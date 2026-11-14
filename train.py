import os
import sys
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from pathlib import Path

# --- HELPER FUNCTIONS FOR DWT ---
def haar_dwt_2d(x):
    lvl1_LL = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] + x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 2.0
    lvl1_LH = (-x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] - x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 2.0
    lvl1_HL = (-x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] + x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 2.0
    lvl1_HH = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] - x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 2.0
    return torch.cat([lvl1_LL, lvl1_LH, lvl1_HL, lvl1_HH], dim=1)

# --- DATASET & DATA LOADER ---
class CASIADataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=(192, 192)):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        self.samples = []

        root_path = Path(root_dir)

        # Quét tất cả ảnh Au và Tp một cách an toàn (không phân biệt hoa thường)
        all_files = list(root_path.rglob("*"))
        au_files = []
        tp_files = []
        gt_files = {}

        for p in all_files:
            if p.is_file() and p.suffix.lower() in [".jpg", ".png", ".tif", ".bmp", ".jpeg"]:
                parts_lower = [part.lower() for part in p.parts]
                # Kiểm tra thư mục chứa ảnh
                if "au" in parts_lower or any("au" in part for part in parts_lower if part != p.name.lower()):
                    if not "groundtruth" in parts_lower and not "gt_mask" in parts_lower:
                        au_files.append(p)
                elif "tp" in parts_lower or any("tp" in part for part in parts_lower if part != p.name.lower()):
                    if not "groundtruth" in parts_lower and not "gt_mask" in parts_lower:
                        tp_files.append(p)

                # Kiểm tra thư mục chứa Ground Truth
                if any(x in parts_lower for x in ["gt_mask", "masks", "casia 2 groundtruth", "groundtruth"]):
                    stem = p.stem.lower()
                    gt_files[stem] = p
                    # Đăng ký thêm biến thể loại bỏ các hậu tố phổ biến
                    stem_clean = stem.replace("_gt", "").replace("gt", "").replace("_mask", "")
                    gt_files[stem_clean] = p

        # Sắp xếp lại danh sách
        au_files = sorted(au_files)
        tp_files = sorted(tp_files)

        for path in au_files:
            self.samples.append({"img_path": str(path), "mask_path": None, "label": 0.0})

        matched_mask_count = 0
        for path in tp_files:
            stem = path.stem.lower()
            matched_mask_path = None
            # Tạo danh sách các ứng viên so khớp phong phú
            candidates = [
                stem,
                stem + "_gt",
                stem.replace("_tp", "_gt"),
                stem + "_mask",
                stem.replace("tp_", "gt_"),
                stem.replace("_tp", "")
            ]
            for cand in candidates:
                if cand in gt_files:
                    matched_mask_path = str(gt_files[cand])
                    break
            if matched_mask_path:
                matched_mask_count += 1
            self.samples.append({"img_path": str(path), "mask_path": matched_mask_path, "label": 1.0})

        print(f"[DATASET] Tổng số ảnh Au: {len(au_files)} | Ảnh Tp: {len(tp_files)} | Khớp thành công Mask: {matched_mask_count}")
        if len(tp_files) > 0 and matched_mask_count == 0:
            raise RuntimeError("❌ LỖI: Không khớp được bất kỳ mask nào cho tập ảnh Tp!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        img = img.resize(self.img_size, Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        mask_tensor = torch.zeros((1, self.img_size[1], self.img_size[0]), dtype=torch.float32)
        if s["label"] == 1.0 and s["mask_path"]:
            try:
                mask = Image.open(s["mask_path"]).convert("L")
                mask = mask.resize(self.img_size, Image.NEAREST)
                mask_np = np.array(mask)
                mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float() / 255.0
                mask_tensor = (mask_tensor > 0.5).float()
            except Exception:
                pass
        return img_tensor, mask_tensor, torch.tensor([s["label"]], dtype=torch.float32)

# --- DWT-MOBILEUNET MODEL ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class DWTMobileUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(15, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up2 = DoubleConv(64, 32)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 1)
        self.out_mask = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        dwt_feat = haar_dwt_2d(x)
        dwt_upsampled = F.interpolate(dwt_feat, size=x.shape[2:], mode="nearest")
        x_in = torch.cat([x, dwt_upsampled], dim=1)
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        cls_feat = self.gap(x3).view(x3.size(0), -1)
        logits_cls = self.fc(cls_feat)
        u1 = self.up1(x3)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.conv_up1(u1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.conv_up2(u2)
        logits_mask = self.out_mask(u2)
        return logits_cls, logits_mask


# --- IMPROVED LOSS FUNCTIONS ---
class FocalLoss2D(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class ComboSegLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.focal = FocalLoss2D(alpha=alpha, gamma=gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def dice_loss(self, preds, targets, smooth=1e-5):
        preds = torch.sigmoid(preds)
        intersection = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def forward(self, preds, targets):
        f_loss = self.focal(preds, targets)
        d_loss = self.dice_loss(preds, targets)
        return self.focal_weight * f_loss + self.dice_weight * d_loss

# --- LOSS & METRICS ---
def dice_loss(preds, targets, smooth=1e-5):
    preds = torch.sigmoid(preds)
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def calculate_seg_metrics(preds, targets, threshold=0.5, smooth=1e-5):
    preds = (torch.sigmoid(preds) > threshold).float()
    targets = (targets > 0.5).float()
    valid_mask = (targets.sum(dim=(1, 2, 3)) > 0)
    if not valid_mask.any():
        return None, None
    preds = preds[valid_mask]
    targets = targets[valid_mask]
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    f1 = (2. * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (union - intersection + smooth)
    return f1.mean().item(), iou.mean().item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dry_run:
        print("[DRY-RUN] Bắt đầu xác minh mô hình...")
        # Model loaded dynamically below
        x = torch.randn(4, 3, 192, 192).to(device)
        y_cls, y_seg = model(x)
        print(f"[DRY-RUN] Tổng số tham số: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
        sys.exit(0)

    dataset = CASIADataset(args.data_dir, img_size=(192, 192))
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model loaded dynamically below
    
    model = DWTMobileUNet().to(device)
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    if os.path.exists(best_model_path) and not args.dry_run:
        print(f"🔄 Phát hiện checkpoint tốt nhất cũ! Đang nạp trọng số từ: {best_model_path} để fine-tune...")
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print("✅ Nạp trọng số thành công!")
        except Exception as e:
            print(f"⚠️ Lỗi nạp trọng số cũ (Sẽ train từ đầu): {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda", enabled=args.amp)

    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_seg_bce = ComboSegLoss(alpha=0.25, gamma=2.0, dice_weight=1.5, focal_weight=1.0)
    best_score = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks, labels in train_loader:
            imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast("cuda", enabled=args.amp):
                logits_cls, logits_mask = model(imgs)
                loss_cls = criterion_cls(logits_cls, labels)
                tp_mask = (labels == 1.0).squeeze()
                if tp_mask.sum() > 0:
                    logits_mask_tp = logits_mask[tp_mask]
                    masks_tp = masks[tp_mask]
                    loss_seg = criterion_seg_bce(logits_mask_tp, masks_tp) + dice_loss(logits_mask_tp, masks_tp)
                else:
                    loss_seg = 0.0
                total_loss = loss_cls + 2.5 * loss_seg
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += total_loss.item()
        scheduler.step()

        model.eval()
        val_cls_correct = 0
        val_total = 0
        val_pixel_f1s, val_pixel_ious = [], []
        with torch.no_grad():
            for imgs, masks, labels in val_loader:
                imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
                with autocast("cuda", enabled=args.amp):
                    logits_cls, logits_mask = model(imgs)
                preds_cls = (torch.sigmoid(logits_cls) > 0.5).float()
                val_cls_correct += (preds_cls == labels).sum().item()
                val_total += labels.size(0)
                f1, iou = calculate_seg_metrics(logits_mask, masks)
                if f1 is not None:
                    val_pixel_f1s.append(f1)
                    val_pixel_ious.append(iou)

        val_acc = val_cls_correct / val_total
        avg_pixel_f1 = np.mean(val_pixel_f1s) if val_pixel_f1s else 0.0
        avg_pixel_iou = np.mean(val_pixel_ious) if val_pixel_ious else 0.0
        composite_score = 0.5 * val_acc + 0.5 * avg_pixel_f1
        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}% | Val Pixel-F1: {avg_pixel_f1:.4f} | Val IoU: {avg_pixel_iou:.4f}")

        if composite_score > best_score:
            best_score = composite_score
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print(f"  --> [Saved New Best Model] Composite Score: {best_score:.4f}")
