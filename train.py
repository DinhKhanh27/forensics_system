"""
Google Colab CUDA Multi-Round Training CLI for DWT-MobileUNet (`train.py`).

Fixes Addressing Code Review:
1. `--dry-run` NEVER writes to `best_model.pth` (saves only to a separate `dry_run.pth`
   or skips overwriting real checkpoints so an interrupted training session never exports
   untrained random weights).
2. Group-Aware Stratified Validation Split (`--val_dir` optional for cross-dataset eval).
3. Refuses to load `dry_run` checkpoints during real training.
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from torch.amp import GradScaler as _AmpGradScaler
    from torch.amp import autocast as _amp_autocast

    def create_grad_scaler(device_type: str, enabled: bool):
        return _AmpGradScaler(device_type, enabled=enabled)

    def get_autocast(device_type: str, enabled: bool):
        return _amp_autocast(device_type, enabled=enabled)
except (ImportError, TypeError):
    from torch.cuda.amp import GradScaler as _CudaGradScaler
    from torch.cuda.amp import autocast as _cuda_autocast

    def create_grad_scaler(device_type: str, enabled: bool):
        return _CudaGradScaler(enabled=enabled)

    def get_autocast(device_type: str, enabled: bool):
        return _cuda_autocast(enabled=enabled)

from src.dataset import build_dataloaders
from src.model import (
    DWTMobileUNet,
    MultiTaskForensicLoss,
    count_parameters,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(payload: Dict, target_path: str) -> None:
    """Save checkpoint atomically via a temporary file to prevent corruption on disconnect."""
    tmp_path = f"{target_path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, target_path)


def _atomic_json_save(data: List[Dict], target_path: str) -> None:
    """Save training history JSON atomically."""
    tmp_path = f"{target_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, target_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DWT-MobileUNet (Multi-Round Supported) on Google Colab CUDA GPU"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/CASIA2",
        help="Dataset root directory containing Au/, Tp/, and Gt_mask / CASIA 2 Groundtruth",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=None,
        help="Optional independent validation dataset directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save best_model.pth and round checkpoints (Google Drive)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs per round (default: 15)",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument(
        "--round_id",
        type=int,
        default=1,
        help="Current training round index (1, 2, 3...)",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=1,
        help="If > 1, automatically executes multiple consecutive rounds with LR decay",
    )
    parser.add_argument(
        "--round_lr_decay",
        type=float,
        default=0.4,
        help="Learning rate multiplier for each subsequent round when --num_rounds > 1",
    )
    parser.add_argument(
        "--pos_pixel_weight",
        type=float,
        default=3.0,
        help="Positive foreground pixel weight in BCE loss (default: 3.0)",
    )
    parser.add_argument(
        "--native_crop_prob",
        type=float,
        default=0.5,
        help="Probability of extracting native 1:1 8x8 JPEG-grid aligned 192x192 crop vs full resize (default: 0.5)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable Automatic Mixed Precision (FP16) on CUDA",
    )
    parser.add_argument(
        "--use-forensic-maps",
        action="store_true",
        help="Use 11-channel CPU MRA input (RGB + 6 DWT + Donoho MAD + ELA)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Explicit checkpoint path to load before starting the round",
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="Automatically resume an interrupted round from last_model.pth (restoring optimizer, scheduler, scaler, and epoch)",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Ignore existing best_model.pth and train Round 1 from scratch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a 3-second synthetic Forward/Loss/Backward/Optimizer check (saves only dry_run.pth)",
    )
    return parser.parse_args()


@torch.no_grad()
def run_real_sanity_check(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    """
    Picks 1 Authentic (Au) and 1 Tampered (Tp) sample from `val_loader`, runs inference
    with Sigmoid probabilities, and prints real-world verification metrics.
    """
    model.eval()
    au_sample = None
    tp_sample = None

    for imgs, masks, labels in val_loader:
        for i in range(labels.size(0)):
            lbl = float(labels[i].item())
            if lbl == 0.0 and au_sample is None:
                au_sample = (imgs[i : i + 1].to(device), masks[i : i + 1].to(device))
            elif lbl == 1.0 and tp_sample is None:
                tp_sample = (imgs[i : i + 1].to(device), masks[i : i + 1].to(device))
            if au_sample is not None and tp_sample is not None:
                break
        if au_sample is not None and tp_sample is not None:
            break

    print("  [SANITY CHECK THỰC NGHIỆM TRÊN TẬP VAL]")
    if au_sample is not None:
        logits_c, logits_m = model(au_sample[0])
        p_fake_au = float(torch.sigmoid(logits_c).item()) * 100.0
        fp_area_au = float((torch.sigmoid(logits_m) >= 0.5).float().mean().item()) * 100.0
        verdict_au = "ĐÚNG (ẢNH THẬT)" if p_fake_au < 50.0 else "SAI (BÁO GIẢ)"
        print(
            f"    • Mẫu Ảnh Thật (Au) : model_prob_cls = {p_fake_au:5.2f}% | Vùng báo giả = {fp_area_au:5.2f}% -> {verdict_au}"
        )

    if tp_sample is not None:
        logits_c, logits_m = model(tp_sample[0])
        p_fake_tp = float(torch.sigmoid(logits_c).item()) * 100.0
        prob_m = (torch.sigmoid(logits_m) >= 0.5).float()
        gt_m = (tp_sample[1] >= 0.5).float()
        inter = float((prob_m * gt_m).sum().item())
        union = float((prob_m.sum() + gt_m.sum()).item())
        f1_tp = (2.0 * inter + 1e-5) / (union + 1e-5)
        verdict_tp = "ĐÚNG (ẢNH GIẢ)" if p_fake_tp >= 50.0 else "SAI (BỎ SÓT)"
        print(
            f"    • Mẫu Ảnh Giả  (Tp) : model_prob_cls = {p_fake_tp:5.2f}% | Pixel-F1 = {f1_tp:.4f}     -> {verdict_tp}"
        )


def main() -> None:
    args = parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 11 if args.use_forensic_maps else 3
    amp_enabled = bool(args.amp and device.type == "cuda")

    # 1. Fast 3-second --dry-run mode (NEVER writes to best_model.pth!)
    if args.dry_run:
        t0 = time.perf_counter()
        print("=" * 72)
        print(f"[DRY-RUN] Bắt đầu xác minh mô hình DWT-MobileUNet trên {device}...")
        model = DWTMobileUNet(
            in_channels=in_channels, width_multiplier=0.5, return_logits=True
        ).to(device)
        total_params, est_int8_mb = count_parameters(model)
        print(
            f"[DRY-RUN] Tổng số tham số: {total_params:,} (~{est_int8_mb:.3f} MB ở dạng INT8, Mục tiêu < 2.0 MB)"
        )

        criterion = MultiTaskForensicLoss(
            cls_weight=0.3, seg_weight=0.7, pos_pixel_weight=args.pos_pixel_weight
        )
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = create_grad_scaler(device.type, enabled=amp_enabled)

        x = torch.rand(4, in_channels, 192, 192, device=device)
        y_cls = torch.tensor([[1.0], [0.0], [1.0], [0.0]], device=device)
        y_mask = torch.zeros((4, 1, 192, 192), device=device)
        y_mask[0, :, 48:128, 48:128] = 1.0
        y_mask[2, :, 64:144, 64:144] = 1.0

        model.train()
        optimizer.zero_grad(set_to_none=True)
        with get_autocast(device.type, enabled=amp_enabled):
            logits_cls, logits_mask = model(x)
            loss, loss_info = criterion(logits_cls, logits_mask, y_cls, y_mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        os.makedirs(args.save_dir, exist_ok=True)
        dry_ckpt_path = os.path.join(args.save_dir, "dry_run.pth")
        torch.save(
            {
                "epoch": 0,
                "round_id": 0,
                "dry_run": True,
                "arch": "DWTMobileUNet",
                "in_channels": in_channels,
                "width_multiplier": 0.5,
                "best_score": 0.0,
                "model_state_dict": model.state_dict(),
            },
            dry_ckpt_path,
        )
        dt = time.perf_counter() - t0
        print(
            f"[DRY-RUN] Thành công trong {dt:.2f}s! Loss={loss_info['loss_total']:.4f} "
            f"(Lưu file kiểm thử riêng tại: {dry_ckpt_path}, KHÔNG ghi đè best_model.pth)"
        )
        print("=" * 72)
        sys.exit(0)

    # 2. Load Dataset with Leak-Free Group-Aware Stratified Split & Mask Validation
    os.makedirs(args.save_dir, exist_ok=True)
    train_loader, val_loader, stats = build_dataloaders(
        data_dir=args.data_dir,
        val_dir=args.val_dir,
        img_size=192,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=0.85,
        use_forensic_maps=args.use_forensic_maps,
        native_crop_prob=args.native_crop_prob,
        seed=42,
    )
    print(
        f"[DATASET] Tổng mẫu hợp lệ: {stats['total']} (Train={stats['train']}, Val={stats['val']}) | "
        f"Au={stats['authentic']} | Tp (Mask hợp lệ)={stats['tampered']} | "
        f"Loại bỏ Tp thiếu/lỗi Mask={stats['skipped_unmatched_tp'] + stats['corrupted_or_empty_masks']} | "
        f"Hybrid Native-Crop Prob={args.native_crop_prob*100:.0f}%"
    )

    # 3. Initialize DWT-MobileUNet
    model = DWTMobileUNet(
        in_channels=in_channels, width_multiplier=0.5, return_logits=True
    ).to(device)
    total_params, est_int8_mb = count_parameters(model)
    print(
        f"[MODEL] Kiến trúc DWTMobileUNet (2-Level Haar DWT + Full-Res Skip + MobileNetV3-Small): "
        f"{total_params:,} params (~{est_int8_mb:.3f} MB INT8)"
    )

    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    last_model_path = os.path.join(args.save_dir, "last_model.pth")
    history_path = os.path.join(args.save_dir, "training_history.json")

    best_score = 0.0
    history: List[Dict] = []
    if os.path.exists(history_path) and (not args.from_scratch or args.auto_resume):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    resumed_ckpt: Optional[Dict] = None
    ckpt_to_load: Optional[str] = None
    if args.resume:
        ckpt_to_load = args.resume
    elif args.auto_resume and os.path.exists(last_model_path):
        ckpt_to_load = last_model_path
    elif not args.from_scratch:
        ckpt_to_load = best_model_path

    if ckpt_to_load and os.path.exists(ckpt_to_load):
        print(f"🔄 Phát hiện checkpoint tại: {ckpt_to_load}. Đang kiểm tra tương thích...")
        try:
            raw_ckpt = torch.load(ckpt_to_load, map_location=device)
            if isinstance(raw_ckpt, dict) and raw_ckpt.get("dry_run", False):
                print("⚠️ Bỏ qua checkpoint được sinh từ --dry-run (trọng số ngẫu nhiên). Sẽ train từ đầu!")
            else:
                sd = (
                    raw_ckpt.get("model_state_dict", raw_ckpt)
                    if isinstance(raw_ckpt, dict)
                    else raw_ckpt
                )
                model.load_state_dict(sd, strict=True)
                if isinstance(raw_ckpt, dict):
                    best_score = float(raw_ckpt.get("best_score", 0.0))
                    resumed_ckpt = raw_ckpt
                print(
                    f"✅ Nạp trọng số DWTMobileUNet thành công (Global best_score={best_score:.4f})!"
                )
        except Exception as e:
            print(
                f"ℹ️ Checkpoint cũ thuộc kiến trúc trước hoặc bị lỗi ({type(e).__name__}). "
                f"Khởi tạo huấn luyện mới với kiến trúc chuẩn DWTMobileUNet (best_score=0.0)."
            )
            best_score = 0.0
            resumed_ckpt = None

    # 4. Execute Training across 1 or multiple rounds
    start_round = args.round_id
    total_rounds = max(1, args.num_rounds)

    for r_offset in range(total_rounds):
        current_round = start_round + r_offset
        round_lr = args.lr * (args.round_lr_decay ** r_offset)

        start_epoch = 1
        round_best_score = -1.0
        is_resuming_round = False

        if resumed_ckpt is not None and (args.resume or args.auto_resume):
            ckpt_round = int(resumed_ckpt.get("round_id", -1))
            ckpt_epoch = int(resumed_ckpt.get("epoch", 0))
            if args.auto_resume and not args.resume:
                if ckpt_round > current_round:
                    print(
                        f"⏭️ [ROUND {current_round}] Phát hiện last_model.pth đang ở Round {ckpt_round}. "
                        f"Bỏ qua Round {current_round} để tránh ghi đè tiến trình mới hơn."
                    )
                    continue
                if ckpt_round == current_round and ckpt_epoch >= args.epochs:
                    print(
                        f"✅ [ROUND {current_round}] Đã hoàn tất trước đó ({ckpt_epoch}/{args.epochs} epochs). Bỏ qua!"
                    )
                    continue
            if ckpt_round == current_round and ckpt_epoch < args.epochs:
                start_epoch = ckpt_epoch + 1
                round_best_score = float(resumed_ckpt.get("round_best_score", -1.0))
                is_resuming_round = True
            elif ckpt_round < current_round and r_offset == 0 and os.path.exists(best_model_path):
                # Previous round finished; start the new round from best_model.pth
                best_ckpt = torch.load(best_model_path, map_location=device)
                if not best_ckpt.get("dry_run", False):
                    model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
                    best_score = float(best_ckpt.get("best_score", best_score))
                    print(
                        f"\n🔄 [ROUND {current_round}] Đã nạp best_model.pth từ Round {ckpt_round} "
                        f"(Best Score = {best_score:.4f}) để tinh chỉnh tiếp với LR = {round_lr:.6f}"
                    )

        if r_offset > 0 and not is_resuming_round and os.path.exists(best_model_path):
            best_ckpt = torch.load(best_model_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
            print(
                f"\n🔄 [ROUND {current_round}] Đã nạp lại best_model.pth từ Round {current_round - 1} "
                f"(Best Score = {best_score:.4f}) để tinh chỉnh tiếp với LR = {round_lr:.6f}"
            )

        print("=" * 72)
        if is_resuming_round:
            print(
                f"🔄 RESUME ROUND {current_round} từ Epoch {start_epoch}/{args.epochs} | LR: {round_lr:.6f} | Batch: {args.batch_size}"
            )
        else:
            print(
                f"🚀 BẮT ĐẦU ROUND {current_round} | Epochs: {args.epochs} | LR: {round_lr:.6f} | Batch: {args.batch_size}"
            )
        print("=" * 72)

        criterion = MultiTaskForensicLoss(
            cls_weight=0.3,
            seg_weight=0.7,
            pos_pixel_weight=args.pos_pixel_weight,
        )
        optimizer = AdamW(model.parameters(), lr=round_lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=round_lr * 0.02
        )
        scaler = create_grad_scaler(device.type, enabled=amp_enabled)

        if is_resuming_round and resumed_ckpt is not None:
            if "optimizer_state_dict" in resumed_ckpt:
                optimizer.load_state_dict(resumed_ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in resumed_ckpt:
                scheduler.load_state_dict(resumed_ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in resumed_ckpt and amp_enabled:
                scaler.load_state_dict(resumed_ckpt["scaler_state_dict"])
            history = [
                h
                for h in history
                if not (
                    h.get("round_id") == current_round
                    and int(h.get("epoch", 0)) >= start_epoch
                )
            ]
            # Clear resumed_ckpt after consuming it for this round
            resumed_ckpt = None

        round_best_path = os.path.join(
            args.save_dir, f"round_{current_round}_best.pth"
        )

        for epoch in range(start_epoch, args.epochs + 1):
            t_ep = time.perf_counter()
            model.train()
            train_loss_sum = 0.0

            for imgs, masks, labels in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with get_autocast(device.type, enabled=amp_enabled):
                    logits_cls, logits_mask = model(imgs)
                    total_loss, _ = criterion(logits_cls, logits_mask, labels, masks)

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss_sum += float(total_loss.item())

            scheduler.step()

            # Validation phase
            model.eval()
            val_cls_correct = 0
            val_total = 0
            tp_f1_list, tp_iou_list = [], []
            au_fp_rates = []

            with torch.no_grad():
                for imgs, masks, labels in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with get_autocast(device.type, enabled=amp_enabled):
                        logits_cls, logits_mask = model(imgs)

                    probs_cls = torch.sigmoid(logits_cls)
                    probs_mask = torch.sigmoid(logits_mask)

                    preds_cls = (probs_cls >= 0.5).float()
                    val_cls_correct += (preds_cls == labels).sum().item()
                    val_total += labels.size(0)

                    pred_bin = (probs_mask >= 0.5).float()
                    true_bin = (masks >= 0.5).float()

                    tp_idx = labels.view(-1) > 0.5
                    if tp_idx.any():
                        p_tp = pred_bin[tp_idx]
                        t_tp = true_bin[tp_idx]
                        inter = (p_tp * t_tp).sum(dim=(1, 2, 3))
                        union = p_tp.sum(dim=(1, 2, 3)) + t_tp.sum(dim=(1, 2, 3))
                        f1_batch = (2.0 * inter + 1e-5) / (union + 1e-5)
                        iou_batch = (inter + 1e-5) / (union - inter + 1e-5)
                        tp_f1_list.extend(f1_batch.cpu().tolist())
                        tp_iou_list.extend(iou_batch.cpu().tolist())

                    au_idx = ~tp_idx
                    if au_idx.any():
                        p_au = pred_bin[au_idx]
                        fp_rate = p_au.mean(dim=(1, 2, 3))
                        au_fp_rates.extend(fp_rate.cpu().tolist())

            val_acc = val_cls_correct / max(1, val_total)
            avg_pixel_f1 = float(np.mean(tp_f1_list)) if tp_f1_list else 0.0
            avg_pixel_iou = float(np.mean(tp_iou_list)) if tp_iou_list else 0.0
            avg_au_fp = float(np.mean(au_fp_rates)) if au_fp_rates else 0.0
            au_clean_score = max(0.0, 1.0 - avg_au_fp * 5.0)

            composite_score = (
                0.30 * val_acc + 0.55 * avg_pixel_f1 + 0.15 * au_clean_score
            )
            avg_train_loss = train_loss_sum / max(1, len(train_loader))
            dt = time.perf_counter() - t_ep

            print(
                f"[Round {current_round} | Epoch {epoch:02d}/{args.epochs:02d}] ({dt:.1f}s) | "
                f"Loss: {avg_train_loss:.4f} | "
                f"Val Acc: {val_acc*100:.2f}% | "
                f"Pixel-F1: {avg_pixel_f1:.4f} | "
                f"IoU: {avg_pixel_iou:.4f} | "
                f"Au-FP: {avg_au_fp*100:.2f}%"
            )

            if composite_score > round_best_score:
                round_best_score = composite_score
            new_global_best = max(best_score, composite_score)

            ckpt_payload = {
                "round_id": current_round,
                "epoch": epoch,
                "dry_run": False,
                "arch": "DWTMobileUNet",
                "in_channels": in_channels,
                "width_multiplier": 0.5,
                "best_score": new_global_best,
                "round_best_score": round_best_score,
                "val_acc": val_acc,
                "val_pixel_f1": avg_pixel_f1,
                "val_iou": avg_pixel_iou,
                "au_fp_rate": avg_au_fp,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }
            _atomic_torch_save(ckpt_payload, last_model_path)

            if composite_score >= round_best_score:
                _atomic_torch_save(ckpt_payload, round_best_path)

            if composite_score > best_score:
                best_score = composite_score
                ckpt_payload["best_score"] = best_score
                _atomic_torch_save(ckpt_payload, best_model_path)
                print(
                    f"  --> [Saved New Global Best Model] Composite Score: {best_score:.4f} (Pixel-F1: {avg_pixel_f1:.4f}, Au-FP: {avg_au_fp*100:.2f}%)"
                )

            history.append(
                {
                    "round_id": current_round,
                    "epoch": epoch,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "train_loss": avg_train_loss,
                    "val_acc": val_acc,
                    "val_pixel_f1": avg_pixel_f1,
                    "val_iou": avg_pixel_iou,
                    "au_fp_rate": avg_au_fp,
                    "composite_score": composite_score,
                }
            )
            _atomic_json_save(history, history_path)

        run_real_sanity_check(model, val_loader, device)
        print(
            f"✅ Hoàn tất Round {current_round}! Round Best: {round_best_score:.4f} | Global Best: {best_score:.4f} -> {best_model_path}"
        )


if __name__ == "__main__":
    main()
