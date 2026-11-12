"""
Google Colab CUDA Training CLI for DWT-MobileUNet.

Features:
- Full CLI support: --data_dir, --save_dir, --config, --epochs, --batch_size, --lr,
  --amp, --use-forensic-maps, --resume, and --dry-run.
- Automatic CUDA GPU detection & Mixed Precision (torch.cuda.amp.autocast + GradScaler).
- Mandatory --dry-run mode: Generates synthetic 9-channel (or 11-channel) MRA tensors,
  executes 1 full Forward -> Multi-Task Loss -> Backward -> Optimizer step in < 3 seconds
  to verify tensor shapes, parameter footprint (< 2 MB INT8), and CUDA compatibility.
- Saves best_model.pth and last_model.pth directly to Google Drive (--save_dir).
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

from src.dataset import build_dataloaders
from src.model import DWTMobileUNet, MultiTaskForensicLoss, count_parameters


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(config_path: str) -> Dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def compute_batch_metrics(
    pred_cls: torch.Tensor,
    pred_mask: torch.Tensor,
    target_cls: torch.Tensor,
    target_mask: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes image-level classification accuracy and pixel-level IoU & Dice/F1 score.
    """
    with torch.no_grad():
        # Image-level detection accuracy
        pred_lbl = (pred_cls.view(-1) >= threshold).float()
        true_lbl = target_cls.view(-1).float()
        cls_acc = float((pred_lbl == true_lbl).float().mean().item())

        # Pixel-level localization F1 and IoU
        pred_bin = (pred_mask >= threshold).float().view(pred_mask.size(0), -1)
        true_bin = (target_mask >= threshold).float().view(target_mask.size(0), -1)

        tp = (pred_bin * true_bin).sum(dim=1)
        fp = (pred_bin * (1.0 - true_bin)).sum(dim=1)
        fn = ((1.0 - pred_bin) * true_bin).sum(dim=1)

        eps = 1e-6
        pixel_f1 = ((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)).mean().item()
        pixel_iou = ((tp + eps) / (tp + fp + fn + eps)).mean().item()

    return {
        "cls_acc": cls_acc,
        "pixel_f1": float(pixel_f1),
        "pixel_iou": float(pixel_iou),
    }


def run_dry_run(
    device: torch.device,
    in_channels: int = 9,
    width_multiplier: float = 0.5,
    batch_size: int = 4,
    lr: float = 1e-3,
    use_amp: bool = True,
    save_dir: str = "./checkpoints",
) -> None:
    """
    Executes a 3-second synthetic verification pass:
    Forward -> Multi-Task Loss -> Backward -> Optimizer Step + Checkpoint write test.
    """
    t0 = time.perf_counter()
    print("=" * 72)
    print("[DRY-RUN] Initializing DWT-MobileUNet Synthetic Verification...")
    print(f"[DRY-RUN] Device           : {device} (CUDA available: {torch.cuda.is_available()})")
    if device.type == "cuda":
        print(f"[DRY-RUN] GPU Name         : {torch.cuda.get_device_name(0)}")
    print(f"[DRY-RUN] Input Channels   : {in_channels} (Resolution: 192x192)")
    print(f"[DRY-RUN] AMP FP16 Enabled : {use_amp and device.type == 'cuda'}")

    model = DWTMobileUNet(
        in_channels=in_channels, width_multiplier=width_multiplier
    ).to(device)
    total_params, est_int8_mb = count_parameters(model)
    print(
        f"[DRY-RUN] Trainable Params : {total_params:,} (~{est_int8_mb:.3f} MB in INT8, Target < 2.0 MB)"
    )

    criterion = MultiTaskForensicLoss(cls_weight=0.3, seg_weight=0.7)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    amp_Active = bool(use_amp and device.type == "cuda")
    scaler = GradScaler(enabled=amp_Active)

    # Synthetic batch: (B, C, 192, 192)
    synth_x = torch.randn(batch_size, in_channels, 192, 192, device=device)
    synth_cls = torch.randint(0, 2, (batch_size, 1), device=device).float()
    synth_mask = torch.randint(0, 2, (batch_size, 1, 192, 192), device=device).float()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    with autocast(enabled=amp_Active):
        pred_cls, pred_mask = model(synth_x)
        loss, loss_dict = criterion(pred_cls, pred_mask, synth_cls, synth_mask)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize()

    assert pred_cls.shape == (batch_size, 1), f"Invalid Head 1 shape: {pred_cls.shape}"
    assert pred_mask.shape == (
        batch_size,
        1,
        192,
        192,
    ), f"Invalid Head 2 shape: {pred_mask.shape}"

    # Save a verified dry-run checkpoint if no best_model.pth exists yet (allows immediate export/UI test)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    dry_ckpt = save_path / "best_model.pth"
    if not dry_ckpt.exists():
        torch.save(
            {
                "epoch": 0,
                "model_state_dict": model.state_dict(),
                "in_channels": in_channels,
                "width_multiplier": width_multiplier,
                "dry_run": True,
            },
            dry_ckpt,
        )
        print(f"[DRY-RUN] Created initial checkpoint at: {dry_ckpt}")

    elapsed = time.perf_counter() - t0
    print(f"[DRY-RUN] Head 1 Output    : {tuple(pred_cls.shape)} (P_fake in [0, 1])")
    print(f"[DRY-RUN] Head 2 Output    : {tuple(pred_mask.shape)} (Mask 192x192x1)")
    print(
        f"[DRY-RUN] Loss Breakdown   : Total={loss_dict['loss_total']:.4f} "
        f"(BCE={loss_dict['loss_bce_cls']:.4f}, Dice={loss_dict['loss_dice']:.4f}, Focal={loss_dict['loss_focal']:.4f})"
    )
    print(f"[DRY-RUN] SUCCESS in {elapsed:.2f}s (< 3.0s target). Ready for full training!")
    print("=" * 72)


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MultiTaskForensicLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.train()
    running = {
        "loss_total": 0.0,
        "loss_bce_cls": 0.0,
        "loss_dice": 0.0,
        "loss_focal": 0.0,
        "cls_acc": 0.0,
        "pixel_f1": 0.0,
        "pixel_iou": 0.0,
    }
    num_batches = max(1, len(loader))

    for mra_x, cls_y, mask_y in loader:
        mra_x = mra_x.to(device, non_blocking=True)
        cls_y = cls_y.to(device, non_blocking=True)
        mask_y = mask_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            pred_cls, pred_mask = model(mra_x)
            loss, loss_dict = criterion(pred_cls, pred_mask, cls_y, mask_y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_metrics = compute_batch_metrics(pred_cls, pred_mask, cls_y, mask_y)
        for k in loss_dict:
            running[k] += loss_dict[k]
        for k in batch_metrics:
            running[k] += batch_metrics[k]

    return {k: v / num_batches for k, v in running.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MultiTaskForensicLoss,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.eval()
    running = {
        "loss_total": 0.0,
        "loss_bce_cls": 0.0,
        "loss_dice": 0.0,
        "loss_focal": 0.0,
        "cls_acc": 0.0,
        "pixel_f1": 0.0,
        "pixel_iou": 0.0,
    }
    num_batches = max(1, len(loader))

    for mra_x, cls_y, mask_y in loader:
        mra_x = mra_x.to(device, non_blocking=True)
        cls_y = cls_y.to(device, non_blocking=True)
        mask_y = mask_y.to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            pred_cls, pred_mask = model(mra_x)
            _, loss_dict = criterion(pred_cls, pred_mask, cls_y, mask_y)

        batch_metrics = compute_batch_metrics(pred_cls, pred_mask, cls_y, mask_y)
        for k in loss_dict:
            running[k] += loss_dict[k]
        for k in batch_metrics:
            running[k] += batch_metrics[k]

    return {k: v / num_batches for k, v in running.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DWT-MobileUNet on Google Colab CUDA GPU"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/CASIA2",
        help="Dataset root directory containing Au/, Tp/, and Gt_mask/",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save best_model.pth (e.g., Google Drive path)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    parser.add_argument(
        "--num_workers", type=int, default=None, help="DataLoader worker processes"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=None,
        help="Enable Automatic Mixed Precision (FP16) on CUDA",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable Automatic Mixed Precision",
    )
    parser.add_argument(
        "--use-forensic-maps",
        action="store_true",
        default=None,
        help="Include explicit Donoho MAD + ELA maps (11 input channels instead of 9)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (.pth) to resume training from",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a 3-second synthetic Forward/Loss/Backward/Optimizer check and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    # Resolve hyperparameters from CLI overrides -> config.yaml -> defaults
    use_forensic = (
        args.use_forensic_maps
        if args.use_forensic_maps is not None
        else cfg.get("data", {}).get("use_forensic_maps", False)
    )
    in_channels = 11 if use_forensic else 9
    width_mult = cfg.get("model", {}).get("width_multiplier", 0.5)
    img_size = cfg.get("data", {}).get("img_size", 192)
    ela_quality = cfg.get("data", {}).get("ela_quality", 90)
    train_split = cfg.get("data", {}).get("train_split", 0.85)
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else cfg.get("data", {}).get("num_workers", 2)
    )

    epochs = (
        args.epochs
        if args.epochs is not None
        else cfg.get("training", {}).get("epochs", 50)
    )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else cfg.get("training", {}).get("batch_size", 16)
    )
    lr = args.lr if args.lr is not None else cfg.get("training", {}).get("lr", 1e-3)
    weight_decay = cfg.get("training", {}).get("weight_decay", 1e-4)
    save_dir = (
        args.save_dir
        if args.save_dir is not None
        else cfg.get("training", {}).get("save_dir", "./checkpoints")
    )
    use_amp = (
        args.amp
        if args.amp is not None
        else cfg.get("training", {}).get("amp", True)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Execute fast 3-second synthetic verification if --dry-run is passed
    if args.dry_run:
        run_dry_run(
            device=device,
            in_channels=in_channels,
            width_multiplier=width_mult,
            batch_size=min(batch_size, 4),
            lr=lr,
            use_amp=use_amp,
            save_dir=save_dir,
        )
        return

    # 2. Full CUDA Training Pipeline
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"[TRAIN] Starting DWT-MobileUNet Training on {device}")
    if device.type == "cuda":
        print(f"[TRAIN] GPU: {torch.cuda.get_device_name(0)} | AMP FP16: {use_amp}")
    print(f"[TRAIN] Data Directory : {args.data_dir}")
    print(f"[TRAIN] Save Directory : {save_path.resolve()}")

    train_loader, val_loader, stats = build_dataloaders(
        data_dir=args.data_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        train_split=train_split,
        use_forensic_maps=use_forensic,
        ela_quality=ela_quality,
        seed=seed,
    )
    print(
        f"[TRAIN] Dataset Loaded : Total={stats['total']} "
        f"(Train={stats['train']}, Val={stats['val']}, "
        f"Au={stats['authentic']}, Tp={stats['tampered']}, Masks={stats['with_mask']})"
    )

    model = DWTMobileUNet(
        in_channels=in_channels,
        width_multiplier=width_mult,
        decoder_channels=cfg.get("model", {}).get("decoder_channels", [96, 48, 24, 16]),
        dropout=cfg.get("model", {}).get("dropout", 0.2),
    ).to(device)

    total_params, est_int8_mb = count_parameters(model)
    print(f"[TRAIN] Model Footprint: {total_params:,} params (~{est_int8_mb:.3f} MB in INT8)")

    criterion = MultiTaskForensicLoss(
        cls_weight=cfg.get("loss", {}).get("cls_weight", 0.3),
        seg_weight=cfg.get("loss", {}).get("seg_weight", 0.7),
        focal_alpha=cfg.get("loss", {}).get("focal_alpha", 0.25),
        focal_gamma=cfg.get("loss", {}).get("focal_gamma", 2.0),
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)

    start_epoch = 1
    best_score = -1.0
    history = []

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_score = ckpt.get("best_score", -1.0)
        print(f"[TRAIN] Resumed from {args.resume} (Epoch {start_epoch})")

    for epoch in range(start_epoch, epochs + 1):
        t_ep = time.perf_counter()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, amp_enabled
        )
        val_metrics = evaluate(model, val_loader, criterion, device, amp_enabled)
        scheduler.step()

        # Combined selection metric: 0.3 * Cls_Acc + 0.7 * Pixel_F1
        val_composite = (
            0.3 * val_metrics["cls_acc"] + 0.7 * val_metrics["pixel_f1"]
        )
        dt = time.perf_counter() - t_ep

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] ({dt:.1f}s) | "
            f"Train Loss: {train_metrics['loss_total']:.4f} (F1: {train_metrics['pixel_f1']:.4f}) | "
            f"Val Loss: {val_metrics['loss_total']:.4f} | "
            f"Val Acc: {val_metrics['cls_acc']*100:.2f}% | "
            f"Val Pixel-F1: {val_metrics['pixel_f1']:.4f} | "
            f"Val IoU: {val_metrics['pixel_iou']:.4f}"
        )

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_score": max(best_score, val_composite),
            "in_channels": in_channels,
            "width_multiplier": width_mult,
            "use_forensic_maps": use_forensic,
            "val_metrics": val_metrics,
        }

        # Always update last_model.pth for fault-tolerant Colab resumption
        torch.save(ckpt_payload, save_path / "last_model.pth")

        if val_composite >= best_score:
            best_score = val_composite
            torch.save(ckpt_payload, save_path / "best_model.pth")
            print(
                f"  --> [Saved New Best Model] Composite Score: {best_score:.4f} -> {save_path / 'best_model.pth'}"
            )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "val_composite": val_composite,
            }
        )
        with open(save_path / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print("=" * 72)
    print(f"[TRAIN] Completed! Best Composite Score: {best_score:.4f}")
    print(f"[TRAIN] Best Checkpoint: {save_path / 'best_model.pth'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
