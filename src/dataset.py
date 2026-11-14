"""
Forensic Image Dataset & Stratified DataLoader Module (CASIA v2.0, Columbia, NIST16).

Key Improvements:
1. Hybrid Multi-Scale Training (50% Native JPEG-Aligned Crop + 50% Full-Image Resize):
   - Directly solves the Train-Test Domain Gap between 'fast' mode (resized full image) and
     'tile' mode (192x192 sliding window patches at native resolution).
   - With 50% probability during training, extracts a native-resolution 192x192 patch aligned
     to the 8x8 JPEG grid (x0 % 8 == 0, y0 % 8 == 0), guided toward forgery boundaries on Tp
     images so the 2-Level Haar DWT learns un-interpolated micro-noise and JPEG block artifacts.
   - With 50% probability, resizes the full image to 192x192 so the model also learns global
     scene context.
2. Stratified Train/Validation Split & Joint Geometry-Preserving Augmentations.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.dwt import MRAPreprocessor

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def scan_casia_forensic_dir(root_dir: str) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Recursively scans `root_dir` (using paths relative to `root_dir`) for Authentic (`Au`),
    Tampered (`Tp`), and Ground-Truth mask files (`Gt_mask`, `CASIA 2 Groundtruth`, `masks`).
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root_dir}")

    all_files = list(root_path.rglob("*"))
    au_files: List[Path] = []
    tp_files: List[Path] = []
    gt_files: Dict[str, Path] = {}

    exact_gt_names = {"gt", "gt_mask", "mask", "masks", "groundtruth", "ground_truth"}
    substr_gt_names = ("gt_mask", "groundtruth", "ground_truth", "casia 2 groundtruth", "mask")

    for p in all_files:
        if not p.is_file() or p.suffix.lower() not in VALID_IMAGE_EXTS:
            continue

        rel_parts = [part.lower() for part in p.relative_to(root_path).parts[:-1]]
        is_gt_folder = any(
            (part in exact_gt_names) or any(gk in part for gk in substr_gt_names)
            for part in rel_parts
        )

        if is_gt_folder:
            stem = p.stem.lower()
            gt_files[stem] = p
            for suffix in ("_gt", "_mask", "-gt", "-mask", ".gt"):
                if stem.endswith(suffix):
                    gt_files[stem[: -len(suffix)]] = p
            stem_clean = stem.replace("_gt", "").replace("gt", "").replace("_mask", "")
            if stem_clean:
                gt_files[stem_clean] = p
            continue

        name_lower = p.name.lower()
        if any(part == "au" or "authentic" in part for part in rel_parts) or name_lower.startswith("au_"):
            au_files.append(p)
        elif any(part == "tp" or "tamper" in part or "forged" in part for part in rel_parts) or name_lower.startswith("tp_"):
            tp_files.append(p)

    au_files.sort()
    tp_files.sort()

    samples: List[Dict] = []
    for path in au_files:
        samples.append({"img_path": str(path), "mask_path": None, "label": 0.0})

    matched_mask_count = 0
    unmatched_tp = 0
    for path in tp_files:
        stem = path.stem.lower()
        matched_mask_path = None
        candidates = [
            stem,
            stem + "_gt",
            stem.replace("_tp", "_gt"),
            stem + "_mask",
            stem.replace("tp_", "gt_"),
            stem.replace("_tp", ""),
            stem.replace("_gt", "").replace("gt", "").replace("_mask", ""),
        ]
        for cand in candidates:
            if cand in gt_files:
                matched_mask_path = str(gt_files[cand])
                break

        if matched_mask_path is not None:
            matched_mask_count += 1
            samples.append(
                {"img_path": str(path), "mask_path": matched_mask_path, "label": 1.0}
            )
        else:
            unmatched_tp += 1

    if len(tp_files) > 0 and matched_mask_count == 0:
        raise RuntimeError(
            f"❌ LỖI: Tìm thấy {len(tp_files)} ảnh Tp nhưng không khớp được mask nào trong {root_dir}!"
        )

    stats = {
        "total": len(samples),
        "authentic": len(au_files),
        "tampered": matched_mask_count,
        "with_mask": matched_mask_count,
        "skipped_unmatched_tp": unmatched_tp,
    }
    return samples, stats


class CASIADataset(Dataset):
    """
    Dataset returning `(img_or_mra_tensor, mask_tensor, label_tensor)`.
    Uses Hybrid Crop + Resize during training so the model excels at BOTH native-resolution
    sliding window patches ('tile' mode) and full-image resizes ('fast' mode).
    """

    def __init__(
        self,
        samples_or_root,
        img_size: Tuple[int, int] = (192, 192),
        is_train: bool = True,
        return_mra: bool = False,
        use_forensic_maps: bool = False,
        ela_quality: int = 90,
        native_crop_prob: float = 0.5,
    ) -> None:
        super().__init__()
        if isinstance(samples_or_root, (str, Path)):
            self.samples, self.stats = scan_casia_forensic_dir(str(samples_or_root))
            print(
                f"[DATASET] Tổng số ảnh Au: {self.stats['authentic']} | "
                f"Ảnh Tp (có Mask): {self.stats['tampered']} | "
                f"Bỏ qua Tp thiếu Mask: {self.stats['skipped_unmatched_tp']}"
            )
        else:
            self.samples = samples_or_root
            self.stats = {}

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size

        self.is_train = is_train
        self.native_crop_prob = native_crop_prob
        self.return_mra = return_mra or use_forensic_maps
        self.mra = (
            MRAPreprocessor(
                target_size=self.img_size,
                use_forensic_maps=use_forensic_maps,
                ela_quality=ela_quality,
            )
            if self.return_mra
            else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_jpeg_aligned_crop(
        self,
        img_full: np.ndarray,
        mask_full: np.ndarray,
        target_h: int,
        target_w: int,
        label: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extracts a native-resolution (target_h x target_w) patch whose top-left coordinates
        (y0, x0) are multiples of 8 (preserving the 8x8 JPEG block grid phase).
        For Tampered (Tp) images, centers the crop around a forged pixel so the patch contains
        actual tampered boundaries.
        """
        h, w = img_full.shape[:2]
        max_y = h - target_h
        max_x = w - target_w

        if label == 1.0:
            fg_ys, fg_xs = np.where(mask_full > 0.5)
            if len(fg_ys) > 0:
                idx = random.randint(0, len(fg_ys) - 1)
                cy, cx = int(fg_ys[idx]), int(fg_xs[idx])
                jitter_y = random.randint(-target_h // 3, target_h // 3)
                jitter_x = random.randint(-target_w // 3, target_w // 3)
                y0 = max(0, min(max_y, cy - target_h // 2 + jitter_y))
                x0 = max(0, min(max_x, cx - target_w // 2 + jitter_x))
                # Snap to 8x8 JPEG block grid
                y0 = min(max_y, (y0 // 8) * 8)
                x0 = min(max_x, (x0 // 8) * 8)
                crop_img = img_full[y0 : y0 + target_h, x0 : x0 + target_w]
                crop_mask = mask_full[y0 : y0 + target_h, x0 : x0 + target_w]
                eff_label = 1.0 if float(crop_mask.sum()) >= 16.0 else 0.0
                return crop_img, crop_mask, eff_label

        # Random 8x8-aligned crop for Authentic images
        y_steps = max(0, max_y // 8)
        x_steps = max(0, max_x // 8)
        y0 = random.randint(0, y_steps) * 8
        x0 = random.randint(0, x_steps) * 8
        crop_img = img_full[y0 : y0 + target_h, x0 : x0 + target_w]
        crop_mask = mask_full[y0 : y0 + target_h, x0 : x0 + target_w]
        eff_label = 1.0 if float(crop_mask.sum()) >= 16.0 else 0.0
        return crop_img, crop_mask, eff_label

    def _augment_joint(
        self, img_np: np.ndarray, mask_np: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            img_np = np.ascontiguousarray(img_np[:, ::-1, :])
            mask_np = np.ascontiguousarray(mask_np[:, ::-1])

        if random.random() < 0.5:
            img_np = np.ascontiguousarray(img_np[::-1, :, :])
            mask_np = np.ascontiguousarray(mask_np[::-1, :])

        k = random.randint(0, 3)
        if k > 0:
            img_np = np.ascontiguousarray(np.rot90(img_np, k, axes=(0, 1)))
            mask_np = np.ascontiguousarray(np.rot90(mask_np, k, axes=(0, 1)))

        return img_np, mask_np

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        target_w, target_h = self.img_size[1], self.img_size[0]
        label = float(s["label"])

        with Image.open(s["img_path"]) as img_pil:
            img_rgb = img_pil.convert("RGB")
            orig_w, orig_h = img_rgb.size

            use_native_crop = (
                self.is_train
                and orig_h >= target_h
                and orig_w >= target_w
                and random.random() < self.native_crop_prob
            )

            if use_native_crop:
                img_full = np.array(img_rgb, dtype=np.uint8)
                mask_full = np.zeros((orig_h, orig_w), dtype=np.float32)
                if label == 1.0 and s["mask_path"]:
                    try:
                        with Image.open(s["mask_path"]) as mask_pil:
                            mask_pil = mask_pil.convert("L")
                            if mask_pil.size != (orig_w, orig_h):
                                mask_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
                            mask_full = (
                                np.array(mask_pil, dtype=np.float32) / 255.0 > 0.5
                            ).astype(np.float32)
                    except Exception:
                        pass
                img_np, mask_np, label = self._extract_jpeg_aligned_crop(
                    img_full, mask_full, target_h, target_w, label
                )
            else:
                img_resized = img_rgb.resize((target_w, target_h), Image.BILINEAR)
                img_np = np.array(img_resized, dtype=np.uint8)
                mask_np = np.zeros((target_h, target_w), dtype=np.float32)
                if label == 1.0 and s["mask_path"]:
                    try:
                        with Image.open(s["mask_path"]) as mask_pil:
                            mask_pil = mask_pil.convert("L").resize(
                                (target_w, target_h), Image.NEAREST
                            )
                            mask_raw = np.array(mask_pil, dtype=np.float32) / 255.0
                            mask_np = (mask_raw > 0.5).astype(np.float32)
                    except Exception:
                        pass

        if self.is_train:
            img_np, mask_np = self._augment_joint(img_np, mask_np)

        if self.return_mra and self.mra is not None:
            feat_np = self.mra(img_np)
            img_tensor = torch.from_numpy(feat_np).float()
        else:
            img_tensor = (
                torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            )

        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
        label_tensor = torch.tensor([label], dtype=torch.float32)
        return img_tensor, mask_tensor, label_tensor


def build_dataloaders(
    data_dir: str,
    img_size: int = 192,
    batch_size: int = 16,
    num_workers: int = 2,
    train_split: float = 0.85,
    use_forensic_maps: bool = False,
    ela_quality: int = 90,
    native_crop_prob: float = 0.5,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    samples, stats = scan_casia_forensic_dir(data_dir)

    au_samples = [s for s in samples if s["label"] == 0.0]
    tp_samples = [s for s in samples if s["label"] == 1.0]

    rng = random.Random(seed)
    rng.shuffle(au_samples)
    rng.shuffle(tp_samples)

    n_au_train = int(len(au_samples) * train_split)
    n_tp_train = int(len(tp_samples) * train_split)

    train_samples = au_samples[:n_au_train] + tp_samples[:n_tp_train]
    val_samples = au_samples[n_au_train:] + tp_samples[n_tp_train:]

    if len(val_samples) == 0:
        val_samples = train_samples

    rng.shuffle(train_samples)

    train_ds = CASIADataset(
        train_samples,
        img_size=(img_size, img_size),
        is_train=True,
        return_mra=use_forensic_maps,
        use_forensic_maps=use_forensic_maps,
        ela_quality=ela_quality,
        native_crop_prob=native_crop_prob,
    )
    val_ds = CASIADataset(
        val_samples,
        img_size=(img_size, img_size),
        is_train=False,
        return_mra=use_forensic_maps,
        use_forensic_maps=use_forensic_maps,
        ela_quality=ela_quality,
        native_crop_prob=0.0,
    )

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=(len(train_ds) >= batch_size),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )

    stats["train"] = len(train_ds)
    stats["val"] = len(val_ds)
    return train_loader, val_loader, stats
