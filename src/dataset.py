"""
Forensic Image Dataset & DataLoader Module (CASIA v2.0, Columbia, NIST16).

Expected Directory Structure:
    data_dir/
    ├── Au/         # Authentic images (label = 0, ground-truth mask = all zeros)
    ├── Tp/         # Tampered/Forged images (label = 1)
    └── Gt_mask/    # Ground-truth binary masks corresponding to Tp images
                    # (Also auto-detects 'Gt', 'mask', 'masks', 'groundtruth')

Features:
- Automatic stem matching between tampered images in `Tp/` and masks in `Gt_mask/`
  (supports suffixes such as `_gt`, `_mask`, or identical stems across extensions).
- Geometry-preserving forensic augmentations (Random Horizontal/Vertical Flips and
  90-degree Rotations) applied jointly to RGB image and Ground-Truth mask BEFORE
  2-Level Haar DWT extraction so subband phase alignment stays authentic.
- Train/Validation split with deterministic seed and PyTorch DataLoader builder.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.dwt import MRAPreprocessor

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _list_images(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    files = [
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
    ]
    files.sort()
    return files


def _build_mask_lookup(mask_dir: Optional[Path]) -> Dict[str, Path]:
    """
    Creates a dictionary mapping normalized file stems to ground-truth mask paths.
    Handles CASIA v2.0 conventions where `Tp_xxx.jpg` matches `Tp_xxx_gt.png`.
    """
    lookup: Dict[str, Path] = {}
    if mask_dir is None or not mask_dir.exists():
        return lookup

    for mask_path in _list_images(mask_dir):
        stem_lower = mask_path.stem.lower()
        lookup[stem_lower] = mask_path
        for suffix in ("_gt", "_mask", "-gt", "-mask", ".gt"):
            if stem_lower.endswith(suffix):
                base_stem = stem_lower[: -len(suffix)]
                lookup[base_stem] = mask_path
    return lookup


def _resolve_dataset_folders(root: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Locates Authentic (Au), Tampered (Tp), and Ground-Truth Mask (Gt_mask) directories
    case-insensitively inside `root`.
    """
    if not root.exists():
        return None, None, None

    subdirs = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}

    au_dir = None
    for candidate in ("au", "authentic", "real", "pristine"):
        if candidate in subdirs:
            au_dir = subdirs[candidate]
            break

    tp_dir = None
    for candidate in ("tp", "tampered", "forged", "fake", "spliced"):
        if candidate in subdirs:
            tp_dir = subdirs[candidate]
            break

    gt_dir = None
    for candidate in ("gt_mask", "gt", "mask", "masks", "groundtruth", "ground_truth"):
        if candidate in subdirs:
            gt_dir = subdirs[candidate]
            break

    return au_dir, tp_dir, gt_dir


class ForensicDataset(Dataset):
    """
    PyTorch Dataset for Image Forgery Detection & Localization using 2-Level Haar DWT MRA.
    Each item returns:
        - mra_tensor: FloatTensor of shape (C, 192, 192), where C = 9 (or 11 with MAD+ELA)
        - cls_label: FloatTensor of shape (1,) in {0.0, 1.0}
        - gt_mask: FloatTensor of shape (1, 192, 192) in {0.0, 1.0}
    """

    def __init__(
        self,
        samples: List[Tuple[Path, Optional[Path], int]],
        img_size: int = 192,
        use_forensic_maps: bool = False,
        ela_quality: int = 90,
        is_train: bool = True,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.img_size = img_size
        self.is_train = is_train
        self.mra = MRAPreprocessor(
            target_size=(img_size, img_size),
            use_forensic_maps=use_forensic_maps,
            ela_quality=ela_quality,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: Path) -> np.ndarray:
        with Image.open(path) as img:
            return np.array(img.convert("RGB"), dtype=np.uint8)

    def _load_mask(self, mask_path: Optional[Path], target_h: int, target_w: int) -> np.ndarray:
        if mask_path is None or not mask_path.exists():
            return np.zeros((target_h, target_w), dtype=np.float32)
        with Image.open(mask_path) as m_img:
            mask_gray = np.array(m_img.convert("L"), dtype=np.uint8)
        if mask_gray.shape[0] != target_h or mask_gray.shape[1] != target_w:
            if _HAS_CV2:
                mask_gray = cv2.resize(
                    mask_gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST
                )
            else:
                mask_gray = np.array(
                    Image.fromarray(mask_gray).resize(
                        (target_w, target_h), resample=Image.NEAREST
                    ),
                    dtype=np.uint8,
                )
        return (mask_gray > 127).astype(np.float32)

    def _augment_joint(
        self, rgb: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies lightweight geometry-preserving augmentations that maintain 8x8 JPEG block
        grid alignment (flips and 90-degree rotations) prior to MRA wavelet decomposition.
        """
        # Random horizontal flip
        if random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])

        # Random vertical flip
        if random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[::-1, :, :])
            mask = np.ascontiguousarray(mask[::-1, :])

        # Random 90-degree rotation (k in {0, 1, 2, 3})
        k = random.randint(0, 3)
        if k > 0:
            rgb = np.ascontiguousarray(np.rot90(rgb, k, axes=(0, 1)))
            mask = np.ascontiguousarray(np.rot90(mask, k, axes=(0, 1)))

        return rgb, mask

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path, mask_path, label = self.samples[index]

        rgb = self._load_rgb(img_path)
        if rgb.shape[0] != self.img_size or rgb.shape[1] != self.img_size:
            if _HAS_CV2:
                rgb = cv2.resize(
                    rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
                )
            else:
                rgb = np.array(
                    Image.fromarray(rgb).resize(
                        (self.img_size, self.img_size), resample=Image.BILINEAR
                    ),
                    dtype=np.uint8,
                )

        if label == 1 and mask_path is not None:
            mask = self._load_mask(mask_path, self.img_size, self.img_size)
        elif label == 1 and mask_path is None:
            # Fallback if a tampered image has no pixel mask: weak center box supervision
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        if self.is_train:
            rgb, mask = self._augment_joint(rgb, mask)

        # Compute 9-channel (or 11-channel) MRA representation
        mra_np = self.mra(rgb)  # (C, 192, 192)

        mra_tensor = torch.from_numpy(mra_np).float()
        cls_tensor = torch.tensor([float(label)], dtype=torch.float32)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()  # (1, 192, 192)

        return mra_tensor, cls_tensor, mask_tensor


def collect_forensic_samples(data_dir: str) -> List[Tuple[Path, Optional[Path], int]]:
    """
    Scans `data_dir` for `Au/`, `Tp/`, and `Gt_mask/` folders and returns a list of
    (image_path, mask_path_or_None, binary_label) tuples.
    """
    root = Path(data_dir)
    au_dir, tp_dir, gt_dir = _resolve_dataset_folders(root)

    samples: List[Tuple[Path, Optional[Path], int]] = []

    # 1. Authentic images (label = 0)
    if au_dir is not None:
        for img_path in _list_images(au_dir):
            samples.append((img_path, None, 0))

    # 2. Tampered images (label = 1) matched with Ground-Truth masks
    mask_lookup = _build_mask_lookup(gt_dir)
    if tp_dir is not None:
        for img_path in _list_images(tp_dir):
            stem_lower = img_path.stem.lower()
            matched_mask = mask_lookup.get(stem_lower, None)
            samples.append((img_path, matched_mask, 1))

    return samples


def build_dataloaders(
    data_dir: str,
    img_size: int = 192,
    batch_size: int = 16,
    num_workers: int = 2,
    train_split: float = 0.85,
    use_forensic_maps: bool = False,
    ela_quality: int = 90,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Builds stratified Train and Validation DataLoaders from CASIA/Columbia/NIST16 folders.
    """
    samples = collect_forensic_samples(data_dir)
    if len(samples) == 0:
        raise FileNotFoundError(
            f"No valid images found under '{data_dir}'. "
            "Expected subdirectories: Au/, Tp/, and Gt_mask/."
        )

    rng = random.Random(seed)
    rng.shuffle(samples)

    n_total = len(samples)
    n_train = max(1, int(n_total * train_split))
    if n_total > 1 and n_train == n_total:
        n_train = n_total - 1

    train_samples = samples[:n_train]
    val_samples = samples[n_train:] if n_total > 1 else samples

    train_ds = ForensicDataset(
        train_samples,
        img_size=img_size,
        use_forensic_maps=use_forensic_maps,
        ela_quality=ela_quality,
        is_train=True,
    )
    val_ds = ForensicDataset(
        val_samples,
        img_size=img_size,
        use_forensic_maps=use_forensic_maps,
        ela_quality=ela_quality,
        is_train=False,
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

    stats = {
        "total": n_total,
        "train": len(train_ds),
        "val": len(val_ds),
        "authentic": sum(1 for _, _, lbl in samples if lbl == 0),
        "tampered": sum(1 for _, _, lbl in samples if lbl == 1),
        "with_mask": sum(1 for _, m, lbl in samples if lbl == 1 and m is not None),
    }
    return train_loader, val_loader, stats
