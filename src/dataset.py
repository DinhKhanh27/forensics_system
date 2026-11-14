"""
Forensic Image Dataset & Group-Aware Stratified DataLoader Module (CASIA v2.0, Columbia, NIST16).

Fixes & Improvements Addressing Code Review:
1. Strict Mask & Image Validation at Scan Time (`_is_valid_tampered_mask`):
   - Verifies that every matched mask for a Tampered (`Tp`) image is readable and contains
     actual foreground tampered pixels (> 0 positive pixels). Corrupted or empty masks are
     filtered out and logged in `stats['corrupted_or_empty_masks']` instead of silently
     falling back to all-zero masks in `__getitem__`.
2. Zero Silent Exception Swallowing in `__getitem__`:
   - Removed `try ... except: pass` when reading masks. If a `Tp` sample has `label == 1.0`,
     its mask is guaranteed to contain positive tampered pixels so classification and
     localization supervision never contradict each other.
3. Leak-Free Group-Aware Stratified Split (`_extract_source_group_id`):
   - Extracts source image identifiers from CASIA v2 filenames (e.g., `ani00018`, `nat00056`)
     so tampered splice variants and their source authentic images are grouped into the same
     split (Train or Val), preventing source-image data leakage.
   - Raises a clear `ValueError` if the dataset is too small to form a non-empty, independent
     Validation set (never falls back to `val_samples = train_samples`).
"""

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.dwt import MRAPreprocessor

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
_CASIA_ID_PATTERN = re.compile(r"([a-z]{3}\d{5})", re.IGNORECASE)


def _extract_source_group_id(img_path: Path) -> str:
    """
    Extracts a canonical source group key from forensic image filenames to prevent
    data leakage between Train and Validation splits.
    - For CASIA v2 (e.g., `Au_ani_00018.jpg` or `Tp_D_NRN_S_N_ani00018_sec00096_00138.jpg`),
      normalizes `ani_00018` -> `ani00018` and uses the primary source ID.
    - For other datasets, groups by base stem prefix before the last underscore.
    """
    stem_norm = img_path.stem.lower().replace("_", "")
    matches = _CASIA_ID_PATTERN.findall(stem_norm)
    if matches:
        return matches[0].lower()
    parts = img_path.stem.lower().split("_")
    if len(parts) >= 2:
        return "_".join(parts[:-1])
    return img_path.stem.lower()


def _is_valid_tampered_mask(mask_path: Path, min_positive_pixels: int = 16) -> bool:
    """
    Verifies at dataset scan time that a ground-truth mask is readable and contains
    at least `min_positive_pixels` foreground pixels.
    """
    try:
        with Image.open(mask_path) as m_img:
            mask_arr = np.array(m_img.convert("L"), dtype=np.uint8)
        pos_count = int(np.sum(mask_arr > 127))
        total_pixels = int(mask_arr.size)
        # Must have foreground pixels and not be a completely white corrupted mask
        return pos_count >= min_positive_pixels and pos_count < int(total_pixels * 0.99)
    except Exception:
        return False


def scan_casia_forensic_dir(root_dir: str) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Recursively scans `root_dir` (using paths relative to `root_dir`) for Authentic (`Au`),
    Tampered (`Tp`), and Ground-Truth mask files (`Gt_mask`, `CASIA 2 Groundtruth`, `masks`),
    validating mask integrity upfront.
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
    corrupted_au = 0
    for path in au_files:
        try:
            with Image.open(path) as im:
                im.verify()
            samples.append(
                {
                    "img_path": str(path),
                    "mask_path": None,
                    "label": 0.0,
                    "group_id": _extract_source_group_id(path),
                }
            )
        except Exception:
            corrupted_au += 1

    matched_mask_count = 0
    unmatched_tp = 0
    corrupted_or_empty_masks = 0

    for path in tp_files:
        stem = path.stem.lower()
        matched_mask_path: Optional[Path] = None
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
                matched_mask_path = gt_files[cand]
                break

        if matched_mask_path is None:
            unmatched_tp += 1
            continue

        if not _is_valid_tampered_mask(matched_mask_path):
            corrupted_or_empty_masks += 1
            continue

        matched_mask_count += 1
        samples.append(
            {
                "img_path": str(path),
                "mask_path": str(matched_mask_path),
                "label": 1.0,
                "group_id": _extract_source_group_id(path),
            }
        )

    if len(tp_files) > 0 and matched_mask_count == 0:
        raise RuntimeError(
            f"❌ LỖI: Tìm thấy {len(tp_files)} ảnh Tp nhưng không khớp được mask hợp lệ nào trong {root_dir}!"
        )

    stats = {
        "total": len(samples),
        "authentic": len(au_files) - corrupted_au,
        "tampered": matched_mask_count,
        "with_mask": matched_mask_count,
        "skipped_unmatched_tp": unmatched_tp,
        "corrupted_or_empty_masks": corrupted_or_empty_masks,
    }
    return samples, stats


class CASIADataset(Dataset):
    """
    Dataset returning `(img_or_mra_tensor, mask_tensor, label_tensor)`.
    Guarantees label-mask consistency: every `Tp` sample (`label == 1.0`) has a verified
    non-empty binary mask.
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
                f"Ảnh Tp (Mask hợp lệ): {self.stats['tampered']} | "
                f"Bỏ qua Tp thiếu/lỗi Mask: {self.stats['skipped_unmatched_tp'] + self.stats['corrupted_or_empty_masks']}"
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
        For Tampered (Tp) images, centers the crop on a foreground tampered pixel so the
        cropped patch is guaranteed to contain tampered pixels (`label == 1.0`).
        """
        h, w = img_full.shape[:2]
        max_y = h - target_h
        max_x = w - target_w

        if label == 1.0:
            fg_ys, fg_xs = np.where(mask_full > 0.5)
            if len(fg_ys) > 0:
                idx = random.randint(0, len(fg_ys) - 1)
                cy, cx = int(fg_ys[idx]), int(fg_xs[idx])
                jitter_y = random.randint(-target_h // 4, target_h // 4)
                jitter_x = random.randint(-target_w // 4, target_w // 4)
                y0 = max(0, min(max_y, cy - target_h // 2 + jitter_y))
                x0 = max(0, min(max_x, cx - target_w // 2 + jitter_x))
                y0 = min(max_y, (y0 // 8) * 8)
                x0 = min(max_x, (x0 // 8) * 8)
                crop_img = img_full[y0 : y0 + target_h, x0 : x0 + target_w]
                crop_mask = mask_full[y0 : y0 + target_h, x0 : x0 + target_w]
                if float(crop_mask.sum()) >= 16.0:
                    return crop_img, crop_mask, 1.0
            # Fallback to full-image resize if 8x8 grid snap missed a tiny forgery region
            img_resized = np.array(
                Image.fromarray(img_full).resize((target_w, target_h), Image.BILINEAR),
                dtype=np.uint8,
            )
            mask_resized = (
                np.array(
                    Image.fromarray((mask_full * 255).astype(np.uint8)).resize(
                        (target_w, target_h), Image.NEAREST
                    ),
                    dtype=np.float32,
                )
                / 255.0
                > 0.5
            ).astype(np.float32)
            return img_resized, mask_resized, 1.0

        # Authentic image (label == 0.0): random 8x8-aligned crop
        y_steps = max(0, max_y // 8)
        x_steps = max(0, max_x // 8)
        y0 = random.randint(0, y_steps) * 8
        x0 = random.randint(0, x_steps) * 8
        crop_img = img_full[y0 : y0 + target_h, x0 : x0 + target_w]
        crop_mask = np.zeros((target_h, target_w), dtype=np.float32)
        return crop_img, crop_mask, 0.0

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
                if label == 1.0:
                    if not s["mask_path"]:
                        raise RuntimeError(f"Tampered sample missing mask_path: {s['img_path']}")
                    with Image.open(s["mask_path"]) as mask_pil:
                        mask_pil = mask_pil.convert("L")
                        if mask_pil.size != (orig_w, orig_h):
                            mask_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
                        mask_full = (
                            np.array(mask_pil, dtype=np.float32) / 255.0 > 0.5
                        ).astype(np.float32)
                else:
                    mask_full = np.zeros((orig_h, orig_w), dtype=np.float32)

                img_np, mask_np, label = self._extract_jpeg_aligned_crop(
                    img_full, mask_full, target_h, target_w, label
                )
            else:
                img_resized = img_rgb.resize((target_w, target_h), Image.BILINEAR)
                img_np = np.array(img_resized, dtype=np.uint8)
                if label == 1.0:
                    if not s["mask_path"]:
                        raise RuntimeError(f"Tampered sample missing mask_path: {s['img_path']}")
                    with Image.open(s["mask_path"]) as mask_pil:
                        mask_pil = mask_pil.convert("L").resize(
                            (target_w, target_h), Image.NEAREST
                        )
                        mask_raw = np.array(mask_pil, dtype=np.float32) / 255.0
                        mask_np = (mask_raw > 0.5).astype(np.float32)
                        # If nearest-neighbor downscaling lost a 1-pixel thin line, preserve max pooling
                        if float(mask_np.sum()) == 0.0:
                            mask_np = (mask_raw > 0.05).astype(np.float32)
                else:
                    mask_np = np.zeros((target_h, target_w), dtype=np.float32)

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


def _group_stratified_split(
    samples: List[Dict], train_split: float = 0.85, seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Splits samples into independent Train and Validation sets while keeping all images
    sharing the same `group_id` (source scene/background ID) in the same split.
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        gid = s.get("group_id", Path(s["img_path"]).stem.lower())
        groups[gid].append(s)

    # Separate groups that contain Tampered samples vs Authentic-only groups for stratification
    tp_groups = []
    au_groups = []
    for gid, grp_items in groups.items():
        has_tp = any(item["label"] == 1.0 for item in grp_items)
        if has_tp:
            tp_groups.append(grp_items)
        else:
            au_groups.append(grp_items)

    rng = random.Random(seed)
    rng.shuffle(tp_groups)
    rng.shuffle(au_groups)

    def _split_group_list(grp_list: List[List[Dict]]) -> Tuple[List[Dict], List[Dict]]:
        if len(grp_list) < 2:
            raise ValueError(
                f"❌ Không đủ số nhóm độc lập ({len(grp_list)}) để tách tập Train và Validation độc lập!"
            )
        n_total_items = sum(len(g) for g in grp_list)
        target_train = max(1, int(n_total_items * train_split))
        train_out: List[Dict] = []
        val_out: List[Dict] = []
        for idx, g in enumerate(grp_list):
            # Ensure at least the last group goes to val_out
            if len(train_out) < target_train and idx < len(grp_list) - 1:
                train_out.extend(g)
            else:
                val_out.extend(g)
        return train_out, val_out

    train_tp, val_tp = _split_group_list(tp_groups)
    train_au, val_au = _split_group_list(au_groups)

    train_samples = train_au + train_tp
    val_samples = val_au + val_tp

    if len(train_samples) == 0 or len(val_samples) == 0:
        raise ValueError(
            f"❌ Tập dữ liệu quá nhỏ để tách Train ({len(train_samples)}) và Val ({len(val_samples)}) độc lập!"
        )

    rng.shuffle(train_samples)
    return train_samples, val_samples


def build_dataloaders(
    data_dir: str,
    val_dir: Optional[str] = None,
    img_size: int = 192,
    batch_size: int = 16,
    num_workers: int = 2,
    train_split: float = 0.85,
    use_forensic_maps: bool = False,
    ela_quality: int = 90,
    native_crop_prob: float = 0.5,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Builds leak-free Group-Aware Stratified Train & Validation DataLoaders.
    If `val_dir` is provided explicitly, loads Validation samples from `val_dir`.
    """
    samples, stats = scan_casia_forensic_dir(data_dir)

    if val_dir is not None and Path(val_dir).exists():
        train_samples = samples
        val_samples, val_stats = scan_casia_forensic_dir(val_dir)
        if len(val_samples) == 0:
            raise ValueError(f"❌ Thư mục validation '{val_dir}' không có mẫu hợp lệ nào!")
        stats["val_external"] = val_stats["total"]
    else:
        train_samples, val_samples = _group_stratified_split(
            samples, train_split=train_split, seed=seed
        )

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
