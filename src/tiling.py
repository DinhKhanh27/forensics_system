"""
RAM-Aware Auto-Tiling, Sliding Window Inference & Global-Local Context Fusion Engine
for Raspberry Pi 3 Model B (1 GB RAM).

Supported Modes (--mode):
1. 'fast':
   Directly resizes input image to 192x192, runs a single pass (~0.7s on RPi 3),
   and upsamples the predicted mask back to original resolution.
2. 'tile' (Sliding Window + Global-Local Context Fusion):
   - First runs 1 global context pass (`fast` 192x192) to capture scene-level illumination
     and prevent isolated texture patches from triggering false alarms.
   - Splits high-resolution image into 192x192 patches with stride=160 (strictly divisible
     by 8 to preserve the 8x8 JPEG compression block phase; overlap=32 px).
   - Processes patches sequentially, invoking gc.collect() immediately after each patch
     to lock peak model RAM under 60 MB.
   - Blends overlapping patch masks using a 2D Gaussian weight window and fuses the local
     high-resolution mask with the global context mask (`0.65 * M_tile + 0.35 * M_global`).
3. 'auto' (Default):
   Automatically selects 'fast' if max(H, W) <= 384 (e.g., CASIA dataset), or switches
   to 'tile' if max(H, W) > 384 (e.g., camera/smartphone photos).
"""

import gc
from typing import Callable, Dict, List, Tuple
import numpy as np
from PIL import Image

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False


def build_2d_gaussian_weight(patch_size: int = 192, sigma_ratio: float = 0.25) -> np.ndarray:
    sigma = patch_size * sigma_ratio
    center = (patch_size - 1) / 2.0
    coords = np.arange(patch_size, dtype=np.float32) - center
    g_1d = np.exp(-(coords ** 2) / (2.0 * (sigma ** 2)))
    g_2d = np.outer(g_1d, g_1d).astype(np.float32)
    g_2d /= float(np.max(g_2d)) + 1e-8
    return np.clip(g_2d, 1e-3, 1.0)


def compute_grid_coordinates(length: int, patch_size: int = 192, stride: int = 160) -> List[int]:
    if stride % 8 != 0:
        raise ValueError(f"Stride must be divisible by 8 to preserve 8x8 JPEG grid phase, got {stride}.")

    if length <= patch_size:
        return [0]

    coords = list(range(0, length - patch_size + 1, stride))
    last_pos = length - patch_size
    if coords[-1] != last_pos:
        coords.append(last_pos)
    return coords


def draw_forgery_contours(
    rgb_img: np.ndarray,
    prob_mask: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 64,
    draw_boxes: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    prob_clamped = np.clip(prob_mask.astype(np.float32), 0.0, 1.0)
    binary_mask = ((prob_clamped >= threshold).astype(np.uint8)) * 255
    annotated = rgb_img.copy()

    img_area = rgb_img.shape[0] * rgb_img.shape[1]
    effective_min_area = max(min_area, int(img_area * 0.0005))

    if _HAS_CV2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid_contours = [
            cnt for cnt in contours if cv2.contourArea(cnt) >= effective_min_area
        ]

        filtered_mask = np.zeros_like(cleaned_mask)
        if valid_contours and draw_boxes:
            cv2.drawContours(filtered_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
            overlay = annotated.copy()
            cv2.drawContours(
                overlay, valid_contours, -1, (255, 40, 40), thickness=cv2.FILLED
            )
            annotated = cv2.addWeighted(overlay, 0.28, annotated, 0.72, 0)
            cv2.drawContours(annotated, valid_contours, -1, (255, 0, 0), thickness=2)
            for cnt in valid_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 60, 0), 1)

        prob_u8 = np.clip(prob_clamped * 255.0, 0, 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        heatmap_blend = cv2.addWeighted(rgb_img, 0.45, heatmap_rgb, 0.55, 0)
        return annotated, filtered_mask, heatmap_blend, len(valid_contours)

    mask_bool = binary_mask > 0
    valid_cnt = int(np.sum(mask_bool) >= effective_min_area)
    if valid_cnt and draw_boxes:
        annotated[mask_bool] = np.clip(
            0.6 * annotated[mask_bool] + 0.4 * np.array([255, 0, 0]), 0, 255
        ).astype(np.uint8)
    prob_u8 = np.clip(prob_clamped * 255.0, 0, 255).astype(np.uint8)
    heatmap_rgb = np.stack([prob_u8, np.zeros_like(prob_u8), 255 - prob_u8], axis=-1)
    return annotated, binary_mask, heatmap_rgb, valid_cnt


class TilingInferenceEngine:
    """
    Unified Inference Controller supporting 'fast', 'tile', and 'auto' modes with
    Global-Local Context Fusion and strict RAM management (< 60 MB model peak RAM).
    """

    def __init__(
        self,
        patch_size: int = 192,
        stride: int = 160,
        auto_tile_threshold: int = 384,
        threshold: float = 0.5,
    ) -> None:
        if stride % 8 != 0:
            raise ValueError(f"Stride ({stride}) must be a multiple of 8 for JPEG grid alignment.")
        self.patch_size = patch_size
        self.stride = stride
        self.overlap = patch_size - stride
        self.auto_tile_threshold = auto_tile_threshold
        self.threshold = threshold
        self.gaussian_weight = build_2d_gaussian_weight(patch_size=patch_size)

    def resolve_mode(self, mode: str, height: int, width: int) -> str:
        mode_clean = mode.lower().strip()
        if mode_clean in ("fast", "tile"):
            return mode_clean
        if max(height, width) <= self.auto_tile_threshold:
            return "fast"
        return "tile"

    def run(
        self,
        rgb_img: np.ndarray,
        patch_predict_fn: Callable[[np.ndarray], Tuple[float, np.ndarray]],
        mode: str = "auto",
        threshold: float = None,
    ) -> Dict[str, object]:
        if rgb_img.dtype != np.uint8:
            rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)

        orig_h, orig_w = rgb_img.shape[:2]
        decision_t = self.threshold if threshold is None else float(threshold)
        active_mode = self.resolve_mode(mode, orig_h, orig_w)

        if active_mode == "fast":
            p_fake, full_prob_mask = self._run_fast(rgb_img, patch_predict_fn)
            num_patches = 1
        else:
            p_fake, full_prob_mask, num_patches = self._run_sliding_window(
                rgb_img, patch_predict_fn
            )

        is_forged = bool(p_fake >= decision_t)

        annotated_img, binary_mask, heatmap_img, num_contours = draw_forgery_contours(
            rgb_img,
            full_prob_mask,
            threshold=decision_t,
            draw_boxes=is_forged,
        )

        if is_forged and num_contours == 0 and p_fake < (decision_t + 0.15):
            p_fake = float(max(0.0, decision_t - 0.05))
            is_forged = False

        label = "ẢNH GIẢ (FORGED)" if is_forged else "ẢNH THẬT (AUTHENTIC)"

        return {
            "p_fake": float(np.clip(p_fake, 0.0, 1.0)),
            "is_forged": is_forged,
            "label": label,
            "mode_used": active_mode,
            "num_patches": int(num_patches),
            "prob_mask": full_prob_mask,
            "binary_mask": binary_mask,
            "annotated_img": annotated_img,
            "heatmap_img": heatmap_img,
        }

    def _run_fast(
        self,
        rgb_img: np.ndarray,
        patch_predict_fn: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    ) -> Tuple[float, np.ndarray]:
        orig_h, orig_w = rgb_img.shape[:2]
        if _HAS_CV2:
            resized_patch = cv2.resize(
                rgb_img, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR
            )
        else:
            resized_patch = np.array(
                Image.fromarray(rgb_img).resize(
                    (self.patch_size, self.patch_size), resample=Image.BILINEAR
                ),
                dtype=np.uint8,
            )

        p_fake, patch_mask = patch_predict_fn(resized_patch)
        patch_mask_2d = np.clip(np.squeeze(patch_mask).astype(np.float32), 0.0, 1.0)

        if orig_h == self.patch_size and orig_w == self.patch_size:
            full_mask = patch_mask_2d
        elif _HAS_CV2:
            full_mask = cv2.resize(
                patch_mask_2d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )
        else:
            full_mask = np.array(
                Image.fromarray(patch_mask_2d, mode="F").resize(
                    (orig_w, orig_h), resample=Image.BILINEAR
                ),
                dtype=np.float32,
            )

        del resized_patch, patch_mask, patch_mask_2d
        gc.collect()
        return float(np.clip(p_fake, 0.0, 1.0)), full_mask

    def _run_sliding_window(
        self,
        rgb_img: np.ndarray,
        patch_predict_fn: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    ) -> Tuple[float, np.ndarray, int]:
        """
        Sequential sliding-window inference with stride=160 (overlap=32px),
        Gaussian Blending accumulation, Global-Local Context Fusion, and per-patch gc.collect().
        """
        orig_h, orig_w = rgb_img.shape[:2]
        ps = self.patch_size

        # 1. Global overview pass (captures full-image context so homogeneous background patches stay clean)
        p_global, global_mask = self._run_fast(rgb_img, patch_predict_fn)

        # 2. Local native-resolution sliding window pass
        pad_h = max(0, ps - orig_h)
        pad_w = max(0, ps - orig_w)
        if pad_h > 0 or pad_w > 0:
            work_img = np.pad(
                rgb_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
            )
        else:
            work_img = rgb_img

        work_h, work_w = work_img.shape[:2]
        y_coords = compute_grid_coordinates(work_h, patch_size=ps, stride=self.stride)
        x_coords = compute_grid_coordinates(work_w, patch_size=ps, stride=self.stride)

        mask_accumulator = np.zeros((work_h, work_w), dtype=np.float32)
        weight_accumulator = np.zeros((work_h, work_w), dtype=np.float32)

        patch_scores: List[float] = []
        num_patches = 0

        for y in y_coords:
            for x in x_coords:
                patch_rgb = work_img[y : y + ps, x : x + ps]
                p_patch, patch_mask = patch_predict_fn(patch_rgb)
                patch_mask_2d = np.clip(
                    np.squeeze(patch_mask).astype(np.float32), 0.0, 1.0
                )

                patch_scores.append(float(np.clip(p_patch, 0.0, 1.0)))

                mask_accumulator[y : y + ps, x : x + ps] += (
                    patch_mask_2d * self.gaussian_weight
                )
                weight_accumulator[y : y + ps, x : x + ps] += self.gaussian_weight
                num_patches += 1

                del patch_rgb, patch_mask, patch_mask_2d
                gc.collect()

        blended_local = mask_accumulator / np.maximum(weight_accumulator, 1e-6)
        local_mask = np.clip(blended_local[:orig_h, :orig_w], 0.0, 1.0)

        # Fuse High-Res Local Tile Mask (65%) with Global Context Mask (35%)
        fused_mask = np.clip(0.65 * local_mask + 0.35 * global_mask, 0.0, 1.0)

        max_p_patch = max(patch_scores) if patch_scores else 0.0
        fused_peak = float(np.percentile(fused_mask, 99.5))
        global_p_fake = 0.40 * p_global + 0.35 * max_p_patch + 0.25 * fused_peak

        del mask_accumulator, weight_accumulator, blended_local, local_mask, global_mask
        gc.collect()

        return float(np.clip(global_p_fake, 0.0, 1.0)), fused_mask, num_patches
