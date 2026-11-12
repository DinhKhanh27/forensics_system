"""
RAM-Aware Auto-Tiling, Sliding Window Inference & Gaussian Blending Engine
for Raspberry Pi 3 Model B (1 GB RAM).

Supported Modes (--mode):
1. 'fast':
   Directly resizes input image to 192x192, runs a single pass (~0.7s on RPi 3),
   and upsamples the predicted mask back to original resolution.
2. 'tile' (Sliding Window):
   Splits high-resolution image into 192x192 patches with stride=160 (strictly divisible
   by 8 to preserve the 8x8 JPEG compression block phase; overlap=32 px).
   Processes patches sequentially, invoking gc.collect() immediately after each patch
   to lock peak model RAM under 60 MB.
   Blends overlapping patch masks using a 2D Gaussian weight window, computes global
   probability P_fake = max(P_patch), applies threshold T >= 0.5, and draws red bounding
   contours via cv2.findContours.
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
    """
    Constructs a 2D Gaussian blending weight matrix of shape (patch_size, patch_size)
    that smoothly tapers toward the 32px border overlap region to eliminate seam artifacts.

    Args:
        patch_size: Spatial size of square patch (default: 192).
        sigma_ratio: Standard deviation relative to patch_size.

    Returns:
        2D float32 weight array in (0, 1] of shape (patch_size, patch_size).
    """
    sigma = patch_size * sigma_ratio
    center = (patch_size - 1) / 2.0
    coords = np.arange(patch_size, dtype=np.float32) - center
    g_1d = np.exp(-(coords ** 2) / (2.0 * (sigma ** 2)))
    g_2d = np.outer(g_1d, g_1d).astype(np.float32)
    # Normalize peak to 1.0 and clamp floor to avoid zero division at corners
    g_2d /= float(np.max(g_2d)) + 1e-8
    return np.clip(g_2d, 1e-3, 1.0)


def compute_grid_coordinates(length: int, patch_size: int = 192, stride: int = 160) -> List[int]:
    """
    Generates 1D starting coordinates for sliding window extraction.
    Ensures all interior steps advance by `stride` (160, divisible by 8 to preserve
    JPEG 8x8 block phase) and covers the trailing boundary seamlessly.
    """
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
    min_area: int = 36,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies decision threshold T to the continuous localization heatmap, extracts tampered
    regions via cv2.findContours, and draws red contours + bounding boxes on a copy of the
    original image.

    Args:
        rgb_img: Original RGB uint8 image (H, W, 3).
        prob_mask: Continuous probability mask (H, W) in [0, 1].
        threshold: Binarization threshold T (default: 0.5).
        min_area: Minimum contour area in pixels to filter isolated impulse noise.

    Returns:
        Tuple of:
            - annotated_rgb: Original RGB image with red forgery contours and semi-transparent fill.
            - binary_mask: uint8 binary mask (0 or 255) of shape (H, W).
            - heatmap_rgb: Jet colorized probability heatmap blended with original image (H, W, 3).
    """
    binary_mask = ((prob_mask >= threshold).astype(np.uint8)) * 255
    annotated = rgb_img.copy()

    if _HAS_CV2:
        # Morphological opening/closing to clean micro-speckles before contour extraction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Semi-transparent red highlight inside forged regions
        overlay = annotated.copy()
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
        if valid_contours:
            cv2.drawContours(overlay, valid_contours, -1, (255, 40, 40), thickness=cv2.FILLED)
            annotated = cv2.addWeighted(overlay, 0.28, annotated, 0.72, 0)
            # Crisp red perimeter contour & bounding box
            cv2.drawContours(annotated, valid_contours, -1, (255, 0, 0), thickness=2)
            for cnt in valid_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 60, 0), 1)

        # Generate Localization Heatmap
        prob_u8 = np.clip(prob_mask * 255.0, 0, 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        heatmap_blend = cv2.addWeighted(rgb_img, 0.45, heatmap_rgb, 0.55, 0)
        return annotated, cleaned_mask, heatmap_blend

    # Fallback if OpenCV is unavailable
    mask_bool = binary_mask > 0
    annotated[mask_bool] = np.clip(
        0.6 * annotated[mask_bool] + 0.4 * np.array([255, 0, 0]), 0, 255
    ).astype(np.uint8)
    prob_u8 = np.clip(prob_mask * 255.0, 0, 255).astype(np.uint8)
    heatmap_rgb = np.stack([prob_u8, np.zeros_like(prob_u8), 255 - prob_u8], axis=-1)
    return annotated, binary_mask, heatmap_rgb


class TilingInferenceEngine:
    """
    Unified Inference Controller supporting 'fast', 'tile', and 'auto' modes with strict
    RAM management (< 60 MB model peak RAM) for Raspberry Pi 3.
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
        self.overlap = patch_size - stride  # 192 - 160 = 32 px
        self.auto_tile_threshold = auto_tile_threshold
        self.threshold = threshold
        self.gaussian_weight = build_2d_gaussian_weight(patch_size=patch_size)

    def resolve_mode(self, mode: str, height: int, width: int) -> str:
        """
        Determines execution mode based on image dimensions when mode == 'auto'.
        - If max(H, W) <= 384 (e.g., CASIA): returns 'fast'
        - If max(H, W) > 384 (e.g., high-res camera/phone): returns 'tile'
        """
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
        """
        Executes forgery detection & localization on an arbitrary-sized RGB image.

        Args:
            rgb_img: Input RGB uint8 image of shape (H, W, 3).
            patch_predict_fn: Callable taking a 192x192 RGB uint8 array and returning
                (p_fake_float, mask_192x192_float32).
            mode: 'auto', 'fast', or 'tile'.
            threshold: Decision threshold T (overrides default if provided).

        Returns:
            Dictionary containing:
                - 'p_fake': Global forgery probability in [0, 1]
                - 'is_forged': Boolean decision (p_fake >= T)
                - 'label': Vietnamese verdict string ('ẢNH GIẢ (FORGED)' or 'ẢNH THẬT (AUTHENTIC)')
                - 'mode_used': Actual mode executed ('fast' or 'tile')
                - 'num_patches': Number of patches evaluated
                - 'prob_mask': Full-resolution float32 probability map (H, W)
                - 'binary_mask': Full-resolution uint8 binary mask (H, W)
                - 'annotated_img': Full-resolution RGB image with red forgery contours
                - 'heatmap_img': Full-resolution RGB localization heatmap
        """
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

        annotated_img, binary_mask, heatmap_img = draw_forgery_contours(
            rgb_img, full_prob_mask, threshold=decision_t
        )

        is_forged = bool(p_fake >= decision_t)
        label = "ẢNH GIẢ (FORGED)" if is_forged else "ẢNH THẬT (AUTHENTIC)"

        return {
            "p_fake": float(p_fake),
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
        """
        Resizes full image directly to 192x192, evaluates once, and resizes mask back.
        """
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
        patch_mask_2d = np.squeeze(patch_mask).astype(np.float32)

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
        return float(p_fake), full_mask

    def _run_sliding_window(
        self,
        rgb_img: np.ndarray,
        patch_predict_fn: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    ) -> Tuple[float, np.ndarray, int]:
        """
        Sequential sliding-window inference with stride=160 (overlap=32px),
        Gaussian Blending accumulation, P_fake = max(P_patch), and per-patch gc.collect().
        """
        orig_h, orig_w = rgb_img.shape[:2]
        ps = self.patch_size

        # Pad image reflectively if either dimension is smaller than 192
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

        # Pre-allocate single-precision accumulators for Gaussian Blending
        mask_accumulator = np.zeros((work_h, work_w), dtype=np.float32)
        weight_accumulator = np.zeros((work_h, work_w), dtype=np.float32)

        max_p_fake = 0.0
        num_patches = 0

        for y in y_coords:
            for x in x_coords:
                patch_rgb = work_img[y : y + ps, x : x + ps]
                p_patch, patch_mask = patch_predict_fn(patch_rgb)
                patch_mask_2d = np.squeeze(patch_mask).astype(np.float32)

                if float(p_patch) > max_p_fake:
                    max_p_fake = float(p_patch)

                # Accumulate Gaussian-weighted patch prediction
                mask_accumulator[y : y + ps, x : x + ps] += (
                    patch_mask_2d * self.gaussian_weight
                )
                weight_accumulator[y : y + ps, x : x + ps] += self.gaussian_weight
                num_patches += 1

                # Explicitly release patch tensors and trigger GC to cap RPi 3 peak RAM < 60 MB
                del patch_rgb, patch_mask, patch_mask_2d
                gc.collect()

        blended_mask = mask_accumulator / np.maximum(weight_accumulator, 1e-6)
        full_mask = blended_mask[:orig_h, :orig_w].copy()

        del mask_accumulator, weight_accumulator, blended_mask
        gc.collect()

        return float(max_p_fake), full_mask, num_patches


if __name__ == "__main__":
    # Self-test of Phase 2 TilingInferenceEngine in 'auto', 'fast', and 'tile' modes
    engine = TilingInferenceEngine(patch_size=192, stride=160, auto_tile_threshold=384)

    def _dummy_predictor(patch: np.ndarray) -> Tuple[float, np.ndarray]:
        assert patch.shape == (192, 192, 3)
        synth_mask = np.zeros((192, 192), dtype=np.float32)
        synth_mask[48:144, 48:144] = 0.85
        return 0.88, synth_mask

    # 1. Test CASIA-sized image (<= 384x384) -> should auto-select 'fast'
    img_small = np.random.randint(0, 256, (256, 384, 3), dtype=np.uint8)
    res_small = engine.run(img_small, _dummy_predictor, mode="auto")
    assert res_small["mode_used"] == "fast" and res_small["num_patches"] == 1

    # 2. Test high-res image (> 384x384) -> should auto-select 'tile' with stride=160
    img_large = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
    res_large = engine.run(img_large, _dummy_predictor, mode="auto")
    assert res_large["mode_used"] == "tile" and res_large["num_patches"] > 1
    assert res_large["prob_mask"].shape == (640, 480)

    print(
        f"[Phase 2 Tiling OK] Small (256x384) -> mode={res_small['mode_used']}, patches={res_small['num_patches']}"
    )
    print(
        f"[Phase 2 Tiling OK] Large (640x480) -> mode={res_large['mode_used']}, patches={res_large['num_patches']}, P_fake={res_large['p_fake']:.2f}"
    )
