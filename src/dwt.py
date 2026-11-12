"""
Multiresolution Analysis (MRA) Preprocessing Module.
Implements:
1. Pure NumPy 2-Level 2D Haar Discrete Wavelet Transform (DWT) - zero PyWavelets dependency
   for ultra-low RAM overhead on Raspberry Pi 3.
2. Optional Explicit Forensic Maps (--use-forensic-maps):
   - Donoho MAD Local Noise Variance Map from HH1 subband.
   - Error Level Analysis (ELA) Map via JPEG Q=90 recompression.
3. Early Fusion: Bilinear interpolation of high-frequency subbands to 192x192,
   Z-score normalization, and concatenation with RGB into a 9-channel (or 11-channel) tensor.
"""

import io
from typing import Dict, Tuple
import numpy as np
from PIL import Image

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False


def _resize_2d_or_3d(arr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Bilinear resize using OpenCV (fast C++ path) with PIL fallback."""
    if arr.shape[0] == target_h and arr.shape[1] == target_w:
        return arr
    if _HAS_CV2:
        return cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if arr.ndim == 2:
        pil_img = Image.fromarray(arr.astype(np.float32), mode="F")
        return np.array(pil_img.resize((target_w, target_h), resample=Image.BILINEAR), dtype=np.float32)
    pil_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return np.array(pil_img.resize((target_w, target_h), resample=Image.BILINEAR), dtype=arr.dtype)


def haar_dwt2d_single_level(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes 1-level 2D Haar Discrete Wavelet Transform using fast vectorized slicing.
    Pads odd dimensions symmetrically if needed.

    Args:
        gray: 2D float32 array of shape (H, W).

    Returns:
        Tuple of (LL, LH, HL, HH), each of shape (H // 2, W // 2).
    """
    h, w = gray.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h != 0 or pad_w != 0:
        gray = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="reflect")

    # 2x2 block decomposition
    x00 = gray[0::2, 0::2]
    x01 = gray[0::2, 1::2]
    x10 = gray[1::2, 0::2]
    x11 = gray[1::2, 1::2]

    # Normalized 2D Haar filters (scale factor 0.5)
    ll = 0.5 * (x00 + x01 + x10 + x11)
    lh = 0.5 * (x00 - x01 + x10 - x11)  # Horizontal high-frequency details
    hl = 0.5 * (x00 + x01 - x10 - x11)  # Vertical high-frequency details
    hh = 0.5 * (x00 - x01 - x10 + x11)  # Diagonal high-frequency details

    return ll, lh, hl, hh


def compute_2level_haar_dwt(rgb_img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Performs 2-Level 2D Haar DWT on the luminance channel of an RGB image.
    - Level 1 (W/2 x H/2): [LL1, LH1, HL1, HH1] captures micro-noise & splicing boundaries.
    - Level 2 (W/4 x H/4): Decomposes LL1 into [LL2, LH2, HL2, HH2] to capture 8x8 JPEG block
      grid inconsistencies and abnormal illumination gradients.

    Args:
        rgb_img: uint8 or float32 RGB image of shape (H, W, 3).

    Returns:
        Dictionary containing LL1, LH1, HL1, HH1, LL2, LH2, HL2, HH2 as float32 arrays.
    """
    if rgb_img.dtype == np.uint8:
        rgb_float = rgb_img.astype(np.float32)
    else:
        rgb_float = rgb_img.astype(np.float32)
        if rgb_float.max() <= 1.0:
            rgb_float = rgb_float * 255.0

    # ITU-R BT.601 luminance extraction
    luminance = (
        0.299 * rgb_float[:, :, 0]
        + 0.587 * rgb_float[:, :, 1]
        + 0.114 * rgb_float[:, :, 2]
    )

    ll1, lh1, hl1, hh1 = haar_dwt2d_single_level(luminance)
    ll2, lh2, hl2, hh2 = haar_dwt2d_single_level(ll1)

    return {
        "LL1": ll1,
        "LH1": lh1,
        "HL1": hl1,
        "HH1": hh1,
        "LL2": ll2,
        "LH2": lh2,
        "HL2": hl2,
        "HH2": hh2,
    }


def compute_donoho_mad_map(hh1: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Computes the local noise standard deviation map inspired by Donoho's Median Absolute
    Deviation (MAD) estimator from the HH1 subband:
        sigma_hat = median(|HH1|) / 0.6745

    Args:
        hh1: 2D float32 array of the HH1 subband (H/2, W/2).
        window_size: Local sliding window size for spatial noise estimation.

    Returns:
        2D float32 local noise discrepancy map at (H/2, W/2) resolution.
    """
    abs_hh1 = np.abs(hh1).astype(np.float32)
    global_sigma = float(np.median(abs_hh1)) / 0.6745

    ksize = window_size if window_size % 2 == 1 else window_size + 1
    if _HAS_CV2 and ksize <= 5:
        local_stat = cv2.medianBlur(abs_hh1, ksize)
    elif _HAS_CV2:
        local_stat = cv2.GaussianBlur(abs_hh1, (ksize, ksize), 0)
    else:
        # Pure NumPy separable box filter fallback
        pad = ksize // 2
        padded = np.pad(abs_hh1, pad, mode="reflect")
        kernel = np.ones(ksize, dtype=np.float32) / ksize
        temp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=padded)
        local_stat = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=1, arr=temp)

    local_sigma_map = local_stat / 0.6745
    mad_discrepancy = np.abs(local_sigma_map - global_sigma)
    return mad_discrepancy.astype(np.float32)


def compute_ela_map(rgb_img: np.ndarray, quality: int = 90) -> np.ndarray:
    """
    Computes Error Level Analysis (ELA) map by recompressing the RGB image at JPEG Q=90%
    in memory (no disk I/O overhead) and measuring the mean absolute pixel error across channels.

    Args:
        rgb_img: uint8 RGB image of shape (H, W, 3).
        quality: JPEG compression quality (default: 90).

    Returns:
        2D float32 ELA map of shape (H, W).
    """
    if rgb_img.dtype != np.uint8:
        img_uint8 = np.clip(rgb_img, 0, 255).astype(np.uint8)
    else:
        img_uint8 = rgb_img

    if _HAS_CV2:
        bgr_img = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        success, encoded_buf = cv2.imencode(".jpg", bgr_img, encode_param)
        if success:
            recompressed_bgr = cv2.imdecode(encoded_buf, cv2.IMREAD_COLOR)
            recompressed_rgb = cv2.cvtColor(recompressed_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            diff = np.abs(img_uint8.astype(np.float32) - recompressed_rgb)
            return np.mean(diff, axis=2).astype(np.float32)

    pil_img = Image.fromarray(img_uint8)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed_rgb = np.array(Image.open(buffer).convert("RGB"), dtype=np.float32)
    diff = np.abs(img_uint8.astype(np.float32) - recompressed_rgb)
    return np.mean(diff, axis=2).astype(np.float32)


def zscore_normalize(channel: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Applies per-channel Z-score normalization: (X - mu) / (sigma + eps).
    """
    mean = float(np.mean(channel))
    std = float(np.std(channel))
    return ((channel - mean) / (std + eps)).astype(np.float32)


class MRAPreprocessor:
    """
    Multiresolution Analysis (MRA) Preprocessor for DWT-MobileUNet.
    Constructs a 9-channel (or 11-channel with --use-forensic-maps) tensor of shape (C, H, W):
      - Channels 0..2: Z-score normalized RGB (resized to target_size x target_size)
      - Channels 3..5: Bilinearly interpolated & Z-score normalized Level-1 DWT [LH1, HL1, HH1]
      - Channels 6..8: Bilinearly interpolated & Z-score normalized Level-2 DWT [LH2, HL2, HH2]
      - Channels 9..10 (Optional): Donoho MAD noise map + ELA (Q=90) map
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (192, 192),
        use_forensic_maps: bool = False,
        ela_quality: int = 90,
    ) -> None:
        self.target_size = target_size  # (H, W) = (192, 192)
        self.use_forensic_maps = use_forensic_maps
        self.ela_quality = ela_quality
        self.out_channels = 11 if use_forensic_maps else 9

    def __call__(self, rgb_img: np.ndarray) -> np.ndarray:
        """
        Transforms an input RGB image into an early-fusion multi-channel float32 tensor.

        Args:
            rgb_img: Input RGB numpy array (H, W, 3) in uint8 [0, 255].

        Returns:
            Float32 array of shape (C, 192, 192) where C = 9 or 11.
        """
        target_h, target_w = self.target_size

        if rgb_img.dtype != np.uint8:
            rgb_uint8 = np.clip(rgb_img, 0, 255).astype(np.uint8)
        else:
            rgb_uint8 = rgb_img

        # Resize RGB to target_size (192x192) if not already matching
        rgb_resized = _resize_2d_or_3d(rgb_uint8, target_w, target_h)

        # 1. Compute 2-Level Haar DWT on the target patch/image
        dwt_subbands = compute_2level_haar_dwt(rgb_resized)

        # 2. Prepare output tensor (C, 192, 192)
        fused = np.empty((self.out_channels, target_h, target_w), dtype=np.float32)

        # Channels 0..2: Z-score normalized RGB
        rgb_float = rgb_resized.astype(np.float32)
        for c in range(3):
            fused[c] = zscore_normalize(rgb_float[:, :, c])

        # Channels 3..8: Bilinear interpolation of [LH1, HL1, HH1, LH2, HL2, HH2] -> 192x192 + Z-score
        high_freq_keys = ["LH1", "HL1", "HH1", "LH2", "HL2", "HH2"]
        for idx, key in enumerate(high_freq_keys, start=3):
            subband = dwt_subbands[key]
            upsampled = _resize_2d_or_3d(subband, target_w, target_h)
            fused[idx] = zscore_normalize(upsampled)

        # Channels 9..10 (Optional): Donoho MAD from HH1 + ELA (Q=90)
        if self.use_forensic_maps:
            mad_map = compute_donoho_mad_map(dwt_subbands["HH1"])
            mad_upsampled = _resize_2d_or_3d(mad_map, target_w, target_h)
            fused[9] = zscore_normalize(mad_upsampled)

            ela_map = compute_ela_map(rgb_resized, quality=self.ela_quality)
            fused[10] = zscore_normalize(ela_map)

        return fused

    def extract_visualizations(self, rgb_img: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generates human-interpretable uint8 visualization maps of Level-1/Level-2 DWT energy,
        Donoho MAD noise map, and ELA map for the Gradio & Flask UIs.
        """
        target_h, target_w = self.target_size
        rgb_resized = _resize_2d_or_3d(rgb_img, target_w, target_h)
        dwt = compute_2level_haar_dwt(rgb_resized)

        def _norm_uint8(arr: np.ndarray) -> np.ndarray:
            arr_abs = np.abs(arr)
            p99 = float(np.percentile(arr_abs, 99.0)) + 1e-6
            scaled = np.clip(arr_abs / p99 * 255.0, 0, 255).astype(np.uint8)
            return _resize_2d_or_3d(scaled, target_w, target_h)

        # High-frequency energy composite at Level 1 and Level 2
        e_l1 = np.sqrt(dwt["LH1"] ** 2 + dwt["HL1"] ** 2 + dwt["HH1"] ** 2)
        e_l2 = np.sqrt(dwt["LH2"] ** 2 + dwt["HL2"] ** 2 + dwt["HH2"] ** 2)

        def _colorize(gray_u8: np.ndarray, cv_cmap: int) -> np.ndarray:
            if _HAS_CV2:
                bgr = cv2.applyColorMap(gray_u8, cv_cmap)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)

        return {
            "dwt_level1": _colorize(_norm_uint8(e_l1), 16 if not _HAS_CV2 else cv2.COLORMAP_VIRIDIS),
            "dwt_level2": _colorize(_norm_uint8(e_l2), 13 if not _HAS_CV2 else cv2.COLORMAP_MAGMA),
            "mad_map": _colorize(
                _norm_uint8(compute_donoho_mad_map(dwt["HH1"])),
                14 if not _HAS_CV2 else cv2.COLORMAP_INFERNO,
            ),
            "ela_map": _colorize(
                _norm_uint8(compute_ela_map(rgb_resized, self.ela_quality)),
                2 if not _HAS_CV2 else cv2.COLORMAP_JET,
            ),
        }


if __name__ == "__main__":
    # Quick self-test of Phase 1 MRA Preprocessor (both 9-channel and 11-channel modes)
    dummy_rgb = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)

    mra_9ch = MRAPreprocessor(target_size=(192, 192), use_forensic_maps=False)
    tensor_9ch = mra_9ch(dummy_rgb)
    assert tensor_9ch.shape == (9, 192, 192), f"Expected (9, 192, 192), got {tensor_9ch.shape}"

    mra_11ch = MRAPreprocessor(target_size=(192, 192), use_forensic_maps=True)
    tensor_11ch = mra_11ch(dummy_rgb)
    assert tensor_11ch.shape == (11, 192, 192), f"Expected (11, 192, 192), got {tensor_11ch.shape}"

    print(f"[Phase 1 OK] 9-channel tensor shape : {tensor_9ch.shape}, dtype={tensor_9ch.dtype}")
    print(f"[Phase 1 OK] 11-channel tensor shape: {tensor_11ch.shape}, dtype={tensor_11ch.dtype}")
