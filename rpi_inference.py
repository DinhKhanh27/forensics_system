"""
Cross-Platform Embedded Inference Engine (Google Colab, PC, and Raspberry Pi 3 Model B).

Fixes & Improvements after Experimental Review:
1. Automatic Sigmoid Calibration Guard (`_ensure_probability`):
   - If an exported ONNX or PyTorch checkpoint outputs raw logits (e.g., values < 0 or > 1
     from commit 8d0513a), automatically applies `sigmoid(x) = 1 / (1 + exp(-x))` so
     `p_fake` and `mask_2d` are strictly valid probabilities in [0, 1].
2. Automatic Input Channel Matching:
   - Inspects the ONNX or PyTorch model's expected input channel dimension (`3`, `9`, or `11`)
     and feeds either normalized 3-channel RGB `[0, 1]` (when 2-Level Haar DWT is inside the
     graph) or the 9/11-channel `MRAPreprocessor` tensor.
3. Universal `.pth` checkpoint loading (`DWTMobileUNet` and `LegacyDWTUNet`).
"""

import argparse
import gc
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image

from src.dwt import MRAPreprocessor
from src.tiling import TilingInferenceEngine

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None
    _HAS_PSUTIL = False

try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    ort = None
    _HAS_ORT = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False


def get_process_ram_mb() -> float:
    if _HAS_PSUTIL:
        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss) / (1024.0 * 1024.0)

    status_file = Path("/proc/self/status")
    if status_file.exists():
        for line in status_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                kb = float(line.split()[1])
                return kb / 1024.0
    return 0.0


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x.astype(np.float32), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def _ensure_probability(arr: np.ndarray) -> np.ndarray:
    """
    Converts raw model output to probabilities in [0, 1] if the model returned raw logits
    (i.e. values < -1e-4 or > 1.0 + 1e-4).
    """
    arr_f32 = np.asarray(arr, dtype=np.float32)
    if float(np.min(arr_f32)) < -1e-4 or float(np.max(arr_f32)) > 1.0001:
        return _sigmoid_np(arr_f32)
    return np.clip(arr_f32, 0.0, 1.0)


class EmbeddedForensicPredictor:
    """
    High-efficiency inference wrapper for DWT-MobileUNet (INT8 ONNX or PyTorch checkpoint).
    """

    def __init__(
        self,
        model_path: Optional[str] = "checkpoints/model_int8.onnx",
        use_forensic_maps: bool = False,
        patch_size: int = 192,
        stride: int = 160,
        auto_tile_threshold: int = 384,
        threshold: float = 0.5,
        num_threads: int = 4,
    ) -> None:
        self.patch_size = patch_size
        self.stride = stride
        self.auto_tile_threshold = auto_tile_threshold
        self.threshold = threshold
        self.use_forensic_maps = use_forensic_maps
        self.expected_in_channels = 11 if use_forensic_maps else 3
        self.backend = "heuristic_mra"
        self.session = None
        self.torch_model = None
        self.input_name = "mra_input"

        candidates = []
        if model_path:
            candidates.append(Path(model_path))
        candidates.extend(
            [
                Path("checkpoints/model_int8.onnx"),
                Path("checkpoints/model_fp32.onnx"),
                Path("checkpoints/best_model.pth"),
            ]
        )

        resolved_path: Optional[Path] = None
        for cand in candidates:
            if cand.exists() and cand.is_file():
                resolved_path = cand
                break

        if resolved_path is not None and resolved_path.suffix.lower() == ".onnx" and _HAS_ORT:
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = num_threads
            sess_opts.inter_op_num_threads = 1
            sess_opts.enable_mem_pattern = False
            sess_opts.enable_cpu_mem_arena = False
            sess_opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self.session = ort.InferenceSession(
                str(resolved_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            inp_meta = self.session.get_inputs()[0]
            self.input_name = inp_meta.name
            if len(inp_meta.shape) >= 2 and isinstance(inp_meta.shape[1], int):
                self.expected_in_channels = int(inp_meta.shape[1])
                self.use_forensic_maps = self.expected_in_channels == 11
            self.backend = f"ONNX ({resolved_path.name}, in_ch={self.expected_in_channels})"

        elif resolved_path is not None and resolved_path.suffix.lower() == ".pth" and _HAS_TORCH:
            from src.model import load_forensic_checkpoint

            self.torch_model, self.expected_in_channels, arch_name = (
                load_forensic_checkpoint(
                    str(resolved_path),
                    device="cpu",
                    return_logits=False,
                    default_in_channels=self.expected_in_channels,
                )
            )
            self.torch_model.eval()
            self.use_forensic_maps = self.expected_in_channels == 11
            self.backend = f"PyTorch ({resolved_path.name} - {arch_name})"

        self.mra = MRAPreprocessor(
            target_size=(patch_size, patch_size),
            use_forensic_maps=self.use_forensic_maps,
        )
        self.tiling_engine = TilingInferenceEngine(
            patch_size=patch_size,
            stride=stride,
            auto_tile_threshold=auto_tile_threshold,
            threshold=threshold,
        )

    def _prepare_input_tensor(self, patch_rgb: np.ndarray) -> np.ndarray:
        """
        Prepares `(1, C, 192, 192)` float32 tensor matching `self.expected_in_channels`.
        - If model expects 3 channels (in-graph 2-Level Haar DWT): returns RGB in [0, 1].
        - If model expects 9 or 11 channels: returns CPU `MRAPreprocessor` tensor.
        """
        if self.expected_in_channels == 3:
            img_float = patch_rgb.astype(np.float32) / 255.0
            rgb_chw = np.transpose(img_float, (2, 0, 1))
            return np.expand_dims(rgb_chw, axis=0)
        mra_chw = self.mra(patch_rgb)
        return np.expand_dims(mra_chw, axis=0)

    def _predict_single_patch(self, patch_rgb: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Runs model inference on a single 192x192 RGB patch and guarantees calibrated
        probabilities in [0, 1] for both `p_fake` and `mask_2d`.
        """
        if self.session is not None or self.torch_model is not None:
            batched = self._prepare_input_tensor(patch_rgb)

            if self.session is not None:
                raw_cls, raw_mask = self.session.run(None, {self.input_name: batched})
            else:
                with torch.no_grad():
                    t_in = torch.from_numpy(batched)
                    pt_cls, pt_mask = self.torch_model(t_in)
                    raw_cls = pt_cls.cpu().numpy()
                    raw_mask = pt_mask.cpu().numpy()

            # Automatically convert raw logits -> Sigmoid probabilities if needed
            prob_cls_arr = _ensure_probability(raw_cls)
            mask_2d = np.squeeze(_ensure_probability(raw_mask)).astype(np.float32)
            p_cls = float(np.squeeze(prob_cls_arr))

            # Suppress tiny isolated background probabilities when classification head says Authentic
            # and couple patch forgery score with actual spatial mask presence
            forged_area_ratio = float(np.mean(mask_2d >= 0.5))
            if forged_area_ratio < 0.002 and p_cls < 0.65:
                # Less than 0.2% pixels active (< 75 pixels in 192x192) -> suppress false-positive noise
                mask_2d = mask_2d * 0.5
                p_fake = min(p_cls, float(np.percentile(mask_2d, 99)))
            else:
                p_fake = float(0.65 * p_cls + 0.35 * float(np.percentile(mask_2d, 98)))

            del batched
            return p_fake, mask_2d

        # Heuristic MRA fallback when no model file exists yet
        mra_tensor = self.mra(patch_rgb)
        hf_energy = np.mean(np.abs(mra_tensor[3:9]), axis=0)
        norm_e = (hf_energy - hf_energy.min()) / (
            hf_energy.max() - hf_energy.min() + 1e-6
        )
        mask_2d = np.clip(norm_e, 0.0, 1.0).astype(np.float32)
        p_fake = float(np.percentile(mask_2d, 95))
        del mra_tensor
        return p_fake, mask_2d

    def predict_image(
        self,
        rgb_img: np.ndarray,
        mode: str = "auto",
        threshold: Optional[float] = None,
        include_mra_vis: bool = True,
    ) -> Dict[str, object]:
        gc.collect()
        t_start = time.perf_counter()

        result = self.tiling_engine.run(
            rgb_img=rgb_img,
            patch_predict_fn=self._predict_single_patch,
            mode=mode,
            threshold=threshold,
        )

        if include_mra_vis:
            result["mra_vis"] = self.mra.extract_visualizations(rgb_img)

        latency_s = time.perf_counter() - t_start
        ram_mb = get_process_ram_mb()

        result["latency_s"] = float(latency_s)
        result["latency_ms"] = float(latency_s * 1000.0)
        result["ram_mb"] = float(ram_mb)
        result["backend"] = self.backend
        result["resolution"] = f"{rgb_img.shape[1]}x{rgb_img.shape[0]}"
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embedded Image Forgery Detection & Localization Inference (RPi 3 / Colab)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (if omitted, runs on a synthetic test image)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/model_int8.onnx",
        help="Path to ONNX INT8 (.onnx) or PyTorch (.pth) model",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "fast", "tile"],
        help="Inference mode: 'fast' (192x192 resize), 'tile' (stride=160 sliding window), or 'auto' (<=384 fast, >384 tile)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold T in [0, 1] (default: 0.5)",
    )
    parser.add_argument(
        "--use-forensic-maps",
        action="store_true",
        help="Enable explicit Donoho MAD + ELA channels (11 channels)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save annotated result and localization heatmap",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    predictor = EmbeddedForensicPredictor(
        model_path=args.model,
        use_forensic_maps=args.use_forensic_maps,
        threshold=args.threshold,
    )

    if args.image and os.path.isfile(args.image):
        rgb_img = np.array(Image.open(args.image).convert("RGB"), dtype=np.uint8)
        img_name = Path(args.image).stem
    else:
        print("[INFERENCE] No --image provided; generating synthetic 640x480 test image...")
        rgb_img = np.random.randint(40, 200, size=(480, 640, 3), dtype=np.uint8)
        rgb_img[140:300, 200:420] = np.clip(rgb_img[140:300, 200:420] + 55, 0, 255)
        img_name = "synthetic_test"

    res = predictor.predict_image(
        rgb_img, mode=args.mode, threshold=args.threshold, include_mra_vis=False
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = out_dir / f"{img_name}_annotated.jpg"
    heatmap_path = out_dir / f"{img_name}_heatmap.jpg"
    mask_path = out_dir / f"{img_name}_mask.png"

    Image.fromarray(res["annotated_img"]).save(annotated_path, quality=95)
    Image.fromarray(res["heatmap_img"]).save(heatmap_path, quality=95)
    Image.fromarray(res["binary_mask"]).save(mask_path)

    print("=" * 72)
    print("[EMBEDDED INFERENCE REPORT]")
    print(f"  Backend Engine     : {res['backend']}")
    print(f"  Input Resolution   : {res['resolution']}")
    print(
        f"  Execution Mode     : {res['mode_used'].upper()} (Patches evaluated: {res['num_patches']})"
    )
    print(f"  Verdict            : {res['label']}")
    print(
        f"  Probability P_fake : {res['p_fake'] * 100:.2f}% (Threshold T = {args.threshold:.2f})"
    )
    print(
        f"  Processing Latency : {res['latency_ms']:.1f} ms ({res['latency_s']:.3f} s)"
    )
    print(f"  Process RAM Usage  : {res['ram_mb']:.2f} MB")
    print(f"  Saved Outputs      : {annotated_path} | {heatmap_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
