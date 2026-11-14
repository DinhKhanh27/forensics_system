"""
Model Export & INT8 Quantization Pipeline (PyTorch -> ONNX FP32 -> ONNX INT8).

Fixes & Improvements:
1. Universal Checkpoint Support: Automatically detects and loads either `DWTMobileUNet`
   (MobileNetV3-Small + 2-Level Haar DWT) or `LegacyDWTUNet` (commit 8d0513a checkpoint)
   with `return_logits=False` so that `torch.sigmoid` is ALWAYS baked into the exported
   ONNX graph (`prob_fake` in [0, 1] and `mask_prob` in [0, 1]).
2. Strict < 2.0 MB INT8 size verification and latency benchmarking.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

from src.model import (
    DWTMobileUNet,
    count_parameters,
    load_forensic_checkpoint,
)


class SyntheticMRADataReader(CalibrationDataReader):
    """
    Calibration dataset reader generating representative inputs in [0, 1] (for 3-ch RGB)
    or Z-score normalized (for 9/11-ch MRA) for static INT8 quantization.
    """

    def __init__(self, input_name: str, in_channels: int = 3, num_samples: int = 24) -> None:
        self.input_name = input_name
        self.in_channels = in_channels
        self.num_samples = num_samples
        self.current_idx = 0
        rng = np.random.default_rng(seed=42)
        if in_channels == 3:
            self.samples = [
                rng.uniform(0.0, 1.0, size=(1, 3, 192, 192)).astype(np.float32)
                for _ in range(num_samples)
            ]
        else:
            self.samples = [
                rng.standard_normal(size=(1, in_channels, 192, 192)).astype(np.float32)
                for _ in range(num_samples)
            ]

    def get_next(self) -> dict:
        if self.current_idx >= self.num_samples:
            return None
        sample = {self.input_name: self.samples[self.current_idx]}
        self.current_idx += 1
        return sample


def load_or_init_pytorch_model(
    checkpoint_path: Path, use_forensic_maps: bool = False
) -> Tuple[nn.Module, int]:
    """
    Loads trained weights from `checkpoint_path` with `return_logits=False` (ensuring
    Sigmoid probabilities in [0, 1] are exported into the ONNX graph).
    """
    default_in_ch = 11 if use_forensic_maps else 3

    if checkpoint_path and checkpoint_path.exists():
        model, in_channels, arch_name = load_forensic_checkpoint(
            str(checkpoint_path),
            device="cpu",
            return_logits=False,
            default_in_channels=default_in_ch,
        )
        print(
            f"[EXPORT] Loaded trained '{arch_name}' weights from {checkpoint_path} (in_channels={in_channels}, Sigmoid=ON)"
        )
    else:
        print(
            f"[EXPORT] Checkpoint '{checkpoint_path}' not found. Initializing fresh DWTMobileUNet for export verification."
        )
        in_channels = default_in_ch
        model = DWTMobileUNet(
            in_channels=in_channels, width_multiplier=0.5, return_logits=False
        )

    model.eval()
    return model, in_channels


def export_to_onnx(
    model: nn.Module, in_channels: int, onnx_fp32_path: Path, opset: int = 13
) -> None:
    """
    Exports PyTorch model to ONNX FP32 with calibrated Sigmoid outputs:
      - prob_fake: (B, 1) in [0, 1]
      - mask_prob: (B, 1, 192, 192) in [0, 1]
    """
    dummy_input = torch.rand(1, in_channels, 192, 192, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_fp32_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["mra_input"],
        output_names=["prob_fake", "mask_prob"],
        dynamic_axes={
            "mra_input": {0: "batch_size"},
            "prob_fake": {0: "batch_size"},
            "mask_prob": {0: "batch_size"},
        },
    )
    onnx_model = onnx.load(str(onnx_fp32_path))
    onnx.checker.check_model(onnx_model)


def quantize_onnx_int8(
    onnx_fp32_path: Path,
    onnx_int8_path: Path,
    in_channels: int = 3,
    quant_mode: str = "dynamic",
) -> None:
    if quant_mode == "static":
        reader = SyntheticMRADataReader(
            input_name="mra_input", in_channels=in_channels, num_samples=24
        )
        quantize_static(
            model_input=str(onnx_fp32_path),
            model_output=str(onnx_int8_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            per_channel=False,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
        )
    else:
        quantize_dynamic(
            model_input=str(onnx_fp32_path),
            model_output=str(onnx_int8_path),
            weight_type=QuantType.QUInt8,
        )


def benchmark_and_verify(
    torch_model: nn.Module,
    onnx_fp32_path: Path,
    onnx_int8_path: Path,
    in_channels: int = 3,
) -> None:
    dummy_np = np.random.uniform(0.0, 1.0, size=(1, in_channels, 192, 192)).astype(
        np.float32
    )

    with torch.no_grad():
        pt_cls, pt_mask = torch_model(torch.from_numpy(dummy_np))
        pt_cls_np = pt_cls.numpy()
        pt_mask_np = pt_mask.numpy()

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess_fp32 = ort.InferenceSession(
        str(onnx_fp32_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )
    sess_int8 = ort.InferenceSession(
        str(onnx_int8_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )

    ort_fp32_cls, ort_fp32_mask = sess_fp32.run(None, {"mra_input": dummy_np})
    ort_int8_cls, ort_int8_mask = sess_int8.run(None, {"mra_input": dummy_np})

    t0 = time.perf_counter()
    for _ in range(10):
        sess_int8.run(None, {"mra_input": dummy_np})
    avg_int8_ms = (time.perf_counter() - t0) * 1000.0 / 10.0

    fp32_mb = os.path.getsize(onnx_fp32_path) / (1024.0 * 1024.0)
    int8_mb = os.path.getsize(onnx_int8_path) / (1024.0 * 1024.0)

    cls_diff = float(np.max(np.abs(pt_cls_np - ort_int8_cls)))
    mask_diff = float(np.max(np.abs(pt_mask_np - ort_int8_mask)))

    print("=" * 72)
    print("[QUANTIZATION & VERIFICATION REPORT]")
    print(f"  ONNX FP32 Model Size : {fp32_mb:.3f} MB ({onnx_fp32_path})")
    print(
        f"  ONNX INT8 Model Size : {int8_mb:.3f} MB ({onnx_int8_path}) -> Target < 2.0 MB: {'PASS' if int8_mb < 2.0 else 'WARN'}"
    )
    print(
        f"  Output 1 (prob_fake) : shape={ort_int8_cls.shape}, range=[{ort_int8_cls.min():.4f}, {ort_int8_cls.max():.4f}], max_diff={cls_diff:.5f}"
    )
    print(
        f"  Output 2 (mask_prob) : shape={ort_int8_mask.shape}, range=[{ort_int8_mask.min():.4f}, {ort_int8_mask.max():.4f}], max_diff={mask_diff:.5f}"
    )
    print(f"  Avg INT8 CPU Latency : {avg_int8_ms:.2f} ms / patch (192x192)")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PyTorch DWT-MobileUNet to ONNX FP32 & INT8 for Raspberry Pi 3"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to trained PyTorch checkpoint (.pth)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model_fp32.onnx and model_int8.onnx",
    )
    parser.add_argument(
        "--quant_mode",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
        help="ONNX Runtime quantization mode (default: dynamic)",
    )
    parser.add_argument(
        "--use-forensic-maps",
        action="store_true",
        help="Use 11-channel input (RGB + 6 DWT + MAD + ELA) if checkpoint is not found",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    onnx_fp32_path = out_dir / "model_fp32.onnx"
    onnx_int8_path = out_dir / "model_int8.onnx"

    model, in_channels = load_or_init_pytorch_model(
        ckpt_path, use_forensic_maps=args.use_forensic_maps
    )
    params, _ = count_parameters(model)
    print(f"[EXPORT] Model Parameters: {params:,} | Input Channels: {in_channels}")

    print(f"[EXPORT] Exporting FP32 ONNX graph -> {onnx_fp32_path} ...")
    export_to_onnx(model, in_channels=in_channels, onnx_fp32_path=onnx_fp32_path)

    print(
        f"[EXPORT] Applying {args.quant_mode.upper()} INT8 quantization -> {onnx_int8_path} ..."
    )
    quantize_onnx_int8(
        onnx_fp32_path=onnx_fp32_path,
        onnx_int8_path=onnx_int8_path,
        in_channels=in_channels,
        quant_mode=args.quant_mode,
    )

    benchmark_and_verify(
        torch_model=model,
        onnx_fp32_path=onnx_fp32_path,
        onnx_int8_path=onnx_int8_path,
        in_channels=in_channels,
    )


if __name__ == "__main__":
    main()
