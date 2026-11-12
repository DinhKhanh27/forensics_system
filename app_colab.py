"""
Interactive Gradio UI for Google Colab & Laptop (`app_colab.py`).

Features:
- Runs directly inside a Google Colab cell (generates public `share=True` link) or local PC browser.
- Upload image -> Select mode (`auto` / `fast` / `tile`) & Decision Threshold `T` -> Displays side-by-side:
  1. Final Verdict Badge (ẢNH THẬT / ẢNH GIẢ) + P_fake (%) probability bar + Latency & RAM stats.
  2. Original Image with red contour bounding boxes around tampered regions.
  3. Localization Heatmap (Jet colormap overlay) + Binary Mask.
  4. Multiresolution Analysis (MRA) Feature Maps: 2-Level Haar DWT (Level 1 & Level 2),
     Donoho MAD Local Noise Map, and ELA (Q=90%) Map.
"""

import argparse
from typing import Dict, Tuple
import numpy as np
import gradio as gr

from rpi_inference import EmbeddedForensicPredictor


def build_gradio_app(predictor: EmbeddedForensicPredictor) -> gr.Blocks:
    def run_forensic_analysis(
        input_img: np.ndarray, mode: str, threshold: float
    ) -> Tuple[Dict[str, float], str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if input_img is None:
            # Generate synthetic demo image with a spliced square if user clicks Analyze without uploading
            input_img = np.random.randint(60, 190, size=(384, 512, 3), dtype=np.uint8)
            input_img[110:260, 170:340] = np.clip(
                input_img[110:260, 170:340] + 60, 0, 255
            )

        res = predictor.predict_image(
            rgb_img=input_img,
            mode=mode,
            threshold=float(threshold),
            include_mra_vis=True,
        )

        p_fake = float(res["p_fake"])
        p_real = float(1.0 - p_fake)
        verdict_probs = {
            "ẢNH GIẢ (FORGED)": round(p_fake, 4),
            "ẢNH THẬT (AUTHENTIC)": round(p_real, 4),
        }

        badge_color = "#dc2626" if res["is_forged"] else "#16a34a"
        summary_md = (
            f"<div style='padding:12px;border-radius:8px;background:{badge_color};color:white;font-weight:bold;font-size:16px;'>"
            f"KẾT LUẬN: {res['label']} — Xác suất giả mạo P_fake = {p_fake * 100:.2f}% (Ngưỡng T = {threshold:.2f})"
            f"</div>\n\n"
            f"- **Backend:** `{res['backend']}` | **Độ phân giải gốc:** `{res['resolution']}`\n"
            f"- **Chế độ suy luận thực tế:** `{res['mode_used'].upper()}` (`{res['num_patches']}` patch $192 \\times 192$, `stride=160`)\n"
            f"- **Thời gian xử lý:** `{res['latency_ms']:.1f} ms` (`{res['latency_s']:.3f} s`) | **RAM tiêu thụ:** `{res['ram_mb']:.1f} MB`"
        )

        mra_vis = res["mra_vis"]
        return (
            verdict_probs,
            summary_md,
            res["annotated_img"],
            res["heatmap_img"],
            mra_vis["dwt_level1"],
            mra_vis["dwt_level2"],
            mra_vis["ela_map"],
        )

    with gr.Blocks(
        title="DWT-MobileUNet: Image Forgery Detection & Localization"
    ) as demo:
        gr.Markdown(
            """
            # 🔍 Hệ thống Phát hiện & Định vị Giả mạo Ảnh (DWT-MobileUNet INT8)
            **Đề tài:** *Design of an Embedded System for Image Forgery Detection and Localization using Multiresolution Analysis and Deep Learning*
            - **Tiền xử lý MRA:** 2-Level 2D Haar DWT ($[LH_1, HL_1, HH_1, LH_2, HL_2, HH_2]$) + Donoho MAD & ELA ($Q=90\\%$).
            - **Cơ chế Auto-Tiling:** Tự động chạy `fast` nếu ảnh $\le 384 \times 384$, hoặc chuyển sang `tile` (`stride=160`, Gaussian Blending, `gc.collect()`) nếu ảnh $> 384 \times 384$.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="numpy",
                    label="1. Upload Ảnh Kiểm tra (RGB)",
                )
                mode_radio = gr.Radio(
                    choices=["auto", "fast", "tile"],
                    value="auto",
                    label="2. Chế độ Suy luận (--mode)",
                    info="auto: <=384px chạy fast, >384px tự động chia cửa sổ trượt stride=160",
                )
                threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="3. Ngưỡng quyết định T (Decision Threshold)",
                )
                run_btn = gr.Button(
                    "🚀 Phân tích & Định vị Giả mạo", variant="primary"
                )

            with gr.Column(scale=2):
                summary_box = gr.Markdown(
                    "👉 *Hãy tải ảnh lên và nhấn nút **Phân tích & Định vị Giả mạo**.*"
                )
                prob_label = gr.Label(
                    num_top_classes=2,
                    label="Nhãn Kết luận & Thanh Xác suất P_fake",
                )
                with gr.Row():
                    annotated_out = gr.Image(
                        type="numpy",
                        label="Ảnh Gốc Khoanh vùng Viền Đỏ (Tampered Contours)",
                    )
                    heatmap_out = gr.Image(
                        type="numpy",
                        label="Mặt nạ Nhiệt Định vị (Localization Heatmap)",
                    )

        gr.Markdown("### 🌊 Trực quan hóa Đặc trưng Phân tích Đa phân giải (MRA Wavelet & ELA)")
        with gr.Row():
            dwt_l1_out = gr.Image(
                type="numpy",
                label="Haar DWT Level 1 (Nhiễu vi mô & Biên cắt dán W/2 x H/2)",
            )
            dwt_l2_out = gr.Image(
                type="numpy",
                label="Haar DWT Level 2 (Sai lệch lưới khối JPEG 8x8 W/4 x H/4)",
            )
            ela_out = gr.Image(
                type="numpy",
                label="Bản đồ Sai lệch Nén ELA (Q=90%) / Donoho MAD",
            )

        run_btn.click(
            fn=run_forensic_analysis,
            inputs=[input_image, mode_radio, threshold_slider],
            outputs=[
                prob_label,
                summary_box,
                annotated_out,
                heatmap_out,
                dwt_l1_out,
                dwt_l2_out,
                ela_out,
            ],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Gradio Forensic UI on Google Colab or Laptop"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/model_int8.onnx",
        help="Path to ONNX INT8 or PyTorch checkpoint",
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Local port for Gradio server"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio share link (recommended on Google Colab)",
    )
    parser.add_argument(
        "--use-forensic-maps",
        action="store_true",
        help="Enable 11-channel MRA mode (with Donoho MAD + ELA)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = EmbeddedForensicPredictor(
        model_path=args.model_path,
        use_forensic_maps=args.use_forensic_maps,
    )
    app = build_gradio_app(predictor)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
