"""
Ultra-Lightweight Embedded Flask Web UI for Raspberry Pi 3 Model B (`rpi_web_ui.py`).

Design Goals:
- Pure Flask + Vanilla HTML/CSS/JS frontend (zero heavy frontend frameworks; < 15 MB web server overhead).
- Accessible across local Wi-Fi network at `http://<ip-cua-pi>:5000` from Laptop/Smartphone or Pi display.
- Supports:
  1. Image Upload from client browser (`POST /api/analyze`).
  2. Direct capture from Raspberry Pi Camera (`POST /api/capture`) via `rpicam-still`,
     `libcamera-still`, or V4L2 `cv2.VideoCapture(0)`, with automatic synthetic test frame
     fallback when testing on Colab/Laptop without a physical CSI camera attached.
  3. Returns Base64-encoded annotated contour image, localization heatmap, DWT Level 1/2
     feature visualization, forgery probability P_fake (%), execution latency (ms/s),
     and live Raspberry Pi 3 RAM usage (MB).
"""

import argparse
import base64
import io
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request

from rpi_inference import EmbeddedForensicPredictor, get_process_ram_mb

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False

app = Flask(__name__, template_folder="templates")
PREDICTOR: Optional[EmbeddedForensicPredictor] = None


def _np_rgb_to_base64_jpeg(rgb_arr: np.ndarray, quality: int = 88) -> str:
    """Encodes an RGB uint8 numpy image to a compact Data-URI JPEG string in memory."""
    pil_img = Image.fromarray(np.clip(rgb_arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64_str}"


def capture_from_pi_camera(width: int = 640, height: int = 480) -> Tuple[np.ndarray, str]:
    """
    Attempts to capture a live RGB frame from the Raspberry Pi Camera module using:
      1. `rpicam-still` / `libcamera-still` (Raspberry Pi OS Bookworm / Bullseye 64-bit)
      2. OpenCV V4L2 `cv2.VideoCapture(0)` (USB webcam / legacy Pi Camera)
      3. Synthetic spliced test frame fallback (for headless Colab / PC verification)
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        for cmd_bin in ("rpicam-still", "libcamera-still"):
            try:
                cmd = [
                    cmd_bin,
                    "-n",
                    "--immediate",
                    "--width",
                    str(width),
                    "--height",
                    str(height),
                    "-o",
                    str(tmp_path),
                ]
                proc = subprocess.run(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4
                )
                if proc.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
                    img = np.array(Image.open(tmp_path).convert("RGB"), dtype=np.uint8)
                    return img, f"Pi Camera ({cmd_bin})"
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                continue
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    if _HAS_CV2:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame_bgr = cap.read()
            cap.release()
            if ret and frame_bgr is not None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                return frame_rgb, "Camera V4L2 / USB (cv2)"

    # Fallback test frame so the Capture button can be tested on Laptop/Colab without camera hardware
    synth = np.random.randint(50, 195, size=(height, width, 3), dtype=np.uint8)
    synth[120:320, 180:440] = np.clip(synth[120:320, 180:440] + 55, 0, 255)
    return synth, "Simulated Camera Frame (No physical camera detected)"


def _format_api_response(rgb_img: np.ndarray, mode: str, threshold: float, source_tag: str):
    assert PREDICTOR is not None, "Predictor not initialized"
    res = PREDICTOR.predict_image(
        rgb_img=rgb_img,
        mode=mode,
        threshold=threshold,
        include_mra_vis=True,
    )
    mra_vis = res["mra_vis"]
    return jsonify(
        {
            "status": "ok",
            "source": source_tag,
            "label": res["label"],
            "is_forged": res["is_forged"],
            "p_fake_pct": round(res["p_fake"] * 100.0, 2),
            "threshold": round(threshold, 2),
            "mode_used": res["mode_used"].upper(),
            "num_patches": res["num_patches"],
            "resolution": res["resolution"],
            "latency_ms": round(res["latency_ms"], 1),
            "latency_s": round(res["latency_s"], 3),
            "ram_mb": round(res["ram_mb"], 2),
            "backend": res["backend"],
            "images": {
                "annotated": _np_rgb_to_base64_jpeg(res["annotated_img"]),
                "heatmap": _np_rgb_to_base64_jpeg(res["heatmap_img"]),
                "dwt_l1": _np_rgb_to_base64_jpeg(mra_vis["dwt_level1"]),
                "dwt_l2": _np_rgb_to_base64_jpeg(mra_vis["dwt_level2"]),
            },
        }
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", ram_mb=round(get_process_ram_mb(), 2))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "Vui lòng chọn file ảnh để tải lên."}), 400

    file = request.files["image"]
    mode = request.form.get("mode", "auto")
    threshold = float(request.form.get("threshold", 0.5))

    try:
        pil_img = Image.open(file.stream).convert("RGB")
        rgb_img = np.array(pil_img, dtype=np.uint8)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Lỗi đọc ảnh: {exc}"}), 400

    return _format_api_response(rgb_img, mode, threshold, source_tag=file.filename or "Uploaded Image")


@app.route("/api/capture", methods=["POST"])
def api_capture():
    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode", "auto")
    threshold = float(payload.get("threshold", 0.5))

    rgb_img, cam_source = capture_from_pi_camera()
    return _format_api_response(rgb_img, mode, threshold, source_tag=cam_source)


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight Embedded Flask Web Server for Raspberry Pi 3 Model B"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/model_int8.onnx",
        help="Path to quantized INT8 ONNX model",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host address")
    parser.add_argument("--port", type=int, default=5000, help="Bind port (default: 5000)")
    parser.add_argument(
        "--use-forensic-maps",
        action="store_true",
        help="Enable 11-channel MRA mode (MAD + ELA)",
    )
    return parser.parse_args()


def main() -> None:
    global PREDICTOR
    args = parse_args()
    PREDICTOR = EmbeddedForensicPredictor(
        model_path=args.model_path,
        use_forensic_maps=args.use_forensic_maps,
    )
    local_ip = get_local_ip()
    print("=" * 72)
    print("[RPi 3 WEB UI] DWT-MobileUNet Embedded Server Ready!")
    print(f"  Local URL   : http://127.0.0.1:{args.port}")
    print(f"  Wi-Fi LAN   : http://{local_ip}:{args.port}")
    print(f"  Backend     : {PREDICTOR.backend}")
    print(f"  Initial RAM : {get_process_ram_mb():.2f} MB")
    print("=" * 72)
    app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
