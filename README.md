# DWT-MobileUNet: Embedded Image Forgery Detection & Localization

**Topic:** *Design of an Embedded System for Image Forgery Detection and Localization using Multiresolution Analysis and Deep Learning*

## 1. Overview
- **Multiresolution Analysis (MRA):** 2-Level 2D Haar Discrete Wavelet Transform ($[LH_1, HL_1, HH_1, LH_2, HL_2, HH_2]$) fused with RGB into a $9 \times 192 \times 192$ tensor (or $11 \times 192 \times 192$ with `--use-forensic-maps` adding Donoho MAD & ELA $Q=90\%$).
- **Deep Learning Architecture (`DWT-MobileUNet`):**
  - **Encoder:** Custom `MobileNetV3-Small` (`width_multiplier=0.5`, `< 2 MB` INT8 footprint).
  - **Head 1 (Detection):** Bottleneck ($1/32$) $\rightarrow$ Global Average Pooling $\rightarrow$ Dense(64) $\rightarrow$ Dropout(0.2) $\rightarrow$ Dense(1) $\rightarrow$ Sigmoid ($P_{fake} \in [0, 1]$).
  - **Head 2 (Localization):** 4 Upsampling Decoder blocks + Skip Connections + Depthwise Separable Conv $\rightarrow$ Conv $1\times 1$ + Sigmoid ($192 \times 192 \times 1$ mask).
  - **Multi-Task Loss:** $\mathcal{L}_{total} = 0.3 \cdot \mathcal{L}_{BCE\_cls} + 0.7 \cdot (\mathcal{L}_{Dice} + \mathcal{L}_{Focal})$.
- **RAM-Aware Auto-Tiling on Raspberry Pi 3 (1 GB RAM):**
  - `fast`: Direct $192 \times 192$ inference (~0.7s/image).
  - `tile`: Sliding window $192 \times 192$ with `stride = 160` (divisible by 8 to preserve $8 \times 8$ JPEG block grid, `overlap = 32 px`), **2D Gaussian Blending**, $P_{fake} = \max(P_{patch})$, and per-patch `gc.collect()` (< 60 MB model peak RAM).
  - `auto`: Automatically chooses `fast` if $\max(H, W) \le 384$ and switches to `tile` if $> 384$.

---

## 2. Project Structure
```text
├── configs/
│   └── config.yaml              # Hyperparameters for MRA, Model, Loss, and Auto-Tiling
├── src/
│   ├── __init__.py
│   ├── dwt.py                   # 2-Level 2D Haar DWT + Donoho MAD & ELA (Q=90%)
│   ├── model.py                 # DWT-MobileUNet Dual-Head & MultiTaskForensicLoss
│   ├── tiling.py                # Sliding window (stride=160), Gaussian Blending & Contour drawing
│   └── dataset.py               # CASIA v2.0 / Columbia / NIST16 DataLoader
├── templates/
│   └── index.html               # Lightweight HTML5/CSS3/JS UI for Raspberry Pi 3
├── train.py                     # Google Colab CUDA CLI Trainer (AMP FP16 + --dry-run)
├── colab_train.ipynb            # End-to-End Google Colab Training & Quantization Notebook
├── export_quantize.py           # PyTorch (.pth) -> ONNX FP32 -> ONNX INT8 (< 2 MB)
├── rpi_inference.py             # Cross-platform Inference Engine (auto / fast / tile)
├── app_colab.py                 # Gradio Web UI for Google Colab & Laptop
├── rpi_web_ui.py                # Ultra-lightweight Flask Web Server (< 15 MB RAM) for RPi 3
├── requirements_colab.txt       # Dependencies for Google Colab / PC
└── requirements_rpi.txt         # Dependencies for Raspberry Pi 3 Model B (64-bit)
```

---

## 3. Quick Start

### A. Training & INT8 Export on Google Colab
Open `colab_train.ipynb` in Google Colab, or run via CLI:
```bash
pip install -r requirements_colab.txt

# 1. 3-second synthetic verification pass
python train.py --dry-run --amp

# 2. Full training (saves best_model.pth to Google Drive)
python train.py --data_dir /content/dataset/CASIA2 --save_dir /content/drive/MyDrive/checkpoints --epochs 50 --batch_size 16 --amp

# 3. Export & INT8 Quantization (< 2 MB)
python export_quantize.py --checkpoint /content/drive/MyDrive/checkpoints/best_model.pth --output_dir ./checkpoints

# 4. Launch Gradio Demo UI
python app_colab.py --model_path ./checkpoints/model_int8.onnx --share
```

### B. Deployment on Raspberry Pi 3 Model B
```bash
git clone <your-repo-url>
cd DATN_VER2
pip install -r requirements_rpi.txt

# 1. CLI Inference (auto / fast / tile)
python rpi_inference.py --image sample.jpg --model checkpoints/model_int8.onnx --mode auto

# 2. Launch Embedded Web UI at http://<pi-ip>:5000
python rpi_web_ui.py --model_path checkpoints/model_int8.onnx --port 5000
```
