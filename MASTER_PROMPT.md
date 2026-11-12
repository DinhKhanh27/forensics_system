# MASTER PROMPT: EMBEDDED IMAGE FORGERY DETECTION & LOCALIZATION (COLAB + RASPBERRY PI 3 + UI)

Hãy đóng vai trò là **Kỹ sư AI Nhúng (Embedded AI Engineer)**. Nhiệm vụ của bạn là xây dựng hoàn chỉnh mã nguồn cho đề tài:
> **"Design of an Embedded System for Image Forgery Detection and Localization using Multiresolution Analysis and Deep Learning"**

---

## 1. NỀN TẢNG PHẦN CỨNG & MÔI TRƯỜNG MỤC TIÊU

- **Môi trường Huấn luyện (Training — 100% Google Colab):**
  - Huấn luyện hoàn toàn trên Google Colab (GPU T4 / L4 / A100) thông qua file Notebook điều phối chuẩn (`colab_train.ipynb`).
  - Hỗ trợ tự động Mount Google Drive để lưu trữ lâu dài tập dữ liệu và trọng số checkpoint (tránh mất dữ liệu khi Colab hết phiên), đồng bộ mã nguồn hai chiều qua GitHub.
- **Thiết bị triển khai (Target Edge Device):** Raspberry Pi 3 Model B (ARM Cortex-A53 Quad-core, 1 GB RAM, Raspberry Pi OS 64-bit).
- **Yêu cầu tương thích chéo (Cross-platform):** Toàn bộ pipeline suy luận và giao diện phải chạy kiểm thử mượt mà trên Google Colab / Laptop trước khi `git pull` xuống bring-up trên Raspberry Pi 3.

---

## 2. KIẾN TRÚC THUẬT TOÁN & MÔ HÌNH CỐ ĐỊNH

### 2.1. Tiền xử lý Phân tích đa phân giải (Multiresolution Analysis - MRA)
- **Thuật toán cốt lõi — 2-Level 2D Discrete Wavelet Transform (Haar):**
  - **Level 1 ($\frac{W}{2} \times \frac{H}{2}$):** Tách thành $[LL_1, LH_1, HL_1, HH_1]$ để bắt nhiễu vi mô và đường biên cắt dán (Splicing).
  - **Level 2 ($\frac{W}{4} \times \frac{H}{4}$):** Tiếp tục phân rã $LL_1$ thành $[LH_2, HL_2, HH_2]$ để bắt sai lệch lưới khối nén JPEG $8 \times 8$ và độ dốc chiếu sáng bất thường.
  - **Early Fusion:** Nội suy Bilinear 6 dải tần số cao $[LH_1, HL_1, HH_1, LH_2, HL_2, HH_2]$ về kích thước chuẩn $192 \times 192$, chuẩn hóa Z-score và ghép với ảnh RGB thành Tensor 9 kênh kích thước $9 \times 192 \times 192$.
- **Tùy chọn đặc trưng giải tích tường minh (cờ `--use-forensic-maps`):**
  - Tích hợp tùy chọn tính bản đồ phương sai nhiễu cục bộ **Donoho MAD** từ băng tần $HH_1$ ($\hat{\sigma} = \frac{\text{median}(|HH_1|)}{0.6745}$) và bản đồ sai lệch nén **ELA (Error Level Analysis, chất lượng nén lại $Q=90\%$)** để tăng tính giải thích học thuật khi cần.

### 2.2. Kiến trúc Deep Learning (DWT-MobileUNet)
- **Encoder:** Sử dụng `MobileNetV3-Small` tùy biến `in_channels=9` (hoặc `11` nếu bật thêm MAD+ELA), hệ số co kênh `width_multiplier=0.5` để giữ kích thước mô hình INT8 dưới 2 MB.
- **Đầu ra đa nhiệm (Dual-Head Output):**
  - **Head 1 — Detection (Nhận diện Thật/Giả):** Lấy đặc trưng tại nút thắt sâu nhất ($1/32$) $\rightarrow$ Global Average Pooling $\rightarrow$ Dense(64) $\rightarrow$ Dropout(0.2) $\rightarrow$ Dense(1) $\rightarrow$ Sigmoid, xuất xác suất $P_{fake} \in [0, 1]$.
  - **Head 2 — Localization (Định vị vùng giả mạo):** Decoder gồm 4 khối Upsampling $2\times$ + Depthwise Separable Conv + Skip Connections từ Encoder $\rightarrow$ Conv $1 \times 1$ + Sigmoid, xuất mặt nạ nhị phân $192 \times 192 \times 1$.
- **Hàm mất mát (Multi-Task Loss):**
  $$\mathcal{L}_{total} = 0.3 \cdot \mathcal{L}_{BCE\_cls} + 0.7 \cdot (\mathcal{L}_{Dice} + \mathcal{L}_{Focal})$$

---

## 3. CƠ CHẾ QUẢN LÝ RAM & AUTO-TILING TRÊN RASPBERRY PI 3

Để giải quyết giới hạn 1 GB RAM của Raspberry Pi 3 mà không làm mất dấu vết vi mô trên ảnh chụp độ phân giải cao, script suy luận (`rpi_inference.py`) bắt buộc hỗ trợ 3 chế độ qua tham số `--mode`:
1. **`fast`:** Resize thẳng ảnh đầu vào về $192 \times 192$ và chạy 1 lượt (~0.7 giây/ảnh trên RPi 3).
2. **`tile` (Sliding Window):**
   - Cắt ảnh gốc thành các patch kích thước $192 \times 192$ với bước trượt `stride = 160` (bắt buộc chia hết cho 8 để giữ nguyên pha lưới nén JPEG $8 \times 8$, vùng chồng lấn `overlap = 32 px`).
   - Xử lý tuần tự từng patch qua mô hình INT8, gọi `gc.collect()` giải phóng tensor ngay sau mỗi patch để khóa cứng RAM đỉnh của mô hình dưới 60 MB.
   - Ghép các patch mask lại bằng thuật toán **Gaussian Blending** tại dải chồng lấn 32px; tính điểm toàn ảnh $P_{fake} = \max(P_{patch})$; áp ngưỡng $T \ge 0.5$ và dùng `cv2.findContours` vẽ khung viền đỏ quanh vùng bị chỉnh sửa.
3. **`auto` (Mặc định):** Nếu kích thước ảnh $\le 384 \times 384$ (như CASIA) thì tự động chạy `fast`; nếu $> 384 \times 384$ (ảnh camera/điện thoại) thì tự động chuyển sang `tile`.

---

## 4. YÊU CẦU GIAO DIỆN NGƯỜI DÙNG (USER INTERFACE - BẮT BUỘC)

Hệ thống phải có giao diện trực quan để demo, kiểm thử và nghiệm thu ở cả 2 môi trường:

1. **Giao diện trên Google Colab & Laptop (`app_colab.py` — Gradio):**
   - Chạy được trực tiếp trong cell Google Colab (tạo link public `share=True`) và trên trình duyệt Laptop.
   - Tính năng: Upload ảnh $\rightarrow$ Chọn chế độ (`auto` / `fast` / `tile`) và ngưỡng quyết định $T$ $\rightarrow$ Hiển thị song song:
     - Nhãn kết luận (**ẢNH THẬT** / **ẢNH GIẢ**) kèm thanh xác suất $P_{fake}$ (%).
     - Ảnh gốc có khoanh vùng viền đỏ tại vị trí chỉnh sửa.
     - Mặt nạ nhiệt (Localization Heatmap) và ảnh trực quan hóa đặc trưng MRA (DWT Level 1/2).
2. **Giao diện nhúng siêu nhẹ trên Raspberry Pi 3 (`rpi_web_ui.py` — Flask + HTML/CSS/JS thuần):**
   - Thiết kế tối giản không dùng framework nặng (tiêu thụ < 15 MB RAM), chạy như một Local Web Server trên Raspberry Pi 3 (`http://<ip-cua-pi>:5000`) để người dùng điều khiển từ trình duyệt Laptop/Điện thoại cùng mạng Wi-Fi (hoặc mở trên màn hình Pi).
   - Tính năng: Upload ảnh hoặc bấm nút **Chụp từ Camera Pi** $\rightarrow$ Chạy `rpi_inference.py` (INT8) $\rightarrow$ Trả về ảnh khoanh vùng đỏ, xác suất giả mạo $P_{fake}$, thời gian xử lý (ms/s) và dung lượng RAM đang dùng thực tế.

---

## 5. TÀI LIỆU THAM KHẢO & DỮ LIỆU HUẤN LUYỆN

- **Tham khảo kiến trúc & pipeline:** MVSS-Net / MVSS-Net++ (Multi-View Multi-Scale Supervision), TruFor (ECCV 2022), Mantra-Net.
- **Tập dữ liệu hỗ trợ:** CASIA v2.0, Columbia, NIST16 (cấu trúc thư mục gồm ảnh gốc `Au`, ảnh giả `Tp`, và mặt nạ `Gt_mask`).

---

## 6. QUY ĐỊNH THỰC THI TRÊN COLAB & TIẾT KIỆM TOKEN

- **Tự động hóa trên Google Colab:**
  - File `train.py` chuẩn CLI (`--data_dir`, `--save_dir`, `--epochs`, `--batch_size`, `--lr`, `--amp`, `--dry-run`), tự động nhận diện CUDA GPU của Colab và bật Mixed Precision (`torch.cuda.amp.autocast` + `GradScaler`).
  - File `colab_train.ipynb` phải đóng gói sẵn các cell chạy theo thứ tự: Mount Google Drive $\rightarrow$ Clone/Pull GitHub $\rightarrow$ Cài thư viện $\rightarrow$ Chạy `--dry-run` $\rightarrow$ Chạy Training chính thức (lưu `best_model.pth` thẳng vào Google Drive) $\rightarrow$ Chạy Export INT8 $\rightarrow$ Khởi chạy giao diện Gradio demo ngay trên Colab.
- **Quy tắc tiết kiệm token (Bắt buộc):**
  - Tuyệt đối không tự ý in log huấn luyện giả lập hoặc chờ quá trình train dài. Chỉ xuất mã nguồn sạch, hoàn chỉnh, có chú thích rõ ràng.
  - Bắt buộc tích hợp cờ `--dry-run` trong `train.py`: Khi chạy `!python train.py --dry-run` trên Colab, script tự tạo synthetic tensor ngẫu nhiên, chạy 1 bước Forward $\rightarrow$ Loss $\rightarrow$ Backward $\rightarrow$ Optimizer trong 3 giây để xác minh toàn bộ shape và CUDA hoạt động không lỗi trước khi train thật.

---

## 7. KẾ HOẠCH BÀN GIAO THEO 4 PHA (DỪNG SAU MỖI PHA ĐỂ KIỂM TRA)

Hãy triển khai dự án lần lượt theo 4 pha độc lập dưới đây. Sau khi xuất xong mỗi pha, dừng lại và chờ tôi xác nhận trước khi sang pha tiếp theo:

- **Pha 1 — Khởi tạo & Tiền xử lý MRA:**
  - Cấu trúc thư mục chuẩn (đồng bộ GitHub giữa Colab/PC và RPi 3), `configs/config.yaml`, `requirements_colab.txt`, `requirements_rpi.txt`, và module `src/dwt.py` (2-Level Haar DWT 9 kênh + tùy chọn MAD/ELA).
- **Pha 2 — Kiến trúc Mô hình & Bộ chia cửa sổ trượt:**
  - `src/model.py` (DWT-MobileUNet Dual-Head với MobileNetV3-Small `width=0.5`, hàm loss tổng hợp Dice + Focal + BCE).
  - `src/tiling.py` (Sliding Window `stride=160` + Gaussian Blending + giải phóng bộ nhớ `gc.collect()`).
- **Pha 3 — Dữ liệu, Training CUDA & Notebook Google Colab:**
  - `src/dataset.py` (Dataloader nạp CASIA/Columbia/NIST16 kèm Data Augmentation nhẹ).
  - `train.py` (CLI CUDA training, AMP FP16, cờ `--dry-run` 3 giây, checkpointing vào Google Drive).
  - `colab_train.ipynb` (Notebook hoàn chỉnh chạy 100% trên Google Colab từ bước Mount Drive đến Export).
- **Pha 4 — Lượng tử hóa INT8, Bring-up RPi 3 & Giao diện người dùng (UI):**
  - `export_quantize.py` (PyTorch $\rightarrow$ ONNX $\rightarrow$ lượng tử hóa INT8 sang TFLite / ONNX Runtime, dung lượng $< 2\text{ MB}$).
  - `rpi_inference.py` (Core inference chạy được trên cả Colab/Laptop và Raspberry Pi 3 với `--mode auto/fast/tile`).
  - `app_colab.py` (Giao diện Gradio chạy trên Google Colab / Laptop).
  - `rpi_web_ui.py` + `templates/index.html` (Giao diện Web Flask siêu nhẹ < 15 MB RAM để chạy demo trực tiếp trên Raspberry Pi 3).
