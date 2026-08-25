# AI-Assisted Parallel Image Processing

Dự án xử lý ảnh song song bằng C++17, OpenMP và CUDA. Hệ thống cho phép chạy cùng một thuật toán trên CPU tuần tự, CPU đa luồng và GPU để so sánh tốc độ, khả năng mở rộng và độ chính xác trên cùng dữ liệu.

## Thành phần chính

- **Thuật toán:** Gaussian Blur, Sobel Edge Detection và Histogram Equalization.
- **Backend:** CPU Sequential, OpenMP, CUDA Basic và CUDA Optimized.
- **Giao diện:** Streamlit hỗ trợ xử lý một ảnh, pipeline nhiều bước và benchmark 300 ảnh.
- **AI hỗ trợ:** chuyển yêu cầu tiếng Việt thành cấu hình pipeline; việc xử lý pixel vẫn do core C++/CUDA thực hiện.
- **Đánh giá:** thời gian chạy, speedup, throughput, MAE, MSE và sai lệch pixel lớn nhất.

## Kiến trúc

```text
Người dùng → Streamlit/AI → Python adapter → Core C++
                                           ├─ CPU Sequential
                                           ├─ OpenMP
                                           ├─ CUDA Basic
                                           └─ CUDA Optimized
```

Kết quả gồm ảnh đã xử lý, thời gian thực thi và dữ liệu CSV phục vụ benchmark. Dataset chính là BSDS300 gồm 300 ảnh.

## Chạy trên Google Colab

[![Mở bằng Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chicken20145/ai-assisted-parallel-image-processing/blob/main/notebooks/colab_setup.ipynb)

Chọn **T4 GPU**, thêm secret `NGROK_AUTHTOKEN`, chạy cell **1 → 2 → 3**, rồi bấm **MỞ GIAO DIỆN PIXEL LAB**.

## Tài liệu

- [`docs/PROJECT.md`](docs/PROJECT.md): mục tiêu, kiến trúc và cách thiết lập.
- [`docs/C_GUIDE.md`](docs/C_GUIDE.md): quy trình benchmark và phân tích kết quả.
