# Môi trường phát triển đã xác minh

## Máy Windows cục bộ

- CPU: Intel Core i9-12900H, 14 nhân, 20 luồng.
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB VRAM.
- CUDA Toolkit: 13.3.
- NVIDIA Driver: 595.97.
- Compiler: MSVC 19.51 (Visual Studio Community 2026).
- Build system: CMake 4.3.1 và Ninja 1.13.2 đi kèm Visual Studio.
- Python: 3.13, sử dụng môi trường riêng `.venv`.

Kết quả probe Release đã xác nhận OpenMP dùng được 20 luồng và CUDA nhận 1 thiết bị. Thông tin động của mỗi phiên benchmark phải được ghi lại bằng:

```powershell
.\scripts\check_environment.ps1
```

File sinh ra nằm tại `data/external/environment.txt` và không được commit vì có thông tin cụ thể của máy/ngày chạy.

## Google Colab

Colab dùng máy ảo tạm thời và loại GPU không cố định. Mỗi phiên phải:

1. Chọn GPU runtime trước khi setup.
2. Chạy `bash scripts/setup_colab.sh`.
3. Giữ file `data/external/environment_colab.txt` cùng kết quả benchmark của phiên đó.
4. Benchmark trên ổ cục bộ `/content`, sau đó mới sao chép kết quả sang Drive.
5. Không so sánh hoặc gộp trực tiếp kết quả giữa hai phiên có GPU/CPU khác nhau.

Notebook khởi động nằm tại `notebooks/colab_setup.ipynb`. Dữ liệu và thư viện phải được thiết lập lại sau khi runtime bị hủy; CSV/biểu đồ quan trọng cần lưu sang Drive ở cuối phiên.

### Phiên Colab đã xác nhận ngày 27/07/2026

- GPU: NVIDIA T4.
- CUDA compiler: 12.8.93.
- Host compiler: GNU 11.4.0.
- OpenMP: 2 luồng.
- CUDA devices: 1.
- CTest: 100% test đạt.
- Probe và benchmark smoke test: đạt.

Cấu hình này chỉ xác nhận script hoạt động trên một phiên cụ thể, không phải cấu hình cố định của Colab và không được mặc định dùng cho phiên benchmark khác.

## Thành phần Python

`requirements.txt` bao gồm thư viện xử lý ảnh/dữ liệu, biểu đồ, validation JSON, giao diện Streamlit, OpenAI API và pytest. Cài đặt bằng `scripts/setup_windows.ps1`; không cài trực tiếp vào Python hệ thống.

## Biến môi trường AI

Khi phần AI được triển khai, API key phải đặt trong biến môi trường `OPENAI_API_KEY` hoặc file `.env` cục bộ. Không ghi API key vào source, notebook, log hay Git history.
