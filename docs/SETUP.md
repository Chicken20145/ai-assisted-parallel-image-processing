# Thiết lập và chạy dự án

## Windows

### Yêu cầu

- Windows 10/11 64-bit.
- Visual Studio với **Desktop development with C++**, MSVC x64/x86, C++ CMake tools và Windows SDK.
- NVIDIA Driver và CUDA Toolkit.
- Python 3.10 trở lên.
- Git.
- Khuyến nghị RAM từ 16 GB và GPU NVIDIA cho CUDA.

Kiểm tra:

```powershell
nvidia-smi
nvcc --version
python --version
git --version
```

### Setup đầy đủ

```powershell
git clone https://github.com/Chicken20145/ai-assisted-parallel-image-processing.git
Set-Location .\ai-assisted-parallel-image-processing
git switch main
git pull --ff-only origin main

.\scripts\setup_windows.ps1
.\scripts\download_datasets.ps1
.\.venv\Scripts\python.exe .\scripts\prepare_benchmark_data.py
.\scripts\check_environment.ps1
.\scripts\test_windows.ps1 -BuildFirst
```

Nếu PowerShell chặn script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Script tự tìm MSVC, CMake và Ninja đi kèm Visual Studio; không cần thêm thủ công vào PATH.

### Build/test riêng

```powershell
.\scripts\build_windows.ps1 -Configuration Release
.\scripts\build_windows.ps1 -Configuration Release -Clean
.\scripts\test_windows.ps1
```

Kết quả đúng phải có `100% tests passed`, số luồng OpenMP dương và ít nhất một CUDA device trên máy NVIDIA.

## Google Colab

Repository là private. Mỗi thành viên phải là collaborator và cấp quyền GitHub cho Colab.

Mở notebook nhánh hiện tại:

```text
https://colab.research.google.com/github/Chicken20145/ai-assisted-parallel-image-processing/blob/main/notebooks/colab_setup.ipynb
```

Nếu báo 404:

1. Vào **File → Open notebook → GitHub**.
2. Bật **Bao gồm các kho lưu trữ riêng tư**.
3. Authorize GitHub bằng tài khoản có quyền repository.
4. Chọn branch `main` và `notebooks/colab_setup.ipynb`.

Sau đó:

1. Chọn **Runtime → Change runtime type → GPU**.
2. Chạy các cell từ trên xuống.
3. Không nhấn **Lưu trong GitHub** nếu chỉ chạy notebook.

Setup tự động thực hiện: kiểm tra GPU/nvcc, cài dependency, tải và xác minh BSDS300, tạo 15 ảnh benchmark, build Release, chạy CTest/probe/smoke benchmark và ghi `environment_colab.txt`.

Trong cùng runtime, cập nhật code mới mà không setup lại:

```bash
%cd /content/ai-assisted-parallel-image-processing
!git pull --ff-only origin main
!bash scripts/build_colab.sh
!bash scripts/test_colab.sh
```

Kết quả thành công:

```text
100% tests passed
CUDA devices available: 1
CTest, probe CUDA/OpenMP và benchmark smoke test trên Colab đều đạt.
Colab setup completed successfully.
```

Phiên xác nhận ngày 27/07/2026 dùng NVIDIA T4, CUDA compiler 12.8.93, GNU 11.4.0, OpenMP 2 luồng và một CUDA device. Colab có thể cấp phần cứng khác ở phiên sau.

## Dữ liệu sinh ra

```text
data/downloads/                         Archive tải về
data/external/BSDS300/images/           300 ảnh nguồn
data/external/benchmark_suite/          15 ảnh benchmark
data/external/benchmark_suite/metadata.csv
data/external/environment*.txt          Cấu hình từng máy/phiên
```

Các thư mục trên không được commit. Nếu mạng trả file lỗi hoặc tải dở, downloader tự xóa cache hỏng, retry ba lần và chỉ thay dataset đã giải nén sau khi xác minh đủ 300 ảnh. Không đổi checksum theo một file tải lỗi.

## Benchmark smoke test

Windows:

```powershell
.\build\image_benchmark.exe `
  --algorithm gaussian_blur `
  --backend sequential `
  --width 1920 --height 1080 --channels 3 `
  --kernel-size 5 --sigma 1.2 `
  --warmup 3 --runs 20
```

Colab:

```bash
./build-colab/image_benchmark \
  --algorithm gaussian_blur \
  --backend sequential \
  --width 1920 --height 1080 --channels 3 \
  --kernel-size 5 --sigma 1.2 \
  --warmup 3 --runs 20 > gaussian_sequential.csv
```

CLI hiện tạo ảnh tổng hợp để kiểm tra pipeline; chưa dùng số liệu này làm kết luận benchmark cuối cùng trên dataset.

Benchmark OpenMP với số thread cụ thể:

```powershell
.\build\image_benchmark.exe `
  --algorithm gaussian_blur `
  --backend openmp --threads 4 `
  --width 1920 --height 1080 --channels 3 `
  --kernel-size 5 --sigma 1.2 `
  --warmup 3 --runs 20
```

Thử `--threads 1`, `2`, `4`, `8` và số thread tối đa hợp lý của máy. Giá trị 0 để OpenMP runtime tự chọn. Không so sánh trực tiếp Colab 2 thread với Windows 20 thread như cùng một môi trường.

## Lưu kết quả Colab

Benchmark trên `/content`, sau đó mới mount Drive và sao chép kết quả để I/O mạng không ảnh hưởng phép đo:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Luôn lưu CSV cùng `environment_colab.txt`. Không gộp dữ liệu giữa hai phiên có phần cứng khác nhau.

## API key AI

- Windows: dùng biến môi trường `OPENAI_API_KEY` hoặc `.env` cục bộ.
- Colab: dùng Secrets với tên `OPENAI_API_KEY`.
- Không commit, in ra output hoặc gửi secret qua chat.

## Lỗi thường gặp

- Không có MSVC/CMake: bổ sung workload C++ và CMake tools trong Visual Studio Installer.
- Không có `nvcc`: cài CUDA Toolkit hoặc chọn lại Colab GPU runtime.
- `nvidia-smi` không nhận GPU: kiểm tra driver/runtime.
- Colab 404: authorize private repository, chọn branch `main` rồi mở lại notebook.
- Colab báo checksum BSDS300 sai: chạy lại cell setup; downloader tự xóa file lỗi và thử tối đa ba lần. Không sửa checksum theo hash lỗi. Nếu vẫn thất bại, kết nối lại runtime rồi chạy lại.
- Runtime Colab reset: chạy lại notebook từ đầu; chỉ kết quả đã chép sang Drive được giữ lại.
- Drive chậm: không dùng Drive làm thư mục benchmark trực tiếp.
