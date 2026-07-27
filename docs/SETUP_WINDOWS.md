# Hướng dẫn thiết lập dự án trên Windows

Tài liệu này dành cho thành viên chạy code C++/OpenMP/CUDA, chuẩn bị dữ liệu hoặc phát triển giao diện trên Windows. Các lệnh bên dưới được chạy trong PowerShell tại thư mục gốc repository.

## 1. Yêu cầu phần cứng

- Windows 10 hoặc Windows 11 64-bit.
- GPU NVIDIA hỗ trợ CUDA để build và chạy backend CUDA.
- Tối thiểu 8 GB RAM; khuyến nghị 16 GB trở lên khi benchmark ảnh 2K/4K.
- Khoảng 3 GB trống cho toolchain và môi trường Python; dữ liệu hiện tại cần dưới 100 MB.

Máy không có GPU NVIDIA vẫn có thể phát triển CPU/OpenMP, nhưng cấu hình CMake hiện tại yêu cầu CUDA Toolkit để build toàn bộ chương trình.

## 2. Cài công cụ bắt buộc

### Visual Studio

Mở Visual Studio Installer và bảo đảm đã chọn:

1. Workload **Desktop development with C++**.
2. Component **MSVC x64/x86 build tools**.
3. Component **C++ CMake tools for Windows**.
4. Windows SDK phù hợp với hệ điều hành.

Không cần thêm thủ công `cl`, CMake hoặc Ninja vào `PATH`; script của dự án tự tìm chúng qua Visual Studio Installer.

### CUDA Toolkit và driver NVIDIA

Cài NVIDIA Driver và CUDA Toolkit. Sau đó mở PowerShell mới và kiểm tra:

```powershell
nvidia-smi
nvcc --version
```

`nvidia-smi` phải hiển thị GPU; `nvcc --version` phải hiển thị phiên bản CUDA compiler.

### Python và Git

Kiểm tra:

```powershell
python --version
git --version
```

Dự án hỗ trợ Python 3.10 trở lên. Không cài package vào Python hệ thống; script sẽ tạo `.venv` riêng.

## 3. Clone repository

```powershell
git clone https://github.com/Chicken20145/ai-assisted-parallel-image-processing.git
Set-Location .\ai-assisted-parallel-image-processing
git status
```

Nếu PowerShell chặn script cục bộ, chỉ áp dụng chính sách cho tiến trình hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Không thay đổi execution policy toàn máy nếu không cần thiết.

## 4. Thiết lập môi trường Python và toolchain

```powershell
.\scripts\setup_windows.ps1
```

Script thực hiện:

- tìm Visual Studio, MSVC, CMake và Ninja;
- kiểm tra `cl` và `nvcc`;
- tạo `.venv` nếu chưa có;
- cài package từ `requirements.txt`.

Khi chỉ muốn kiểm tra lại toolchain mà không cài Python package:

```powershell
.\scripts\setup_windows.ps1 -SkipPythonPackages
```

## 5. Tải và chuẩn bị dữ liệu

```powershell
.\scripts\download_datasets.ps1
.\.venv\Scripts\python.exe .\scripts\prepare_benchmark_data.py
```

Kết quả mong đợi:

- `data/external/BSDS300/images/`: 300 ảnh gốc;
- `data/external/benchmark_suite/`: 15 ảnh benchmark;
- `data/external/benchmark_suite/metadata.csv`: nguồn, phép biến đổi, kích thước và SHA-256.

Script tải kiểm tra SHA-256 trước khi giải nén. Không commit `data/downloads/` hoặc `data/external/` vào Git.

## 6. Ghi cấu hình máy

```powershell
.\scripts\check_environment.ps1
```

Thông tin CPU, RAM, GPU, driver và CUDA được ghi vào `data/external/environment.txt`. Chạy lại lệnh này trước mỗi đợt benchmark chính thức.

## 7. Build và chạy

Build Release:

```powershell
.\scripts\build_windows.ps1 -Configuration Release
```

Build sạch hoàn toàn:

```powershell
.\scripts\build_windows.ps1 -Configuration Release -Clean
```

Kết quả probe đúng phải hiển thị số luồng OpenMP lớn hơn 0 và ít nhất một CUDA device trên máy có GPU NVIDIA.

## 8. Kiểm tra nhanh sau setup

```powershell
.\scripts\test_windows.ps1 -BuildFirst
.\.venv\Scripts\python.exe -c "import PIL,numpy,pandas,matplotlib,jsonschema,pydantic,streamlit,openai,pytest; print('Python dependencies OK')"
Get-ChildItem .\data\external\benchmark_suite\*.png | Measure-Object
.\build\parallel_image_processing.exe
```

Số ảnh PNG phải là 15.

## 9. Lỗi thường gặp

### Không tìm thấy MSVC hoặc CMake

Mở Visual Studio Installer, bổ sung workload **Desktop development with C++** và **C++ CMake tools for Windows**, sau đó mở lại PowerShell.

### `nvcc` không tồn tại

CUDA Toolkit chưa được cài hoặc PowerShell được mở trước khi cài. Cài CUDA Toolkit rồi mở terminal mới.

### `nvidia-smi` không nhận GPU

Kiểm tra NVIDIA Driver, khởi động lại Windows nếu vừa cài driver và xác nhận GPU không bị vô hiệu hóa trong Device Manager.

### Checksum dữ liệu không đúng

Xóa riêng file `data/downloads/BSDS300-images.tgz`, sau đó chạy lại script tải. Không xóa toàn bộ repository.

### Muốn tạo lại môi trường Python

Đóng các tiến trình đang dùng `.venv`, xóa riêng thư mục `.venv`, rồi chạy lại `scripts/setup_windows.ps1`.
