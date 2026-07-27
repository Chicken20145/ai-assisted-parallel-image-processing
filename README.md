# Xử lý ảnh song song có AI hỗ trợ bằng OpenMP và CUDA

## Mục tiêu dự án

Thiết kế, hiện thực và đánh giá một hệ thống xử lý ảnh song song gồm các phiên bản CPU tuần tự, OpenMP, CUDA cơ bản và CUDA tối ưu. Lớp AI hỗ trợ chuyển yêu cầu ngôn ngữ tự nhiên thành pipeline xử lý ảnh đã được kiểm tra tính hợp lệ.

## Phạm vi cố định

Dự án tập trung vào đúng ba thuật toán:

1. Gaussian Blur
2. Sobel Edge Detection
3. Histogram Equalization

Mỗi thuật toán có bốn phiên bản:

- CPU tuần tự làm mốc so sánh
- CPU song song bằng OpenMP
- CUDA cơ bản
- CUDA tối ưu

Dự án được xem là hoàn thành khi các phiên bản cho kết quả đúng, benchmark có thể tái lập, giao diện hoạt động ổn định và AI tạo được pipeline hợp lệ. Video, đa GPU, MPI, huấn luyện mô hình và triển khai cloud nằm ngoài phạm vi cốt lõi.

## Cấu trúc repository

```text
app/            Giao diện và tích hợp pipeline AI
benchmarks/     Định nghĩa benchmark và kết quả sinh ra
data/samples/   Ảnh kiểm thử dung lượng nhỏ
docs/           Mục tiêu, nhiệm vụ, thiết kế và báo cáo
include/        Header C++/CUDA dùng chung
scripts/        Công cụ build và benchmark
src/cpu/        Phiên bản CPU tuần tự
src/openmp/     Phiên bản OpenMP
src/cuda/       Phiên bản CUDA
tests/          Kiểm tra tính đúng đắn và trường hợp biên
```

## Thiết lập Windows

Hướng dẫn đầy đủ: [`docs/SETUP_WINDOWS.md`](docs/SETUP_WINDOWS.md).

Yêu cầu: Visual Studio có workload **Desktop development with C++**, CUDA Toolkit và Python 3.10 trở lên. Thiết lập nhanh trong PowerShell:

```powershell
.\scripts\setup_windows.ps1
.\scripts\download_datasets.ps1
.\.venv\Scripts\python.exe .\scripts\prepare_benchmark_data.py
.\scripts\check_environment.ps1
.\scripts\build_windows.ps1 -Configuration Release
```

Script tự tìm MSVC, CMake và Ninja đi kèm Visual Studio, kiểm tra checksum dữ liệu, tạo 15 ảnh benchmark ở năm độ phân giải và build chương trình ở chế độ Release.

Chạy kiểm thử lõi CPU:

```powershell
.\scripts\test_windows.ps1 -BuildFirst
```

Chạy benchmark tổng hợp tối thiểu:

```powershell
.\build\image_benchmark.exe --algorithm gaussian_blur --backend sequential --width 1920 --height 1080 --channels 3 --kernel-size 5 --sigma 1.2 --warmup 3 --runs 20
```

API tích hợp được mô tả trong [`docs/API.md`](docs/API.md); công thức và quy tắc biên nằm trong [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md).

## Thiết lập Google Colab

Hướng dẫn đầy đủ: [`docs/SETUP_COLAB.md`](docs/SETUP_COLAB.md).

1. Mở [notebook thiết lập Colab](notebooks/colab_setup.ipynb) trên Google Colab.
2. Chọn **Runtime → Change runtime type → GPU**.
3. Chạy lần lượt các cell; cell chính gọi `scripts/setup_colab.sh` để cài dependency, tải dữ liệu, build Release và ghi cấu hình runtime.
4. Chạy benchmark trên `/content`; chỉ sao chép kết quả cuối sang Google Drive để độ trễ Drive không ảnh hưởng phép đo.

Có thể chạy trực tiếp trong một repository đã clone:

```bash
bash scripts/setup_colab.sh
```

Khi đã setup trong cùng runtime và chỉ cần cập nhật code mới:

```bash
git pull --ff-only
bash scripts/build_colab.sh
bash scripts/test_colab.sh
```

GPU, CPU, CUDA và giới hạn Colab có thể thay đổi giữa các phiên. Không gộp số đo của các phiên có cấu hình khác nhau nếu không ghi chú rõ.

## Build thủ công

Yêu cầu: CMake, trình biên dịch C++, CUDA Toolkit và trình biên dịch hỗ trợ OpenMP.

```powershell
cmake -S . -B build
cmake --build build --config Release
./build/Release/parallel_image_processing.exe
```

Với trình build một cấu hình, file thực thi có thể nằm trực tiếp trong `build/`.

## Chỉ số đánh giá

- Tổng thời gian xử lý end-to-end
- Thời gian CUDA kernel và truyền dữ liệu CPU–GPU
- Speedup: `S(p) = T(1) / T(p)`
- Hiệu suất OpenMP: `E(p) = S(p) / p`
- Thông lượng triệu pixel/giây
- MAE/MSE so với phiên bản CPU tuần tự

Xem `docs/GOALS.md` và `docs/TASKS.md` để biết phạm vi và kế hoạch công việc đã thống nhất.
