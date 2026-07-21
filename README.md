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

## Build ban đầu

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

