# Tổng quan dự án

## Mục tiêu

Xây dựng ứng dụng xử lý ảnh song song chính xác, đo lường được và giải thích được khi nào CPU tuần tự, OpenMP hoặc CUDA phù hợp. Lớp AI chuyển yêu cầu tiếng Việt thành pipeline JSON hợp lệ; AI không trực tiếp tính toán ảnh và không tự tạo số liệu benchmark.

## Phạm vi cốt lõi

Dự án chỉ triển khai ba thuật toán:

1. Gaussian Blur.
2. Sobel Edge Detection.
3. Histogram Equalization.

Mỗi thuật toán có bốn backend:

1. CPU tuần tự làm kết quả tham chiếu.
2. OpenMP.
3. CUDA cơ bản.
4. CUDA tối ưu.

Ngoài phạm vi hiện tại: video thời gian thực, MPI, đa GPU, huấn luyện mô hình, nhận dạng vật thể/khuôn mặt, chẩn đoán y tế và triển khai cloud nhiều người dùng.

## Thành phần của repository

```text
app/            Giao diện và adapter AI
benchmarks/     CLI, cấu hình và kết quả benchmark
data/           Ảnh mẫu và dữ liệu ngoài Git
docs/           Ba tài liệu chung của dự án
include/        API C++ công khai
notebooks/      Notebook Google Colab
scripts/        Setup, build, test và chuẩn bị dữ liệu
src/common/     Dispatcher và error metrics
src/cpu/        CPU tuần tự
src/openmp/     OpenMP
src/cuda/       CUDA
tests/          Kiểm thử tính đúng đắn
```

## Trạng thái hiện tại

- Nhánh nền ổn định cho cả nhóm: `main`.
- Đã có setup Windows và Google Colab có thể tái lập.
- Đã tải BSDS300 và tạo 15 ảnh benchmark cục bộ.
- Đã có API chung, CPU tuần tự, OpenMP, CTest, MAE/MSE và CLI benchmark tổng hợp.
- Đã triển khai grayscale, Gaussian Blur, Sobel và Histogram Equalization tuần tự.
- Đã có manual UI Streamlit, JSON schema, prompt corpus và fallback chạy qua mock adapter tuần tự.
- Chưa triển khai CUDA Basic, CUDA Optimized, adapter C++ thật, AI prompt parser và đọc ảnh thật trong benchmark CLI.
- Backend chưa triển khai trả `BackendUnavailable`; đây là hành vi có chủ ý.

Các PR nền tảng đã merge:

- [PR #1 – Setup Windows, Colab và dữ liệu](https://github.com/Chicken20145/ai-assisted-parallel-image-processing/pull/1) — đã merge.
- [PR #2 – Core API và CPU tuần tự](https://github.com/Chicken20145/ai-assisted-parallel-image-processing/pull/2) — đã merge.
- [PR #4 – Backend OpenMP](https://github.com/Chicken20145/ai-assisted-parallel-image-processing/pull/4) — đã merge và đạt test Windows/Colab.

Mọi nhánh nhiệm vụ mới phải tạo từ `main` mới nhất.

## Hợp đồng ảnh

```cpp
struct Image {
    int width;
    int height;
    int channels;
    std::vector<std::uint8_t> pixels;
};
```

- Hỗ trợ grayscale một kênh và RGB ba kênh.
- RGB xen kẽ dạng `RGBRGB...`.
- Không có padding giữa các hàng.
- Buffer phải có đúng `width * height * channels` phần tử.
- Các hàm không sửa ảnh đầu vào và không đọc/ghi file trực tiếp.

## API xử lý chung

```cpp
pip::ProcessingResult result = pip::process(
    input,
    pip::Algorithm::GaussianBlur,
    params,
    pip::Backend::Sequential);
```

Mapping dùng giữa JSON và C++:

| JSON | C++ |
|---|---|
| `gaussian_blur` | `Algorithm::GaussianBlur` |
| `sobel` | `Algorithm::Sobel` |
| `histogram_equalization` | `Algorithm::HistogramEqualization` |
| `sequential` | `Backend::Sequential` |
| `openmp` | `Backend::OpenMP` |
| `cuda_basic` | `Backend::CudaBasic` |
| `cuda_optimized` | `Backend::CudaOptimized` |

Tham số:

- Gaussian: `kernel_size` là 3, 5 hoặc 7; `sigma` từ 0.1 đến 10.0.
- Sobel: `threshold` từ 0 đến 255.
- OpenMP: `thread_count` từ 1 đến 1024; giá trị 0 để runtime tự chọn.
- Sobel threshold 0 trả gradient 0–255; threshold lớn hơn 0 trả ảnh nhị phân.

Đầu ra:

| Thuật toán | Số kênh đầu ra |
|---|---|
| Gaussian Blur | Giữ nguyên đầu vào |
| Sobel | Grayscale một kênh |
| Histogram Equalization | Grayscale một kênh |

`ProcessingResult::threads_used` cho biết số thread được cấu hình cho lần chạy. Mã lỗi chung: `None`, `InvalidImage`, `InvalidParameters`, `BackendUnavailable`, `InternalError`. B phải kiểm tra `result.ok()` trước khi đọc ảnh đầu ra.

Timing chung gồm `allocation_ms`, `h2d_ms`, `kernel_ms`, `d2h_ms`, `total_ms`. CPU tuần tự hiện dùng `kernel_ms == total_ms`; CUDA phải tách riêng các giai đoạn.

## OpenMP

- Gaussian và Sobel song song hóa vòng lặp theo hàng với `schedule(static)`.
- Chuyển RGB sang grayscale cũng được song song hóa khi backend là OpenMP.
- Histogram Equalization dùng histogram 256 mức riêng cho từng thread, sau đó hợp nhất.
- Ánh xạ CDF được song song hóa theo pixel.
- Backend OpenMP giữ nguyên công thức, cách làm tròn, quy tắc biên và kiểu đầu ra của Sequential.
- Test yêu cầu kết quả khớp tuyệt đối: MAE = 0, MSE = 0, max absolute error = 0.

## Quy tắc tính toán

### Grayscale

```text
gray = (77R + 150G + 29B + 128) >> 8
```

### Gaussian Blur

```text
G(x,y) = exp(-(x²+y²)/(2σ²))
```

Kernel được chuẩn hóa để tổng bằng 1, tính độc lập từng kênh và dùng biên replicate/clamp.

### Sobel

```text
Gx = -1  0  1      Gy = -1 -2 -1
     -2  0  2            0  0  0
     -1  0  1            1  2  1
```

Độ lớn gradient là `clamp(round(sqrt(Gx² + Gy²)), 0, 255)`.

### Histogram Equalization

```text
output(v) = round((cdf(v) - cdf_min) * 255 / (N - cdf_min))
```

Ảnh chỉ có một mức xám được giữ nguyên để tránh chia cho 0.

## Dữ liệu

BSDS300 được tải từ Computer Vision Group, UC Berkeley cho mục đích nghiên cứu/giáo dục phi thương mại:

- 300 ảnh JPEG: 200 train, 100 test.
- SHA-256 archive: `A5F7D0E49FE135C75518A3543CED24470156FD69305AE77845DFF2A5138652B4`.
- SHA-256 logic của tên và nội dung 300 JPEG: `44584B06A9D2F22028D345F087F99D2428A5B6C410E8BEDE069770A60A8A24EF`.
- Nguồn: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/
- Vị trí cục bộ: `data/external/BSDS300/images/`.
- Downloader chấp nhận lớp đóng gói tar/gzip khác nhau chỉ khi checksum archive đã biết hoặc checksum logic của đúng 300 ảnh khớp; file lạ vẫn bị từ chối.

Script tạo 15 ảnh benchmark từ ba nhóm nội dung và năm kích thước: 256×256, 512×512, Full HD, 2K và 4K. Metadata và checksum từng ảnh nằm trong `data/external/benchmark_suite/metadata.csv`.

DIV2K và LOL mới được khảo sát, chưa dùng trong quy trình chính thức do dung lượng hoặc điều khoản phân phối chưa đủ rõ.

## Tiêu chí hoàn thành

- Cả ba thuật toán chạy đúng với ảnh thường và ảnh biên.
- OpenMP/CUDA được so với CPU bằng MAE, MSE và max absolute error.
- Benchmark Release có warm-up 3–5 lần và ít nhất 20 lần đo.
- CUDA tách H2D, kernel, D2H và end-to-end.
- Báo cáo có mean, standard deviation, speedup, efficiency và throughput.
- Giao diện hiển thị ảnh, pipeline hợp lệ, backend thực tế và timing.
- Clone sạch có thể setup, build và test theo tài liệu.
