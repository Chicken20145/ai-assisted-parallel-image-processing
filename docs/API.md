# API lõi xử lý ảnh

Tài liệu này là hợp đồng bàn giao từ Thành viên A cho Thành viên B và C. API công khai nằm trong `include/`; code giao diện không include trực tiếp header nội bộ trong `src/`.

## Kiểu ảnh

```cpp
pip::Image image;
image.width = 1920;
image.height = 1080;
image.channels = 3;
image.pixels = rgb_buffer;
```

Quy ước:

- Chỉ hỗ trợ ảnh grayscale một kênh hoặc RGB ba kênh.
- Buffer liên tục theo hàng, pixel RGB xen kẽ dạng `RGBRGB...`.
- Mỗi kênh dùng `uint8_t`, giá trị 0–255.
- Không có padding giữa các hàng.
- `pixels.size()` phải bằng `width * height * channels`.
- `Image::is_valid()` kiểm tra toàn bộ điều kiện trên.

## Thuật toán và backend

```cpp
enum class Algorithm {
    GaussianBlur,
    Sobel,
    HistogramEqualization,
};

enum class Backend {
    Sequential,
    OpenMP,
    CudaBasic,
    CudaOptimized,
};
```

Backend `Sequential` đã hoạt động. Ba backend còn lại đã có giá trị enum ổn định nhưng hiện trả `BackendUnavailable`; nhờ vậy B có thể hoàn thành adapter mà không cần đổi API sau này.

Mapping JSON đề xuất:

| JSON | C++ |
|---|---|
| `gaussian_blur` | `Algorithm::GaussianBlur` |
| `sobel` | `Algorithm::Sobel` |
| `histogram_equalization` | `Algorithm::HistogramEqualization` |
| `sequential` | `Backend::Sequential` |
| `openmp` | `Backend::OpenMP` |
| `cuda_basic` | `Backend::CudaBasic` |
| `cuda_optimized` | `Backend::CudaOptimized` |

## Tham số

```cpp
pip::ProcessingParams params;
params.kernel_size = 5;
params.sigma = 1.2F;
params.threshold = 100;
```

- `kernel_size`: chỉ 3, 5 hoặc 7; dùng cho Gaussian Blur.
- `sigma`: số hữu hạn trong khoảng 0.1–10.0; dùng cho Gaussian Blur.
- `threshold`: số nguyên 0–255; dùng cho Sobel.
- Tham số không liên quan đến thuật toán đang chạy được bỏ qua.

Sobel có hai chế độ:

- `threshold == 0`: trả độ lớn gradient 0–255.
- `threshold > 0`: trả ảnh nhị phân; pixel đạt ngưỡng là 255, còn lại là 0.

## Gọi dispatcher

```cpp
#include "processing_api.hpp"

pip::ProcessingResult result = pip::process(
    input,
    pip::Algorithm::GaussianBlur,
    params,
    pip::Backend::Sequential);

if (!result.ok()) {
    show_error(result.error_message);
    return;
}

display(result.output);
display_time(result.timing.total_ms);
```

Dispatcher bắt lỗi tham số và trả `ProcessingResult`; B không cần bắt exception cho luồng xử lý thông thường.

## Kiểu đầu ra

| Thuật toán | Đầu vào | Đầu ra |
|---|---|---|
| Gaussian Blur | grayscale hoặc RGB | Giữ nguyên số kênh |
| Sobel | grayscale hoặc RGB | Grayscale một kênh |
| Histogram Equalization | grayscale hoặc RGB | Grayscale một kênh |

## Timing

```cpp
struct Timing {
    double allocation_ms;
    double h2d_ms;
    double kernel_ms;
    double d2h_ms;
    double total_ms;
};
```

Ở backend CPU tuần tự, `kernel_ms` là thời gian toàn bộ phép xử lý trong lõi và `total_ms` bằng `kernel_ms`; các trường CUDA bằng 0. Backend CUDA sau này phải điền riêng allocation, H2D, kernel và D2H bằng CUDA Event hoặc cơ chế đo phù hợp.

## Mã lỗi

| Mã | Ý nghĩa |
|---|---|
| `None` | Thành công |
| `InvalidImage` | Kích thước, số kênh hoặc buffer sai |
| `InvalidParameters` | Kernel, sigma hoặc threshold ngoài phạm vi |
| `BackendUnavailable` | Backend chưa build, chưa triển khai hoặc không khả dụng |
| `InternalError` | Lỗi không dự kiến trong lõi |

UI nên hiển thị `error_message` cho người dùng và ghi `to_string(error)` vào log.

## So sánh kết quả

```cpp
#include "error_metrics.hpp"

pip::ErrorMetrics metrics = pip::compare_images(reference, candidate);
```

Hai ảnh phải có cùng width, height và channels. Hàm trả MAE, MSE và sai số tuyệt đối lớn nhất. C dùng CPU tuần tự làm `reference` khi đánh giá OpenMP/CUDA.

## Trạng thái thread safety

Các hàm CPU hiện không dùng trạng thái toàn cục và có thể được gọi độc lập. Không sửa buffer đầu vào. Quy tắc này phải được giữ khi thêm OpenMP/CUDA.
