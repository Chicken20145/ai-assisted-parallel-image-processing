# Thuật toán và quy tắc tính toán

Tài liệu mô tả bản CPU tuần tự dùng làm kết quả tham chiếu cho OpenMP và CUDA.

## Quy tắc chung

- Pixel dùng `uint8_t` trong khoảng 0–255.
- Ảnh RGB lưu xen kẽ theo thứ tự R, G, B.
- Mọi convolution dùng xử lý biên replicate: tọa độ ngoài ảnh được clamp về pixel biên gần nhất.
- Hàm chỉ xử lý buffer, không đọc hoặc ghi file.
- Kết quả CPU tuần tự ưu tiên tính rõ ràng và xác định được hơn tối ưu sớm.

## Chuyển RGB sang grayscale

Công thức số nguyên cố định:

```text
gray = (77R + 150G + 29B + 128) >> 8
```

Hệ số xấp xỉ 0.299R + 0.587G + 0.114B và tránh khác biệt làm tròn giữa backend. Ảnh grayscale đầu vào được trả lại không thay đổi.

Độ phức tạp thời gian `O(W × H)` và bộ nhớ đầu ra `O(W × H)`.

## Gaussian Blur

Kernel hai chiều được sinh từ:

```text
G(x,y) = exp(-(x²+y²)/(2σ²))
```

Sau đó toàn bộ hệ số được chia cho tổng kernel để tổng bằng 1. Hỗ trợ kernel 3×3, 5×5 và 7×7; sigma từ 0.1 đến 10.0.

Thuật toán xử lý độc lập từng kênh và giữ nguyên số kênh đầu vào. Giá trị tích lũy dùng `float`, sau đó làm tròn gần nhất và clamp 0–255.

Độ phức tạp thời gian `O(W × H × C × K²)`; bộ nhớ đầu ra `O(W × H × C)`.

## Sobel Edge Detection

Nếu đầu vào là RGB, ảnh được chuyển sang grayscale trước. Hai kernel:

```text
Gx = -1  0  1      Gy = -1 -2 -1
     -2  0  2            0  0  0
     -1  0  1            1  2  1
```

Độ lớn gradient:

```text
magnitude = clamp(round(sqrt(Gx² + Gy²)), 0, 255)
```

Với threshold bằng 0, trả trực tiếp magnitude. Với threshold lớn hơn 0, trả 255 nếu `magnitude >= threshold`, ngược lại trả 0.

Độ phức tạp thời gian `O(W × H)`; bộ nhớ đầu ra `O(W × H)`.

## Histogram Equalization

Nếu đầu vào là RGB, ảnh được chuyển sang grayscale. Quy trình:

1. Đếm histogram 256 mức xám.
2. Tính cumulative distribution function (CDF).
3. Tìm `cdf_min` khác 0 đầu tiên.
4. Ánh xạ:

```text
output(v) = round((cdf(v) - cdf_min) × 255 / (N - cdf_min))
```

Nếu toàn bộ ảnh chỉ có một mức xám, trả ảnh grayscale không đổi để tránh chia cho 0.

Độ phức tạp thời gian `O(W × H + 256)`; bộ nhớ phụ `O(256)` ngoài ảnh đầu ra.

## Trường hợp biên đã kiểm tra

- Ảnh 1×1 và 2×2.
- Ảnh 7×5 không liên quan kích thước block.
- Ảnh nhỏ hơn Gaussian kernel.
- Ảnh grayscale và RGB.
- Ảnh hằng số.
- Buffer sai kích thước và số kênh không hỗ trợ.
- Kernel, sigma và threshold ngoài phạm vi.

## Nguyên tắc cho OpenMP/CUDA

- Không thay đổi công thức, quy tắc biên hoặc cách làm tròn nếu chưa cập nhật test và tài liệu.
- Mọi backend phải so sánh với CPU tuần tự bằng MAE, MSE và max absolute error.
- Không sửa buffer đầu vào.
- Giữ nguyên kiểu đầu ra và hợp đồng lỗi trong `processing_api.hpp`.
