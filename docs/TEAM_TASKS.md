# Phân công và kế hoạch nhóm

## Nguyên tắc chung

- Chỉ tập trung Gaussian Blur, Sobel và Histogram Equalization.
- Mỗi thuật toán phải có Sequential, OpenMP, CUDA Basic và CUDA Optimized.
- AI chỉ tạo pipeline JSON hợp lệ, không thay thế C++/CUDA và không tạo số liệu.
- Mỗi thay đổi phải build/test và được ít nhất một thành viên khác review trước khi merge.
- Không commit dataset lớn, build, `.venv`, kết quả tạm hoặc API key.

## Thành viên A – lõi xử lý song song

### Trách nhiệm

- Kiểu ảnh, API, validation và dispatcher.
- CPU tuần tự làm kết quả tham chiếu.
- OpenMP, CUDA cơ bản và CUDA tối ưu.
- Error metrics, test kỹ thuật, timing và tài liệu công thức.

### Thứ tự thực hiện

1. CPU tuần tự và test trường hợp biên — đã có trên `feature/core-api-cpu`.
2. OpenMP Gaussian/Sobel theo pixel hoặc hàng với `schedule(static)`.
3. OpenMP Histogram bằng histogram riêng từng thread rồi hợp nhất.
4. Thử 1, 2, 4, 8 và số thread hợp lý; không hard-code theo máy.
5. CUDA Basic: một thread/pixel, kiểm tra bounds và mọi CUDA error.
6. Tách allocation, H2D, kernel, D2H và total bằng CUDA Event.
7. CUDA Optimized: shared memory + halo cho convolution, shared histogram theo block.
8. So sánh mọi backend với Sequential bằng MAE/MSE/max error.

### Bàn giao cho B

- Header trong `include/` và API `pip::process()` ổn định.
- Mapping thuật toán/backend, tham số hợp lệ, kiểu đầu ra và mã lỗi.
- Timing và fallback khi backend không khả dụng.
- Ví dụ gọi hàm; B không include trực tiếp code trong `src/`.

### Bàn giao cho C

- Lệnh build/test/benchmark.
- Thread count, schedule và block size cần thử.
- Ngưỡng sai số chấp nhận được.
- Giải thích kỹ thuật cho từng tối ưu.

### Kiểm tra bắt buộc

- Ảnh 1×1, 2×2, 7×5 và ảnh nhỏ hơn kernel.
- RGB/grayscale, ảnh hằng số, kích thước không chia hết block.
- Không data race, không bỏ qua CUDA error.
- API và kiểu đầu ra không thay đổi giữa backend.

## Thành viên B – AI và ứng dụng

### Trách nhiệm

- JSON schema, validation, prompt và bộ prompt test.
- Giao diện tải/xem/lưu ảnh, chọn thuật toán/backend và hiển thị timing.
- Adapter gọi lõi C++/CUDA; chế độ thủ công và fallback.

### Thứ tự thực hiện

1. Schema chỉ nhận ba thuật toán; kernel 3/5/7, sigma 0.1–10, threshold 0–255.
2. Giới hạn số operation và từ chối JSON sai trước khi gọi backend.
3. Tạo ít nhất 20 prompt: hợp lệ, mơ hồ, ngoài phạm vi, injection và tiếng Việt không dấu.
4. Xây UI Streamlit/Gradio với chế độ thủ công dùng được khi mất AI.
5. Adapter chỉ gọi `pip::process()`, kiểm tra `result.ok()` và hiển thị lỗi thân thiện.
6. Hiển thị pipeline đã validate, backend thực tế, kernel/transfer/total time và speedup.
7. Auto fallback sang OpenMP khi CUDA không khả dụng.

### Bàn giao

- Schema và mapping cho A.
- Hướng dẫn chạy UI, prompt mẫu, ảnh chụp và lỗi thường gặp cho C.
- Không sao chép lại thuật toán của A bằng Python.
- API key chỉ nằm trong environment/Colab Secrets.

## Thành viên C – dữ liệu, benchmark và báo cáo

### Trách nhiệm

- Cấu hình máy/Colab, dataset, metadata và checksum.
- Quy trình benchmark, CSV thô, biểu đồ và phân tích.
- Git/PR, báo cáo, slide và demo dự phòng.

### Thứ tự thực hiện

1. Chạy setup và giữ `environment*.txt` của từng phiên.
2. Xác nhận đủ ba nhóm nội dung và năm độ phân giải.
3. Build Release, warm-up 3–5 lần, đo ít nhất 20 lần.
4. Lưu từng lần đo; CUDA phải tách H2D/kernel/D2H/total.
5. Tính mean, standard deviation, speedup, OpenMP efficiency và throughput.
6. Vẽ ảnh hưởng thread count, block size và CUDA Basic/Optimized.
7. Không trộn máy Windows và Colab hoặc hai phiên Colab khác cấu hình.
8. Mỗi biểu đồ có tên, đơn vị, cấu hình và nhận xét.

CSV tối thiểu:

```text
timestamp,environment,algorithm,image_width,image_height,backend,
threads,block_x,block_y,kernel_size,run,h2d_ms,kernel_ms,d2h_ms,total_ms,mae
```

C có quyền từ chối số liệu thiếu cấu hình máy, số lần chạy, CSV thô hoặc không tái lập được.

## Phối hợp hiện tại

### B dùng ngay

- Đọc `include/image_types.hpp` và `include/processing_api.hpp`.
- Tích hợp `Sequential`; giữ UI cho backend khác nhưng xử lý `BackendUnavailable`.
- Không phụ thuộc benchmark CLI để đọc ảnh thật vì tính năng đó chưa có.

### C dùng ngay

- Setup Windows/Colab và kiểm tra 15 ảnh benchmark.
- Chuẩn bị pipeline CSV bằng CLI ảnh tổng hợp.
- Không dùng smoke benchmark để kết luận hiệu năng cuối cùng.

### A làm tiếp

- Triển khai ba thuật toán OpenMP trên API hiện tại.
- Thêm test so sánh OpenMP với Sequential.
- Sau khi OpenMP ổn định mới chuyển sang CUDA Basic.

## Mốc dự án

1. Nền tảng/setup/dataset.
2. CPU tuần tự và schema validation.
3. OpenMP và benchmark thread/schedule.
4. CUDA Basic/Optimized và benchmark block/transfer.
5. AI, UI và Auto backend.
6. Benchmark chính thức, báo cáo và demo.

Một mốc chỉ hoàn thành khi code/tài liệu đúng vị trí, có cách kiểm tra, test đạt, không chứa secret/file sinh ra và đã được review/merge.

## Quy trình Git

1. Pull branch nền mới nhất.
2. Mỗi nhiệm vụ dùng branch riêng.
3. Commit nhỏ, thông điệp rõ.
4. Push và mở PR kèm cách test.
5. Một thành viên khác review.
6. Chỉ merge khi build/test đạt.

Không nhấn **Lưu trong GitHub** từ Colab nếu chỉ chạy notebook; thao tác đó tạo commit trực tiếp và dễ gây xung đột.
