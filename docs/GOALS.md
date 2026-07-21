# Mục tiêu và tiêu chí thành công

## Mục tiêu chính

Xây dựng một ứng dụng xử lý ảnh song song chính xác, có thể đo lường và giải thích được khi nào, tại sao OpenMP hoặc CUDA cải thiện hiệu năng so với CPU tuần tự.

## Mục tiêu kỹ thuật

1. Hiện thực Gaussian Blur, Sobel Edge Detection và Histogram Equalization.
2. Cung cấp phiên bản CPU tuần tự, OpenMP, CUDA cơ bản và CUDA tối ưu.
3. Tối ưu ít nhất một phép convolution bằng shared memory.
4. Đo riêng thời gian kernel, truyền dữ liệu và tổng thời gian end-to-end.
5. Kiểm tra kết quả song song với bản tuần tự bằng MAE hoặc MSE.
6. Tìm số thread OpenMP và kích thước CUDA block phù hợp bằng thực nghiệm.
7. Xây dựng chế độ Auto dựa trên quy tắc rút ra từ benchmark.
8. Chuyển yêu cầu ngôn ngữ tự nhiên thành pipeline JSON được kiểm tra hợp lệ.

## Tiêu chí thành công

- Cả ba thuật toán vượt qua kiểm thử với ảnh thông thường và kích thước biên.
- Benchmark bao gồm ảnh 256x256, 512x512, Full HD, 2K và 4K.
- Mỗi cấu hình được chạy khởi động và đo ít nhất 20 lần.
- Báo cáo có thời gian trung bình, độ lệch chuẩn, speedup, efficiency và throughput.
- Kết quả CUDA tách thời gian kernel khỏi thời gian truyền dữ liệu.
- Giao diện hiển thị ảnh đầu vào, đầu ra, backend được chọn và thời gian đo.
- Một bản clone sạch có thể build và chạy theo đúng hướng dẫn.

## Ngoài phạm vi

- Xử lý video thời gian thực trong phiên bản chính
- MPI hoặc chạy đa GPU
- Huấn luyện hoặc fine-tune mô hình AI
- Nhận dạng vật thể hoặc khuôn mặt
- Đưa ra kết luận chẩn đoán y tế
- Triển khai cloud hoặc tài khoản đa người dùng
- Thêm bộ lọc mới trước khi hoàn thành ba thuật toán cốt lõi

