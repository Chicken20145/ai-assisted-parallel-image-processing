# Bảng nhiệm vụ của nhóm

## Phân công vai trò

- **Thành viên A – Tính toán song song:** thuật toán, tính đúng đắn, OpenMP, CUDA và tối ưu.
- **Thành viên B – AI và ứng dụng:** pipeline JSON hợp lệ, giao diện và tích hợp backend.
- **Thành viên C – Hậu cần và đánh giá:** dữ liệu, chạy benchmark, biểu đồ, tài liệu và điều phối demo.

Mọi phần code phải được ít nhất một thành viên khác review trước khi merge.

## Mốc 0 – Nền tảng

- [x] Tạo cấu trúc dự án và Git repository.
- [x] Tạo dự án CMake có kiểm tra OpenMP và CUDA.
- [ ] Xác nhận build Release trên máy NVIDIA cục bộ.
- [ ] Ghi lại thông tin CPU, GPU, RAM, hệ điều hành, compiler và CUDA.
- [ ] Thêm một ảnh mẫu nhỏ có giấy phép sử dụng phù hợp.

## Mốc 1 – Phiên bản tuần tự

- [ ] Thiết kế bộ đệm ảnh dùng chung và quy tắc xử lý biên.
- [ ] Hiện thực chuyển đổi grayscale.
- [ ] Hiện thực Gaussian Blur tuần tự.
- [ ] Hiện thực Sobel Edge Detection tuần tự.
- [ ] Hiện thực Histogram Equalization tuần tự.
- [ ] Viết kiểm thử tính đúng đắn và trường hợp biên.

**Điều kiện hoàn thành:** tất cả thuật toán tuần tự tạo được kết quả tham chiếu đã kiểm chứng.

## Mốc 2 – OpenMP

- [ ] Hiện thực ba phiên bản OpenMP.
- [ ] Thử nghiệm 1, 2, 4, 8 và số thread tối đa phù hợp với phần cứng.
- [ ] So sánh static, dynamic và guided scheduling khi phù hợp.
- [ ] Loại bỏ data race và đối chiếu kết quả với bản tham chiếu.

**Điều kiện hoàn thành:** kết quả OpenMP chính xác và số liệu speedup có thể lặp lại.

## Mốc 3 – CUDA

- [ ] Hiện thực CUDA kernel cơ bản cho ba thuật toán.
- [ ] Thêm kiểm tra lỗi CUDA và đo thời gian bằng CUDA Event.
- [ ] Đo riêng thời gian host-to-device, kernel và device-to-host.
- [ ] Tối ưu Gaussian/Sobel bằng shared memory.
- [ ] Tối ưu histogram bằng histogram cục bộ trong shared memory của block.
- [ ] Thử block 8x8, 16x16, 32x8 và 32x16.

**Điều kiện hoàn thành:** CUDA cơ bản và tối ưu cho kết quả đúng, mức cải thiện được ghi nhận.

## Mốc 4 – Benchmark và phân tích

- [ ] Chuẩn bị ảnh 256x256, 512x512, Full HD, 2K và 4K.
- [ ] Thêm lần chạy khởi động và ít nhất 20 lần đo chính thức.
- [ ] Xuất kết quả ra CSV.
- [ ] Tính trung bình, độ lệch chuẩn, speedup, efficiency và throughput.
- [ ] Vẽ biểu đồ so sánh.
- [ ] Rút ra quy tắc lựa chọn backend Auto.

**Điều kiện hoàn thành:** kết quả có thể tái lập và mọi nhận định hiệu năng đều có số liệu hỗ trợ.

## Mốc 5 – Giao diện và AI hỗ trợ

- [ ] Tải lên và xem trước ảnh.
- [ ] Chọn CPU tuần tự, OpenMP, CUDA hoặc Auto.
- [ ] Hiển thị ảnh đầu ra, thời gian và speedup.
- [ ] Định nghĩa JSON schema nghiêm ngặt cho pipeline được hỗ trợ.
- [ ] Chuyển yêu cầu tự nhiên thành JSON đúng schema.
- [ ] Kiểm tra tên thao tác và khoảng tham số trước khi thực thi.
- [ ] Bảo đảm AI chỉ giải thích từ số liệu đã đo.

**Điều kiện hoàn thành:** toàn bộ kịch bản demo chạy mà không cần sửa code thủ công.

## Mốc 6 – Nộp bài

- [ ] Kiểm tra build lại từ một bản clone sạch.
- [ ] Hoàn thiện báo cáo, sơ đồ, bảng số liệu và giới hạn dự án.
- [ ] Chuẩn bị slide và kịch bản demo từ năm đến bảy phút.
- [ ] Quay video demo dự phòng.
- [ ] Gắn tag Git cuối cùng là `v1.0.0`.

## Phần tùy chọn – chỉ làm sau khi đạt mọi điều kiện

- [ ] Xử lý hàng loạt một thư mục ảnh.
- [ ] Demo một đoạn video offline ngắn.
- [ ] So sánh GPU cục bộ với một GPU Google Colab như hai môi trường riêng biệt.

