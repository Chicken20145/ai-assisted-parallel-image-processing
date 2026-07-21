# Kế hoạch và hướng dẫn công việc cho nhóm

Tài liệu này mô tả cụ thể mỗi thành viên phải làm gì, tạo ra file nào, kiểm tra như thế nào và bàn giao cho ai. Khi nhận việc, thành viên đọc toàn bộ phần của mình trước khi bắt đầu.

## 1. Phạm vi bắt buộc

Dự án chỉ tập trung vào ba thuật toán:

1. Gaussian Blur.
2. Sobel Edge Detection.
3. Histogram Equalization.

Mỗi thuật toán phải có bốn phiên bản:

1. CPU tuần tự làm kết quả tham chiếu.
2. CPU song song bằng OpenMP.
3. CUDA cơ bản.
4. CUDA tối ưu.

AI chỉ nhận yêu cầu tự nhiên và tạo pipeline JSON hợp lệ. AI không thay thế phần tính toán OpenMP/CUDA và không được tự tạo số liệu benchmark.

## 2. Phân công tổng quát

| Vai trò | Trách nhiệm chính | Sản phẩm phải bàn giao |
|---|---|---|
| Thành viên A – Tính toán song song | Thuật toán, C++, OpenMP, CUDA, kiểm tra sai số và tối ưu | Source code, unit test kỹ thuật, tài liệu thuật toán và thông số kernel |
| Thành viên B – AI và ứng dụng | Giao diện, JSON schema, prompt, điều phối backend và tích hợp thư viện C++/CUDA | Ứng dụng chạy được, module AI, bộ prompt test và hướng dẫn sử dụng |
| Thành viên C – Hậu cần và đánh giá | Dữ liệu, môi trường, benchmark, biểu đồ, Git, báo cáo và demo | Bộ dữ liệu, CSV, biểu đồ, nhật ký thí nghiệm, báo cáo, slide và video |

Mọi phần code phải được ít nhất một thành viên khác review trước khi merge. Thành viên C có quyền từ chối số liệu nếu thiếu cấu hình máy, số lần chạy hoặc không thể tái lập.

---

# 3. Hướng dẫn chi tiết cho Thành viên A – Tính toán song song

## A.1. Mục tiêu

Xây dựng phần lõi xử lý ảnh chính xác trước, sau đó song song hóa và tối ưu. Thành viên A chịu trách nhiệm chứng minh rằng kết quả nhanh hơn nhưng không sai so với phiên bản tuần tự.

## A.2. Thứ tự thực hiện

### Bước 1 – Thiết kế kiểu dữ liệu ảnh dùng chung

- Tạo cấu trúc ảnh chứa `width`, `height`, `channels` và vùng nhớ pixel liên tục.
- Quy định ảnh grayscale dùng một kênh, ảnh RGB dùng ba kênh.
- Quy định rõ kiểu dữ liệu đầu vào/đầu ra, ưu tiên `uint8_t`; phép tính trung gian có thể dùng `float` hoặc `int`.
- Chọn một quy tắc xử lý biên và dùng thống nhất, ví dụ clamp hoặc replicate.
- Kiểm tra cả ảnh có chiều rộng/chiều cao không chia hết cho CUDA block.

**Đầu ra:** header kiểu dữ liệu ảnh, tài liệu quy tắc biên và test ảnh rất nhỏ như 1x1, 2x2, 7x5.

### Bước 2 – Viết phiên bản CPU tuần tự

Thực hiện theo thứ tự:

1. Chuyển RGB sang grayscale.
2. Gaussian Blur với kernel 3x3, sau đó hỗ trợ 5x5 và 7x7.
3. Sobel theo hai chiều `Gx`, `Gy`, tính độ lớn gradient và threshold.
4. Histogram Equalization gồm histogram 256 mức, CDF và ánh xạ lại pixel.

Yêu cầu:

- Hàm xử lý không đọc/ghi file trực tiếp; chỉ nhận và trả buffer ảnh.
- Không tối ưu sớm. Code tuần tự phải dễ đọc để làm bản tham chiếu.
- Mỗi thuật toán có test với kết quả biết trước hoặc so sánh với OpenCV.
- Ghi lại công thức và độ phức tạp tính toán trong `docs/ALGORITHMS.md`.

**Đầu ra:** code trong `src/cpu/`, header trong `include/`, test trong `tests/`.

### Bước 3 – Viết phiên bản OpenMP

- Song song hóa vòng lặp theo pixel hoặc theo hàng ảnh.
- Dùng `schedule(static)` làm cấu hình ban đầu cho Gaussian và Sobel.
- Thử `static`, `dynamic`, `guided` nhưng chỉ giữ cấu hình tốt nhất làm mặc định.
- Histogram không được để nhiều thread ghi không kiểm soát vào cùng một mảng.
- Với Histogram Equalization, tạo histogram riêng cho từng thread rồi hợp nhất.
- Kiểm tra với 1, 2, 4, 8 và số thread tối đa hợp lý của CPU.
- Chạy công cụ hoặc test lặp để phát hiện data race và kết quả không ổn định.

**Đầu ra:** code trong `src/openmp/`, API giống phiên bản tuần tự, bảng cấu hình thread đề xuất.

### Bước 4 – Viết CUDA cơ bản

- Mỗi CUDA thread xử lý một pixel đầu ra.
- Kiểm tra `x < width` và `y < height` trước khi đọc/ghi.
- Thêm macro hoặc hàm kiểm tra lỗi sau CUDA API và kernel launch.
- Dùng CUDA Event để đo kernel, không dùng đồng hồ CPU cho riêng kernel.
- Tách thời gian:
  - cấp phát bộ nhớ;
  - host-to-device;
  - kernel;
  - device-to-host;
  - tổng end-to-end.
- Thử block 8x8, 16x16, 32x8 và 32x16.

**Đầu ra:** CUDA kernel cơ bản trong `src/cuda/`, log lỗi rõ ràng và hàm timing trả về cấu trúc thống nhất.

### Bước 5 – Tối ưu CUDA

- Gaussian và Sobel: dùng shared memory có vùng halo.
- Đặt kernel cố định trong constant memory nếu phù hợp.
- Giảm số lần truy cập global memory và bảo đảm truy cập liền kề.
- Histogram: tạo histogram cục bộ cho từng block trong shared memory, sau đó hợp nhất bằng atomic operation.
- Không tuyên bố “đã tối ưu” nếu chưa có số liệu so sánh với CUDA cơ bản.
- Ghi rõ đánh đổi giữa shared memory, số thread và occupancy.

**Đầu ra:** CUDA tối ưu, bảng so sánh trước/sau và giải thích nguyên nhân cải thiện hoặc không cải thiện.

## A.3. Cách tự kiểm tra

Trước khi bàn giao một thuật toán:

- [ ] Chạy được với ảnh grayscale và/hoặc RGB đúng theo thiết kế.
- [ ] Chạy được với kích thước không chia hết cho block.
- [ ] Không crash với ảnh nhỏ hơn kernel.
- [ ] Kết quả OpenMP/CUDA được so sánh với CPU tuần tự.
- [ ] MAE/MSE nằm trong ngưỡng nhóm thống nhất.
- [ ] Không có data race hoặc lỗi CUDA bị bỏ qua.
- [ ] API và tham số đã được ghi chú để Thành viên B gọi được.
- [ ] Cung cấp lệnh build và test cho Thành viên C.

## A.4. Bàn giao cho Thành viên B và C

Mỗi hàm xử lý phải có tài liệu ngắn gồm:

- tên thuật toán;
- backend hỗ trợ;
- kiểu ảnh đầu vào;
- tham số và khoảng hợp lệ;
- kiểu ảnh đầu ra;
- cấu trúc timing trả về;
- mã lỗi có thể xảy ra;
- ví dụ gọi hàm.

---

# 4. Hướng dẫn chi tiết cho Thành viên B – AI và ứng dụng

## B.1. Mục tiêu

Xây dựng một giao diện đơn giản nhưng ổn định, cho phép người dùng tải ảnh, yêu cầu xử lý bằng tiếng Việt, chọn backend và xem kết quả. Thành viên B phải bảo đảm AI chỉ tạo lệnh hợp lệ mà chương trình có thể thực thi an toàn.

## B.2. Thứ tự thực hiện

### Bước 1 – Thiết kế JSON schema trước khi gọi AI

Schema chỉ cho phép ba thao tác cốt lõi. Ví dụ:

```json
{
  "operations": [
    {"name": "gaussian_blur", "kernel_size": 5, "sigma": 1.2},
    {"name": "sobel", "threshold": 100}
  ],
  "backend": "auto"
}
```

Quy định đề xuất:

- `name`: chỉ nhận `gaussian_blur`, `sobel`, `histogram_equalization`.
- `kernel_size`: chỉ nhận 3, 5 hoặc 7.
- `sigma`: đặt giới hạn hợp lý, ví dụ 0.1–10.0.
- `threshold`: số nguyên 0–255.
- `backend`: `sequential`, `openmp`, `cuda` hoặc `auto`.
- Giới hạn số thao tác trong một pipeline để tránh yêu cầu quá dài.

**Đầu ra:** schema, hàm validation và thông báo lỗi tiếng Việt dễ hiểu.

### Bước 2 – Xây bộ prompt

- System prompt phải mô tả đúng ba thao tác và yêu cầu chỉ trả JSON.
- Không cho AI tạo tên thuật toán hoặc tham số ngoài schema.
- Tạo ít nhất 20 câu kiểm thử, gồm:
  - yêu cầu hợp lệ;
  - yêu cầu mơ hồ;
  - yêu cầu chứa thuật toán chưa hỗ trợ;
  - tham số vượt giới hạn;
  - prompt injection đơn giản;
  - yêu cầu bằng tiếng Việt không dấu.
- Nếu AI trả sai JSON, hệ thống phải báo lỗi hoặc thử sửa theo quy tắc giới hạn; không thực thi trực tiếp.
- API key chỉ đọc từ biến môi trường hoặc Colab Secrets, không commit vào Git.

**Đầu ra:** prompt template, bộ test prompt và bảng tỷ lệ JSON hợp lệ.

### Bước 3 – Xây giao diện

Giao diện Streamlit hoặc Gradio tối thiểu gồm:

1. Tải ảnh.
2. Xem trước ảnh gốc.
3. Nhập yêu cầu tiếng Việt hoặc chọn bộ lọc thủ công.
4. Chọn backend hoặc Auto.
5. Hiển thị pipeline JSON sau validation.
6. Chạy xử lý.
7. Hiển thị ảnh kết quả.
8. Hiển thị backend thực tế, kernel time, transfer time, tổng thời gian và speedup.
9. Cho phép lưu ảnh kết quả.

Luôn có chế độ chọn bộ lọc thủ công để demo vẫn hoạt động khi mất mạng hoặc API AI gặp lỗi.

### Bước 4 – Tích hợp lõi xử lý ảnh

- Không sao chép lại thuật toán của Thành viên A bằng Python nếu không cần thiết.
- Tạo một lớp adapter gọi chung theo dạng `process(image, pipeline, backend)`.
- Chuyển đổi dữ liệu ảnh đúng layout mà C++/CUDA yêu cầu.
- Hiển thị lỗi thân thiện nếu không tìm thấy GPU hoặc CUDA thất bại.
- Backend Auto đọc quy tắc từ file cấu hình do benchmark tạo ra.
- Nếu GPU không khả dụng, Auto phải fallback sang OpenMP thay vì crash.

### Bước 5 – AI giải thích kết quả

- Chỉ đưa cho AI số liệu thực tế từ chương trình.
- Yêu cầu AI giải thích ngắn: backend nào nhanh nhất và nguyên nhân có thể có.
- Không để AI tự bổ sung thời gian, speedup hoặc cấu hình phần cứng.
- Luôn hiển thị bảng số liệu gốc bên cạnh phần giải thích.

## B.3. Cách tự kiểm tra

- [ ] Ảnh sai định dạng hoặc quá lớn được xử lý an toàn.
- [ ] JSON sai schema không được chuyển xuống backend.
- [ ] 20 prompt test có kết quả được ghi lại.
- [ ] Chế độ thủ công chạy được khi tắt AI.
- [ ] Cả bốn lựa chọn backend hiển thị đúng trạng thái.
- [ ] Auto fallback đúng khi không có CUDA.
- [ ] Không có API key trong source, notebook, log hoặc Git history.
- [ ] Giao diện demo được từ đầu đến cuối mà không sửa code.

## B.4. Bàn giao cho Thành viên C

- Hướng dẫn cài và chạy giao diện.
- Danh sách thao tác demo ổn định.
- Năm câu prompt mẫu và kết quả mong đợi.
- Ảnh chụp màn hình giao diện.
- Danh sách lỗi thường gặp và cách xử lý.

---

# 5. Hướng dẫn chi tiết cho Thành viên C – Hậu cần và đánh giá

## C.1. Mục tiêu

Bảo đảm dự án có dữ liệu hợp lệ, số liệu có thể tái lập, tiến độ rõ ràng và tài liệu đủ để người khác build, chạy, đánh giá. Vai trò này không chỉ làm slide mà chịu trách nhiệm về chất lượng bằng chứng của dự án.

## C.2. Thứ tự thực hiện

### Bước 1 – Quản lý môi trường và Git

- Ghi cấu hình máy cục bộ: CPU, số nhân/luồng, GPU, VRAM, RAM, hệ điều hành, compiler, CUDA và driver.
- Ghi cấu hình Google Colab ở từng phiên bằng `nvidia-smi`, `nvcc --version` và `lscpu`.
- Cập nhật hướng dẫn build Windows và Colab trong README.
- Theo dõi issue, milestone, branch và pull request.
- Không merge nếu code chưa được review hoặc test chưa qua.
- Nhắc nhóm không commit `build/`, dữ liệu lớn, `.env`, API key và file kết quả tạm.

### Bước 2 – Chuẩn bị dữ liệu

- Chọn ảnh có quyền sử dụng rõ ràng hoặc ảnh tự tạo.
- Chuẩn bị ít nhất ba loại nội dung: phong cảnh, nhiều cạnh/chi tiết và tương phản thấp.
- Tạo các kích thước 256x256, 512x512, 1920x1080, 2560x1440 và 3840x2160.
- Giữ một file metadata ghi nguồn, giấy phép, kích thước và checksum.
- Chỉ đưa ảnh mẫu nhỏ vào Git; ảnh lớn lưu trong Drive và ghi hướng dẫn tải.

### Bước 3 – Thiết kế benchmark

Quy trình bắt buộc cho mỗi cấu hình:

1. Dùng cùng ảnh và tham số giữa các backend.
2. Build ở chế độ Release.
3. Chạy khởi động 3–5 lần.
4. Chạy đo chính thức ít nhất 20 lần.
5. Lưu từng lần đo, không chỉ lưu giá trị trung bình.
6. Ghi trạng thái nguồn điện và hạn chế ứng dụng nền nếu có thể.
7. Với CUDA, lưu riêng H2D, kernel, D2H và end-to-end.
8. Không so sánh trực tiếp hai máy khác cấu hình mà không ghi chú.

CSV tối thiểu gồm:

```text
timestamp,environment,algorithm,image_width,image_height,backend,
threads,block_x,block_y,kernel_size,run,h2d_ms,kernel_ms,d2h_ms,total_ms,mae
```

### Bước 4 – Phân tích số liệu

Tính và vẽ:

- trung bình và độ lệch chuẩn;
- speedup so với CPU tuần tự;
- efficiency của OpenMP;
- throughput triệu pixel/giây;
- tỷ trọng thời gian truyền dữ liệu CUDA;
- ảnh hưởng của số thread;
- ảnh hưởng của block size;
- CUDA cơ bản so với CUDA tối ưu;
- MAE/MSE để chứng minh tính đúng đắn.

Mỗi biểu đồ phải có tên, đơn vị, chú giải, cấu hình máy và nhận xét 2–4 câu. Không chỉ đưa biểu đồ mà không giải thích.

### Bước 5 – Báo cáo và demo

Báo cáo phải trả lời bốn phần của giảng viên:

1. Bài toán ban đầu và lý do cần song song.
2. Phương pháp toán học, logic và code.
3. Phương pháp đánh giá, bảng số liệu và thông số tối ưu.
4. Hướng ứng dụng cụ thể.

Kịch bản demo 5–7 phút:

1. Giới thiệu bài toán và kiến trúc.
2. Nhập một yêu cầu xử lý ảnh bằng tiếng Việt.
3. Cho xem JSON đã validation.
4. Chạy CPU tuần tự, OpenMP và CUDA.
5. So sánh ảnh đầu ra và số liệu.
6. Trình bày một tối ưu CUDA có bằng chứng.
7. Nêu giới hạn và hướng phát triển.

Luôn quay một video demo dự phòng và chuẩn bị chế độ thủ công nếu API AI không hoạt động.

## C.3. Cách tự kiểm tra

- [ ] Người khác clone repository và làm theo README có thể chạy được.
- [ ] Mỗi số liệu truy ngược được về file CSV và cấu hình máy.
- [ ] Mỗi benchmark có đủ số lần chạy.
- [ ] Biểu đồ không trộn dữ liệu máy cục bộ với Colab một cách gây hiểu nhầm.
- [ ] Báo cáo phân biệt kernel time và end-to-end time.
- [ ] Nguồn ảnh và giấy phép được ghi lại.
- [ ] Slide đúng với số liệu trong báo cáo.
- [ ] Demo chính và video dự phòng đều chạy được.

---

# 6. Quy ước bàn giao giữa các thành viên

## A bàn giao cho B

- Header/API đã ổn định.
- Ví dụ gọi từng thuật toán.
- Danh sách tham số hợp lệ.
- Cấu trúc kết quả và timing.
- File build thành công và test đi kèm.

## A bàn giao cho C

- Lệnh benchmark.
- Các cấu hình thread/block cần thử.
- Ngưỡng MAE/MSE chấp nhận được.
- Giải thích kỹ thuật cho từng tối ưu.

## B bàn giao cho A

- JSON schema và danh sách tham số UI sử dụng.
- Adapter không làm thay đổi layout dữ liệu ngoài thỏa thuận.
- Báo lỗi tích hợp có log đủ để debug.

## B bàn giao cho C

- Hướng dẫn sử dụng giao diện.
- Prompt mẫu và pipeline mong đợi.
- Kịch bản chạy khi online và khi mất kết nối AI.

## C phản hồi cho A và B

- File CSV thô, không chỉ ảnh biểu đồ.
- Các trường hợp kết quả sai hoặc tốc độ bất thường.
- Issue có bước tái hiện, môi trường, đầu vào và log.

---

# 7. Kế hoạch theo mốc

## Mốc 0 – Nền tảng

- [x] Tạo cấu trúc dự án và Git repository.
- [x] Tạo dự án CMake có kiểm tra OpenMP và CUDA.
- [ ] Xác nhận build Release trên máy NVIDIA cục bộ. — A + C
- [ ] Ghi cấu hình môi trường máy và Colab. — C
- [ ] Thêm ảnh mẫu nhỏ có giấy phép phù hợp. — C
- [ ] Thống nhất kiểu dữ liệu ảnh và API lõi. — A + B

## Mốc 1 – CPU tuần tự

- [ ] Hoàn thành ba thuật toán tuần tự. — A
- [ ] Hoàn thành test tham chiếu và trường hợp biên. — A
- [ ] Hoàn thành JSON schema và validation độc lập với AI. — B
- [ ] Chuẩn bị dữ liệu theo năm độ phân giải. — C

**Điều kiện hoàn thành:** kết quả CPU tuần tự đã kiểm chứng; schema từ chối dữ liệu sai.

## Mốc 2 – OpenMP

- [ ] Hoàn thành ba phiên bản OpenMP. — A
- [ ] Kiểm tra data race và sai số. — A
- [ ] Tạo giao diện chọn ảnh, thuật toán và backend. — B
- [ ] Chạy benchmark thread/scheduling. — C, có A hỗ trợ

**Điều kiện hoàn thành:** OpenMP đúng và có số liệu speedup lặp lại được.

## Mốc 3 – CUDA

- [ ] Hoàn thành ba CUDA kernel cơ bản. — A
- [ ] Hoàn thành ít nhất một tối ưu shared memory. — A
- [ ] Tích hợp CUDA vào adapter và giao diện. — B
- [ ] Chạy benchmark block size và thời gian truyền dữ liệu. — C, có A hỗ trợ

**Điều kiện hoàn thành:** CUDA cơ bản/tối ưu đúng và mức cải thiện có số liệu.

## Mốc 4 – AI và Auto backend

- [ ] Hoàn thành prompt và bộ 20 prompt test. — B
- [ ] Kết nối pipeline AI với giao diện. — B
- [ ] Rút quy tắc Auto từ benchmark. — A + C
- [ ] Cài quy tắc Auto và fallback OpenMP. — B

**Điều kiện hoàn thành:** AI tạo JSON hợp lệ, chế độ thủ công và fallback luôn dùng được.

## Mốc 5 – Đánh giá cuối

- [ ] Chạy toàn bộ benchmark chính thức. — C
- [ ] Xác minh MAE/MSE. — A + C
- [ ] Vẽ biểu đồ và phân tích kết quả. — C, có A review
- [ ] Kiểm thử kịch bản giao diện hoàn chỉnh. — B + C

**Điều kiện hoàn thành:** mọi nhận định đều truy ngược được về dữ liệu đo.

## Mốc 6 – Nộp bài

- [ ] Build lại từ một bản clone sạch. — C
- [ ] Hoàn thiện báo cáo. — C chủ trì, A/B viết phần chuyên môn
- [ ] Chuẩn bị slide và kịch bản demo 5–7 phút. — C
- [ ] Quay video demo dự phòng. — B + C
- [ ] Review chéo toàn bộ tài liệu. — A + B + C
- [ ] Gắn tag `v1.0.0`. — C

---

# 8. Quy trình Git cho cả nhóm

Mỗi thành viên tạo branch riêng:

```text
feature/cpu-algorithms
feature/openmp
feature/cuda
feature/ui
feature/ai-pipeline
experiment/benchmarks
docs/report
```

Quy trình làm việc:

1. Cập nhật nhánh `main` trước khi bắt đầu.
2. Tạo branch đúng với nhiệm vụ.
3. Commit nhỏ, thông điệp mô tả rõ thay đổi.
4. Push branch lên GitHub.
5. Tạo pull request.
6. Ghi cách test trong pull request.
7. Một thành viên khác review.
8. Chỉ merge khi build và test qua.

Không commit trực tiếp lên `main`, trừ chỉnh sửa tài liệu rất nhỏ đã được cả nhóm thống nhất.

# 9. Định nghĩa hoàn thành chung

Một nhiệm vụ chỉ được đánh dấu `[x]` khi:

- code hoặc tài liệu đã được lưu đúng vị trí;
- có cách chạy hoặc cách kiểm tra rõ ràng;
- kết quả đúng với tiêu chí đã thống nhất;
- không chứa khóa bí mật hoặc file sinh ra không cần thiết;
- đã được ít nhất một thành viên khác review;
- pull request đã merge hoặc tài liệu đã được duyệt;
- người nhận bàn giao xác nhận có thể sử dụng đầu ra.

# 10. Phần tùy chọn

Chỉ thực hiện sau khi tất cả điều kiện cốt lõi đã đạt:

- [ ] Xử lý hàng loạt một thư mục ảnh.
- [ ] Demo một đoạn video offline ngắn.
- [ ] So sánh GPU cục bộ với một GPU Google Colab như hai môi trường riêng biệt.

Không mở rộng sang nhận dạng vật thể, đa GPU, MPI hoặc huấn luyện mô hình AI trong phiên bản chính.
