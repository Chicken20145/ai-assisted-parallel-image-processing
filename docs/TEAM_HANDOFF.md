# Ghi chú bàn giao và phối hợp A/B/C

Tài liệu này là điểm bắt đầu chung cho cả nhóm. Đọc phần “Trạng thái hiện tại” trước khi lấy code hoặc chạy benchmark.

## Trạng thái hiện tại

- Nhánh đang phát triển: `feature/core-api-cpu`.
- PR setup Windows/Colab: [PR #1](https://github.com/Chicken20145/ai-assisted-parallel-image-processing/pull/1).
- PR core API và CPU tuần tự: [PR #2](https://github.com/Chicken20145/ai-assisted-parallel-image-processing/pull/2).
- Đã hoạt động: API công khai, grayscale, Gaussian Blur, Sobel, Histogram Equalization CPU tuần tự, error metrics, CTest và benchmark tổng hợp.
- Chưa hoạt động: thuật toán OpenMP, CUDA Basic, CUDA Optimized, đọc ảnh thật trong benchmark CLI và adapter giao diện.
- Các backend chưa triển khai trả `BackendUnavailable`; đây là hành vi có chủ ý, không phải lỗi setup.

PR #2 đang xếp chồng lên PR #1. Thứ tự merge: PR #1 trước, đổi base PR #2 sang `main`, sau đó mới merge PR #2.

## Truy cập Google Colab

Repository là private. Mỗi thành viên phải được thêm làm collaborator và cấp quyền GitHub cho Colab.

Notebook theo nhánh hiện tại:

```text
https://colab.research.google.com/github/Chicken20145/ai-assisted-parallel-image-processing/blob/feature%2Fcore-api-cpu/notebooks/colab_setup.ipynb
```

Nếu Colab báo 404 hoặc không tìm thấy notebook:

1. Chọn **File → Open notebook → GitHub**.
2. Bật **Bao gồm các kho lưu trữ riêng tư**.
3. Authorize GitHub bằng tài khoản có quyền repository.
4. Chọn repository, branch `feature/core-api-cpu` và file `notebooks/colab_setup.ipynb`.

Không nhấn **Lưu trong GitHub** nếu chỉ chạy notebook. Thao tác đó tạo commit trực tiếp từ Colab và có thể gây xung đột.

## Cập nhật code trong cùng runtime Colab

Sau lần setup đầu tiên, khi A push code mới:

```bash
%cd /content/ai-assisted-parallel-image-processing
!git pull --ff-only origin feature/core-api-cpu
!bash scripts/build_colab.sh
!bash scripts/test_colab.sh
```

Không cần tải lại dataset hoặc cài lại package trong cùng runtime. Khi runtime bị reset, chạy lại notebook từ đầu.

## Thành viên A – lõi xử lý

A tiếp tục phát triển trên API trong `include/processing_api.hpp`:

1. Giữ CPU tuần tự làm kết quả tham chiếu.
2. Thêm OpenMP nhưng không đổi layout ảnh, enum, kiểu đầu ra hoặc quy tắc biên.
3. So sánh OpenMP với Sequential bằng `compare_images()`.
4. Sau OpenMP mới triển khai CUDA Basic và CUDA Optimized.
5. Mỗi lần push phải chạy:

```powershell
.\scripts\test_windows.ps1 -BuildFirst
```

Và kiểm tra Colab bằng:

```bash
bash scripts/build_colab.sh
bash scripts/test_colab.sh
```

## Thành viên B – AI và ứng dụng

B có thể bắt đầu adapter dựa trên:

- `include/image_types.hpp`;
- `include/processing_api.hpp`;
- `docs/API.md`.

Quy tắc tích hợp:

- Chỉ gọi `pip::process()`, không include file trong `src/cpu/` hoặc `src/cuda/`.
- Kiểm tra `result.ok()` trước khi đọc ảnh đầu ra.
- Hiển thị `error_message` thân thiện và ghi `to_string(result.error)` vào log.
- Không viết lại ba thuật toán bằng Python để thay thế lõi C++.
- UI có thể hiển thị lựa chọn backend ngay, nhưng phải báo “chưa khả dụng” khi nhận `BackendUnavailable`.
- Schema JSON dùng đúng mapping trong `docs/API.md`.

B chưa nên phụ thuộc vào benchmark CLI để đọc ảnh thật vì CLI hiện tạo ảnh tổng hợp. Adapter ảnh thật sẽ được bổ sung ở mốc tích hợp.

## Thành viên C – dữ liệu và benchmark

C có thể thực hiện ngay:

1. Chạy setup Windows hoặc Colab.
2. Xác minh đủ 15 ảnh trong `data/external/benchmark_suite/`.
3. Lưu `environment.txt` hoặc `environment_colab.txt` cho mỗi phiên.
4. Dùng `image_benchmark` để kiểm tra CSV và quy trình warm-up/runs.
5. Chuẩn bị cấu trúc lưu CSV, biểu đồ và metadata môi trường.

Ví dụ smoke benchmark:

```bash
./build-colab/image_benchmark \
  --algorithm gaussian_blur \
  --backend sequential \
  --width 1920 --height 1080 --channels 3 \
  --kernel-size 5 --sigma 1.2 \
  --warmup 3 --runs 20 > gaussian_sequential.csv
```

Đây là benchmark ảnh tổng hợp để kiểm tra pipeline, chưa phải số liệu chính thức trên ảnh dataset. Không dùng nó để kết luận chất lượng hoặc hiệu năng cuối cùng.

## Kết quả Colab đã xác nhận ngày 27/07/2026

Một phiên Colab GPU đã chạy thành công với:

- GPU runtime: NVIDIA T4.
- CUDA compiler: 12.8.93.
- Host compiler: GNU 11.4.0.
- OpenMP threads available: 2.
- CUDA devices available: 1.
- CTest: 100% test đạt.
- CUDA/OpenMP probe và benchmark smoke test: đạt.

Đây chỉ là bằng chứng setup hoạt động. Colab có thể cấp phần cứng khác ở phiên sau; luôn dùng file môi trường sinh tại thời điểm benchmark thay vì sao chép cấu hình trên.

## Quy tắc Git chung

- Không commit trực tiếp từ Colab nếu chỉ chạy thử nghiệm.
- Không commit dataset, `build/`, `build-colab/`, `.venv`, CSV tạm hoặc API key.
- Không push lên branch của người khác khi chưa thống nhất.
- Mỗi PR phải ghi cách build/test và được một thành viên khác review.
- Khi phát hiện lỗi, ghi branch, commit SHA, môi trường, lệnh chạy và output liên quan.
