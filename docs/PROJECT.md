# Dự án AI-Assisted Parallel Image Processing

Đây là note tổng hợp duy nhất về mục tiêu, kiến trúc, trạng thái, thiết lập và cách phối hợp dự án. Quy trình benchmark chi tiết của thành viên C nằm riêng tại [`C_GUIDE.md`](C_GUIDE.md).

## 1. Mục tiêu

Xây dựng ứng dụng xử lý ảnh có thể:

1. Nhận yêu cầu thủ công hoặc câu tiếng Việt.
2. Chuyển yêu cầu AI thành pipeline JSON đã khóa schema.
3. Chạy cùng thuật toán trên CPU tuần tự, OpenMP và CUDA.
4. Đo thời gian, kiểm tra độ chính xác và giải thích backend nào phù hợp.
5. Tái lập được trên Windows và Google Colab.

AI chỉ tạo pipeline. Toàn bộ xử lý pixel và timing chính thức phải đến từ core C++/CUDA; AI không được tự tính ảnh hoặc tạo số liệu benchmark.

## 2. Phạm vi cố định

Ba thuật toán:

- Gaussian Blur: `kernel_size` 3/5/7, `sigma` 0.1–10.
- Sobel Edge Detection: `threshold` 0–255.
- Histogram Equalization: không có tham số riêng.

Bốn backend:

- `sequential`: CPU tuần tự, kết quả tham chiếu.
- `openmp`: CPU đa luồng.
- `cuda_basic`: một CUDA thread/pixel.
- `cuda_optimized`: shared memory/halo hoặc histogram theo block.

Ngoài phạm vi: video thời gian thực, MPI, đa GPU, nhận diện khuôn mặt/vật thể, huấn luyện model, chỉnh màu kiểu mạng xã hội, chẩn đoán y tế và triển khai cloud nhiều người dùng.

## 3. Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| Setup Windows/Colab, dataset BSDS300, 15 ảnh benchmark | Hoàn thành trên `main` |
| API chung và CPU Sequential | Hoàn thành trên `main` |
| OpenMP cho ba thuật toán | Hoàn thành trên `main` |
| CUDA Basic/Optimized | Chưa triển khai; A làm tiếp |
| Manual UI, schema, prompt corpus | Hoàn thành trên Draft PR #6 |
| Adapter UI → core C++ thật | Hoàn thành trên Draft PR #6 |
| AI prompt parser Structured Outputs | Hoàn thành code trên Draft PR #6 |
| Runner CSV/summary/biểu đồ cho C | Hoàn thành code trên Draft PR #6 |
| Benchmark chính thức và báo cáo | C chạy sau khi PR #6 merge |

PR đang dùng: [PR #6 – Manual UI, AI schema và C++ adapter](https://github.com/Chicken20145/ai-assisted-parallel-image-processing/pull/6). PR vẫn để Draft cho đến khi một thành viên khác review.

## 4. Kiến trúc

```text
Người dùng
   ├─ Manual mode ───────────────┐
   └─ AI prompt                  │
        └─ OpenAI Responses API  │
             └─ Structured Output (Pydantic)
                                ↓
                     Pipeline validation lần hai
                                ↓
                    app/pipeline_adapter.py
                                ↓
                    app/core_adapter.py
                                ↓ PPM/PGM lossless
                    image_pipeline_cli (C++)
                                ↓
                        pip::process()
                  ┌─────────────┼─────────────┐
              Sequential      OpenMP        CUDA
```

Quy tắc quan trọng:

- Output AI luôn bị Pydantic kiểm tra lại trước khi chạy.
- Python không sao chép thuật toán C++ trong đường chạy thật.
- `image_pipeline_cli` đọc ảnh P5/P6, gọi duy nhất `pip::process()` và trả JSON.
- CUDA chưa có trả `BackendUnavailable`; UI fallback CUDA → OpenMP → Sequential và hiển thị backend thực tế.
- Benchmark chính thức từ chối fallback để không gắn nhãn sai số liệu.

## 5. Cấu trúc repository

```text
app/          Streamlit UI, AI parser, schema và adapter C++
include/      API C++ công khai
src/cpu/      Thuật toán CPU tuần tự
src/openmp/   Thuật toán OpenMP
src/cuda/     CUDA probe và backend CUDA tương lai
scripts/      Setup, build, dataset, benchmark và phân tích
tests/        CTest, pytest unit/integration
data/         Dataset cục bộ, không commit
benchmarks/   Kết quả benchmark cục bộ, không commit mặc định
notebooks/    Notebook setup Google Colab
docs/         PROJECT.md và C_GUIDE.md
```

## 6. Hợp đồng core

```cpp
pip::ProcessingResult result = pip::process(
    input,
    pip::Algorithm::GaussianBlur,
    params,
    pip::Backend::OpenMP);
```

Ảnh là buffer liên tục, RGB xen kẽ hoặc grayscale:

```cpp
struct Image {
    int width;
    int height;
    int channels; // 1 hoặc 3
    std::vector<std::uint8_t> pixels;
};
```

`ProcessingResult` cung cấp:

- `output`: ảnh kết quả.
- `backend_used`: backend chạy thật.
- `threads_used`: số luồng OpenMP thực tế được cấu hình.
- `timing`: allocation, H2D, kernel, D2H và total.
- `error`: `None`, `InvalidImage`, `InvalidParameters`, `BackendUnavailable`, `InternalError`.

Gaussian giữ số kênh đầu vào. Sobel và Histogram Equalization trả grayscale một kênh. Sequential/OpenMP hiện phải khớp tuyệt đối: MAE/MSE/max error đều bằng 0.

## 7. Thiết lập Windows

Yêu cầu: Windows 10/11, Visual Studio C++ workload, CMake tools, Windows SDK, CUDA Toolkit/driver, Python 3.10+, Git.

```powershell
git clone https://github.com/Chicken20145/ai-assisted-parallel-image-processing.git
Set-Location .\ai-assisted-parallel-image-processing
git switch main
git pull --ff-only origin main

.\scripts\setup_windows.ps1
.\scripts\download_datasets.ps1
.\.venv\Scripts\python.exe .\scripts\prepare_benchmark_data.py
.\scripts\check_environment.ps1
.\scripts\test_windows.ps1 -BuildFirst
```

Nếu PowerShell chặn script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Build/test riêng:

```powershell
.\scripts\build_windows.ps1 -Configuration Release
.\scripts\test_windows.ps1
.\.venv\Scripts\python.exe -m pytest tests -q
```

## 8. Thiết lập Google Colab

Repository private nên tài khoản phải là collaborator. Mở:

```text
https://colab.research.google.com/github/Chicken20145/ai-assisted-parallel-image-processing/blob/main/notebooks/colab_setup.ipynb
```

Nếu 404: **File → Open notebook → GitHub**, bật kho private, authorize GitHub, chọn branch và notebook. Sau đó chọn **Runtime → Change runtime type → GPU** rồi chạy từ trên xuống.

Cập nhật code trong cùng runtime:

```bash
%cd /content/ai-assisted-parallel-image-processing
!git pull --ff-only origin main
!bash scripts/build_colab.sh
!bash scripts/test_colab.sh
```

Không lưu notebook vào GitHub nếu chỉ chạy thử. Không mount Drive trong lúc benchmark; chỉ sao chép kết quả sang Drive sau khi đo xong.

## 9. Chạy ứng dụng B

Build core trước:

```powershell
.\scripts\build_windows.ps1 -Configuration Release
.\.venv\Scripts\python.exe -m streamlit run .\app\app.py
```

UI có ba chế độ:

1. Một thuật toán.
2. Pipeline thủ công tối đa năm bước.
3. AI từ mô tả tiếng Việt.

Thiết lập OpenAI trên Windows chỉ trong phiên terminal:

```powershell
$env:OPENAI_API_KEY = 'key-cua-ban'
$env:OPENAI_MODEL = 'gpt-5.6-luna' # tùy chọn
```

Google Colab: tạo Secret `OPENAI_API_KEY`; code tự đọc Colab Secrets. Không commit `.env`, API key hoặc in key ra output. Khi không có key/API lỗi, manual mode vẫn hoạt động.

AI parser dùng Responses API + Pydantic Structured Outputs, sau đó validate lần hai bằng schema nội bộ. Tài liệu kỹ thuật tham khảo: [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

Sau khi có API key, B chạy bộ đánh giá 26 prompt (lệnh này gọi API và có thể phát sinh chi phí):

```powershell
.\.venv\Scripts\python.exe .\scripts\evaluate_ai_prompts.py `
  --model gpt-5.6-luna `
  --output .\benchmarks\results\ai_prompt_evaluation.csv
```

Có thể dùng `--limit 1` để kiểm tra key/model trước khi chạy đủ bộ.

## 10. Dataset

Nguồn chính: BSDS300 của UC Berkeley, 300 JPEG dùng cho nghiên cứu/giáo dục phi thương mại.

- Archive SHA-256 đã biết: `A5F7D0E49FE135C75518A3543CED24470156FD69305AE77845DFF2A5138652B4`.
- Logical checksum 300 ảnh: `44584B06A9D2F22028D345F087F99D2428A5B6C410E8BEDE069770A60A8A24EF`.
- Bộ benchmark: ba nhóm nội dung × năm độ phân giải = 15 ảnh.
- Metadata: `data/external/benchmark_suite/metadata.csv`.

Dataset, archive, build, API key và kết quả tạm không được commit.

## 11. Phân công còn lại

### A – Core

- Triển khai CUDA Basic và CUDA Optimized.
- Tách allocation/H2D/kernel/D2H/total bằng CUDA Event.
- So sánh CUDA với Sequential bằng error metrics.

### B – AI/UI

- Code chức năng đã hoàn thiện trên PR #6.
- Việc thủ công còn lại: chạy prompt eval bằng key cá nhân, chụp ảnh demo và nhờ thành viên khác review PR.
- Không gửi API key cho người khác và không lưu output chứa thông tin nhạy cảm.

### C – Benchmark/report

- Code runner, summary và plot đã viết sẵn.
- Thực hiện theo [`C_GUIDE.md`](C_GUIDE.md).
- Không dùng timing một lần chạy trên UI làm số liệu báo cáo.

## 12. Git và tiêu chí merge

```powershell
git switch main
git pull --ff-only origin main
git switch -c feature/ten-nhiem-vu
# sửa, test
git add -- <dung-cac-file-cua-minh>
git commit -m "mo ta ngan gon"
git push -u origin feature/ten-nhiem-vu
```

- Không push trực tiếp `main`.
- Mỗi PR cần ít nhất một người khác review.
- Không merge khi test đỏ, có conversation chưa resolve hoặc lẫn dataset/build/secret.
- PR #6 phải merge trước khi C tạo branch báo cáo từ `main`.

## 13. Tiêu chí hoàn thành dự án

- Ba thuật toán đúng trên ảnh thường và ảnh biên.
- Sequential/OpenMP/CUDA dùng cùng hợp đồng API và kiểu output.
- CUDA có Basic/Optimized và timing tách giai đoạn.
- Benchmark Release warm-up 3–5, ít nhất 20 lần/cấu hình.
- Có CSV thô, environment, mean/std, speedup, efficiency, throughput và error metrics.
- UI manual/AI chạy được, hiển thị backend thật và tải ảnh kết quả.
- Clone sạch có thể setup/build/test lại theo note này.
