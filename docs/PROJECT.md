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
| CUDA Basic/Optimized | Đã triển khai thật; tự báo không khả dụng nếu runtime không có GPU |
| Manual UI, schema, prompt corpus | Hoàn thành trên `main` |
| Adapter UI → core C++ thật | Hoàn thành trên `main` |
| AI prompt parser Structured Outputs | Hoàn thành code trên `main` |
| Runner CSV/summary/biểu đồ cho C | Hoàn thành code trên `main` |
| Benchmark chính thức và báo cáo | C có thể bắt đầu từ `main` |

[PR #6 – UI/AI adapter và benchmark](https://github.com/Chicken20145/ai-assisted-parallel-image-processing/pull/6) đã merge vào `main` tại commit `816ce98`.

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
- CUDA chạy kernel thật và tách thời gian allocation/H2D/kernel/D2H/total; khi không có GPU trả `BackendUnavailable`.
- Benchmark chính thức từ chối fallback để không gắn nhãn sai số liệu.

## 5. Cấu trúc repository

```text
app/          Streamlit UI, AI parser, schema và adapter C++
include/      API C++ công khai
src/cpu/      Thuật toán CPU tuần tự
src/openmp/   Thuật toán OpenMP
src/cuda/     CUDA probe, CUDA Basic và CUDA Optimized
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

Gaussian giữ số kênh đầu vào. Sobel và Histogram Equalization trả grayscale một kênh.
Sequential/OpenMP phải khớp tuyệt đối. Benchmark lưu tổng sai lệch nguyên, tổng bình phương,
số giá trị đã so sánh và max error; MAE/MSE chính xác là phân số của các tổng này. Mức thay
đổi so với ảnh đầu vào được ghi riêng và không được gọi là ground-truth error.

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

Mở notebook chính thức trên nhánh `main`:

```text
https://colab.research.google.com/github/Chicken20145/ai-assisted-parallel-image-processing/blob/main/notebooks/colab_setup.ipynb
```

Nếu không mở được: **File → Open notebook → GitHub**, chọn repository, branch `main` và `notebooks/colab_setup.ipynb`.

Trình tự chạy lần đầu:

1. Chọn **Runtime → Change runtime type → T4 GPU**, rồi **Connect**.
2. Mở **Secrets** (biểu tượng chìa khóa), tạo `NGROK_AUTHTOKEN` từ [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken) và bật quyền notebook. Chỉ tạo thêm `OPENAI_API_KEY` khi cần chế độ AI.
3. Chạy từng cell từ trên xuống; không chuyển cell khi còn đang chạy hoặc có traceback đỏ.
4. Xác nhận cell in `ĐÃ ĐỒNG BỘ GITHUB` cùng branch/commit, setup hoàn thành và test báo `100% tests passed`.
5. Chạy cell **Mở Streamlit UI trên Colab**, rồi bấm **Mở Pixel Lab Streamlit UI**.

Giao diện hiển thị `Phiên bản Git đang chạy` ngay dưới tiêu đề. Mã này phải giống mã commit mà cell cập nhật vừa in; nếu khác, chạy lại cell cập nhật và cell mở Streamlit để khởi động lại server.

Không chạy cell cũ chứa `serve_kernel_port_as_iframe`, không mở `localhost:8501` và không ghi token trực tiếp vào notebook.

Cập nhật code trong cùng runtime:

```bash
%cd /content/ai-assisted-parallel-image-processing
!git pull --ff-only origin main
!bash scripts/build_colab.sh
!bash scripts/test_colab.sh
```

Không lưu notebook vào GitHub nếu chỉ chạy thử. Không mount Drive trong lúc benchmark; chỉ sao chép kết quả sang Drive sau khi đo xong.

### Mở UI trên Colab

Không chạy `streamlit run` rồi mở `localhost:8501` vì localhost nằm trong máy ảo Colab. Colab kernel proxy không phù hợp với WebSocket của Streamlit và có thể trả 404 hoặc trang trắng. Trong notebook, chạy mục **Mở Streamlit UI trên Colab**. Trước lần đầu, tạo secret `NGROK_AUTHTOKEN` trong mục Secrets (biểu tượng chìa khóa) và bật quyền truy cập cho notebook. Cell sẽ:

1. Kiểm tra `build-colab/image_pipeline_cli`.
2. Chạy Streamlit nền và chờ health endpoint.
3. Tạo liên kết HTTPS **Mở Pixel Lab Streamlit UI** bằng ngrok mà không in token ra output.
4. Dừng server/tunnel cũ nếu cell được chạy lại.

URL ngrok là URL công khai trong thời gian runtime còn hoạt động. Không chia sẻ URL và không tải dữ liệu nhạy cảm. Khi dùng xong, đóng tunnel bằng `ngrok.disconnect(ui_url)` hoặc ngắt runtime Colab.

Nếu không chạy được:

- `FileNotFoundError`: chạy lại setup/build.
- Trang trắng hoặc HTTP 404: notebook có thể đang là bản cũ; mở lại notebook từ `main` và kiểm tra cell có `ngrok.connect`.
- Lỗi xác thực ngrok: kiểm tra secret đúng tên `NGROK_AUTHTOKEN`, token còn hiệu lực và quyền notebook đã bật.
- Colab vừa kết nối lại runtime: chạy lại toàn bộ cell từ đầu.
- Link được tạo nhưng ứng dụng lỗi: xem log:

```python
print(open('/tmp/pixel_lab_streamlit.log', encoding='utf-8').read())
```

## 9. Chạy ứng dụng B

Build core trước:

```powershell
.\scripts\build_windows.ps1 -Configuration Release
.\.venv\Scripts\python.exe -m streamlit run .\app\app.py
```

UI có hai tab:

1. **Xử lý một ảnh**: tải ảnh riêng hoặc chọn trực tiếp một trong 300 ảnh BSDS300; chọn một bước, nhiều bước hoặc nhập yêu cầu AI; bấm **Chạy và xem chỉ số**.
2. **Benchmark**: chọn kiểm tra đủ 300 ảnh hoặc đo hiệu năng báo cáo; bấm một nút để chạy Sequential/OpenMP và hiện thời gian, speedup, throughput, MAE cùng file CSV tải về.

Ba cách tạo yêu cầu xử lý ảnh:

1. Một thuật toán.
2. Pipeline thủ công tối đa năm bước.
3. AI từ mô tả tiếng Việt.

- **Kiểm tra đủ 300 ảnh**: warm-up 1, đo 3 lần; dùng để kiểm tra độ đúng và độ phủ.
- **Đo hiệu năng để làm báo cáo**: 15 ảnh nhiều độ phân giải, warm-up 3, đo 20 lần và thử các mức luồng phù hợp với CPU; khớp quy trình tại [`C_GUIDE.md`](C_GUIDE.md).
- UI tự kiểm tra CUDA trước benchmark; chỉ thêm CUDA khi core trả đúng backend, không ghi fallback dưới nhãn CUDA.

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

- CUDA Basic và CUDA Optimized đã có trong core; tiếp tục đo để chọn block size tốt nhất theo GPU.
- Timing allocation/H2D/kernel/D2H/total đã tách bằng CUDA Event và đồng hồ host.
- Test CUDA so sánh kết quả với Sequential; tự bỏ qua có thông báo trên máy không có GPU.

### B – AI/UI

- Code chức năng đã merge vào `main` qua PR #6.
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
- PR #6 đã merge; C tạo branch báo cáo từ `main` mới nhất.

## 13. Tiêu chí hoàn thành dự án

- Ba thuật toán đúng trên ảnh thường và ảnh biên.
- Sequential/OpenMP/CUDA dùng cùng hợp đồng API và kiểu output.
- CUDA có Basic/Optimized và timing tách giai đoạn.
- Benchmark Release warm-up 3–5, ít nhất 20 lần/cấu hình.
- Có CSV thô, environment, mean/std, speedup, efficiency, throughput và error metrics.
- UI manual/AI chạy được, hiển thị backend thật và tải ảnh kết quả.
- Clone sạch có thể setup/build/test lại theo note này.
