# AI-Assisted Parallel Image Processing

Hệ thống xử lý ảnh song song sử dụng C++17, OpenMP và CUDA, kèm giao diện Streamlit và lớp AI chuyển yêu cầu tiếng Việt thành pipeline có cấu trúc. Dự án tập trung vào tính đúng đắn, khả năng tái lập benchmark và so sánh công bằng giữa CPU tuần tự, CPU đa luồng và GPU.

## Tính năng chính

- Ba thuật toán: Gaussian Blur, Sobel Edge Detection và Histogram Equalization.
- Bốn backend: CPU Sequential, OpenMP, CUDA Basic và CUDA Optimized.
- API C++ thống nhất `pip::process()` cho mọi thuật toán và backend.
- Giao diện Streamlit xử lý một ảnh, pipeline nhiều bước và benchmark bằng một nút.
- AI parser dùng Structured Outputs; kết quả luôn được Pydantic kiểm tra trước khi thực thi.
- Dataset BSDS300 gồm 300 ảnh và bộ benchmark 15 ảnh ở năm độ phân giải.
- Timing CUDA tách riêng allocation, H2D, kernel, D2H và total.
- Đánh giá speedup, efficiency, throughput và độ chính xác pixel bằng tổng sai lệch nguyên,
  phân số MAE/MSE chính xác và sai lệch pixel lớn nhất.

## Kiến trúc

```text
Người dùng
   │
   ▼
Streamlit UI ── AI parser ── Pydantic validation
   │
   ▼
Python adapter
   │
   ▼
image_pipeline_cli
   │
   ▼
pip::process()
   ├── CPU Sequential
   ├── OpenMP
   ├── CUDA Basic
   └── CUDA Optimized
   │
   ▼
Ảnh kết quả + timing + CSV benchmark
```

AI chỉ tạo cấu hình pipeline. Toàn bộ xử lý pixel và số liệu timing chính thức được thực hiện bởi core C++/CUDA.

## Bắt đầu nhanh trên Google Colab

Đây là cách được khuyến nghị vì Colab cung cấp sẵn GPU NVIDIA:

[![Mở bằng Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chicken20145/ai-assisted-parallel-image-processing/blob/main/notebooks/colab_setup.ipynb)

1. Bấm nút **Mở bằng Google Colab** ở trên.
2. Chọn **Runtime → Change runtime type → T4 GPU**.
3. Tạo Colab Secret `NGROK_AUTHTOKEN` rồi chạy ba cell được đánh số 1, 2, 3.
4. Khi cell số 2 hiện `CÀI ĐẶT HOÀN TẤT`, chạy cell số 3.
5. Bấm nút **MỞ GIAO DIỆN PIXEL LAB** để bắt đầu demo.

Notebook tự clone `main`, tải và xác minh đủ 300 ảnh, build Release, chạy test và chuẩn bị giao diện. Chế độ xử lý thủ công không cần khóa OpenAI. Các mục cập nhật GitHub và lưu Google Drive là tùy chọn, không cần dùng trong demo thông thường.

## Thiết lập trên Windows

### Yêu cầu

- Windows 10/11.
- Visual Studio với workload **Desktop development with C++** và Windows SDK.
- NVIDIA CUDA Toolkit và driver tương thích.
- Python 3.10 trở lên.
- Git.

### Thiết lập và build

```powershell
git clone https://github.com/Chicken20145/ai-assisted-parallel-image-processing.git
cd ai-assisted-parallel-image-processing
.\scripts\setup_windows.ps1
.\scripts\download_datasets.ps1
.\.venv\Scripts\python.exe .\scripts\prepare_benchmark_data.py
.\scripts\build_windows.ps1 -Configuration Release
.\scripts\test_windows.ps1
```

Nếu đã có đầy đủ dependency, có thể build trực tiếp:

```powershell
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## Chạy giao diện

Sau khi build core và kích hoạt môi trường Python:

```powershell
$env:PIP_CORE_CLI = "$PWD\build\Release\image_pipeline_cli.exe"
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Giao diện hỗ trợ:

- Tải ảnh hoặc chọn ảnh từ BSDS300.
- Chọn thuật toán, backend và tham số.
- Ghép pipeline tối đa năm bước.
- Tạo pipeline từ yêu cầu tiếng Việt khi có `OPENAI_API_KEY`.
- Chạy kiểm tra đủ 300 ảnh hoặc benchmark hiệu năng chính thức.

## Benchmark

### Kiểm tra độ phủ

Chế độ **Kiểm tra đủ 300 ảnh** chạy ba thuật toán trên toàn bộ BSDS300, đo ba lần mỗi cấu hình. Mục tiêu là xác nhận mọi backend chạy ổn định và cho kết quả đúng.

### Đo hiệu năng chính thức

Chế độ **Đo hiệu năng để làm báo cáo** sử dụng 15 ảnh ở năm độ phân giải, warm-up ba lần và đo 20 lần. Runner từ chối fallback để không ghi số liệu dưới sai backend.

Chạy bằng dòng lệnh:

```bash
python scripts/run_benchmarks.py \
  --input-dir data/external/benchmark_suite \
  --output benchmarks/results/raw_results.csv \
  --backends sequential openmp cuda_basic cuda_optimized \
  --threads 1,2,4 \
  --warmup 3 \
  --runs 20

python scripts/analyze_benchmarks.py \
  --input benchmarks/results/raw_results.csv \
  --summary benchmarks/results/summary.csv \
  --plots benchmarks/results/plots
```

Không gộp kết quả từ các runtime có CPU/GPU khác nhau. Luôn lưu kèm file thông tin môi trường.

## Kiểm thử

CTest bao gồm:

- Kiểm tra thuật toán CPU tuần tự.
- Đối chiếu OpenMP với Sequential.
- Đối chiếu CUDA Basic/Optimized với Sequential; tự bỏ qua có thông báo nếu không có GPU.
- Kiểm tra tải và xác minh dataset.

Các backend song song được so sánh với CPU tuần tự bằng tổng sai lệch, tổng bình phương sai
lệch, đúng số giá trị pixel-kênh đã so sánh và sai lệch tuyệt đối lớn nhất. MAE/MSE được lưu
thêm dưới dạng phân số `tổng/số phần tử`, nên số 0 chỉ xuất hiện khi mọi pixel thực sự khớp.
Benchmark cũng ghi mức thay đổi so với ảnh đầu vào; đây không phải ground truth chất lượng.

Gaussian CPU/OpenMP/CUDA Optimized dùng phép lọc tách hai chiều, giảm chi phí từ `K²` xuống
`2K`. OpenMP giữ một parallel region qua nhiều pha; CUDA Optimized dùng bộ nhớ chia sẻ cho
Sobel/Histogram và tính CDF trực tiếp trên GPU.

## Cấu trúc repository

```text
app/          Streamlit UI, AI parser, schema và Python adapter
benchmarks/   Cấu hình và kết quả benchmark sinh ra
data/         Ảnh mẫu; dataset tải về không commit
docs/         Kiến trúc, thiết lập và hướng dẫn benchmark
include/      Public headers của core C++/CUDA
notebooks/    Notebook Google Colab
scripts/      Setup, build, download, benchmark và phân tích
src/common/   API chung và error metrics
src/cpu/      CPU Sequential
src/openmp/   OpenMP
src/cuda/     CUDA Basic và CUDA Optimized
tests/        CTest và Python tests
```

## Phạm vi

Dự án tập trung vào xử lý ảnh tĩnh trên một CPU và một GPU. Video thời gian thực, MPI, đa GPU, huấn luyện mô hình và triển khai production cloud không thuộc phạm vi hiện tại.

## Tài liệu

- [`docs/PROJECT.md`](docs/PROJECT.md): mục tiêu, kiến trúc, API, thiết lập và trạng thái dự án.
- [`docs/C_GUIDE.md`](docs/C_GUIDE.md): quy trình benchmark, phân tích CSV và checklist báo cáo.

## Bảo mật và dữ liệu

- Không commit API key, ngrok token, dataset archive hoặc kết quả benchmark dung lượng lớn.
- Không tải ảnh nhạy cảm lên URL ngrok công khai.
- Chỉ sử dụng nguồn dữ liệu có giấy phép và lưu metadata/checksum để tái lập thí nghiệm.
