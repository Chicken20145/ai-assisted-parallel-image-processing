# Hướng dẫn chi tiết cho thành viên C – Dataset, Benchmark và Báo cáo

Đây là note duy nhất C cần theo để chạy benchmark. Code đã được chuẩn bị sẵn; C không cần tự viết lại thuật toán hay tự ghép CSV bằng tay.

## 1. Kết quả C phải bàn giao

1. File environment của đúng phiên/máy đo.
2. CSV thô lưu từng lần chạy.
3. CSV summary có mean, standard deviation, speedup, efficiency và throughput.
4. Biểu đồ OpenMP theo thread count; sau này bổ sung CUDA theo block size.
5. Xác nhận MAE/MSE/max error so với Sequential.
6. Nhận xét kỹ thuật, cấu hình máy và cách tái lập.

Không chấp nhận số liệu chỉ chụp từ UI, thiếu CSV thô, thiếu environment, trộn hai máy/phiên Colab hoặc backend đã fallback.

## 2. Code đã viết sẵn cho C

- `image_pipeline_cli`: đọc ảnh thật, warm-up và chạy nhiều lần trong cùng một process.
- `scripts/run_benchmarks.py`: duyệt dataset, gọi core, kiểm tra backend, tính error metrics và ghi CSV.
- `scripts/analyze_benchmarks.py`: tổng hợp mean/std/speedup/efficiency/throughput và vẽ biểu đồ.
- `tests/test_benchmark_workflow.py`: test ma trận cấu hình, error metrics và công thức summary.

Runner cố ý từ chối nếu backend chạy thật khác backend yêu cầu. Vì vậy CUDA chưa triển khai sẽ báo lỗi thay vì ghi nhầm số liệu fallback thành CUDA.

## 3. Khi nào C bắt đầu

C có thể smoke test ngay trên branch `feature/ui-manual-mode`. Để làm kết quả chính thức:

1. Chờ PR #6 được review và merge.
2. Tạo branch riêng từ `main` mới nhất.

```powershell
git switch main
git pull --ff-only origin main
git switch -c feature/benchmark-report
```

C không sửa trực tiếp branch của B và không push CSV lớn/dataset lên Git.

## 4. Chuẩn bị Windows

```powershell
.\scripts\setup_windows.ps1
.\scripts\download_datasets.ps1
.\.venv\Scripts\python.exe .\scripts\prepare_benchmark_data.py
.\scripts\check_environment.ps1
.\scripts\build_windows.ps1 -Configuration Release
.\scripts\test_windows.ps1
.\.venv\Scripts\python.exe -m pytest tests -q
```

Phải thấy CTest và pytest đạt trước khi đo. Kiểm tra đủ 15 ảnh:

```powershell
(Get-ChildItem .\data\external\benchmark_suite\* -File -Include *.png,*.jpg).Count
Get-Content .\data\external\benchmark_suite\metadata.csv -TotalCount 5
```

Giữ `data/external/environment.txt` cùng bộ kết quả. Không đổi driver, power plan hoặc chạy ứng dụng nặng trong lúc đo.

## 5. Smoke test bắt buộc

Chạy một ảnh, một thuật toán, 1–2 luồng và hai lần đo:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_benchmarks.py `
  --input-dir .\data\external\benchmark_suite `
  --output .\benchmarks\results\smoke_raw.csv `
  --algorithms sobel `
  --backends sequential openmp `
  --threads 1,2 `
  --warmup 1 `
  --runs 2 `
  --max-images 1

.\.venv\Scripts\python.exe .\scripts\analyze_benchmarks.py `
  --input .\benchmarks\results\smoke_raw.csv `
  --summary .\benchmarks\results\smoke_summary.csv `
  --plots .\benchmarks\results\smoke_plots
```

Kết quả đúng:

- 6 dòng raw: 1 ảnh × 1 thuật toán × 3 cấu hình × 2 runs.
- Backend gồm Sequential, OpenMP 1 thread, OpenMP 2 thread.
- `mae`, `mse`, `max_abs_error` đều bằng 0.
- Có `smoke_summary.csv` và ảnh `speedup_sobel.png`.

## 6. Benchmark CPU/OpenMP chính thức

Mặc định runner dùng ba thuật toán, Sequential, OpenMP 1/2/4/8 thread, warm-up 3 và 20 lần đo:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_benchmarks.py `
  --input-dir .\data\external\benchmark_suite `
  --output .\benchmarks\results\windows_cpu_openmp_raw.csv `
  --algorithms gaussian_blur sobel histogram_equalization `
  --backends sequential openmp `
  --threads 1,2,4,8 `
  --warmup 3 `
  --runs 20 `
  --environment windows_may_c
```

Với 15 ảnh, ba thuật toán và năm cấu hình, số dòng mong đợi:

```text
15 × 3 × (1 Sequential + 4 OpenMP) × 20 = 4.500 dòng
```

Nếu máy có ít hơn 8 logical CPU, dùng danh sách hợp lý nhưng phải ghi rõ. Có thể thêm số luồng tối đa của máy để khảo sát, không hard-code kết luận từ một máy.

## 7. Tổng hợp và biểu đồ

```powershell
.\.venv\Scripts\python.exe .\scripts\analyze_benchmarks.py `
  --input .\benchmarks\results\windows_cpu_openmp_raw.csv `
  --summary .\benchmarks\results\windows_cpu_openmp_summary.csv `
  --plots .\benchmarks\results\windows_cpu_openmp_plots
```

Summary có:

- `kernel_ms_mean`, `kernel_ms_std`.
- `total_ms_mean`, `total_ms_std`.
- `throughput_mpix_s_mean`.
- `sequential_kernel_ms`.
- `speedup = T_sequential / T_backend`.
- `efficiency = speedup / threads` cho OpenMP.
- `mae_max`.

Mỗi biểu đồ phải ghi thuật toán, đơn vị, môi trường và số luồng. Speedup dưới 1 ở ảnh nhỏ không phải lỗi; overhead song song có thể lớn hơn lợi ích.

## 8. Kiểm tra chất lượng CSV

```powershell
$csv = Import-Csv .\benchmarks\results\windows_cpu_openmp_raw.csv
$csv.Count
$csv | Group-Object algorithm,backend,threads | Select-Object Name,Count
$csv | Measure-Object mae -Maximum
$csv | Where-Object { $_.requested_backend -ne $_.backend }
```

Yêu cầu:

- Đúng số dòng dự kiến.
- Mỗi cấu hình có đúng 20 runs.
- `requested_backend == backend` cho mọi dòng.
- Sequential/OpenMP: MAE/MSE/max error bằng 0.
- Không có timing âm/rỗng.
- Standard deviation không bất thường; nếu có outlier phải chạy lại trong môi trường sạch và ghi chú.

## 9. Google Colab

Chọn GPU runtime, chạy notebook setup, sau đó:

```bash
%cd /content/ai-assisted-parallel-image-processing
!git switch main
!git pull --ff-only origin main
!bash scripts/build_colab.sh
!bash scripts/test_colab.sh

!python scripts/run_benchmarks.py \
  --input-dir data/external/benchmark_suite \
  --output benchmarks/results/colab_cpu_openmp_raw.csv \
  --algorithms gaussian_blur sobel histogram_equalization \
  --backends sequential openmp \
  --threads 1,2 \
  --warmup 3 --runs 20 \
  --environment colab_t4_session_YYYYMMDD

!python scripts/analyze_benchmarks.py \
  --input benchmarks/results/colab_cpu_openmp_raw.csv \
  --summary benchmarks/results/colab_cpu_openmp_summary.csv \
  --plots benchmarks/results/colab_cpu_openmp_plots
```

Colab T4 thường chỉ cấp ít CPU thread; không so sánh trực tiếp với Windows như cùng một môi trường. Chỉ mount Drive sau khi benchmark:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Sao chép cả raw CSV, summary, plots và `environment_colab.txt` sang cùng một thư mục Drive.

## 10. CUDA sau khi A bàn giao

Không thêm `cuda_basic` hoặc `cuda_optimized` vào benchmark chính thức khi core còn trả `BackendUnavailable`.

Sau khi A merge CUDA và test chính xác đạt:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_benchmarks.py `
  --input-dir .\data\external\benchmark_suite `
  --output .\benchmarks\results\windows_cuda_raw.csv `
  --backends sequential cuda_basic cuda_optimized `
  --warmup 5 --runs 20 `
  --environment windows_gpu_c
```

Khi CUDA hoàn chỉnh, C bổ sung block size 8×8, 16×16, 32×8 hoặc cấu hình A bàn giao; CSV phải có H2D/kernel/D2H/total và error metrics. Runner hiện để sẵn `block_x`, `block_y`; A/C mở rộng CLI khi backend CUDA nhận block size.

## 11. Cách phân tích báo cáo

Với từng thuật toán và độ phân giải, trả lời:

1. OpenMP bắt đầu nhanh hơn Sequential từ kích thước nào?
2. Thread count nào tốt nhất trên từng máy?
3. Speedup có gần tuyến tính không; efficiency giảm ở đâu?
4. Gaussian, Sobel hay Histogram hưởng lợi nhiều nhất và vì sao?
5. Standard deviation có ổn định không?
6. CUDA kernel nhanh nhưng total có bị H2D/D2H chi phối không?
7. Mọi backend có giữ sai số trong ngưỡng không?

Không kết luận “backend X luôn nhanh nhất” từ một ảnh hoặc một lần chạy.

## 12. Checklist trước khi bàn giao

- [ ] PR #6 đã merge và C tạo branch từ `main` mới nhất.
- [ ] Build Release, CTest và pytest đạt.
- [ ] Đủ dataset/metadata/checksum.
- [ ] Có environment đúng phiên.
- [ ] Smoke test đạt trước full benchmark.
- [ ] Mỗi cấu hình warm-up 3–5 và chạy ít nhất 20 lần.
- [ ] Raw CSV đúng số dòng, không fallback, không timing lỗi.
- [ ] Error metrics đạt.
- [ ] Summary và plots tạo lại được bằng một lệnh.
- [ ] Không commit dataset, build, API key hoặc file tạm.
- [ ] Báo cáo ghi rõ máy, driver, compiler, CUDA, thread count và đơn vị.
