# Hướng dẫn thiết lập dự án trên Google Colab

Colab phù hợp để build, chạy và benchmark CUDA khi thành viên không dùng cùng một máy NVIDIA. Runtime Colab là tạm thời: GPU, CPU, CUDA và thời lượng phiên có thể thay đổi, vì vậy mỗi phiên benchmark phải lưu cấu hình riêng.

## 1. Mở notebook

Sau khi nhánh setup được merge vào `main`, mở:

```text
https://colab.research.google.com/github/Chicken20145/ai-assisted-parallel-image-processing/blob/main/notebooks/colab_setup.ipynb
```

Hoặc vào Colab, chọn **File → Open notebook → GitHub**, nhập URL repository rồi chọn `notebooks/colab_setup.ipynb`.

## 2. Bật GPU trước khi chạy

1. Chọn **Runtime → Change runtime type**.
2. Chọn **Hardware accelerator: GPU**.
3. Giữ runtime version mặc định mới nhất nếu nhóm chưa chốt phiên bản khác.
4. Nhấn **Save** và chờ Colab kết nối lại.

Chạy cell kiểm tra:

```bash
!nvidia-smi
!nvcc --version
```

Nếu `nvidia-smi` không hiển thị GPU, không tiếp tục benchmark CUDA.

## 3. Clone và setup tự động

Notebook đã có sẵn cell:

```python
%cd /content
!test -d ai-assisted-parallel-image-processing || git clone https://github.com/Chicken20145/ai-assisted-parallel-image-processing.git
%cd /content/ai-assisted-parallel-image-processing
!git pull --ff-only
!bash scripts/setup_colab.sh
```

Script tự động:

1. Xác nhận runtime có GPU NVIDIA và `nvcc`.
2. Cài CMake từ 3.24, Ninja và Python dependencies.
3. Tải BSDS300 từ UC Berkeley và kiểm tra SHA-256.
4. Giải nén 300 ảnh nguồn.
5. Tạo 15 ảnh benchmark ở 5 độ phân giải.
6. Cấu hình CUDA theo GPU thực tế bằng `CMAKE_CUDA_ARCHITECTURES=native`.
7. Build Release trong `build-colab/`.
8. Chạy chương trình probe.
9. Ghi cấu hình vào `data/external/environment_colab.txt`.

Kết thúc thành công sẽ có dòng:

```text
Colab setup completed successfully.
```

## 4. Kiểm tra kết quả setup

```bash
!ls data/external/benchmark_suite/*.png | wc -l
!cat data/external/environment_colab.txt
!./build-colab/parallel_image_processing
```

Kết quả mong đợi:

- Có 15 ảnh PNG benchmark.
- `CUDA devices available` bằng 1 hoặc lớn hơn.
- File môi trường ghi rõ GPU, CUDA, CPU, RAM và thời điểm chạy.

## 5. Quy tắc benchmark trên Colab

- Chạy code và đọc ảnh từ ổ cục bộ `/content`.
- Không benchmark trực tiếp trên thư mục Google Drive vì độ trễ mạng làm sai lệch thời gian end-to-end.
- Build Release, chạy warm-up 3–5 lần và đo ít nhất 20 lần.
- Lưu từng lần đo, không chỉ lưu trung bình.
- Không gộp số liệu giữa các phiên có GPU hoặc CPU khác nhau.
- Luôn giữ `environment_colab.txt` cùng file CSV tương ứng.

## 6. Lưu kết quả sang Google Drive

Chỉ mount Drive sau khi benchmark hoặc ngay trước lúc lưu kết quả:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Sao chép kết quả:

```python
from pathlib import Path
import shutil

source = Path('/content/ai-assisted-parallel-image-processing/benchmarks/results')
destination = Path('/content/drive/MyDrive/parallel-image-processing/results')
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(source, destination, dirs_exist_ok=True)
```

Nên tạo thư mục riêng theo ngày và cấu hình GPU để tránh ghi đè kết quả của phiên trước.

## 7. Sử dụng API key AI

Không ghi API key trực tiếp vào notebook. Dùng mục **Secrets** của Colab với tên `OPENAI_API_KEY`, sau đó đọc trong code:

```python
from google.colab import userdata
import os

os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
```

Không in biến này ra output và không lưu output chứa secret.

## 8. Chạy lại sau khi runtime bị reset

File trong `/content` sẽ mất khi runtime bị hủy. Trong phiên mới:

1. Bật lại GPU.
2. Chạy lại cell clone/setup.
3. Chạy benchmark trên `/content`.
4. Chép CSV, biểu đồ và file môi trường sang Drive.

Không cần lưu `.venv`, `build-colab/` hoặc toàn bộ dataset lên Drive; script có thể tái tạo chúng.

## 9. Lỗi thường gặp

### `No NVIDIA GPU detected`

Runtime chưa bật GPU hoặc Colab chưa cấp GPU. Chọn lại runtime GPU và reconnect. Nếu vẫn không có, thử lại khi tài nguyên khả dụng.

### Không tìm thấy `nvcc`

Runtime hiện tại không có CUDA compiler. Chọn runtime GPU mặc định mới nhất và chạy lại từ đầu.

### CMake không nhận CUDA compiler

Chạy lại cell setup trong runtime mới. Không tự đổi `CMAKE_CUDA_COMPILER` nếu chưa ghi lại đầy đủ cấu hình CUDA.

### Runtime bị ngắt giữa benchmark

Kết quả chưa chép sang Drive sẽ mất. Chỉ sử dụng CSV có đủ số lần chạy; không ghép một phần dữ liệu từ hai cấu hình runtime khác nhau.

### Drive chậm hoặc báo lỗi I/O

Giảm số file nhỏ trên Drive, đóng gói kết quả khi cần và chỉ sao chép sau benchmark. Không dùng Drive làm thư mục làm việc chính.
