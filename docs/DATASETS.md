# Nguồn dữ liệu ảnh

Tài liệu này ghi nguồn, điều kiện sử dụng và trạng thái tải của các bộ dữ liệu được cân nhắc cho benchmark. Dữ liệu gốc được lưu trong `data/external/` và file tải về trong `data/downloads/`; cả hai không được commit vào Git.

## Đã tải

### Berkeley Segmentation Dataset (BSDS300)

- Đơn vị phát hành: Computer Vision Group, University of California, Berkeley.
- Mục đích sử dụng trong dự án: ảnh tự nhiên có nhiều biên và chi tiết để kiểm thử Sobel, Gaussian Blur và benchmark kích thước nhỏ.
- Quy mô đã tải: 300 ảnh JPEG, gồm 200 ảnh train và 100 ảnh test.
- Điều kiện: dùng cho nghiên cứu và giáo dục phi thương mại; khi sử dụng phải trích dẫn công trình của D. Martin và cộng sự tại ICCV 2001. Trang nguồn còn yêu cầu công bố kết quả khi dùng bộ dữ liệu để đánh giá segmentation hoặc boundary detection.
- Trang chính thức: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/
- File nguồn: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz
- SHA-256 file tải: `A5F7D0E49FE135C75518A3543CED24470156FD69305AE77845DFF2A5138652B4`
- Vị trí cục bộ: `data/external/BSDS300/images/`
- Script tải đa nền tảng: `python scripts/download_datasets.py`.

### Bộ benchmark dẫn xuất cục bộ

- Script tạo: `scripts/prepare_benchmark_data.py`.
- Nguồn: ba ảnh từ BSDS300, đại diện cho phong cảnh, nhiều chi tiết và tương phản thấp.
- Kích thước: 256x256, 512x512, 1920x1080, 2560x1440 và 3840x2160.
- Số lượng: 15 ảnh PNG (3 nhóm nội dung x 5 kích thước).
- Biến đổi: crop giữa ảnh rồi resize; nhóm tương phản thấp được giảm contrast và brightness theo tham số cố định.
- Metadata: `data/external/benchmark_suite/metadata.csv`, gồm file nguồn, biến đổi, kích thước và SHA-256 từng ảnh.
- Lưu ý: đây là dữ liệu dẫn xuất để benchmark hiệu năng giữa các backend, không phải ground truth đánh giá chất lượng thuật toán.

## Đã xác minh, chưa tải

### DIV2K

- Đơn vị phát hành: Computer Vision Lab, ETH Zurich.
- Mục đích dự kiến: ảnh RGB chất lượng cao độ phân giải 2K, phù hợp tạo bộ benchmark ở nhiều độ phân giải.
- Quy mô: 1.000 ảnh 2K, chia 800 train, 100 validation và 100 test.
- Điều kiện: chỉ dành cho nghiên cứu học thuật; bản quyền từng ảnh thuộc chủ sở hữu gốc. Không đưa ảnh từ bộ này vào Git nếu chưa xác minh quyền phân phối của từng ảnh.
- Trang chính thức: https://data.vision.ee.ethz.ch/cvl/DIV2K/
- Trạng thái: chưa tải vì gói ảnh HR lớn; cần chốt nơi lưu dữ liệu ngoài repository.

### LOL (LOw-Light paired dataset)

- Đơn vị phát hành: nhóm tác giả công trình Deep Retinex Decomposition for Low-Light Enhancement, BMVC 2018.
- Mục đích dự kiến: ảnh thiếu sáng/tương phản thấp cho Histogram Equalization.
- Quy mô: 500 cặp ảnh thiếu sáng và phơi sáng bình thường; ảnh có độ phân giải 400x600.
- Trang dự án chính thức: https://daooshee.github.io/BMVC2018website/
- Trạng thái: chưa tải. Trang dự án cung cấp liên kết Google Drive nhưng không nêu giấy phép sử dụng/phân phối rõ ràng; cần xin phép hoặc xác minh điều khoản trước khi đưa dữ liệu vào quy trình chính thức.

## Nguyên tắc sử dụng

- Chỉ commit ảnh mẫu khi quyền phân phối lại được ghi rõ.
- Dữ liệu benchmark dung lượng lớn phải nằm ngoài Git và có hướng dẫn tải lại.
- Mỗi ảnh được chọn cần có nguồn, giấy phép/điều khoản, kích thước và checksum.
- Không trộn ảnh train/test theo cách làm sai lệch đánh giá; với dự án này, ảnh chỉ dùng làm đầu vào hiệu năng và kiểm tra tính đúng đắn giữa các backend.
