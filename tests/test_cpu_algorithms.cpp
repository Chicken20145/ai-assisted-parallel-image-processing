#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cpu_algorithms.hpp"
#include "error_metrics.hpp"
#include "processing_api.hpp"

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[FAIL] " << message << '\n';
        ++failures;
    }
}

void test_image_validation() {
    check(pip::Image{1, 1, 1, {42}}.is_valid(), "Ảnh grayscale 1x1 phải hợp lệ");
    check(pip::Image{1, 1, 3, {1, 2, 3}}.is_valid(), "Ảnh RGB 1x1 phải hợp lệ");
    check(!pip::Image{1, 1, 2, {1, 2}}.is_valid(), "Ảnh 2 kênh phải bị từ chối");
    check(!pip::Image{2, 2, 1, {1, 2}}.is_valid(), "Buffer sai kích thước phải bị từ chối");
}

void test_grayscale() {
    const pip::Image rgb{3, 1, 3, {255, 0, 0, 0, 255, 0, 0, 0, 255}};
    const pip::Image gray = pip::cpu::to_grayscale(rgb);
    check(gray.channels == 1 && gray.pixels == std::vector<std::uint8_t>({77, 149, 29}),
          "RGB sang grayscale phải dùng công thức cố định");
}

void test_gaussian() {
    const pip::Image one_pixel{1, 1, 1, {123}};
    check(pip::cpu::gaussian_blur(one_pixel, 7, 2.0F).pixels == one_pixel.pixels,
          "Gaussian phải xử lý ảnh nhỏ hơn kernel");

    const pip::Image constant{7, 5, 3, std::vector<std::uint8_t>(7 * 5 * 3, 80)};
    check(pip::cpu::gaussian_blur(constant, 5, 1.2F).pixels == constant.pixels,
          "Gaussian phải giữ nguyên ảnh hằng số RGB");
}

void test_sobel() {
    const pip::Image constant{2, 2, 1, {40, 40, 40, 40}};
    check(pip::cpu::sobel(constant, 0).pixels == std::vector<std::uint8_t>({0, 0, 0, 0}),
          "Sobel ảnh hằng số phải bằng 0");

    const pip::Image edge{3, 3, 1, {0, 0, 255, 0, 0, 255, 0, 0, 255}};
    const pip::Image binary = pip::cpu::sobel(edge, 100);
    check(binary.pixels[1] == 255, "Sobel phải phát hiện cạnh dọc");
    for (std::uint8_t value : binary.pixels) {
        check(value == 0 || value == 255, "Sobel có threshold chỉ được trả 0 hoặc 255");
    }
}

void test_histogram_equalization() {
    const pip::Image image{4, 1, 1, {0, 0, 128, 255}};
    check(pip::cpu::histogram_equalization(image).pixels ==
              std::vector<std::uint8_t>({0, 0, 128, 255}),
          "Histogram Equalization phải ánh xạ theo CDF đã chuẩn hóa");

    const pip::Image constant{1, 1, 3, {10, 20, 30}};
    const pip::Image result = pip::cpu::histogram_equalization(constant);
    check(result.channels == 1 && result.pixels.size() == 1,
          "Histogram Equalization RGB phải trả grayscale và xử lý ảnh 1x1");
}

void test_dispatcher_and_errors() {
    const pip::Image image{1, 1, 1, {100}};
    pip::ProcessingParams params;
    params.kernel_size = 4;
    const auto invalid = pip::process(image, pip::Algorithm::GaussianBlur, params, pip::Backend::Sequential);
    check(!invalid.ok() && invalid.error == pip::ProcessingError::InvalidParameters,
          "Dispatcher phải trả lỗi tham số, không crash");

    const auto unavailable = pip::process(image, pip::Algorithm::Sobel, {}, pip::Backend::CudaBasic);
    check(!unavailable.ok() && unavailable.error == pip::ProcessingError::BackendUnavailable,
          "Dispatcher phải báo backend CUDA chưa sẵn sàng");

    const auto success = pip::process(image, pip::Algorithm::HistogramEqualization, {}, pip::Backend::Sequential);
    check(success.ok() && success.output.is_valid() && success.timing.total_ms >= 0.0,
          "Dispatcher CPU phải trả ảnh và timing hợp lệ");
}

void test_error_metrics() {
    const pip::Image reference{2, 1, 1, {10, 20}};
    const pip::Image candidate{2, 1, 1, {12, 16}};
    const pip::ErrorMetrics metrics = pip::compare_images(reference, candidate);
    check(std::abs(metrics.mae - 3.0) < 1e-12, "MAE phải chính xác");
    check(std::abs(metrics.mse - 10.0) < 1e-12, "MSE phải chính xác");
    check(metrics.max_absolute_error == 4, "Max absolute error phải chính xác");
}

}  // namespace

int main() {
    test_image_validation();
    test_grayscale();
    test_gaussian();
    test_sobel();
    test_histogram_equalization();
    test_dispatcher_and_errors();
    test_error_metrics();
    if (failures != 0) {
        std::cerr << failures << " kiểm tra thất bại.\n";
        return EXIT_FAILURE;
    }
    std::cout << "Tất cả kiểm tra CPU tuần tự đã đạt.\n";
    return EXIT_SUCCESS;
}
