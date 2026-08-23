#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

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

pip::Image make_image(int width, int height, int channels) {
    pip::Image image{width, height, channels, {}};
    image.pixels.resize(image.expected_size());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < channels; ++channel) {
                const std::size_t index =
                    (static_cast<std::size_t>(y) * width + x) * channels + channel;
                image.pixels[index] = static_cast<std::uint8_t>((x * 17 + y * 31 + channel * 47) & 0xFF);
            }
        }
    }
    return image;
}

void compare_backend(
    const pip::Image& input,
    pip::Algorithm algorithm,
    pip::ProcessingParams params,
    int threads,
    const std::string& name) {
    const auto sequential = pip::process(input, algorithm, params, pip::Backend::Sequential);
    params.thread_count = threads;
    const auto parallel = pip::process(input, algorithm, params, pip::Backend::OpenMP);
    check(sequential.ok(), name + ": Sequential phải thành công");
    check(parallel.ok(), name + ": OpenMP phải thành công");
    if (!sequential.ok() || !parallel.ok()) {
        return;
    }
    const pip::ErrorMetrics metrics = pip::compare_images(sequential.output, parallel.output);
    check(metrics.mae == 0.0 && metrics.mse == 0.0 && metrics.max_absolute_error == 0,
          name + ": OpenMP phải khớp chính xác Sequential");
    check(parallel.backend_used == pip::Backend::OpenMP, name + ": backend_used phải là OpenMP");
    check(parallel.threads_used == threads, name + ": threads_used phải đúng cấu hình");
}

void test_all_algorithms_and_layouts() {
    const std::vector<int> thread_counts{1, 2, 4};
    const std::vector<pip::Image> images{
        make_image(1, 1, 1),
        make_image(2, 2, 3),
        make_image(7, 5, 1),
        make_image(19, 11, 3),
    };

    pip::ProcessingParams gaussian;
    gaussian.kernel_size = 5;
    gaussian.sigma = 1.2F;
    pip::ProcessingParams sobel;
    sobel.threshold = 0;

    for (int threads : thread_counts) {
        for (const auto& image : images) {
            const std::string suffix = std::to_string(image.width) + "x" + std::to_string(image.height) +
                                       "x" + std::to_string(image.channels) + " threads=" +
                                       std::to_string(threads);
            compare_backend(image, pip::Algorithm::GaussianBlur, gaussian, threads, "Gaussian " + suffix);
            compare_backend(image, pip::Algorithm::Sobel, sobel, threads, "Sobel " + suffix);
            compare_backend(image, pip::Algorithm::HistogramEqualization, {}, threads, "Histogram " + suffix);
        }
    }
}

void test_dispatcher_validation() {
    const pip::Image image = make_image(2, 2, 1);
    pip::ProcessingParams params;
    params.thread_count = -1;
    const auto invalid = pip::process(image, pip::Algorithm::Sobel, params, pip::Backend::OpenMP);
    check(!invalid.ok() && invalid.error == pip::ProcessingError::InvalidParameters,
          "Thread âm phải trả InvalidParameters");

    params.thread_count = 1025;
    const auto too_many = pip::process(image, pip::Algorithm::Sobel, params, pip::Backend::OpenMP);
    check(!too_many.ok() && too_many.error == pip::ProcessingError::InvalidParameters,
          "Thread vượt giới hạn bảo vệ phải trả InvalidParameters");

    params.thread_count = 0;
    const auto automatic = pip::process(image, pip::Algorithm::Sobel, params, pip::Backend::OpenMP);
    check(automatic.ok() && automatic.threads_used > 0,
          "Thread 0 phải dùng cấu hình tự động của OpenMP runtime");

}

}  // namespace

int main() {
    test_all_algorithms_and_layouts();
    test_dispatcher_validation();
    if (failures != 0) {
        std::cerr << failures << " kiểm tra OpenMP thất bại.\n";
        return EXIT_FAILURE;
    }
    std::cout << "Tất cả kiểm tra OpenMP đã đạt.\n";
    return EXIT_SUCCESS;
}
