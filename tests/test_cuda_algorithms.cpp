#include <cstdlib>
#include <iostream>
#include <string>

#include "error_metrics.hpp"
#include "processing_api.hpp"

namespace {
int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) { std::cerr << "[FAIL] " << message << '\n'; ++failures; }
}

pip::Image make_image() {
    pip::Image image{19, 11, 3, {}};
    image.pixels.resize(image.expected_size());
    for (std::size_t i = 0; i < image.pixels.size(); ++i) image.pixels[i] = static_cast<std::uint8_t>((i * 37) & 0xFF);
    return image;
}

void compare(pip::Algorithm algorithm, pip::ProcessingParams params, pip::Backend backend, const char* name) {
    const auto expected = pip::process(make_image(), algorithm, params, pip::Backend::Sequential);
    const auto actual = pip::process(make_image(), algorithm, params, backend);
    check(expected.ok() && actual.ok(), std::string(name) + " phải chạy thành công");
    if (!expected.ok() || !actual.ok()) return;
    const auto metrics = pip::compare_images(expected.output, actual.output);
    const int tolerance = algorithm == pip::Algorithm::GaussianBlur ? 1 : 0;
    check(metrics.max_absolute_error <= tolerance, std::string(name) + " phải khớp CPU trong sai số cho phép");
    check(actual.backend_used == backend, std::string(name) + " phải báo đúng backend");
}
}  // namespace

int main() {
    const auto probe = pip::process(make_image(), pip::Algorithm::HistogramEqualization, {}, pip::Backend::CudaBasic);
    if (!probe.ok() && probe.error == pip::ProcessingError::BackendUnavailable) {
        std::cout << "Bỏ qua test CUDA: " << probe.error_message << '\n';
        return EXIT_SUCCESS;
    }
    pip::ProcessingParams gaussian;
    gaussian.kernel_size = 5;
    gaussian.sigma = 1.2F;
    pip::ProcessingParams sobel;
    sobel.threshold = 0;
    for (const auto backend : {pip::Backend::CudaBasic, pip::Backend::CudaOptimized}) {
        compare(pip::Algorithm::GaussianBlur, gaussian, backend, "Gaussian CUDA");
        compare(pip::Algorithm::Sobel, sobel, backend, "Sobel CUDA");
        compare(pip::Algorithm::HistogramEqualization, {}, backend, "Histogram CUDA");
    }
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
