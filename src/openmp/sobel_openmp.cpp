#include "openmp_algorithms.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pip::openmp {

Image sobel(const Image& input, int threshold, int thread_count) {
    if (!input.is_valid()) {
        throw std::invalid_argument("Ảnh đầu vào không hợp lệ.");
    }
    if (threshold < 0 || threshold > 255) {
        throw std::invalid_argument("Sobel threshold phải nằm trong khoảng 0–255.");
    }
    const int threads = effective_thread_count(thread_count);
    const Image grayscale = to_grayscale(input, threads);
    Image output{input.width, input.height, 1, {}};
    output.pixels.resize(grayscale.pixels.size());
    constexpr int gx_kernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    constexpr int gy_kernel[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

#pragma omp parallel for schedule(static) num_threads(threads)
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            int gx = 0;
            int gy = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                const int source_y = std::clamp(y + ky, 0, input.height - 1);
                for (int kx = -1; kx <= 1; ++kx) {
                    const int source_x = std::clamp(x + kx, 0, input.width - 1);
                    const int value = grayscale.pixels[static_cast<std::size_t>(source_y) * input.width + source_x];
                    gx += value * gx_kernel[ky + 1][kx + 1];
                    gy += value * gy_kernel[ky + 1][kx + 1];
                }
            }
            const int magnitude = std::clamp(
                static_cast<int>(std::lround(std::sqrt(static_cast<double>(gx * gx + gy * gy)))), 0, 255);
            output.pixels[static_cast<std::size_t>(y) * input.width + x] =
                threshold == 0 ? static_cast<std::uint8_t>(magnitude)
                               : static_cast<std::uint8_t>(magnitude >= threshold ? 255 : 0);
        }
    }
    return output;
}

}  // namespace pip::openmp
