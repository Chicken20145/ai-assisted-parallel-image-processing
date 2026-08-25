#include "openmp_algorithms.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace pip::openmp {
namespace {

std::vector<float> make_kernel(int kernel_size, float sigma) {
    if ((kernel_size != 3 && kernel_size != 5 && kernel_size != 7) ||
        !std::isfinite(sigma) || sigma < 0.1F || sigma > 10.0F) {
        throw std::invalid_argument("Gaussian Blur chỉ nhận kernel 3/5/7 và sigma trong khoảng 0.1–10.0.");
    }

    const int radius = kernel_size / 2;
    const float denominator = 2.0F * sigma * sigma;
    std::vector<float> kernel(static_cast<std::size_t>(kernel_size));
    float sum = 0.0F;
    for (int offset = -radius; offset <= radius; ++offset) {
        const float value = std::exp(-static_cast<float>(offset * offset) / denominator);
        kernel[static_cast<std::size_t>(offset + radius)] = value;
        sum += value;
    }
    for (float& value : kernel) {
        value /= sum;
    }
    return kernel;
}

}  // namespace

Image gaussian_blur(const Image& input, int kernel_size, float sigma, int thread_count) {
    if (!input.is_valid()) {
        throw std::invalid_argument("Ảnh đầu vào không hợp lệ.");
    }
    const int threads = effective_thread_count(thread_count);
    const std::vector<float> kernel = make_kernel(kernel_size, sigma);
    const int radius = kernel_size / 2;
    Image output{input.width, input.height, input.channels, {}};
    output.pixels.resize(input.pixels.size());
    std::vector<float> intermediate(input.pixels.size());

#pragma omp parallel num_threads(threads)
    {
#pragma omp for schedule(static)
        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                for (int channel = 0; channel < input.channels; ++channel) {
                    float sum = 0.0F;
                    for (int kx = -radius; kx <= radius; ++kx) {
                        const int source_x = std::clamp(x + kx, 0, input.width - 1);
                        const std::size_t source_index =
                            (static_cast<std::size_t>(y) * input.width + source_x) * input.channels + channel;
                        sum += static_cast<float>(input.pixels[source_index]) *
                               kernel[static_cast<std::size_t>(kx + radius)];
                    }
                    const std::size_t intermediate_index =
                        (static_cast<std::size_t>(y) * input.width + x) * input.channels + channel;
                    intermediate[intermediate_index] = sum;
                }
            }
        }

#pragma omp for schedule(static)
        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                for (int channel = 0; channel < input.channels; ++channel) {
                    float sum = 0.0F;
                    for (int ky = -radius; ky <= radius; ++ky) {
                        const int source_y = std::clamp(y + ky, 0, input.height - 1);
                        const std::size_t source_index =
                            (static_cast<std::size_t>(source_y) * input.width + x) * input.channels + channel;
                        sum += intermediate[source_index] * kernel[static_cast<std::size_t>(ky + radius)];
                    }
                    const std::size_t output_index =
                        (static_cast<std::size_t>(y) * input.width + x) * input.channels + channel;
                    output.pixels[output_index] =
                        static_cast<std::uint8_t>(std::clamp(std::lround(sum), 0L, 255L));
                }
            }
        }
    }
    return output;
}

}  // namespace pip::openmp
