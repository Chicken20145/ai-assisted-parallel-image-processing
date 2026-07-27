#include "cpu_algorithms.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace pip::cpu {
namespace {

std::vector<float> make_kernel(int kernel_size, float sigma) {
    if ((kernel_size != 3 && kernel_size != 5 && kernel_size != 7) ||
        !std::isfinite(sigma) || sigma < 0.1F || sigma > 10.0F) {
        throw std::invalid_argument("Gaussian Blur chỉ nhận kernel 3/5/7 và sigma trong khoảng 0.1–10.0.");
    }

    const int radius = kernel_size / 2;
    const float denominator = 2.0F * sigma * sigma;
    std::vector<float> kernel(static_cast<std::size_t>(kernel_size * kernel_size));
    float sum = 0.0F;
    for (int ky = -radius; ky <= radius; ++ky) {
        for (int kx = -radius; kx <= radius; ++kx) {
            const float value = std::exp(-static_cast<float>(kx * kx + ky * ky) / denominator);
            const std::size_t index = static_cast<std::size_t>((ky + radius) * kernel_size + (kx + radius));
            kernel[index] = value;
            sum += value;
        }
    }
    for (float& value : kernel) {
        value /= sum;
    }
    return kernel;
}

}  // namespace

Image gaussian_blur(const Image& input, int kernel_size, float sigma) {
    if (!input.is_valid()) {
        throw std::invalid_argument("Ảnh đầu vào không hợp lệ.");
    }
    const std::vector<float> kernel = make_kernel(kernel_size, sigma);
    const int radius = kernel_size / 2;
    Image output{input.width, input.height, input.channels, {}};
    output.pixels.resize(input.pixels.size());

    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            for (int channel = 0; channel < input.channels; ++channel) {
                float sum = 0.0F;
                for (int ky = -radius; ky <= radius; ++ky) {
                    const int source_y = std::clamp(y + ky, 0, input.height - 1);
                    for (int kx = -radius; kx <= radius; ++kx) {
                        const int source_x = std::clamp(x + kx, 0, input.width - 1);
                        const std::size_t source_index =
                            (static_cast<std::size_t>(source_y) * input.width + source_x) * input.channels + channel;
                        const std::size_t kernel_index =
                            static_cast<std::size_t>((ky + radius) * kernel_size + (kx + radius));
                        sum += static_cast<float>(input.pixels[source_index]) * kernel[kernel_index];
                    }
                }
                const std::size_t output_index =
                    (static_cast<std::size_t>(y) * input.width + x) * input.channels + channel;
                output.pixels[output_index] = static_cast<std::uint8_t>(std::clamp(std::lround(sum), 0L, 255L));
            }
        }
    }
    return output;
}

}  // namespace pip::cpu
