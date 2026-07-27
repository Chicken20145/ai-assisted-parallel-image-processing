#include "cpu_algorithms.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

namespace pip::cpu {

Image histogram_equalization(const Image& input) {
    if (!input.is_valid()) {
        throw std::invalid_argument("Ảnh đầu vào không hợp lệ.");
    }
    const Image grayscale = to_grayscale(input);
    std::array<std::size_t, 256> histogram{};
    for (std::uint8_t value : grayscale.pixels) {
        ++histogram[value];
    }

    std::array<std::size_t, 256> cdf{};
    cdf[0] = histogram[0];
    for (std::size_t i = 1; i < cdf.size(); ++i) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    std::size_t cdf_min = 0;
    for (std::size_t value : cdf) {
        if (value != 0) {
            cdf_min = value;
            break;
        }
    }
    const std::size_t pixel_count = grayscale.pixels.size();
    if (cdf_min == pixel_count) {
        return grayscale;
    }

    Image output{input.width, input.height, 1, {}};
    output.pixels.resize(pixel_count);
    const double denominator = static_cast<double>(pixel_count - cdf_min);
    for (std::size_t i = 0; i < pixel_count; ++i) {
        const std::uint8_t value = grayscale.pixels[i];
        const double mapped = static_cast<double>(cdf[value] - cdf_min) * 255.0 / denominator;
        output.pixels[i] = static_cast<std::uint8_t>(std::lround(mapped));
    }
    return output;
}

}  // namespace pip::cpu
