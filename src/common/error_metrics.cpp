#include "error_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pip {

ErrorMetrics compare_images(const Image& reference, const Image& candidate) {
    if (!reference.is_valid() || !candidate.is_valid() ||
        reference.width != candidate.width || reference.height != candidate.height ||
        reference.channels != candidate.channels) {
        throw std::invalid_argument("Hai ảnh phải hợp lệ và có cùng width, height, channels.");
    }

    double absolute_sum = 0.0;
    double squared_sum = 0.0;
    int maximum = 0;
    for (std::size_t i = 0; i < reference.pixels.size(); ++i) {
        const int difference = std::abs(static_cast<int>(reference.pixels[i]) - candidate.pixels[i]);
        absolute_sum += difference;
        squared_sum += static_cast<double>(difference * difference);
        maximum = std::max(maximum, difference);
    }
    const double count = static_cast<double>(reference.pixels.size());
    return {absolute_sum / count, squared_sum / count, static_cast<std::uint8_t>(maximum)};
}

}  // namespace pip
