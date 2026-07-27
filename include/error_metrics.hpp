#pragma once

#include <cstdint>

#include "image_types.hpp"

namespace pip {

struct ErrorMetrics {
    double mae = 0.0;
    double mse = 0.0;
    std::uint8_t max_absolute_error = 0;
};

[[nodiscard]] ErrorMetrics compare_images(const Image& reference, const Image& candidate);

}  // namespace pip
