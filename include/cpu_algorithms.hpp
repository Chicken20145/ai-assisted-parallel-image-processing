#pragma once

#include "image_types.hpp"

namespace pip::cpu {

[[nodiscard]] Image to_grayscale(const Image& input);
[[nodiscard]] Image gaussian_blur(const Image& input, int kernel_size, float sigma);
[[nodiscard]] Image sobel(const Image& input, int threshold);
[[nodiscard]] Image histogram_equalization(const Image& input);

}  // namespace pip::cpu
