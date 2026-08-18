#pragma once

#include "image_types.hpp"

namespace pip::openmp {

[[nodiscard]] int effective_thread_count(int requested_threads);
[[nodiscard]] Image to_grayscale(const Image& input, int thread_count = 0);
[[nodiscard]] Image gaussian_blur(const Image& input, int kernel_size, float sigma, int thread_count = 0);
[[nodiscard]] Image sobel(const Image& input, int threshold, int thread_count = 0);
[[nodiscard]] Image histogram_equalization(const Image& input, int thread_count = 0);

}  // namespace pip::openmp
