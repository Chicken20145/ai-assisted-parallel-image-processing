#include "openmp_algorithms.hpp"

#include <omp.h>

#include <stdexcept>

namespace pip::openmp {

Image to_grayscale(const Image& input, int thread_count) {
    if (!input.is_valid()) {
        throw std::invalid_argument("Ảnh đầu vào không hợp lệ.");
    }
    const int threads = effective_thread_count(thread_count);
    if (input.channels == 1) {
        return input;
    }

    const auto pixel_count = static_cast<std::int64_t>(input.width) * input.height;
    Image output{input.width, input.height, 1, {}};
    output.pixels.resize(static_cast<std::size_t>(pixel_count));

#pragma omp parallel for schedule(static) num_threads(threads)
    for (std::int64_t pixel = 0; pixel < pixel_count; ++pixel) {
        const std::size_t input_index = static_cast<std::size_t>(pixel) * 3;
        const int red = input.pixels[input_index];
        const int green = input.pixels[input_index + 1];
        const int blue = input.pixels[input_index + 2];
        output.pixels[static_cast<std::size_t>(pixel)] =
            static_cast<std::uint8_t>((77 * red + 150 * green + 29 * blue + 128) >> 8);
    }
    return output;
}

}  // namespace pip::openmp
