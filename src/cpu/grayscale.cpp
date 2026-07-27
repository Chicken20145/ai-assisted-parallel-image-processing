#include "cpu_algorithms.hpp"

#include <stdexcept>

namespace pip::cpu {

Image to_grayscale(const Image& input) {
    if (!input.is_valid()) {
        throw std::invalid_argument("Ảnh đầu vào không hợp lệ.");
    }
    if (input.channels == 1) {
        return input;
    }

    Image output{input.width, input.height, 1, {}};
    output.pixels.resize(static_cast<std::size_t>(input.width) * input.height);
    for (std::size_t i = 0, pixel = 0; i < input.pixels.size(); i += 3, ++pixel) {
        const int red = input.pixels[i];
        const int green = input.pixels[i + 1];
        const int blue = input.pixels[i + 2];
        output.pixels[pixel] = static_cast<std::uint8_t>((77 * red + 150 * green + 29 * blue + 128) >> 8);
    }
    return output;
}

}  // namespace pip::cpu
