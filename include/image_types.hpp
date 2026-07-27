#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace pip {

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<std::uint8_t> pixels;

    [[nodiscard]] bool has_supported_layout() const noexcept {
        return channels == 1 || channels == 3;
    }

    [[nodiscard]] bool dimensions_are_valid() const noexcept {
        return width > 0 && height > 0;
    }

    [[nodiscard]] std::size_t expected_size() const noexcept {
        if (!dimensions_are_valid() || channels <= 0) {
            return 0;
        }
        const auto w = static_cast<std::size_t>(width);
        const auto h = static_cast<std::size_t>(height);
        const auto c = static_cast<std::size_t>(channels);
        if (w > std::numeric_limits<std::size_t>::max() / h ||
            w * h > std::numeric_limits<std::size_t>::max() / c) {
            return 0;
        }
        return w * h * c;
    }

    [[nodiscard]] bool is_valid() const noexcept {
        const std::size_t size = expected_size();
        return has_supported_layout() && size != 0 && pixels.size() == size;
    }
};

}  // namespace pip
