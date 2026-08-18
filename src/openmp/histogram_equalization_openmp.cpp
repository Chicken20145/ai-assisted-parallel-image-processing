#include "openmp_algorithms.hpp"

#include <omp.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace pip::openmp {

Image histogram_equalization(const Image& input, int thread_count) {
    if (!input.is_valid()) {
        throw std::invalid_argument("Ảnh đầu vào không hợp lệ.");
    }
    const int threads = effective_thread_count(thread_count);
    const Image grayscale = to_grayscale(input, threads);
    const auto pixel_count = static_cast<std::int64_t>(grayscale.pixels.size());
    std::vector<std::array<std::size_t, 256>> local_histograms(static_cast<std::size_t>(threads));

#pragma omp parallel num_threads(threads)
    {
        const int thread_id = omp_get_thread_num();
        auto& histogram = local_histograms[static_cast<std::size_t>(thread_id)];
#pragma omp for schedule(static)
        for (std::int64_t i = 0; i < pixel_count; ++i) {
            ++histogram[grayscale.pixels[static_cast<std::size_t>(i)]];
        }
    }

    std::array<std::size_t, 256> histogram{};
    for (const auto& local : local_histograms) {
        for (std::size_t value = 0; value < histogram.size(); ++value) {
            histogram[value] += local[value];
        }
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
    const std::size_t count = grayscale.pixels.size();
    if (cdf_min == count) {
        return grayscale;
    }

    Image output{input.width, input.height, 1, {}};
    output.pixels.resize(count);
    const double denominator = static_cast<double>(count - cdf_min);
#pragma omp parallel for schedule(static) num_threads(threads)
    for (std::int64_t i = 0; i < pixel_count; ++i) {
        const std::uint8_t value = grayscale.pixels[static_cast<std::size_t>(i)];
        const double mapped = static_cast<double>(cdf[value] - cdf_min) * 255.0 / denominator;
        output.pixels[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(std::lround(mapped));
    }
    return output;
}

}  // namespace pip::openmp
