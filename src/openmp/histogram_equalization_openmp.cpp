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
    Image grayscale{input.width, input.height, 1, {}};
    grayscale.pixels.resize(static_cast<std::size_t>(input.width) * input.height);
    if (input.channels == 1) {
        grayscale.pixels = input.pixels;
    }
    const auto pixel_count = static_cast<std::int64_t>(grayscale.pixels.size());
    std::vector<std::array<std::size_t, 256>> local_histograms(static_cast<std::size_t>(threads));
    std::array<std::size_t, 256> histogram{};
    std::array<std::size_t, 256> cdf{};
    std::size_t cdf_min = 0;
    bool uniform_image = false;
    Image output{input.width, input.height, 1, {}};
    output.pixels.resize(grayscale.pixels.size());

#pragma omp parallel num_threads(threads)
    {
        if (input.channels == 3) {
#pragma omp for schedule(static)
            for (std::int64_t pixel = 0; pixel < pixel_count; ++pixel) {
                const std::size_t source = static_cast<std::size_t>(pixel) * 3;
                grayscale.pixels[static_cast<std::size_t>(pixel)] = static_cast<std::uint8_t>(
                    (77 * input.pixels[source] + 150 * input.pixels[source + 1] +
                     29 * input.pixels[source + 2] + 128) >> 8);
            }
        }

        const int thread_id = omp_get_thread_num();
        auto& local_histogram = local_histograms[static_cast<std::size_t>(thread_id)];
#pragma omp for schedule(static)
        for (std::int64_t i = 0; i < pixel_count; ++i) {
            ++local_histogram[grayscale.pixels[static_cast<std::size_t>(i)]];
        }

#pragma omp single
        {
            for (const auto& local : local_histograms) {
                for (std::size_t value = 0; value < histogram.size(); ++value) {
                    histogram[value] += local[value];
                }
            }
            cdf[0] = histogram[0];
            for (std::size_t value = 1; value < cdf.size(); ++value) {
                cdf[value] = cdf[value - 1] + histogram[value];
            }
            for (std::size_t value : cdf) {
                if (value != 0) {
                    cdf_min = value;
                    break;
                }
            }
            uniform_image = cdf_min == grayscale.pixels.size();
        }

        if (!uniform_image) {
            const double denominator = static_cast<double>(grayscale.pixels.size() - cdf_min);
#pragma omp for schedule(static)
            for (std::int64_t i = 0; i < pixel_count; ++i) {
                const std::uint8_t value = grayscale.pixels[static_cast<std::size_t>(i)];
                const double mapped = static_cast<double>(cdf[value] - cdf_min) * 255.0 / denominator;
                output.pixels[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(std::lround(mapped));
            }
        }
    }
    return uniform_image ? grayscale : output;
}

}  // namespace pip::openmp
