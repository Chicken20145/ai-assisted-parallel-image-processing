#pragma once

#include <cstdint>
#include <string>

#include "image_types.hpp"

namespace pip {

enum class Algorithm {
    GaussianBlur,
    Sobel,
    HistogramEqualization,
};

enum class Backend {
    Sequential,
    OpenMP,
    CudaBasic,
    CudaOptimized,
};

enum class ProcessingError {
    None,
    InvalidImage,
    InvalidParameters,
    BackendUnavailable,
    InternalError,
};

struct ProcessingParams {
    int kernel_size = 3;
    float sigma = 1.0F;
    int threshold = 100;
};

struct Timing {
    double allocation_ms = 0.0;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double total_ms = 0.0;
};

struct ProcessingResult {
    Image output;
    Timing timing;
    Backend backend_used = Backend::Sequential;
    ProcessingError error = ProcessingError::None;
    std::string error_message;

    [[nodiscard]] bool ok() const noexcept {
        return error == ProcessingError::None;
    }
};

[[nodiscard]] ProcessingResult process(
    const Image& input,
    Algorithm algorithm,
    const ProcessingParams& params,
    Backend backend);

[[nodiscard]] const char* to_string(Algorithm algorithm) noexcept;
[[nodiscard]] const char* to_string(Backend backend) noexcept;
[[nodiscard]] const char* to_string(ProcessingError error) noexcept;

}  // namespace pip
