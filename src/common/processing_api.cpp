#include "processing_api.hpp"

#include <chrono>
#include <exception>
#include <stdexcept>
#include <utility>

#include "cpu_algorithms.hpp"

namespace pip {
namespace {

ProcessingResult make_error(Backend backend, ProcessingError error, std::string message) {
    ProcessingResult result;
    result.backend_used = backend;
    result.error = error;
    result.error_message = std::move(message);
    return result;
}

}  // namespace

ProcessingResult process(
    const Image& input,
    Algorithm algorithm,
    const ProcessingParams& params,
    Backend backend) {
    if (!input.is_valid()) {
        return make_error(backend, ProcessingError::InvalidImage,
                          "Ảnh phải có width/height dương, 1 hoặc 3 kênh và buffer liên tục đúng kích thước.");
    }
    if (backend != Backend::Sequential) {
        return make_error(backend, ProcessingError::BackendUnavailable,
                          "Backend này chưa được triển khai trong mốc CPU tuần tự.");
    }

    const auto started = std::chrono::steady_clock::now();
    try {
        Image output;
        switch (algorithm) {
            case Algorithm::GaussianBlur:
                output = cpu::gaussian_blur(input, params.kernel_size, params.sigma);
                break;
            case Algorithm::Sobel:
                output = cpu::sobel(input, params.threshold);
                break;
            case Algorithm::HistogramEqualization:
                output = cpu::histogram_equalization(input);
                break;
        }
        const auto finished = std::chrono::steady_clock::now();
        ProcessingResult result;
        result.output = std::move(output);
        result.backend_used = backend;
        result.timing.kernel_ms = std::chrono::duration<double, std::milli>(finished - started).count();
        result.timing.total_ms = result.timing.kernel_ms;
        return result;
    } catch (const std::invalid_argument& error) {
        return make_error(backend, ProcessingError::InvalidParameters, error.what());
    } catch (const std::exception& error) {
        return make_error(backend, ProcessingError::InternalError, error.what());
    }
}

const char* to_string(Algorithm algorithm) noexcept {
    switch (algorithm) {
        case Algorithm::GaussianBlur: return "gaussian_blur";
        case Algorithm::Sobel: return "sobel";
        case Algorithm::HistogramEqualization: return "histogram_equalization";
    }
    return "unknown";
}

const char* to_string(Backend backend) noexcept {
    switch (backend) {
        case Backend::Sequential: return "sequential";
        case Backend::OpenMP: return "openmp";
        case Backend::CudaBasic: return "cuda_basic";
        case Backend::CudaOptimized: return "cuda_optimized";
    }
    return "unknown";
}

const char* to_string(ProcessingError error) noexcept {
    switch (error) {
        case ProcessingError::None: return "none";
        case ProcessingError::InvalidImage: return "invalid_image";
        case ProcessingError::InvalidParameters: return "invalid_parameters";
        case ProcessingError::BackendUnavailable: return "backend_unavailable";
        case ProcessingError::InternalError: return "internal_error";
    }
    return "unknown";
}

}  // namespace pip
