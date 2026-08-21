#pragma once

#include "processing_api.hpp"

namespace pip::cuda_backend {

[[nodiscard]] ProcessingResult process_cuda(
    const Image& input,
    Algorithm algorithm,
    const ProcessingParams& params,
    Backend backend);

}  // namespace pip::cuda_backend
