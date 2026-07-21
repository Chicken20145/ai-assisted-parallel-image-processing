#include <cuda_runtime.h>
#include "parallel_probe.hpp"

int cuda_device_count() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    return status == cudaSuccess ? count : 0;
}

