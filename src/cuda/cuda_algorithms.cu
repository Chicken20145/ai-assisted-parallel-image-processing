#include "cuda_algorithms.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace pip::cuda_backend {
namespace {

void check_cuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
    }
}

__device__ int clamp_device(int value, int low, int high) {
    return max(low, min(value, high));
}

__global__ void grayscale_kernel(const unsigned char* input, unsigned char* output, int pixels, int channels) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixels) return;
    if (channels == 1) {
        output[index] = input[index];
        return;
    }
    const int source = index * 3;
    output[index] = static_cast<unsigned char>(
        (77 * input[source] + 150 * input[source + 1] + 29 * input[source + 2] + 128) >> 8);
}

__global__ void gaussian_kernel(
    const unsigned char* input, unsigned char* output, int width, int height, int channels,
    const float* weights, int kernel_size) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int radius = kernel_size / 2;
    for (int channel = 0; channel < channels; ++channel) {
        float sum = 0.0F;
        for (int ky = -radius; ky <= radius; ++ky) {
            const int sy = clamp_device(y + ky, 0, height - 1);
            for (int kx = -radius; kx <= radius; ++kx) {
                const int sx = clamp_device(x + kx, 0, width - 1);
                const float product = __fmul_rn(
                    static_cast<float>(input[(sy * width + sx) * channels + channel]),
                    weights[(ky + radius) * kernel_size + kx + radius]);
                sum = __fadd_rn(sum, product);
            }
        }
        output[(y * width + x) * channels + channel] =
            static_cast<unsigned char>(clamp_device(static_cast<int>(floorf(sum + 0.5F)), 0, 255));
    }
}

__global__ void gaussian_horizontal_kernel(
    const unsigned char* input, float* intermediate, int width, int height, int channels,
    const float* weights, int kernel_size) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int radius = kernel_size / 2;
    for (int channel = 0; channel < channels; ++channel) {
        float sum = 0.0F;
        for (int kx = -radius; kx <= radius; ++kx) {
            const int source_x = clamp_device(x + kx, 0, width - 1);
            const float product = __fmul_rn(
                static_cast<float>(input[(y * width + source_x) * channels + channel]),
                weights[kx + radius]);
            sum = __fadd_rn(sum, product);
        }
        intermediate[(y * width + x) * channels + channel] = sum;
    }
}

__global__ void gaussian_vertical_kernel(
    const float* intermediate, unsigned char* output, int width, int height, int channels,
    const float* weights, int kernel_size) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int radius = kernel_size / 2;
    for (int channel = 0; channel < channels; ++channel) {
        float sum = 0.0F;
        for (int ky = -radius; ky <= radius; ++ky) {
            const int source_y = clamp_device(y + ky, 0, height - 1);
            const float product = __fmul_rn(
                intermediate[(source_y * width + x) * channels + channel],
                weights[ky + radius]);
            sum = __fadd_rn(sum, product);
        }
        output[(y * width + x) * channels + channel] = static_cast<unsigned char>(
            clamp_device(static_cast<int>(floorf(sum + 0.5F)), 0, 255));
    }
}

__global__ void sobel_kernel(
    const unsigned char* grayscale, unsigned char* output, int width, int height, int threshold) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int gx_values[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    const int gy_values[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    int gx = 0;
    int gy = 0;
    for (int ky = -1; ky <= 1; ++ky) {
        const int sy = clamp_device(y + ky, 0, height - 1);
        for (int kx = -1; kx <= 1; ++kx) {
            const int sx = clamp_device(x + kx, 0, width - 1);
            const int value = grayscale[sy * width + sx];
            const int kernel_index = (ky + 1) * 3 + kx + 1;
            gx += value * gx_values[kernel_index];
            gy += value * gy_values[kernel_index];
        }
    }
    const int magnitude = clamp_device(static_cast<int>(floor(sqrt(static_cast<double>(gx * gx + gy * gy)) + 0.5)), 0, 255);
    output[y * width + x] = static_cast<unsigned char>(threshold == 0 ? magnitude : (magnitude >= threshold ? 255 : 0));
}

__global__ void sobel_tiled_kernel(
    const unsigned char* grayscale, unsigned char* output, int width, int height, int threshold) {
    extern __shared__ unsigned char tile[];
    constexpr int radius = 1;
    const int tile_width = blockDim.x + 2 * radius;
    const int tile_height = blockDim.y + 2 * radius;
    const int tile_values = tile_width * tile_height;
    const int local_thread = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_threads = blockDim.x * blockDim.y;
    for (int linear = local_thread; linear < tile_values; linear += block_threads) {
        const int tile_x = linear % tile_width;
        const int tile_y = linear / tile_width;
        const int source_x = clamp_device(
            static_cast<int>(blockIdx.x * blockDim.x) + tile_x - radius, 0, width - 1);
        const int source_y = clamp_device(
            static_cast<int>(blockIdx.y * blockDim.y) + tile_y - radius, 0, height - 1);
        tile[linear] = grayscale[source_y * width + source_x];
    }
    __syncthreads();

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    constexpr int gx_values[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    constexpr int gy_values[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    int gx = 0;
    int gy = 0;
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            const int value = tile[(threadIdx.y + ky + radius) * tile_width + threadIdx.x + kx + radius];
            const int kernel_index = (ky + 1) * 3 + kx + 1;
            gx += value * gx_values[kernel_index];
            gy += value * gy_values[kernel_index];
        }
    }
    const int magnitude = clamp_device(
        static_cast<int>(floor(sqrt(static_cast<double>(gx * gx + gy * gy)) + 0.5)), 0, 255);
    output[y * width + x] = static_cast<unsigned char>(
        threshold == 0 ? magnitude : (magnitude >= threshold ? 255 : 0));
}

__global__ void histogram_basic_kernel(const unsigned char* image, unsigned int* histogram, int pixels) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < pixels) atomicAdd(&histogram[image[index]], 1U);
}

__global__ void histogram_optimized_kernel(const unsigned char* image, unsigned int* histogram, int pixels) {
    __shared__ unsigned int local[256];
    local[threadIdx.x] = 0;
    __syncthreads();
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < pixels; index += blockDim.x * gridDim.x) {
        atomicAdd(&local[image[index]], 1U);
    }
    __syncthreads();
    if (local[threadIdx.x] != 0) atomicAdd(&histogram[threadIdx.x], local[threadIdx.x]);
}

__global__ void histogram_to_cdf_kernel(unsigned int* histogram, unsigned int* cdf_min) {
    __shared__ unsigned int scan[256];
    const int index = threadIdx.x;
    scan[index] = histogram[index];
    __syncthreads();
    for (int offset = 1; offset < 256; offset *= 2) {
        const unsigned int addition = index >= offset ? scan[index - offset] : 0U;
        __syncthreads();
        scan[index] += addition;
        __syncthreads();
    }
    histogram[index] = scan[index];
    if (index == 0) {
        unsigned int first = 0;
        for (int value = 0; value < 256; ++value) {
            if (scan[value] != 0) {
                first = scan[value];
                break;
            }
        }
        *cdf_min = first;
    }
}

__global__ void equalize_kernel(
    const unsigned char* input, unsigned char* output, int pixels,
    const unsigned int* cdf, const unsigned int* cdf_min_pointer) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixels) return;
    const unsigned int cdf_min = *cdf_min_pointer;
    if (cdf_min == static_cast<unsigned int>(pixels)) {
        output[index] = input[index];
        return;
    }
    const double mapped = static_cast<double>(cdf[input[index]] - cdf_min) * 255.0 /
                          static_cast<double>(pixels - cdf_min);
    output[index] = static_cast<unsigned char>(floor(mapped + 0.5));
}

std::vector<float> make_gaussian_kernel(int size, float sigma) {
    if ((size != 3 && size != 5 && size != 7) || !std::isfinite(sigma) || sigma < 0.1F || sigma > 10.0F) {
        throw std::invalid_argument("Gaussian Blur chỉ nhận bộ lọc 3/5/7 và sigma trong khoảng 0.1–10.0.");
    }
    const int radius = size / 2;
    const float denominator = 2.0F * sigma * sigma;
    std::vector<float> values(static_cast<std::size_t>(size * size));
    float sum = 0.0F;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            const float value = std::exp(-static_cast<float>(x * x + y * y) / denominator);
            values[(y + radius) * size + x + radius] = value;
            sum += value;
        }
    }
    for (float& value : values) value /= sum;
    return values;
}

std::vector<float> make_gaussian_kernel_1d(int size, float sigma) {
    if ((size != 3 && size != 5 && size != 7) || !std::isfinite(sigma) || sigma < 0.1F || sigma > 10.0F) {
        throw std::invalid_argument("Gaussian Blur chỉ nhận bộ lọc 3/5/7 và sigma trong khoảng 0.1–10.0.");
    }
    const int radius = size / 2;
    const float denominator = 2.0F * sigma * sigma;
    std::vector<float> values(static_cast<std::size_t>(size));
    float sum = 0.0F;
    for (int offset = -radius; offset <= radius; ++offset) {
        const float value = std::exp(-static_cast<float>(offset * offset) / denominator);
        values[static_cast<std::size_t>(offset + radius)] = value;
        sum += value;
    }
    for (float& value : values) value /= sum;
    return values;
}

}  // namespace

ProcessingResult process_cuda(
    const Image& input, Algorithm algorithm, const ProcessingParams& params, Backend backend) {
    ProcessingResult result;
    result.backend_used = backend;
    result.threads_used = 256;
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        result.error = ProcessingError::BackendUnavailable;
        result.error_message = device_status == cudaSuccess ? "Không tìm thấy GPU CUDA." : cudaGetErrorString(device_status);
        return result;
    }

    unsigned char* device_input = nullptr;
    unsigned char* device_output = nullptr;
    unsigned char* device_gray = nullptr;
    float* device_gaussian_intermediate = nullptr;
    float* device_weights = nullptr;
    unsigned int* device_histogram = nullptr;
    unsigned int* device_cdf_min = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    const auto total_start = std::chrono::steady_clock::now();
    try {
        if (algorithm == Algorithm::Sobel && (params.threshold < 0 || params.threshold > 255)) {
            throw std::invalid_argument("Sobel threshold phải nằm trong khoảng 0–255.");
        }
        std::vector<float> weights;
        if (algorithm == Algorithm::GaussianBlur) {
            weights = backend == Backend::CudaOptimized
                          ? make_gaussian_kernel_1d(params.kernel_size, params.sigma)
                          : make_gaussian_kernel(params.kernel_size, params.sigma);
        }
        const int pixels = input.width * input.height;
        const std::size_t input_bytes = input.pixels.size();
        const std::size_t output_bytes = algorithm == Algorithm::GaussianBlur ? input_bytes : static_cast<std::size_t>(pixels);
        const auto allocation_start = std::chrono::steady_clock::now();
        check_cuda(cudaMalloc(&device_input, input_bytes), "Cấp phát ảnh đầu vào");
        check_cuda(cudaMalloc(&device_output, output_bytes), "Cấp phát ảnh đầu ra");
        if (algorithm == Algorithm::GaussianBlur && backend == Backend::CudaOptimized) {
            check_cuda(
                cudaMalloc(&device_gaussian_intermediate, input_bytes * sizeof(float)),
                "Cấp phát bộ đệm Gaussian tách chiều");
        }
        if (algorithm != Algorithm::GaussianBlur) check_cuda(cudaMalloc(&device_gray, pixels), "Cấp phát ảnh xám");
        if (!weights.empty()) check_cuda(cudaMalloc(&device_weights, weights.size() * sizeof(float)), "Cấp phát bộ lọc");
        if (algorithm == Algorithm::HistogramEqualization) {
            check_cuda(cudaMalloc(&device_histogram, 256 * sizeof(unsigned int)), "Cấp phát histogram");
            check_cuda(cudaMalloc(&device_cdf_min, sizeof(unsigned int)), "Cấp phát CDF min");
        }
        result.timing.allocation_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - allocation_start).count();
        check_cuda(cudaEventCreate(&start), "Tạo CUDA event");
        check_cuda(cudaEventCreate(&stop), "Tạo CUDA event");

        check_cuda(cudaEventRecord(start), "Bắt đầu H2D");
        check_cuda(cudaMemcpy(device_input, input.pixels.data(), input_bytes, cudaMemcpyHostToDevice), "Chép ảnh lên GPU");
        if (!weights.empty()) check_cuda(cudaMemcpy(device_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice), "Chép bộ lọc lên GPU");
        check_cuda(cudaEventRecord(stop), "Kết thúc H2D");
        check_cuda(cudaEventSynchronize(stop), "Đợi H2D");
        float elapsed = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "Đo H2D");
        result.timing.h2d_ms = elapsed;

        check_cuda(cudaEventRecord(start), "Bắt đầu kernel");
        const dim3 block2d = backend == Backend::CudaOptimized ? dim3(32, 8) : dim3(16, 16);
        const dim3 grid2d((input.width + block2d.x - 1) / block2d.x, (input.height + block2d.y - 1) / block2d.y);
        const int blocks = (pixels + 255) / 256;
        if (algorithm == Algorithm::GaussianBlur) {
            if (backend == Backend::CudaOptimized) {
                gaussian_horizontal_kernel<<<grid2d, block2d>>>(
                    device_input, device_gaussian_intermediate, input.width, input.height, input.channels,
                    device_weights, params.kernel_size);
                gaussian_vertical_kernel<<<grid2d, block2d>>>(
                    device_gaussian_intermediate, device_output, input.width, input.height,
                    input.channels, device_weights, params.kernel_size);
            } else {
                gaussian_kernel<<<grid2d, block2d>>>(
                    device_input, device_output, input.width, input.height, input.channels,
                    device_weights, params.kernel_size);
            }
        } else {
            grayscale_kernel<<<blocks, 256>>>(device_input, device_gray, pixels, input.channels);
            if (algorithm == Algorithm::Sobel) {
                if (backend == Backend::CudaOptimized) {
                    const std::size_t shared_bytes =
                        static_cast<std::size_t>(block2d.x + 2) * (block2d.y + 2);
                    sobel_tiled_kernel<<<grid2d, block2d, shared_bytes>>>(
                        device_gray, device_output, input.width, input.height, params.threshold);
                } else {
                    sobel_kernel<<<grid2d, block2d>>>(
                        device_gray, device_output, input.width, input.height, params.threshold);
                }
            } else {
                check_cuda(cudaMemset(device_histogram, 0, 256 * sizeof(unsigned int)), "Xóa histogram");
                if (backend == Backend::CudaOptimized) histogram_optimized_kernel<<<std::min(blocks, 1024), 256>>>(device_gray, device_histogram, pixels);
                else histogram_basic_kernel<<<blocks, 256>>>(device_gray, device_histogram, pixels);
                histogram_to_cdf_kernel<<<1, 256>>>(device_histogram, device_cdf_min);
                equalize_kernel<<<blocks, 256>>>(
                    device_gray, device_output, pixels, device_histogram, device_cdf_min);
            }
        }
        check_cuda(cudaGetLastError(), "Chạy CUDA kernel");
        check_cuda(cudaEventRecord(stop), "Kết thúc kernel");
        check_cuda(cudaEventSynchronize(stop), "Đợi kernel");
        check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "Đo kernel");
        result.timing.kernel_ms = elapsed;

        result.output = Image{input.width, input.height, algorithm == Algorithm::GaussianBlur ? input.channels : 1, {}};
        result.output.pixels.resize(output_bytes);
        check_cuda(cudaEventRecord(start), "Bắt đầu D2H");
        check_cuda(cudaMemcpy(result.output.pixels.data(), device_output, output_bytes, cudaMemcpyDeviceToHost), "Chép kết quả về CPU");
        check_cuda(cudaEventRecord(stop), "Kết thúc D2H");
        check_cuda(cudaEventSynchronize(stop), "Đợi D2H");
        check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "Đo D2H");
        result.timing.d2h_ms = elapsed;
        result.timing.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total_start).count();
    } catch (const std::invalid_argument& error) {
        result.error = ProcessingError::InvalidParameters;
        result.error_message = error.what();
    } catch (const std::exception& error) {
        result.error = ProcessingError::InternalError;
        result.error_message = error.what();
    }
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    cudaFree(device_cdf_min);
    cudaFree(device_histogram);
    cudaFree(device_weights);
    cudaFree(device_gaussian_intermediate);
    cudaFree(device_gray);
    cudaFree(device_output);
    cudaFree(device_input);
    return result;
}

}  // namespace pip::cuda_backend
