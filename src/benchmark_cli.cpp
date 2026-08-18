#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "processing_api.hpp"

namespace {

struct Options {
    pip::Algorithm algorithm = pip::Algorithm::GaussianBlur;
    pip::Backend backend = pip::Backend::Sequential;
    pip::ProcessingParams params;
    int width = 512;
    int height = 512;
    int channels = 3;
    int threads = 0;
    int warmup = 3;
    int runs = 20;
};

int parse_int(const char* text, std::string_view name) {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != std::string(text).size()) {
        throw std::invalid_argument("Giá trị không hợp lệ cho " + std::string(name));
    }
    return value;
}

float parse_float(const char* text, std::string_view name) {
    std::size_t consumed = 0;
    const float value = std::stof(text, &consumed);
    if (consumed != std::string(text).size()) {
        throw std::invalid_argument("Giá trị không hợp lệ cho " + std::string(name));
    }
    return value;
}

pip::Algorithm parse_algorithm(std::string_view value) {
    if (value == "gaussian_blur") return pip::Algorithm::GaussianBlur;
    if (value == "sobel") return pip::Algorithm::Sobel;
    if (value == "histogram_equalization") return pip::Algorithm::HistogramEqualization;
    throw std::invalid_argument("Thuật toán phải là gaussian_blur, sobel hoặc histogram_equalization.");
}

pip::Backend parse_backend(std::string_view value) {
    if (value == "sequential") return pip::Backend::Sequential;
    if (value == "openmp") return pip::Backend::OpenMP;
    if (value == "cuda_basic") return pip::Backend::CudaBasic;
    if (value == "cuda_optimized") return pip::Backend::CudaOptimized;
    throw std::invalid_argument("Backend không hợp lệ.");
}

void print_help() {
    std::cout
        << "Synthetic benchmark cho lõi xử lý ảnh.\n\n"
        << "Tùy chọn:\n"
        << "  --algorithm gaussian_blur|sobel|histogram_equalization\n"
        << "  --backend sequential|openmp|cuda_basic|cuda_optimized\n"
        << "  --width N --height N --channels 1|3\n"
        << "  --threads N (0 = OpenMP runtime tự chọn)\n"
        << "  --kernel-size 3|5|7 --sigma 0.1..10 --threshold 0..255\n"
        << "  --warmup N --runs N\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        if (argument == "--help") {
            print_help();
            std::exit(EXIT_SUCCESS);
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("Thiếu giá trị sau " + std::string(argument));
        }
        const char* value = argv[++i];
        if (argument == "--algorithm") options.algorithm = parse_algorithm(value);
        else if (argument == "--backend") options.backend = parse_backend(value);
        else if (argument == "--width") options.width = parse_int(value, argument);
        else if (argument == "--height") options.height = parse_int(value, argument);
        else if (argument == "--channels") options.channels = parse_int(value, argument);
        else if (argument == "--threads") options.threads = parse_int(value, argument);
        else if (argument == "--kernel-size") options.params.kernel_size = parse_int(value, argument);
        else if (argument == "--sigma") options.params.sigma = parse_float(value, argument);
        else if (argument == "--threshold") options.params.threshold = parse_int(value, argument);
        else if (argument == "--warmup") options.warmup = parse_int(value, argument);
        else if (argument == "--runs") options.runs = parse_int(value, argument);
        else throw std::invalid_argument("Tùy chọn không xác định: " + std::string(argument));
    }
    if (options.width <= 0 || options.height <= 0 ||
        (options.channels != 1 && options.channels != 3) ||
        options.warmup < 0 || options.runs <= 0 || options.threads < 0) {
        throw std::invalid_argument("width/height/runs phải dương; channels là 1/3; warmup/threads không âm.");
    }
    options.params.thread_count = options.threads;
    return options;
}

pip::Image make_input(const Options& options) {
    pip::Image image{options.width, options.height, options.channels, {}};
    image.pixels.resize(image.expected_size());
    for (int y = 0; y < options.height; ++y) {
        for (int x = 0; x < options.width; ++x) {
            for (int channel = 0; channel < options.channels; ++channel) {
                const std::size_t index =
                    (static_cast<std::size_t>(y) * options.width + x) * options.channels + channel;
                image.pixels[index] = static_cast<std::uint8_t>((x * 17 + y * 31 + channel * 47) & 0xFF);
            }
        }
    }
    return image;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const pip::Image input = make_input(options);
        for (int run = 0; run < options.warmup; ++run) {
            const auto result = pip::process(input, options.algorithm, options.params, options.backend);
            if (!result.ok()) {
                std::cerr << "Warm-up thất bại: " << result.error_message << '\n';
                return EXIT_FAILURE;
            }
        }

        std::cout << "algorithm,image_width,image_height,channels,backend,threads,kernel_size,sigma,threshold,run,kernel_ms,total_ms\n";
        std::cout << std::fixed << std::setprecision(6);
        for (int run = 1; run <= options.runs; ++run) {
            const auto result = pip::process(input, options.algorithm, options.params, options.backend);
            if (!result.ok()) {
                std::cerr << "Benchmark thất bại: " << result.error_message << '\n';
                return EXIT_FAILURE;
            }
            std::cout << pip::to_string(options.algorithm) << ','
                      << options.width << ',' << options.height << ',' << options.channels << ','
                      << pip::to_string(result.backend_used) << ',' << result.threads_used << ','
                      << options.params.kernel_size << ','
                      << options.params.sigma << ',' << options.params.threshold << ',' << run << ','
                      << result.timing.kernel_ms << ',' << result.timing.total_ms << '\n';
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Lỗi tham số: " << error.what() << "\nDùng --help để xem hướng dẫn.\n";
        return EXIT_FAILURE;
    }
}
