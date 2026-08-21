#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "processing_api.hpp"

namespace {

struct Options {
    std::string input_path;
    std::string output_path;
    pip::Algorithm algorithm = pip::Algorithm::GaussianBlur;
    pip::Backend backend = pip::Backend::Sequential;
    pip::ProcessingParams params;
};

std::string json_escape(std::string_view value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const unsigned char character : value) {
        switch (character) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (character < 0x20) {
                    constexpr char digits[] = "0123456789abcdef";
                    escaped += "\\u00";
                    escaped += digits[(character >> 4) & 0x0F];
                    escaped += digits[character & 0x0F];
                } else {
                    escaped += static_cast<char>(character);
                }
        }
    }
    return escaped;
}

int parse_int(const char* text, std::string_view name) {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != std::string(text).size()) {
        throw std::invalid_argument("Invalid integer for " + std::string(name));
    }
    return value;
}

float parse_float(const char* text, std::string_view name) {
    std::size_t consumed = 0;
    const float value = std::stof(text, &consumed);
    if (consumed != std::string(text).size()) {
        throw std::invalid_argument("Invalid number for " + std::string(name));
    }
    return value;
}

pip::Algorithm parse_algorithm(std::string_view value) {
    if (value == "gaussian_blur") return pip::Algorithm::GaussianBlur;
    if (value == "sobel") return pip::Algorithm::Sobel;
    if (value == "histogram_equalization") return pip::Algorithm::HistogramEqualization;
    throw std::invalid_argument("Unsupported algorithm");
}

pip::Backend parse_backend(std::string_view value) {
    if (value == "sequential") return pip::Backend::Sequential;
    if (value == "openmp") return pip::Backend::OpenMP;
    if (value == "cuda_basic") return pip::Backend::CudaBasic;
    if (value == "cuda_optimized") return pip::Backend::CudaOptimized;
    throw std::invalid_argument("Unsupported backend");
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (index + 1 >= argc) {
            throw std::invalid_argument("Missing value after " + std::string(argument));
        }
        const char* value = argv[++index];
        if (argument == "--input") options.input_path = value;
        else if (argument == "--output") options.output_path = value;
        else if (argument == "--algorithm") options.algorithm = parse_algorithm(value);
        else if (argument == "--backend") options.backend = parse_backend(value);
        else if (argument == "--kernel-size") options.params.kernel_size = parse_int(value, argument);
        else if (argument == "--sigma") options.params.sigma = parse_float(value, argument);
        else if (argument == "--threshold") options.params.threshold = parse_int(value, argument);
        else if (argument == "--threads") options.params.thread_count = parse_int(value, argument);
        else throw std::invalid_argument("Unknown option: " + std::string(argument));
    }
    if (options.input_path.empty() || options.output_path.empty()) {
        throw std::invalid_argument("--input and --output are required");
    }
    if (options.params.thread_count < 0 || options.params.thread_count > 1024) {
        throw std::invalid_argument("threads must be in range 0..1024");
    }
    return options;
}

std::string read_token(std::istream& input) {
    std::string token;
    while (input >> std::ws && input.peek() == '#') {
        input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    if (!(input >> token)) {
        throw std::runtime_error("Unexpected end of PNM header");
    }
    return token;
}

pip::Image read_pnm(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open input image");

    const std::string magic = read_token(input);
    const int channels = magic == "P5" ? 1 : (magic == "P6" ? 3 : 0);
    if (channels == 0) throw std::runtime_error("Only binary P5/P6 PNM images are supported");
    const int width = parse_int(read_token(input).c_str(), "width");
    const int height = parse_int(read_token(input).c_str(), "height");
    const int max_value = parse_int(read_token(input).c_str(), "max value");
    if (width <= 0 || height <= 0 || max_value != 255) {
        throw std::runtime_error("Invalid PNM dimensions or max value");
    }
    const int separator = input.get();
    if (separator == std::char_traits<char>::eof()) {
        throw std::runtime_error("Missing PNM pixel data");
    }

    pip::Image image{width, height, channels, {}};
    image.pixels.resize(image.expected_size());
    input.read(reinterpret_cast<char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
    if (input.gcount() != static_cast<std::streamsize>(image.pixels.size())) {
        throw std::runtime_error("PNM pixel data is incomplete");
    }
    return image;
}

void write_pnm(const std::string& path, const pip::Image& image) {
    if (!image.is_valid()) throw std::runtime_error("Core returned an invalid image");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot open output image");
    output << (image.channels == 1 ? "P5\n" : "P6\n")
           << image.width << ' ' << image.height << "\n255\n";
    output.write(reinterpret_cast<const char*>(image.pixels.data()),
                 static_cast<std::streamsize>(image.pixels.size()));
    if (!output) throw std::runtime_error("Cannot write output image");
}

void print_result(const pip::ProcessingResult& result) {
    std::cout << std::fixed << std::setprecision(6)
              << "{\"ok\":" << (result.ok() ? "true" : "false")
              << ",\"error_code\":\"" << pip::to_string(result.error) << '"'
              << ",\"error_message\":\"" << json_escape(result.error_message) << '"'
              << ",\"backend_used\":\"" << pip::to_string(result.backend_used) << '"'
              << ",\"threads_used\":" << result.threads_used
              << ",\"timing\":{"
              << "\"allocation_ms\":" << result.timing.allocation_ms << ','
              << "\"h2d_ms\":" << result.timing.h2d_ms << ','
              << "\"kernel_ms\":" << result.timing.kernel_ms << ','
              << "\"d2h_ms\":" << result.timing.d2h_ms << ','
              << "\"total_ms\":" << result.timing.total_ms << "}}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const pip::Image input = read_pnm(options.input_path);
        const pip::ProcessingResult result = pip::process(
            input, options.algorithm, options.params, options.backend);
        if (result.ok()) write_pnm(options.output_path, result.output);
        print_result(result);
        return result.ok() ? EXIT_SUCCESS : 2;
    } catch (const std::exception& error) {
        pip::ProcessingResult result;
        result.error = pip::ProcessingError::InternalError;
        result.error_message = error.what();
        print_result(result);
        return EXIT_FAILURE;
    }
}
