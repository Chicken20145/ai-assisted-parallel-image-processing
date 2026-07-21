#include <iostream>
#include "parallel_probe.hpp"

int main() {
    std::cout << "AI-Assisted Parallel Image Processing\n";
    std::cout << "OpenMP threads available: " << openmp_thread_count() << '\n';
    std::cout << "CUDA devices available: " << cuda_device_count() << '\n';
    return 0;
}

