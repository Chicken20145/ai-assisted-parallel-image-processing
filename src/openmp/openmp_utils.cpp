#include "openmp_algorithms.hpp"

#include <omp.h>

#include <stdexcept>

namespace pip::openmp {

int effective_thread_count(int requested_threads) {
    if (requested_threads < 0) {
        throw std::invalid_argument("Số OpenMP thread không được âm.");
    }
    const int threads = requested_threads == 0 ? omp_get_max_threads() : requested_threads;
    constexpr int maximum_supported_threads = 1024;
    if (threads <= 0 || threads > maximum_supported_threads) {
        throw std::invalid_argument("Số OpenMP thread phải nằm trong khoảng 1–1024, hoặc bằng 0 để tự chọn.");
    }
    return threads;
}

}  // namespace pip::openmp
