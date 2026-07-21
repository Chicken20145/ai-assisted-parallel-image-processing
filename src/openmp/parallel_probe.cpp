#include <omp.h>
#include "parallel_probe.hpp"

int openmp_thread_count() {
    int count = 1;
#pragma omp parallel
    {
#pragma omp single
        count = omp_get_num_threads();
    }
    return count;
}

