cmake_minimum_required(VERSION 3.24)
project(parallel_image_processing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_library(image_processing_core
    src/common/error_metrics.cpp
    src/common/processing_api.cpp
    src/cpu/grayscale.cpp
    src/cpu/gaussian.cpp
    src/cpu/sobel.cpp
    src/cpu/histogram_equalization.cpp
    src/openmp/openmp_utils.cpp
    src/openmp/grayscale_openmp.cpp
    src/openmp/gaussian_openmp.cpp
    src/openmp/sobel_openmp.cpp
    src/openmp/histogram_equalization_openmp.cpp
)
target_include_directories(image_processing_core PUBLIC include)
target_link_libraries(image_processing_core PUBLIC OpenMP::OpenMP_CXX)

add_executable(parallel_image_processing
    src/main.cpp
    src/openmp/parallel_probe.cpp
    src/cuda/cuda_probe.cu
)

target_include_directories(parallel_image_processing PRIVATE include)
target_link_libraries(parallel_image_processing PRIVATE image_processing_core OpenMP::OpenMP_CXX)
set_target_properties(parallel_image_processing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(image_benchmark src/benchmark_cli.cpp)
target_link_libraries(image_benchmark PRIVATE image_processing_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(cpu_algorithm_tests tests/test_cpu_algorithms.cpp)
    target_link_libraries(cpu_algorithm_tests PRIVATE image_processing_core)
    add_test(NAME cpu_algorithm_tests COMMAND cpu_algorithm_tests)

    add_executable(openmp_algorithm_tests tests/test_openmp_algorithms.cpp)
    target_link_libraries(openmp_algorithm_tests PRIVATE image_processing_core)
    add_test(NAME openmp_algorithm_tests COMMAND openmp_algorithm_tests)
endif()
