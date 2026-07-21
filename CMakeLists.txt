cmake_minimum_required(VERSION 3.24)
project(parallel_image_processing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(parallel_image_processing
    src/main.cpp
    src/openmp/parallel_probe.cpp
    src/cuda/cuda_probe.cu
)

target_include_directories(parallel_image_processing PRIVATE include)
target_link_libraries(parallel_image_processing PRIVATE OpenMP::OpenMP_CXX)
set_target_properties(parallel_image_processing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

