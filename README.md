# AI-Assisted Parallel Image Processing Using OpenMP and CUDA

## Project objective

Design, implement, and evaluate a parallel image-processing system using sequential CPU, OpenMP, basic CUDA, and optimized CUDA implementations. An AI-assisted layer translates natural-language requests into a validated image-processing pipeline.

## Fixed scope

The core project contains exactly three algorithms:

1. Gaussian Blur
2. Sobel Edge Detection
3. Histogram Equalization

Each algorithm will have four implementations:

- Sequential CPU baseline
- OpenMP CPU version
- Basic CUDA version
- Optimized CUDA version

The project is complete when all implementations are correct, reproducible benchmarks are available, the UI works reliably, and AI can return a validated pipeline. Video, multi-GPU, MPI, model training, and cloud deployment are outside the core scope.

## Repository structure

```text
app/            User interface and AI pipeline integration
benchmarks/     Benchmark definitions and generated results
data/samples/   Small test images
docs/           Goals, task board, design and reports
include/        Shared C++/CUDA headers
scripts/        Build and benchmark helpers
src/cpu/        Sequential implementations
src/openmp/     OpenMP implementations
src/cuda/       CUDA implementations
tests/          Correctness and edge-case tests
```

## Initial build

Requirements: CMake, a C++ compiler, CUDA Toolkit, and an OpenMP-capable compiler.

```powershell
cmake -S . -B build
cmake --build build --config Release
./build/Release/parallel_image_processing.exe
```

On single-config generators, the executable may be located directly under `build/`.

## Evaluation metrics

- End-to-end execution time
- CUDA kernel time and host-device transfer time
- Speedup: `S(p) = T(1) / T(p)`
- OpenMP efficiency: `E(p) = S(p) / p`
- Throughput in megapixels per second
- MAE/MSE against the sequential reference

See `docs/GOALS.md` and `docs/TASKS.md` for the agreed scope and work plan.

