# Project Goals and Success Criteria

## Primary goal

Build a correct and measurable parallel image-processing application that demonstrates when and why OpenMP or CUDA improves performance over sequential CPU execution.

## Technical goals

1. Implement Gaussian Blur, Sobel Edge Detection, and Histogram Equalization.
2. Provide sequential CPU, OpenMP, basic CUDA, and optimized CUDA versions.
3. Optimize at least one convolution with shared memory.
4. Measure kernel, transfer, and end-to-end execution time separately.
5. Validate parallel output against the sequential reference using MAE or MSE.
6. Determine suitable OpenMP thread counts and CUDA block sizes experimentally.
7. Implement an Auto backend using benchmark-derived rules.
8. Translate natural-language requests into a validated JSON pipeline.

## Success criteria

- All three algorithms pass correctness tests on normal and boundary-sized images.
- Benchmarks cover 256x256, 512x512, Full HD, 2K, and 4K inputs.
- Each benchmark configuration is warmed up and measured at least 20 times.
- Reports include mean time, standard deviation, speedup, efficiency, and throughput.
- CUDA reports distinguish kernel time from data-transfer time.
- The interface displays input/output images, selected backend, and measured time.
- A clean checkout can be built and run from documented instructions.

## Explicit non-goals

- Real-time video processing in the core submission
- MPI or multi-GPU execution
- Training or fine-tuning an AI model
- Object/face recognition
- Medical diagnosis claims
- Cloud deployment or multi-user accounts
- More filters before the three core algorithms are complete

