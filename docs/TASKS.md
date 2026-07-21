# Team Task Board

## Roles

- **Member A — Parallel computing:** algorithms, correctness, OpenMP, CUDA, optimization.
- **Member B — AI and application:** validated JSON pipeline, UI, backend integration.
- **Member C — Operations and evaluation:** datasets, benchmark execution, charts, documentation, demo coordination.

All code must be reviewed by at least one other member before merging.

## Milestone 0 — Foundation

- [x] Create repository structure and Git repository.
- [x] Add CMake project with OpenMP and CUDA probes.
- [ ] Confirm Release build on the local NVIDIA machine.
- [ ] Record CPU, GPU, RAM, OS, compiler, and CUDA versions.
- [ ] Add one small, license-compatible sample image.

## Milestone 1 — Sequential reference

- [ ] Define a shared image buffer and border-handling convention.
- [ ] Implement grayscale conversion.
- [ ] Implement sequential Gaussian Blur.
- [ ] Implement sequential Sobel Edge Detection.
- [ ] Implement sequential Histogram Equalization.
- [ ] Add correctness and edge-case tests.

**Exit condition:** all sequential algorithms produce verified reference output.

## Milestone 2 — OpenMP

- [ ] Implement the three OpenMP versions.
- [ ] Test 1, 2, 4, 8, and hardware-appropriate maximum threads.
- [ ] Compare static, dynamic, and guided scheduling where relevant.
- [ ] Remove data races and verify output against the reference.

**Exit condition:** correct OpenMP output with repeatable speedup data.

## Milestone 3 — CUDA

- [ ] Implement basic CUDA kernels for all three algorithms.
- [ ] Add CUDA error checking and event-based timing.
- [ ] Measure host-to-device, kernel, and device-to-host time separately.
- [ ] Optimize Gaussian/Sobel using shared memory.
- [ ] Optimize histogram using block-local shared histograms.
- [ ] Test 8x8, 16x16, 32x8, and 32x16 blocks.

**Exit condition:** correct basic and optimized CUDA versions with documented gains.

## Milestone 4 — Benchmark and analysis

- [ ] Prepare 256x256, 512x512, Full HD, 2K, and 4K inputs.
- [ ] Add warm-up runs and at least 20 measured repetitions.
- [ ] Export results to CSV.
- [ ] Calculate mean, standard deviation, speedup, efficiency, and throughput.
- [ ] Produce comparison charts.
- [ ] Derive rules for the Auto backend.

**Exit condition:** results are reproducible and every performance claim is supported by data.

## Milestone 5 — UI and AI assistance

- [ ] Upload and preview an image.
- [ ] Select sequential, OpenMP, CUDA, or Auto backend.
- [ ] Display output image, runtime, and speedup.
- [ ] Define a strict JSON schema for supported pipelines.
- [ ] Convert natural-language requests to schema-compliant JSON.
- [ ] Validate operation names and parameter ranges before execution.
- [ ] Make AI explanations use measured data only.

**Exit condition:** the full demo works without manual code changes.

## Milestone 6 — Submission

- [ ] Reproduce the build from a clean checkout.
- [ ] Finalize report, diagrams, tables, and limitations.
- [ ] Prepare slides and a five-to-seven-minute demo script.
- [ ] Record a backup demonstration video.
- [ ] Tag the final Git commit as `v1.0.0`.

## Optional work — only after all exit conditions pass

- [ ] Batch processing for a directory of images.
- [ ] A short offline video demonstration.
- [ ] Compare the local GPU with one Google Colab GPU as separate environments.

