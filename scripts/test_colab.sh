#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ ! -d build-colab ]]; then
  echo 'Chưa có build-colab. Hãy chạy scripts/build_colab.sh trước.' >&2
  exit 1
fi

ctest --test-dir build-colab --output-on-failure
./build-colab/parallel_image_processing
./build-colab/image_benchmark \
  --algorithm gaussian_blur \
  --backend sequential \
  --width 64 --height 48 --channels 3 \
  --kernel-size 5 --sigma 1.2 \
  --warmup 1 --runs 2 >/tmp/pip_colab_smoke_test.csv

test "$(wc -l </tmp/pip_colab_smoke_test.csv)" -eq 3
echo 'CTest, probe CUDA/OpenMP và benchmark smoke test trên Colab đều đạt.'
