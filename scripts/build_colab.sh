#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo 'Không tìm thấy nvcc. Hãy chọn Runtime > Change runtime type > GPU rồi chạy lại setup.' >&2
  exit 1
fi
if ! command -v cmake >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
  echo 'Thiếu CMake hoặc Ninja. Hãy chạy scripts/setup_colab.sh trước.' >&2
  exit 1
fi

cmake -S . -B build-colab -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-colab --parallel "$(nproc)"

echo 'Build Release trên Colab đã hoàn tất.'
