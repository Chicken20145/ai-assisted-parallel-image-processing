#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo 'No NVIDIA GPU detected. In Colab choose Runtime > Change runtime type > GPU, then reconnect.' >&2
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo 'CUDA compiler nvcc is missing from this runtime.' >&2
  exit 1
fi

python3 -m pip install --quiet --upgrade 'cmake>=3.24' ninja
python3 -m pip install --quiet -r requirements.txt

python3 scripts/download_datasets.py
python3 scripts/prepare_benchmark_data.py

bash scripts/build_colab.sh
bash scripts/test_colab.sh

bash scripts/check_environment_colab.sh
echo 'Colab setup completed successfully.'
