#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${repo_root}/data/external/environment_colab.txt"
mkdir -p "$(dirname "${output}")"

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "os=$(grep '^PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '\"')"
  echo "cpu=$(lscpu | awk -F: '/Model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}')"
  echo "cpu_logical_processors=$(nproc)"
  echo "ram_gb=$(awk '/MemTotal/ {printf "%.2f", $2 / 1024 / 1024}' /proc/meminfo)"
  echo "python=$(python3 --version 2>&1)"
  echo "cmake=$(cmake --version | head -n 1)"
  echo "compiler=$(g++ --version | head -n 1)"
  echo "nvcc=$(nvcc --version | tail -n 1)"
  echo 'nvidia_smi_begin'
  nvidia-smi
  echo 'nvidia_smi_end'
} > "${output}"

echo "Colab environment written to ${output}"
