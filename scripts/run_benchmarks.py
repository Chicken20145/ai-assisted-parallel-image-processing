from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core_adapter import find_core_cli  # noqa: E402


ALGORITHM_PARAMS = {
    "gaussian_blur": {"kernel_size": 5, "sigma": 1.5, "threshold": 100},
    "sobel": {"kernel_size": 3, "sigma": 1.0, "threshold": 0},
    "histogram_equalization": {"kernel_size": 3, "sigma": 1.0, "threshold": 100},
}
CSV_FIELDS = [
    "timestamp", "environment", "image_name", "algorithm", "image_width", "image_height",
    "channels", "requested_backend", "backend", "threads", "block_x", "block_y",
    "kernel_size", "sigma", "threshold", "run", "allocation_ms", "h2d_ms",
    "kernel_ms", "d2h_ms", "total_ms", "mae", "mse", "max_abs_error",
    "throughput_mpix_s",
]


def discover_images(input_dir: Path, max_images: int | None = None) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".pgm"}
    images = sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in extensions)
    if max_images is not None:
        images = images[:max_images]
    if not images:
        raise FileNotFoundError(f"Không tìm thấy ảnh benchmark trong {input_dir}")
    return images


def run_configuration(
    executable: Path,
    image: Image.Image,
    algorithm: str,
    backend: str,
    params: dict,
    threads: int,
    warmup: int,
    runs: int,
) -> tuple[dict, Image.Image]:
    with tempfile.TemporaryDirectory(prefix="pip-benchmark-") as temp_dir:
        input_path = Path(temp_dir) / "input.ppm"
        output_path = Path(temp_dir) / "output.pnm"
        normalized = image if image.mode == "L" else image.convert("RGB")
        normalized.save(input_path, format="PPM")
        command = [
            str(executable),
            "--input", str(input_path),
            "--output", str(output_path),
            "--algorithm", algorithm,
            "--backend", backend,
            "--kernel-size", str(params["kernel_size"]),
            "--sigma", str(params["sigma"]),
            "--threshold", str(params["threshold"]),
            "--threads", str(threads),
            "--warmup", str(warmup),
            "--runs", str(runs),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
            check=False,
        )
        try:
            payload = json.loads(completed.stdout.strip())
        except json.JSONDecodeError as error:
            raise RuntimeError(f"CLI trả JSON lỗi: {completed.stdout or completed.stderr}") from error
        if completed.returncode != 0 or not payload.get("ok"):
            raise RuntimeError(payload.get("error_message") or completed.stderr or "Core benchmark thất bại")
        if payload.get("backend_used") != backend:
            raise RuntimeError(
                f"Yêu cầu {backend} nhưng core dùng {payload.get('backend_used')}; "
                "benchmark chính thức không chấp nhận fallback."
            )
        timings = payload.get("timings")
        if timings is None and runs == 1 and isinstance(payload.get("timing"), dict):
            timings = [payload["timing"]]
            payload["timings"] = timings
        if not isinstance(timings, list) or len(timings) != runs:
            raise RuntimeError("Số timing core trả về không khớp --runs.")
        with Image.open(BytesIO(output_path.read_bytes())) as decoded:
            decoded.load()
            output_image = decoded.copy()
        return payload, output_image


def error_metrics(reference: Image.Image, candidate: Image.Image) -> tuple[float, float, int]:
    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if reference_array.shape != candidate_array.shape:
        raise ValueError(f"Sai shape: reference={reference_array.shape}, candidate={candidate_array.shape}")
    difference = np.abs(reference_array - candidate_array)
    return float(difference.mean()), float(np.square(difference).mean()), int(difference.max())


def backend_configurations(backends: list[str], threads: list[int]) -> list[tuple[str, int]]:
    configurations: list[tuple[str, int]] = []
    for backend in backends:
        if backend == "openmp":
            configurations.extend((backend, count) for count in threads)
        else:
            configurations.append((backend, 1 if backend == "sequential" else 0))
    return configurations


def parse_thread_list(raw: str) -> list[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or any(value < 1 or value > 1024 for value in values):
        raise argparse.ArgumentTypeError("--threads phải là danh sách số 1..1024, ví dụ 1,2,4,8")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark ảnh thật cho Sequential/OpenMP/CUDA.")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/external/benchmark_suite")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/raw_results.csv")
    parser.add_argument("--environment", default=platform.node() or platform.system())
    parser.add_argument(
        "--algorithms", nargs="+", choices=sorted(ALGORITHM_PARAMS), default=sorted(ALGORITHM_PARAMS)
    )
    parser.add_argument(
        "--backends", nargs="+",
        choices=["sequential", "openmp", "cuda_basic", "cuda_optimized"],
        default=["sequential", "openmp"],
    )
    parser.add_argument("--threads", type=parse_thread_list, default=parse_thread_list("1,2,4,8"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--core-cli", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup < 0 or args.runs <= 0:
        raise SystemExit("--warmup không âm và --runs phải dương.")
    executable = args.core_cli.resolve() if args.core_cli else find_core_cli()
    if executable is None or not executable.is_file():
        raise SystemExit("Không tìm thấy image_pipeline_cli; hãy build project trước.")
    images = discover_images(args.input_dir, args.max_images)
    configurations = backend_configurations(args.backends, args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for image_path in images:
            with Image.open(image_path) as opened:
                opened.load()
                image = opened.copy()
            normalized = image if image.mode == "L" else image.convert("RGB")
            width, height = normalized.size
            channels = 1 if normalized.mode == "L" else 3

            for algorithm in args.algorithms:
                params = ALGORITHM_PARAMS[algorithm]
                _, reference = run_configuration(
                    executable, normalized, algorithm, "sequential", params, 1, 0, 1
                )
                for backend, thread_count in configurations:
                    payload, candidate = run_configuration(
                        executable,
                        normalized,
                        algorithm,
                        backend,
                        params,
                        thread_count,
                        args.warmup,
                        args.runs,
                    )
                    mae, mse, max_error = error_metrics(reference, candidate)
                    for run_number, timing in enumerate(payload["timings"], start=1):
                        kernel_ms = float(timing["kernel_ms"])
                        writer.writerow(
                            {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "environment": args.environment,
                                "image_name": image_path.name,
                                "algorithm": algorithm,
                                "image_width": width,
                                "image_height": height,
                                "channels": channels,
                                "requested_backend": backend,
                                "backend": payload["backend_used"],
                                "threads": payload["threads_used"],
                                "block_x": "",
                                "block_y": "",
                                "kernel_size": params["kernel_size"],
                                "sigma": params["sigma"],
                                "threshold": params["threshold"],
                                "run": run_number,
                                **{key: timing[key] for key in (
                                    "allocation_ms", "h2d_ms", "kernel_ms", "d2h_ms", "total_ms"
                                )},
                                "mae": mae,
                                "mse": mse,
                                "max_abs_error": max_error,
                                "throughput_mpix_s": (
                                    width * height / (kernel_ms * 1000.0) if kernel_ms > 0 else ""
                                ),
                            }
                        )
                        row_count += 1
                    output_file.flush()
                    print(
                        f"[OK] {image_path.name} | {algorithm} | {backend} "
                        f"| threads={payload['threads_used']} | {args.runs} runs"
                    )
    print(f"Đã ghi {row_count} dòng vào {args.output}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
