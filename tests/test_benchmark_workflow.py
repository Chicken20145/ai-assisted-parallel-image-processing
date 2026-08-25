from __future__ import annotations

import pandas as pd

from scripts.analyze_benchmarks import build_summary
from scripts.run_benchmarks import (
    backend_configurations,
    error_metrics,
    exact_error_metrics,
    parse_thread_list,
)


def test_benchmark_configuration_matrix() -> None:
    assert parse_thread_list("4,1,2,2") == [1, 2, 4]
    assert backend_configurations(["sequential", "openmp"], [1, 2, 4]) == [
        ("sequential", 1), ("openmp", 1), ("openmp", 2), ("openmp", 4)
    ]


def test_summary_computes_speedup_and_efficiency() -> None:
    common = {
        "environment": "test", "image_name": "sample.png", "algorithm": "sobel",
        "image_width": 10, "image_height": 10, "channels": 3,
        "kernel_size": 3, "sigma": 1.0, "threshold": 0,
        "total_ms": 10.0, "throughput_mpix_s": 0.01, "mae": 0.0,
    }
    frame = pd.DataFrame(
        [
            {**common, "backend": "sequential", "threads": 1, "run": 1, "kernel_ms": 10.0},
            {**common, "backend": "sequential", "threads": 1, "run": 2, "kernel_ms": 10.0},
            {**common, "backend": "openmp", "threads": 2, "run": 1, "kernel_ms": 5.0},
            {**common, "backend": "openmp", "threads": 2, "run": 2, "kernel_ms": 5.0},
        ]
    )
    summary = build_summary(frame)
    openmp = summary.loc[summary["backend"] == "openmp"].iloc[0]
    assert openmp["speedup"] == 2.0
    assert openmp["efficiency"] == 1.0


def test_error_metrics_are_zero_for_identical_images() -> None:
    from PIL import Image

    image = Image.new("RGB", (3, 2), color=(10, 20, 30))
    assert error_metrics(image, image.copy()) == (0.0, 0.0, 0)
    exact = exact_error_metrics(image, image.copy())
    assert exact.mae_exact == "0/18"
    assert exact.mse_exact == "0/18"


def test_error_metrics_keep_exact_integer_totals() -> None:
    from PIL import Image

    reference = Image.new("L", (3, 1))
    reference.putdata([10, 20, 30])
    candidate = Image.new("L", (3, 1))
    candidate.putdata([11, 18, 33])

    exact = exact_error_metrics(reference, candidate)
    assert exact.absolute_sum == 6
    assert exact.squared_sum == 14
    assert exact.count == 3
    assert exact.maximum == 3
    assert exact.mae_exact == "6/3"
    assert exact.mse_exact == "14/3"
