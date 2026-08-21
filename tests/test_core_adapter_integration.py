from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core_adapter import find_core_cli, pip_process
from app.pipeline_adapter import run_pipeline_step


@pytest.fixture(scope="module", autouse=True)
def require_core_cli() -> Path:
    executable = find_core_cli()
    if executable is None:
        pytest.skip("Cần build image_pipeline_cli trước khi chạy integration test.")
    os.environ["PIP_CORE_CLI"] = str(executable)
    return executable


def sample_image() -> Image.Image:
    y, x = np.mgrid[0:17, 0:19]
    pixels = np.stack(
        ((x * 13 + y * 3) % 256, (x * 5 + y * 17) % 256, (x * 23 + y * 7) % 256),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


@pytest.mark.parametrize(
    ("algorithm", "params"),
    [
        ("gaussian_blur", {"kernel_size": 5, "sigma": 1.5}),
        ("sobel", {"threshold": 80}),
        ("histogram_equalization", {}),
    ],
)
def test_real_openmp_matches_real_sequential(algorithm: str, params: dict) -> None:
    image = sample_image()
    sequential = pip_process(image, algorithm, "sequential", params)
    openmp = pip_process(image, algorithm, "openmp", {**params, "thread_count": 2})

    assert sequential.ok, sequential.error_message
    assert openmp.ok, openmp.error_message
    assert sequential.backend_used == "sequential"
    assert openmp.backend_used == "openmp"
    assert openmp.threads_used == 2
    assert openmp.timing["kernel_ms"] >= 0
    np.testing.assert_array_equal(np.asarray(openmp.output_image), np.asarray(sequential.output_image))


def test_real_cuda_unavailable_falls_back_to_openmp() -> None:
    response = run_pipeline_step(
        sample_image(),
        "sobel",
        "cuda_optimized",
        {"threshold": 0},
        thread_count=2,
    )
    assert response.ok, response.friendly_error
    assert response.requested_backend == "cuda_optimized"
    assert response.actual_backend == "openmp"
    assert response.fallback_happened
    assert response.threads_used == 2


def test_real_core_error_is_mapped_for_python() -> None:
    result = pip_process(
        sample_image(),
        "gaussian_blur",
        "sequential",
        {"kernel_size": 4, "sigma": 1.0},
    )
    assert not result.ok
    assert result.error_code == "InvalidParameters"
    assert result.error_message
