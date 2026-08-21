from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from app.mock_adapter import pip_process
from app.pipeline_adapter import run_pipeline_step
from app.pipeline_schema import MAX_OPERATIONS, validate_pipeline
from app.prompt_test import TEST_PROMPTS, summary


ROOT = Path(__file__).resolve().parents[1]


def _reference_sobel(image: Image.Image, threshold: int) -> np.ndarray:
    gray = np.array(image.convert("L"), dtype=np.uint8)
    output = np.zeros_like(gray)
    gx_kernel = ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))
    gy_kernel = ((-1, -2, -1), (0, 0, 0), (1, 2, 1))
    height, width = gray.shape
    for y in range(height):
        for x in range(width):
            gx = 0
            gy = 0
            for ky in range(-1, 2):
                source_y = min(max(y + ky, 0), height - 1)
                for kx in range(-1, 2):
                    source_x = min(max(x + kx, 0), width - 1)
                    value = int(gray[source_y, source_x])
                    gx += value * gx_kernel[ky + 1][kx + 1]
                    gy += value * gy_kernel[ky + 1][kx + 1]
            magnitude = min(math.floor(math.sqrt(gx * gx + gy * gy) + 0.5), 255)
            output[y, x] = magnitude if threshold == 0 else (255 if magnitude >= threshold else 0)
    return output


def test_schema_accepts_valid_pipeline_and_rejects_invalid_fields() -> None:
    valid = {
        "operations": [
            {
                "algorithm": "gaussian_blur",
                "backend": "openmp",
                "params": {"kernel_size": 5, "sigma": 1.5},
            },
            {"algorithm": "histogram_equalization", "backend": "sequential"},
        ]
    }
    pipeline, error = validate_pipeline(valid)
    assert pipeline is not None and error is None

    invalid = {
        "operations": [
            {
                "algorithm": "gaussian_blur",
                "backend": "sequential",
                "params": {"kernel_size": 4, "sigma": 1.0, "admin_override": True},
            }
        ]
    }
    pipeline, error = validate_pipeline(invalid)
    assert pipeline is None
    assert error is not None and "Pipeline không hợp lệ" in error


def test_schema_limits_pipeline_length() -> None:
    operation = {
        "algorithm": "sobel",
        "backend": "sequential",
        "params": {"threshold": 100},
    }
    pipeline, error = validate_pipeline({"operations": [operation] * (MAX_OPERATIONS + 1)})
    assert pipeline is None
    assert error is not None


def test_numpy_sobel_matches_core_rules_without_scipy() -> None:
    pixels = np.array(
        [
            [[10, 30, 90], [20, 40, 100], [30, 50, 110]],
            [[40, 60, 120], [50, 70, 130], [60, 80, 140]],
            [[70, 90, 150], [80, 100, 160], [90, 110, 170]],
        ],
        dtype=np.uint8,
    )
    image = Image.fromarray(pixels, mode="RGB")
    for threshold in (0, 100):
        result = pip_process(image, "sobel", "sequential", {"threshold": threshold})
        assert result.ok
        np.testing.assert_array_equal(np.array(result.output_image), _reference_sobel(image, threshold))


def test_backend_unavailable_falls_back_to_sequential() -> None:
    image = Image.new("RGB", (5, 5), color=(20, 40, 60))
    response = run_pipeline_step(image, "sobel", "openmp", {"threshold": 0})
    assert response.ok
    assert response.requested_backend == "openmp"
    assert response.actual_backend == "sequential"
    assert response.fallback_happened


def test_invalid_image_does_not_trigger_backend_fallback() -> None:
    response = run_pipeline_step(None, "sobel", "openmp", {"threshold": 0})
    assert not response.ok
    assert response.actual_backend == "openmp"
    assert not response.fallback_happened
    assert response.friendly_error


def test_prompt_corpus_is_complete_and_has_unique_ids() -> None:
    assert len(TEST_PROMPTS) >= 20
    assert len({case["id"] for case in TEST_PROMPTS}) == len(TEST_PROMPTS)
    assert summary() == {
        "valid": 6,
        "ambiguous": 5,
        "out_of_scope": 5,
        "injection": 5,
        "unaccented": 5,
    }


def test_streamlit_app_renders_without_exception() -> None:
    app = AppTest.from_file(str(ROOT / "app" / "app.py")).run(timeout=15)
    assert not app.exception
    assert app.title[0].value == "PIXEL LAB"
