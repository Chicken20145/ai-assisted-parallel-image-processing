from __future__ import annotations

import time
import sys
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter, ImageOps

IMPLEMENTED_BACKENDS = {"sequential"}


@dataclass
class ProcessingResult:
    """Tương ứng pip::ProcessingResult bên C++."""
    ok: bool
    error_code: str  # None, InvalidImage, InvalidParameters, BackendUnavailable, InternalError
    output_image: Image.Image | None = None
    timing: dict = field(default_factory=dict)  # allocation_ms, h2d_ms, kernel_ms, d2h_ms, total_ms


def _run_gaussian_blur(img: Image.Image, params: dict) -> Image.Image:
    # Dùng PIL GaussianBlur để có kết quả demo trực quan (không phải core CUDA thật,
    # công thức radius không khớp kernel Gaussian toán học của spec)
    radius = params["kernel_size"] / 2
    return img.filter(ImageFilter.GaussianBlur(radius=radius * (params["sigma"] / 2)))


def _run_sobel(img: Image.Image, params: dict) -> Image.Image:
    gray = np.array(img.convert("L"), dtype=np.float32)
    padded = np.pad(gray, pad_width=1, mode="edge")
    top_left = padded[:-2, :-2]
    top = padded[:-2, 1:-1]
    top_right = padded[:-2, 2:]
    middle_left = padded[1:-1, :-2]
    middle_right = padded[1:-1, 2:]
    bottom_left = padded[2:, :-2]
    bottom = padded[2:, 1:-1]
    bottom_right = padded[2:, 2:]

    gx = -top_left + top_right - 2 * middle_left + 2 * middle_right - bottom_left + bottom_right
    gy = -top_left - 2 * top - top_right + bottom_left + 2 * bottom + bottom_right
    # Magnitude không âm nên floor(x + 0.5) tương đương std::lround của core C++.
    magnitude = np.clip(np.floor(np.sqrt(gx ** 2 + gy ** 2) + 0.5), 0, 255)

    threshold = params["threshold"]
    if threshold > 0:
        # Spec: magnitude >= threshold -> 255, ngược lại 0 (dùng >=, không phải >)
        magnitude = np.where(magnitude >= threshold, 255, 0)

    return Image.fromarray(magnitude.astype(np.uint8))


def _run_histogram_equalization(img: Image.Image, params: dict) -> Image.Image:
    return ImageOps.equalize(img.convert("L"))


_ALGORITHM_FUNCS = {
    "gaussian_blur": _run_gaussian_blur,
    "sobel": _run_sobel,
    "histogram_equalization": _run_histogram_equalization,
}


def _is_valid_image(image) -> bool:
    """Tương ứng input.is_valid() bên C++ - kiểm tra tối thiểu trước khi xử lý."""
    if image is None or not isinstance(image, Image.Image):
        return False
    width, height = image.size
    return width > 0 and height > 0


def pip_process(image: Image.Image, algorithm: str, backend: str, params: dict) -> ProcessingResult:
    """
    Chữ ký tương ứng pip::process(input, Algorithm, params, Backend) bên C++.
    Adapter thật (adapter.py) chỉ gọi hàm này - không quan tâm bên trong
    là mock hay C++ thật.
    """
    if not _is_valid_image(image):
        return ProcessingResult(ok=False, error_code="InvalidImage")

    if backend not in IMPLEMENTED_BACKENDS:
        # Đúng hành vi thật hiện tại: backend chưa triển khai -> BackendUnavailable
        return ProcessingResult(ok=False, error_code="BackendUnavailable")

    t_start = time.perf_counter()
    try:
        output = _ALGORITHM_FUNCS[algorithm](image, params)
    except (KeyError, TypeError, ValueError):
        return ProcessingResult(ok=False, error_code="InternalError")
    t_end = time.perf_counter()

    kernel_ms = (t_end - t_start) * 1000
    # CPU sequential: kernel_ms == total_ms, không có transfer H2D/D2H (đúng ghi chú tài liệu)
    timing = {
        "allocation_ms": 0.0,
        "h2d_ms": 0.0,
        "kernel_ms": round(kernel_ms, 3),
        "d2h_ms": 0.0,
        "total_ms": round(kernel_ms, 3),
    }
    return ProcessingResult(ok=True, error_code="None", output_image=output, timing=timing)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    img = Image.new("L", (4, 4), color=100)
    img.putpixel((2, 2), 200)

    r1 = pip_process(img, "sobel", "sequential", {"threshold": 0})
    arr = np.array(r1.output_image)
    test_val = int(arr[1, 2])

    r2 = pip_process(img, "sobel", "sequential", {"threshold": test_val})
    arr2 = np.array(r2.output_image)
    assert arr2[1, 2] == 255, "Lỗi: threshold == magnitude phải cho ra 255 (>=)"
    print(f"[OK] threshold == magnitude ({test_val}) -> 255 đúng spec '>='")

    r3 = pip_process(img, "sobel", "openmp", {"threshold": 0})
    assert r3.error_code == "BackendUnavailable"
    print(f"[OK] backend chưa hỗ trợ -> BackendUnavailable")

    r4 = pip_process(None, "sobel", "sequential", {"threshold": 0})
    assert r4.error_code == "InvalidImage"
    print(f"[OK] image=None -> InvalidImage (không còn bị nhầm InternalError)")

    r5 = pip_process(img, "gaussian_blur", "sequential", {"kernel_size": 5, "sigma": 1.5})
    assert r5.ok and r5.timing["kernel_ms"] == r5.timing["total_ms"]
    print(f"[OK] gaussian_blur chạy được, timing.kernel_ms == timing.total_ms (đúng CPU sequential)")

    r6 = pip_process(img, "histogram_equalization", "sequential", {})
    assert r6.ok
    print(f"[OK] histogram_equalization chạy được")
