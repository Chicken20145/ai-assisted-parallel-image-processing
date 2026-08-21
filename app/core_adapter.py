from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ERROR_CODES = {
    "none": "None",
    "invalid_image": "InvalidImage",
    "invalid_parameters": "InvalidParameters",
    "backend_unavailable": "BackendUnavailable",
    "internal_error": "InternalError",
}


@dataclass
class ProcessingResult:
    ok: bool
    error_code: str
    output_image: Image.Image | None = None
    timing: dict[str, float] = field(default_factory=dict)
    backend_used: str = "sequential"
    threads_used: int = 1
    error_message: str = ""


def find_core_cli() -> Path | None:
    """Tìm executable build sẵn; có thể ghi đè bằng PIP_CORE_CLI."""
    configured = os.environ.get("PIP_CORE_CLI")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(
        [
            ROOT / "build" / "image_pipeline_cli.exe",
            ROOT / "build" / "image_pipeline_cli",
            ROOT / "build" / "Release" / "image_pipeline_cli.exe",
            ROOT / "build-colab" / "image_pipeline_cli",
        ]
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _failure(code: str, message: str, backend: str) -> ProcessingResult:
    return ProcessingResult(
        ok=False,
        error_code=code,
        backend_used=backend,
        error_message=message,
    )


def pip_process(
    image: Image.Image,
    algorithm: str,
    backend: str,
    params: dict,
    *,
    timeout_seconds: int = 120,
) -> ProcessingResult:
    """Gọi `pip::process()` qua image_pipeline_cli bằng PPM/PGM lossless."""
    if image is None or not isinstance(image, Image.Image) or min(image.size) <= 0:
        return _failure("InvalidImage", "Ảnh đầu vào không hợp lệ.", backend)

    executable = find_core_cli()
    if executable is None:
        return _failure(
            "InternalError",
            "Không tìm thấy image_pipeline_cli. Hãy build project trước khi chạy UI.",
            backend,
        )

    with tempfile.TemporaryDirectory(prefix="pip-ui-") as temp_dir:
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
            "--kernel-size", str(params.get("kernel_size", 3)),
            "--sigma", str(params.get("sigma", 1.0)),
            "--threshold", str(params.get("threshold", 100)),
            "--threads", str(params.get("thread_count", 0)),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return _failure("InternalError", f"Không thể chạy core C++: {error}", backend)

        try:
            payload = json.loads(completed.stdout.strip())
        except (json.JSONDecodeError, TypeError):
            detail = completed.stderr.strip() or completed.stdout.strip() or "không có output"
            return _failure("InternalError", f"Core C++ trả dữ liệu không hợp lệ: {detail}", backend)

        error_code = ERROR_CODES.get(payload.get("error_code"), "InternalError")
        ok = bool(payload.get("ok")) and completed.returncode == 0
        output_image = None
        if ok:
            try:
                # Giải mã từ RAM để Pillow không giữ file handle PGM/PPM trên Windows.
                with Image.open(BytesIO(output_path.read_bytes())) as decoded:
                    decoded.load()
                    output_image = decoded.copy()
            except (OSError, FileNotFoundError) as error:
                return _failure("InternalError", f"Không đọc được ảnh từ core C++: {error}", backend)

        timing = {
            key: float(payload.get("timing", {}).get(key, 0.0))
            for key in ("allocation_ms", "h2d_ms", "kernel_ms", "d2h_ms", "total_ms")
        }
        return ProcessingResult(
            ok=ok,
            error_code="None" if ok else error_code,
            output_image=output_image,
            timing=timing,
            backend_used=str(payload.get("backend_used", backend)),
            threads_used=int(payload.get("threads_used", 1)),
            error_message=str(payload.get("error_message", "")),
        )
