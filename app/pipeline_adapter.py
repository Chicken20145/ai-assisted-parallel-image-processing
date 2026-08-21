from __future__ import annotations
from dataclasses import dataclass
from PIL import Image

try:
    from app.mock_adapter import pip_process
except ModuleNotFoundError:  # Cho phép `streamlit run app/app.py` từ thư mục project.
    from mock_adapter import pip_process

FALLBACK_CHAIN = {
    "cuda_optimized": ["cuda_basic", "openmp", "sequential"],
    "cuda_basic": ["openmp", "sequential"],
    "openmp": ["sequential"],
    "sequential": [],
}

FRIENDLY_ERRORS = {
    "InvalidImage": "Ảnh đầu vào không hợp lệ (sai định dạng hoặc rỗng).",
    "InvalidParameters": "Tham số xử lý không hợp lệ.",
    "BackendUnavailable": "Backend được chọn hiện chưa khả dụng.",
    "InternalError": "Có lỗi nội bộ xảy ra khi xử lý ảnh.",
}

@dataclass
class AdapterResponse:
    ok: bool
    output_image: Image.Image | None
    requested_backend: str
    actual_backend: str
    fallback_happened: bool
    timing: dict
    friendly_error: str | None

def run_pipeline_step(image, algorithm, backend, params):
    requested_backend = backend
    attempted_backends = [backend, *FALLBACK_CHAIN.get(backend, [])]
    actual_backend = backend
    result = None
    for candidate in attempted_backends:
        actual_backend = candidate
        result = pip_process(image, algorithm, candidate, params)
        if result.ok or result.error_code != "BackendUnavailable":
            break

    assert result is not None
    return AdapterResponse(
        ok=result.ok,
        output_image=result.output_image,
        requested_backend=requested_backend,
        actual_backend=actual_backend,
        fallback_happened=actual_backend != requested_backend,
        timing=result.timing,
        friendly_error=None if result.ok else FRIENDLY_ERRORS.get(result.error_code, "Lỗi không xác định."),
    )
