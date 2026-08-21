from __future__ import annotations
from dataclasses import dataclass
from PIL import Image

try:
    from app.core_adapter import pip_process
except ModuleNotFoundError:  # Cho phép `streamlit run app/app.py` từ thư mục project.
    from core_adapter import pip_process

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
    threads_used: int
    friendly_error: str | None
    technical_error: str | None

def run_pipeline_step(image, algorithm, backend, params, thread_count=0):
    requested_backend = backend
    attempted_backends = [backend, *FALLBACK_CHAIN.get(backend, [])]
    actual_backend = backend
    result = None
    call_params = {**params, "thread_count": thread_count}
    for candidate in attempted_backends:
        actual_backend = candidate
        result = pip_process(image, algorithm, candidate, call_params)
        if result.ok or result.error_code != "BackendUnavailable":
            break

    assert result is not None
    actual_backend = result.backend_used
    friendly_error = None
    if not result.ok:
        friendly_error = FRIENDLY_ERRORS.get(result.error_code, "Lỗi không xác định.")
        if result.error_message:
            friendly_error = f"{friendly_error} {result.error_message}"
    return AdapterResponse(
        ok=result.ok,
        output_image=result.output_image,
        requested_backend=requested_backend,
        actual_backend=actual_backend,
        fallback_happened=actual_backend != requested_backend,
        timing=result.timing,
        threads_used=result.threads_used,
        friendly_error=friendly_error,
        technical_error=None if result.ok else result.error_message,
    )
