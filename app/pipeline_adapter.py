from __future__ import annotations
from dataclasses import dataclass
from PIL import Image
from mock_adapter import ProcessingResult, pip_process

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
    tried = [backend]
    result = pip_process(image, algorithm, backend, params)
    if not result.ok and result.error_code == "BackendUnavailable":
        for fb in FALLBACK_CHAIN.get(backend, []):
            tried.append(fb)
            result = pip_process(image, algorithm, fb, params)
            if result.ok:
                backend = fb
                break
    return AdapterResponse(
        ok=result.ok,
        output_image=result.output_image,
        requested_backend=tried[0],
        actual_backend=backend,
        fallback_happened=len(tried) > 1,
        timing=result.timing,
        friendly_error=None if result.ok else FRIENDLY_ERRORS.get(result.error_code, "Lỗi không xác định."),
    )
