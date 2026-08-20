from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

MAX_OPERATIONS = 5


# Enum
class AlgorithmEnum(str, Enum):
    GAUSSIAN_BLUR = "gaussian_blur"
    SOBEL = "sobel"
    HISTOGRAM_EQUALIZATION = "histogram_equalization"


class BackendEnum(str, Enum):
    SEQUENTIAL = "sequential"
    OPENMP = "openmp"
    CUDA_BASIC = "cuda_basic"
    CUDA_OPTIMIZED = "cuda_optimized"


# Params thuat toan
class GaussianBlurParams(BaseModel):
    kernel_size: Literal[3, 5, 7]
    sigma: float = Field(..., ge=0.1, le=10.0)

    model_config = {"extra": "forbid"}


class SobelParams(BaseModel):
    threshold: int = Field(..., ge=0, le=255)

    model_config = {"extra": "forbid"}


class HistogramEqualizationParams(BaseModel):
    model_config = {"extra": "forbid"}


# Operation
class GaussianBlurOperation(BaseModel):
    algorithm: Literal[AlgorithmEnum.GAUSSIAN_BLUR]
    backend: BackendEnum
    params: GaussianBlurParams

    model_config = {"extra": "forbid"}


class SobelOperation(BaseModel):
    algorithm: Literal[AlgorithmEnum.SOBEL]
    backend: BackendEnum
    params: SobelParams

    model_config = {"extra": "forbid"}


class HistogramEqualizationOperation(BaseModel):
    algorithm: Literal[AlgorithmEnum.HISTOGRAM_EQUALIZATION]
    backend: BackendEnum
    # cho phép thiếu params -> mặc định object rỗng
    params: HistogramEqualizationParams = HistogramEqualizationParams()

    model_config = {"extra": "forbid"}


Operation = Annotated[
    Union[
        GaussianBlurOperation,
        SobelOperation,
        HistogramEqualizationOperation,
    ],
    Field(discriminator="algorithm"),
]


class Pipeline(BaseModel):
    operations: List[Operation] = Field(
        ..., min_length=1, max_length=MAX_OPERATIONS
    )

    model_config = {"extra": "forbid"}

    @field_validator("operations")
    @classmethod
    def _check_not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("Pipeline phải có ít nhất 1 operation")
        return v


def validate_pipeline(raw: dict | str) -> tuple[Pipeline | None, str | None]:
    """
    Nhận JSON thô (dict hoặc chuỗi JSON), trả về (Pipeline, None) nếu hợp lệ
    hoặc (None, thong_bao_loi_than_thien) nếu sai.

    Đây là điểm chặn duy nhất trước khi dữ liệu được đưa xuống adapter
    gọi pip::process() - không JSON sai nào được lọt qua.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            return None, f"JSON không hợp lệ: {e}"

    try:
        pipeline = Pipeline.model_validate(raw)
        return pipeline, None
    except ValidationError as e:
        return None, "Pipeline không hợp lệ:\n" + "\n".join(_translate_errors(e))


# Field cuối cùng trong loc -> câu tiếng Việt (khớp key error type của Pydantic)
_FIELD_MESSAGES = {
    ("kernel_size", "literal_error"): "kernel_size chỉ được nhận giá trị 3, 5 hoặc 7.",
    ("sigma", "greater_than_equal"): "sigma phải lớn hơn hoặc bằng 0.1.",
    ("sigma", "less_than_equal"): "sigma phải nhỏ hơn hoặc bằng 10.0.",
    ("threshold", "greater_than_equal"): "threshold phải lớn hơn hoặc bằng 0.",
    ("threshold", "less_than_equal"): "threshold phải nhỏ hơn hoặc bằng 255.",
    ("backend", "enum"): "backend không hợp lệ (chỉ nhận sequential, openmp, cuda_basic, cuda_optimized).",
    ("algorithm", "union_tag_invalid"): "algorithm không hợp lệ (chỉ nhận gaussian_blur, sobel, histogram_equalization).",
    ("operations", "too_short"): f"Pipeline phải có ít nhất 1 operation.",
    ("operations", "too_long"): f"Pipeline không được vượt quá {MAX_OPERATIONS} operation.",
}


def _translate_errors(exc: ValidationError) -> list[str]:
    """Dịch từng lỗi Pydantic thành 1 dòng tiếng Việt kèm vị trí lỗi trong pipeline.

    Với các loại lỗi chưa được map sẵn trong _FIELD_MESSAGES, rơi về message
    gốc của Pydantic để không bao giờ nuốt mất thông tin lỗi.
    """
    lines = []
    for err in exc.errors():
        loc = err["loc"]
        field = loc[-1] if loc else ""
        key = (str(field), err["type"])

        # Vị trí lỗi trong pipeline: "operations -> 1 -> ..." -> "operation #2"
        position = ""
        if len(loc) >= 2 and loc[0] == "operations" and isinstance(loc[1], int):
            position = f"operation #{loc[1] + 1}: "

        if key == ("extra_hack", "extra_forbidden") or err["type"] == "extra_forbidden":
            message = f"có field không được phép: '{field}'."
        else:
            message = _FIELD_MESSAGES.get(key, err["msg"])

        lines.append(f"- {position}{message}")
    return lines


def pipeline_to_dicts(pipeline: Pipeline) -> list[dict]:
    return [op.model_dump(mode="json") for op in pipeline.operations]


if __name__ == "__main__":
    test_cases = [
        # (mô tả, JSON)
        ("Hợp lệ - Gaussian Blur", {
            "operations": [
                {"algorithm": "gaussian_blur", "backend": "sequential",
                 "params": {"kernel_size": 5, "sigma": 1.5}}
            ]
        }),
        ("Hợp lệ - Sobel + Histogram nối tiếp", {
            "operations": [
                {"algorithm": "sobel", "backend": "cuda_basic",
                 "params": {"threshold": 100}},
                {"algorithm": "histogram_equalization", "backend": "openmp",
                 "params": {}}
            ]
        }),
        ("Sai - kernel_size không hợp lệ", {
            "operations": [
                {"algorithm": "gaussian_blur", "backend": "sequential",
                 "params": {"kernel_size": 4, "sigma": 1.0}}
            ]
        }),
        ("Sai - sigma ngoài khoảng", {
            "operations": [
                {"algorithm": "gaussian_blur", "backend": "sequential",
                 "params": {"kernel_size": 3, "sigma": 20.0}}
            ]
        }),
        ("Sai - backend không tồn tại", {
            "operations": [
                {"algorithm": "sobel", "backend": "gpu_fast",
                 "params": {"threshold": 50}}
            ]
        }),
        ("Sai - vượt quá số operation cho phép", {
            "operations": [
                {"algorithm": "sobel", "backend": "sequential",
                 "params": {"threshold": 10}}
            ] * (MAX_OPERATIONS + 1)
        }),
        ("Sai - field thừa không nằm trong schema", {
            "operations": [
                {"algorithm": "sobel", "backend": "sequential",
                 "params": {"threshold": 10, "extra_hack": True}}
            ]
        }),
    ]

    for desc, payload in test_cases:
        pipeline, err = validate_pipeline(payload)
        status = "OK" if pipeline else "REJECT"
        print(f"[{status}] {desc}")
        if err:
            print(f"   -> {err}\n")
        else:
            print(f"   -> {pipeline_to_dicts(pipeline)}\n")
