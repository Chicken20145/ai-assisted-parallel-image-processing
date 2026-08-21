from __future__ import annotations

import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

try:
    from app.ai_parser import DEFAULT_MODEL, api_is_configured, parse_prompt
    from app.core_adapter import find_core_cli
    from app.pipeline_adapter import run_pipeline_step
    from app.pipeline_schema import MAX_OPERATIONS, validate_pipeline
except ModuleNotFoundError:  # Cho phép chạy trực tiếp: streamlit run app/app.py
    from ai_parser import DEFAULT_MODEL, api_is_configured, parse_prompt
    from core_adapter import find_core_cli
    from pipeline_adapter import run_pipeline_step
    from pipeline_schema import MAX_OPERATIONS, validate_pipeline


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "external" / "BSDS300" / "images"
RESULT_DIR = ROOT / "benchmarks" / "results"
RAW_RESULT = RESULT_DIR / "ui_raw_results.csv"
SUMMARY_RESULT = RESULT_DIR / "ui_summary.csv"
PLOT_DIR = RESULT_DIR / "ui_plots"

ALGORITHM_LABELS = {
    "gaussian_blur": "Làm mờ Gaussian",
    "sobel": "Tìm biên Sobel",
    "histogram_equalization": "Cân bằng histogram",
}
BACKEND_LABELS = {
    "sequential": "CPU tuần tự",
    "openmp": "CPU OpenMP",
    "cuda_basic": "CUDA cơ bản",
    "cuda_optimized": "CUDA tối ưu",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm"}


def dataset_images() -> list[Path]:
    if not DATASET_DIR.is_dir():
        return []
    return sorted(path for path in DATASET_DIR.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def image_to_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def operation_controls(prefix: str) -> dict:
    algorithm = st.selectbox(
        "Thuật toán",
        list(ALGORITHM_LABELS),
        format_func=ALGORITHM_LABELS.get,
        key=f"{prefix}_algorithm",
    )
    backend = st.selectbox(
        "Cách chạy",
        list(BACKEND_LABELS),
        format_func=BACKEND_LABELS.get,
        key=f"{prefix}_backend",
    )
    thread_count = 0
    if backend != "sequential":
        thread_count = int(
            st.number_input(
                "Số luồng",
                min_value=0,
                max_value=1024,
                value=0,
                help="Để 0 để hệ thống tự chọn.",
                key=f"{prefix}_threads",
            )
        )

    params: dict[str, float | int] = {}
    if algorithm == "gaussian_blur":
        params["kernel_size"] = st.select_slider(
            "Kích thước kernel", options=[3, 5, 7], value=5, key=f"{prefix}_kernel"
        )
        params["sigma"] = st.slider(
            "Sigma", 0.1, 10.0, 1.5, step=0.1, key=f"{prefix}_sigma"
        )
    elif algorithm == "sobel":
        params["threshold"] = st.slider(
            "Ngưỡng", 0, 255, 100, key=f"{prefix}_threshold"
        )

    return {
        "algorithm": algorithm,
        "backend": backend,
        "params": params,
        "thread_count": thread_count,
    }


def show_step_metrics(index: int, algorithm: str, response, sequential_timing) -> None:
    st.markdown(f"#### Bước {index}: {ALGORITHM_LABELS[algorithm]}")
    if response.fallback_happened:
        st.warning(
            f"Không chạy được {BACKEND_LABELS[response.requested_backend]}; "
            f"đã chuyển sang {BACKEND_LABELS[response.actual_backend]}."
        )
    metrics = st.columns(4)
    metrics[0].metric("Backend", BACKEND_LABELS[response.actual_backend])
    metrics[1].metric("Số luồng", response.threads_used)
    metrics[2].metric("Kernel", f"{response.timing['kernel_ms']:.3f} ms")
    metrics[3].metric("Tổng", f"{response.timing['total_ms']:.3f} ms")
    if sequential_timing and response.timing["kernel_ms"] > 0:
        speedup = sequential_timing["kernel_ms"] / response.timing["kernel_ms"]
        st.metric("Tăng tốc so với CPU tuần tự", f"{speedup:.2f} lần")


def render_single_image(core_cli: Path | None, images: list[Path]) -> None:
    st.subheader("Xử lý một ảnh")
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("#### 1. Chọn ảnh")
            source_options = ["Tải ảnh lên"]
            if images:
                source_options.append(f"Chọn trong BSDS300 ({len(images)} ảnh)")
            source = st.radio(
                "Nguồn ảnh",
                source_options,
                horizontal=True,
                key="single_image_source",
                label_visibility="collapsed",
            )
            input_image = None
            input_name = ""
            if source == "Tải ảnh lên":
                uploaded = st.file_uploader(
                    "Tải ảnh JPG, PNG hoặc BMP",
                    type=["jpg", "jpeg", "png", "bmp"],
                    key="single_upload",
                )
                if uploaded is not None:
                    input_image = Image.open(uploaded).convert("RGB")
                    input_name = uploaded.name
            else:
                selected = st.selectbox(
                    "Ảnh trong bộ dữ liệu",
                    images,
                    format_func=lambda path: path.name,
                    key="single_dataset_image",
                )
                with Image.open(selected) as opened:
                    opened.load()
                    input_image = opened.convert("RGB")
                input_name = selected.name

        with st.container(border=True):
            st.markdown("#### 2. Chọn cách xử lý")
            mode = st.radio(
                "Cách tạo yêu cầu",
                ["Một bước", "Nhiều bước", "Nhập bằng tiếng Việt"],
                horizontal=True,
                key="single_mode",
            )
            steps_config: list[dict] = []
            ai_prompt = ""
            selected_model = DEFAULT_MODEL
            if mode == "Một bước":
                steps_config = [operation_controls("single")]
            elif mode == "Nhiều bước":
                step_count = int(
                    st.number_input(
                        "Số bước",
                        min_value=1,
                        max_value=MAX_OPERATIONS,
                        value=2,
                        key="pipeline_step_count",
                    )
                )
                for index in range(step_count):
                    with st.expander(f"Bước {index + 1}", expanded=True):
                        steps_config.append(operation_controls(f"pipeline_{index}"))
            else:
                ai_prompt = st.text_area(
                    "Yêu cầu",
                    placeholder="Ví dụ: Làm mờ ảnh rồi tìm biên bằng OpenMP.",
                    key="ai_prompt",
                )
                selected_model = st.text_input("Model", value=DEFAULT_MODEL, key="ai_model")
                if not api_is_configured():
                    st.info("Chưa có OPENAI_API_KEY. Hai chế độ thủ công vẫn dùng được.")

            run_single = st.button(
                "Chạy và xem chỉ số",
                type="primary",
                use_container_width=True,
                key="run_single_image",
                disabled=core_cli is None,
            )

    with right:
        with st.container(border=True):
            st.markdown("#### Ảnh đầu vào")
            if input_image is None:
                st.info("Hãy tải ảnh hoặc chọn ảnh trong BSDS300.")
            else:
                st.image(input_image, caption=input_name, use_container_width=True)

    if not run_single:
        return
    if input_image is None:
        st.error("Chưa chọn ảnh.")
        return

    if mode == "Nhập bằng tiếng Việt":
        parsed = parse_prompt(ai_prompt, model=selected_model)
        pipeline = parsed.pipeline
        error = parsed.error
    else:
        pipeline, error = validate_pipeline({"operations": steps_config})
    if pipeline is None:
        st.error(error or "Yêu cầu không hợp lệ.")
        return

    current_image = input_image
    results = []
    for index, operation in enumerate(pipeline.operations, start=1):
        response = run_pipeline_step(
            current_image,
            operation.algorithm.value,
            operation.backend.value,
            operation.params.model_dump(),
            operation.thread_count,
        )
        if not response.ok:
            st.error(f"Bước {index} không chạy được: {response.friendly_error}")
            return
        reference_timing = None
        if response.actual_backend != "sequential":
            reference = run_pipeline_step(
                current_image,
                operation.algorithm.value,
                "sequential",
                operation.params.model_dump(),
            )
            if reference.ok:
                reference_timing = reference.timing
        results.append((index, operation.algorithm.value, response, reference_timing))
        current_image = response.output_image

    st.divider()
    result_col, metric_col = st.columns([1, 1.25], gap="large")
    with result_col:
        st.subheader("Ảnh kết quả")
        st.image(current_image, use_container_width=True)
        st.download_button(
            "Tải ảnh PNG",
            data=image_to_png(current_image),
            file_name="ket_qua.png",
            mime="image/png",
            use_container_width=True,
            key="download_single_result",
        )
    with metric_col:
        st.subheader("Chỉ số")
        for result in results:
            show_step_metrics(*result)


def run_full_benchmark(core_cli: Path, thread_count: int) -> tuple[bool, str]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_command = [
        sys.executable,
        str(ROOT / "scripts" / "run_benchmarks.py"),
        "--input-dir", str(DATASET_DIR),
        "--output", str(RAW_RESULT),
        "--backends", "sequential", "openmp",
        "--threads", str(thread_count),
        "--warmup", "1",
        "--runs", "3",
        "--core-cli", str(core_cli),
    ]
    benchmark = subprocess.run(
        benchmark_command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=7200,
        check=False,
    )
    if benchmark.returncode != 0:
        return False, benchmark.stderr or benchmark.stdout

    analyze_command = [
        sys.executable,
        str(ROOT / "scripts" / "analyze_benchmarks.py"),
        "--input", str(RAW_RESULT),
        "--summary", str(SUMMARY_RESULT),
        "--plots", str(PLOT_DIR),
    ]
    analysis = subprocess.run(
        analyze_command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    if analysis.returncode != 0:
        return False, analysis.stderr or analysis.stdout
    return True, benchmark.stdout


def show_benchmark_results() -> None:
    if not RAW_RESULT.is_file() or not SUMMARY_RESULT.is_file():
        return
    raw = pd.read_csv(RAW_RESULT)
    summary = pd.read_csv(SUMMARY_RESULT)
    sequential = raw.loc[raw["backend"] == "sequential", "kernel_ms"]
    openmp = raw.loc[raw["backend"] == "openmp", "kernel_ms"]
    speedup = summary.loc[summary["backend"] == "openmp", "speedup"]

    st.subheader("Kết quả")
    first = st.columns(4)
    first[0].metric("Ảnh đã chạy", raw["image_name"].nunique())
    first[1].metric("Số lần đo", len(raw))
    first[2].metric("CPU tuần tự", f"{sequential.mean():.3f} ms")
    first[3].metric("OpenMP", f"{openmp.mean():.3f} ms")
    second = st.columns(3)
    second[0].metric("Tăng tốc trung bình", f"{speedup.mean():.2f} lần")
    second[1].metric("Sai số MAE lớn nhất", f"{raw['mae'].max():.3f}")
    second[2].metric("Thông lượng OpenMP", f"{raw.loc[raw['backend'] == 'openmp', 'throughput_mpix_s'].mean():.2f} MP/s")

    table = (
        summary.groupby(["algorithm", "backend"], as_index=False)
        .agg(
            kernel_ms=("kernel_ms_mean", "mean"),
            total_ms=("total_ms_mean", "mean"),
            speedup=("speedup", "mean"),
            mae_max=("mae_max", "max"),
        )
        .rename(
            columns={
                "algorithm": "Thuật toán",
                "backend": "Backend",
                "kernel_ms": "Kernel trung bình (ms)",
                "total_ms": "Tổng trung bình (ms)",
                "speedup": "Tăng tốc",
                "mae_max": "MAE lớn nhất",
            }
        )
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    downloads = st.columns(2)
    downloads[0].download_button(
        "Tải kết quả chi tiết",
        RAW_RESULT.read_bytes(),
        file_name="benchmark_300_anh.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_raw_benchmark",
    )
    downloads[1].download_button(
        "Tải bảng tổng hợp",
        SUMMARY_RESULT.read_bytes(),
        file_name="benchmark_tong_hop.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_summary_benchmark",
    )


def render_benchmark(core_cli: Path | None, images: list[Path]) -> None:
    st.subheader("Benchmark 300 ảnh")
    st.write("Nút bên dưới chạy cả 3 thuật toán bằng CPU tuần tự và OpenMP, sau đó tự tính chỉ số.")
    status = st.columns(3)
    status[0].metric("Ảnh tìm thấy", f"{len(images)}/300")
    status[1].metric("Thuật toán", 3)
    status[2].metric("Backend", 2)

    if images:
        with st.expander(f"Danh sách {len(images)} ảnh BSDS300"):
            st.dataframe(
                pd.DataFrame({"STT": range(1, len(images) + 1), "Tên ảnh": [p.name for p in images]}),
                use_container_width=True,
                hide_index=True,
                height=320,
            )

    ready = len(images) == 300 and core_cli is not None
    if len(images) != 300:
        st.error("Chưa đủ 300 ảnh. Hãy chạy cell setup Colab trước.")
    if core_cli is None:
        st.error("Chưa tìm thấy core C++. Hãy chạy cell build trước.")

    thread_count = max(1, min(4, os.cpu_count() or 1))
    if st.button(
        "Chạy benchmark 300 ảnh và xem chỉ số",
        type="primary",
        use_container_width=True,
        key="run_full_benchmark",
        disabled=not ready,
    ):
        with st.spinner("Đang chạy 300 ảnh. Không đóng Colab hoặc ngắt runtime..."):
            ok, detail = run_full_benchmark(core_cli, thread_count)
        if ok:
            st.success("Đã chạy xong benchmark.")
        else:
            st.error("Benchmark không chạy được.")
            st.code(detail[-5000:] if detail else "Không có log.")

    show_benchmark_results()


def main() -> None:
    st.set_page_config(page_title="So sánh xử lý ảnh", page_icon="🖼️", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: #f6f7f9; color: #1f2937; }
        [data-testid="stHeader"] { background: #ffffff; border-bottom: 1px solid #e5e7eb; }
        .block-container { max-width: 1220px; padding-top: 2rem; padding-bottom: 3rem; }
        h1, h2, h3, h4 { color: #111827 !important; letter-spacing: 0 !important; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff; border-color: #e5e7eb !important; border-radius: 12px;
        }
        [data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px;
        }
        .stButton > button[kind="primary"] { background: #2563eb; border-color: #2563eb; }
        .stButton > button { border-radius: 8px; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("So sánh xử lý ảnh")
    st.caption("Xử lý một ảnh hoặc chạy benchmark toàn bộ BSDS300.")

    core = find_core_cli()
    available_images = dataset_images()
    if core is not None:
        st.success("Core C++ đã sẵn sàng.")
    else:
        st.warning("Chưa có core C++. Hãy chạy setup/build trước.")

    single_tab, benchmark_tab = st.tabs(["Xử lý một ảnh", "Benchmark 300 ảnh"])
    with single_tab:
        render_single_image(core, available_images)
    with benchmark_tab:
        render_benchmark(core, available_images)


if __name__ == "__main__":
    main()
