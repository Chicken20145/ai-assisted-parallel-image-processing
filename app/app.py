import streamlit as st
from PIL import Image

try:
    from app.pipeline_adapter import run_pipeline_step
    from app.pipeline_schema import MAX_OPERATIONS, validate_pipeline
except ModuleNotFoundError:  # Cho phép Streamlit chạy file trực tiếp từ thư mục app.
    from pipeline_adapter import run_pipeline_step
    from pipeline_schema import MAX_OPERATIONS, validate_pipeline

# ----------------------------------------------------------------------------
# Cấu hình trang + giao diện tuỳ chỉnh
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Pixel Lab · Manual Image Pipeline", layout="wide", page_icon="◧")

ALGO_LABELS = {
    "gaussian_blur": "Gaussian Blur",
    "sobel": "Sobel Edge Detection",
    "histogram_equalization": "Histogram Equalization",
}
ALGO_KEYS = list(ALGO_LABELS.keys())
BACKEND_LABELS = {
    "sequential": "CPU tuần tự",
    "openmp": "OpenMP",
    "cuda_basic": "CUDA cơ bản",
    "cuda_optimized": "CUDA tối ưu",
}
BACKEND_KEYS = list(BACKEND_LABELS.keys())

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --bp-bg: #0E1B2E;
    --bp-panel: #16263F;
    --bp-panel-2: #1C3050;
    --bp-grid: rgba(120, 170, 220, 0.08);
    --bp-line: rgba(120, 170, 220, 0.22);
    --bp-text: #E7EEF7;
    --bp-text-dim: #9AB0C9;
    --bp-cyan: #5FD8E8;
    --bp-amber: #F2A65A;
    --bp-mint: #6FCF97;
    --bp-coral: #E8615A;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: var(--bp-text); }

.stApp {
    background-color: var(--bp-bg);
    background-image:
        linear-gradient(var(--bp-grid) 1px, transparent 1px),
        linear-gradient(90deg, var(--bp-grid) 1px, transparent 1px);
    background-size: 28px 28px;
}

section[data-testid="stSidebar"] {
    background-color: var(--bp-panel);
    border-right: 1px solid var(--bp-line);
}

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.01em; }

h1 { color: var(--bp-text) !important; font-weight: 600 !important; }

.hero-sub { color: var(--bp-text-dim); font-size: 0.95rem; margin-top: -0.6rem; margin-bottom: 1.2rem; }

/* Badge kiểu "algorithm chip" - lấy cảm hứng từ chính tên thuật toán trong schema */
.algo-chip {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    padding: 3px 10px;
    border-radius: 5px;
    border: 1px solid var(--bp-line);
    background: var(--bp-panel-2);
    color: var(--bp-cyan);
    margin-right: 6px;
    margin-bottom: 6px;
}
.algo-chip.active { border-color: var(--bp-cyan); box-shadow: 0 0 0 1px var(--bp-cyan) inset; }
.algo-chip.dim { color: var(--bp-text-dim); }

.step-arrow { color: var(--bp-text-dim); font-family: 'IBM Plex Mono', monospace; margin: 0 4px; }

/* Buttons */
.stButton > button {
    background: var(--bp-cyan) !important;
    color: #0B1520 !important;
    border: none !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 6px !important;
}
.stButton > button:hover { background: #7FE3F0 !important; }

/* Panels quanh mỗi step */
div[data-testid="stExpander"] {
    background: var(--bp-panel);
    border: 1px solid var(--bp-line);
    border-radius: 8px;
}

hr { border-color: var(--bp-line) !important; }
</style>
""", unsafe_allow_html=True)

st.title("PIXEL LAB")
st.markdown('<div class="hero-sub">Manual Image Processing Pipeline — 3 thuật toán, 4 backend, 1 pipeline JSON</div>', unsafe_allow_html=True)
st.info(
    "Mốc hiện tại dùng mock adapter để kiểm thử UI/schema. Mock chỉ xử lý sequential; "
    "OpenMP/CUDA sẽ fallback cho tới khi adapter C++ thật được nối."
)


def algo_chip_row(active_keys=None, dim=True):
    active_keys = active_keys or []
    parts = []
    for i, key in enumerate(ALGO_KEYS):
        cls = "algo-chip active" if key in active_keys else ("algo-chip dim" if dim else "algo-chip")
        parts.append(f'<span class="{cls}">{key}</span>')
    st.markdown("".join(parts), unsafe_allow_html=True)


algo_chip_row()

# ----------------------------------------------------------------------------
# Sidebar: chọn chế độ + cấu hình
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Cấu hình")

    mode = st.radio(
        "Chế độ xử lý",
        ["Một thuật toán", "Ghép chuỗi nhiều bước"],
        help="Một thuật toán: chạy riêng 1 bước để xem kỹ kết quả/backend/timing. "
             "Ghép chuỗi: nối nhiều thuật toán liên tiếp trên cùng 1 ảnh.",
    )

    steps_config = []

    if mode == "Một thuật toán":
        algo_label = st.selectbox("Thuật toán", list(ALGO_LABELS.values()))
        algorithm = [k for k, v in ALGO_LABELS.items() if v == algo_label][0]
        backend = st.selectbox("Backend", BACKEND_KEYS, format_func=lambda k: BACKEND_LABELS[k])

        params = {}
        if algorithm == "gaussian_blur":
            params["kernel_size"] = st.select_slider("Kernel size", options=[3, 5, 7], value=5)
            params["sigma"] = st.slider("Sigma", 0.1, 10.0, 1.5, step=0.1)
        elif algorithm == "sobel":
            params["threshold"] = st.slider("Threshold", 0, 255, 100)

        steps_config = [{"algorithm": algorithm, "backend": backend, "params": params}]

    else:
        n_steps = st.number_input("Số bước", min_value=1, max_value=MAX_OPERATIONS, value=2, step=1)
        st.caption(f"Tối đa {MAX_OPERATIONS} bước / pipeline.")
        for i in range(int(n_steps)):
            with st.expander(f"Bước {i + 1}", expanded=True):
                algo_label = st.selectbox("Thuật toán", list(ALGO_LABELS.values()), key=f"algo_{i}")
                algorithm = [k for k, v in ALGO_LABELS.items() if v == algo_label][0]
                backend = st.selectbox("Backend", BACKEND_KEYS, format_func=lambda k: BACKEND_LABELS[k], key=f"backend_{i}")

                params = {}
                if algorithm == "gaussian_blur":
                    params["kernel_size"] = st.select_slider("Kernel size", options=[3, 5, 7], value=5, key=f"ks_{i}")
                    params["sigma"] = st.slider("Sigma", 0.1, 10.0, 1.5, step=0.1, key=f"sigma_{i}")
                elif algorithm == "sobel":
                    params["threshold"] = st.slider("Threshold", 0, 255, 100, key=f"thr_{i}")

                steps_config.append({"algorithm": algorithm, "backend": backend, "params": params})

    run = st.button("Chạy xử lý", use_container_width=True)

uploaded = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "bmp"])

col_input, col_output = st.columns(2)
input_image = None
if uploaded is not None:
    input_image = Image.open(uploaded).convert("RGB")
    with col_input:
        st.subheader("Ảnh gốc")
        st.image(input_image, use_container_width=True)
else:
    st.info("Vui lòng tải ảnh lên để bắt đầu.")

# ----------------------------------------------------------------------------
# Chạy pipeline
# ----------------------------------------------------------------------------
if run:
    if input_image is None:
        st.error("Chưa có ảnh đầu vào.")
    else:
        candidate = {"operations": steps_config}
        pipeline, err = validate_pipeline(candidate)

        if pipeline is None:
            st.error(f"Pipeline không hợp lệ, đã chặn trước khi gọi backend:\n\n{err}")
        else:
            algo_chip_row(active_keys=[op["algorithm"] for op in steps_config])

            current_image = input_image
            step_results = []
            failed = False

            for i, op in enumerate(pipeline.operations):
                response = run_pipeline_step(
                    current_image, op.algorithm.value, op.backend.value, op.params.model_dump()
                )
                step_results.append((op.algorithm.value, response))
                if not response.ok:
                    st.error(f"Bước {i + 1} ({op.algorithm.value}) thất bại: {response.friendly_error}")
                    failed = True
                    break
                current_image = response.output_image

            if not failed:
                with col_output:
                    st.subheader("Ảnh kết quả")
                    st.image(current_image, use_container_width=True)

                st.subheader("Chi tiết từng bước")
                for i, (algo_name, response) in enumerate(step_results):
                    with st.expander(f"Bước {i + 1}: {algo_name} → backend: {response.actual_backend}", expanded=(i == len(step_results) - 1)):
                        if response.fallback_happened:
                            st.warning(
                                f"Backend yêu cầu ({response.requested_backend}) chưa khả dụng "
                                f"→ tự động chuyển sang **{response.actual_backend}**."
                            )
                        else:
                            st.success(f"Chạy thành công trên backend: **{response.actual_backend}**")

                        t = response.timing
                        tcols = st.columns(5)
                        tcols[0].metric("Allocation (ms)", t["allocation_ms"])
                        tcols[1].metric("H2D (ms)", t["h2d_ms"])
                        tcols[2].metric("Kernel (ms)", t["kernel_ms"])
                        tcols[3].metric("D2H (ms)", t["d2h_ms"])
                        tcols[4].metric("Total (ms)", t["total_ms"])

                        if response.actual_backend == "sequential":
                            st.caption(
                                "Speedup sẽ hiển thị sau khi UI nối adapter C++ thật. "
                                "Core A đã có Sequential/OpenMP; mock UI hiện chỉ chạy sequential."
                            )
