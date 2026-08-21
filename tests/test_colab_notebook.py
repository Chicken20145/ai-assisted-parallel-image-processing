from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_colab_notebook_contains_streamlit_proxy_launcher() -> None:
    notebook = json.loads((ROOT / "notebooks" / "colab_setup.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert notebook["nbformat"] == 4
    assert "ngrok.connect(8501, bind_tls=True)" in source
    assert "userdata.get('NGROK_AUTHTOKEN')" in source
    assert "build-colab' / 'image_pipeline_cli" in source
    assert "127.0.0.1:8501/_stcore/health" in source
    assert "Mở Pixel Lab Streamlit UI" in source

    launcher = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "ngrok.connect(8501, bind_tls=True)" in "".join(cell.get("source", []))
    )
    compile(launcher, "colab_streamlit_launcher", "exec")
