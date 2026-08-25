from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GROUP_KEYS = [
    "environment", "image_name", "algorithm", "image_width", "image_height", "channels",
    "backend", "threads", "kernel_size", "sigma", "threshold",
]
MATCH_KEYS = [
    "environment", "image_name", "algorithm", "image_width", "image_height", "channels",
    "kernel_size", "sigma", "threshold",
]


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(GROUP_KEYS) | {"run", "kernel_ms", "total_ms", "throughput_mpix_s", "mae"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CSV thiếu cột: {', '.join(missing)}")
    metric_aggregations = {
        "mae_max": ("mae", "max"),
        "mse_max": ("mse", "max") if "mse" in frame else ("mae", "max"),
        "max_abs_error": ("max_abs_error", "max") if "max_abs_error" in frame else ("mae", "max"),
    }
    if "backend_abs_error_sum" in frame:
        metric_aggregations.update(
            backend_abs_error_sum=("backend_abs_error_sum", "max"),
            backend_squared_error_sum=("backend_squared_error_sum", "max"),
            backend_compared_values=("backend_compared_values", "max"),
            backend_max_abs_error=("backend_max_abs_error", "max"),
            input_abs_change_sum=("input_abs_change_sum", "max"),
            input_squared_change_sum=("input_squared_change_sum", "max"),
            input_compared_values=("input_compared_values", "max"),
            input_change_max_abs_error=("input_change_max_abs_error", "max"),
        )
    summary = (
        frame.groupby(GROUP_KEYS, dropna=False)
        .agg(
            runs=("run", "count"),
            kernel_ms_mean=("kernel_ms", "mean"),
            kernel_ms_std=("kernel_ms", "std"),
            total_ms_mean=("total_ms", "mean"),
            total_ms_std=("total_ms", "std"),
            throughput_mpix_s_mean=("throughput_mpix_s", "mean"),
            **metric_aggregations,
        )
        .reset_index()
    )
    baseline = (
        summary.loc[summary["backend"] == "sequential", MATCH_KEYS + ["kernel_ms_mean"]]
        .rename(columns={"kernel_ms_mean": "sequential_kernel_ms"})
        .drop_duplicates(MATCH_KEYS)
    )
    if baseline.empty:
        raise ValueError("CSV phải có backend sequential để tính speedup.")
    summary = summary.merge(baseline, on=MATCH_KEYS, how="left", validate="many_to_one")
    if summary["sequential_kernel_ms"].isna().any():
        raise ValueError("Thiếu baseline sequential tương ứng cho một số cấu hình.")
    summary["speedup"] = summary["sequential_kernel_ms"] / summary["kernel_ms_mean"]
    summary["efficiency"] = np.where(
        summary["backend"] == "openmp",
        summary["speedup"] / summary["threads"],
        1.0,
    )
    return summary.sort_values(GROUP_KEYS).reset_index(drop=True)


def write_speedup_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    openmp = summary.loc[summary["backend"] == "openmp"]
    for algorithm, group in openmp.groupby("algorithm"):
        plotted = group.groupby("threads", as_index=False).agg(speedup_median=("speedup", "median"))
        figure, axis = plt.subplots(figsize=(7, 4.5))
        for thread_count, measurements in group.groupby("threads"):
            axis.scatter(
                np.full(len(measurements), thread_count), measurements["speedup"],
                alpha=0.28, s=20, color="tab:blue",
                label="Từng ảnh đo được" if thread_count == group["threads"].min() else None,
            )
        axis.plot(
            plotted["threads"], plotted["speedup_median"], marker="o",
            color="tab:blue", label="Trung vị đo được",
        )
        axis.plot(plotted["threads"], plotted["threads"], "--", label="Speedup lý tưởng")
        axis.set(title=f"OpenMP speedup – {algorithm}", xlabel="Số luồng", ylabel="Speedup (×)")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"speedup_{algorithm}.png", dpi=160)
        plt.close(figure)

    cuda = summary.loc[summary["backend"].isin(["cuda_basic", "cuda_optimized"])]
    backend_order = ["sequential", "cuda_basic", "cuda_optimized"]
    for algorithm in sorted(cuda["algorithm"].unique()):
        group = summary.loc[
            (summary["algorithm"] == algorithm) & summary["backend"].isin(backend_order)
        ].copy()
        if group.empty:
            continue
        figure, axis = plt.subplots(figsize=(7, 4.5))
        positions = {backend: index for index, backend in enumerate(backend_order)}
        medians: list[float] = []
        present: list[str] = []
        for backend in backend_order:
            measurements = group.loc[group["backend"] == backend, "speedup"]
            if measurements.empty:
                continue
            position = positions[backend]
            axis.scatter(
                np.full(len(measurements), position), measurements,
                alpha=0.28, s=20, color="tab:blue",
                label="Từng ảnh đo được" if not present else None,
            )
            present.append(backend)
            medians.append(float(measurements.median()))
        axis.plot([positions[name] for name in present], medians, marker="o", label="Trung vị đo được")
        axis.axhline(1.0, color="gray", linestyle="--", label="CPU tuần tự (1×)")
        axis.set_xticks([positions[name] for name in present], present)
        axis.set_yscale("log")
        axis.set(title=f"CUDA speedup – {algorithm}", xlabel="Backend", ylabel="Speedup (×)")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"speedup_cuda_{algorithm}.png", dpi=160)
        plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tổng hợp CSV và vẽ speedup OpenMP.")
    parser.add_argument("--input", type=Path, default=ROOT / "benchmarks/results/raw_results.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "benchmarks/results/summary.csv")
    parser.add_argument("--plots", type=Path, default=ROOT / "benchmarks/results/plots")
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.input)
    summary = build_summary(frame)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    write_speedup_plots(summary, args.plots)
    print(f"Đã ghi summary: {args.summary}")
    print(f"Đã ghi biểu đồ: {args.plots}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
