from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


METHOD_ORDER = ["dense", "file_fts", "graph_path", "fusion", "oracle"]
FORMAL_METHODS = ["dense", "file_fts", "graph_path", "fusion"]
TASK_ORDER = ["semantic_fact", "multi_hop_relation", "exact_file_lookup"]
TASK_LABELS = {
    "semantic_fact": "Semantic Fact",
    "multi_hop_relation": "Multi-hop Relation",
    "exact_file_lookup": "Exact/File Lookup",
}
METHOD_LABELS = {
    "dense": "Dense",
    "file_fts": "File-FTS",
    "graph_path": "Graph-Path",
    "fusion": "Fusion",
    "oracle": "Oracle",
}
METRIC_LABELS = {
    "evidence_recall": "Evidence Recall",
    "complete_evidence_recall": "Complete Evidence Recall",
    "mrr": "MRR",
    "search_success_rate": "Search Success Rate",
    "average_tool_calls": "Average Tool Calls",
    "latency_avg_ms": "Latency Avg (ms)",
    "latency_p95_ms": "Latency P95 (ms)",
    "evidence_gain_per_step": "Evidence Gain per Step",
    "stop_accuracy": "Stop Accuracy",
}
QUALITY_METRICS = [
    "evidence_recall",
    "complete_evidence_recall",
    "mrr",
    "search_success_rate",
]
COST_METRICS = ["average_tool_calls", "latency_avg_ms", "latency_p95_ms"]
ALL_METRICS = QUALITY_METRICS + COST_METRICS + ["evidence_gain_per_step", "stop_accuracy"]
PERCENT_METRICS = {
    "evidence_recall",
    "complete_evidence_recall",
    "search_success_rate",
    "stop_accuracy",
}
QUALITY_COLOR = "#3B6EA8"
METHOD_COLORS = {
    "dense": "#F58518",
    "file_fts": "#2F9C95",
    "graph_path": "#33691E",
    "fusion": "#E45756",
    "oracle": "#8C8C8C",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_clean_dir(path: Path) -> None:
    resolved = path.resolve()
    if not resolved.name.startswith("visualization"):
        raise ValueError(f"Refusing to clear non-visualization directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def ordered_methods(df: pd.DataFrame, include_oracle: bool = True) -> list[str]:
    allowed = METHOD_ORDER if include_oracle else FORMAL_METHODS
    present = set(df["method"].astype(str))
    return [method for method in allowed if method in present]


def method_labels(methods: Iterable[str]) -> list[str]:
    return [METHOD_LABELS.get(method, method) for method in methods]


def values_for_methods(df: pd.DataFrame, metric: str, methods: list[str]) -> list[float]:
    lookup = df.set_index("method")[metric].to_dict()
    return [float(lookup.get(method, np.nan)) for method in methods]


def prettify_metric_value(metric: str, value: float) -> str:
    if math.isnan(value):
        return "N/A"
    if metric in PERCENT_METRICS:
        return f"{value * 100:.1f}%"
    if metric.startswith("latency"):
        return f"{value:.0f}"
    if metric == "average_tool_calls":
        return f"{value:.1f}"
    return f"{value:.3f}"


def set_metric_axis(ax: plt.Axes, metric: str, values: list[float]) -> None:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return
    if metric in PERCENT_METRICS or metric == "mrr":
        ax.set_ylim(0, 1.08)
        ticks = np.linspace(0, 1.0, 6)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick * 100:.0f}%" for tick in ticks])
    else:
        top = max(clean) * 1.18 if max(clean) > 0 else 1.0
        ax.set_ylim(0, top)


def annotate_bars(ax: plt.Axes, bars, metric: str) -> None:
    for bar in bars:
        value = float(bar.get_height())
        if math.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            prettify_metric_value(metric, value),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def annotate_bars_by_metric(ax: plt.Axes, bars, metrics: list[str]) -> None:
    for bar, metric in zip(bars, metrics):
        value = float(bar.get_height())
        if math.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            prettify_metric_value(metric, value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def add_oracle_note(ax: plt.Axes) -> None:
    ax.text(
        0.99,
        -0.20,
        "Oracle is analysis-only upper-bound reference.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#666666",
    )


def plot_metric_bar(overall_k5: pd.DataFrame, metric: str, out_dir: Path) -> None:
    values_df = overall_k5.dropna(subset=[metric])
    if values_df.empty:
        return
    methods = ordered_methods(values_df)
    values = values_for_methods(values_df, metric, methods)
    colors = [METHOD_COLORS[method] for method in methods]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(method_labels(methods), values, color=colors, edgecolor="#222222", linewidth=0.5)
    if "oracle" in methods:
        bars[methods.index("oracle")].set_hatch("//")
    set_metric_axis(ax, metric, values)
    annotate_bars(ax, bars, metric)
    ax.set_title(f"Overall @5: {METRIC_LABELS[metric]}")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_xlabel("")
    ax.grid(axis="x", visible=False)
    if "oracle" in methods:
        add_oracle_note(ax)
    save_figure(fig, out_dir / "by_metric" / f"overall_at5_{metric}")


def plot_latency_grouped(overall_k5: pd.DataFrame, out_dir: Path) -> None:
    methods = ordered_methods(overall_k5)
    x = np.arange(len(methods))
    width = 0.36
    avg = values_for_methods(overall_k5, "latency_avg_ms", methods)
    p95 = values_for_methods(overall_k5, "latency_p95_ms", methods)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - width / 2, avg, width, label="Average", color="#4C78A8", edgecolor="#222222", linewidth=0.4)
    ax.bar(x + width / 2, p95, width, label="P95", color="#F58518", edgecolor="#222222", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels(methods))
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Overall @5: Average and P95 Latency")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper left")
    if "oracle" in methods:
        add_oracle_note(ax)
    save_figure(fig, out_dir / "overall" / "latency_avg_vs_p95")


def plot_overall_quality_grouped(overall_k5: pd.DataFrame, out_dir: Path) -> None:
    methods = ordered_methods(overall_k5)
    x = np.arange(len(methods))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(QUALITY_METRICS))
    colors = ["#4C78A8", "#E45756", "#54A24B", "#B279A2"]
    for metric, offset, color in zip(QUALITY_METRICS, offsets, colors):
        values = values_for_methods(overall_k5, metric, methods)
        ax.bar(
            x + offset,
            values,
            width,
            label=METRIC_LABELS[metric],
            color=color,
            edgecolor="#222222",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels(methods))
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_yticklabels([f"{tick * 100:.0f}%" for tick in np.linspace(0, 1.0, 6)])
    ax.set_ylabel("Score")
    ax.set_title("Overall @5: Quality Metrics by Baseline")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="lower right")
    if "oracle" in methods:
        add_oracle_note(ax)
    save_figure(fig, out_dir / "overall" / "overall_quality_metrics_grouped")


def plot_quality_cost_scatter(overall_k5: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for _, row in overall_k5.iterrows():
        method = str(row["method"])
        marker = "X" if method == "oracle" else "o"
        size = 95 if method == "oracle" else 80
        ax.scatter(
            float(row["latency_avg_ms"]),
            float(row["complete_evidence_recall"]),
            s=size,
            marker=marker,
            color=METHOD_COLORS.get(method, QUALITY_COLOR),
            edgecolor="#222222",
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            METHOD_LABELS.get(method, method),
            (float(row["latency_avg_ms"]), float(row["complete_evidence_recall"])),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xscale("symlog", linthresh=20)
    ax.set_ylim(0.70, 1.01)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Average latency (ms, symlog)")
    ax.set_ylabel("Complete Evidence Recall@5")
    ax.set_title("Quality-Cost Trade-off")
    save_figure(fig, out_dir / "overall" / "quality_cost_tradeoff_complete_recall_latency")


def plot_k_sensitivity(all_metrics: pd.DataFrame, out_dir: Path) -> None:
    overall = all_metrics[all_metrics["group_type"] == "overall"].copy()
    for metric in ["evidence_recall", "complete_evidence_recall", "mrr", "search_success_rate"]:
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        for method in ordered_methods(overall):
            subset = overall[overall["method"] == method].sort_values("k")
            ax.plot(
                subset["k"],
                subset[metric],
                marker="o",
                linewidth=2.0,
                markersize=5,
                color=METHOD_COLORS.get(method, QUALITY_COLOR),
                label=METHOD_LABELS.get(method, method),
                linestyle="--" if method == "oracle" else "-",
            )
        ax.set_xticks([1, 3, 5])
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.linspace(0, 1.0, 6))
        ax.set_yticklabels([f"{tick * 100:.0f}%" for tick in np.linspace(0, 1.0, 6)])
        ax.set_xlabel("k")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"@k Sensitivity: {METRIC_LABELS[metric]}")
        ax.legend(ncol=3, loc="lower right")
        save_figure(fig, out_dir / "k_sensitivity" / f"k_sensitivity_{metric}")


def plot_task_grouped(task_k5: pd.DataFrame, metric: str, out_dir: Path) -> None:
    methods = ordered_methods(task_k5)
    x = np.arange(len(TASK_ORDER))
    width = 0.12
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    offsets = np.linspace(-(len(methods) - 1) / 2 * width, (len(methods) - 1) / 2 * width, len(methods))
    for method, offset in zip(methods, offsets):
        values = []
        for task in TASK_ORDER:
            subset = task_k5[(task_k5["method"] == method) & (task_k5["task_type"] == task)]
            values.append(float(subset[metric].iloc[0]) if not subset.empty else np.nan)
        ax.bar(
            x + offset,
            values,
            width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, QUALITY_COLOR),
            edgecolor="#222222",
            linewidth=0.35,
            hatch="//" if method == "oracle" else None,
        )
    if metric in PERCENT_METRICS or metric == "mrr":
        ax.set_ylim(0, 1.08)
        ax.set_yticks(np.linspace(0, 1.0, 6))
        ax.set_yticklabels([f"{tick * 100:.0f}%" for tick in np.linspace(0, 1.0, 6)])
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[task] for task in TASK_ORDER])
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"Task-wise @5: {METRIC_LABELS[metric]}")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.30))
    save_figure(fig, out_dir / "by_task" / f"taskwise_at5_{metric}_grouped")


def plot_task_heatmap(task_k5: pd.DataFrame, metric: str, out_dir: Path) -> None:
    methods = ordered_methods(task_k5)
    matrix = np.full((len(methods), len(TASK_ORDER)), np.nan)
    for i, method in enumerate(methods):
        for j, task in enumerate(TASK_ORDER):
            subset = task_k5[(task_k5["method"] == method) & (task_k5["task_type"] == task)]
            if not subset.empty:
                matrix[i, j] = float(subset[metric].iloc[0])
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if metric in PERCENT_METRICS or metric == "mrr" else None)
    ax.set_xticks(np.arange(len(TASK_ORDER)))
    ax.set_xticklabels([TASK_LABELS[task] for task in TASK_ORDER], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(method_labels(methods))
    for i in range(len(methods)):
        for j in range(len(TASK_ORDER)):
            value = matrix[i, j]
            if math.isnan(value):
                label = "N/A"
            elif metric in PERCENT_METRICS:
                label = f"{value * 100:.1f}"
            else:
                label = f"{value:.3f}"
            ax.text(j, i, label, ha="center", va="center", color="#111111", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(METRIC_LABELS[metric])
    ax.set_title(f"Task x Baseline Heatmap: {METRIC_LABELS[metric]} @5")
    ax.grid(False)
    save_figure(fig, out_dir / "heatmaps" / f"task_method_heatmap_{metric}")


def plot_method_metric_heatmap(overall_k5: pd.DataFrame, out_dir: Path) -> None:
    metrics = QUALITY_METRICS + COST_METRICS
    methods = ordered_methods(overall_k5)
    matrix = np.full((len(methods), len(metrics)), np.nan)
    for i, method in enumerate(methods):
        row = overall_k5[overall_k5["method"] == method]
        if row.empty:
            continue
        for j, metric in enumerate(metrics):
            matrix[i, j] = float(row[metric].iloc[0])

    display = matrix.copy()
    for j, metric in enumerate(metrics):
        column = display[:, j]
        valid = ~np.isnan(column)
        if not valid.any():
            continue
        min_value = np.nanmin(column)
        max_value = np.nanmax(column)
        if max_value == min_value:
            display[valid, j] = 1.0
        elif metric in COST_METRICS:
            display[valid, j] = 1.0 - (column[valid] - min_value) / (max_value - min_value)
        else:
            display[valid, j] = (column[valid] - min_value) / (max_value - min_value)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    im = ax.imshow(display, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([METRIC_LABELS[metric] for metric in metrics], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(method_labels(methods))
    for i in range(len(methods)):
        for j, metric in enumerate(metrics):
            raw = matrix[i, j]
            label = prettify_metric_value(metric, raw) if not math.isnan(raw) else "N/A"
            ax.text(j, i, label, ha="center", va="center", fontsize=7.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Normalized desirability (higher is better; latency/calls inverted)")
    ax.set_title("Baseline Profile Matrix @5")
    ax.grid(False)
    save_figure(fig, out_dir / "method_profiles" / "method_metric_profile_heatmap")


def plot_individual_method_profiles(overall_k5: pd.DataFrame, task_k5: pd.DataFrame, out_dir: Path) -> None:
    for method in ordered_methods(overall_k5):
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
        row = overall_k5[overall_k5["method"] == method].iloc[0]
        quality_values = [float(row[metric]) for metric in QUALITY_METRICS]
        bars = axes[0].bar(
            [METRIC_LABELS[metric].replace(" ", "\n") for metric in QUALITY_METRICS],
            quality_values,
            color=[METHOD_COLORS.get(method, QUALITY_COLOR)] * len(QUALITY_METRICS),
            edgecolor="#222222",
            linewidth=0.4,
        )
        axes[0].set_ylim(0, 1.08)
        axes[0].set_yticks(np.linspace(0, 1.0, 6))
        axes[0].set_yticklabels([f"{tick * 100:.0f}%" for tick in np.linspace(0, 1.0, 6)])
        axes[0].set_title("Overall Quality @5")
        axes[0].grid(axis="x", visible=False)
        annotate_bars_by_metric(axes[0], bars, QUALITY_METRICS)

        task_values = []
        for task in TASK_ORDER:
            subset = task_k5[(task_k5["method"] == method) & (task_k5["task_type"] == task)]
            task_values.append(float(subset["complete_evidence_recall"].iloc[0]) if not subset.empty else np.nan)
        bars = axes[1].bar(
            [TASK_LABELS[task].replace(" ", "\n") for task in TASK_ORDER],
            task_values,
            color=[METHOD_COLORS.get(method, QUALITY_COLOR)] * len(TASK_ORDER),
            edgecolor="#222222",
            linewidth=0.4,
        )
        axes[1].set_ylim(0, 1.08)
        axes[1].set_yticks(np.linspace(0, 1.0, 6))
        axes[1].set_yticklabels([f"{tick * 100:.0f}%" for tick in np.linspace(0, 1.0, 6)])
        axes[1].set_title("Complete Evidence Recall@5 by Task")
        axes[1].grid(axis="x", visible=False)
        annotate_bars(axes[1], bars, "complete_evidence_recall")
        title_suffix = " (analysis-only)" if method == "oracle" else ""
        fig.suptitle(f"{METHOD_LABELS.get(method, method)} Profile{title_suffix}", y=1.02, fontsize=13)
        save_figure(fig, out_dir / "method_profiles" / f"profile_{method}")


def plot_failure_counts(failure_cases: pd.DataFrame, out_dir: Path) -> None:
    if failure_cases.empty:
        return
    counts = (
        failure_cases.groupby(["method", "task_type"])
        .size()
        .reset_index(name="failures")
    )
    methods = ordered_methods(counts)
    x = np.arange(len(methods))
    bottoms = np.zeros(len(methods))
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    task_colors = {
        "semantic_fact": "#4C78A8",
        "multi_hop_relation": "#E45756",
        "exact_file_lookup": "#54A24B",
    }
    for task in TASK_ORDER:
        values = []
        for method in methods:
            subset = counts[(counts["method"] == method) & (counts["task_type"] == task)]
            values.append(int(subset["failures"].iloc[0]) if not subset.empty else 0)
        ax.bar(
            x,
            values,
            bottom=bottoms,
            label=TASK_LABELS[task],
            color=task_colors[task],
            edgecolor="#222222",
            linewidth=0.35,
        )
        bottoms += np.array(values)
    for index, total in enumerate(bottoms):
        ax.text(index, total, f"{int(total)}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels(methods))
    ax.set_ylabel("Complete@5 failure count")
    ax.set_title("Failure Cases by Baseline and Task @5")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    save_figure(fig, out_dir / "failures" / "failure_counts_stacked_by_task")

    formal = failure_cases[failure_cases["method_role"] == "formal_baseline"]
    if not formal.empty:
        pivot = (
            formal.groupby(["method", "task_type"]).size().unstack(fill_value=0)
            .reindex(FORMAL_METHODS)
            .reindex(columns=TASK_ORDER)
            .fillna(0)
        )
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        matrix = pivot.to_numpy()
        im = ax.imshow(matrix, cmap="Reds")
        ax.set_xticks(np.arange(len(TASK_ORDER)))
        ax.set_xticklabels([TASK_LABELS[task] for task in TASK_ORDER], rotation=20, ha="right")
        ax.set_yticks(np.arange(len(FORMAL_METHODS)))
        ax.set_yticklabels(method_labels(FORMAL_METHODS))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Failure count")
        ax.set_title("Formal Baseline Failure Heatmap @5")
        ax.grid(False)
        save_figure(fig, out_dir / "failures" / "formal_failure_heatmap_by_task")


def plot_failure_rates(task_k5: pd.DataFrame, out_dir: Path) -> None:
    rows = task_k5.copy()
    rows["failure_rate"] = 1.0 - rows["complete_evidence_recall"].astype(float)
    methods = ordered_methods(rows)
    matrix = np.full((len(methods), len(TASK_ORDER)), np.nan)
    for i, method in enumerate(methods):
        for j, task in enumerate(TASK_ORDER):
            subset = rows[(rows["method"] == method) & (rows["task_type"] == task)]
            if not subset.empty:
                matrix[i, j] = float(subset["failure_rate"].iloc[0])
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    im = ax.imshow(matrix, cmap="OrRd", vmin=0, vmax=max(0.01, np.nanmax(matrix)))
    ax.set_xticks(np.arange(len(TASK_ORDER)))
    ax.set_xticklabels([TASK_LABELS[task] for task in TASK_ORDER], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(method_labels(methods))
    for i in range(len(methods)):
        for j in range(len(TASK_ORDER)):
            ax.text(j, i, f"{matrix[i, j] * 100:.1f}%", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Failure rate")
    ax.set_title("Complete@5 Failure Rate by Task")
    ax.grid(False)
    save_figure(fig, out_dir / "failures" / "failure_rate_heatmap_by_task")


def plot_formal_vs_oracle_gap(overall_k5: pd.DataFrame, task_k5: pd.DataFrame, out_dir: Path) -> None:
    oracle_overall = overall_k5[overall_k5["method"] == "oracle"]
    if oracle_overall.empty:
        return
    oracle_complete = float(oracle_overall["complete_evidence_recall"].iloc[0])
    formal = overall_k5[overall_k5["method"].isin(FORMAL_METHODS)].copy()
    formal["oracle_gap"] = oracle_complete - formal["complete_evidence_recall"].astype(float)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    methods = ordered_methods(formal, include_oracle=False)
    values = values_for_methods(formal, "oracle_gap", methods)
    bars = ax.bar(
        method_labels(methods),
        values,
        color="#B279A2",
        edgecolor="#222222",
        linewidth=0.4,
    )
    annotate_bars(ax, bars, "complete_evidence_recall")
    ax.set_ylabel("Oracle gap in Complete Recall@5")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Distance to Oracle Upper Bound")
    ax.grid(axis="x", visible=False)
    save_figure(fig, out_dir / "overall" / "formal_gap_to_oracle_complete_recall")

    oracle_task = task_k5[task_k5["method"] == "oracle"].set_index("task_type")["complete_evidence_recall"].to_dict()
    rows = []
    for _, row in task_k5[task_k5["method"].isin(FORMAL_METHODS)].iterrows():
        rows.append(
            {
                "method": row["method"],
                "task_type": row["task_type"],
                "gap": float(oracle_task[row["task_type"]]) - float(row["complete_evidence_recall"]),
            }
        )
    gap_df = pd.DataFrame(rows)
    methods = FORMAL_METHODS
    matrix = np.full((len(methods), len(TASK_ORDER)), np.nan)
    for i, method in enumerate(methods):
        for j, task in enumerate(TASK_ORDER):
            subset = gap_df[(gap_df["method"] == method) & (gap_df["task_type"] == task)]
            if not subset.empty:
                matrix[i, j] = float(subset["gap"].iloc[0])
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    im = ax.imshow(matrix, cmap="Purples", vmin=0, vmax=max(0.01, np.nanmax(matrix)))
    ax.set_xticks(np.arange(len(TASK_ORDER)))
    ax.set_xticklabels([TASK_LABELS[task] for task in TASK_ORDER], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(method_labels(methods))
    for i in range(len(methods)):
        for j in range(len(TASK_ORDER)):
            ax.text(j, i, f"{matrix[i, j] * 100:.1f}%", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Oracle gap")
    ax.set_title("Task-wise Gap to Oracle Complete@5")
    ax.grid(False)
    save_figure(fig, out_dir / "overall" / "taskwise_gap_to_oracle_complete_recall")


def plot_overview_dashboard(overall_k5: pd.DataFrame, task_k5: pd.DataFrame, out_dir: Path) -> None:
    methods = ordered_methods(overall_k5)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))

    ax = axes[0, 0]
    values = values_for_methods(overall_k5, "complete_evidence_recall", methods)
    bars = ax.bar(method_labels(methods), values, color=[METHOD_COLORS[m] for m in methods], edgecolor="#222222", linewidth=0.4)
    annotate_bars(ax, bars, "complete_evidence_recall")
    ax.set_ylim(0, 1.08)
    ax.set_title("Complete Evidence Recall@5")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", visible=False)

    ax = axes[0, 1]
    values = values_for_methods(overall_k5, "mrr", methods)
    bars = ax.bar(method_labels(methods), values, color=[METHOD_COLORS[m] for m in methods], edgecolor="#222222", linewidth=0.4)
    annotate_bars(ax, bars, "mrr")
    ax.set_ylim(0, 1.08)
    ax.set_title("MRR@5")
    ax.grid(axis="x", visible=False)

    ax = axes[1, 0]
    for method in ordered_methods(task_k5):
        subset = task_k5[task_k5["method"] == method].set_index("task_type").reindex(TASK_ORDER)
        ax.plot(
            [TASK_LABELS[task] for task in TASK_ORDER],
            subset["complete_evidence_recall"].astype(float).to_numpy(),
            marker="o",
            linewidth=2,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            linestyle="--" if method == "oracle" else "-",
        )
    ax.set_ylim(0, 1.05)
    ax.set_title("Task-wise Complete Recall@5")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=12)
    ax.legend(ncol=3, fontsize=8)

    ax = axes[1, 1]
    ax.scatter(
        overall_k5["latency_avg_ms"].astype(float),
        overall_k5["complete_evidence_recall"].astype(float),
        s=75,
        color=[METHOD_COLORS[m] for m in overall_k5["method"]],
        edgecolor="#222222",
        linewidth=0.5,
    )
    for _, row in overall_k5.iterrows():
        ax.annotate(
            METHOD_LABELS[str(row["method"])],
            (float(row["latency_avg_ms"]), float(row["complete_evidence_recall"])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("symlog", linthresh=20)
    ax.set_ylim(0.70, 1.01)
    ax.set_xlabel("Average latency (ms, symlog)")
    ax.set_ylabel("Complete Recall@5")
    ax.set_title("Quality-Cost Trade-off")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    fig.suptitle("Baseline Evaluation Summary on Frozen Final Dataset", fontsize=14, y=1.02)
    save_figure(fig, out_dir / "summary" / "baseline_evaluation_dashboard")


def write_readme(out_dir: Path, generated: list[Path]) -> None:
    rel_files = sorted(path.relative_to(out_dir).as_posix() for path in generated if path.suffix == ".png")
    lines = [
        "# Baseline Visualization Index",
        "",
        "本目录基于 `results/analysis/baselines` 中的真实 CSV 指标生成，不重新运行检索，也不读取 gold 以外的新信息。",
        "",
        "图表同时保存为 PNG 和 PDF：PNG 适合 PPT，PDF 适合论文或报告排版。",
        "",
        "## Directory Guide",
        "",
        "- `summary/`: 汇报总览图。",
        "- `overall/`: overall @5、质量-成本关系、Oracle gap。",
        "- `by_metric/`: 每个指标上不同 baseline 的柱状对比。",
        "- `by_task/`: 三类任务上的 grouped bar 对比。",
        "- `heatmaps/`: 方法 x 任务的指标热力图。",
        "- `method_profiles/`: 每个 baseline 的指标画像，以及整体 profile heatmap。",
        "- `k_sensitivity/`: @1/@3/@5 曲线。",
        "- `failures/`: 失败案例数量和失败率可视化。",
        "",
        "## Notes",
        "",
        "- Oracle 标记为 analysis-only upper-bound reference，不作为正式 baseline。",
        "- Evidence Gain per Step 和 Stop Accuracy 对当前单步 baseline 不适用，因此不会生成数值图。",
        "- 所有方法输出均为 Top-k retrieval output，不是答案生成预测。",
        "",
        "## PNG Files",
        "",
    ]
    lines.extend(f"- `{name}`" for name in rel_files)
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme(out_dir: Path, generated: list[Path]) -> None:
    rel_files = sorted(path.relative_to(out_dir).as_posix() for path in generated if path.suffix == ".png")
    lines = [
        "# Baseline Visualization Index",
        "",
        "This directory is generated from `results/analysis/baselines` CSV files. It does not rerun retrieval and does not read gold data beyond exported evaluation metrics.",
        "",
        "Charts are saved as PNG and PDF. PNG files are convenient for slides; PDF files are convenient for reports and papers.",
        "",
        "## Directory Guide",
        "",
        "- `summary/`: overview dashboard.",
        "- `overall/`: overall @5 and quality-cost charts.",
        "- `by_metric/`: one chart per metric comparing baselines and supplemental references.",
        "- `by_task/`: task-wise grouped comparisons.",
        "- `heatmaps/`: method x task metric heatmaps.",
        "- `method_profiles/`: per-method metric profiles.",
        "- `k_sensitivity/`: @1/@3/@5 curves.",
        "- `failures/`: failure counts and failure-rate visualizations.",
        "",
        "## Notes",
        "",
        "- Formal runnable baselines: `dense`, `file_fts`, `graph_path`, `fusion`.",
        "- `oracle` is an analysis-only upper-bound reference.",
        "- Evidence Gain per Step and Stop Accuracy are not applicable to these one-step baselines.",
        "- All outputs are Top-k retrieval outputs, not answer-generation predictions.",
        "",
        "## PNG Files",
        "",
    ]
    lines.extend(f"- `{name}`" for name in rel_files)
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_visualizations(baseline_dir: Path, out_dir: Path) -> list[Path]:
    all_metrics = pd.read_csv(baseline_dir / "baseline_metrics_all.csv")
    failure_cases = pd.read_csv(baseline_dir / "failure_cases_k5.csv")
    all_metrics = all_metrics[all_metrics["method"].isin(METHOD_ORDER)].copy()
    failure_cases = failure_cases[failure_cases["method"].isin(METHOD_ORDER)].copy()
    numeric_columns = [
        "k",
        "count",
        "evidence_recall",
        "complete_evidence_recall",
        "mrr",
        "search_success_rate",
        "average_tool_calls",
        "latency_avg_ms",
        "latency_p95_ms",
        "evidence_gain_per_step",
        "stop_accuracy",
    ]
    for column in numeric_columns:
        if column in all_metrics.columns:
            all_metrics[column] = pd.to_numeric(all_metrics[column], errors="coerce")

    overall_k5 = all_metrics[(all_metrics["group_type"] == "overall") & (all_metrics["k"] == 5)].copy()
    task_k5 = all_metrics[(all_metrics["group_type"] == "task_type") & (all_metrics["k"] == 5)].copy()
    overall_k5["method"] = pd.Categorical(overall_k5["method"], METHOD_ORDER, ordered=True)
    task_k5["method"] = pd.Categorical(task_k5["method"], METHOD_ORDER, ordered=True)
    task_k5["task_type"] = pd.Categorical(task_k5["task_type"], TASK_ORDER, ordered=True)
    overall_k5 = overall_k5.sort_values("method")
    task_k5 = task_k5.sort_values(["method", "task_type"])

    configure_style()
    ensure_clean_dir(out_dir)

    plot_overview_dashboard(overall_k5, task_k5, out_dir)
    plot_overall_quality_grouped(overall_k5, out_dir)
    plot_latency_grouped(overall_k5, out_dir)
    plot_quality_cost_scatter(overall_k5, out_dir)
    plot_formal_vs_oracle_gap(overall_k5, task_k5, out_dir)
    plot_k_sensitivity(all_metrics, out_dir)
    plot_method_metric_heatmap(overall_k5, out_dir)
    plot_individual_method_profiles(overall_k5, task_k5, out_dir)
    plot_failure_counts(failure_cases, out_dir)
    plot_failure_rates(task_k5, out_dir)

    for metric in ALL_METRICS:
        if metric in overall_k5.columns and overall_k5[metric].notna().any():
            plot_metric_bar(overall_k5, metric, out_dir)
    for metric in QUALITY_METRICS:
        plot_task_grouped(task_k5, metric, out_dir)
        plot_task_heatmap(task_k5, metric, out_dir)
    for metric in COST_METRICS:
        plot_task_grouped(task_k5, metric, out_dir)

    generated = sorted(out_dir.rglob("*"))
    write_readme(out_dir, generated)
    return sorted(path for path in out_dir.rglob("*") if path.is_file())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize baseline result CSV files.")
    parser.add_argument(
        "--baseline-dir",
        default="results/analysis/baselines",
        help="Directory containing baseline_metrics_*.csv and failure_cases_k5.csv.",
    )
    parser.add_argument(
        "--out-dir",
        default="figures/baselines/all_methods",
        help="Output visualization directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    baseline_dir = Path(args.baseline_dir)
    out_dir = Path(args.out_dir)
    files = build_visualizations(baseline_dir, out_dir)
    print(f"Wrote {len(files)} files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
