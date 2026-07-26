from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "final"
OUT_DIR = ROOT / "figures" / "agent"
MAIN_RESULTS = RESULTS_DIR / "main_results.csv"

METHOD_ORDER = [
    "dense",
    "file_fts",
    "graph_path",
    "rule_router",
    "fusion",
    "oracle",
    "two_stage_agent",
]
METHOD_LABEL = {
    "dense": "Dense",
    "file_fts": "File FTS",
    "graph_path": "Graph Path",
    "rule_router": "Rule Router",
    "fusion": "Fusion",
    "oracle": "Oracle",
    "two_stage_agent": "Agent",
}
TASK_ORDER = ["semantic_fact", "multi_hop_relation", "exact_file_lookup"]
TASK_LABEL = {
    "semantic_fact": "Semantic Fact",
    "multi_hop_relation": "Multi-hop Relation",
    "exact_file_lookup": "Exact File Lookup",
}
METRIC_LABEL = {
    "evidence_recall_at_k": "Evidence Recall",
    "complete_recall_at_k": "Complete Evidence Recall",
    "search_success": "Search Success",
    "mrr": "MRR",
    "tool_calls": "Avg. Tool Calls",
    "latency_ms": "Avg. Latency (ms)",
    "latency_ms_p95": "P95 Latency (ms)",
    "text_read_tokens": "Text Read Tokens",
    "evidence_gain_step2": "Evidence Gain per Step",
    "stop_accuracy": "Stop Accuracy",
}
COLORS = {
    "dense": "#4C78A8",
    "file_fts": "#54A24B",
    "graph_path": "#F58518",
    "rule_router": "#7E57C2",
    "fusion": "#8C8C8C",
    "oracle": "#222222",
    "two_stage_agent": "#D62728",
}


def main() -> None:
    configure_style()
    clean_output_dirs()
    df = load_results()

    create_overview_figures(df)
    create_task_breakdown_figures(df)
    create_cost_efficiency_figures(df)
    create_agent_analysis_figures(df)
    create_table_figures(df)
    create_index(df)

    print(f"Visualizations written to: {OUT_DIR}")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.2,
            "figure.titlesize": 15,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_output_dirs() -> None:
    for name in [
        "00_overview",
        "01_task_breakdown",
        "02_cost_efficiency",
        "03_agent_analysis",
        "04_tables",
    ]:
        path = OUT_DIR / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def load_results() -> pd.DataFrame:
    df = pd.read_csv(MAIN_RESULTS)
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    numeric_cols = [
        "k",
        "count",
        "evidence_recall_at_k",
        "complete_recall_at_k",
        "search_success",
        "mrr",
        "tool_calls",
        "latency_ms",
        "latency_ms_p95",
        "text_read_tokens",
        "evidence_gain_step2",
        "stop_accuracy",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["method_label"] = df["method"].map(METHOD_LABEL)
    df["task_label"] = df["group_name"].map(TASK_LABEL).fillna(df["group_name"])
    df["method_rank"] = df["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    df["task_rank"] = df["group_name"].map({t: i for i, t in enumerate(TASK_ORDER)})
    return df.sort_values(["k", "group_type", "task_rank", "method_rank"])


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def pct_axis(ax: plt.Axes, upper: float = 1.05) -> None:
    ax.set_ylim(0, upper)
    ticks = np.linspace(0, 1.0, 6)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{int(t * 100)}%" for t in ticks])


def annotate_bars(ax: plt.Axes, *, percentage: bool = True, fontsize: float = 8.0) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if np.isnan(height) or height <= 0:
            continue
        label = f"{height * 100:.1f}%" if percentage else f"{height:.2f}"
        ax.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=0,
        )


def overall_rows(df: pd.DataFrame, k: int) -> pd.DataFrame:
    return df[(df["group_type"] == "overall") & (df["k"] == k)].copy()


def task_rows(df: pd.DataFrame, k: int) -> pd.DataFrame:
    rows = df[(df["group_type"] == "task_type") & (df["k"] == k)].copy()
    return rows.sort_values(["task_rank", "method_rank"])


def create_overview_figures(df: pd.DataFrame) -> None:
    out = OUT_DIR / "00_overview"
    metrics = ["evidence_recall_at_k", "complete_recall_at_k", "mrr"]

    for k in [3, 5]:
        rows = overall_rows(df, k)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
        for ax, metric in zip(axes, metrics):
            values = rows.set_index("method").loc[METHOD_ORDER, metric]
            labels = [METHOD_LABEL[m] for m in METHOD_ORDER]
            colors = [COLORS[m] for m in METHOD_ORDER]
            bars = ax.bar(labels, values, color=colors, edgecolor="#333333", linewidth=0.5)
            pct_axis(ax)
            ax.set_title(METRIC_LABEL[metric])
            ax.tick_params(axis="x", rotation=35)
            annotate_bars(ax)
            if metric == "complete_recall_at_k":
                for bar, method in zip(bars, METHOD_ORDER):
                    if method == "two_stage_agent":
                        bar.set_linewidth(2.2)
                        bar.set_edgecolor("#111111")
        fig.suptitle(f"Overall Retrieval Performance @ {k}")
        save_figure(fig, out / f"overall_metrics_at{k}")

    # Complete@3 vs Complete@5.
    rows3 = overall_rows(df, 3).set_index("method")
    rows5 = overall_rows(df, 5).set_index("method")
    x = np.arange(len(METHOD_ORDER))
    width = 0.34
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(
        x - width / 2,
        rows3.loc[METHOD_ORDER, "complete_recall_at_k"],
        width,
        label="Complete@3",
        color="#88BDE6",
        edgecolor="#333333",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        rows5.loc[METHOD_ORDER, "complete_recall_at_k"],
        width,
        label="Complete@5",
        color="#D62728",
        edgecolor="#333333",
        linewidth=0.5,
    )
    pct_axis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], rotation=25, ha="right")
    ax.set_ylabel("Complete Evidence Recall")
    ax.set_title("Complete Evidence Recall at Different Cutoffs")
    ax.legend(frameon=False, ncol=2)
    annotate_bars(ax)
    save_figure(fig, out / "complete_recall_at3_at5")

    # Evidence/complete gap at K=5.
    rows = overall_rows(df, 5).set_index("method").loc[METHOD_ORDER]
    gap = rows["evidence_recall_at_k"] - rows["complete_recall_at_k"]
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.bar(
        [METHOD_LABEL[m] for m in METHOD_ORDER],
        gap,
        color=[COLORS[m] for m in METHOD_ORDER],
        edgecolor="#333333",
        linewidth=0.5,
    )
    pct_axis(ax, upper=max(0.32, gap.max() + 0.05))
    ax.set_ylabel("Evidence@5 - Complete@5")
    ax.set_title("Partial Evidence Gap @ 5")
    ax.tick_params(axis="x", rotation=25)
    annotate_bars(ax)
    save_figure(fig, out / "partial_evidence_gap_at5")

    # Per-method metric profile.
    for k in [3, 5]:
        rows = overall_rows(df, k).set_index("method").loc[METHOD_ORDER]
        profile_metrics = ["evidence_recall_at_k", "complete_recall_at_k", "mrr"]
        x = np.arange(len(profile_metrics))
        fig, ax = plt.subplots(figsize=(10.8, 5.4))
        for method in METHOD_ORDER:
            linewidth = 2.8 if method == "two_stage_agent" else 1.8
            marker = "*" if method == "two_stage_agent" else "o"
            ax.plot(
                x,
                rows.loc[method, profile_metrics],
                color=COLORS[method],
                marker=marker,
                linewidth=linewidth,
                markersize=9 if method == "two_stage_agent" else 5.5,
                label=METHOD_LABEL[method],
            )
        pct_axis(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABEL[m] for m in profile_metrics])
        ax.set_title(f"Per-method Metric Profile @ {k}")
        ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.28))
        save_figure(fig, out / f"method_metric_profile_at{k}")


def create_task_breakdown_figures(df: pd.DataFrame) -> None:
    out = OUT_DIR / "01_task_breakdown"
    for metric in ["evidence_recall_at_k", "complete_recall_at_k", "mrr"]:
        for k in [3, 5]:
            create_heatmap(
                df,
                k=k,
                metric=metric,
                out_path=out / f"{metric.replace('_at_k', '')}_heatmap_at{k}",
            )

    for k in [3, 5]:
        rows = task_rows(df, k)
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
        for ax, task in zip(axes, TASK_ORDER):
            sub = rows[rows["group_name"] == task].set_index("method").loc[METHOD_ORDER]
            ax.bar(
                [METHOD_LABEL[m] for m in METHOD_ORDER],
                sub["complete_recall_at_k"],
                color=[COLORS[m] for m in METHOD_ORDER],
                edgecolor="#333333",
                linewidth=0.5,
            )
            pct_axis(ax)
            ax.set_title(TASK_LABEL[task])
            ax.tick_params(axis="x", rotation=35)
            annotate_bars(ax, fontsize=7.2)
        axes[0].set_ylabel(f"Complete Evidence Recall@{k}")
        fig.suptitle(f"Task-wise Complete Evidence Recall @ {k}")
        save_figure(fig, out / f"taskwise_complete_recall_at{k}")

    # Evidence vs complete paired bars by task for K=5.
    for task in TASK_ORDER:
        rows = task_rows(df, 5)
        sub = rows[rows["group_name"] == task].set_index("method").loc[METHOD_ORDER]
        x = np.arange(len(METHOD_ORDER))
        width = 0.34
        fig, ax = plt.subplots(figsize=(11.2, 5.2))
        ax.bar(
            x - width / 2,
            sub["evidence_recall_at_k"],
            width,
            color="#6BAED6",
            label="Evidence@5",
            edgecolor="#333333",
            linewidth=0.5,
        )
        ax.bar(
            x + width / 2,
            sub["complete_recall_at_k"],
            width,
            color="#D62728",
            label="Complete@5",
            edgecolor="#333333",
            linewidth=0.5,
        )
        pct_axis(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], rotation=25, ha="right")
        ax.set_title(f"Evidence vs Complete Evidence: {TASK_LABEL[task]} @ 5")
        ax.legend(frameon=False, ncol=2)
        annotate_bars(ax, fontsize=7.8)
        save_figure(fig, out / f"evidence_vs_complete_{task}_at5")


def create_heatmap(df: pd.DataFrame, *, k: int, metric: str, out_path: Path) -> None:
    rows = task_rows(df, k)
    matrix = (
        rows.pivot(index="method", columns="group_name", values=metric)
        .loc[METHOD_ORDER, TASK_ORDER]
        .to_numpy()
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(TASK_ORDER)))
    ax.set_xticklabels([TASK_LABEL[t] for t in TASK_ORDER], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels([METHOD_LABEL[m] for m in METHOD_ORDER])
    ax.set_title(f"{METRIC_LABEL[metric]} Heatmap @ {k}")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value >= 0.72 else "#111111"
            ax.text(j, i, f"{value * 100:.1f}%", ha="center", va="center", fontsize=8.5, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Score", rotation=270, labelpad=14)
    save_figure(fig, out_path)


def create_cost_efficiency_figures(df: pd.DataFrame) -> None:
    out = OUT_DIR / "02_cost_efficiency"
    rows = overall_rows(df, 5).set_index("method").loc[METHOD_ORDER]
    deployable = [m for m in METHOD_ORDER if m != "oracle"]

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for method in METHOD_ORDER:
        row = rows.loc[method]
        marker = "*" if method == "two_stage_agent" else ("X" if method == "oracle" else "o")
        size = 220 if method == "two_stage_agent" else (150 if method == "oracle" else 120)
        ax.scatter(
            row["tool_calls"],
            row["complete_recall_at_k"],
            s=size,
            c=COLORS[method],
            marker=marker,
            edgecolor="#222222",
            linewidth=0.8,
            alpha=0.95,
            label=METHOD_LABEL[method],
        )
        ax.text(row["tool_calls"] + 0.035, row["complete_recall_at_k"], METHOD_LABEL[method], va="center", fontsize=9)
    pct_axis(ax, upper=0.96)
    ax.set_xlim(0.8, 3.25)
    ax.set_xlabel("Average Tool Calls")
    ax.set_ylabel("Complete Evidence Recall@5")
    ax.set_title("Cost-effectiveness: Complete@5 vs Tool Calls")
    ax.grid(True, alpha=0.25)
    save_figure(fig, out / "cost_effectiveness_complete_at5")

    ratio = rows.loc[deployable, "complete_recall_at_k"] / rows.loc[deployable, "tool_calls"]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(
        [METHOD_LABEL[m] for m in deployable],
        ratio,
        color=[COLORS[m] for m in deployable],
        edgecolor="#333333",
        linewidth=0.5,
    )
    ax.set_ylabel("Complete@5 per Tool Call")
    ax.set_title("Cost-normalized Complete Evidence Recall")
    ax.tick_params(axis="x", rotation=25)
    annotate_bars(ax, percentage=False)
    save_figure(fig, out / "complete_per_tool_call_at5")

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    p95 = rows.loc[METHOD_ORDER, "latency_ms_p95"]
    ax.bar(
        [METHOD_LABEL[m] for m in METHOD_ORDER],
        p95,
        color=[COLORS[m] for m in METHOD_ORDER],
        edgecolor="#333333",
        linewidth=0.5,
    )
    ax.set_ylabel("P95 Latency (ms)")
    ax.set_title("P95 Latency by Method")
    ax.tick_params(axis="x", rotation=25)
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.0f}",
            (patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    save_figure(fig, out / "p95_latency_by_method_at5")

    # Task-wise tool calls.
    rows = task_rows(df, 5)
    deployable = ["dense", "file_fts", "graph_path", "rule_router", "fusion", "two_stage_agent"]
    x = np.arange(len(TASK_ORDER))
    width = 0.12
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    for i, method in enumerate(deployable):
        sub = rows[rows["method"] == method].set_index("group_name").loc[TASK_ORDER]
        offsets = x + (i - (len(deployable) - 1) / 2) * width
        ax.bar(
            offsets,
            sub["tool_calls"],
            width,
            color=COLORS[method],
            edgecolor="#333333",
            linewidth=0.4,
            label=METHOD_LABEL[method],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL[t] for t in TASK_ORDER], rotation=10)
    ax.set_ylabel("Average Tool Calls")
    ax.set_title("Task-wise Tool Calls @ 5")
    ax.legend(frameon=False, ncol=3)
    save_figure(fig, out / "taskwise_tool_calls_at5")


def create_agent_analysis_figures(df: pd.DataFrame) -> None:
    out = OUT_DIR / "03_agent_analysis"
    rows = overall_rows(df, 5).set_index("method").loc[METHOD_ORDER]
    metrics = ["evidence_recall_at_k", "complete_recall_at_k", "mrr"]

    # Agent vs selected baselines.
    compare_methods = ["dense", "file_fts", "graph_path", "rule_router", "fusion", "two_stage_agent"]
    x = np.arange(len(metrics))
    width = 0.12
    fig, ax = plt.subplots(figsize=(11.2, 5.5))
    for i, method in enumerate(compare_methods):
        offsets = x + (i - (len(compare_methods) - 1) / 2) * width
        ax.bar(
            offsets,
            rows.loc[method, metrics],
            width,
            color=COLORS[method],
            edgecolor="#333333",
            linewidth=0.4,
            label=METHOD_LABEL[method],
        )
    pct_axis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABEL[m] for m in metrics])
    ax.set_title("Agent vs Baselines on Overall Metrics @ 5")
    ax.legend(frameon=False, ncol=3)
    save_figure(fig, out / "agent_vs_baselines_overall_at5")

    # Task profile of final Agent.
    agent_rows = task_rows(df, 5)
    agent = agent_rows[agent_rows["method"] == "two_stage_agent"].set_index("group_name").loc[TASK_ORDER]
    x = np.arange(len(TASK_ORDER))
    width = 0.28
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x - width, agent["evidence_recall_at_k"], width, label="Evidence@5", color="#6BAED6", edgecolor="#333333")
    ax.bar(x, agent["complete_recall_at_k"], width, label="Complete@5", color="#D62728", edgecolor="#333333")
    ax.bar(x + width, agent["mrr"], width, label="MRR", color="#59A14F", edgecolor="#333333")
    pct_axis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL[t] for t in TASK_ORDER], rotation=10)
    ax.set_title("Final Agent Task Profile @ 5")
    ax.legend(frameon=False, ncol=3)
    annotate_bars(ax, fontsize=8)
    save_figure(fig, out / "agent_task_profile_at5")

    # Complementarity of core retrievers and final agent.
    task5 = task_rows(df, 5)
    complement_methods = ["dense", "file_fts", "graph_path", "two_stage_agent"]
    x = np.arange(len(TASK_ORDER))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    for i, method in enumerate(complement_methods):
        sub = task5[task5["method"] == method].set_index("group_name").loc[TASK_ORDER]
        offsets = x + (i - (len(complement_methods) - 1) / 2) * width
        ax.bar(
            offsets,
            sub["complete_recall_at_k"],
            width,
            color=COLORS[method],
            edgecolor="#333333",
            linewidth=0.5,
            label=METHOD_LABEL[method],
        )
    pct_axis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL[t] for t in TASK_ORDER], rotation=10)
    ax.set_ylabel("Complete Evidence Recall@5")
    ax.set_title("Core Retriever Complementarity and Final Agent @ 5")
    ax.legend(frameon=False, ncol=4)
    save_figure(fig, out / "core_retriever_complementarity_at5")

    # Agent gain over rule router and best single retriever.
    task5 = task_rows(df, 5)
    gain_rows = []
    for task in TASK_ORDER:
        sub = task5[task5["group_name"] == task].set_index("method")
        agent_value = sub.loc["two_stage_agent", "complete_recall_at_k"]
        router_value = sub.loc["rule_router", "complete_recall_at_k"]
        best_single_value = sub.loc[["dense", "file_fts", "graph_path"], "complete_recall_at_k"].max()
        gain_rows.append((task, agent_value - router_value, agent_value - best_single_value))
    x = np.arange(len(TASK_ORDER))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    router_gain = [g[1] for g in gain_rows]
    single_gain = [g[2] for g in gain_rows]
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.bar(x - width / 2, router_gain, width, label="Agent - Rule Router", color="#C8A2C8", edgecolor="#333333")
    ax.bar(x + width / 2, single_gain, width, label="Agent - Best Single Retriever", color="#D62728", edgecolor="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL[t] for t in TASK_ORDER], rotation=10)
    ax.set_ylabel("Complete@5 Difference")
    ax.set_title("Task-wise Agent Gain on Complete Evidence Recall @ 5")
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    save_figure(fig, out / "agent_gain_by_task_at5")

    # Balance chart: average vs worst-task complete@5.
    task_complete = (
        task5.pivot(index="method", columns="group_name", values="complete_recall_at_k")
        .loc[METHOD_ORDER, TASK_ORDER]
    )
    avg_complete = task_complete.mean(axis=1)
    worst_complete = task_complete.min(axis=1)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for method in METHOD_ORDER:
        marker = "*" if method == "two_stage_agent" else ("X" if method == "oracle" else "o")
        size = 220 if method == "two_stage_agent" else (150 if method == "oracle" else 115)
        ax.scatter(
            worst_complete.loc[method],
            avg_complete.loc[method],
            s=size,
            color=COLORS[method],
            edgecolor="#222222",
            linewidth=0.8,
            marker=marker,
        )
        ax.text(worst_complete.loc[method] + 0.01, avg_complete.loc[method], METHOD_LABEL[method], va="center", fontsize=9)
    pct_axis(ax, upper=0.98)
    ax.set_xlim(0.35, 1.0)
    ax.set_xlabel("Worst-task Complete@5")
    ax.set_ylabel("Average Task Complete@5")
    ax.set_title("Task Balance: Average vs Worst-task Complete@5")
    save_figure(fig, out / "task_balance_complete_at5")


def create_table_figures(df: pd.DataFrame) -> None:
    out = OUT_DIR / "04_tables"
    for k in [3, 5]:
        rows = overall_rows(df, k).set_index("method").loc[METHOD_ORDER]
        table_data = []
        for method in METHOD_ORDER:
            r = rows.loc[method]
            table_data.append(
                [
                    METHOD_LABEL[method],
                    f"{r['evidence_recall_at_k'] * 100:.2f}%",
                    f"{r['complete_recall_at_k'] * 100:.2f}%",
                    f"{r['mrr'] * 100:.2f}%",
                    f"{r['tool_calls']:.2f}",
                ]
            )
        create_table_image(
            out / f"overall_table_at{k}",
            title=f"Overall Results @ {k}",
            headers=["Method", f"Evidence@{k}", f"Complete@{k}", "MRR", "Calls"],
            rows=table_data,
            figsize=(9.0, 3.8),
        )

    for k in [3, 5]:
        rows = task_rows(df, k)
        table_data = []
        for method in METHOD_ORDER:
            method_rows = rows[rows["method"] == method].set_index("group_name").loc[TASK_ORDER]
            table_data.append(
                [
                    METHOD_LABEL[method],
                    f"{method_rows.loc['semantic_fact', 'complete_recall_at_k'] * 100:.2f}%",
                    f"{method_rows.loc['multi_hop_relation', 'complete_recall_at_k'] * 100:.2f}%",
                    f"{method_rows.loc['exact_file_lookup', 'complete_recall_at_k'] * 100:.2f}%",
                    f"{method_rows['tool_calls'].mean():.2f}",
                ]
            )
        create_table_image(
            out / f"task_complete_table_at{k}",
            title=f"Task-wise Complete Evidence Recall @ {k}",
            headers=["Method", "Semantic", "Multi-hop", "Exact File", "Calls"],
            rows=table_data,
            figsize=(9.5, 3.8),
        )


def create_table_image(path: Path, *, title: str, headers: list[str], rows: list[list[str]], figsize: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=10)
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.38)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#D9EAF7")
            cell.set_text_props(weight="bold")
        elif rows[row - 1][0] == "Agent":
            cell.set_facecolor("#FDE0DD")
        elif rows[row - 1][0] == "Oracle":
            cell.set_facecolor("#EEEEEE")
    save_figure(fig, path)


def create_index(df: pd.DataFrame) -> None:
    files = []
    for directory in ["00_overview", "01_task_breakdown", "02_cost_efficiency", "03_agent_analysis", "04_tables"]:
        for png in sorted((OUT_DIR / directory).glob("*.png")):
            files.append(png.relative_to(OUT_DIR).as_posix())

    rows5 = overall_rows(df, 5).set_index("method")
    agent = rows5.loc["two_stage_agent"]
    fusion = rows5.loc["fusion"]
    router = rows5.loc["rule_router"]
    content = [
        "# Visualization Index",
        "",
        "Data source: `results/final/main_results.csv`.",
        "",
        "The figures use the frozen method set: `dense`, `file_fts`, `graph_path`, `rule_router`, `fusion`, `oracle`, and the final `two_stage_agent`.",
        "",
        "Key numbers at @5:",
        f"- Agent Complete@5: {agent['complete_recall_at_k'] * 100:.2f}%",
        f"- Agent Evidence@5: {agent['evidence_recall_at_k'] * 100:.2f}%",
        f"- Agent MRR: {agent['mrr'] * 100:.2f}%",
        f"- Agent average tool calls: {agent['tool_calls']:.2f}",
        f"- Fusion Complete@5: {fusion['complete_recall_at_k'] * 100:.2f}% with {fusion['tool_calls']:.2f} tool calls",
        f"- Rule Router Complete@5: {router['complete_recall_at_k'] * 100:.2f}% with {router['tool_calls']:.2f} tool calls",
        "",
        "Generated figures:",
        "",
    ]
    content.extend(f"- `{file}`" for file in files)
    (OUT_DIR / "figure_index.md").write_text("\n".join(content) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
