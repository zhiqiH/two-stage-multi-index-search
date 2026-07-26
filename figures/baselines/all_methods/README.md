# Baseline Visualization Index

This directory is generated from `results/analysis/baselines` CSV files. It does not rerun retrieval and does not read gold data beyond exported evaluation metrics.

Charts are saved as PNG and PDF. PNG files are convenient for slides; PDF files are convenient for reports and papers.

## Directory Guide

- `summary/`: overview dashboard.
- `overall/`: overall @5 and quality-cost charts.
- `by_metric/`: one chart per metric comparing baselines and supplemental references.
- `by_task/`: task-wise grouped comparisons.
- `heatmaps/`: method x task metric heatmaps.
- `method_profiles/`: per-method metric profiles.
- `k_sensitivity/`: @1/@3/@5 curves.
- `failures/`: failure counts and failure-rate visualizations.

## Notes

- Formal runnable baselines: `dense`, `file_fts`, `graph_path`, `fusion`.
- `oracle` is an analysis-only upper-bound reference.
- Evidence Gain per Step and Stop Accuracy are not applicable to these one-step baselines.
- All outputs are Top-k retrieval outputs, not answer-generation predictions.

## PNG Files

- `by_metric/overall_at5_average_tool_calls.png`
- `by_metric/overall_at5_complete_evidence_recall.png`
- `by_metric/overall_at5_evidence_recall.png`
- `by_metric/overall_at5_latency_avg_ms.png`
- `by_metric/overall_at5_latency_p95_ms.png`
- `by_metric/overall_at5_mrr.png`
- `by_metric/overall_at5_search_success_rate.png`
- `by_task/taskwise_at5_average_tool_calls_grouped.png`
- `by_task/taskwise_at5_complete_evidence_recall_grouped.png`
- `by_task/taskwise_at5_evidence_recall_grouped.png`
- `by_task/taskwise_at5_latency_avg_ms_grouped.png`
- `by_task/taskwise_at5_latency_p95_ms_grouped.png`
- `by_task/taskwise_at5_mrr_grouped.png`
- `by_task/taskwise_at5_search_success_rate_grouped.png`
- `failures/failure_counts_stacked_by_task.png`
- `failures/failure_rate_heatmap_by_task.png`
- `failures/formal_failure_heatmap_by_task.png`
- `heatmaps/task_method_heatmap_complete_evidence_recall.png`
- `heatmaps/task_method_heatmap_evidence_recall.png`
- `heatmaps/task_method_heatmap_mrr.png`
- `heatmaps/task_method_heatmap_search_success_rate.png`
- `k_sensitivity/k_sensitivity_complete_evidence_recall.png`
- `k_sensitivity/k_sensitivity_evidence_recall.png`
- `k_sensitivity/k_sensitivity_mrr.png`
- `k_sensitivity/k_sensitivity_search_success_rate.png`
- `method_profiles/method_metric_profile_heatmap.png`
- `method_profiles/profile_dense.png`
- `method_profiles/profile_file_fts.png`
- `method_profiles/profile_fusion.png`
- `method_profiles/profile_graph_path.png`
- `method_profiles/profile_oracle.png`
- `overall/formal_gap_to_oracle_complete_recall.png`
- `overall/latency_avg_vs_p95.png`
- `overall/overall_quality_metrics_grouped.png`
- `overall/quality_cost_tradeoff_complete_recall_latency.png`
- `overall/taskwise_gap_to_oracle_complete_recall.png`
- `summary/baseline_evaluation_dashboard.png`
