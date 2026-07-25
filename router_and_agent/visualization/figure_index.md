# Visualization Index

Data source: `router_and_agent/results/final/main_results.csv`.

The figures use the frozen method set: `dense`, `file_fts`, `graph_path`, `rule_router`, `fusion`, `oracle`, and the final `two_stage_agent`.

Key numbers at @5:
- Agent Complete@5: 85.56%
- Agent Evidence@5: 96.67%
- Agent MRR: 87.81%
- Agent average tool calls: 1.44
- Fusion Complete@5: 85.56% with 3.00 tool calls
- Rule Router Complete@5: 82.22% with 1.00 tool calls

Generated figures:

- `00_overview/complete_recall_at3_at5.png`
- `00_overview/method_metric_profile_at3.png`
- `00_overview/method_metric_profile_at5.png`
- `00_overview/overall_metrics_at3.png`
- `00_overview/overall_metrics_at5.png`
- `00_overview/partial_evidence_gap_at5.png`
- `01_task_breakdown/complete_recall_heatmap_at3.png`
- `01_task_breakdown/complete_recall_heatmap_at5.png`
- `01_task_breakdown/evidence_recall_heatmap_at3.png`
- `01_task_breakdown/evidence_recall_heatmap_at5.png`
- `01_task_breakdown/evidence_vs_complete_exact_file_lookup_at5.png`
- `01_task_breakdown/evidence_vs_complete_multi_hop_relation_at5.png`
- `01_task_breakdown/evidence_vs_complete_semantic_fact_at5.png`
- `01_task_breakdown/mrr_heatmap_at3.png`
- `01_task_breakdown/mrr_heatmap_at5.png`
- `01_task_breakdown/taskwise_complete_recall_at3.png`
- `01_task_breakdown/taskwise_complete_recall_at5.png`
- `02_cost_efficiency/complete_per_tool_call_at5.png`
- `02_cost_efficiency/cost_effectiveness_complete_at5.png`
- `02_cost_efficiency/p95_latency_by_method_at5.png`
- `02_cost_efficiency/taskwise_tool_calls_at5.png`
- `03_agent_analysis/agent_gain_by_task_at5.png`
- `03_agent_analysis/agent_task_profile_at5.png`
- `03_agent_analysis/agent_vs_baselines_overall_at5.png`
- `03_agent_analysis/core_retriever_complementarity_at5.png`
- `03_agent_analysis/task_balance_complete_at5.png`
- `04_tables/overall_table_at3.png`
- `04_tables/overall_table_at5.png`
- `04_tables/task_complete_table_at3.png`
- `04_tables/task_complete_table_at5.png`
