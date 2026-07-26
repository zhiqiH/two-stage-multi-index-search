# 结果目录

## `final/`

正式冻结输出：

- `{method}_final_predictions.jsonl`：Top-k 检索结果与运行轨迹
- `{method}_final_metrics.csv`：逐题、逐 k 指标
- `{method}_final_summary.json`：聚合指标与置信区间
- `main_results.csv`：所有保留方法的统一结果表

保留方法：`dense`、`file_fts`、`graph_path`、`rule_router`、`fusion`、`oracle`、`two_stage_agent`。

## `analysis/`

由正式结果派生的报告用表格、失败案例和历史冻结记录：

- `analysis/baselines/`：基线指标导出和失败分析
- `analysis/agent/`：Router / Agent 的紧凑对比表与原始冻结包记录

不要手工修改 `final/` 中的正式结果。使用 `experiments/` 下的脚本重新生成。
