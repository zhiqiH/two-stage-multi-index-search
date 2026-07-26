# 图表目录

- `agent/`：Router、Two-Stage Agent、成本效率和任务分解图。
- `baselines/all_methods/`：Dense、File-FTS、Graph-Path、Fusion、Oracle 的完整比较。
- `baselines/core_only/`：不含补充方法的历史核心基线图。

重新生成：

```powershell
python experiments/visualize_baseline_results.py
python tools/generate_agent_visualizations.py
```

两个脚本会清理各自的输出目录后重新绘图。
