# Router and Agent Frozen Experiment Commands

Working directory:
`D:\mark_ai_project\projects\project\Two_Stage_Lightweight_Task_Aware_Multi_Index_Agentic_Search`

All DeepInfra calls used the `DEEPINFRA_TOKEN` environment variable. The token value is intentionally not recorded.

Representative commands:

```powershell
.\.venv\Scripts\python.exe .\experiments\run_experiment_box.py --config config.yaml --method dense --split final --ks 1,3,5
.\.venv\Scripts\python.exe .\experiments\run_experiment_box.py --config config.yaml --method file_fts --split final --ks 1,3,5
.\.venv\Scripts\python.exe .\experiments\run_experiment_box.py --config config.yaml --method graph_path --split final --ks 1,3,5
.\.venv\Scripts\python.exe .\experiments\run_experiment_box.py --config config.yaml --method rule_router --split final --ks 1,3,5
.\.venv\Scripts\python.exe .\experiments\run_experiment_box.py --config config.yaml --method two_stage_agent --split final --ks 1,3,5
.\.venv\Scripts\python.exe .\experiments\run_fusion_oracle.py --config config.yaml --split final --ks 1,3,5
.\.venv\Scripts\python.exe .\experiments\summarize_results.py --config config.yaml
```

Frozen methods included in this package:
`dense`, `file_fts`, `graph_path`, `rule_router`, `fusion`, `oracle`, `two_stage_agent`.

The final two-stage implementation is stored under the official method name `two_stage_agent`.
