# Two-Stage Lightweight Task-Aware Multi-Index Agentic Search

This repository is a script-only research project. The frozen final dataset is evaluated through an anti-contamination experiment box.

No notebooks are required for the reproducible workflow.

## Final Dataset

Frozen files:

- `data/final/corpus.jsonl`: 3,500 public retrieval documents.
- `data/final/questions.jsonl`: 180 final questions.
- `data/final/splits.json`: compatibility split file; all final questions use `split = final`.
- `data/final/data_check.csv`: dataset audit sheet.

Question distribution:

- `semantic_fact`: 60
- `multi_hop_relation`: 60
- `exact_file_lookup`: 60

All non-gold corpus documents are distractors/noise. Runtime retrieval sees only public question text and public corpus text. Gold documents are opened only by evaluation scripts after predictions have already been written.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dense retrieval uses DeepInfra embeddings. Set the token only as an environment variable while running dense:

```powershell
$env:DEEPINFRA_TOKEN="..."
.\experiments\run_baseline.ps1 -Method dense -RebuildDense
Remove-Item Env:\DEEPINFRA_TOKEN
```

Do not write API keys into project files.

## Formal Baselines

The final formal baseline set is:

- `dense`: DeepInfra embedding retrieval with `sentence-transformers/all-MiniLM-L6-v2`.
- `file_fts`: SQLite FTS5 file-style retrieval with field weighting, chunk aggregation, and exact-feature rescoring.
- `graph_path`: pure graph path retrieval over a document-sentence-entity-relation graph.

Discarded historical methods are not formal baselines and have been removed from the runnable baseline entrypoints.

## Run Baselines

Run from the project root:

```powershell
.\experiments\run_baseline.ps1 -Method file_fts
.\experiments\run_baseline.ps1 -Method graph_path

$env:DEEPINFRA_TOKEN="..."
.\experiments\run_baseline.ps1 -Method dense -RebuildDense
Remove-Item Env:\DEEPINFRA_TOKEN
```

The wrapper writes predictions, metrics, summaries, leakage checks, and `results/final/main_results.csv`.

Direct experiment-box usage:

```powershell
python experiments/run_experiment_box.py --method file_fts --split final --ks 1,3,5 --save-indexes
python experiments/run_experiment_box.py --method graph_path --split final --ks 1,3,5 --save-indexes
python experiments/run_experiment_box.py --method dense --split final --ks 1,3,5 --save-indexes --rebuild-dense
```

Validate the public runtime view:

```powershell
python experiments/run_experiment_box.py --method dense --split final --validate-inputs-only
```

## Export Results

```powershell
python experiments/export_baseline_results.py `
  --config config.yaml `
  --out-dir D:\mark_ai_project\projects\project\baseline_result

python experiments/visualize_baseline_results.py `
  --baseline-dir D:\mark_ai_project\projects\project\baseline_result `
  --out-dir D:\mark_ai_project\projects\project\baseline_result\visualization
```

Exported report files:

- `baseline_result/baseline_metrics_all.csv`
- `baseline_result/baseline_metrics_overall.csv`
- `baseline_result/baseline_metrics_by_task.csv`
- `baseline_result/failure_cases_k5.csv`
- `baseline_result/baseline_failure_analysis.md`
- `baseline_result/visualization/`

## Anti-Contamination Rules

- Do not tune Router, Judge, retriever weights, thresholds, prompts, or the dataset using final results.
- The final dataset is for fixed evaluation only.
- Bootstrap or separate validation data must be used for future method design.
- All methods must pass `experiments/check_no_gold_leakage.py` before evaluation results are trusted.
