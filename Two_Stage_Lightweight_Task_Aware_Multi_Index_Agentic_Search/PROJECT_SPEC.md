# Project Protocol

## Scope

The project evaluates three fixed retrieval baselines and then uses them as the tool pool for later router/agent experiments.

Final formal baselines:

- `dense`
- `file_fts`
- `graph_path`

Final task types:

- `semantic_fact`
- `multi_hop_relation`
- `exact_file_lookup`

The final dataset is fixed at 180 questions and 3,500 corpus documents:

- 60 `semantic_fact`
- 60 `multi_hop_relation`
- 60 `exact_file_lookup`

`exact_file_lookup` is evenly divided into:

- 20 `title_anchor`
- 20 `date_number_lookup`
- 20 `exact_phrase_lookup`

## Workflow

All reproducible work must be done through scripts in `experiments/` and library code in `src/`. Do not use notebooks as a required project step.

All formal retrieval runs must go through the anti-contamination experiment box:

```powershell
python experiments/run_experiment_box.py --method <dense|file_fts|graph_path> --split final
```

## Isolation Rule

Do not tune Router, Judge, retriever weights, thresholds, prompts, or question wording using final results. Method changes must be fixed before final evaluation and justified without looking at final performance.

Runtime methods can read only:

- public question ID
- public question text
- public corpus document ID, title, sentences, and full text

Evaluation-only data, including `gold_documents`, `gold_sentences`, source IDs, answers, and corpus role metadata, is opened only by evaluation/export scripts after prediction files exist.

## Checkpoints

Expected outputs after a complete baseline run:

- `indexes/final/dense_embeddings.npy`
- `indexes/final/dense_doc_ids.json`
- `indexes/final/file_fts.sqlite`
- `indexes/final/graph_path.pkl`
- `results/final/dense_final_predictions.jsonl`
- `results/final/file_fts_final_predictions.jsonl`
- `results/final/graph_path_final_predictions.jsonl`
- `results/final/main_results.csv`
- `D:\mark_ai_project\projects\project\baseline_result`
- `D:\mark_ai_project\projects\project\baseline_result\visualization`

Bootstrap metrics are for engineering checks only and must not be reported as final experiment results.
