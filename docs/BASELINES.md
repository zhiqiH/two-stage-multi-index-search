# Final Baselines

This project now uses three formal retrieval baselines on the frozen final dataset.

The final dataset is fixed under `data/final/`:

- `corpus.jsonl`: 3,500 public retrieval documents.
- `questions.jsonl`: 180 questions, 60 per task type.
- Task types: `semantic_fact`, `multi_hop_relation`, `exact_file_lookup`.
- Runtime retrieval reads only the public corpus and public question fields.
- Gold documents are read only after retrieval by the evaluator.

## Formal Methods

| Method | ID | Purpose | Implementation |
|---|---|---|---|
| Dense | `dense` | Lightweight semantic vector baseline for complementarity analysis. | DeepInfra embeddings with `sentence-transformers/all-MiniLM-L6-v2`. |
| File-FTS | `file_fts` | Standard file/search baseline with lexical and exact lookup advantages. | SQLite FTS5, weighted fields, chunk retrieval, document aggregation, and exact-feature rescoring. |
| Graph-Path | `graph_path` | Pure graph retrieval baseline for relation and multi-hop evidence discovery. | Document-sentence-entity-relation graph with query-aware beam search over explicit paths. |

Only the three methods above are formal baselines.

## Run Order

Run from the project root:

```powershell
.\experiments\run_baseline.ps1 -Method file_fts
.\experiments\run_baseline.ps1 -Method graph_path

$env:DEEPINFRA_TOKEN="..."
.\experiments\run_baseline.ps1 -Method dense -RebuildDense
Remove-Item Env:\DEEPINFRA_TOKEN
```

## Output Files

Each method writes:

- `results/final/{method}_final_predictions.jsonl`
- `results/final/{method}_final_metrics.csv`
- `results/final/{method}_final_summary.json`

The shared summary is:

- `results/final/main_results.csv`

Indexes are rebuilt under:

- `indexes/final/dense_embeddings.npy`
- `indexes/final/dense_doc_ids.json`
- `indexes/final/file_fts.sqlite`
- `indexes/final/graph_path.pkl`

The report-ready export is generated under:

- `results/analysis/baselines/`
- `figures/baselines/all_methods/`

## Required Metrics

The exported metrics include:

- Evidence Recall@1/@3/@5
- Complete Evidence Recall@1/@3/@5
- MRR
- Search Success Rate
- Average Tool Calls
- Average and P95 Latency
- Evidence Gain per Step, blank for one-step baselines
- Stop Accuracy, blank for one-step baselines

Task-wise metrics are always exported separately for:

- `semantic_fact`
- `multi_hop_relation`
- `exact_file_lookup`
