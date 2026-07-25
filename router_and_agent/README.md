# Router and Agent Frozen Package

This directory freezes the router and final two-stage agent experiments.

## Contents

- `data/final/`: frozen final dataset used by all reported methods.
- `code/`: source snapshot for retrievers, router, evidence judge, agent, metrics, and experiment box.
- `results/final/`: frozen predictions, per-question metrics, and summary JSON for the retained methods.
- `experiment_records/`: compact result tables, method inventory, command record, and source README snapshots.
- `reports/`: Word report for the router and agent design/results.
- `manifest/freeze_manifest.json`: file hashes, dataset counts, method list, and freeze metadata.

## Frozen Methods

The retained methods are:

- `dense`
- `file_fts`
- `graph_path`
- `rule_router`
- `fusion`
- `oracle`
- `two_stage_agent`

The final coverage-judge/protected-fusion agent is stored under the official method name `two_stage_agent`.

## Dataset

The frozen final dataset has 3500 public corpus documents and 180 questions:

- `semantic_fact`: 60
- `multi_hop_relation`: 60
- `exact_file_lookup`: 60

All methods retrieve from the same 3500-document corpus. Gold labels are used only inside the evaluation code.

## Main Report

Open `reports/router_and_agent_design_and_results.docx` for the detailed design, implementation notes, results, comparisons, and limitations.
