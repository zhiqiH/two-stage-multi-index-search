# Final Evaluation Dataset

This directory is reserved for the frozen final evaluation dataset. The
bootstrap set is isolated in `data/bootstrap/` and must not be used for final
experiment tables.

## Fixed Target

- total questions: 180
- corpus documents: exactly 3500
- split: every question uses `"split": "final"`
- no dev/test split inside final

## Task Quotas

- `semantic_fact`: 60
- `multi_hop_relation`: 60
- `exact_file_lookup`: 60

`exact_file_lookup` is subdivided as:

- `title_anchor`: 20
- `date_number_lookup`: 20
- `exact_phrase_lookup`: 20

## Corpus Policy

All gold documents must be included. After gold and seed-context documents are
included, fill the corpus to exactly 3500 documents with unrelated HotpotQA
candidate documents. Every document not referenced by a question's
`gold_documents` is treated as noise/distractor material.

## Method Isolation

Router, judge, retriever weights, and thresholds must not be tuned on this
dataset. Use bootstrap only for engineering checks. Once final is frozen, run
experiments and analyze results without changing the method based on final
performance.
