# Final Dataset Card

Status: frozen
Build seed: 42
Selection manifest: `data\final\selection_manifest.json`

## Size

- Questions: 180
- Corpus documents: 3500

## Task Counts

- exact_file_lookup: 60
- multi_hop_relation: 60
- semantic_fact: 60

## Exact Lookup Subtypes

- date_number_lookup: 20
- exact_phrase_lookup: 20
- title_anchor: 20

## Isolation Rule

This final dataset is for evaluation only. Do not tune Router, Judge, retriever parameters, or question wording on final experiment failures.
