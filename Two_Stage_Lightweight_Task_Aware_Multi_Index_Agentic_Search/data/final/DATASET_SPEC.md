# Final Dataset Specification

## Version

Status: frozen.

The final dataset is immutable because `FROZEN.txt` has been created.

## Files To Be Filled

- `corpus.jsonl`: exactly 3500 documents
- `questions.jsonl`: exactly 180 questions
- `data_check.csv`: quality-control records
- `splits.json`: compatibility file with one key: `final`
- `selection_manifest.json`: selected candidate IDs and source hashes
- `DATASET_CARD.md`: final dataset description
- `FROZEN.txt`: freeze record

## Construction Workflow

1. Build staging candidates with `experiments/build_final_staging.py`.
2. Use `experiments/prepare_final_questions_with_llm.py` to produce LLM-assisted
   `*_llm.csv` files.
3. Select only accepted, quality-checked rows from the LLM-assisted files.
4. Build `questions.jsonl`, `splits.json`, and `data_check.csv`.
5. Freeze only after approving the final 180-question dataset.

LLM usage is part of dataset construction only. After `FROZEN.txt` exists, do
not use final experiment results to tune Router, Judge, retriever parameters, or
question wording.

The LLM-assisted step has task-specific rules:

- `semantic_fact`: LLM may rewrite the candidate into a natural semantic
  question, then an independent verifier checks it.
- `exact_file_lookup`: LLM may generate the final exact/file lookup question,
  then an independent verifier checks subtype constraints.
- `multi_hop_relation`: LLM may only verify and screen the candidate; it must
  not rewrite the original multi-hop question.

## Question Schema

Each question must include:

```json
{
  "question_id": "sf_0001",
  "question": "...",
  "task_type": "semantic_fact",
  "gold_documents": ["doc_..."],
  "gold_sentences": [{"doc_id": "doc_...", "sent_id": 0, "text": "..."}],
  "source_hotpot_id": "...",
  "split": "final",
  "quality_checked": true,
  "metadata": {
    "construction": "...",
    "subtype": "...",
    "answer": "..."
  }
}
```

## Corpus Schema

Each document must include:

```json
{
  "doc_id": "doc_...",
  "title": "...",
  "sentences": ["..."],
  "full_text": "...",
  "source_question_ids": ["..."],
  "metadata": {
    "corpus_role": "gold_or_seed_context_or_noise"
  }
}
```

`corpus_role` is descriptive. Evaluation uses `gold_documents`, not this field.

## Freeze Rule

After final freeze:

- do not edit questions to improve any method
- do not tune Router/Judge/retriever parameters on final failures
- only fix objectively invalid records, and record the fix in a changelog
