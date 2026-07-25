# Data Schemas

## `corpus.jsonl`

One JSON object per document:

```json
{
  "doc_id": "doc_001",
  "title": "Document Title",
  "sentences": ["Sentence one.", "Sentence two."],
  "full_text": "Sentence one. Sentence two.",
  "source_question_ids": ["hotpot_abc"]
}
```

`full_text` may be omitted during loading; the loader will join `sentences`.

## `questions.jsonl`

One JSON object per evaluation question:

```json
{
  "question_id": "mh_001",
  "question": "Which country ...?",
  "task_type": "multi_hop",
  "gold_documents": ["doc_001", "doc_014"],
  "gold_sentences": [
    {"doc_id": "doc_001", "sent_id": 2},
    {"doc_id": "doc_014", "sent_id": 4}
  ],
  "source_hotpot_id": "hotpot_abc",
  "split": "test",
  "quality_checked": true
}
```

Allowed `task_type` values for the main experiment:

- `semantic_fact`
- `multi_hop`
- `exact_file_lookup`

## `splits.json`

```json
{
  "dev": ["q_001", "q_002"],
  "test": ["q_003", "q_004"]
}
```

Gold labels are for offline evaluation only. Router and evidence judge code
must not read `gold_documents` or `gold_sentences`.
