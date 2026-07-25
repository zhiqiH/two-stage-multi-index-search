from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from .common import (
    CorpusDocument,
    RetrievedDocument,
    SearchOutput,
    estimate_tokens,
    now_ms,
)


class DenseSearcher:
    def __init__(
        self,
        *,
        corpus: list[CorpusDocument],
        embeddings: np.ndarray,
        provider: str,
        model_name: str,
        normalize_embeddings: bool,
        batch_size: int,
        max_document_tokens: int,
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        if len(corpus) != len(embeddings):
            raise ValueError("Corpus and embedding counts must match.")
        self.corpus = corpus
        self.embeddings = embeddings.astype("float32")
        self.provider = provider
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_document_tokens = max_document_tokens
        self.base_url = base_url
        self.api_key_env = api_key_env

    @classmethod
    def build(
        cls,
        corpus: list[CorpusDocument],
        *,
        provider: str,
        model_name: str,
        batch_size: int,
        normalize_embeddings: bool,
        max_document_tokens: int,
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> "DenseSearcher":
        texts = [
            truncate_words(materialize_dense_document(doc), max_document_tokens)
            for doc in corpus
        ]
        embeddings = encode_texts(
            texts,
            provider=provider,
            model_name=model_name,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            base_url=base_url,
            api_key_env=api_key_env,
        )
        return cls(
            corpus=corpus,
            embeddings=embeddings,
            provider=provider,
            model_name=model_name,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            max_document_tokens=max_document_tokens,
            base_url=base_url,
            api_key_env=api_key_env,
        )

    @classmethod
    def load(
        cls,
        *,
        corpus: list[CorpusDocument],
        embeddings_path: str | Path,
        doc_ids_path: str | Path,
        provider: str,
        model_name: str,
        batch_size: int,
        normalize_embeddings: bool,
        max_document_tokens: int,
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> "DenseSearcher":
        embeddings = np.load(embeddings_path).astype("float32")
        doc_ids = json.loads(Path(doc_ids_path).read_text(encoding="utf-8"))
        corpus_by_id = {doc.doc_id: doc for doc in corpus}
        ordered_corpus = [corpus_by_id[doc_id] for doc_id in doc_ids]
        return cls(
            corpus=ordered_corpus,
            embeddings=embeddings,
            provider=provider,
            model_name=model_name,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            max_document_tokens=max_document_tokens,
            base_url=base_url,
            api_key_env=api_key_env,
        )

    def save(self, *, embeddings_path: str | Path, doc_ids_path: str | Path) -> None:
        embeddings_path = Path(embeddings_path)
        doc_ids_path = Path(doc_ids_path)
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        doc_ids_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, self.embeddings)
        doc_ids_path.write_text(
            json.dumps([doc.doc_id for doc in self.corpus], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def search(
        self,
        *,
        question_id: str,
        query: str,
        top_k: int = 5,
    ) -> SearchOutput:
        start_ms = now_ms()
        query_embedding = encode_texts(
            [query],
            provider=self.provider,
            model_name=self.model_name,
            batch_size=1,
            normalize_embeddings=self.normalize_embeddings,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
        )[0]
        return self.search_with_embedding(
            question_id=question_id,
            query_embedding=query_embedding,
            top_k=top_k,
            query_latency_ms=now_ms() - start_ms,
            trace_prefix=f"encode_query:{self.provider}",
        )

    def search_with_embedding(
        self,
        *,
        question_id: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        query_latency_ms: float = 0.0,
        trace_prefix: str = "encode_query",
    ) -> SearchOutput:
        start_ms = now_ms()
        scores = self.embeddings @ query_embedding
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), self.corpus[index].title, self.corpus[index].doc_id),
        )[:top_k]

        docs = []
        text_read_tokens = 0
        for rank, doc_index in enumerate(ranked_indices, start=1):
            doc = self.corpus[doc_index]
            text = truncate_words(materialize_dense_document(doc), self.max_document_tokens)
            text_read_tokens += estimate_tokens(text)
            docs.append(
                RetrievedDocument(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    score=float(scores[doc_index]),
                    rank=rank,
                    tool="dense",
                )
            )

        return SearchOutput(
            question_id=question_id,
            tool="dense",
            ranked_documents=docs,
            latency_ms=query_latency_ms + (now_ms() - start_ms),
            text_read_tokens=text_read_tokens,
            trace=[
                trace_prefix,
                "dot_product_search",
                f"return_top_{top_k}",
            ],
        )


def materialize_dense_document(doc: CorpusDocument) -> str:
    return f"{doc.title}. {doc.full_text}".strip()


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        trimmed = text
    else:
        trimmed = " ".join(words[:max_words])

    # DeepInfra's BGE endpoint enforces a 512-token model limit. The project
    # token counter is word-based, so add a conservative character cap to keep
    # the API input below the service-side tokenizer limit without using gold.
    max_chars = max(256, max_words * 4)
    if len(trimmed) <= max_chars:
        return trimmed
    shortened = trimmed[:max_chars].rsplit(" ", 1)[0].strip()
    return shortened or trimmed[:max_chars].strip()


def encode_texts(
    texts: list[str],
    *,
    provider: str,
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> np.ndarray:
    provider = provider.lower()
    if provider == "local":
        return encode_texts_local(
            texts,
            model_name=model_name,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        )
    if provider == "deepinfra":
        return encode_texts_openai_compatible(
            texts,
            model_name=model_name,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            base_url=base_url,
            api_key_env=api_key_env or "DEEPINFRA_TOKEN",
        )
    raise ValueError(f"Unsupported dense provider: {provider}")


def encode_texts_local(
    texts: list[str],
    *,
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Local dense retrieval requires sentence-transformers. "
            "Run: pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype="float32")


def encode_texts_openai_compatible(
    texts: list[str],
    *,
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
    base_url: str | None,
    api_key_env: str,
) -> np.ndarray:
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            OpenAI,
            RateLimitError,
        )
    except ImportError as exc:
        raise RuntimeError("DeepInfra dense retrieval requires openai. Run: pip install openai") from exc

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env} is not set.")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=0)
    all_embeddings: list[list[float]] = []
    for batch in batched(texts, batch_size):
        response = None
        for attempt in range(5):
            try:
                response = client.embeddings.create(model=model_name, input=batch)
                break
            except APIStatusError as exc:
                if exc.status_code < 500 and exc.status_code != 429:
                    raise
                if attempt == 4:
                    raise
                time.sleep(min(2.0 ** attempt, 8.0))
            except (APIConnectionError, APITimeoutError, RateLimitError):
                if attempt == 4:
                    raise
                time.sleep(min(2.0 ** attempt, 8.0))
        if response is None:
            raise RuntimeError("Embedding request failed without a response.")
        ordered = sorted(response.data, key=lambda item: item.index)
        all_embeddings.extend([item.embedding for item in ordered])

    embeddings = np.asarray(all_embeddings, dtype="float32")
    if normalize_embeddings:
        embeddings = l2_normalize(embeddings)
    return embeddings


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embeddings / norms


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
