from __future__ import annotations

import re
from collections import Counter

from .common import normalize_text


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "which",
    "who",
    "with",
}


ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][a-zA-Z0-9'&.-]+)(?:\s+(?:[A-Z][a-zA-Z0-9'&.-]+|of|the|and|de|la))*"
)
NUMBER_DATE_PATTERN = re.compile(r"\b\d{4}\b|\b\d+(?:\.\d+)?\b")
QUOTED_PHRASE_PATTERN = re.compile(r'"([^"]+)"')


def simple_entities(text: str) -> list[str]:
    entities = []
    for match in ENTITY_PATTERN.finditer(text):
        entity = normalize_entity(match.group(0))
        if entity and entity not in STOPWORDS:
            entities.append(entity)
    return sorted(set(entities))


def normalize_entity(entity: str) -> str:
    return normalize_text(entity)


def numbers_and_dates(text: str) -> list[str]:
    return sorted(set(match.group(0) for match in NUMBER_DATE_PATTERN.finditer(text)))


def quoted_phrases(text: str) -> list[str]:
    return [match.group(1).strip() for match in QUOTED_PHRASE_PATTERN.finditer(text)]


def keywords(text: str, *, max_keywords: int = 20) -> list[str]:
    tokens = [
        token
        for token in normalize_text(text).split()
        if len(token) > 2 and token not in STOPWORDS
    ]
    counts = Counter(tokens)
    return [token for token, _ in counts.most_common(max_keywords)]
