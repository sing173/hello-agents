"""Deterministic lexical helpers shared by the offline components."""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+", re.IGNORECASE)
_CJK_PATTERN = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")
_STEM_SUFFIXES = tuple(
    sorted(
        {
            "ational",
            "ation",
            "ements",
            "ement",
            "ically",
            "ingly",
            "ments",
            "ment",
            "ized",
            "izes",
            "izing",
            "ised",
            "ises",
            "ising",
            "izations",
            "ization",
            "isations",
            "isation",
            "edly",
            "ing",
            "ies",
            "ize",
            "ise",
            "ed",
            "es",
            "al",
            "e",
            "s",
        },
        key=len,
        reverse=True,
    )
)


def normalize_text(text: str) -> str:
    """Case-fold text and collapse whitespace."""
    return " ".join(text.casefold().split())


_ALIASES = {
    "recollection": "retriev",
    "retrieval": "retriev",
    "retrieve": "retriev",
}

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


def _normalize_token(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    for suffix in _STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return _ALIASES.get(word[: -len(suffix)], word[: -len(suffix)])
    return _ALIASES.get(word, word)


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize Latin words and CJK text without external dictionaries.

    CJK runs are represented as characters, adjacent bigrams, and the complete
    run. Latin words receive a small suffix-stripping normalization so queries
    such as ``optimize`` and documents such as ``optimization`` overlap.
    """
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text):
        value = match.group(0).casefold()
        if _CJK_PATTERN.fullmatch(value):
            tokens.extend(value)
            tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
            if len(value) > 2:
                tokens.append(value)
        else:
            if value not in _STOPWORDS:
                tokens.append(_normalize_token(value))
    return tuple(tokens)


def lexical_similarity(query: str, document: str) -> tuple[float, float, float]:
    """Return TF cosine, Jaccard, and query-coverage scores."""
    query_tokens = tokenize(query)
    document_tokens = tokenize(document)
    if not query_tokens or not document_tokens:
        return (0.0, 0.0, 0.0)

    query_set = set(query_tokens)
    document_set = set(document_tokens)
    intersection = query_set & document_set
    union = query_set | document_set

    query_counter = Counter(query_tokens)
    document_counter = Counter(document_tokens)
    dot_product = sum(
        count * document_counter.get(token, 0) for token, count in query_counter.items()
    )
    query_norm = math.sqrt(sum(count * count for count in query_counter.values()))
    document_norm = math.sqrt(
        sum(count * count for count in document_counter.values())
    )
    cosine = dot_product / (query_norm * document_norm) if query_norm and document_norm else 0.0
    jaccard = len(intersection) / len(union) if union else 0.0
    coverage = len(intersection) / len(query_set)

    return (cosine, jaccard, coverage)


def hybrid_lexical_score(query: str, document: str) -> float:
    """Combine lexical signals into a stable 0..1 relevance score."""
    cosine, jaccard, coverage = lexical_similarity(query, document)
    return min(1.0, 0.6 * cosine + 0.2 * jaccard + 0.2 * coverage)


def estimate_tokens(text: str) -> int:
    """Estimate tokens as one per CJK character or Latin number/word."""
    cjk_count = 0
    latin_count = 0
    for match in _TOKEN_PATTERN.finditer(text):
        value = match.group(0)
        if _CJK_PATTERN.fullmatch(value):
            cjk_count += len(value)
        else:
            latin_count += 1
    return cjk_count + latin_count