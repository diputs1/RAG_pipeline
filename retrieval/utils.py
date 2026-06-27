"""
retrieval/utils.py
==================
Shared helper functions used across retrieval modules.
"""

from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document


_SCORE_KEYS = ("relevance_score", "rrf_score", "hybrid_score", "dbsf_score", "bm25_score", "score")


def _dedup_key(doc: Document) -> tuple[str, str]:
    doc_id = doc.metadata.get("doc_id") or doc.metadata.get("id")
    source = doc.metadata.get("source") or doc.metadata.get("document_id") or ""
    if doc_id:
        return ("id", f"{source}:{doc_id}")
    return ("content", doc.page_content.strip())


def _best_score(doc: Document) -> float:
    for key in _SCORE_KEYS:
        value = doc.metadata.get(key)
        if value is not None:
            return float(value)
    return 0.0


def deduplicate(docs: list[Document]) -> list[Document]:
    """Deduplicate by document id when present, otherwise by content.

    If a duplicate id appears later with a better retrieval score, keep the
    stronger candidate while preserving the first-seen position.
    """
    chosen: dict[tuple[str, str], Document] = {}
    order: list[tuple[str, str]] = []

    for doc in docs:
        key = _dedup_key(doc)
        if key not in chosen:
            chosen[key] = doc
            order.append(key)
            continue
        if _best_score(doc) > _best_score(chosen[key]):
            chosen[key] = doc

    return [chosen[key] for key in order]


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """
    Reciprocal Rank Fusion (RRF) — merge multiple ranked lists.

    RRF score:  score(d) = Σ  1 / (k + rank_i(d))

    k=60 is the standard constant from Cormack et al. (2009).
    Higher k → smoother differences; lower k → stronger top-rank bias.

    Returns documents sorted by descending RRF score.
    """
    scores:  dict[str, float]    = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key           = doc.page_content.strip()
            scores[key]  += 1.0 / (k + rank)
            doc_map[key]  = doc

    for doc in doc_map.values():
        doc.metadata["rrf_score"] = scores[doc.page_content.strip()]

    return [doc_map[k] for k in sorted(scores, key=scores.__getitem__, reverse=True)]


def weighted_fusion(
    dense_docs:  list[Document],
    sparse_docs: list[Document],
    alpha: float = 0.5,
) -> list[Document]:
    """
    Weighted linear combination of dense and sparse scores.

    Final score = alpha * dense_norm + (1 - alpha) * sparse_norm
    Scores are min-max normalised within each list before combining.

    alpha = 1.0 → pure dense; alpha = 0.0 → pure sparse.
    """
    def normalise(docs: list[Document], key: str) -> dict[str, float]:
        raw = [doc.metadata.get(key, 0.0) for doc in docs]
        # Fall back to rank-based scoring when explicit scores are absent
        # (most dense VectorStore backends don't set relevance_score by default)
        if not raw or all(s == 0.0 for s in raw):
            raw = [1.0 / (i + 1) for i in range(len(docs))]
        lo, hi = min(raw), max(raw)
        rng    = hi - lo or 1.0
        return {doc.page_content.strip(): (raw[i] - lo) / rng for i, doc in enumerate(docs)}

    d_scores = normalise(dense_docs,  "relevance_score")
    s_scores = normalise(sparse_docs, "bm25_score")

    all_docs: dict[str, Document] = {
        doc.page_content.strip(): doc
        for doc in dense_docs + sparse_docs
    }
    combined: dict[str, float] = {
        key: alpha * d_scores.get(key, 0.0) + (1 - alpha) * s_scores.get(key, 0.0)
        for key in all_docs
    }
    for key, doc in all_docs.items():
        doc.metadata["hybrid_score"] = combined[key]

    return [all_docs[k] for k in sorted(combined, key=combined.__getitem__, reverse=True)]


def distribution_based_fusion(ranked_lists: list[list[Document]]) -> list[Document]:
    """
    Distribution-Based Score Fusion (DBSF).

    Z-score normalises each list's scores, then averages across lists.
    More robust than min-max to score outliers.
    """
    import math

    key_scores: dict[str, list[float]] = defaultdict(list)
    doc_map:    dict[str, Document]    = {}

    for ranked in ranked_lists:
        raw = [doc.metadata.get("relevance_score", 0.0) for doc in ranked]
        if not raw:
            continue
        # Fall back to rank-based when explicit scores are absent
        if all(s == 0.0 for s in raw):
            raw = [1.0 / (i + 1) for i in range(len(ranked))]
        mean = sum(raw) / len(raw)
        std  = math.sqrt(sum((x - mean) ** 2 for x in raw) / len(raw)) or 1.0
        for doc, score in zip(ranked, raw):
            key = doc.page_content.strip()
            key_scores[key].append((score - mean) / std)
            doc_map[key] = doc

    final = {k: sum(v) / len(v) for k, v in key_scores.items()}
    for doc in doc_map.values():
        doc.metadata["dbsf_score"] = final.get(doc.page_content.strip(), 0.0)

    return [doc_map[k] for k in sorted(final, key=final.__getitem__, reverse=True)]
