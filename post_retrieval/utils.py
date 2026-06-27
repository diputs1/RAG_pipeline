"""
post_retrieval/utils.py
=======================
Shared helpers used across post-retrieval modules.
"""

from __future__ import annotations

from langchain_core.documents import Document


_SCORE_KEYS = ("rerank_score", "relevance_score", "rrf_score", "hybrid_score", "bm25_score", "score")


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


def call_llm(
    prompt:      str,
    provider:    str,
    model:       str,
    max_tokens:  int   = 512,
    temperature: float = 0.0,
) -> str:
    """Call an LLM and return the raw text response."""
    if provider == "openai":
        from openai import OpenAI
        r = OpenAI().chat.completions.create(
            model=model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content.strip()

    if provider == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text.strip()

    if provider == "google":
        import google.generativeai as genai
        return genai.GenerativeModel(model).generate_content(prompt).text.strip()

    raise ValueError(f"Unsupported LLM provider: '{provider}'")
