"""Context selection: dedupe, keep rank order, optional same-article neighbour expansion, char budget.

Budget unit = characters of the rendered context block (header + text), the same unit the
starter uses. Rough token estimate for Vietnamese with qwen2.5 tokenizer: ~1 token / 3 chars.
"""
from __future__ import annotations

from .common import load_chunks

_BY_ARTICLE: dict[tuple[str, int], list[dict]] | None = None


def article_siblings(chunk: dict) -> list[dict]:
    """All chunks of the same (law, article) in reading order (Day 6 metadata part/parts)."""
    global _BY_ARTICLE
    if _BY_ARTICLE is None:
        _BY_ARTICLE = {}
        for c in load_chunks():
            _BY_ARTICLE.setdefault((c["source_id"], c["page"]), []).append(c)
        for v in _BY_ARTICLE.values():
            v.sort(key=lambda c: c["metadata"].get("part", 0))
    return _BY_ARTICLE.get((chunk["source_id"], chunk["page"]), [chunk])


def render(c: dict) -> str:
    year = c["metadata"].get("year")
    return f"[{c['id']}] (Bộ luật Lao động {year}, Điều {c['page']})\n{c['text']}"


def select_context(ranked: list[dict], *, budget: int, expand_article: bool = False) -> tuple[str, list[dict]]:
    """Greedy by rank. With expand_article, the other parts of a hit's article follow it right after
    (keeps clauses/conditions of one Điều together). Blocks that do not fit are skipped, not cut."""
    order, seen = [], set()
    for c in ranked:
        group = article_siblings(c) if expand_article else [c]
        if expand_article:  # the retrieved part first, then its siblings
            group = [c] + [s for s in group if s["id"] != c["id"]]
        for g in group:
            if g["id"] not in seen:
                seen.add(g["id"])
                order.append(g if g["id"] != c["id"] else c)
    blocks, chosen, used = [], [], 0
    for c in order:
        b = render(c)
        extra = len(b) + (2 if blocks else 0)
        if used + extra <= budget:
            blocks.append(b)
            chosen.append(c)
            used += extra
    return "\n\n".join(blocks), chosen
