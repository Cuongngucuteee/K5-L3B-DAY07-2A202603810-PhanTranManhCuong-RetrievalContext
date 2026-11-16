"""Shared paths, config, normalization and evidence-hit definition for Day 7.

Hit definition (identical for every retrieval method and agent run, copied from Day 6):
a chunk is an evidence hit for a question when it belongs to a gold (source_id, article) pair
AND contains the whole gold evidence string after NFC + lowercase + whitespace normalization.
Alternatives in `gold` are accepted equally. Gold is only read by evaluation code, never by
retrievers or the agent.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "day7"
BENCH = ROOT / "benchmark"
RESULTS = ROOT / "results"


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


load_env()
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding")
DAY6_STRATEGY = os.environ.get("DAY6_STRATEGY", "B")  # article/clause-aligned chunks from Day 6
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
AGENT_MODEL = os.environ.get("AGENT_MODEL", "qwen2.5:7b")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def chunk_hits(chunk: dict, gold: list[dict]) -> bool:
    arts = chunk["metadata"].get("articles", [chunk["page"]])
    return any(chunk["source_id"] == g["source_id"] and g["page"] in arts and norm(g["evidence"]) in norm(chunk["text"])
               for g in gold)


YEAR_RE = re.compile(r"\b(2012|2019)\b")


def infer_year_filter(question: str) -> dict | None:
    """Filter inferred from the user's question text only (never from gold).

    Rule: exactly one of the two law years is written in the question -> {"year": that}.
    Zero or both years -> no filter. This reproduces Day 6's user-supplied `filter` field
    on all 24 benchmark questions (checked in build_data), but is computed from the text.
    """
    years = set(YEAR_RE.findall(question))
    return {"year": int(years.pop())} if len(years) == 1 else None


def load_chunks() -> list[dict]:
    return read_json(DATA / "chunks.json")


def load_questions(split: str | None = None, answerable: bool | None = None) -> list[dict]:
    qs = read_json(DATA / "questions.json")
    if split:
        qs = [q for q in qs if q["split"] == split]
    if answerable is not None:
        qs = [q for q in qs if q["answerable"] == answerable]
    return qs
