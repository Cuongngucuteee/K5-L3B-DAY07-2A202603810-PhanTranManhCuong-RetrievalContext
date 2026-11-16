"""Export the Day 6 index (strategy B rows of pgvector) and build the Day 7 benchmark.

    python -m src.day7.build_data --day6 ../K5-L3B-DAY06-PhanTranManhCuong-2A202603810-DataIngestion

Writes data/day7/chunks.json (Day 7 chunk format, no vectors), data/day7/questions.json
(20 answerable Day 6 questions + benchmark/unanswerable.json) and data/day7/cases.json.
Every gold evidence string is verified to sit inside an exported chunk of the right article.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import psycopg

from .common import BENCH, DATA, DATABASE_URL, DAY6_STRATEGY, EMBEDDING_MODEL, chunk_hits, infer_year_filter, now, read_json, write_json


def export_chunks() -> tuple[list[dict], dict]:
    with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute("""SELECT chunk_id, document_id, source, page, articles, year, text, metadata, model,
                              vector_dims(embedding)
                       FROM chunks WHERE strategy = %s ORDER BY chunk_id""", (DAY6_STRATEGY,))
        rows = cur.fetchall()
    if not rows:
        raise SystemExit(f"Không có chunk strategy {DAY6_STRATEGY} trong pgvector — chạy ingest của Day 6 trước.")
    models = {r[8] for r in rows}
    dims = {r[9] for r in rows}
    if models != {EMBEDDING_MODEL}:
        raise SystemExit(f"Index dùng model {models}, cấu hình Day 7 là {EMBEDDING_MODEL}")
    chunks = []
    for cid, doc, source, page, arts, year, text, meta, _, _ in rows:
        meta = meta if isinstance(meta, dict) else json.loads(meta)
        chunks.append({"id": cid, "text": text, "source_id": doc, "source": source, "page": page,
                       "metadata": {**meta, "articles": list(arts), "year": year}})
    info = {"exported_at": now(), "strategy": DAY6_STRATEGY, "rows": len(chunks), "embedding_model": models.pop(),
            "dimensions": dims.pop(), "articles": len({(c["source_id"], c["page"]) for c in chunks})}
    return chunks, info


def build_questions(day6: Path, chunks: list[dict]) -> list[dict]:
    out = []
    for line in (day6 / "benchmark" / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        gold = [{"source_id": q["gold_source_id"], "page": q["gold_page"], "evidence": q["evidence_quote"]}]
        gold += [{"source_id": a["source_id"], "page": a["page"], "evidence": a["evidence_quote"]}
                 for a in q.get("alternatives") or []]
        for g in gold:
            if not any(chunk_hits(c, [g]) for c in chunks):
                raise SystemExit(f"{q['id']}: evidence không nằm trọn trong chunk nào của {g['source_id']} Điều {g['page']}")
        inferred = infer_year_filter(q["question"])
        if inferred != (q.get("filter") or None):
            raise SystemExit(f"{q['id']}: filter suy từ câu hỏi {inferred} khác filter người dùng Day 6 {q.get('filter')}")
        out.append({"id": q["id"], "split": q["split"], "type": q["type"], "question": q["question"],
                    "answerable": True, "expected_behavior": "search_and_answer",
                    "expected_answer": q["expected_answer"], "gold": gold, "filter": q.get("filter")})
    for u in read_json(BENCH / "unanswerable.json"):
        if infer_year_filter(u["question"]) is not None:
            u = {**u, "filter": infer_year_filter(u["question"])}
        out.append({**u, "filter": u.get("filter")})
    ids = [q["id"] for q in out]
    if len(ids) != len(set(ids)):
        raise SystemExit("ID câu hỏi trùng")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--day6", type=Path, required=True, help="Thư mục repo Day 6 (chứa benchmark/questions.jsonl)")
    args = p.parse_args()
    chunks, info = export_chunks()
    questions = build_questions(args.day6, chunks)
    write_json(DATA / "chunks.json", chunks)
    write_json(DATA / "questions.json", questions)
    write_json(DATA / "cases.json", [{"id": q["id"], "split": q["split"], "answerable": q["answerable"]} for q in questions])
    counts = {s: {"answerable": sum(q["split"] == s and q["answerable"] for q in questions),
                  "unanswerable": sum(q["split"] == s and not q["answerable"] for q in questions)} for s in ("dev", "test")}
    info.update({"questions": len(questions), "split_counts": counts})
    write_json(DATA / "export_info.json", info)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
