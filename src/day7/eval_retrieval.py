"""Retrieval + context benchmark (no LLM): original question as query, same corpus/index/labels.

    python -m src.day7.eval_retrieval --split dev                 # all configs in CONFIGS
    python -m src.day7.eval_retrieval --split test --configs R1 R5

Per config writes results/retrieval/<split>/<config>.json (per-query trace: filter, top-10 ranking
with hit flags, context ids) and results/retrieval/<split>/summary.{json,md}.
Metrics on answerable questions only: Hit@1, Evidence Hit@3, Context Evidence Rate (n/N), MRR@10.
Unanswerable questions are run (trace kept) but excluded from the denominators.
"""
from __future__ import annotations

import argparse
import statistics
import time

from .common import RESULTS, chunk_hits, infer_year_filter, load_questions, now, write_json
from .context import select_context
from .retrievers import build_retriever, timed_search

# Fixed before running dev. top_k = chunks offered to context; ranking metrics always use top-3/top-10.
CONFIGS = {
    "R0": {"method": "dense", "filter": False, "top_k": 3, "budget": 4000, "expand_article": False,
           "hypothesis": "Day 6 dense (index B) không filter — điểm xuất phát."},
    "R1": {"method": "dense", "filter": True, "top_k": 3, "budget": 4000, "expand_article": False,
           "hypothesis": "BASELINE Day 7: dense + filter năm suy từ câu hỏi (= Day 6 C)."},
    "R2": {"method": "bm25_syllable", "filter": True, "top_k": 3, "budget": 4000, "expand_article": False,
           "hypothesis": "Lexical theo âm tiết: khớp tốt số liệu/thuật ngữ pháp lý nhưng mất nghĩa từ ghép."},
    "R3": {"method": "bm25_bigram", "filter": True, "top_k": 3, "budget": 4000, "expand_article": False,
           "hypothesis": "Thêm bigram âm tiết xấp xỉ từ ghép tiếng Việt -> xếp hạng lexical tốt hơn R2."},
    "R4": {"method": "hybrid", "filter": True, "top_k": 3, "budget": 4000, "expand_article": False,
           "hypothesis": "RRF dense + BM25 bigram: sửa câu dense xếp evidence ở hạng 2-3 (q01,q05,q12)."},
    "R5": {"method": "hybrid_rerank", "filter": True, "top_k": 3, "budget": 4000, "expand_article": False,
           "hypothesis": "Cross-encoder rerank 20 candidate hybrid -> tăng Hit@1."},
    "R6": {"method": "hybrid_rerank", "filter": True, "top_k": 3, "budget": 6000, "expand_article": True,
           "hypothesis": "Giữ các phần cùng Điều cạnh hit -> context đủ điều kiện hơn; budget 6000 để vừa."},
    "R7": {"method": "hybrid", "filter": True, "top_k": 3, "budget": 6000, "expand_article": True,
           "hypothesis": "Thêm sau lỗi q05/q12 trên dev: hit nằm ở phần khác cùng Điều -> mở rộng Điều, không rerank."},
}


def run_config(name: str, cfg: dict, questions: list[dict], cache: dict) -> dict:
    retr = build_retriever(cfg["method"], cache=cache)
    traces = []
    for q in questions:
        filt = infer_year_filter(q["question"]) if cfg["filter"] else None
        rows, ms = timed_search(retr, q["question"], filters=filt, top_k=max(10, cfg["top_k"]))
        context, chosen = select_context(rows[:cfg["top_k"]], budget=cfg["budget"], expand_article=cfg["expand_article"])
        t = {"query_id": q["id"], "split": q["split"], "answerable": q["answerable"], "question": q["question"],
             "query": q["question"], "filter": filt, "elapsed_ms": round(ms, 1),
             "results": [{"rank": r["rank"], "id": r["id"], "source_id": r["source_id"], "page": r["page"],
                          "score": r["score"], **({"components": r["components"]} if "components" in r else {}),
                          **({"first_stage": r["first_stage"]} if "first_stage" in r else {}),
                          "hit": chunk_hits(r, q["gold"]) if q["answerable"] else None} for r in rows],
             "context_ids": [c["id"] for c in chosen], "context_chars": len(context)}
        if q["answerable"]:
            ranks = [r["rank"] for r in rows if chunk_hits(r, q["gold"])]
            t.update({"hit1": bool(ranks) and ranks[0] == 1, "hit3": any(r <= 3 for r in ranks),
                      "first_hit_rank": ranks[0] if ranks else None,
                      "context_hit": any(chunk_hits(c, q["gold"]) for c in chosen)})
        traces.append(t)
    ans = [t for t in traces if t["answerable"]]
    n = len(ans)
    summary = {"config": name, **cfg, "split": traces[0]["split"], "N_answerable": n,
               "hit1": sum(t["hit1"] for t in ans), "hit3": sum(t["hit3"] for t in ans),
               "context_evidence": sum(t["context_hit"] for t in ans),
               "mrr10": round(sum(1 / t["first_hit_rank"] for t in ans if t["first_hit_rank"]) / n, 4),
               "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces)),
               "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 1),
               "cost_usd": 0, "misses_top3": [t["query_id"] for t in ans if not t["hit3"]],
               "not_rank1": [t["query_id"] for t in ans if not t["hit1"]]}
    return {"summary": summary, "traces": traces}


def frac(n, d):
    return f"{n}/{d} ({n / d:.0%})"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--configs", nargs="*", default=list(CONFIGS))
    args = p.parse_args()
    questions = load_questions(args.split)
    cache: dict = {}
    out_dir = RESULTS / "retrieval" / args.split
    summaries = []
    for name in args.configs:
        t = time.perf_counter()
        res = run_config(name, CONFIGS[name], questions, cache)
        res["summary"]["wall_seconds"] = round(time.perf_counter() - t, 1)
        res["summary"]["run_at"] = now()
        write_json(out_dir / f"{name}.json", res)
        summaries.append(res["summary"])
        s = res["summary"]
        print(f"{name:3} {s['method']:14} H@1 {frac(s['hit1'], s['N_answerable']):12} H@3 {frac(s['hit3'], s['N_answerable']):12} "
              f"Ctx {frac(s['context_evidence'], s['N_answerable']):12} MRR {s['mrr10']} {s['median_ms']}ms miss3={s['misses_top3']} notR1={s['not_rank1']}")
    write_json(out_dir / "summary.json", summaries)
    lines = ["| Cấu hình | Phương pháp | Filter | top-k ctx | Budget (ký tự) | Mở rộng Điều | Hit@1 | Evidence Hit@3 | Context Evidence Rate | MRR@10 | Ctx TB (ký tự) | Median ms/query | Chi phí |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        N = s["N_answerable"]
        lines.append(f"| {s['config']} | {s['method']} | {'năm' if s['filter'] else '-'} | {s['top_k']} | {s['budget']} | "
                     f"{'có' if s['expand_article'] else '-'} | {frac(s['hit1'], N)} | {frac(s['hit3'], N)} | "
                     f"{frac(s['context_evidence'], N)} | {s['mrr10']} | {s['avg_context_chars']} | {s['median_ms']} | $0 |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
