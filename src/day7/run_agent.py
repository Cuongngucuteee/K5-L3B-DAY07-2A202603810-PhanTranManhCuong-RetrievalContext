"""Run the real agent on a split for one or more configs; save full traces.

    python -m src.day7.run_agent --split dev --configs A1 A2 A3
    python -m src.day7.run_agent --split test --configs A1 A3      # only after the choice is frozen

Output: results/agent/<split>/<config>/traces.json (one record per question, starter-compatible:
query_id, status, events [search events normalized], citations) and auto_summary.json with
automatic metrics only (tool use, #calls, context evidence, latency, tokens). Answer quality,
citation support and abstention correctness come from human labels in reviews.json.
"""
from __future__ import annotations

import argparse
import statistics
import time
import traceback

from .agent import Agent
from .common import AGENT_MODEL, RESULTS, chunk_hits, load_chunks, load_questions, now, write_json
from .retrievers import build_retriever

# Frozen before dev runs (A3 settings derived from dev retrieval analysis, see report).
CONFIGS = {
    "A1": {"retriever": "dense", "top_k": 3, "max_searches": 3, "budget": 6000, "expand_article": False, "seed_original": False,
           "hypothesis": "Baseline agent: tool = dense pgvector (R1), model tự viết query."},
    "A2": {"retriever": "hybrid", "top_k": 3, "max_searches": 3, "budget": 6000, "expand_article": False, "seed_original": False,
           "hypothesis": "Tool hybrid RRF (R4: Hit@1 13/15 so với 12/15) -> evidence lên hạng cao hơn, ít câu sai/thiếu hơn."},
    "A3": {"retriever": "hybrid", "top_k": 3, "max_searches": 3, "budget": 8000, "expand_article": True, "seed_original": True,
           "hypothesis": "Sửa lỗi dev A1 q12 (query model quá ngắn, tìm 1 lần rồi dừng): lượt đầu luôn dùng câu hỏi gốc; "
                         "+ mở rộng cùng Điều (R7, Ctx 15/15). Budget 8000 để chứa phần mở rộng."},
    "A4": {"retriever": "hybrid_rerank", "top_k": 3, "max_searches": 3, "budget": 8000, "expand_article": True, "seed_original": True,
           "hypothesis": "A3 + cross-encoder rerank (R6) -> tăng Hit@1, giảm chunk nhiễu trong context."},
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=("dev", "test"))
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--ids", nargs="*", help="chỉ chạy các ID này (debug)")
    args = p.parse_args()
    questions = load_questions(args.split)
    if args.ids:
        questions = [q for q in questions if q["id"] in args.ids]
    chunk_by_id = {c["id"]: c for c in load_chunks()}
    cache: dict = {}
    for name in args.configs:
        cfg = CONFIGS[name]
        agent = Agent(build_retriever(cfg["retriever"], cache=cache), top_k=cfg["top_k"], max_searches=cfg["max_searches"],
                      budget=cfg["budget"], expand_article=cfg["expand_article"], seed_original=cfg["seed_original"])
        traces = []
        t0 = time.perf_counter()
        for q in questions:
            try:
                r = agent.run(q["question"])
            except Exception as e:  # recorded, never replaced by a fake answer
                r = {"status": "error", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
                     "events": [], "citations": [], "answer": None, "context_ids": []}
            rec = {"query_id": q["id"], "split": q["split"], "config": name, "question": q["question"], **r}
            if q["answerable"] and r["status"] != "error":  # evaluation-side only
                rec["auto_context_evidence"] = any(chunk_hits(chunk_by_id[i], q["gold"]) for i in r["context_ids"])
                rec["auto_cited_evidence"] = any(chunk_hits(chunk_by_id[i], q["gold"]) for i in r["citations"] if i in chunk_by_id)
            traces.append(rec)
            print(f"[{name}] {q['id']} {rec['status']:8} searches={r.get('n_searches')} {r.get('elapsed_ms')}ms "
                  f"ctx_ev={rec.get('auto_context_evidence')} | {(r.get('answer') or r.get('error') or '')[:110]!r}", flush=True)
        ok = [t for t in traces if t["status"] != "error"]
        ans = [t for t in ok if "auto_context_evidence" in t]
        summary = {"config": name, **cfg, "model": AGENT_MODEL, "split": args.split, "run_at": now(),
                   "n": len(traces), "errors": len(traces) - len(ok),
                   "tool_use": f"{sum(t['n_searches'] > 0 for t in ok)}/{len(ok)}",
                   "searches_distribution": {k: sum(t['n_searches'] == k for t in ok) for k in range(0, cfg["max_searches"] + 1)},
                   "context_evidence": f"{sum(t['auto_context_evidence'] for t in ans)}/{len(ans)}",
                   "cited_evidence": f"{sum(t['auto_cited_evidence'] for t in ans)}/{len(ans)}",
                   "answered": sum(t["status"] == "answered" for t in ok), "abstained": sum(t["status"] == "abstain" for t in ok),
                   "invalid_citation_ids": sum(len(t.get("invalid_citations", [])) for t in ok),
                   "median_s": round(statistics.median(t["elapsed_ms"] for t in ok) / 1000, 1) if ok else None,
                   "total_s": round(time.perf_counter() - t0, 1),
                   "avg_prompt_tokens": round(statistics.mean(t["usage"]["prompt_tokens"] for t in ok)) if ok else None,
                   "avg_completion_tokens": round(statistics.mean(t["usage"]["completion_tokens"] for t in ok)) if ok else None,
                   "avg_context_chars": round(statistics.mean(t["context_chars"] for t in ok)) if ok else None,
                   "cost_usd": 0}
        out = RESULTS / "agent" / args.split / name
        write_json(out / ("traces.json" if not args.ids else "traces_debug.json"), traces)
        if not args.ids:
            write_json(out / "auto_summary.json", summary)
        print(summary)


if __name__ == "__main__":
    main()
