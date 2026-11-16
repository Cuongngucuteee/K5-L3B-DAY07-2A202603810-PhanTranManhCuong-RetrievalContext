"""Manual review helpers.

    python -m src.day7.review show --split dev --config A1        # print sheet to read and label
    python -m src.day7.review summarize --split dev --configs A1 A2 A3 A4

Labels live in results/agent/<split>/<config>/reviews.json, written by a human reviewer:
[{query_id, answer_grade (0/1/2, answerable only), citation_support [bool per citation],
  abstention_correct (unanswerable only), reason}]. `summarize` runs starter/agent_metrics.aggregate
on them and writes agent_metrics.json + results/agent/<split>/summary.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

from .common import DATA, RESULTS, ROOT, load_chunks, load_questions, read_json, write_json

sys.path.insert(0, str(ROOT / "starter"))
from agent_metrics import aggregate  # noqa: E402


def show(split: str, config: str) -> None:
    chunks = {c["id"]: c for c in load_chunks()}
    qs = {q["id"]: q for q in load_questions(split)}
    for t in read_json(RESULTS / "agent" / split / config / "traces.json"):
        q = qs[t["query_id"]]
        print("=" * 100)
        print(f"{t['query_id']} [{'có đáp án' if q['answerable'] else 'THIẾU'}] status={t['status']} | {q['question']}")
        print("  EXPECTED:", q.get("expected_answer") or q.get("absence_reason"))
        for e in t["events"]:
            if e["event"] == "search":
                print(f"  SEARCH q={e['query']!r} filter={e['filter']} -> {[r['id'] for r in e['results']]}")
            else:
                print("  EVENT", e["event"])
        print("  ANSWER:", (t.get("answer") or t.get("error") or "").replace("\n", " "))
        print("  CITED:", t["citations"], "invalid:", t.get("invalid_citations"), "ctx_ev:", t.get("auto_context_evidence"))
        for cid in t["citations"]:
            if cid in chunks:
                print(f"    [{cid}] {chunks[cid]['text'][:700].replace(chr(10), ' ')}")


def summarize(split: str, configs: list[str]) -> None:
    cases = [c for c in read_json(DATA / "cases.json") if c["split"] == split]
    rows = []
    for name in configs:
        d = RESULTS / "agent" / split / name
        traces = read_json(d / "traces.json")
        reviews = read_json(d / "reviews.json")
        m = {"split": split, "config": name, **aggregate(cases, traces, reviews)}
        auto = read_json(d / "auto_summary.json")
        write_json(d / "agent_metrics.json", m)
        rows.append((name, m, auto))
    f = lambda x: f"{x['n']}/{x['N']}" + (f" ({x['n'] / x['N']:.0%})" if x["N"] else " (không áp dụng)")
    lines = ["| Cấu hình | Tool | Tool use | Số lượt tìm (phân bố) | Context evidence | Grounded 0/1/2 | TB /2 | Citation hợp lệ | Câu trả lời có citation | Abstention đúng | Từ chối nhầm | Lỗi | Median s/câu | Prompt+completion tokens TB | Chi phí |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m, a in rows:
        g = m["grounded_answer"]
        dist = ", ".join(f"{k} lượt: {v}" for k, v in a["searches_distribution"].items() if v)
        lines.append(f"| {name} | {a['retriever']} | {f(m['tool_use'])} | {dist} | {a['context_evidence']} | {g['0']}/{g['1']}/{g['2']} | "
                     f"{g['mean']:.2f} | {f(m['citation_validity'])} | {f(m['answer_citation_coverage'])} | {f(m['abstention'])} | "
                     f"{f(m['false_abstention'])} | {m['provider_errors']} | {a['median_s']} | {a['avg_prompt_tokens']}+{a['avg_completion_tokens']} | $0 |")
    (RESULTS / "agent" / split / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show"); s.add_argument("--split", required=True); s.add_argument("--config", required=True)
    m = sub.add_parser("summarize"); m.add_argument("--split", required=True); m.add_argument("--configs", nargs="+", required=True)
    a = p.parse_args()
    show(a.split, a.config) if a.cmd == "show" else summarize(a.split, a.configs)


if __name__ == "__main__":
    main()
