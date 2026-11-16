"""Offline checks for src/day7 (no DB, no model). Fakes only test code paths, not quality."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.day7 import agent as agent_mod  # noqa: E402
from src.day7 import context as ctx_mod  # noqa: E402
from src.day7.common import chunk_hits, infer_year_filter  # noqa: E402
from src.day7.retrievers import BM25Retriever, HybridRetriever  # noqa: E402


def chunk(cid, text, year=2019, page=1, part=0):
    return {"id": cid, "text": text, "source_id": f"blld-{year}", "source": "x", "page": page,
            "metadata": {"year": year, "articles": [page], "part": part}}


CHUNKS = [
    chunk("blld-2019-d107-B-00", "Điều 107. Làm thêm giờ không quá 40 giờ trong 01 tháng", page=107),
    chunk("blld-2012-d106-B-00", "Điều 106. Làm thêm giờ không quá 30 giờ trong 01 tháng", year=2012, page=106),
    chunk("blld-2019-d113-B-00", "Điều 113. Nghỉ hằng năm 12 ngày làm việc", page=113),
]


class Common(unittest.TestCase):
    def test_year_filter_from_text_only(self):
        self.assertEqual(infer_year_filter("Theo Bộ luật Lao động 2019, ..."), {"year": 2019})
        self.assertIsNone(infer_year_filter("So sánh 2012 và 2019"))
        self.assertIsNone(infer_year_filter("Làm thêm bao nhiêu giờ?"))

    def test_hit_needs_article_and_full_evidence(self):
        g = [{"source_id": "blld-2019", "page": 107, "evidence": "không quá  40 GIỜ"}]
        self.assertTrue(chunk_hits(CHUNKS[0], g))
        self.assertFalse(chunk_hits(CHUNKS[1], g))
        self.assertFalse(chunk_hits(CHUNKS[0], [{**g[0], "page": 108}]))


class Retrieval(unittest.TestCase):
    def test_bm25_filter_before_ranking(self):
        r = BM25Retriever(CHUNKS, bigrams=True)
        top = r.search("làm thêm giờ trong tháng", filters={"year": 2012}, top_k=3)
        self.assertEqual([c["id"] for c in top], ["blld-2012-d106-B-00"])
        self.assertEqual(top[0]["rank"], 1)
        with self.assertRaises(ValueError):
            r.search("x", filters={"chapter": "I"})

    def test_rrf_uses_ranks(self):
        class Fake:
            def __init__(self, name, order): self.name, self.order = name, order
            def search(self, q, *, filters=None, top_k=3):
                return [{**next(c for c in CHUNKS if c["id"] == i), "rank": n, "score": 100 - n}
                        for n, i in enumerate(self.order[:top_k], 1)]
        a = Fake("dense", ["blld-2019-d113-B-00", "blld-2019-d107-B-00"])
        b = Fake("bm25_bigram", ["blld-2019-d107-B-00", "blld-2012-d106-B-00"])
        top = HybridRetriever(a, b, k=60).search("q", top_k=3)
        self.assertEqual(top[0]["id"], "blld-2019-d107-B-00")  # in both lists
        self.assertEqual(set(top[0]["components"]), {"dense", "bm25_bigram"})


class Context(unittest.TestCase):
    def test_budget_dedupe_and_expansion(self):
        sib = [chunk("blld-2019-d020-B-00", "phần 0", page=20, part=0), chunk("blld-2019-d020-B-01", "phần 1", page=20, part=1)]
        with mock.patch.object(ctx_mod, "_BY_ARTICLE", {("blld-2019", 20): sib}):
            text, chosen = ctx_mod.select_context([sib[1], sib[1]], budget=10_000, expand_article=True)
            self.assertEqual([c["id"] for c in chosen], ["blld-2019-d020-B-01", "blld-2019-d020-B-00"])
            text, chosen = ctx_mod.select_context([sib[1]], budget=60, expand_article=True)
            self.assertLessEqual(len(text), 60)


class AgentLoop(unittest.TestCase):
    def setUp(self):
        self.retr = BM25Retriever(CHUNKS, bigrams=True)
        patch = mock.patch.object(agent_mod, "article_siblings", lambda c: [c])
        patch.start()
        self.addCleanup(patch.stop)

    def run_with(self, replies, question="Theo Bộ luật Lao động 2012, làm thêm bao nhiêu giờ?", **kw):
        it = iter(replies)
        with mock.patch.object(agent_mod, "ollama_chat", lambda *a, **k: next(it)):
            return agent_mod.Agent(self.retr, **kw).run(question)

    @staticmethod
    def call(q, year=None):
        args = {"query": q, **({"year": year} if year else {})}
        return {"message": {"content": "", "tool_calls": [{"function": {"name": "search_law", "arguments": args}}]}}

    @staticmethod
    def say(text):
        return {"message": {"content": text}, "prompt_eval_count": 10, "eval_count": 5}

    def test_user_year_overrides_model_year_and_cites(self):
        r = self.run_with([self.call("làm thêm giờ", year=2019), self.say("30 giờ [blld-2012-d106-B-00]")])
        self.assertEqual(r["status"], "answered")
        self.assertEqual(r["events"][0]["filter"], {"year": 2012})
        self.assertEqual(r["events"][0]["filter_source"], "user_question")
        self.assertEqual(r["citations"], ["blld-2012-d106-B-00"])

    def test_hallucinated_citation_abstains(self):
        r = self.run_with([self.call("làm thêm giờ"), self.say("30 giờ [blld-2019-d999-B-00]")])
        self.assertEqual(r["status"], "abstain")
        self.assertEqual(r["invalid_citations"], ["blld-2019-d999-B-00"])

    def test_no_search_reprompt_then_limit(self):
        r = self.run_with([self.say("search_law(...)"), self.call("a"), self.call("b"), self.say("Không đủ bằng chứng để trả lời.")],
                          max_searches=2)
        events = [e["event"] for e in r["events"]]
        self.assertEqual(events[0], "no_search_reprompt")
        self.assertIn("search_limit", events)
        self.assertEqual(r["n_searches"], 2)
        self.assertEqual(r["status"], "abstain")

    def test_seed_original_and_citation_formats(self):
        r = self.run_with([self.say("30 giờ [chunk-id: blld-2012-d106-B-00]")], seed_original=True)
        self.assertEqual(r["events"][0]["query"], "Theo Bộ luật Lao động 2012, làm thêm bao nhiêu giờ?")
        self.assertEqual(r["citations"], ["blld-2012-d106-B-00"])

    def test_retriever_error_propagates(self):
        class Broken:
            def search(self, *a, **k): raise ConnectionError("db down")
        with mock.patch.object(agent_mod, "ollama_chat", lambda *a, **k: self.call("x")):
            with self.assertRaises(ConnectionError):
                agent_mod.Agent(Broken()).run("q")


if __name__ == "__main__":
    unittest.main()
