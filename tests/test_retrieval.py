"""
Isolated retrieval-quality tests for LifeVault (Phase 2).

Safety guarantees (by construction):
  * lifevault.py is imported only with sentence_transformers and ollama
    replaced by fakes, CWD switched to pytest's tmp_path, and a guard that the
    CWD is not the project root. No real model, Ollama server, vault,
    chat_history.md or production ChromaDB is touched.
  * ask_llm / ask_llm_stream / save_chat_memory are replaced per test with
    recorders, so no LLM call or memory write happens.
  * TestRealChroma uses the REAL chromadb package (if installed) but only
    against ./.lifevault_db inside tmp_path, with telemetry disabled.

Run:  python -m pytest tests/test_retrieval.py -v
"""

import logging
import math
import os
import random
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ══════════════════════════════════════════════════════════════════════════════
# Fakes
# ══════════════════════════════════════════════════════════════════════════════
VOCAB = ["cat", "dog", "tax", "car", "python", "recipe", "shadow", "whiskers",
         "color", "purple", "garden", "invoice", "budget", "vacation"]


class _Vec(list):
    def tolist(self):
        return list(self)


def keyword_embed(text: str) -> _Vec:
    """Deterministic bag-of-keywords vector, unit-normalised (like MiniLM's
    Normalize layer). A tiny bias dimension avoids zero vectors."""
    words = [w.strip(".,!?:;()[]\"'*#").lower() for w in str(text).split()]
    v = [float(sum(w == k for w in words)) for k in VOCAB] + [1e-3]
    n = math.sqrt(sum(x * x for x in v))
    return _Vec(x / n for x in v)


class _FakeCollection:
    """Minimal Chroma stand-in: squared-L2 ranking, `where` filter, include."""

    def __init__(self):
        self.items: dict[str, dict] = {}
        self.fail_on: set = set()  # e.g. {"document"} → filtered query raises

    def count(self):
        return len(self.items)

    def get(self, ids=None, where=None, **_):
        if ids is not None:
            return {"ids": [i for i in ids if i in self.items]}
        return {"ids": [i for i, v in self.items.items()
                        if not where or all(v["meta"].get(k) == w for k, w in where.items())]}

    def add(self, ids, embeddings, documents, metadatas):
        for i, e, d, m in zip(ids, embeddings, documents, metadatas):
            self.items[i] = {"emb": list(e), "doc": d, "meta": m}

    def query(self, query_embeddings, n_results, include=None, where=None):
        if n_results <= 0:
            raise TypeError("Number of requested results 0, cannot be negative, or zero.")
        if where and where.get("type") in self.fail_on:
            raise RuntimeError("simulated filtered-query failure")
        q = query_embeddings[0]
        pool = [(i, v) for i, v in self.items.items()
                if not where or all(v["meta"].get(k) == w for k, w in where.items())]
        d2 = lambda e: sum((a - b) ** 2 for a, b in zip(q, e))  # noqa: E731
        pool.sort(key=lambda iv: d2(iv[1]["emb"]))
        pool = pool[:n_results]
        return {
            "ids": [[i for i, _ in pool]],
            "documents": [[v["doc"] for _, v in pool]],
            "metadatas": [[v["meta"] for _, v in pool]],
            "distances": [[d2(v["emb"]) for _, v in pool]],
            "embeddings": [[v["emb"] for _, v in pool]],
        }


def _fake_st():
    st = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, *a, **k):
            pass

        def encode(self, text):
            return keyword_embed(text)

    st.SentenceTransformer = SentenceTransformer
    return st


def _fake_ollama():
    o = types.ModuleType("ollama")
    o.list = lambda: {"models": []}

    def _no_network(*a, **k):
        raise RuntimeError("ollama is stubbed in tests")

    o.chat = _no_network
    return o


def _import_engine(tmp_path, monkeypatch, real_chroma: bool):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LIFEVAULT_ALLOWED_ROOTS", raising=False)
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    monkeypatch.setitem(sys.modules, "sentence_transformers", _fake_st())
    monkeypatch.setitem(sys.modules, "ollama", _fake_ollama())
    if not real_chroma:
        col = _FakeCollection()
        chromadb = types.ModuleType("chromadb")
        chromadb.PersistentClient = lambda path=None, **_: types.SimpleNamespace(
            get_or_create_collection=lambda name, **__: col)
        monkeypatch.setitem(sys.modules, "chromadb", chromadb)
    monkeypatch.delitem(sys.modules, "lifevault", raising=False)
    # Guard: never import the engine against the real project directory.
    assert Path.cwd().resolve() == tmp_path.resolve() != ROOT
    import lifevault

    return lifevault


def _teardown_engine():
    lg = logging.getLogger("LifeVault")
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()
    sys.modules.pop("lifevault", None)


class LLM:
    """Records prompts; replies from a script (callable or list)."""

    def __init__(self, replies=None, stream_reply="ok"):
        self.prompts: list[str] = []
        self.stream_prompts: list[str] = []
        self.replies = replies
        self.stream_reply = stream_reply

    def ask(self, prompt, timeout=None):
        self.prompts.append(prompt)
        r = self.replies
        if callable(r):
            return r(prompt)
        if isinstance(r, list):
            return r.pop(0) if r else "answer"
        return "answer"

    def stream(self, prompt):
        self.stream_prompts.append(prompt)
        yield from self.stream_reply.split(" ")


@pytest.fixture
def lv(tmp_path, monkeypatch):
    mod = _import_engine(tmp_path, monkeypatch, real_chroma=False)
    saved = []
    monkeypatch.setattr(mod, "save_chat_memory", lambda q, a, epoch=None: saved.append((q, a)))
    mod._test_saved = saved
    yield mod
    _teardown_engine()


def add(mod, cid, text, kind="document", source=None, emb=None):
    mod.collection.add(
        ids=[cid],
        embeddings=[list(emb) if emb is not None else mod.embedder.encode(text).tolist()],
        documents=[text],
        metadatas=[{"source": source or (f"{cid}.md" if kind == "document" else "chat_history.md"),
                    "type": kind}],
    )


def mem_text(day: int, words: str, pad: int = 700) -> str:
    return (f"### [2026-01-{day:02d} 10:00 AM] User: {words}\n\n"
            f"### [2026-01-{day:02d} 10:00 AM] LifeVault: {words} " + "x" * pad)


def sections(ctx: str) -> tuple[str, str]:
    head_d = "### Notes from your files\n"
    head_m = "\n\n### Conversation memories (oldest → newest)\n"
    assert ctx.startswith(head_d) and head_m in ctx
    doc, mem = ctx[len(head_d):].split(head_m)
    return doc, mem


def chunk(cid, text, kind="document", rel=0.5, source=None):
    return {"id": cid, "text": text, "source": source or f"{cid}.md", "_type": kind,
            "distance": None, "relevance": rel}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Relevance scoring
# ══════════════════════════════════════════════════════════════════════════════
class TestCosine:
    @pytest.mark.parametrize("a,b,expected", [
        ([1, 0, 0], [1, 0, 0], 1.0),
        ([1, 0, 0], [0, 1, 0], 0.0),
        ([1, 0, 0], [-1, 0, 0], -1.0),
        ([1, 0, 0], [5, 0, 0], 1.0),           # norm-independent (1 - d/2 would give -7)
        ([1, 1, 0], [1, 0, 0], 1 / math.sqrt(2)),
    ])
    def test_values(self, lv, a, b, expected):
        assert lv._cosine(a, b) == pytest.approx(expected)

    @pytest.mark.parametrize("a,b", [
        (None, [1.0]), ([1.0], None), ([], []), ([1, 0], [1, 0, 0]),
        ([0, 0], [1, 0]), ([float("nan"), 1], [1, 1]), ([float("inf"), 1], [1, 1]),
        (["a", "b"], [1, 1]),
    ])
    def test_undefined_returns_none(self, lv, a, b):
        assert lv._cosine(a, b) is None

    def test_clamped_to_unit_interval(self, lv):
        v = [0.1] * 384
        assert -1.0 <= lv._cosine(v, v) <= 1.0


class TestSelect:
    def rows(self, *rels, layer="memory"):
        return [dict(chunk(f"c{i}", f"text {i}", layer, r), _rank=i) for i, r in enumerate(rels)]

    def test_threshold_boundary_inclusive(self, lv):
        out = lv._select(self.rows(0.25, 0.2499, 0.9), "memory", 10, 0.25)
        assert [c["relevance"] for c in out] == [0.9, 0.25]

    @pytest.mark.parametrize("thr", [0.0, None, -0.5, float("nan"), float("inf"), "0.3", True])
    def test_disabled_or_invalid_threshold_filters_nothing(self, lv, thr):
        assert len(lv._select(self.rows(0.9, 0.01, -0.4), "memory", 10, thr)) == 3

    def test_unscored_never_filtered_and_ranked_last(self, lv):
        out = lv._select(self.rows(None, 0.1, 0.8), "memory", 10, 0.25)
        assert [c["relevance"] for c in out] == [0.8, None]

    def test_dedup_by_normalised_text_keeps_best(self, lv):
        rows = [dict(chunk("a", "My  Cat is\nShadow", "document", 0.3), _rank=0),
                dict(chunk("b", "my cat is shadow", "document", 0.7), _rank=1)]
        out = lv._select(rows, "document", 10, 0.0)
        assert [c["id"] for c in out] == ["b"]

    def test_dedup_by_id(self, lv):
        rows = [dict(chunk("a", "t1", "document", 0.7), _rank=0),
                dict(chunk("a", "t1", "document", 0.7), _rank=0)]
        assert len(lv._select(rows, "document", 10, 0.0)) == 1

    def test_k_cap_and_layer_separation(self, lv):
        rows = self.rows(0.9, 0.8, 0.7) + self.rows(0.95, layer="document")
        assert len(lv._select(rows, "memory", 2, 0.0)) == 2
        assert all(c["_type"] == "memory" for c in lv._select(rows, "memory", 5, 0.0))

    def test_rank_key_removed(self, lv):
        assert "_rank" not in lv._select(self.rows(0.5), "memory", 1, 0.0)[0]


class TestRetrieveFake:
    def test_returns_docs_then_memories_with_scores(self, lv):
        add(lv, "d1", "my cat whiskers")
        add(lv, "m1", mem_text(1, "cat food"), "memory")
        out = lv.retrieve("cat")
        assert [c["_type"] for c in out] == ["document", "memory"]
        assert all(set(c) >= {"id", "text", "source", "_type", "distance", "relevance"} for c in out)
        assert out[0]["relevance"] > 0.5

    def test_unrelated_memory_filtered_documents_never(self, lv):
        add(lv, "d1", "invoice budget")                # unrelated document
        add(lv, "m1", mem_text(1, "tax"), "memory")    # unrelated memory
        out = lv.retrieve("cat whiskers")
        assert [c["id"] for c in out] == ["d1"]        # doc kept (threshold 0.0), memory dropped

    def test_fallback_preserves_real_metadata_type(self, lv):
        add(lv, "d1", "cat whiskers")
        add(lv, "m1", mem_text(1, "cat"), "memory")
        lv.collection.fail_on = {"document"}
        out = lv.retrieve("cat")
        kinds = {c["id"]: c["_type"] for c in out}
        assert kinds == {"d1": "document", "m1": "memory"}  # not mislabelled

    def test_small_collection_and_zero_k(self, lv):
        add(lv, "d1", "cat")
        assert len(lv.retrieve("cat", memory_k=10, notes_k=8)) == 1
        assert lv.retrieve("cat", memory_k=0, notes_k=0) == []

    def test_empty_db(self, lv):
        assert lv.retrieve("anything") == []

    def test_duplicate_documents_collapsed(self, lv):
        add(lv, "a", "cat whiskers shadow", source="a/notes.md")
        add(lv, "b", "Cat  whiskers SHADOW", source="b/notes.md")
        assert len([c for c in lv.retrieve("cat") if c["_type"] == "document"]) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 2. Context budgets
# ══════════════════════════════════════════════════════════════════════════════
class TestBuildContext:
    def test_memories_cannot_exclude_documents(self, lv):
        mems = [chunk(f"m{i}", mem_text(i + 1, "cat"), "memory", 0.99) for i in range(10)]
        docs = [chunk(f"d{i}", "doc " + "y" * 700, "document", 0.4) for i in range(3)]
        ctx, used = lv.build_context(mems + docs, 4000, 2000)
        doc_body, mem_body = sections(ctx)
        assert sum(c["_type"] == "document" for c in used) == 3
        # Docs needed < 4000, so only their spare budget may go to memories.
        assert len(mem_body) <= 2000 + (4000 - len(doc_body))
        assert len(doc_body) + len(mem_body) <= 6000

    def test_memory_reservation_holds_when_documents_need_full_budget(self, lv):
        mems = [chunk(f"m{i}", mem_text(i + 1, "cat"), "memory", 0.99) for i in range(10)]
        docs = [chunk(f"d{i}", "doc " + "y" * 900, "document", 0.1) for i in range(10)]
        ctx, used = lv.build_context(mems + docs, 4000, 2000)
        doc_body, mem_body = sections(ctx)
        assert len(mem_body) <= 2000 and len(doc_body) <= 4000
        assert sum(c["_type"] == "document" for c in used) >= 4

    def test_separators_counted_exactly(self, lv):
        docs = [chunk(f"d{i}", "z" * 100, rel=0.5) for i in range(50)]
        ctx, used = lv.build_context(docs, 1000, 0)
        doc_body, _ = sections(ctx)
        entries = [lv._entry(c) for c in used if not c.get("truncated")]
        assert len(doc_body) <= 1000
        assert doc_body.count(lv._CONTEXT_SEP) == len(used) - 1
        assert sum(map(len, entries)) + len(lv._CONTEXT_SEP) * (len(used) - 1) <= 1000

    def test_oversize_chunk_skipped_and_scan_continues(self, lv):
        big = chunk("big", "b" * 900, rel=0.9)
        small = chunk("small", "s", rel=0.1)
        first = chunk("first", "f" * 300, rel=0.95)
        budget = len(lv._entry(first)) + len(lv._CONTEXT_SEP) + len(lv._entry(small)) + 50
        ctx, used = lv.build_context([first, big, small], budget, 0, min_truncated=200)
        assert [c["id"] for c in used] == ["first", "small"]

    def test_truncates_only_with_enough_room(self, lv):
        first = chunk("first", "f" * 100, rel=0.9)
        big = chunk("big", "b" * 2000, rel=0.8)
        budget = len(lv._entry(first)) + len(lv._CONTEXT_SEP) + 250
        ctx, used = lv.build_context([first, big], budget, 0, min_truncated=200)
        assert [c["id"] for c in used] == ["first", "big"] and used[1]["truncated"] is True
        doc_body, _ = sections(ctx)
        assert len(doc_body) == budget and doc_body.endswith(lv._TRUNC_MARKER)

    def test_unused_memory_budget_rolls_over_to_documents(self, lv):
        docs = [chunk(f"d{i}", "y" * 900, rel=0.5) for i in range(10)]
        _, used_none = lv.build_context(docs, 4000, 2000)
        doc_len = len(sections(lv.build_context(docs, 4000, 2000)[0])[0])
        assert 4000 < doc_len <= 6000
        assert len(used_none) > len(lv.build_context(docs, 4000, 0)[1])

    def test_unused_document_budget_rolls_over_to_memories(self, lv):
        mems = [chunk(f"m{i}", mem_text(i + 1, "cat"), "memory") for i in range(10)]
        _, mem_body = sections(lv.build_context(mems, 4000, 2000)[0])
        assert 2000 < len(mem_body) <= 6000

    def test_partial_rollover_is_exact(self, lv):
        mems = [chunk("m1", mem_text(1, "cat", pad=100), "memory")]
        docs = [chunk(f"d{i}", "y" * 900, rel=0.5) for i in range(10)]
        doc_body, mem_body = sections(lv.build_context(mems + docs, 4000, 2000)[0])
        assert len(doc_body) <= 4000 + (2000 - len(mem_body))
        assert len(doc_body) + len(mem_body) <= 6000

    def test_empty_sources_are_explicit(self, lv):
        ctx, used = lv.build_context([], 4000, 2000)
        assert used == []
        assert "(No relevant notes found in your files.)" in ctx
        assert "(No relevant conversation memories.)" in ctx

    def test_source_attribution_and_scores(self, lv):
        ctx, _ = lv.build_context([chunk("d1", "hello", rel=0.62, source="taxes/2025.pdf"),
                                   chunk("m1", "hi", "memory", rel=None, source="chat_history.md")])
        assert "[Source: taxes/2025.pdf · note · relevance 0.62]" in ctx
        assert "[Source: chat_history.md · memory · relevance n/a]" in ctx

    def test_memories_selected_by_relevance_shown_chronologically(self, lv):
        new = chunk("new", mem_text(9, "cat", pad=10), "memory", 0.9)
        old = chunk("old", mem_text(2, "cat", pad=10), "memory", 0.5)
        _, mem_body = sections(lv.build_context([new, old], 0, 2000)[0])
        assert mem_body.index("2026-01-02") < mem_body.index("2026-01-09")

    def test_negative_and_zero_budgets(self, lv):
        ctx, used = lv.build_context([chunk("d", "x")], -5, 0)
        assert used == []

    def test_randomised_budget_invariants(self, lv):
        rng = random.Random(1234)
        for _ in range(200):
            cs = [chunk(f"c{i}", "w" * rng.randint(1, 1500),
                        rng.choice(["document", "memory"]), rng.random()) for i in range(rng.randint(0, 15))]
            d, m = rng.randint(0, 5000), rng.randint(0, 3000)
            ctx, used = lv.build_context(cs, d, m)
            doc_body, mem_body = sections(ctx)
            real_d = 0 if not any(c["_type"] == "document" for c in used) else len(doc_body)
            real_m = 0 if not any(c["_type"] == "memory" for c in used) else len(mem_body)
            assert real_d + real_m <= d + m
            assert {c["id"] for c in used} <= {c["id"] for c in cs}


# ══════════════════════════════════════════════════════════════════════════════
# 3. Sub-question parsing
# ══════════════════════════════════════════════════════════════════════════════
class TestParser:
    def test_preamble_ignored(self, lv):
        out = lv._parse_sub_questions("Here are the sub-questions:\n1. What is X?\n2) When did Y happen?")
        assert out == ["What is X?", "When did Y happen?"]

    def test_bullets_and_markdown(self, lv):
        out = lv._parse_sub_questions("- **Who paid the invoice?**\n• Which car was bought\n* `Where is it`?")
        assert out == ["Who paid the invoice?", "Which car was bought?", "Where is it?"]

    def test_duplicates_and_restatement_dropped(self, lv):
        text = "1. What is my cat's name?\n2. what is my CAT'S  name\n3. When was it renamed?"
        assert lv._parse_sub_questions(text, original="What is my cat's name") == ["When was it renamed?"]

    @pytest.mark.parametrize("text", ["", "   ", None, "❌ LLM failed after 3 attempts: boom",
                                      "Sure! I can help with that.", "First, consider the budget.",
                                      "1. Why?", "---\n***"])
    def test_unusable_output_returns_empty(self, lv, text):
        assert lv._parse_sub_questions(text) == []

    # [RET-07] Table-driven: (model output, original question, expected sub-questions)
    QWEN3B_REAL = ("1. 2. Compare the sensors used by Project Borealis and Northstar in terms of their names. "
                   "3. 4. Compare the sensors used by Project Borealis and Northstar in terms of their colours. "
                   "5. 6. Determine the maintenance intervals for the sensors used by Project Borealis. "
                   "7. 8. Determine the maintenance intervals for the sensors used by Project Northstar. "
                   "9. Which project needs maintenance more often based on the provided information?")
    TABLE = [
        ("numbered lines",
         "1. What is the sensor sampling rate?\n2. Which sensor uses that rate?", None,
         ["What is the sensor sampling rate?", "Which sensor uses that rate?"]),
        ("bulleted lines",
         "- What is the sensor sampling rate?\n- Which sensor uses that rate?", None,
         ["What is the sensor sampling rate?", "Which sensor uses that rate?"]),
        ("several questions on one unmarked line",
         "What is the sampling rate? Which sensor uses it? Why was it selected?", None,
         ["What is the sampling rate?", "Which sensor uses it?", "Why was it selected?"]),
        ("numbered, semicolon-separated on one line",
         "1. What is the sampling rate?; 2. Which sensor uses it?; 3. Who installed it?", None,
         ["What is the sampling rate?", "Which sensor uses it?", "Who installed it?"]),
        ("numbered, whitespace-separated on one line",
         "1. What is the sampling rate? 2. Which sensor uses it? 3) Who installed it?", None,
         ["What is the sampling rate?", "Which sensor uses it?", "Who installed it?"]),
        ("inline numbering after a prose lead-in",
         "Sub-questions: 1. What is the sampling rate? 2. Which sensor uses it?", None,
         ["What is the sampling rate?", "Which sensor uses it?"]),
        ("real qwen2.5:3b run-on output (empty items, imperatives)", QWEN3B_REAL, None,
         ["Compare the sensors used by Project Borealis and Northstar in terms of their names?",
          "Compare the sensors used by Project Borealis and Northstar in terms of their colours?",
          "Determine the maintenance intervals for the sensors used by Project Borealis?",
          "Determine the maintenance intervals for the sensors used by Project Northstar?",
          "Which project needs maintenance more often based on the provided information?"]),
        ("code fence and surrounding prose",
         "Sure! Here is the breakdown:\n```\n1. What is the sampling rate?\n2. Which sensor uses it?\n```\nHope this helps.",
         None, ["What is the sampling rate?", "Which sensor uses it?"]),
        ("markdown heading and bold numbers",
         "### Sub-questions\n**1.** What is the sampling rate?\n**2.** Which sensor uses it?", None,
         ["What is the sampling rate?", "Which sensor uses it?"]),
        ("duplicates on one line and across lines",
         "What is the sampling rate? What is the  SAMPLING rate?\n1. Which sensor uses it?\n2. Which sensor uses it?",
         None, ["What is the sampling rate?", "Which sensor uses it?"]),
        ("restatement of the original dropped",
         "1. What is the sampling rate of OR-17?\n2. Who calibrates OR-17?", "what is the sampling rate of or-17",
         ["Who calibrates OR-17?"]),
        ("question mark inside quotes is not a boundary",
         'Who wrote "Why now?" in the report? Which team approved it?', None,
         ['Who wrote "Why now?" in the report?', "Which team approved it?"]),
        ("lower-case continuation is not a boundary",
         "1. Is the rate 10 Hz? or 20 Hz?", None, ["Is the rate 10 Hz? or 20 Hz?"]),
        ("decimal numbers and versions are not list markers",
         "1. Did sensor v2. 5 replace the 2.5 Hz unit?\n2. When was it installed?", None,
         ["Did sensor v2. 5 replace the 2.5 Hz unit?", "When was it installed?"]),
        ("mid-sentence number is not a marker",
         "1. What did chapter 2. say about calibration?\n2. Who wrote it?", None,
         ["What did chapter 2. say about calibration?", "Who wrote it?"]),
    ]

    @pytest.mark.parametrize("name,text,original,expected", TABLE, ids=[t[0] for t in TABLE])
    def test_formats(self, lv, name, text, original, expected):
        assert lv._parse_sub_questions(text, original=original) == expected

    # Ambiguous or non-question output → [] so rlm_query uses its direct-answer fallback.
    AMBIGUOUS = [
        "What is the sampling rate?",                                   # one unmarked question
        "Sure! What is the sampling rate? Which sensor uses it?",       # prose mixed into the line
        "Dr. Smith asked what? Then who replied?",                      # abbreviation: unclear boundary
        "The rate is 10 Hz. The sensor is OR-17.",                      # statements only
        "1. 3. 2.",                                                     # markers without content
        "```\n```",                                                     # empty fence
        "Steps: 1. gather data 3. analyse it",                          # non-sequential inline numbers
        "\u0000�??? ;;; 1)",                                       # junk
    ]

    @pytest.mark.parametrize("text", AMBIGUOUS)
    def test_ambiguous_or_malformed_falls_back(self, lv, text):
        assert lv._parse_sub_questions(text) == []


# ══════════════════════════════════════════════════════════════════════════════
# 4. Conversation history
# ══════════════════════════════════════════════════════════════════════════════
class TestHistory:
    def test_gradio6_content_blocks(self, lv):
        h = [{"role": "user", "content": [{"text": "hi", "type": "text"}]},
             {"role": "assistant", "content": [{"text": "hello", "type": "text"}, {"path": "x.png"}]}]
        assert lv._safe_history(h) == "User: hi\nAssistant: hello"

    def test_placeholders_errors_and_junk_filtered(self, lv):
        h = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "⏳ Thinking…"},
             {"role": "assistant", "content": "Thinking..."}, {"role": "assistant", "content": "❌ LLM failed"},
             {"role": "assistant", "content": "   "}, "not-a-dict", {"role": "assistant", "content": None},
             {"role": "assistant", "content": "a1"}]
        assert lv._safe_history(h) == "User: q1\nAssistant: a1"

    def test_window_and_cap(self, lv):
        h = [{"role": "user", "content": f"m{i} " + "z" * 600} for i in range(20)]
        lines = lv._safe_history(h, window=3).split("\n")
        assert len(lines) == 6 and lines[0].startswith("User: m14")
        assert all(len(l) <= len("User: ") + 500 for l in lines)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Fast mode end-to-end (fake Chroma, scripted LLM)
# ══════════════════════════════════════════════════════════════════════════════
class TestFastQuery:
    def run(self, lv, monkeypatch, q, history=None, stream="fine"):
        llm = LLM(stream_reply=stream)
        monkeypatch.setattr(lv, "ask_llm_stream", llm.stream)
        monkeypatch.setattr(lv, "ask_llm", llm.ask)
        out = list(lv.fast_query(q, history=history or []))
        return llm, out

    def test_documents_reach_prompt_despite_many_memories(self, lv, monkeypatch):
        for i in range(10):
            add(lv, f"m{i}", mem_text(i + 1, "cat whiskers"), "memory")
        add(lv, "d1", "cat whiskers vet record " + "v" * 600)
        llm, out = self.run(lv, monkeypatch, "cat whiskers")
        prompt = llm.stream_prompts[0]
        assert "vet record" in prompt and "[Source: d1.md · note" in prompt
        ev = out[-1]["evidence"]
        assert any(c["id"] == "d1" for c in ev)
        for c in ev:  # evidence == chunks actually in the prompt
            assert c["text"][:50] in prompt

    def test_nonempty_db_without_relevant_results(self, lv, monkeypatch):
        add(lv, "m1", mem_text(1, "tax"), "memory")
        llm, out = self.run(lv, monkeypatch, "cat whiskers")
        assert "(No relevant notes found in your files.)" in llm.stream_prompts[0]
        assert "No documents indexed" not in out[-1]["answer"]
        assert out[-1]["evidence"] == []

    def test_empty_db_makes_no_llm_call(self, lv, monkeypatch):
        llm, out = self.run(lv, monkeypatch, "anything")
        assert "No documents indexed" in out[-1]["answer"] and llm.stream_prompts == []

    def test_history_has_no_duplicate_question_or_placeholder(self, lv, monkeypatch):
        add(lv, "d1", "cat")
        prev = [{"role": "user", "content": [{"text": "earlier question", "type": "text"}]},
                {"role": "assistant", "content": [{"text": "earlier answer", "type": "text"}]}]
        llm, _ = self.run(lv, monkeypatch, "what about my cat", history=prev)
        p = llm.stream_prompts[0]
        assert p.count("what about my cat") == 1
        assert "Thinking" not in p and "'type': 'text'" not in p
        assert "User: earlier question\nAssistant: earlier answer" in p


# ══════════════════════════════════════════════════════════════════════════════
# 6. Deep Reasoning end-to-end (fake Chroma, scripted LLM)
# ══════════════════════════════════════════════════════════════════════════════
def _deep_db(lv, n_mem=10):
    for i in range(n_mem):
        add(lv, f"m{i}", mem_text(i + 1, "cat tax"), "memory")
    add(lv, "dcat", "cat whiskers renamed shadow " + "c" * 500)
    add(lv, "dtax", "tax invoice budget filed " + "t" * 500)


class TestDeepMode:
    def run(self, lv, monkeypatch, replies, q="cat and tax?"):
        llm = LLM(replies=replies)
        monkeypatch.setattr(lv, "ask_llm", llm.ask)
        monkeypatch.setattr(lv, "ask_llm_stream", llm.stream)
        out = list(lv.rlm_query(q, history=[]))
        return llm, out

    def test_docstring_is_honest(self, lv):
        assert "NOT a Recursive Language Model" in " ".join(lv.rlm_query.__doc__.split())

    def test_sub_prompts_get_documents_and_synthesis_gets_direct_evidence(self, lv, monkeypatch):
        _deep_db(lv)
        llm, out = self.run(lv, monkeypatch,
                            ["Here you go:\n1. What is the cat called?\n2. Was the tax invoice filed?",
                             "Shadow.", "Yes.", "FINAL"])
        assert len(llm.prompts) == 4  # 1 decompose + 2 sub + 1 synthesis (preamble cost nothing)
        sub1, sub2, synth = llm.prompts[1], llm.prompts[2], llm.prompts[3]
        assert "renamed shadow" in sub1 and "· note ·" in sub1
        assert "invoice budget filed" in sub2
        assert "Direct evidence for the original question:" in synth and "· note ·" in synth.split("Direct evidence")[1]
        assert out[-1]["answer"] == "FINAL"

    def test_evidence_equals_chunks_used_with_labels(self, lv, monkeypatch):
        _deep_db(lv)
        llm, out = self.run(lv, monkeypatch,
                            ["1. What is the cat called?\n2. Was the tax invoice filed?", "a", "b", "c"])
        ev = out[-1]["evidence"]
        all_prompts = "\n".join(llm.prompts[1:])
        assert ev, "evidence must not be empty"
        for c in ev:
            assert c["text"][:40] in all_prompts
            assert set(c["used_in"]) <= {"Q1", "Q2", "synthesis"}
        assert any("Q1" in c["used_in"] for c in ev) and any("synthesis" in c["used_in"] for c in ev)
        assert len({c["id"] for c in ev}) == len(ev)  # no duplicate entries

    @pytest.mark.parametrize("decomp", ["Sure! Happy to help.", "❌ LLM failed after 3 attempts: x",
                                         "1. cat and tax?", ""])
    def test_unusable_decomposition_gives_single_direct_answer(self, lv, monkeypatch, decomp):
        _deep_db(lv)
        llm, out = self.run(lv, monkeypatch, [decomp, "DIRECT"])
        assert len(llm.prompts) == 2  # decompose + direct answer only
        assert "Retrieved context:" in llm.prompts[1] and "renamed shadow" in llm.prompts[1]
        assert out[-1]["answer"] == "DIRECT"
        assert out[-1]["evidence"] and all(c["used_in"] == ["direct"] for c in out[-1]["evidence"])
        assert lv._test_saved == [("cat and tax?", "DIRECT")]

    def test_sub_question_without_evidence_skips_llm_call(self, lv, monkeypatch):
        add(lv, "m1", mem_text(1, "garden"), "memory")  # only an unrelated memory
        llm, out = self.run(lv, monkeypatch, ["1. Was the tax invoice filed?\n2. What car was bought?", "SYN"])
        assert len(llm.prompts) == 2  # decompose + synthesis; both sub-calls skipped
        assert "Not found in notes." in llm.prompts[1]

    def test_call_count_bound(self, lv, monkeypatch):
        _deep_db(lv)
        decomp = "\n".join(f"{i}. Question number {i} about cat?" for i in range(1, 9))
        llm, _ = self.run(lv, monkeypatch, [decomp] + ["x"] * 10)
        assert len(llm.prompts) <= lv.MAX_SUB_CALLS + 2

    def test_empty_db(self, lv, monkeypatch):
        llm, out = self.run(lv, monkeypatch, [])
        assert "No documents indexed" in out[-1]["answer"] and llm.prompts == []


# ══════════════════════════════════════════════════════════════════════════════
# 7. UI history snapshot (app imported under fakes, never launched)
# ══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def app_mod(lv, monkeypatch):
    pytest.importorskip("gradio")
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.delitem(sys.modules, "app", raising=False)
    import app

    yield app
    sys.modules.pop("app", None)


class TestDoChatSnapshot:
    def test_engine_receives_previous_turns_only_and_is_isolated(self, app_mod, monkeypatch):
        seen = {}

        def fake_engine(question, history=None):
            seen["history"] = history
            seen["before"] = repr(history)
            yield {"answer": "step 1", "trajectory": [], "evidence": []}
            yield {"answer": "step 2", "trajectory": [], "evidence": []}
            seen["after"] = repr(history)

        monkeypatch.setattr(app_mod, "fast_query", fake_engine)
        prev = [{"role": "user", "content": [{"text": "old q", "type": "text"}]},
                {"role": "assistant", "content": [{"text": "old a", "type": "text"}]}]
        outs = list(app_mod.do_chat("new q", prev, "⚡ Fast Lookup"))
        assert outs[-1][0][-1]["content"] == "step 2"            # UI still updated
        assert seen["history"] == prev and seen["history"] is not prev
        assert "new q" not in seen["before"] and "Thinking" not in seen["before"]
        assert seen["before"] == seen["after"]                   # not mutated mid-run

    def test_evidence_panel_labels(self, app_mod):
        text = app_mod.format_evidence([
            {"id": "a", "text": "abc", "source": "n.md", "_type": "document", "relevance": 0.5,
             "used_in": ["Q1", "synthesis"], "truncated": True},
            {"id": "b", "text": "xyz", "source": "chat_history.md", "_type": "memory", "relevance": None}])
        assert "📄 Note · n.md · relevance 0.50 · used in: Q1, synthesis · truncated" in text
        assert "💭 Memory · chat_history.md" in text and "relevance n/a" not in text


# ══════════════════════════════════════════════════════════════════════════════
# 8. REAL ChromaDB (temp dir) — validates the scoring assumptions
# ══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def real(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    mod = _import_engine(tmp_path, monkeypatch, real_chroma=True)
    assert Path(mod.chroma._settings.persist_directory if hasattr(mod.chroma, "_settings") else
                tmp_path / ".lifevault_db").resolve().is_relative_to(tmp_path.resolve())
    monkeypatch.setattr(mod, "save_chat_memory", lambda q, a, epoch=None: None)
    yield mod
    try:
        from chromadb.api.client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
    _teardown_engine()


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


DIM = len(VOCAB) + 1


class TestRealChroma:
    def test_db_is_in_tmp_and_metric_is_squared_l2(self, real, tmp_path):
        assert (tmp_path / ".lifevault_db").exists()
        assert not (ROOT / ".lifevault_db").exists()
        cfg = real.collection.configuration_json
        assert cfg["hnsw"]["space"] == "l2"
        a = _unit([1] + [0] * (DIM - 1))
        b = _unit([0, 1] + [0] * (DIM - 2))
        add(real, "a", "alpha", emb=a)
        add(real, "b", "beta", emb=b)
        r = real.collection.query(query_embeddings=[a], n_results=2, include=["distances", "embeddings"])
        dists = r["distances"][0]
        assert isinstance(dists, list) and dists[0] == pytest.approx(0.0, abs=1e-6)
        assert dists[1] == pytest.approx(2.0, abs=1e-5)  # squared L2 between orthonormal vectors
        assert len(r["embeddings"][0][0]) == DIM

    def test_relevance_identical_similar_unrelated_and_nonunit(self, real):
        q = real.embedder.encode("cat whiskers").tolist()
        add(real, "same", "cat whiskers")                      # identical embedding
        add(real, "sim", "cat whiskers shadow garden")         # overlapping
        add(real, "unrel", "tax invoice budget")               # disjoint keywords
        add(real, "big", "scaled", emb=[x * 5 for x in q])     # same direction, norm 5
        rel = {c["id"]: c["relevance"] for c in real.retrieve("cat whiskers", notes_k=10)}
        assert rel["same"] == pytest.approx(1.0, abs=1e-6)
        assert rel["big"] == pytest.approx(1.0, abs=1e-6)     # 1 - d/2 would be ≈ -47
        assert 0.3 < rel["sim"] < 0.95
        assert abs(rel["unrel"]) < 0.01
        for c in real.retrieve("cat whiskers", notes_k=10):    # cosine agrees with 1-d/2 for unit vectors
            if c["id"] != "big":
                assert c["relevance"] == pytest.approx(1 - c["distance"] / 2, abs=1e-5)

    def test_filtered_query_when_k_exceeds_matches(self, real):
        add(real, "d1", "cat")
        add(real, "m1", mem_text(1, "cat"), "memory")
        out = real.retrieve("cat", memory_k=10, notes_k=8)
        assert sorted(c["id"] for c in out) == ["d1", "m1"]

    def test_no_memories_documents_still_returned(self, real):
        add(real, "d1", "cat")
        add(real, "d2", "dog")
        out = real.retrieve("cat")
        assert {c["_type"] for c in out} == {"document"} and len(out) == 2

    def test_memory_threshold_against_real_scores(self, real):
        add(real, "d1", "garden")
        add(real, "m_rel", mem_text(1, "cat whiskers"), "memory")
        add(real, "m_unrel", mem_text(2, "tax invoice"), "memory")
        ids = [c["id"] for c in real.retrieve("cat whiskers")]
        assert "m_rel" in ids and "m_unrel" not in ids and "d1" in ids

    def test_duplicate_text_collapsed(self, real):
        add(real, "x1", "cat whiskers shadow", source="a/n.md")
        add(real, "x2", "cat whiskers shadow", source="b/n.md")
        assert len(real.retrieve("cat")) == 1


# ══════════════════════════════════════════════════════════════════════════════
# [RET-08] Deep Reasoning evidence: per-step uses must mirror the prompts
# ══════════════════════════════════════════════════════════════════════════════
_PROMPT_ENTRY = re.compile(r"\[Source: (.+?) · (note|memory) · relevance (-?[\d.]+|n/a)\]\n(.{0,40})")


def _stage_entries(prompt: str, section: str) -> list[tuple[str, str, str, str]]:
    return _PROMPT_ENTRY.findall(prompt[prompt.index(section):])


class TestDeepEvidence:
    DECOMP = "1. What colour are the cat whiskers?\n2. Is the cat garden in shadow?"

    def run(self, lv, monkeypatch):
        add(lv, "c1", "cat whiskers shadow are long and grey " + "w" * 300, source="notes/cat.md")
        add(lv, "c2", "cat garden purple flowers grow here " + "g" * 300, source="notes/cat.md")
        add(lv, "t1", "tax invoice budget was filed in March " + "t" * 300, source="notes/tax.md")
        llm = LLM(replies=[self.DECOMP, "grey", "yes", "FINAL"])
        monkeypatch.setattr(lv, "ask_llm", llm.ask)
        out = list(lv.rlm_query("cat and tax?", history=[]))
        assert len(llm.prompts) == 4 and out[-1]["answer"] == "FINAL"
        stages = {"Q1": _stage_entries(llm.prompts[1], "Notes:"),
                  "Q2": _stage_entries(llm.prompts[2], "Notes:"),
                  "synthesis": _stage_entries(llm.prompts[3], "Direct evidence")}
        return out[-1]["evidence"], stages

    def test_every_use_matches_the_prompt_of_its_step(self, lv, monkeypatch):
        ev, stages = self.run(lv, monkeypatch)
        sent = {(step, src, rel, head[:30]) for step, es in stages.items() for src, _, rel, head in es}
        shown = set()
        for e in ev:
            assert [u["step"] for u in e["uses"]] == e["used_in"]          # one use per step label
            for u in e["uses"]:
                shown.add((u["step"], e["source"], f"{u['relevance']:.2f}", e["text"][:30]))
        assert shown == sent

    def test_same_excerpt_in_several_steps_keeps_each_relevance(self, lv, monkeypatch):
        ev, _ = self.run(lv, monkeypatch)
        multi = [e for e in ev if len(e["uses"]) >= 2]
        assert multi, "scenario must reuse an excerpt"
        assert any(len({round(u["relevance"], 2) for u in e["uses"]}) >= 2 for e in multi)
        assert len({e["id"] for e in ev}) == len(ev)                       # still one entry per excerpt

    def test_different_excerpts_of_one_document_stay_separate(self, lv, monkeypatch):
        ev, _ = self.run(lv, monkeypatch)
        cat = [e for e in ev if e["source"] == "notes/cat.md"]
        assert {e["id"] for e in cat} == {"c1", "c2"}

    def test_synthesis_reuse_is_recorded(self, lv, monkeypatch):
        ev, stages = self.run(lv, monkeypatch)
        assert stages["synthesis"]
        reused = [e for e in ev if "synthesis" in e["used_in"] and any(s.startswith("Q") for s in e["used_in"])]
        assert reused

    def test_panel_shows_every_step_with_its_relevance(self, lv, monkeypatch, app_mod):
        ev, stages = self.run(lv, monkeypatch)
        panel = app_mod.format_evidence(ev)
        for e in ev:
            rels = {f"{u['relevance']:.2f}" for u in e["uses"]}
            if len(rels) > 1:
                for u in e["uses"]:
                    assert f"{u['step']} ({u['relevance']:.2f}" in panel
        for step, es in stages.items():
            for src, _, rel, _ in es:
                assert re.search(rf"· {re.escape(src)} ·[^\n]*{step} \({rel}|· {re.escape(src)} · relevance {rel} · used in: [^\n]*{step}", panel)


class TestEvidencePanelFormat:
    def test_single_step_and_fast_mode_formats_unchanged(self, app_mod):
        one = {"source": "a.md", "_type": "document", "relevance": 0.5, "text": "alpha " * 10,
               "used_in": ["Q1"], "uses": [{"step": "Q1", "relevance": 0.5, "truncated": False}]}
        fast = {"source": "b.md", "_type": "document", "relevance": 0.25, "text": "beta " * 10}
        out = app_mod.format_evidence([one, fast])
        assert "[1] 📄 Note · a.md · relevance 0.50 · used in: Q1\n" in out
        assert "[2] 📄 Note · b.md · relevance 0.25\n" in out

    def test_steps_with_different_relevance_and_truncation(self, app_mod):
        e = {"source": "a.md", "_type": "document", "relevance": 0.61, "text": "alpha " * 10,
             "used_in": ["Q1", "Q2", "synthesis"],
             "uses": [{"step": "Q1", "relevance": 0.61, "truncated": False},
                      {"step": "Q2", "relevance": 0.58, "truncated": True},
                      {"step": "synthesis", "relevance": 0.61, "truncated": False}]}
        line = app_mod.format_evidence([e]).split("\n")[0]
        assert line == "[1] 📄 Note · a.md · relevance 0.58–0.61 · used in: Q1 (0.61), Q2 (0.58, truncated), synthesis (0.61)"

    @pytest.mark.parametrize("chunk", [
        {"text": "no source"},
        {"source": "x.md"},
        {"source": "x.md", "text": "t", "relevance": float("nan"), "used_in": ["Q1"]},
        {"source": "x.md", "text": "t", "used_in": ["Q1", "Q2"],
         "uses": [None, {"step": "Q2"}, {"relevance": "high"}, "junk"]},
        {"source": "x.md", "text": "t", "uses": "not-a-list", "used_in": None},
    ])
    def test_missing_or_malformed_metadata_never_breaks_the_panel(self, app_mod, chunk):
        out = app_mod.format_evidence([chunk])
        assert out.startswith("[1] ") and "nan" not in out and "{" not in out and "None" not in out
