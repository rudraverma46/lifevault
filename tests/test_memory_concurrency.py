"""
Clear-Memory concurrency tests [MEM-01].

Every test imports the engine in pytest's tmp_path (CWD switched + guarded)
with a REAL ChromaDB 1.5.8 in that temp folder, a fake sentence_transformers
(keyword vectors) and a fake ollama. Races are reproduced deterministically
with threading.Event gates and an instrumented lock — no timing guesses: a
gate holds the save (or ingest) inside its critical section until the test
has observed Clear Memory waiting for it.

Run:  python -m pytest tests/test_memory_concurrency.py -v
"""

import logging
import math
import os
import sys
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GATE_TIMEOUT = 10        # upper bound only; never reached unless the code under test is broken
VOCAB = ["cat", "shadow", "tax", "invoice", "memory", "answer", "gate", "project"]


class _Vec(list):
    def tolist(self):
        return list(self)


def _embed(text):
    words = [w.strip(".,!?:;()[]\"'*#").lower() for w in str(text).split()]
    v = [float(sum(w == k for w in words)) for k in VOCAB] + [1e-3]
    n = math.sqrt(sum(x * x for x in v))
    return _Vec(x / n for x in v)


class _Embedder:
    """Fake embedder; `hold` blocks encode() of texts containing a marker."""

    def __init__(self):
        self.hold = None          # (marker, entered: Event, release: Event)

    def encode(self, text):
        if self.hold and self.hold[0] in str(text):
            marker, entered, release = self.hold
            entered.set()
            assert release.wait(GATE_TIMEOUT), "test gate never released"
        return _embed(text)


@pytest.fixture
def lv(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LIFEVAULT_ALLOWED_ROOTS", raising=False)
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    emb = _Embedder()
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = lambda *a, **k: emb
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    o = types.ModuleType("ollama")
    o.list = lambda: {"models": []}
    monkeypatch.setitem(sys.modules, "ollama", o)
    monkeypatch.delitem(sys.modules, "lifevault", raising=False)
    assert Path.cwd().resolve() == tmp_path.resolve() != ROOT        # guard
    import lifevault

    assert (tmp_path / ".lifevault_db").exists() and not (ROOT / ".lifevault_db").exists()
    (tmp_path / "my_vault").mkdir(exist_ok=True)
    lifevault._test_embedder = emb
    # Record the engine's background save threads so tests can join them.
    started = []

    class _RecThread(threading.Thread):
        def start(self):
            started.append(self)
            super().start()

    monkeypatch.setattr(lifevault, "threading", types.SimpleNamespace(
        Thread=_RecThread, Lock=threading.Lock, Event=threading.Event))
    lifevault._test_threads = started
    lifevault.collection.add(ids=["doc-1"], embeddings=[_embed("cat shadow")],
                             documents=["The cat is called Shadow. " * 4],
                             metadatas=[{"source": "notes/cat.md", "type": "document"}])
    yield lifevault
    try:
        from chromadb.api.client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
    lg = logging.getLogger("LifeVault")
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()
    sys.modules.pop("lifevault", None)


# ── helpers ───────────────────────────────────────────────────────────────────
def mem_ids(lv):
    return lv.collection.get(where={"type": "memory"})["ids"]


def doc_ids(lv):
    return lv.collection.get(where={"type": "document"})["ids"]


def mem_file(tmp_path):
    return tmp_path / "my_vault" / "chat_history.md"


def join_saves(lv):
    for t in list(lv._test_threads):
        t.join(GATE_TIMEOUT)
        assert not t.is_alive()


def save_and_wait(lv, q, a, **kw):
    lv.save_chat_memory(q, a, **kw)
    join_saves(lv)


class _WatchedLock:
    """Wraps the engine's memory lock; signals when the clear thread waits on it."""

    def __init__(self, inner, waiter_name):
        self.inner, self.waiter_name = inner, waiter_name
        self.waiting = threading.Event()

    def acquire(self, blocking=True, timeout=-1):
        if threading.current_thread().name == self.waiter_name:
            self.waiting.set()
        return self.inner.acquire(blocking, timeout)

    def release(self):
        self.inner.release()

    def locked(self):
        return self.inner.locked()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


def start_clear(lv):
    """Run clear_memory() in a named thread; returns (thread, result list, done Event)."""
    out, done = [], threading.Event()

    def run():
        out.append(lv.clear_memory())
        done.set()

    t = threading.Thread(target=run, name="clear-memory-test", daemon=True)
    t.start()
    return t, out, done


def wait_any(*events, timeout=GATE_TIMEOUT):
    gate = threading.Event()
    for _ in range(int(timeout * 100)):
        if any(e.is_set() for e in events):
            return True
        gate.wait(0.01)
    return False


# ── tests ─────────────────────────────────────────────────────────────────────
class TestClearMemory:
    def test_clear_with_nothing_running(self, lv, tmp_path):
        save_and_wait(lv, "What is the cat called?", "The cat is called Shadow.")
        assert mem_ids(lv) and mem_file(tmp_path).read_text(encoding="utf-8").strip()
        msg = lv.clear_memory()
        assert msg.startswith("✅")
        assert mem_ids(lv) == [] and mem_file(tmp_path).read_text(encoding="utf-8") == ""
        assert doc_ids(lv) == ["doc-1"]

    def test_clear_waits_for_a_save_in_progress_and_removes_it(self, lv, tmp_path, monkeypatch):
        entered, release = threading.Event(), threading.Event()
        lv._test_embedder.hold = ("GATED", entered, release)
        watched = _WatchedLock(lv._memory_lock, "clear-memory-test")
        monkeypatch.setattr(lv, "_memory_lock", watched)
        lv.save_chat_memory("GATED question?", "GATED answer.")
        assert entered.wait(GATE_TIMEOUT)                 # the save is mid-way (file written, vector pending)
        clear_t, out, done = start_clear(lv)
        assert wait_any(watched.waiting, done)
        assert not done.is_set(), "clear finished while a save was still in progress"
        release.set()
        join_saves(lv)
        clear_t.join(GATE_TIMEOUT)
        assert out and out[0].startswith("✅")
        assert mem_ids(lv) == []                          # the earlier save did not survive the clear
        assert mem_file(tmp_path).read_text(encoding="utf-8") == ""
        assert doc_ids(lv) == ["doc-1"]

    def test_answer_from_generation_started_before_clear_is_not_saved(self, lv, tmp_path):
        epoch = lv.memory_epoch()                          # question asked here …
        assert lv.clear_memory().startswith(("✅", "ℹ️"))   # … user clears while it is generating …
        lv.save_chat_memory("Old question?", "Answer finished after the clear.", epoch=epoch)
        join_saves(lv)
        assert mem_ids(lv) == []                           # … and the late answer is not resurrected
        assert not mem_file(tmp_path).exists() or mem_file(tmp_path).read_text(encoding="utf-8") == ""

    def test_fast_query_in_flight_during_clear_does_not_repopulate(self, lv, tmp_path, monkeypatch):
        streaming, finish = threading.Event(), threading.Event()

        def stream(prompt, timeout=None):
            yield "Shadow "
            streaming.set()
            assert finish.wait(GATE_TIMEOUT)
            yield "is the cat."

        monkeypatch.setattr(lv, "ask_llm_stream", stream)
        result = []
        gen_t = threading.Thread(target=lambda: result.append(list(lv.fast_query("cat?", history=[]))[-1]))
        gen_t.start()
        assert streaming.wait(GATE_TIMEOUT)                # generation in progress, save not started
        assert lv.clear_memory().startswith(("✅", "ℹ️"))
        finish.set()
        gen_t.join(GATE_TIMEOUT)
        join_saves(lv)
        assert result and result[0]["answer"] == "Shadow is the cat."   # the user still sees the answer
        assert lv._test_threads, "fast_query must still request the save"
        assert mem_ids(lv) == []
        assert not mem_file(tmp_path).exists() or mem_file(tmp_path).read_text(encoding="utf-8") == ""

    def test_new_answer_after_clear_is_saved(self, lv, tmp_path):
        save_and_wait(lv, "First question?", "First answer.")
        assert lv.clear_memory().startswith("✅")
        epoch = lv.memory_epoch()
        save_and_wait(lv, "Second question?", "Second answer.", epoch=epoch)
        assert len(mem_ids(lv)) == 1
        text = mem_file(tmp_path).read_text(encoding="utf-8")
        assert "Second answer." in text and "First answer." not in text

    def test_failed_save_releases_lock_and_clear_still_empties_the_file(self, lv, tmp_path, monkeypatch):
        real_add = lv.collection.add

        class Boom:
            def __getattr__(self, name):
                return getattr(real_col, name)

            def add(self, **kw):
                raise RuntimeError("disk full")

        real_col = lv.collection
        monkeypatch.setattr(lv, "collection", Boom())
        save_and_wait(lv, "Q?", "An answer whose vector could not be stored.")
        monkeypatch.setattr(lv, "collection", real_col)
        assert not lv._memory_lock.locked()
        assert mem_ids(lv) == [] and "could not be stored" in mem_file(tmp_path).read_text(encoding="utf-8")
        msg = lv.clear_memory()                            # file-only memory must be cleared too,
        assert msg.startswith("✅")                         # or the next ingest would restore it
        assert mem_file(tmp_path).read_text(encoding="utf-8") == ""
        assert real_add is not None and doc_ids(lv) == ["doc-1"]

    def test_failed_clear_reports_without_leaking_details(self, lv, tmp_path, monkeypatch):
        save_and_wait(lv, "Q?", "Remember this.")
        real_col = lv.collection

        class Boom:
            def __getattr__(self, name):
                return getattr(real_col, name)

            def delete(self, **kw):
                raise PermissionError(13, "Permission denied", str(tmp_path / ".lifevault_db"))

        monkeypatch.setattr(lv, "collection", Boom())
        msg = lv.clear_memory()
        assert msg.startswith("❌")
        assert str(tmp_path) not in msg and "Permission denied" not in msg and "Errno" not in msg
        assert not lv._memory_lock.locked()                # no lock leak after a failure
        monkeypatch.setattr(lv, "collection", real_col)
        assert doc_ids(lv) == ["doc-1"]
        assert lv.clear_memory().startswith("✅")           # a retry succeeds
        assert mem_ids(lv) == []

    def test_clear_gives_up_when_memory_stays_busy(self, lv, tmp_path, monkeypatch):
        save_and_wait(lv, "Q?", "Keep me.")
        monkeypatch.setattr(lv, "MEMORY_CLEAR_TIMEOUT", 0.05)
        epoch = lv.memory_epoch()
        assert lv._memory_lock.acquire(timeout=GATE_TIMEOUT)    # simulate a stuck writer
        try:
            msg = lv.clear_memory()
        finally:
            lv._memory_lock.release()
        assert msg.startswith("❌") and "nothing was cleared" in msg.lower()
        assert len(mem_ids(lv)) == 1 and lv.memory_epoch() == epoch

    def test_ingest_indexing_memory_cannot_restore_cleared_memories(self, lv, tmp_path, monkeypatch):
        save_and_wait(lv, "Q?", "Remembered GATED fact.")
        assert lv.clear_memory().startswith("✅")
        # Re-create the file as an older copy would look, then let ingestion
        # index it while a clear is requested mid-way.
        save_and_wait(lv, "Q2?", "Another GATED fact.")
        lv.collection.delete(ids=mem_ids(lv))                    # file has the entry, index does not
        entered, release = threading.Event(), threading.Event()
        lv._test_embedder.hold = ("GATED", entered, release)
        watched = _WatchedLock(lv._memory_lock, "clear-memory-test")
        monkeypatch.setattr(lv, "_memory_lock", watched)
        ing = []
        ing_t = threading.Thread(target=lambda: ing.append(list(lv.ingest_vault("./my_vault"))))
        ing_t.start()
        assert entered.wait(GATE_TIMEOUT)                         # ingest is indexing the memory file
        clear_t, out, done = start_clear(lv)
        assert wait_any(watched.waiting, done)
        assert not done.is_set(), "clear finished while ingestion was indexing memories"
        release.set()
        ing_t.join(GATE_TIMEOUT)
        clear_t.join(GATE_TIMEOUT)
        assert out and out[0].startswith("✅")
        assert mem_ids(lv) == [] and mem_file(tmp_path).read_text(encoding="utf-8") == ""
        assert doc_ids(lv) == ["doc-1"]
