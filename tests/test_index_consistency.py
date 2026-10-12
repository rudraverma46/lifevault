"""
Index-consistency regression tests (Phase 3 Task E, Option A — lazy per-file).

Every test runs ingest_vault() against a REAL ChromaDB 1.5.8 created in
pytest's tmp_path (CWD switched + guarded). sentence_transformers and ollama
are fakes; all files are synthetic. Fault injection wraps the real collection
in a proxy (add / get / delete failures). Nothing touches the real vault,
chat_history.md or production database.

Run:  python -m pytest tests/test_index_consistency.py -v
"""

import hashlib
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

VOCAB = ["cat", "dog", "tax", "car", "invoice", "budget", "garden", "recipe",
         "shadow", "whiskers", "alpha", "beta", "gamma", "delta", "project"]


class _Vec(list):
    def tolist(self):
        return list(self)


def keyword_embed(text):
    words = [w.strip(".,!?:;()[]\"'*#").lower() for w in str(text).split()]
    v = [float(sum(w == k for w in words)) for k in VOCAB] + [1e-3]
    n = math.sqrt(sum(x * x for x in v))
    return _Vec(x / n for x in v)


def _import(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LIFEVAULT_ALLOWED_ROOTS", raising=False)
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = type("ST", (), {"__init__": lambda s, *a, **k: None,
                                             "encode": lambda s, t: keyword_embed(t)})
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    o = types.ModuleType("ollama")
    o.list = lambda: {"models": []}
    monkeypatch.setitem(sys.modules, "ollama", o)
    monkeypatch.delitem(sys.modules, "lifevault", raising=False)
    assert Path.cwd().resolve() == tmp_path.resolve() != ROOT        # guard
    import lifevault

    assert (tmp_path / ".lifevault_db").exists() and not (ROOT / ".lifevault_db").exists()
    return lifevault


def _release():
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


@pytest.fixture
def lv(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    mod = _import(tmp_path, monkeypatch)
    (tmp_path / "my_vault").mkdir(exist_ok=True)
    yield mod
    _release()


# ── helpers ───────────────────────────────────────────────────────────────────
def write(tmp_path, rel, text):
    p = tmp_path / "my_vault" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def ingest(lv, path=None):
    return [m for m, _ in (lv.ingest_vault(path) if path else lv.ingest_vault())]


def records(lv, kind="document"):
    r = lv.collection.get(where={"type": kind}, include=["documents", "metadatas"])
    return list(zip(r["ids"], r["documents"], r["metadatas"]))


def texts(lv):
    return " ".join(d for _, d, _ in records(lv))


def by_source(lv, src):
    return [(i, d, m) for i, d, m in records(lv) if m["source"] == src]


def add_legacy(lv, name, text):
    """Seed records exactly as pre-schema-2 LifeVault wrote them."""
    chunks = lv.chunk_text(text, name)
    lv.collection.add(ids=[c["id"] for c in chunks],
                      embeddings=[keyword_embed(c["text"]) for c in chunks],
                      documents=[c["text"] for c in chunks],
                      metadatas=[{"source": name, "type": "document"} for _ in chunks])
    return [c["id"] for c in chunks]


class Proxy:
    """Wrap the real collection; override one method to inject a fault."""

    def __init__(self, real, **overrides):
        self._real, self._o = real, overrides

    def __getattr__(self, name):
        return self._o.get(name) or getattr(self._real, name)


A = "alpha cat notes. " * 10            # 1 chunk
B = "beta tax records. " * 10
C = "delta garden recipe text. " * 8
BIG = "gamma budget line. " * 120       # several chunks


# ══════════════════════════════════════════════════════════════════════════════
# 1, 2, 15 — first ingestion, unchanged re-ingestion, no duplicates
# ══════════════════════════════════════════════════════════════════════════════
class TestBasics:
    def test_first_ingestion_schema(self, lv, tmp_path):
        write(tmp_path, "notes/project.md", A)
        log = ingest(lv)
        assert log[-1].startswith("✅  Done")
        (cid, doc, meta), = records(lv)
        h = hashlib.sha256(A.encode()).hexdigest()
        assert meta == {"source": "notes/project.md", "type": "document",
                        "doc_id": "vault:notes/project.md", "doc_hash": h,
                        "chunk_index": 0, "chunk_count": 1, "schema": 2}
        assert cid == hashlib.md5(f"v2::vault:notes/project.md::{h}::0".encode()).hexdigest()

    def test_reingest_unchanged_is_idempotent_and_stable(self, lv, tmp_path):
        write(tmp_path, "a.md", A)
        write(tmp_path, "big.md", BIG)
        ingest(lv)
        ids1 = sorted(i for i, _, _ in records(lv))
        for _ in range(3):
            log = ingest(lv)
        assert sorted(i for i, _, _ in records(lv)) == ids1          # stable chunk ids
        assert "0 new chunks" in log[-1] and "2 unchanged" in log[-1]
        assert len(texts(lv)) and len(ids1) == len(set(ids1))

    def test_no_duplicate_text_after_many_runs_and_edits(self, lv, tmp_path):
        for k in range(4):
            write(tmp_path, "a.md", A + f" dog {k}" * 5)
            ingest(lv)
        docs = [d for _, d, _ in records(lv)]
        assert len(docs) == len(set(docs)) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 3, 4, 5, 6, 14 — update, delete, rename, same filename, isolation
# ══════════════════════════════════════════════════════════════════════════════
class TestLifecycle:
    def test_updated_document_replaces_old_text(self, lv, tmp_path):
        write(tmp_path, "a.md", A)
        ingest(lv)
        write(tmp_path, "a.md", C)
        log = ingest(lv)
        assert "alpha cat" not in texts(lv) and "delta garden" in texts(lv)
        assert "1 updated" in log[-1]

    def test_shrunk_document_leaves_no_trailing_chunks(self, lv, tmp_path):
        write(tmp_path, "big.md", BIG)
        ingest(lv)
        assert len(by_source(lv, "big.md")) > 1
        write(tmp_path, "big.md", "gamma short. " * 8)
        ingest(lv)
        assert len(by_source(lv, "big.md")) == 1

    def test_deleted_document_not_searchable(self, lv, tmp_path):
        p = write(tmp_path, "a.md", A)
        write(tmp_path, "b.md", B)
        ingest(lv)
        p.unlink()
        log = ingest(lv)
        assert "alpha cat" not in texts(lv) and "beta tax" in texts(lv)
        assert "deleted or renamed" in log[-1]

    def test_renamed_document_leaves_no_old_reference(self, lv, tmp_path):
        p = write(tmp_path, "a.md", A)
        ingest(lv)
        p.rename(tmp_path / "my_vault" / "renamed.md")
        ingest(lv)
        assert {m["source"] for _, _, m in records(lv)} == {"renamed.md"}

    def test_same_filename_in_different_directories(self, lv, tmp_path):
        write(tmp_path, "notes/project.md", A)
        write(tmp_path, "archive/project.md", B)
        ingest(lv)
        assert {m["doc_id"] for _, _, m in records(lv)} == {
            "vault:notes/project.md", "vault:archive/project.md"}
        assert "alpha cat" in texts(lv) and "beta tax" in texts(lv)

    def test_update_and_delete_of_one_never_touch_the_other(self, lv, tmp_path):
        a = write(tmp_path, "notes/project.md", A)
        write(tmp_path, "archive/project.md", B)
        ingest(lv)
        other = sorted(by_source(lv, "archive/project.md"))
        write(tmp_path, "notes/project.md", C)
        ingest(lv)
        assert sorted(by_source(lv, "archive/project.md")) == other
        a.unlink()
        ingest(lv)
        assert sorted(by_source(lv, "archive/project.md")) == other
        assert by_source(lv, "notes/project.md") == []

    def test_subfolder_ingest_uses_same_identity_and_never_deletes(self, lv, tmp_path):
        write(tmp_path, "notes/project.md", A)
        gone = write(tmp_path, "other/x.md", B)
        ingest(lv)
        ids = sorted(i for i, _, _ in records(lv))
        gone.unlink()
        log = ingest(lv, str(tmp_path / "my_vault" / "notes"))
        assert sorted(i for i, _, _ in records(lv)) == ids        # no dup, no deletion
        assert "runs only when the whole vault is ingested" in log[-1]


# ══════════════════════════════════════════════════════════════════════════════
# 7, 8, 9, 12 — failure paths
# ══════════════════════════════════════════════════════════════════════════════
def _fails(msg):
    def f(*a, **k):
        raise RuntimeError(msg)
    return f


class TestFailures:
    def test_insert_failure_keeps_previous_chunks_and_reports(self, lv, tmp_path, monkeypatch):
        write(tmp_path, "a.md", A)
        ingest(lv)
        before = records(lv)
        write(tmp_path, "a.md", C)
        monkeypatch.setattr(lv, "collection", Proxy(lv.collection, add=_fails("disk full")))
        log = ingest(lv)
        lv.collection = lv.collection._real                       # restore for the rerun
        assert log[-1].startswith("⚠️  Finished with problems") and "✅" not in log[-1]
        assert "could not store new chunks" in log[-1]
        assert records(lv) == before                               # old text still searchable
        ingest(lv)                                                 # heals
        assert "delta garden" in texts(lv) and "alpha cat" not in texts(lv)

    def test_verification_failure_keeps_previous_and_removes_unverified(self, lv, tmp_path, monkeypatch):
        write(tmp_path, "a.md", A)
        ingest(lv)
        before = records(lv)
        write(tmp_path, "a.md", C)
        real = lv.collection

        def bad_get(**kw):
            r = real.get(**kw)
            if kw.get("ids") and "documents" in (kw.get("include") or []):
                r = dict(r, documents=["CORRUPTED" for _ in r["ids"]])
            return r

        monkeypatch.setattr(lv, "collection", Proxy(real, get=bad_get))
        log = ingest(lv)
        lv.collection = real
        assert "could not be verified" in log[-1] and not log[-1].startswith("✅")
        assert records(lv) == before                               # unverified chunks removed

    def test_cleanup_failure_is_reported_and_retried(self, lv, tmp_path, monkeypatch):
        write(tmp_path, "a.md", A)
        ingest(lv)
        write(tmp_path, "a.md", C)
        real = lv.collection
        monkeypatch.setattr(lv, "collection", Proxy(real, delete=_fails("locked")))
        log = ingest(lv)
        lv.collection = real
        assert not log[-1].startswith("✅") and "obsolete chunks" in log[-1]
        assert "delta garden" in texts(lv)                          # new content is in
        log = ingest(lv)
        assert log[-1].startswith("✅") and "alpha cat" not in texts(lv)

    def test_partial_failure_other_files_still_indexed(self, lv, tmp_path, monkeypatch):
        write(tmp_path, "a.md", A)
        write(tmp_path, "b.md", B)
        ingest(lv)
        before_a = by_source(lv, "a.md")
        write(tmp_path, "a.md", C)
        write(tmp_path, "new.md", "dog car notes. " * 10)
        real_read = lv.read_file

        def read(path, strict=False):
            if path.name == "a.md":
                raise OSError("I/O error")
            return real_read(path, strict=strict)

        monkeypatch.setattr(lv, "read_file", read)
        log = ingest(lv)
        assert "Finished with problems in 1" in log[-1] and "unreadable" in log[-1]
        assert by_source(lv, "a.md") == before_a                  # failed doc kept, not "deleted"
        assert by_source(lv, "new.md")                            # others proceed

    @pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores perms")
    def test_unreadable_file_and_dir_during_full_scan_are_not_deleted(self, lv, tmp_path):
        f = write(tmp_path, "a.md", A)
        write(tmp_path, "locked/b.md", B)
        ingest(lv)
        n = len(records(lv))
        f.chmod(0)
        (tmp_path / "my_vault" / "locked").chmod(0)
        try:
            log = ingest(lv)
        finally:
            f.chmod(0o600)
            (tmp_path / "my_vault" / "locked").chmod(0o700)
        assert len(records(lv)) == n and "alpha cat" in texts(lv) and "beta tax" in texts(lv)
        assert not log[-1].startswith("✅")

    # (Rev 6) An intentionally emptied file is now a valid update — see TestEmptiedDocuments.

    def test_empty_scan_does_not_trigger_cleanup(self, lv, tmp_path):
        p = write(tmp_path, "a.md", A)
        ingest(lv)
        p.unlink()
        write(tmp_path, "chat_history.md", "")
        log = ingest(lv)
        assert "Cleanup skipped" in log[-1] and "alpha cat" in texts(lv)

    def test_concurrent_ingestion_is_refused(self, lv, tmp_path):
        write(tmp_path, "a.md", A)
        assert lv._ingest_lock.acquire(blocking=False)
        try:
            log = ingest(lv)
        finally:
            lv._ingest_lock.release()
        assert "already running" in log[-1] and records(lv) == []


# ══════════════════════════════════════════════════════════════════════════════
# 10, 11 — chat memory and full-vault cleanup scope
# ══════════════════════════════════════════════════════════════════════════════
ENTRY = ("\n\n### [2026-01-0{d} 10:00 AM] User: cat question {d}?\n\n"
         "### [2026-01-0{d} 10:00 AM] LifeVault: The cat is called Shadow, answer {d}.\n\n---\n")


class TestMemory:
    def test_chat_history_reingest_creates_no_duplicates(self, lv, tmp_path):
        entries = [ENTRY.format(d=d) for d in range(1, 7)]
        write(tmp_path, "chat_history.md", "".join(entries))
        for e in entries[:3]:                                   # as save_chat_memory indexed them
            for c in lv.chunk_text(e, "chat_history.md"):
                lv.collection.add(ids=[c["id"]], embeddings=[keyword_embed(c["text"])],
                                  documents=[c["text"]],
                                  metadatas=[{"source": "chat_history.md", "type": "memory"}])
        ingest(lv)
        n = len(records(lv, "memory"))
        ingest(lv)
        ingest(lv)
        mem = [d for _, d, _ in records(lv, "memory")]
        assert len(mem) == n == 6 and len(set(mem)) == 6

    def test_memory_with_hr_in_answer_still_idempotent(self, lv, tmp_path):
        e = ENTRY.format(d=1).replace("answer 1.", "answer 1.\n\n---\n\nmore text")
        write(tmp_path, "chat_history.md", e + ENTRY.format(d=2))
        ingest(lv)
        n = len(records(lv, "memory"))
        ingest(lv)
        assert len(records(lv, "memory")) == n

    def test_cleanup_never_deletes_memories(self, lv, tmp_path):
        write(tmp_path, "chat_history.md", ENTRY.format(d=1))
        p = write(tmp_path, "a.md", A)
        ingest(lv)
        mem = records(lv, "memory")
        p.unlink()
        (tmp_path / "my_vault" / "chat_history.md").unlink()
        write(tmp_path, "b.md", B)
        ingest(lv)
        assert records(lv, "memory") == mem and mem
        assert by_source(lv, "a.md") == []

    def test_documents_and_memories_stay_distinguishable(self, lv, tmp_path):
        write(tmp_path, "chat_history.md", ENTRY.format(d=1))
        write(tmp_path, "a.md", A)
        ingest(lv)
        assert all("schema" not in m for _, _, m in records(lv, "memory"))
        assert all(m["schema"] == 2 for _, _, m in records(lv, "document"))


# ══════════════════════════════════════════════════════════════════════════════
# 13 — legacy (pre-schema-2) records
# ══════════════════════════════════════════════════════════════════════════════
class TestLegacy:
    def test_identical_legacy_is_kept_and_reported_even_in_most_favourable_case(self, lv, tmp_path):
        # (Rev 6) Unique name + identical text + full vault is NOT proof of ownership:
        # the legacy record may have come from a same-named file outside the vault.
        write(tmp_path, "a.md", A)
        legacy = add_legacy(lv, "a.md", A)
        log = ingest(lv)
        assert lv.collection.get(ids=legacy)["ids"] == legacy
        assert sorted(m.get("schema", 1) for _, _, m in records(lv)) == [1, 2]
        assert f"{len(legacy)} chunks indexed by an older LifeVault version were kept" in log[-1]
        assert f"{len(legacy)} of them repeat text now indexed under its current path" in log[-1]

    def test_identical_legacy_kept_on_subfolder_ingest(self, lv, tmp_path):
        write(tmp_path, "notes/a.md", A)
        legacy = add_legacy(lv, "a.md", A)
        ingest(lv, str(tmp_path / "my_vault" / "notes"))
        assert lv.collection.get(ids=legacy)["ids"] == legacy

    def test_ambiguous_same_name_legacy_kept(self, lv, tmp_path):
        write(tmp_path, "notes/project.md", A)
        write(tmp_path, "archive/project.md", B)
        legacy = add_legacy(lv, "project.md", A)                     # could belong to either
        log = ingest(lv)
        assert lv.collection.get(ids=legacy)["ids"] == legacy
        assert "older LifeVault version were kept" in log[-1]

    def test_stale_legacy_of_edited_file_kept_and_reported(self, lv, tmp_path):
        write(tmp_path, "a.md", C)                                   # file has changed since
        legacy = add_legacy(lv, "a.md", A)
        log = ingest(lv)
        assert lv.collection.get(ids=legacy)["ids"] == legacy
        assert f"{len(legacy)} chunks indexed by an older LifeVault version were kept" in log[-1]

    def test_legacy_of_deleted_file_kept(self, lv, tmp_path):
        write(tmp_path, "b.md", B)
        legacy = add_legacy(lv, "gone.md", A)
        ingest(lv)
        assert lv.collection.get(ids=legacy)["ids"] == legacy

    def test_legacy_with_same_id_but_other_source_metadata_kept(self, lv, tmp_path):
        write(tmp_path, "a.md", A)
        chunks = lv.chunk_text(A, "a.md")
        lv.collection.add(ids=[chunks[0]["id"]], embeddings=[keyword_embed(A)],
                          documents=[chunks[0]["text"]],
                          metadatas=[{"source": "elsewhere.md", "type": "document"}])
        ingest(lv)
        assert lv.collection.get(ids=[chunks[0]["id"]])["ids"] == [chunks[0]["id"]]

    def test_import_and_startup_never_modify_records(self, lv, tmp_path, monkeypatch):
        write(tmp_path, "a.md", A)
        add_legacy(lv, "zzz.md", B)
        ingest(lv)
        snap = sorted(lv.collection.get()["ids"])
        _release()
        mod = _import(tmp_path, monkeypatch)                        # simulated restart
        mod.startup_health_check()
        assert sorted(mod.collection.get()["ids"]) == snap


# ══════════════════════════════════════════════════════════════════════════════
# 16, 17 — retrieval compatibility and security
# ══════════════════════════════════════════════════════════════════════════════
class TestCompatibilityAndSecurity:
    def test_retrieval_and_evidence_use_accurate_sources(self, lv, tmp_path):
        write(tmp_path, "notes/project.md", "project cat whiskers plan. " * 6)
        write(tmp_path, "archive/project.md", "project tax invoice. " * 6)
        ingest(lv)
        out = lv.retrieve("cat whiskers")
        top = out[0]
        assert set(top) >= {"id", "text", "source", "_type", "distance", "relevance"}
        assert top["source"] == "notes/project.md" and top["_type"] == "document"
        ctx, used = lv.build_context(out, 4000, 2000)
        assert "[Source: notes/project.md · note · relevance" in ctx
        assert {c["source"] for c in used} == {"notes/project.md", "archive/project.md"}

    def test_outside_path_still_refused(self, lv, tmp_path):
        (tmp_path / "elsewhere").mkdir()
        (tmp_path / "elsewhere" / "x.md").write_text(A)
        log = ingest(lv, str(tmp_path / "elsewhere"))
        assert log == [log[0]] and "Ingestion refused" in log[0] and records(lv) == []

    def test_symlink_escape_still_skipped(self, lv, tmp_path):
        (tmp_path / "outside").mkdir()
        (tmp_path / "outside" / "secret.md").write_text(B)
        os.symlink(tmp_path / "outside" / "secret.md", tmp_path / "my_vault" / "leak.md")
        write(tmp_path, "ok.md", A)
        ingest(lv)
        assert "beta tax" not in texts(lv) and "alpha cat" in texts(lv)

    def test_allowed_root_outside_vault_gets_abs_identity_and_is_never_swept(
            self, lv, tmp_path, monkeypatch):
        ext = tmp_path / "extra"
        ext.mkdir()
        p = ext / "e.md"
        p.write_text(B)
        monkeypatch.setenv("LIFEVAULT_ALLOWED_ROOTS", str(ext))
        ingest(lv, str(ext))
        (_, _, meta), = records(lv)
        assert meta["doc_id"] == f"abs:{p.resolve().as_posix()}" and meta["source"] == "extra/e.md"
        p.unlink()
        write(tmp_path, "a.md", A)
        ingest(lv)                                                   # full-vault sweep
        assert "beta tax" in texts(lv)                               # outside-vault doc untouched


# ══════════════════════════════════════════════════════════════════════════════
# Rev 6 — Issue A: incomplete scans never drive deletion
# ══════════════════════════════════════════════════════════════════════════════
class TestIncompleteScan:
    @pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores perms")
    def test_lost_directory_with_readable_sibling_keeps_its_documents(self, lv, tmp_path, monkeypatch):
        """A sub-tree becomes unreadable and its files look 'missing' (as after a
        lost mount reporting ENOENT) while a sibling folder is still readable."""
        write(tmp_path, "ok/a.md", A)
        write(tmp_path, "mnt/b.md", B)
        ingest(lv)
        mnt = tmp_path / "my_vault" / "mnt"
        real_lstat = os.lstat

        def lost_mount_lstat(path, *a, **k):
            if str(path).startswith(str(mnt) + os.sep):
                raise FileNotFoundError(2, "No such file or directory", str(path))
            return real_lstat(path, *a, **k)

        monkeypatch.setattr(os, "lstat", lost_mount_lstat)
        mnt.chmod(0)
        try:
            log = ingest(lv)
        finally:
            mnt.chmod(0o700)
        assert "beta tax" in texts(lv) and "alpha cat" in texts(lv)
        assert "Cleanup skipped" in log[-1] and "could not be read" in log[-1]
        assert not log[-1].startswith("✅")

    def test_injected_scan_error_skips_all_cleanup_but_ingests_readable_files(self, lv, tmp_path, monkeypatch):
        gone = write(tmp_path, "a.md", A)
        write(tmp_path, "mnt/b.md", B)
        ingest(lv)
        gone.unlink()                                   # genuinely deleted …
        write(tmp_path, "c.md", "dog car notes. " * 10)
        real = lv._list_dir

        def flaky(path):
            if Path(path).name == "mnt":
                raise OSError(5, "Input/output error")
            return real(path)

        monkeypatch.setattr(lv, "_list_dir", flaky)
        log = ingest(lv)
        assert "alpha cat" in texts(lv)                 # … but not swept: scan was incomplete
        assert "beta tax" in texts(lv) and by_source(lv, "c.md")
        assert "Cleanup skipped" in log[-1] and "mnt" in log[-1]
        monkeypatch.setattr(lv, "_list_dir", real)
        log = ingest(lv)                                # complete scan → now swept
        assert "alpha cat" not in texts(lv) and log[-1].startswith("✅")

    def test_entry_type_error_marks_scan_incomplete(self, lv, tmp_path, monkeypatch):
        gone = write(tmp_path, "a.md", A)
        write(tmp_path, "b.md", B)
        ingest(lv)
        gone.unlink()
        real = lv._list_dir

        class Broken:
            name = "weird"
            path = str(tmp_path / "my_vault" / "weird")

            def is_dir(self, follow_symlinks=True):
                raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(lv, "_list_dir", lambda p: real(p) + ([Broken()] if Path(p) == (tmp_path / "my_vault").resolve() else []))
        log = ingest(lv)
        assert "alpha cat" in texts(lv) and "Cleanup skipped" in log[-1]

    def test_unreadable_root_changes_nothing(self, lv, tmp_path, monkeypatch):
        write(tmp_path, "a.md", A)
        ingest(lv)
        before = records(lv)

        def boom(path):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(lv, "_list_dir", boom)
        log = ingest(lv)
        assert records(lv) == before and "nothing was changed" in log[-1]

    def test_scan_still_finds_files_and_skips_symlinked_dirs(self, lv, tmp_path):
        write(tmp_path, "x/y/deep.md", A)
        write(tmp_path, "UPPER.MD", B)                  # extension match stays case-sensitive
        (tmp_path / "outside").mkdir()
        (tmp_path / "outside" / "s.md").write_text(C)
        os.symlink(tmp_path / "outside", tmp_path / "my_vault" / "linkdir")
        ingest(lv)
        assert {m["source"] for _, _, m in records(lv)} == {"x/y/deep.md"}


# ══════════════════════════════════════════════════════════════════════════════
# Rev 6 — Issue B: successfully emptied documents
# ══════════════════════════════════════════════════════════════════════════════
class TestEmptiedDocuments:
    def test_nonempty_file_becoming_empty_removes_only_its_chunks(self, lv, tmp_path):
        write(tmp_path, "a.md", A)
        write(tmp_path, "b.md", B)
        ingest(lv)
        other = by_source(lv, "b.md")
        write(tmp_path, "a.md", "")
        log = ingest(lv)
        assert by_source(lv, "a.md") == [] and by_source(lv, "b.md") == other
        assert log[-1].startswith("✅") and "emptied" in log[-1]

    def test_whitespace_or_too_short_text_counts_as_empty(self, lv, tmp_path):
        write(tmp_path, "a.md", A)
        ingest(lv)
        write(tmp_path, "a.md", "  hi \n")
        ingest(lv)
        assert by_source(lv, "a.md") == []

    def test_empty_file_ingested_repeatedly_is_stable(self, lv, tmp_path):
        write(tmp_path, "e.md", "")
        write(tmp_path, "b.md", B)
        for _ in range(3):
            log = ingest(lv)
            assert log[-1].startswith("✅")
        assert by_source(lv, "e.md") == [] and len(by_source(lv, "b.md")) == 1

    def test_empty_file_later_receiving_content(self, lv, tmp_path):
        write(tmp_path, "e.md", "")
        ingest(lv)
        write(tmp_path, "e.md", C)
        ingest(lv)
        assert "delta garden" in texts(lv)

    def test_failed_read_is_not_treated_as_empty(self, lv, tmp_path, monkeypatch):
        write(tmp_path, "a.md", A)
        ingest(lv)
        before = by_source(lv, "a.md")
        write(tmp_path, "a.md", "")
        real = lv.read_file
        monkeypatch.setattr(lv, "read_file", lambda p, strict=False: (_ for _ in ()).throw(OSError("EIO"))
                            if p.name == "a.md" else real(p, strict=strict))
        log = ingest(lv)
        assert by_source(lv, "a.md") == before and "unreadable" in log[-1]

    @pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores perms")
    def test_unreadable_file_retains_chunks(self, lv, tmp_path):
        f = write(tmp_path, "a.md", A)
        ingest(lv)
        f.chmod(0)
        try:
            ingest(lv)
        finally:
            f.chmod(0o600)
        assert "alpha cat" in texts(lv)

    def test_emptying_with_delete_failure_keeps_chunks_and_retries(self, lv, tmp_path, monkeypatch):
        write(tmp_path, "a.md", A)
        ingest(lv)
        write(tmp_path, "a.md", "")
        real = lv.collection
        monkeypatch.setattr(lv, "collection", Proxy(real, delete=_fails("locked")))
        log = ingest(lv)
        lv.collection = real
        assert "alpha cat" in texts(lv) and not log[-1].startswith("✅")
        ingest(lv)
        assert by_source(lv, "a.md") == []

    def test_emptied_file_in_subfolder_ingest_also_updates(self, lv, tmp_path):
        write(tmp_path, "notes/a.md", A)
        ingest(lv)
        write(tmp_path, "notes/a.md", "")
        ingest(lv, str(tmp_path / "my_vault" / "notes"))
        assert by_source(lv, "notes/a.md") == []


# ══════════════════════════════════════════════════════════════════════════════
# Rev 6 — Issue C: legacy ownership is never inferred
# ══════════════════════════════════════════════════════════════════════════════
class TestLegacyOwnership:
    def test_same_name_identical_text_from_unrelated_source_is_preserved(self, lv, tmp_path):
        # The legacy record could just as well have come from ~/elsewhere/project.md.
        write(tmp_path, "notes/project.md", A)
        legacy = add_legacy(lv, "project.md", A)
        before = lv.collection.get(ids=legacy, include=["documents", "metadatas"])
        ingest(lv)
        after = lv.collection.get(ids=legacy, include=["documents", "metadatas"])
        assert after["ids"] == legacy
        assert after["documents"] == before["documents"] and after["metadatas"] == before["metadatas"]
        assert {m["source"] for _, _, m in records(lv)} == {"project.md", "notes/project.md"}

    def test_no_unrelated_record_is_touched(self, lv, tmp_path):
        write(tmp_path, "chat_history.md", ENTRY.format(d=1))
        write(tmp_path, "a.md", A)
        write(tmp_path, "b.md", B)
        legacy = add_legacy(lv, "a.md", A) + add_legacy(lv, "other.md", C)
        ingest(lv)
        snap = sorted(lv.collection.get()["ids"])
        write(tmp_path, "a.md", C)
        (tmp_path / "my_vault" / "b.md").unlink()
        ingest(lv)
        now = set(lv.collection.get()["ids"])
        assert set(legacy) <= now                         # every legacy record still there
        assert len(records(lv, "memory")) == 1

    def test_only_schema2_records_with_doc_id_are_ever_swept(self, lv, tmp_path):
        p = write(tmp_path, "a.md", A)
        legacy = add_legacy(lv, "a.md", A)
        ingest(lv)
        p.unlink()
        write(tmp_path, "b.md", B)
        ingest(lv)
        assert lv.collection.get(ids=legacy)["ids"] == legacy          # ambiguous → kept
        assert all(m.get("schema") != 2 or m["source"] == "b.md"
                   for _, _, m in records(lv))                         # demonstrable → swept
