"""
LifeVault v2.0 — Production-Ready Core Engine
==========================================
Stack : Ollama (local LLM) + ChromaDB (vector search) + Gradio (UI)
GPU   : Ollama auto-detects your GPU. No extra config needed.

__version__ = "2.0"

Improvements over v1
--------------------
[BUG-01] Removed duplicate `import os`
[BUG-02] Fixed `stats_live` being static in UI (now reactive)
[BUG-03] Fixed retrieval ratio 10:3 memory→notes (too memory-heavy, now balanced)
[BUG-04] Fixed synthesis prompt having no length guard (could overflow context)
[BUG-05] Fixed `save_chat_memory` blocking the main thread (now runs in background thread)
[BUG-06] Fixed decompose regex breaking when LLM uses "1)" or "First:" format (multi-pattern)
[BUG-07] Fixed chunk ID collision: uses file+position hash, not content hash

[FEAT-01] Startup health check — tells user exactly what's wrong before first query
[FEAT-02] Retry logic on ask_llm — 2 retries with 1s backoff before failing
[FEAT-03] Query result cache — same question isn't re-sent to Ollama
[FEAT-04] ask_llm timeout — 120s hard cap, returns a clean error instead of hanging
[FEAT-05] get_vault_stats() — rich stats dict used by the UI header
[FEAT-06] clear_memory() — wipe only chat_history chunks, keep documents
[FEAT-07] Ollama streaming for fast_query — tokens appear as they're generated
[FEAT-08] MAX_CONTEXT_CHARS guard — truncates sub-answers before synthesis (superseded: SUB_ANSWERS_CHARS / build_context)
[FEAT-09] Configurable retrieval balance via MEMORY_K / NOTES_K constants
[FEAT-10] Context-aware synthesis — history window is safely capped

Phase 2 — retrieval quality
---------------------------
[RET-01] build_context(): separate, roll-over character budgets for documents
         and memories; memories can no longer crowd out all document evidence
[RET-02] Relevance = cosine similarity computed from the stored embeddings
         (valid whatever the vector norm / Chroma metric); optional thresholds
[RET-03] Deduplication by normalised text; true metadata type preserved
[RET-04] Deep mode: initial retrieval reaches the synthesis prompt; evidence
         panel = chunks actually placed in prompts
[RET-05] Stricter sub-question parsing; direct-answer fallback
[RET-06] Conversation history: Gradio 6 content blocks, placeholders, errors

Phase 3 — reliability
---------------------
[LLM-01] Failures raise LLMError(kind) — never returned as answer strings
[LLM-02] ollama.Client(timeout=…) — httpx-enforced timeouts, no worker threads
[LLM-03] Exception classification for ollama 0.6.1 / httpx error types
[LLM-04] Errors are flagged, prefixed, never cached and never saved as memory
[LLM-05] Startup health check requires the exact configured model tag
[MEM-01] Clear Memory is ordered with saves: one lock + a clear counter (epoch)
[RET-07] Sub-question parser also splits one-line lists and question runs
[RET-08] Deep evidence keeps every step's own relevance for a reused excerpt
[PRIV-01] ChromaDB anonymized telemetry disabled per client (defence in depth)
[IDX-01] Document identity = resolved vault-relative path (schema-2 chunk ids)
[IDX-02] Prepare → add missing → verify → only then delete obsolete chunks
[IDX-04] read_file(strict=True): unreadable files never replace/delete chunks
[IDX-05] Legacy records are never deleted (Rev 6: reported only)
[IDX-06] chat_history.md indexed per entry — no duplicate memories
[IDX-07] Full-vault sweep of deleted/renamed documents (confirmed missing only)
[IDX-08] Explicit scan; any traversal error → scan incomplete → no sweep
[IDX-09] Successfully emptied documents drop their old chunks
"""

import os
import re
import time
import hashlib
import logging
import math
import threading
from collections import Counter
from fnmatch import fnmatchcase
from datetime import datetime
from pathlib import Path
from typing import Generator

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

from security import (
    ConfigError, PathValidationError, partition_contained, resolve_ingest_path,
)

# ── LOGGING ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("LifeVault")
logger.setLevel(logging.DEBUG)

_fh = logging.FileHandler("logs/lifevault_debug.log", encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_sh = logging.StreamHandler()
_sh.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
_fh.setFormatter(_fmt)
_sh.setFormatter(_fmt)
logger.addHandler(_fh)
logger.addHandler(_sh)


# ── CONFIG ────────────────────────────────────────────────────────────────────
VAULT_DIR      = "./my_vault"
OLLAMA_MODEL   = "qwen2.5:7b"
EMBED_MODEL    = "all-MiniLM-L6-v2"
CHUNK_SIZE     = 800          # [BUG-07 adj] larger chunks → less mid-sentence cuts
CHUNK_OVERLAP  = 120
MEMORY_K       = 10           # [FEAT-09] memory chunks per query
NOTES_K        = 8            # [FEAT-09] document chunks per query (notes > memory)
MAX_SUB_CALLS  = 4
LLM_TIMEOUT    = 120          # [FEAT-04] seconds before ask_llm gives up
MAX_RETRIES    = 2            # [FEAT-02] retries on LLM failure
HISTORY_WINDOW = 6            # last N messages included in prompts

# ── RETRIEVAL / CONTEXT  [RET-01..05] ─────────────────────────────────────────
# Budgets are Unicode characters (not tokens). Separators between entries count
# toward a budget; the two fixed section headers do not. Unused budget of one
# source rolls over to the other.
DOC_CONTEXT_CHARS           = 4000   # reserved for document evidence
MEMORY_CONTEXT_CHARS        = 2000   # reserved for conversation memory
SUB_DOC_CONTEXT_CHARS       = 1700   # per Deep-mode sub-question
SUB_MEMORY_CONTEXT_CHARS    = 800
SYNTH_DOC_EVIDENCE_CHARS    = 1000   # direct evidence in the synthesis prompt
SYNTH_MEMORY_EVIDENCE_CHARS = 500
SUB_ANSWERS_CHARS           = 5000   # cap on combined sub-answers
MIN_TRUNCATED_CHARS         = 200    # truncate a chunk only if ≥ this room remains
# Relevance = cosine similarity in [-1, 1]. A threshold that is None, ≤ 0 or
# not a finite number disables filtering for that source; chunks without a
# valid score are never filtered. PROVISIONAL values — not empirically tuned.
MIN_DOC_RELEVANCE           = 0.0    # documents are never filtered out
MIN_MEMORY_RELEVANCE        = 0.25
RETRIEVAL_OVERFETCH         = 2      # fetch k × this candidates, then filter/dedup
# ─────────────────────────────────────────────────────────────────────────────


# ── SETUP ─────────────────────────────────────────────────────────────────────
embedder   = SentenceTransformer(EMBED_MODEL)
# [PRIV-01] Opt out of ChromaDB's anonymized product telemetry via its
# documented per-client setting (an explicit value also overrides any
# ANONYMIZED_TELEMETRY env var). In chromadb 1.5.8 the capture is already a
# no-op; this is defence in depth against version changes. It does not affect
# other network traffic (e.g. the first-run embedding-model download).
_ChromaSettings = getattr(chromadb, "Settings", None)
if _ChromaSettings is not None:
    chroma = chromadb.PersistentClient(
        path="./.lifevault_db", settings=_ChromaSettings(anonymized_telemetry=False))
else:  # stand-in modules used by isolated tests
    logger.warning("chromadb.Settings unavailable; telemetry setting not applied.")
    chroma = chromadb.PersistentClient(path="./.lifevault_db")
collection = chroma.get_or_create_collection("vault")

# Simple in-memory query cache  [FEAT-03]
_query_cache: dict[str, str] = {}

# Lock for thread-safe memory writes  [BUG-05]. [MEM-01] Also held by
# clear_memory() and by ingestion while it indexes chat_history.md, so a clear
# never interleaves with a save or a memory re-index.
_memory_lock = threading.Lock()
# [MEM-01] Incremented by every clear_memory() (under _memory_lock). A save
# carries the epoch from when its question was asked; a save whose epoch is
# older than the current one belongs to a conversation the user has cleared
# and is dropped.
_memory_epoch = 0
MEMORY_CLEAR_TIMEOUT = 30   # seconds clear_memory() waits for a running save/index


# ── HEALTH CHECK  [FEAT-01] ───────────────────────────────────────────────────
def startup_health_check() -> dict:
    """
    Run at import time or on UI load.
    Returns {"ok": bool, "server": bool, "model": bool, "vault": bool, "message": str}
    """
    status = {"ok": True, "server": False, "model": False, "vault": False, "message": ""}
    issues = []

    # 1. Is Ollama running?
    try:
        models = _get_llm_client(HEALTH_TIMEOUT).list()  # [LLM-02] bounded
        status["server"] = True
        available = {m["model"] for m in models.get("models", [])}
        # [LLM-05] Exact tag only (an untagged name means ":latest"): Ollama
        # answers 404 for "qwen2.5:7b" even when "qwen2.5:3b" is installed.
        wanted = OLLAMA_MODEL if ":" in OLLAMA_MODEL else f"{OLLAMA_MODEL}:latest"
        if wanted in available or OLLAMA_MODEL in available:
            status["model"] = True
            logger.info(f"Ollama OK — model '{OLLAMA_MODEL}' available.")
        else:
            issues.append(
                f"Model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}"
            )
    except Exception as e:
        issues.append(f"Ollama not running: {e}. Start it with: ollama serve")

    # 2. Is the vault directory accessible?
    vault = Path(VAULT_DIR)
    if vault.exists():
        files = list(vault.rglob("*.md")) + list(vault.rglob("*.txt")) + \
                list(vault.rglob("*.pdf"))
        status["vault"] = True
        logger.info(f"Vault OK — {len(files)} files found.")
    else:
        vault.mkdir(parents=True, exist_ok=True)
        issues.append(f"Vault folder '{VAULT_DIR}' was empty — created it. "
                       "Add .md, .txt, or .pdf files and ingest.")

    # 3. ChromaDB
    try:
        _ = collection.count()
        logger.info(f"ChromaDB OK — {collection.count()} chunks indexed.")
    except Exception as e:
        issues.append(f"ChromaDB error: {e}")

    if issues:
        status["ok"] = False
        status["message"] = " | ".join(issues)
    else:
        status["message"] = (
            f"✅ All systems ready. "
            f"{collection.count()} chunks indexed."
        )
    return status


# ── FILE INGESTION ────────────────────────────────────────────────────────────
def read_file(path: Path, strict: bool = False) -> str:
    """
    Parse .md / .txt / .pdf / .json / .eml → plain text.
    Default: errors are logged and "" (or a "[Skipped …]" note for a missing
    library) is returned, as in v2.0. [IDX-04] strict=True re-raises instead,
    so ingestion can tell "unreadable" from "empty" and keep the old index.
    """
    suffix = path.suffix.lower()
    logger.debug(f"Reading: {path.name}")
    try:
        if suffix in (".md", ".txt"):
            return path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".pdf":
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
                # Strip pages that returned nothing (scanned / image-only)
                return "\n".join(p for p in pages if p.strip())
        elif suffix == ".json":
            import json
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            # Flatten nested JSON to readable text
            return _flatten_json(data)
        elif suffix == ".eml":
            import email
            msg = email.message_from_string(
                path.read_text(encoding="utf-8", errors="ignore")
            )
            parts = []
            if msg.get("subject"):
                parts.append(f"Subject: {msg['subject']}")
            if msg.get("from"):
                parts.append(f"From: {msg['from']}")
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        parts.append(part.get_payload(decode=True).decode("utf-8", errors="ignore"))
            else:
                parts.append(msg.get_payload(decode=True).decode("utf-8", errors="ignore"))
            return "\n".join(parts)
    except ImportError as e:
        logger.warning(f"Missing dependency for {path.name}: {e}")
        if strict:
            raise
        return f"[Skipped {path.name} — install missing library: {e}]"
    except Exception as e:
        logger.error(f"Failed to read {path.name}: {e}")
        if strict:
            raise
        return ""
    return ""


def _flatten_json(obj, prefix: str = "") -> str:
    """Recursively turn a JSON object into readable key: value lines."""
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lines.append(_flatten_json(v, f"{prefix}{k}: "))
    elif isinstance(obj, list):
        for item in obj:
            lines.append(_flatten_json(item, prefix))
    else:
        lines.append(f"{prefix}{obj}")
    return "\n".join(lines)


def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split text into overlapping chunks.
    [BUG-07] ID uses source + position, not content — avoids stale chunk buildup.
    """
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 60:
            if source == "chat_history.md":
                chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            else:
                chunk_id = hashlib.md5(f"{source}::{idx}".encode()).hexdigest()
            chunks.append({"text": chunk, "source": source, "id": chunk_id})
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── INDEX CONSISTENCY  [IDX-01..09] ──────────────────────────────────────────
# Schema-2 document chunks carry their document identity:
#   id       = md5("v2::<doc_id>::<sha256(text)>::<chunk_index>")
#   metadata = {source, type="document", doc_id, doc_hash, chunk_index,
#               chunk_count, schema=2}
# doc_id is the RESOLVED path relative to the default vault ("vault:<rel>"),
# or "abs:<absolute path>" for files under another permitted root.
# Legacy (pre-schema-2) records — id md5("<filename>::<idx>"), metadata
# {source, type} — are never rewritten or deleted (ownership can't be
# proven from a bare filename); see _superseded_legacy_ids().
# Memories (chat_history.md) keep their content-hash ids and metadata.
INDEX_SCHEMA     = 2
MEMORY_FILE_NAME = "chat_history.md"
_MEMORY_ENTRY_RE = re.compile(r"\n\n### \[\d{4}-\d{2}-\d{2} \d{2}:\d{2} [AP]M\] User: ")
_ingest_lock     = threading.Lock()


def _get(**kwargs) -> dict:
    """collection.get() with absent fields normalised to None (ids → list)."""
    r = collection.get(**kwargs)
    return {"ids": list(r.get("ids") or []),
            "documents": r.get("documents"), "metadatas": r.get("metadatas")}


def _vault_root() -> Path:
    return Path(VAULT_DIR).resolve()


def _document_identity(file: Path, folder: Path, vault_root: Path) -> tuple[str, str]:
    """[IDX-01] (doc_id, display source) from the file's resolved path, so
    notes/project.md and archive/project.md are different documents and the
    identity is the same whichever folder (vault or sub-folder) is ingested."""
    real = file.resolve(strict=True)
    try:
        rel = real.relative_to(vault_root).as_posix()
        return f"vault:{rel}", rel
    except ValueError:
        try:
            shown = f"{folder.name}/{real.relative_to(folder).as_posix()}"
        except ValueError:
            shown = real.name
        return f"abs:{real.as_posix()}", shown


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _v2_chunk_id(doc_id: str, doc_hash: str, idx: int) -> str:
    return hashlib.md5(f"v2::{doc_id}::{doc_hash}::{idx}".encode("utf-8", "surrogatepass")).hexdigest()


def _delete_ids(ids: list[str]) -> tuple[int, str | None]:
    """Delete ids and confirm they are gone. Returns (removed, error)."""
    if not ids:
        return 0, None
    try:
        collection.delete(ids=ids)
        remaining = _get(ids=ids)["ids"]
    except Exception as e:
        logger.error(f"Index delete failed: {type(e).__name__}: {e}")
        return 0, "could not remove obsolete chunks (will retry on next ingest)"
    if remaining:
        return len(ids) - len(remaining), f"{len(remaining)} obsolete chunks still present (will retry)"
    return len(ids), None


def _verify_chunks(ids: list[str], texts: list[str], doc_id: str, doc_hash: str) -> bool:
    """[IDX-02] Every prepared chunk is stored with the expected text/metadata."""
    r = _get(ids=ids, include=["documents", "metadatas"])
    if set(r["ids"]) != set(ids):
        return False
    expected = dict(zip(ids, texts))
    if r["documents"] is not None and any(
            d != expected[i] for i, d in zip(r["ids"], r["documents"])):
        return False
    if r["metadatas"] is not None and any(
            (m or {}).get("doc_id") != doc_id or (m or {}).get("doc_hash") != doc_hash
            for m in r["metadatas"]):
        return False
    return True


def _superseded_legacy_ids(file_name: str, text: str) -> list[str]:
    """
    [IDX-05, Rev 6] READ-ONLY. Legacy records whose id is md5("<name>::k"),
    whose metadata is legacy {source == name, type == document}, and whose
    text equals this file's current chunk k — i.e. text that is now ALSO
    indexed under the file's path. They are reported, never deleted: a
    legacy record stores only a bare filename, so it may equally have come
    from a same-named file outside the vault (pre-v2.0 ingestion allowed any
    path; extra allowed roots still exist). Ownership is therefore never
    provable from the stored evidence.
    """
    legacy = chunk_text(text, file_name)
    ids = [c["id"] for c in legacy]
    if not ids:
        return []
    r = _get(ids=ids, include=["documents", "metadatas"])
    if not r["ids"] or r["documents"] is None or r["metadatas"] is None:
        return []
    expected = {c["id"]: c["text"] for c in legacy}
    return [i for i, d, m in zip(r["ids"], r["documents"], r["metadatas"])
            if m and "schema" not in m and "doc_id" not in m
            and m.get("type") == "document" and m.get("source") == file_name
            and d == expected.get(i)]


def _ingest_document(file: Path, folder: Path, vault_root: Path, check_legacy: bool) -> dict:
    """
    [IDX-02/03] Prepare → add missing → verify → only then delete this
    document's obsolete chunks. Any failure before verification leaves the
    previous chunks untouched (and removes only chunks this call created).
    """
    res = {"status": "failed", "added": 0, "removed": 0, "superseded": [], "detail": ""}
    doc_id, source = _document_identity(file, folder, vault_root)
    res["source"], res["doc_id"] = source, doc_id
    try:
        text = read_file(file, strict=True)
    except Exception:
        res["detail"] = "unreadable — previous index entries kept"
        return res
    pieces = chunk_text(text, source) if text.strip() else []
    if not pieces:
        # [IDX-09, Rev 6] The read SUCCEEDED (strict) and yielded no indexable
        # text: that is a validated new state with zero chunks, so the old
        # chunks of this document — and only this doc_id — are removed and
        # the removal is confirmed. Failed reads never reach this point.
        existing = _get(where={"doc_id": doc_id})["ids"]
        if not existing:
            res.update(status="empty", detail="no indexable text")
            return res
        removed, err = _delete_ids(existing)
        res["removed"] = removed
        if err:
            res.update(status="partial", detail=err)
            return res
        res.update(status="emptied", detail=f"now empty — {removed} previous chunks removed")
        return res

    doc_hash = _sha256(text)
    n = len(pieces)
    ids = [_v2_chunk_id(doc_id, doc_hash, i) for i in range(n)]
    texts = [p["text"] for p in pieces]
    metas = [{"source": source, "type": "document", "doc_id": doc_id, "doc_hash": doc_hash,
              "chunk_index": i, "chunk_count": n, "schema": INDEX_SCHEMA} for i in range(n)]
    if len(set(ids)) != n or any(not t.strip() for t in texts):
        res["detail"] = "invalid chunk preparation — previous index entries kept"
        return res

    existing = _get(where={"doc_id": doc_id})["ids"]
    present = set(_get(ids=ids)["ids"])
    missing = [i for i in range(n) if ids[i] not in present]
    created = [ids[i] for i in missing]
    if missing:
        try:
            collection.add(
                ids=created,
                embeddings=[embedder.encode(texts[i]).tolist() for i in missing],
                documents=[texts[i] for i in missing],
                metadatas=[metas[i] for i in missing],
            )
        except Exception as e:
            logger.error(f"Index add failed for {source}: {type(e).__name__}: {e}")
            _delete_ids(created)  # best effort; these ids did not exist before
            res["detail"] = "could not store new chunks — previous index entries kept"
            return res
    try:
        verified = _verify_chunks(ids, texts, doc_id, doc_hash)
    except Exception as e:
        logger.error(f"Index verification error for {source}: {type(e).__name__}: {e}")
        verified = False
    if not verified:
        _delete_ids(created)
        res["detail"] = "new chunks could not be verified — previous index entries kept"
        return res

    keep = set(ids)
    removed, err = _delete_ids([i for i in existing if i not in keep])
    res["added"], res["removed"] = len(missing), removed
    if check_legacy:
        res["superseded"] = _superseded_legacy_ids(file.name, text)
    if missing:
        res["status"] = "updated" if [i for i in existing if i not in keep] else "new"
    else:
        res["status"] = "updated" if removed else "unchanged"
    if err:
        res.update(status="partial", detail=err)
    return res


def _memory_segments(text: str) -> list[str]:
    """Split chat_history.md back into the exact entries save_chat_memory wrote."""
    starts = [m.start() for m in _MEMORY_ENTRY_RE.finditer(text)]
    if not starts:
        return [text]
    segs = [text[:starts[0]]] if starts[0] > 0 else []
    return segs + [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]


def _ingest_memory_file(file: Path) -> dict:
    """
    [IDX-06] Index chat_history.md per entry, exactly as save_chat_memory
    does, so content-hash ids match and re-ingestion adds no duplicates.
    Memories are only ever added here — never deleted.
    """
    res = {"status": "failed", "added": 0, "removed": 0, "superseded": [],
           "detail": "", "source": MEMORY_FILE_NAME, "doc_id": None}
    with _memory_lock:   # [MEM-01] a concurrent clear waits for (and then removes) these
        return _index_memory_file(file, res)


def _index_memory_file(file: Path, res: dict) -> dict:
    try:
        text = read_file(file, strict=True)
    except Exception:
        res["detail"] = "unreadable — existing memories kept"
        return res
    chunks, seen = [], set()
    for seg in _memory_segments(text):
        for c in chunk_text(seg, MEMORY_FILE_NAME):
            if c["id"] not in seen:
                seen.add(c["id"])
                chunks.append(c)
    if not chunks:
        res.update(status="empty", detail="no memories")
        return res
    present = set(_get(ids=[c["id"] for c in chunks])["ids"])
    new = [c for c in chunks if c["id"] not in present]
    if new:
        try:
            collection.add(
                ids=[c["id"] for c in new],
                embeddings=[embedder.encode(c["text"]).tolist() for c in new],
                documents=[c["text"] for c in new],
                metadatas=[{"source": MEMORY_FILE_NAME, "type": "memory"} for _ in new],
            )
        except Exception as e:
            logger.error(f"Memory index add failed: {type(e).__name__}: {e}")
            res["detail"] = "could not store memories (will retry on next ingest)"
            return res
    res.update(status="new" if new else "unchanged", added=len(new))
    return res


def _cleanup_removed_documents(vault_root: Path, scanned: set[str]) -> dict:
    """
    [IDX-07] Full-vault ingest only. Delete schema-2 chunks of vault
    documents that were not scanned AND whose file is confirmed missing
    (os.lstat → FileNotFoundError). Unknown states (permission errors, files
    present but not scanned, ids outside the vault) are kept and reported.
    """
    out = {"removed": 0, "docs": 0, "kept": 0, "error": None, "note": None}
    r = _get(where={"schema": INDEX_SCHEMA}, include=["metadatas"])
    if r["metadatas"] is None:
        out["note"] = "cleanup skipped: index metadata unavailable"
        return out
    by_doc: dict[str, list[str]] = {}
    for cid, m in zip(r["ids"], r["metadatas"]):
        d = (m or {}).get("doc_id")
        if (m or {}).get("type") == "document" and isinstance(d, str) and d.startswith("vault:"):
            by_doc.setdefault(d, []).append(cid)
    for doc_id, ids in by_doc.items():
        if doc_id in scanned:
            continue
        candidate = (vault_root / doc_id[len("vault:"):])
        if ".." in Path(doc_id[len("vault:"):]).parts:
            out["kept"] += 1
            continue
        try:
            os.lstat(candidate)
            out["kept"] += 1              # still on disk, just not scanned → keep
            continue
        except FileNotFoundError:
            pass
        except OSError:
            out["kept"] += 1              # can't tell → keep
            continue
        removed, err = _delete_ids(ids)
        out["removed"] += removed
        out["docs"] += 1 if removed else 0
        if err:
            out["error"] = err
    return out


def _list_dir(path) -> list:
    """os.scandir() as a list (separate so tests can inject failures)."""
    with os.scandir(path) as it:
        return list(it)


def _scan_folder(folder: Path, patterns: list[str]) -> tuple[list[Path], list[str]]:
    """
    [IDX-08] Walk `folder` recording every traversal problem instead of
    silently skipping it (pathlib.rglob ignores unreadable directories —
    verified on Python 3.14). Symlinked directories are not descended into,
    as with rglob's default. Returns (matching files, problems); a non-empty
    problems list means the scan is INCOMPLETE and must not drive deletion.
    Extension matching stays case-sensitive (fnmatchcase), as before.
    """
    files: list[Path] = []
    problems: list[str] = []
    stack = [folder]
    while stack:
        d = stack.pop()
        try:
            entries = _list_dir(d)
        except OSError as e:
            problems.append(f"{_rel(d, folder)} ({type(e).__name__})")
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    continue
            except OSError as e:
                problems.append(f"{_rel(Path(entry.path), folder)} ({type(e).__name__})")
                continue
            if any(fnmatchcase(entry.name, p) for p in patterns):
                files.append(Path(entry.path))
    order = {p: i for i, p in enumerate(patterns)}
    files.sort(key=lambda f: (min(order[p] for p in patterns if fnmatchcase(f.name, p)), f.as_posix()))
    return files, problems


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix() or "."
    except ValueError:
        return path.name


def _count_legacy_documents() -> int | None:
    r = _get(where={"type": "document"}, include=["metadatas"])
    if r["metadatas"] is None:
        return None
    return sum(1 for m in r["metadatas"] if m is not None and "schema" not in m)


def ingest_vault(vault_dir: str = VAULT_DIR) -> Generator:
    """
    Generator: yields (log_string, progress_fraction) tuples.
    [IDX-01..07] Per-file safe replacement keyed by document identity.
    Ingesting the whole default vault additionally removes chunks of
    deleted/renamed documents — but only after a COMPLETE scan (Rev 6).
    Legacy records are never deleted, only reported.
    Reports "✅ Done" only when every step succeeded.
    """
    logger.info(f"Ingestion started: {vault_dir}")
    # [SEC-02] Confine ingestion to permitted roots; only the default vault
    # may be auto-created (resolve_ingest_path handles that).
    try:
        folder = resolve_ingest_path(vault_dir, default_vault=VAULT_DIR)
    except (PathValidationError, ConfigError) as e:
        logger.warning(f"Ingestion refused for requested path: {e}")
        yield f"⛔  Ingestion refused: {e}", 0.0
        return

    if not _ingest_lock.acquire(blocking=False):
        yield "⏳  Another ingestion is already running — try again when it finishes.", 0.0
        return
    try:
        yield from _ingest_folder(folder, vault_dir)
    finally:
        _ingest_lock.release()


def _ingest_folder(folder: Path, vault_dir: str) -> Generator:
    extensions = ["*.md", "*.txt", "*.pdf", "*.json", "*.eml"]
    try:
        files, scan_problems = _scan_folder(folder, extensions)
    except OSError as e:
        logger.error(f"Vault scan failed: {type(e).__name__}: {e}")
        yield "❌  Could not scan the folder; nothing was changed. See the debug log.", 0.0
        return
    for prob in scan_problems:
        logger.warning(f"Scan incomplete — could not read: {prob}")
    if scan_problems and not files:
        yield ("❌  Could not scan the folder (" + "; ".join(scan_problems[:3])
               + "); nothing was changed. See the debug log."), 0.0
        return
    # [BUG-FIX] Keep chat_history.md so memories survive DB rebuilds
    # files = [f for f in files if f.name != "chat_history.md"]

    # [SEC-03] Skip symlinks (and files under symlinked dirs) escaping the tree
    files, escaped = partition_contained(files, folder)
    for f, reason in escaped:
        logger.warning(f"Skipped {f.relative_to(folder)}: {reason}")

    if not files:
        yield f"⚠️  No supported files in {vault_dir}.", 0.0
        return

    vault_root = _vault_root()
    full_vault = folder == vault_root
    total_files = len(files)
    yield f"Found {total_files} files. Starting embedding...", 0.02

    log_lines: list[str] = []
    stats = Counter()
    scanned: set[str] = set()
    superseded: set[str] = set()
    failures = 0
    for i, file in enumerate(files):
        try:
            if file.name == MEMORY_FILE_NAME:
                r = _ingest_memory_file(file)
            else:
                doc_id, _ = _document_identity(file, folder, vault_root)
                scanned.add(doc_id)
                r = _ingest_document(file, folder, vault_root, check_legacy=full_vault)
        except Exception as e:  # unexpected — keep going, keep old entries
            logger.error(f"Ingestion error for {file.name}: {type(e).__name__}: {e}", exc_info=e)
            r = {"status": "failed", "added": 0, "removed": 0, "superseded": [],
                 "detail": "unexpected error — previous index entries kept",
                 "source": file.name}
        stats[r["status"]] += 1
        stats["added"] += r["added"]
        stats["removed"] += r["removed"]
        superseded.update(r["superseded"])
        if r["status"] in ("failed", "partial"):
            failures += 1
            log_lines.append(f"✗  {r['source']}  ({r['detail']})")
        elif r["status"] == "empty":
            log_lines.append(f"⚠  {r['source']}  ({r['detail']})")
        elif r["status"] == "emptied":
            log_lines.append(f"✓  {r['source']}  (emptied: {r['removed']} previous chunks removed)")
        else:
            extra = f", {r['removed']} obsolete removed" if r["removed"] else ""
            log_lines.append(f"✓  {r['source']}  ({r['status']}: {r['added']} new chunks{extra})")
        logger.info(f"{r['source']}: {r['status']} (+{r['added']} / -{r['removed']})")
        yield "\n".join(log_lines), (i + 1) / total_files

    notes: list[str] = []
    if scan_problems:  # [IDX-08] incomplete scan: report; never reconcile deletions
        failures += 1
        shown = "; ".join(scan_problems[:3]) + (" …" if len(scan_problems) > 3 else "")
        notes.append(f"{len(scan_problems)} folder(s) or entries could not be read ({shown}); "
                     f"files in them were not processed.")
    if full_vault:
        if scan_problems:
            notes.append("Cleanup skipped: the scan was incomplete, so no chunks were removed "
                         "for files that look missing.")
        elif not scanned:
            notes.append("Cleanup skipped: no documents found in the scan.")
        else:
            c = _cleanup_removed_documents(vault_root, scanned)
            stats["removed"] += c["removed"]
            if c["docs"]:
                notes.append(f"Removed {c['removed']} chunks of {c['docs']} deleted or renamed documents.")
            if c["kept"]:
                notes.append(f"Kept chunks of {c['kept']} documents whose removal could not be confirmed.")
            if c["note"]:
                notes.append(c["note"])
            if c["error"]:
                failures += 1
                notes.append(f"Cleanup problem: {c['error']}.")
        legacy = _count_legacy_documents()
        if legacy:
            notes.append(f"{legacy} chunks indexed by an older LifeVault version were kept "
                         f"(their owning file can't be proven; {len(superseded)} of them repeat "
                         f"text now indexed under its current path). See README to rebuild.")
    else:
        notes.append("Deleted/renamed-file cleanup runs only when the whole vault is ingested.")

    head = ("✅  Done" if failures == 0 else f"⚠️  Finished with problems in {failures} item(s) — "
            f"their previous index entries were kept")
    summary = (
        f"{head} — {stats['added']} new chunks indexed from {total_files} files "
        f"({stats['new']} new, {stats['updated']} updated, {stats['unchanged']} unchanged, "
        f"{stats['empty'] + stats['emptied']} without text ({stats['emptied']} emptied), "
        f"{stats['failed'] + stats['partial']} with problems); "
        f"{stats['removed']} obsolete chunks removed.\n"
        f"Total vault size: {collection.count()} chunks.\n"
        + ("".join(f"{n}\n" for n in notes)) + "\n"
    )
    logger.info(f"Ingestion complete: +{stats['added']} / -{stats['removed']}, {failures} problems.")
    yield summary + "\n".join(log_lines), 1.0


# ── MEMORY ────────────────────────────────────────────────────────────────────
def memory_epoch() -> int:
    """[MEM-01] Current clear counter; capture it when a question is asked."""
    return _memory_epoch


def save_chat_memory(question: str, answer: str, epoch: int | None = None) -> None:
    """
    [BUG-05] Runs in a background thread so the UI response is never delayed.
    Saves to chat_history.md and immediately indexes into ChromaDB.
    [LLM-04] Empty answers and error messages (UI_ERROR_PREFIX) are never saved.
    [MEM-01] `epoch` = memory_epoch() when the question was asked (default:
    now). If Clear Memory ran since then, the answer is not saved.
    """
    if not str(answer or "").strip() or str(answer).lstrip().startswith(UI_ERROR_PREFIX.strip()):
        logger.debug("Memory save skipped: empty or error answer.")
        return
    if epoch is None:
        epoch = _memory_epoch

    def _write():
        with _memory_lock:
            if epoch != _memory_epoch:
                logger.info("Memory save skipped: memory was cleared after this question was asked.")
                return
            try:
                mem_file = Path(VAULT_DIR) / "chat_history.md"
                mem_file.parent.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y-%m-%d %I:%M %p")
                entry = (
                    f"\n\n### [{ts}] User: {question}\n\n"
                    f"### [{ts}] LifeVault: {answer}\n\n---\n"
                )
                with open(mem_file, "a", encoding="utf-8") as f:
                    f.write(entry)

                chunks = chunk_text(entry, "chat_history.md")
                for chunk in chunks:
                    if not collection.get(ids=[chunk["id"]])["ids"]:
                        emb = embedder.encode(chunk["text"]).tolist()
                        collection.add(
                            ids=[chunk["id"]],
                            embeddings=[emb],
                            documents=[chunk["text"]],
                            metadatas=[{"source": "chat_history.md", "type": "memory"}],
                        )
                logger.debug(f"Memory saved: {len(chunks)} chunks.")
            except Exception as e:
                logger.error(f"Memory save failed: {e}")

    threading.Thread(target=_write, daemon=True).start()


def clear_memory() -> str:
    """
    [FEAT-06] Delete ONLY conversation memory: every type="memory" vector and
    the contents of chat_history.md. Document chunks are never touched.
    [MEM-01] Semantics: waits (≤ MEMORY_CLEAR_TIMEOUT) for a save or memory
    re-index that is already running, then removes it too. Answers to
    questions asked before the clear are never saved afterwards; questions
    asked after it are saved normally. Returns a status string for the UI.
    """
    global _memory_epoch
    if not _memory_lock.acquire(timeout=MEMORY_CLEAR_TIMEOUT):
        logger.warning("clear_memory: memory still busy after "
                       f"{MEMORY_CLEAR_TIMEOUT}s; nothing was cleared.")
        return "❌  Memory is busy (an answer is still being saved). Nothing was cleared — please try again."
    try:
        _memory_epoch += 1   # in-flight answers from before this point are dropped
        ids = collection.get(where={"type": "memory"}).get("ids", [])
        if ids:
            collection.delete(ids=ids)
            left = collection.get(where={"type": "memory"}).get("ids", [])
            if left:
                raise RuntimeError(f"{len(left)} memory chunks still present after delete")
        mem_file = Path(VAULT_DIR) / MEMORY_FILE_NAME
        had_file = mem_file.exists() and mem_file.stat().st_size > 0
        if had_file:
            mem_file.write_text("", encoding="utf-8")
        if ids or had_file:
            msg = f"✅  Cleared {len(ids)} memory chunks. Document vault is intact."
        else:
            msg = "ℹ️  No memory chunks to clear."
        logger.info(msg)
        return msg
    except Exception as e:
        logger.error(f"clear_memory failed: {type(e).__name__}: {e}")
        return ("❌  Could not clear memory completely. Documents were not touched. "
                "Please try again; details are in the LifeVault debug log.")
    finally:
        _memory_lock.release()


# ── VAULT STATS  [FEAT-05] ────────────────────────────────────────────────────
def get_vault_stats() -> dict:
    """
    Returns a rich stats dictionary for the UI header / status bar.
    """
    try:
        total = collection.count()
        try:
            mem = collection.get(where={"type": "memory"})
            mem_count = len(mem.get("ids", []))
        except Exception:
            mem_count = 0
        doc_count = total - mem_count
        return {
            "total": total,
            "documents": doc_count,
            "memories": mem_count,
            "status": "online",
        }
    except Exception:
        return {"total": 0, "documents": 0, "memories": 0, "status": "error"}


# ── RETRIEVAL ─────────────────────────────────────────────────────────────────
_INCLUDE = ["documents", "metadatas", "distances", "embeddings"]
_CONTEXT_SEP = "\n\n---\n\n"
_TRUNC_MARKER = "\n... [truncated]"
_TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\s[APM]{2})\]")


def _cosine(a, b) -> float | None:
    """
    [RET-02] Cosine similarity of two vectors, clamped to [-1, 1] (the clamp
    only absorbs float rounding — cosine is bounded by definition).
    Returns None when undefined: missing vector, length mismatch, zero norm,
    or non-finite values. Independent of vector norm and of Chroma's metric.
    """
    try:
        if a is None or b is None or len(a) == 0 or len(a) != len(b):
            return None
        dot = na = nb = 0.0
        for x, y in zip(a, b):
            x, y = float(x), float(y)
            dot += x * y
            na += x * x
            nb += y * y
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (dot, na, nb)) or na == 0.0 or nb == 0.0:
        return None
    return max(-1.0, min(1.0, dot / math.sqrt(na * nb)))


def _norm_text(text: str) -> str:
    """Whitespace- and case-insensitive key used for deduplication."""
    return " ".join(str(text).split()).casefold()


def _memory_timestamp(text: str) -> datetime:
    """Timestamp of a chat_history entry; datetime.min if absent/invalid."""
    m = _TS_RE.search(text or "")
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d %I:%M %p")
        except ValueError:
            pass
    return datetime.min


def _first(res, key):
    v = res.get(key) if res else None
    return v[0] if v is not None and len(v) > 0 else None


def _query_rows(res, q_emb) -> list[dict]:
    """Flatten one Chroma query result into scored chunk dicts."""
    ids, docs = _first(res, "ids"), _first(res, "documents")
    if ids is None or docs is None:
        return []
    metas, dists, embs = _first(res, "metadatas"), _first(res, "distances"), _first(res, "embeddings")
    rows = []
    for i, (cid, doc) in enumerate(zip(ids, docs)):
        if not doc:
            continue
        meta = (metas[i] if metas is not None and i < len(metas) else None) or {}
        try:
            dist = float(dists[i]) if dists is not None and i < len(dists) else None
            dist = dist if dist is not None and math.isfinite(dist) else None
        except (TypeError, ValueError):
            dist = None
        emb = embs[i] if embs is not None and i < len(embs) else None
        rows.append({
            "id": cid,
            "text": doc,
            "source": meta.get("source", "unknown"),
            # [RET-03] keep the chunk's real type (fallback queries are mixed)
            "_type": "memory" if meta.get("type") == "memory" else "document",
            "distance": dist,
            "relevance": _cosine(q_emb, emb),
            "_rank": i,
        })
    return rows


def _filter_active(threshold) -> bool:
    return (isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
            and math.isfinite(threshold) and threshold > 0)


def _select(rows: list[dict], layer: str, k: int, min_relevance) -> list[dict]:
    """
    Rank by relevance (unscored last, then Chroma order), drop duplicate ids
    and duplicate normalised text (keeping the best), apply the threshold
    only to validly scored chunks, and keep at most k.
    """
    active = _filter_active(min_relevance)
    cands = [r for r in rows if r["_type"] == layer]
    cands.sort(key=lambda r: (r["relevance"] is None, -(r["relevance"] or 0.0), r["_rank"]))
    out, seen_ids, seen_text, dropped = [], set(), set(), 0
    for r in cands:
        key = _norm_text(r["text"])
        if r["id"] in seen_ids or key in seen_text:
            continue
        seen_ids.add(r["id"])
        seen_text.add(key)
        if active and r["relevance"] is not None and r["relevance"] < min_relevance:
            dropped += 1
            continue
        out.append({k_: v for k_, v in r.items() if k_ != "_rank"})
        if len(out) >= k:
            break
    if dropped:
        logger.debug(f"{layer}: {dropped} chunks below relevance {min_relevance}.")
    return out


def retrieve(query: str, memory_k: int = MEMORY_K, notes_k: int = NOTES_K) -> list[dict]:
    """
    [FEAT-09, RET-02/03] Dual-layer retrieval.
    Returns documents (most relevant first) followed by memories (most
    relevant first). Each chunk: {"id", "text", "source", "_type",
    "distance" (Chroma squared-L2, may be None), "relevance" (cosine, may be None)}.
    """
    total = collection.count()
    if total == 0:
        logger.warning("Retrieval: database empty.")
        return []

    q_emb = embedder.encode(query).tolist()
    rows: list[dict] = []
    for layer, k in (("document", notes_k), ("memory", memory_k)):
        if k <= 0:
            continue
        n = max(1, min(k * RETRIEVAL_OVERFETCH, total))  # Chroma rejects n_results=0
        try:
            res = collection.query(query_embeddings=[q_emb], n_results=n,
                                   include=_INCLUDE, where={"type": layer})
        except Exception as e:
            logger.debug(f"{layer.capitalize()} layer query failed: {e}")
            if layer != "document":
                continue
            try:  # Fallback: unfiltered; each chunk keeps its own metadata type
                res = collection.query(query_embeddings=[q_emb], n_results=n, include=_INCLUDE)
            except Exception as e2:
                logger.error(f"Fallback retrieval also failed: {e2}")
                continue
        rows.extend(_query_rows(res, q_emb))

    documents = _select(rows, "document", notes_k, MIN_DOC_RELEVANCE)
    memories = _select(rows, "memory", memory_k, MIN_MEMORY_RELEVANCE)
    logger.debug(f"Retrieved {len(documents)} doc chunks + {len(memories)} memories.")
    return documents + memories


# ── CONTEXT BUILDER  [RET-01] ─────────────────────────────────────────────────
def _entry(c: dict) -> str:
    kind = "memory" if c.get("_type") == "memory" else "note"
    rel = c.get("relevance")
    rel_s = (f"relevance {rel:.2f}"
             if isinstance(rel, (int, float)) and math.isfinite(rel) else "relevance n/a")
    return f"[Source: {c.get('source', 'unknown')} · {kind} · {rel_s}]\n{c.get('text', '')}"


def _entries_len(chunks: list[dict]) -> int:
    return sum(len(_entry(c)) for c in chunks) + len(_CONTEXT_SEP) * max(0, len(chunks) - 1)


def _fill(chunks: list[dict], budget: int, min_truncated: int) -> list[tuple[dict, str]]:
    """Greedy in given (relevance) order; skip what can't fit, keep scanning."""
    picked: list[tuple[dict, str]] = []
    used = 0
    for c in chunks:
        entry = _entry(c)
        sep = len(_CONTEXT_SEP) if picked else 0
        room = budget - used - sep
        if room <= 0:
            break
        if len(entry) <= room:
            picked.append((c, entry))
            used += sep + len(entry)
        elif room >= max(min_truncated, len(_TRUNC_MARKER) + 1):
            picked.append((dict(c, truncated=True), entry[: room - len(_TRUNC_MARKER)] + _TRUNC_MARKER))
            break  # budget is now exactly full
        # else: too little room to be useful — try the next (maybe shorter) chunk
    return picked


def build_context(
    chunks: list[dict],
    doc_chars: int = DOC_CONTEXT_CHARS,
    mem_chars: int = MEMORY_CONTEXT_CHARS,
    min_truncated: int = MIN_TRUNCATED_CHARS,
) -> tuple[str, list[dict]]:
    """
    [RET-01] Build a prompt context with separate budgets for document
    evidence and conversation memory.

    * Each source is filled in the given (relevance) order within its own
      reservation, so memories can never consume the document budget.
    * If one source's chunks all fit with room to spare, the spare amount is
      added to the other source's budget (total ≤ doc_chars + mem_chars).
    * Separators count toward the budget; oversize chunks are skipped unless
      ≥ min_truncated chars remain, in which case they are truncated.
    * Every entry carries [Source · kind · relevance]. Selected memories are
      shown oldest → newest so the "newer wins" rule is visible.
    * Empty sections are stated explicitly rather than omitted.

    Returns (context_text, chunks_actually_included).
    """
    doc_chars, mem_chars = max(0, int(doc_chars)), max(0, int(mem_chars))
    docs = [c for c in chunks if c.get("_type") != "memory"]
    mems = [c for c in chunks if c.get("_type") == "memory"]
    doc_budget = doc_chars + max(0, mem_chars - _entries_len(mems))
    mem_budget = mem_chars + max(0, doc_chars - _entries_len(docs))

    doc_pick = _fill(docs, doc_budget, min_truncated)
    mem_pick = _fill(mems, mem_budget, min_truncated)
    mem_pick.sort(key=lambda p: _memory_timestamp(p[0].get("text", "")))

    doc_text = _CONTEXT_SEP.join(e for _, e in doc_pick) or "(No relevant notes found in your files.)"
    mem_text = _CONTEXT_SEP.join(e for _, e in mem_pick) or "(No relevant conversation memories.)"
    context = (f"### Notes from your files\n{doc_text}\n\n"
               f"### Conversation memories (oldest → newest)\n{mem_text}")
    return context, [c for c, _ in doc_pick] + [c for c, _ in mem_pick]


# ── LLM WRAPPER  [FEAT-02, FEAT-04, LLM-01..05] ──────────────────────────────
# Failures are signalled with LLMError (never with a returned string). The UI
# shows `UI_ERROR_PREFIX + user_message`; _safe_history and the export skip
# messages with that prefix, and save_chat_memory refuses to store them.
UI_ERROR_PREFIX = "❌ "
LLM_RETRY_DELAY = 1.0     # seconds between connection retries
HEALTH_TIMEOUT  = 5       # seconds for the health check's model listing

_LLM_USER_MESSAGES = {
    "unavailable": "The local model server (Ollama) isn't reachable. Start it with `ollama serve`, then try again.",
    "model_not_found": f"The model '{OLLAMA_MODEL}' isn't installed in Ollama. Install it with `ollama pull {OLLAMA_MODEL}`, then try again.",
    "timeout": f"The local model didn't respond within {LLM_TIMEOUT} seconds. Please try again.",
    "http": "The local model server reported an error. Check the LifeVault debug log for details.",
    "empty": "The local model returned an empty response. Please try again.",
    "malformed": "The local model returned a response LifeVault couldn't read. Check the LifeVault debug log for details.",
    "unexpected": "Something went wrong while generating the answer. Check the LifeVault debug log for details.",
}
# Kinds where a later step or fallback would almost certainly fail the same way.
FATAL_LLM_ERRORS = frozenset({"unavailable", "model_not_found", "timeout", "http", "unexpected"})


class LLMError(RuntimeError):
    """
    [LLM-01] A generation failed. `kind` ∈ _LLM_USER_MESSAGES; `user_message`
    is safe to show (no backend text, paths, hosts or request data — those go
    to the local debug log only).
    """

    def __init__(self, kind: str):
        self.kind = kind if kind in _LLM_USER_MESSAGES else "unexpected"
        self.user_message = _LLM_USER_MESSAGES[self.kind]
        super().__init__(f"{self.kind}: {self.user_message}")


_llm_clients: dict = {}
_llm_clients_lock = threading.Lock()


def _get_llm_client(timeout: float = LLM_TIMEOUT):
    """
    [LLM-02] One ollama.Client per timeout value. The timeout is enforced by
    httpx (connect and each read/write), so a timed-out request is actually
    abandoned — no background thread keeps generating. Host resolution is
    unchanged (ollama's default / OLLAMA_HOST). Creating a client is offline.
    """
    with _llm_clients_lock:
        client = _llm_clients.get(timeout)
        if client is None:
            client = _llm_clients[timeout] = ollama.Client(timeout=timeout)
        return client


def _classify_llm_exception(exc: BaseException) -> LLMError:
    """
    [LLM-03] Map ollama 0.6.1 / httpx exceptions to an LLMError kind:
      ResponseError 404 or "model … not found" → model_not_found
      other ResponseError (HTTP status, in-stream error) → http
      TimeoutError / httpx *Timeout*             → timeout
      ConnectionError / httpx transport errors / OSError → unavailable
      ValueError (JSON / pydantic validation)    → malformed
      anything else                              → unexpected
    """
    if isinstance(exc, LLMError):
        return exc
    response_error = getattr(ollama, "ResponseError", None)
    if isinstance(response_error, type) and isinstance(exc, response_error):
        status = getattr(exc, "status_code", -1)
        detail = str(getattr(exc, "error", "") or "").lower()
        if status == 404 or ("model" in detail and "not found" in detail):
            return LLMError("model_not_found")
        return LLMError("http")
    names = {c.__name__ for c in type(exc).__mro__}
    if isinstance(exc, TimeoutError) or "TimeoutException" in names:
        return LLMError("timeout")
    if "HTTPStatusError" in names:
        return LLMError("http")
    if isinstance(exc, OSError) or names & {"TransportError", "NetworkError", "ProtocolError"}:
        return LLMError("unavailable")
    if isinstance(exc, ValueError) or "DecodingError" in names:
        return LLMError("malformed")
    return LLMError("unexpected")


def _log_llm_failure(err: LLMError, exc: BaseException, attempt: int) -> None:
    """Local diagnostics only. Never logs the prompt."""
    detail = f"{type(exc).__name__}: {str(exc)[:300]}"
    if err.kind == "unexpected" and not isinstance(exc, LLMError):
        logger.error(f"LLM call failed unexpectedly (attempt {attempt}): {detail}", exc_info=exc)
    else:
        logger.warning(f"LLM call failed [{err.kind}] (attempt {attempt}): {detail}")


def _message_content(response) -> str:
    """Text of an ollama chat response/chunk (pydantic model or dict)."""
    try:
        content = response["message"]["content"]
    except (KeyError, TypeError, IndexError, AttributeError):
        raise LLMError("malformed") from None
    if content is None:          # ollama Message.content is Optional
        return ""
    if not isinstance(content, str):
        raise LLMError("malformed")
    return content


def ask_llm(prompt: str, timeout: int = LLM_TIMEOUT) -> str:
    """
    [FEAT-02/03/04, LLM-01..04] Blocking chat completion.

    Returns the stripped, NON-EMPTY model text. On any failure raises
    LLMError — it never returns an error string. Only successful answers are
    cached (key = model + prompt). Connection failures are retried up to
    MAX_RETRIES times, LLM_RETRY_DELAY apart (generation is read-only, so a
    retry repeats nothing); timeouts, HTTP / model-not-found errors, empty
    and malformed responses are not retried.
    """
    cache_key = hashlib.md5(f"{OLLAMA_MODEL}\x00{prompt}".encode()).hexdigest()
    cached = _query_cache.get(cache_key)
    if cached:
        logger.debug("Cache hit — skipping LLM call.")
        return cached

    for attempt in range(1, MAX_RETRIES + 2):  # 1 + MAX_RETRIES total attempts
        t0 = time.time()
        try:
            logger.info(f"LLM call (attempt {attempt}/{MAX_RETRIES + 1}) → {OLLAMA_MODEL}")
            response = _get_llm_client(timeout).chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            text = _message_content(response).strip()
            if not text:
                raise LLMError("empty")
        except Exception as e:
            err = _classify_llm_exception(e)
            _log_llm_failure(err, e, attempt)
            if err.kind == "unavailable" and attempt <= MAX_RETRIES:
                time.sleep(LLM_RETRY_DELAY)
                continue
            raise err from None
        logger.info(f"LLM responded in {time.time() - t0:.2f}s")
        _query_cache[cache_key] = text
        return text
    raise LLMError("unexpected")  # unreachable; keeps the contract explicit


def ask_llm_stream(prompt: str, timeout: int = LLM_TIMEOUT):
    """
    [FEAT-07, LLM-01..03] Streaming chat completion; yields text tokens.

    Raises LLMError on failure — possibly after some tokens were yielded
    (mid-stream failure) — and LLMError("empty") if the stream produces no
    non-whitespace text. A connection failure before the first token is
    retried like ask_llm; once tokens have been yielded nothing is retried.
    Never cached. The httpx timeout bounds connecting and each wait for the
    next chunk (not the total generation time).
    """
    produced = False
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            for chunk in _get_llm_client(timeout).chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            ):
                token = _message_content(chunk)
                if token:
                    produced = produced or bool(token.strip())
                    yield token
        except Exception as e:
            err = _classify_llm_exception(e)
            _log_llm_failure(err, e, attempt)
            if err.kind == "unavailable" and not produced and attempt <= MAX_RETRIES:
                time.sleep(LLM_RETRY_DELAY)
                continue
            raise err from None
        break
    if not produced:
        logger.warning("Streaming LLM returned no text.")
        raise LLMError("empty")


def _error_answer(err: LLMError, partial: str = "") -> str:
    """[LLM-04] User-facing error text. Always starts with UI_ERROR_PREFIX so
    history, export and memory recognise it; a partial answer is labelled."""
    msg = f"{UI_ERROR_PREFIX}{err.user_message}"
    if partial.strip():
        msg += f"\n\n(The answer was interrupted — partial text below, not saved to memory.)\n\n{partial}"
    return msg


# ── QUERY PARSING  [BUG-06, RET-05] ───────────────────────────────────────────
_SUBQ_BULLET = re.compile(r"^[-•*]\s+(.*)$")
# A list number such as "1." / "2)" / "**3.**" followed by whitespace or end.
_SUBQ_MARKER = re.compile(r"[*_]{0,2}(\d{1,2})[.)][*_]{0,2}(?=\s|$)")
# After "?": optional ";"/",", whitespace, then a capital, digit or opening quote.
_SUBQ_BOUNDARY = re.compile(r"[;,]?\s+(?=[A-Z0-9\"“])")
MIN_SUBQ_CHARS = 8              # including the trailing "?"


def _question_key(q: str) -> str:
    return _norm_text(q).rstrip("?").strip()


def _split_numbered(line: str) -> list[str] | None:
    """
    [RET-07] Items of a line that holds a numbered list ("1. a 2. b 3) c",
    also after a lead-in like "Sub-questions:"). A number counts as a marker
    only at the start of the line or after sentence punctuation (. ? ! ; : ),
    and only if preceded by whitespace — so "v2.", "2.5" and "chapter 2."
    are not markers. Markers on one line must be consecutive (n, n+1, …).
    Returns None when the line is not such a list.
    """
    marks = []
    for m in _SUBQ_MARKER.finditer(line):
        before = line[:m.start()].rstrip()
        if m.start() and not line[m.start() - 1].isspace():
            continue
        if before and before[-1] not in ".?!;:)":
            continue
        marks.append(m)
    nums = [int(m.group(1)) for m in marks]
    if not marks or (marks[0].start() > 0 and len(marks) < 2) \
            or any(b != a + 1 for a, b in zip(nums, nums[1:])):
        return None
    return [line[m.end():(marks[i + 1].start() if i + 1 < len(marks) else len(line))]
            for i, m in enumerate(marks)]


def _split_questions(line: str) -> list[str]:
    """
    [RET-07] An unmarked line made ONLY of 2+ complete questions ("What is
    X? Which Y? Why Z?"). A boundary is a "?" outside double quotes followed
    by a capital/digit/quote. If any part contains other sentence punctuation
    (prose, "Dr. Smith", "Sure! …") or text follows the last "?", the line is
    ambiguous and yields nothing.
    """
    segs, start, quoted = [], 0, False
    for i, ch in enumerate(line):
        if ch == '"':
            quoted = not quoted
        elif ch in "“”":
            quoted = ch == "“"
        elif ch == "?" and not quoted:
            rest = line[i + 1:]
            b = _SUBQ_BOUNDARY.match(rest)
            if b or not rest.strip():
                segs.append(line[start:i + 1].strip())
                start = i + 1 + (b.end() if b else len(rest))
    if line[start:].strip() or len(segs) < 2 or any(re.search(r"[.!:;]\s", q) for q in segs):
        return []
    return segs


def _parse_sub_questions(text: str, original: str | None = None) -> list[str]:
    """
    [RET-05, RET-07] Extract sub-questions from the decomposition output:
      * numbered ("1." / "1)" / "**1.**") or bulleted ("-", "•", "*") lines;
      * several numbered items on ONE line (space, ";" or punctuation between
        consecutive numbers), e.g. "1. A? 2. B?" or a run-on "1. 2. A. 3. B";
      * an unmarked line consisting only of 2+ complete questions.
    Code-fence lines, headings and prose are ignored; Markdown emphasis is
    stripped; duplicates (case/whitespace-insensitive) and restatements of
    `original` are dropped. A single unmarked question, or a line where
    boundaries are ambiguous, yields nothing.

    Returns [] for empty output, LLM error strings ("❌ …") or when nothing
    usable remains — callers then fall back to a direct answer instead of
    spending LLM calls on junk. Never calls the LLM.
    """
    if not text or not text.strip() or text.lstrip().startswith("❌"):
        return []
    seen = {_question_key(original)} if original else set()
    questions: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or line.startswith(("```", "#")):
            continue
        bullet = _SUBQ_BULLET.match(line)
        if bullet:
            items = _split_numbered(bullet.group(1)) or [bullet.group(1)]
        else:
            items = _split_numbered(line)
            if items is None:
                items = _split_questions(line)
        for item in items:
            # Strip emphasis/quotes, question marks and trailing list punctuation
            # together, so "`Where`?", "**Who?**" and "Compare X." all normalise
            # cleanly, then re-add one "?".
            q = item.strip().strip("*_\"'`? \t.;:,") + "?"
            key = _question_key(q)
            if len(q) >= MIN_SUBQ_CHARS and key and key not in seen:
                seen.add(key)
                questions.append(q)
    return questions


_HISTORY_PLACEHOLDERS = {"Thinking...", "⏳ Thinking…"}


def _content_text(content) -> str:
    """
    [RET-06] Plain text of a chat message's content. Gradio 6.12 delivers
    Chatbot content as a list of {"type": "text", "text": ...} blocks
    (verified); plain strings are also accepted. Non-text parts are ignored.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content["text"]) if "text" in content else ""
    if isinstance(content, (list, tuple)):
        return "\n".join(t for t in (_content_text(p) for p in content) if t)
    return str(content)


def _safe_history(history: list | None, window: int = HISTORY_WINDOW) -> str:
    """
    Format the last N *previous* turns (the caller must not include the
    current question). Skips empty messages, UI placeholders and earlier
    error answers ("❌ …"); each message is capped at 500 chars.
    """
    if not history:
        return ""
    recent: list[tuple[str, str]] = []
    for m in history:
        if not isinstance(m, dict):
            continue
        text = _content_text(m.get("content")).strip()
        if not text or text in _HISTORY_PLACEHOLDERS or text.startswith("❌"):
            continue
        recent.append((str(m.get("role") or "user").capitalize(), text))
    recent = recent[-(window * 2):]  # N pairs
    return "\n".join(f"{role}: {text[:500]}" for role, text in recent)


# ── DEEP REASONING (decomposition-based RAG)  ────────────────────────────────
_ANSWER_RULES = (
    "RULES:\n"
    "- Memory logs have timestamps [YYYY-MM-DD]. Newer = more current truth.\n"
    "- Clearly separate confirmed facts from inferences.\n"
    "- Cite the source filename in brackets when referencing specific notes.\n"
    "- Each source line shows a relevance score (cosine similarity, -1 to 1); "
    "treat low-relevance items with caution.\n"
    "- If the research findings lack the answer, answer using your pre-trained world knowledge "
    "but you MUST explicitly declare: '(Note: This is from general knowledge, not your vault).'\n\n"
)


def _evidence_key(c: dict) -> str:
    return c.get("id") or _norm_text(c.get("text", ""))


def rlm_query(user_question: str, history: list | None = None) -> Generator:
    """
    Deep Reasoning — decomposition-based RAG. This is NOT a Recursive
    Language Model (no recursion, no REPL, no model-written code); the
    function name is kept for compatibility.

      1. retrieve(question)            → direct evidence for synthesis
      2. decompose (1 LLM call)        → ≤ MAX_SUB_CALLS sub-questions
      3. per sub-question: retrieve + build_context + 1 LLM call
         (no call when no evidence is found → "Not found in notes.")
      4. synthesise from sub-answers + direct evidence (1 LLM call)
    If decomposition is unusable (error / malformed / only restates the
    question) → ONE direct-answer call from the step-1 context instead.

    `history` must contain only previous turns. Yields
    {"answer", "trajectory", "evidence"}; evidence lists exactly the chunks
    placed into prompts, each tagged with "used_in" (direct / Q1… / synthesis).
    [RET-08] One entry per excerpt (chunk id); "uses" records every step that
    sent it, with that step's own relevance and truncation, in step order.
    """
    trajectory: list[tuple[str, str]] = []
    evidence: list[dict] = []
    mem_epoch = _memory_epoch   # [MEM-01] a Clear Memory during generation wins

    def _track(used: list[dict], label: str) -> None:
        for c in used:
            key = _evidence_key(c)
            use = {"step": label, "relevance": c.get("relevance"), "truncated": bool(c.get("truncated"))}
            for e in evidence:
                if _evidence_key(e) == key:
                    if label not in e["used_in"]:
                        e["used_in"].append(label)
                        e["uses"].append(use)
                    break
            else:
                evidence.append(dict(c, used_in=[label], uses=[use]))

    def _state(answer: str) -> dict:
        return {"answer": answer, "trajectory": trajectory, "evidence": list(evidence)}

    def _fail(err: LLMError) -> dict:
        # [LLM-04] Error result: flagged, prefixed, never saved to memory.
        trajectory.append(("⚠️ Model error", err.user_message))
        return dict(_state(_error_answer(err)), error=True)

    # 1. Retrieve
    trajectory.append(("🔍 Retrieving sources", "Searching vault..."))
    yield _state("Retrieving sources...")

    if collection.count() == 0:
        yield _state("⚠️ No documents indexed. Go to **Ingest Vault** first.")
        return

    chunks = retrieve(user_question)
    n_doc = sum(c["_type"] == "document" for c in chunks)
    trajectory.append(("📄 Sources found",
                       f"{n_doc} note chunks + {len(chunks) - n_doc} memories retrieved."))
    yield _state("Analysing question...")

    # 2. Decompose  [BUG-06, RET-05]
    decompose_prompt = (
        f'A user asked: "{user_question}"\n\n'
        f"Break this into at most {MAX_SUB_CALLS} numbered sub-questions "
        f"(format: 1. Question) that together fully answer it. "
        f"If the question involves time or recency, include a sub-question about chronological order. "
        f"Reply with ONLY the numbered list, nothing else."
    )
    trajectory.append(("🧠 Decomposing question", ""))
    yield _state("Decomposing question...")

    try:
        decomp_text = ask_llm(decompose_prompt)
    except LLMError as e:
        if e.kind in FATAL_LLM_ERRORS:   # server down / timeout etc. → stop now
            yield _fail(e)
            return
        trajectory.append(("⚠️ Decomposition failed", e.user_message))
        decomp_text = ""                 # empty / malformed → direct-answer fallback
    sub_questions = _parse_sub_questions(decomp_text, original=user_question)[:MAX_SUB_CALLS]
    history_str = _safe_history(history)

    if not sub_questions:
        # [RET-05] Unusable decomposition → one direct answer, no wasted calls
        trajectory.append(("↪ Decomposition unusable", "Answering directly from retrieved context."))
        ctx, used = build_context(chunks, DOC_CONTEXT_CHARS, MEMORY_CONTEXT_CHARS)
        _track(used, "direct")
        yield _state("Answering directly...")
        try:
            answer = ask_llm(
                f"You are answering a question about a user's personal notes and memories.\n\n"
                f"{_ANSWER_RULES}"
                f"Recent conversation:\n{history_str}\n\n"
                f"Question: {user_question}\n\n"
                f"Retrieved context:\n{ctx}\n\n"
                f"Write a clear, well-structured final answer:"
            )
        except LLMError as e:
            yield _fail(e)
            return
        trajectory.append(("🏁 Done", ""))
        save_chat_memory(user_question, answer, epoch=mem_epoch)
        yield _state(answer)
        return

    trajectory.append(("📋 Sub-questions identified",
                       "\n".join(f"{i+1}. {q}" for i, q in enumerate(sub_questions))))
    yield _state("Researching sub-questions...")

    # 3. Sub-calls  [RET-01: documents always get their own budget]
    sub_answers: list[str] = []
    n_sub = len(sub_questions)
    for i, sq in enumerate(sub_questions):
        trajectory.append((f"💬 Sub-question {i+1}/{n_sub}", sq))
        yield _state(f"Researching {i+1}/{n_sub}...")

        sub_chunks = retrieve(sq, memory_k=3, notes_k=5)
        sub_ctx, used = build_context(sub_chunks, SUB_DOC_CONTEXT_CHARS, SUB_MEMORY_CONTEXT_CHARS)
        if used:
            _track(used, f"Q{i+1}")
            try:
                sub_ans = ask_llm(
                    f"Answer this question using ONLY the notes below. "
                    f"Be concise (2–3 sentences). If not found, say 'Not found in notes.'\n\n"
                    f"Question: {sq}\n\nNotes:\n{sub_ctx}\n\nAnswer:"
                )
            except LLMError as e:
                if e.kind in FATAL_LLM_ERRORS:
                    yield _fail(e)
                    return
                sub_ans = None  # empty / malformed → omitted, never fed to synthesis
        else:
            sub_ans = "Not found in notes."  # no evidence → no LLM call
        if sub_ans is None:
            trajectory.append((f"⚠️ Answer {i+1} unavailable", "No usable text; omitted from synthesis."))
        else:
            sub_answers.append(f"**Q: {sq}**\n{sub_ans}")
            trajectory.append((f"✅ Answer {i+1}", sub_ans[:200]))
        yield _state(f"Completed {i+1}/{n_sub}...")

    # 4. Synthesise  [FEAT-08, BUG-04, RET-04]
    trajectory.append(("✨ Synthesising final answer", ""))
    direct_ctx, direct_used = build_context(chunks, SYNTH_DOC_EVIDENCE_CHARS,
                                            SYNTH_MEMORY_EVIDENCE_CHARS)
    _track(direct_used, "synthesis")
    yield _state("Synthesising...")

    all_sub = "\n\n".join(sub_answers)
    if len(all_sub) > SUB_ANSWERS_CHARS:
        all_sub = all_sub[:SUB_ANSWERS_CHARS] + "\n... [truncated for context]"

    try:
        synthesis = ask_llm(
            f"You are answering a question about a user's personal notes and memories.\n\n"
            f"{_ANSWER_RULES}"
            f"Recent conversation:\n{history_str}\n\n"
            f"Question: {user_question}\n\n"
            f"Research findings (answers to sub-questions):\n{all_sub}\n\n"
            f"Direct evidence for the original question:\n{direct_ctx}\n\n"
            f"Write a clear, well-structured final answer:"
        )
    except LLMError as e:
        yield _fail(e)
        return

    trajectory.append(("🏁 Done", ""))
    save_chat_memory(user_question, synthesis, epoch=mem_epoch)  # runs in background thread

    yield _state(synthesis)


# ── FAST LOOKUP (STANDARD RAG) ────────────────────────────────────────────────
def fast_query(user_question: str, history: list | None = None) -> Generator:
    """
    Single-shot retrieve + answer.
    [FEAT-07] Streams tokens from Ollama for immediate feedback.
    [RET-01] Context uses separate document / memory budgets; `evidence` is
    exactly the chunks placed into the prompt. `history` = previous turns only.
    """
    trajectory: list[tuple[str, str]] = []
    mem_epoch = _memory_epoch   # [MEM-01] a Clear Memory during generation wins

    yield {"answer": "Searching vault...", "trajectory": trajectory, "evidence": []}

    if collection.count() == 0:
        yield {"answer": "⚠️ No documents indexed. Go to **Ingest Vault** first.",
               "trajectory": trajectory, "evidence": []}
        return

    chunks = retrieve(user_question)
    ctx, used = build_context(chunks, DOC_CONTEXT_CHARS, MEMORY_CONTEXT_CHARS)
    n_doc = sum(c["_type"] == "document" for c in used)
    trajectory.append(("📄 Sources found",
                       f"{len(chunks)} chunks retrieved; {n_doc} note + "
                       f"{len(used) - n_doc} memory chunks used."))
    yield {"answer": "Reasoning...", "trajectory": trajectory, "evidence": used}

    history_str = _safe_history(history)

    prompt = (
        f"You are LifeVault, a personal AI assistant.\n"
        f"Respond to the user based on the provided notes and the recent conversation history.\n"
        f"IMPORTANT: Memory logs contain timestamps. Always trust the newer timestamp over older conflicting ones.\n"
        f"If the user is telling you a new fact or updating information (e.g., 'I bought a car', 'my dog is named X'), gently acknowledge it and confirm you will remember it.\n"
        f"If the provided notes and history do not contain the answer to the user's question, clearly state that you do not have that information recorded.\n"
        f"When answering questions based on the notes, be direct and cite filenames in brackets.\n"
        f"Each source line shows a relevance score (cosine similarity, -1 to 1); treat low-relevance items with caution.\n\n"
        f"Recent conversation:\n{history_str}\n\n"
        f"User Input: {user_question}\n\n"
        f"Retrieved context:\n{ctx}\n\nAnswer:"
    )

    # Stream tokens  [LLM-04: failures → flagged error result, nothing saved]
    full_answer = ""
    try:
        for token in ask_llm_stream(prompt):
            full_answer += token
            yield {"answer": full_answer, "trajectory": trajectory, "evidence": used}
    except LLMError as e:
        trajectory.append(("⚠️ Model error", e.user_message))
        yield {"answer": _error_answer(e, partial=full_answer), "trajectory": trajectory,
               "evidence": used, "error": True}
        return

    trajectory.append(("✅ Done", ""))
    save_chat_memory(user_question, full_answer, epoch=mem_epoch)
    yield {"answer": full_answer, "trajectory": trajectory, "evidence": used}
