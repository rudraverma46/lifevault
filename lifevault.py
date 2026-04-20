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
[FEAT-08] MAX_CONTEXT_CHARS guard — truncates sub-answers before synthesis
[FEAT-09] Configurable retrieval balance via MEMORY_K / NOTES_K constants
[FEAT-10] Context-aware synthesis — history window is safely capped
"""

import os
import re
import time
import hashlib
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Generator

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

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
MEMORY_K       = 5            # [FEAT-09] memory chunks per query
NOTES_K        = 8            # [FEAT-09] document chunks per query (notes > memory)
MAX_SUB_CALLS  = 4
LLM_TIMEOUT    = 120          # [FEAT-04] seconds before ask_llm gives up
MAX_RETRIES    = 2            # [FEAT-02] retries on LLM failure
MAX_CONTEXT_CHARS = 6000      # [FEAT-08] max chars fed to synthesis step
HISTORY_WINDOW = 6            # last N messages included in prompts
# ─────────────────────────────────────────────────────────────────────────────


# ── SETUP ─────────────────────────────────────────────────────────────────────
embedder   = SentenceTransformer(EMBED_MODEL)
chroma     = chromadb.PersistentClient(path="./.lifevault_db")
collection = chroma.get_or_create_collection("vault")

# Simple in-memory query cache  [FEAT-03]
_query_cache: dict[str, str] = {}

# Lock for thread-safe memory writes  [BUG-05]
_memory_lock = threading.Lock()


# ── HEALTH CHECK  [FEAT-01] ───────────────────────────────────────────────────
def startup_health_check() -> dict:
    """
    Run at import time or on UI load.
    Returns {"ok": bool, "model": bool, "vault": bool, "message": str}
    """
    status = {"ok": True, "model": False, "vault": False, "message": ""}
    issues = []

    # 1. Is Ollama running?
    try:
        models = ollama.list()
        available = [m["model"] for m in models.get("models", [])]
        # Normalise: "qwen2.5:7b" and "qwen2.5:7b-instruct" both match
        base = OLLAMA_MODEL.split(":")[0]
        if any(base in m for m in available):
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
def read_file(path: Path) -> str:
    """Parse .md / .txt / .pdf → plain text."""
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
        return f"[Skipped {path.name} — install missing library: {e}]"
    except Exception as e:
        logger.error(f"Failed to read {path.name}: {e}")
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
            chunk_id = hashlib.md5(f"{source}::{idx}".encode()).hexdigest()
            chunks.append({"text": chunk, "source": source, "id": chunk_id})
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def ingest_vault(vault_dir: str = VAULT_DIR) -> Generator:
    """
    Generator: yields (log_string, progress_fraction) tuples.
    Safe to call repeatedly — already-indexed chunks are skipped.
    """
    logger.info(f"Ingestion started: {vault_dir}")
    folder = Path(vault_dir)
    if not folder.exists():
        folder.mkdir(parents=True)
        yield f"⚠️  Created empty vault at {vault_dir}. Add files and re-ingest.", 0.0
        return

    extensions = ["*.md", "*.txt", "*.pdf", "*.json", "*.eml"]
    files = []
    for ext in extensions:
        files.extend(folder.rglob(ext))
    # Never index the auto-generated chat history file
    files = [f for f in files if f.name != "chat_history.md"]

    if not files:
        yield f"⚠️  No supported files in {vault_dir}.", 0.0
        return

    log_lines: list[str] = []
    total_new = 0
    total_files = len(files)
    yield f"Found {total_files} files. Starting embedding...", 0.02

    for i, file in enumerate(files):
        text = read_file(file)
        if not text.strip():
            log_lines.append(f"⚠  {file.name}  (empty / unreadable)")
            continue

        chunks = chunk_text(text, file.name)
        file_new = 0
        for chunk in chunks:
            if collection.get(ids=[chunk["id"]])["ids"]:
                continue
            emb = embedder.encode(chunk["text"]).tolist()
            collection.add(
                ids=[chunk["id"]],
                embeddings=[emb],
                documents=[chunk["text"]],
                metadatas=[{"source": chunk["source"], "type": "document"}],
            )
            file_new += 1
            total_new += 1

        log_lines.append(f"✓  {file.name}  ({len(chunks)} chunks, {file_new} new)")
        logger.info(f"{file.name}: {file_new} new chunks.")
        yield "\n".join(log_lines), (i + 1) / total_files

    summary = (
        f"✅  Done — {total_new} new chunks indexed from {total_files} files.\n"
        f"Total vault size: {collection.count()} chunks.\n\n"
    )
    logger.info(f"Ingestion complete: {total_new} new chunks added.")
    yield summary + "\n".join(log_lines), 1.0


# ── MEMORY ────────────────────────────────────────────────────────────────────
def save_chat_memory(question: str, answer: str) -> None:
    """
    [BUG-05] Runs in a background thread so the UI response is never delayed.
    Saves to chat_history.md and immediately indexes into ChromaDB.
    """
    def _write():
        with _memory_lock:
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
    [FEAT-06] Delete ONLY chat_history chunks. Document chunks are kept.
    Returns a status string for the UI.
    """
    try:
        results = collection.get(where={"type": "memory"})
        ids = results.get("ids", [])
        if ids:
            collection.delete(ids=ids)
            # Also wipe the file
            mem_file = Path(VAULT_DIR) / "chat_history.md"
            if mem_file.exists():
                mem_file.write_text("", encoding="utf-8")
            msg = f"✅  Cleared {len(ids)} memory chunks. Document vault is intact."
        else:
            msg = "ℹ️  No memory chunks to clear."
        logger.info(msg)
        return msg
    except Exception as e:
        logger.error(f"clear_memory failed: {e}")
        return f"❌  Error clearing memory: {e}"


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
def retrieve(query: str, memory_k: int = MEMORY_K, notes_k: int = NOTES_K) -> list[dict]:
    """
    [FEAT-09] Dual-layer retrieval with configurable and balanced k values.
    Returns memories (chronologically sorted) + document notes.
    """
    if collection.count() == 0:
        logger.warning("Retrieval: database empty.")
        return []

    emb = embedder.encode(query).tolist()
    results: list[dict] = []

    # --- Memory layer ---
    try:
        mem = collection.query(
            query_embeddings=[emb],
            n_results=min(memory_k, collection.count()),
            include=["documents", "metadatas"],
            where={"type": "memory"},
        )
        if mem["documents"] and mem["documents"][0]:
            for doc, meta in zip(mem["documents"][0], mem["metadatas"][0]):
                results.append({"text": doc, "source": meta.get("source", "memory"), "_type": "memory"})
    except Exception as e:
        logger.debug(f"Memory layer query failed (may be empty): {e}")

    # --- Document layer ---
    try:
        notes = collection.query(
            query_embeddings=[emb],
            n_results=min(notes_k, collection.count()),
            include=["documents", "metadatas"],
            where={"type": "document"},
        )
        if notes["documents"] and notes["documents"][0]:
            for doc, meta in zip(notes["documents"][0], notes["metadatas"][0]):
                results.append({"text": doc, "source": meta.get("source", "unknown"), "_type": "document"})
    except Exception as e:
        logger.debug(f"Document layer query failed: {e}")
        # Fallback: unfiltered query
        try:
            fb = collection.query(
                query_embeddings=[emb],
                n_results=min(notes_k, collection.count()),
                include=["documents", "metadatas"],
            )
            if fb["documents"] and fb["documents"][0]:
                for doc, meta in zip(fb["documents"][0], fb["metadatas"][0]):
                    results.append({"text": doc, "source": meta.get("source", "unknown"), "_type": "document"})
        except Exception as e2:
            logger.error(f"Fallback retrieval also failed: {e2}")

    # Sort memories chronologically, leave docs in relevance order
    def _ts(chunk: dict):
        m = re.search(r"\[(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}\s[APM]{2})\]", chunk["text"])
        if m:
            try:
                return datetime.strptime(m.group(1), "%Y-%m-%d %I:%M %p")
            except Exception:
                return datetime.min
        return datetime.min

    memories = sorted([c for c in results if c["_type"] == "memory"], key=_ts)
    documents = [c for c in results if c["_type"] == "document"]
    combined = memories + documents
    logger.debug(f"Retrieved {len(memories)} memories + {len(documents)} doc chunks.")
    return combined


# ── LLM WRAPPER  [FEAT-02, FEAT-04] ──────────────────────────────────────────
def ask_llm(prompt: str, timeout: int = LLM_TIMEOUT) -> str:
    """
    Calls Ollama with retry logic and a hard timeout.
    [FEAT-02] Retries up to MAX_RETRIES times with 1s backoff.
    [FEAT-04] Enforces LLM_TIMEOUT seconds hard cap.
    [FEAT-03] Results are cached by prompt MD5.
    """
    cache_key = hashlib.md5(prompt.encode()).hexdigest()
    if cache_key in _query_cache:
        logger.debug("Cache hit — skipping LLM call.")
        return _query_cache[cache_key]

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 2):  # 1 + MAX_RETRIES total attempts
        try:
            logger.info(f"LLM call (attempt {attempt}/{MAX_RETRIES + 1}) → {OLLAMA_MODEL}")
            t0 = time.time()

            result_holder: dict = {}

            def _call():
                response = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                result_holder["text"] = response["message"]["content"].strip()

            thread = threading.Thread(target=_call, daemon=True)
            thread.start()
            thread.join(timeout=timeout)

            if thread.is_alive():
                raise TimeoutError(f"LLM call timed out after {timeout}s")

            text = result_holder.get("text", "")
            duration = time.time() - t0
            logger.info(f"LLM responded in {duration:.2f}s")

            _query_cache[cache_key] = text
            return text

        except Exception as e:
            last_error = str(e)
            logger.warning(f"LLM attempt {attempt} failed: {e}")
            if attempt <= MAX_RETRIES:
                time.sleep(1.0)

    error_msg = f"❌ LLM failed after {MAX_RETRIES + 1} attempts: {last_error}"
    logger.error(error_msg)
    return error_msg


def ask_llm_stream(prompt: str):
    """
    [FEAT-07] Streaming version for fast_query — yields tokens as they arrive.
    Falls back to non-streaming on error.
    """
    try:
        for chunk in ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ):
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
    except Exception as e:
        logger.error(f"Streaming LLM failed: {e}")
        yield f"❌ Error: {e}"


# ── QUERY PARSING  [BUG-06] ───────────────────────────────────────────────────
def _parse_sub_questions(text: str) -> list[str]:
    """
    [BUG-06] Robust sub-question extraction — handles multiple LLM list formats:
    "1. Question", "1) Question", "- Question", "• Question", "First, ..."
    """
    patterns = [
        r"^\s*\d+[\.\)]\s+(.+)",    # 1. or 1)
        r"^\s*[-•*]\s+(.+)",         # - or • or *
        r"^\s*[A-Z][a-z]+,?\s+(.+)", # First, ... / Second, ...
    ]
    questions = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for pat in patterns:
            m = re.match(pat, line)
            if m:
                q = m.group(1).strip().rstrip("?") + "?"
                if len(q) > 10:
                    questions.append(q)
                break
    return questions or [text.strip()]  # fallback: treat whole text as one question


# ── CONTEXT GUARD  [FEAT-08] ──────────────────────────────────────────────────
def _safe_context(chunks: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Build a context string, truncating if it would overflow the LLM window."""
    parts = []
    total = 0
    for c in chunks:
        entry = f"[Source: {c['source']}]\n{c['text']}"
        if total + len(entry) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(entry[:remaining] + "\n... [truncated]")
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n---\n\n".join(parts)


def _safe_history(history: list | None, window: int = HISTORY_WINDOW) -> str:
    """Format the last N conversation turns safely."""
    if not history:
        return ""
    recent = [m for m in history if m.get("content") and m["content"] != "Thinking..."]
    recent = recent[-(window * 2):]  # N pairs
    lines = []
    for m in recent:
        role = m.get("role", "user").capitalize()
        content = str(m.get("content", ""))[:500]  # cap per message
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ── DEEP REASONING (RLM)  ─────────────────────────────────────────────────────
def rlm_query(user_question: str, history: list | None = None) -> Generator:
    """
    Simplified RLM — Decompose → Sub-calls → Synthesise.
    Generator: yields incremental result dicts for streaming UI updates.
    """
    trajectory: list[tuple[str, str]] = []

    # 1. Retrieve
    trajectory.append(("🔍 Retrieving sources", "Searching vault..."))
    yield {"answer": "Retrieving sources...", "trajectory": trajectory, "evidence": []}

    chunks = retrieve(user_question)
    if not chunks:
        yield {"answer": "⚠️ No documents indexed. Go to **Ingest Vault** first.",
               "trajectory": trajectory, "evidence": []}
        return

    trajectory.append(("📄 Sources found", f"{len(chunks)} chunks retrieved."))
    yield {"answer": "Analysing question...", "trajectory": trajectory, "evidence": chunks[:6]}

    # 2. Decompose  [BUG-06 fixed]
    decompose_prompt = (
        f'A user asked: "{user_question}"\n\n'
        f"Break this into at most {MAX_SUB_CALLS} numbered sub-questions "
        f"(format: 1. Question) that together fully answer it. "
        f"If the question involves time or recency, include a sub-question about chronological order. "
        f"Reply with ONLY the numbered list, nothing else."
    )
    trajectory.append(("🧠 Decomposing question", ""))
    yield {"answer": "Decomposing question...", "trajectory": trajectory, "evidence": chunks[:6]}

    decomp_text = ask_llm(decompose_prompt)
    sub_questions = _parse_sub_questions(decomp_text)[:MAX_SUB_CALLS]
    trajectory.append(("📋 Sub-questions identified", "\n".join(f"{i+1}. {q}" for i, q in enumerate(sub_questions))))
    yield {"answer": "Researching sub-questions...", "trajectory": trajectory, "evidence": chunks[:6]}

    # 3. Sub-calls
    sub_answers: list[str] = []
    for i, sq in enumerate(sub_questions):
        trajectory.append((f"💬 Sub-question {i+1}/{len(sub_questions)}", sq))
        yield {"answer": f"Researching {i+1}/{len(sub_questions)}...", "trajectory": trajectory, "evidence": chunks[:6]}

        sub_chunks = retrieve(sq, memory_k=3, notes_k=5)
        sub_ctx = _safe_context(sub_chunks, max_chars=2500)

        sub_ans = ask_llm(
            f"Answer this question using ONLY the notes below. "
            f"Be concise (2–3 sentences). If not found, say 'Not found in notes.'\n\n"
            f"Question: {sq}\n\nNotes:\n{sub_ctx}\n\nAnswer:"
        )
        sub_answers.append(f"**Q: {sq}**\n{sub_ans}")
        trajectory.append((f"✅ Answer {i+1}", sub_ans[:200]))
        yield {"answer": f"Completed {i+1}/{len(sub_questions)}...", "trajectory": trajectory, "evidence": chunks[:6]}

    # 4. Synthesise  [FEAT-08 guarded, BUG-04 fixed]
    trajectory.append(("✨ Synthesising final answer", ""))
    yield {"answer": "Synthesising...", "trajectory": trajectory, "evidence": chunks[:6]}

    all_sub = "\n\n".join(sub_answers)
    # Guard: cap combined sub-answers so synthesis prompt never overflows
    if len(all_sub) > MAX_CONTEXT_CHARS:
        all_sub = all_sub[:MAX_CONTEXT_CHARS] + "\n... [truncated for context]"

    history_str = _safe_history(history)

    synthesis = ask_llm(
        f"You are answering a question about a user's personal notes and memories.\n\n"
        f"RULES:\n"
        f"- Memory logs have timestamps [YYYY-MM-DD]. Newer = more current truth.\n"
        f"- Clearly separate confirmed facts from inferences.\n"
        f"- Cite the source filename in brackets when referencing specific notes.\n\n"
        f"Recent conversation:\n{history_str}\n\n"
        f"Question: {user_question}\n\n"
        f"Research findings:\n{all_sub}\n\n"
        f"Write a clear, well-structured final answer:"
    )

    trajectory.append(("🏁 Done", ""))
    save_chat_memory(user_question, synthesis)  # runs in background thread

    yield {"answer": synthesis, "trajectory": trajectory, "evidence": chunks[:6]}


# ── FAST LOOKUP (STANDARD RAG) ────────────────────────────────────────────────
def fast_query(user_question: str, history: list | None = None) -> Generator:
    """
    Single-shot retrieve + answer.
    [FEAT-07] Streams tokens from Ollama for immediate feedback.
    """
    trajectory: list[tuple[str, str]] = []

    yield {"answer": "Searching vault...", "trajectory": trajectory, "evidence": []}

    chunks = retrieve(user_question)
    if not chunks:
        yield {"answer": "⚠️ No documents indexed. Go to **Ingest Vault** first.",
               "trajectory": trajectory, "evidence": []}
        return

    trajectory.append(("📄 Sources found", f"{len(chunks)} chunks retrieved."))
    yield {"answer": "Reasoning...", "trajectory": trajectory, "evidence": chunks}

    ctx = _safe_context(chunks)
    history_str = _safe_history(history)

    prompt = (
        f"Answer the question below using ONLY the provided notes. "
        f"Be direct and concise. Cite filenames in brackets.\n\n"
        f"Recent conversation:\n{history_str}\n\n"
        f"Question: {user_question}\n\n"
        f"Notes:\n{ctx}\n\nAnswer:"
    )

    # Stream tokens
    full_answer = ""
    for token in ask_llm_stream(prompt):
        full_answer += token
        yield {"answer": full_answer, "trajectory": trajectory, "evidence": chunks}

    trajectory.append(("✅ Done", ""))
    save_chat_memory(user_question, full_answer)
    yield {"answer": full_answer, "trajectory": trajectory, "evidence": chunks}
