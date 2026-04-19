"""
LifeVault — Simplified College Demo
====================================
Stack: Ollama (local LLM) + ChromaDB (vector search) + Gradio (UI)
GPU:   Ollama auto-detects your GPU. No extra config needed.
"""

import os
import re
import time
import hashlib
import logging
import chromadb
import ollama
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── LOGGING SETUP ─────────────────────────────────────────────────────────────
logger = logging.getLogger("LifeVaultBackend")
logger.setLevel(logging.DEBUG)

import os

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

fh = logging.FileHandler("logs/lifevault_debug.log")
fh.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s')
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)
# ─────────────────────────────────────────────────────────────────────────────


# ── CONFIG ────────────────────────────────────────────────────────────────────
VAULT_DIR      = "./my_vault"          # Put your notes/PDFs here
OLLAMA_MODEL   = "qwen2.5:7b"         # Change to "llama3.2:3b" if you prefer
EMBED_MODEL    = "all-MiniLM-L6-v2"   # Fast, runs on CPU even with GPU laptop
CHUNK_SIZE     = 600                   # characters per chunk
CHUNK_OVERLAP  = 100
TOP_K_CHUNKS   = 12                    # chunks retrieved per query
MAX_SUB_CALLS  = 4                     # recursive sub-questions (the RLM idea)
# ─────────────────────────────────────────────────────────────────────────────


# ── SETUP ────────────────────────────────────────────────────────────────────
embedder   = SentenceTransformer(EMBED_MODEL)
chroma     = chromadb.PersistentClient(path="./.lifevault_db")
collection = chroma.get_or_create_collection("vault")


# ── STEP 1: INGEST YOUR FILES ─────────────────────────────────────────────────

def read_file(path: Path) -> str:
    """Read .md, .txt, or .pdf files into plain text."""
    suffix = path.suffix.lower()
    logger.debug(f"Reading file: {path.name}")
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except ImportError:
            logger.warning("pdfplumber not installed; skipping PDF.")
            return f"[PDF skipped — run: pip install pdfplumber]"
        except Exception as e:
            logger.error(f"Error parsing PDF {path.name}: {e}")
            return f"[PDF skipped — Error parsing {path.name}: {e}]"
    return ""


def chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:          # skip tiny fragments
            chunks.append({
                "text":   chunk,
                "source": source,
                "id":     hashlib.md5(f"{source}:{start}".encode()).hexdigest(),
            })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def ingest_vault(vault_dir: str = VAULT_DIR):
    """Scan vault folder, embed all chunks, store in ChromaDB."""
    logger.info(f"Starting ingestion process on vault: {vault_dir}")
    folder = Path(vault_dir)
    if not folder.exists():
        logger.warning(f"Vault folder missing, creating empty vault at {vault_dir}")
        folder.mkdir(parents=True)
        yield f"Created empty vault at {vault_dir}. Add .md, .txt, or .pdf files, then click Ingest again.", 0.0
        return

    files = list(folder.rglob("*.md")) + list(folder.rglob("*.txt")) + list(folder.rglob("*.pdf"))
    if not files:
        logger.info(f"No files available in vault folder {vault_dir}")
        yield f"No files found in {vault_dir}. Add some notes and try again.", 0.0
        return

    log_lines = []
    new_chunks = 0
    total_files = len(files)
    logger.info(f"Found {total_files} files to process.")

    yield f"Found {total_files} files to process...", 0.0

    for i, file in enumerate(files):
        text = read_file(file)
        if not text.strip():
            continue
        chunks = chunk_text(text, str(file.name))

        file_new_chunks = 0
        for chunk in chunks:
            # Skip if already indexed
            existing = collection.get(ids=[chunk["id"]])
            if existing["ids"]:
                continue

            embedding = embedder.encode(chunk["text"]).tolist()
            collection.add(
                ids        = [chunk["id"]],
                embeddings = [embedding],
                documents  = [chunk["text"]],
                metadatas  = [{"source": chunk["source"]}],
            )
            file_new_chunks += 1
            new_chunks += 1
            logger.debug(f"ChromaDB stored chunk {chunk['id']}")

        log_lines.append(f"✓ {file.name}  ({len(chunks)} chunks, {file_new_chunks} newly indexed)")
        logger.info(f"Processed file {file.name}: {file_new_chunks} new chunks.")
        progress_frac = (i + 1) / total_files
        yield "\n".join(log_lines), progress_frac

    summary = f"Indexed {new_chunks} new chunks from {total_files} files.\n\n"
    logger.info(f"Ingestion complete: Added {new_chunks} overall chunks.")
    yield summary + "\n".join(log_lines), 1.0


# ── STEP 2: RETRIEVAL ─────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K_CHUNKS) -> list[dict]:
    """Find the most relevant chunks for a query using semantic search."""
    logger.debug(f"Retrieving top {top_k} chunks for query: '{query}'")
    if collection.count() == 0:
        logger.warning("Retrieval blocked: No chunks found in database.")
        return []
    embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings = [embedding],
        n_results        = min(top_k, collection.count()),
        include          = ["documents", "metadatas"],
    )
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "source": meta.get("source", "unknown")})
    logger.debug(f"Retrieved {len(chunks)} active chunks.")
    return chunks


def ask_llm(prompt: str) -> str:
    """Send a prompt to Ollama and return the response."""
    logger.info(f"Issuing Ollama chat request to model: {OLLAMA_MODEL}")
    t0 = time.time()
    try:
        response = ollama.chat(
            model    = OLLAMA_MODEL,
            messages = [{"role": "user", "content": prompt}],
        )
        duration = time.time() - t0
        logger.info(f"Ollama response received in {duration:.2f}s")
        return response["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Error communicating with local LLM: {e}")
        return f"Error communicating with local LLM ({OLLAMA_MODEL}): {e}"


# ── STEP 3: RECURSIVE REASONING (the RLM idea, simplified) ───────────────────

def rlm_query(user_question: str) -> dict:
    """
    Simplified RLM:
      1. Retrieve relevant chunks from the vault.
      2. Ask the LLM to decompose the question into sub-questions.
      3. Answer each sub-question against the chunks (recursive sub-calls).
      4. Synthesize a final answer from all sub-answers.

    This mirrors the core idea of Zhang et al. — the LLM doesn't see all chunks
    at once; it decomposes the problem and queries them recursively.
    """
    trajectory = []   # We'll show this in the UI as the "reasoning trace"

    # ── 1. Retrieve chunks ───────────────────────────────────────────────────
    trajectory.append(("🔍 Retrieving sources", "Searching vault for relevant chunks..."))
    yield {"answer": "Retrieving sources...", "trajectory": trajectory, "evidence": []}
    
    chunks = retrieve(user_question)
    if not chunks:
        yield {
            "answer":     "No documents indexed yet. Please ingest your vault first.",
            "trajectory": trajectory,
            "evidence":   [],
        }
        return

    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    trajectory.append((
        "📄 Sources found",
        f"Retrieved {len(chunks)} chunks from your vault."
    ))
    yield {"answer": "Sources retrieved, analyzing question...", "trajectory": trajectory, "evidence": chunks[:6]}

    # ── 2. Decompose the question ────────────────────────────────────────────
    decompose_prompt = f"""You are a reasoning assistant. A user has asked:
"{user_question}"

Break this question into at most {MAX_SUB_CALLS} specific sub-questions that,
when answered, will together answer the full question.
Reply with ONLY a numbered list. No extra text.
Example:
1. What did the user write about X?
2. How did the user describe Y?"""

    trajectory.append(("🧠 Decomposing question", "Breaking into sub-questions..."))
    yield {"answer": "Decomposing question...", "trajectory": trajectory, "evidence": chunks[:6]}
    
    decomposition = ask_llm(decompose_prompt)
    trajectory.append(("📋 Sub-questions", decomposition))
    yield {"answer": "Analyzing sub-questions...", "trajectory": trajectory, "evidence": chunks[:6]}

    # Parse sub-questions from the numbered list
    sub_questions = re.findall(r"\d+\.\s+(.+)", decomposition)
    if not sub_questions:
        sub_questions = [user_question]   # fallback: treat as single question

    # ── 3. Answer each sub-question (recursive sub-calls) ───────────────────
    sub_answers = []
    for i, sub_q in enumerate(sub_questions[:MAX_SUB_CALLS]):
        trajectory.append((f"💬 Sub-question {i+1}", sub_q))
        yield {"answer": f"Researching sub-question {i+1} of {len(sub_questions[:MAX_SUB_CALLS])}...", "trajectory": trajectory, "evidence": chunks[:6]}

        # Retrieve a fresh set of chunks specifically for this sub-question
        sub_chunks = retrieve(sub_q, top_k=6)
        sub_context = "\n\n".join(f"[{c['source']}]: {c['text']}" for c in sub_chunks)

        sub_prompt = f"""Answer this question using ONLY the provided notes. 
Be concise (2-4 sentences). If the notes don't contain the answer, say "Not found in notes."

Question: {sub_q}

Notes:
{sub_context}

Answer:"""
        sub_ans = ask_llm(sub_prompt)
        sub_answers.append(f"**Q: {sub_q}**\n{sub_ans}")
        trajectory.append((f"✅ Answer {i+1}", sub_ans))
        yield {"answer": f"Completed sub-question {i+1}", "trajectory": trajectory, "evidence": chunks[:6]}

    # ── 4. Synthesize final answer ───────────────────────────────────────────
    trajectory.append(("✨ Synthesizing final answer", "Combining all sub-answers..."))
    yield {"answer": "Synthesizing final answer...", "trajectory": trajectory, "evidence": chunks[:6]}
    
    all_sub_answers = "\n\n".join(sub_answers)

    synthesis_prompt = f"""You are answering a user's question about their personal notes.

Original question: {user_question}

Here are answers to related sub-questions found in the notes:
{all_sub_answers}

Write a clear, well-structured final answer that directly addresses the original question.
Cite specific sources in brackets when possible, like [filename.md].
Separate what was found in the notes from any inferences you make."""

    final_answer = ask_llm(synthesis_prompt)
    trajectory.append(("🏁 Final answer ready", ""))

    yield {
        "answer":     final_answer,
        "trajectory": trajectory,
        "evidence":   chunks[:6],           # top 6 chunks shown as evidence
    }


# ── FAST MODE (simple lookup, no recursion) ───────────────────────────────────

def fast_query(user_question: str):
    """For simple lookup questions — no decomposition, just retrieve + answer."""
    logger.info(f"INITIATING FAST QUERY: '{user_question}'")
    trajectory = []
    
    trajectory.append(("🔍 Fast lookup", "Retrieving chunks..."))
    yield {"answer": "Searching the vault...", "trajectory": trajectory, "evidence": []}
    
    chunks = retrieve(user_question, top_k=5)
    if not chunks:
        yield {"answer": "No documents indexed yet.", "trajectory": [], "evidence": []}
        return

    trajectory.append(("📄 Sources found", f"Retrieved {len(chunks)} chunks"))
    yield {"answer": "Reasoning...", "trajectory": trajectory, "evidence": chunks}

    context = "\n\n".join(f"[{c['source']}]: {c['text']}" for c in chunks)
    prompt = f"""Answer the following question using ONLY the provided notes.
Be direct and concise.

Question: {user_question}

Notes:
{context}

Answer:"""
    answer = ask_llm(prompt)
    
    trajectory.append(("✅ Done", ""))
    yield {
        "answer":     answer,
        "trajectory": trajectory,
        "evidence":   chunks,
    }
