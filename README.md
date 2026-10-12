# LifeVault v2.0
**A privacy-first, locally-hosted retrieval-augmented assistant that answers questions over your personal knowledge vault, with an optional question-decomposition ("Deep Reasoning") mode.**

LifeVault solves the problem of "context window rot" inherently found in modern massive LLMs. It combines standard Retrieval-Augmented Generation (RAG) with a question-decomposition mode. That mode is *inspired by* Zhang et al.'s *Recursive Language Models*, but it is **not** an RLM: it has no recursion and no code-executing REPL. It only sends retrieved, size-budgeted excerpts to the local model, so it can work with large collections of PDFs and Markdown notes on consumer hardware. No accuracy or scale benchmarks have been run.

In **v2.0**, LifeVault ships with full conversational memory, a reactive backend, robust background process management, and strict context limit guards to prevent context degradation.

> [!TIP]
> Everything runs **100% locally** on your machine. Your personal notes, medical history, legal files, or lecture notes never leave your hard drive.

---

## 🛠 Prerequisites

- **Machine:** A laptop/desktop. (NVIDIA or AMD GPUs will be natively accelerated by Ollama).
- **Environment:** Python 3.10+ (the floor set by the pinned packages; tested on 3.14.6)
- **LLM Engine:** [Ollama](https://ollama.com), installed system-wide or user-local (see below).
- **Disk Space:** ~8 GB free (for the local `qwen2.5:7b` instruction model mapping).

---

## 🚀 Quickstart Installation

### 1. Download Local Inference Models
First, pull the core reasoning model to your machine:
```bash
ollama pull qwen2.5:7b
```
*(You can test if it works by typing `ollama run qwen2.5:7b "Hello"`)*

The tag must match `OLLAMA_MODEL` in `lifevault.py` (default `qwen2.5:7b`) exactly; `qwen2.5:3b` does not satisfy `qwen2.5:7b`. There is no UI or environment setting for the model. Check with `ollama list`: the `NAME` column must show `qwen2.5:7b` itself.

**User-local Ollama (no sudo, no system service).** If Ollama was unpacked from the official Linux tarball into `~/.local/ollama`, start it yourself in a separate terminal, bound to this machine only, and pull with the same binary:
```bash
OLLAMA_HOST=127.0.0.1:11434 ~/.local/ollama/bin/ollama serve      # leave running
OLLAMA_HOST=127.0.0.1:11434 ~/.local/ollama/bin/ollama pull qwen2.5:7b
```
`scripts/start.sh` never starts or stops Ollama.

### 2. Set up the Environment
Clone the repository and install the backend requirements into a local virtual environment:
```bash
python -m venv LIFEVAULT_1
source LIFEVAULT_1/bin/activate
# optional: CPU-only PyTorch (~200 MB instead of the multi-GB CUDA build; only the
# small embedding model uses it — Ollama does its own GPU acceleration)
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
```
Check the environment (expect the five pins from `requirements.txt` and "No broken requirements found"):
```bash
LIFEVAULT_1/bin/python --version                     # pins need Python >= 3.10
LIFEVAULT_1/bin/python -m pip check
LIFEVAULT_1/bin/python -m pip freeze | grep -iE '^(chromadb|sentence-transformers|ollama|gradio|pdfplumber)=='
LIFEVAULT_1/bin/python -c "import torch; print(torch.__version__, 'CUDA build' if torch.version.cuda else 'CPU build')"
```
The name `LIFEVAULT_1` (in the project root, git-ignored) is required: `scripts/start.sh` runs `./LIFEVAULT_1/bin/python app.py`. On first start, `all-MiniLM-L6-v2` (~90 MB) is downloaded into `~/.cache/huggingface`. This setup (Python 3.14.6, the pins in `requirements.txt`, CPU torch, Ollama 0.40.3 user-local) was verified on 2026-10-12: `LIFEVAULT_1` was created with exactly these commands, `pip check` was clean, and `scripts/start.sh` start/status/restart/stop plus a smoke test with synthetic notes ran in the repository checkout. Only Python 3.14.6 has been tested.

### 3. Start
With Ollama running (step 1): `./scripts/start.sh start`, then open the printed URL (default `http://127.0.0.1:7860`). See **Server Management** below.

---

## 📂 Project Architecture

The workspace is highly modularized for production safety and logging.

```text
trial/
├── lifevault.py          # Core Engine: Handles ChromaDB, Embedding, caching, and recursive LLM loops (v2.0)
├── app.py                # Frontend: Streams generator statuses dynamically using Gradio (v2.0)
├── security.py           # Host/port/login settings + ingestion-path validation (stdlib only)
├── requirements.txt      # Pinned PIP dependencies for reproducibility
├── tests/
│   ├── test_security.py          # Isolated security tests (temp dirs, fakes — safe to run)
│   ├── test_retrieval.py         # Isolated retrieval tests (fakes + optional temp ChromaDB — safe to run)
│   ├── test_reliability.py       # Isolated export / telemetry / Ollama-failure tests (mocked Ollama — safe to run)
│   ├── test_index_consistency.py # Index update/delete/legacy/failure tests on a temporary ChromaDB
│   ├── test_memory_concurrency.py # Clear Memory vs. save/index ordering (temporary ChromaDB, no model)
│   ├── test_real_ollama.py       # Opt-in tests against a real local Ollama model (skipped by default)
│   └── test_memory_retention.py  # Manual script — MODIFIES your real memory/DB, see warning below
├── my_vault/             # YOUR DIRECTORY: Drop .md, .txt, and .pdf files here to be parsed.
├── scripts/              
│   └── start.sh          # Background boot script and process manager
├── logs/                 
│   ├── app.log           # Web-app crash & status log
│   ├── lifevault.pid     # Process identifier for the background service
│   └── lifevault_debug.log # Highly-detailed chunk indexing and Ollama timing log
├── docs/                 # Reference documents
└── .lifevault_db/        # Auto-constructed Vector DB (DO NOT DELETE)
```

---

## 🖥 Server Management (New in v2.0)

You do not need to run Python manually or keep a terminal open. We provide a robust daemon script `start.sh` to control the application cluster in the background.

From the root directory of the project, you have 4 commands:

```bash
# Safely start the server in the background
./scripts/start.sh start

# Check if the server is running and what its Process ID and port are
./scripts/start.sh status

# Kill the server safely
./scripts/start.sh stop

# Hot-restart the server (useful after updating code)
./scripts/start.sh restart
```

### Accessing the Web Application
Once running, open your native browser and traverse to:
👉 **[http://127.0.0.1:7860](http://127.0.0.1:7860)** (or the host/port you configured below)

---

## 🔒 Security & Network Configuration

By default LifeVault listens on **`127.0.0.1` only**, so it is reachable solely from the machine it runs on. Earlier versions listened on all interfaces (`0.0.0.0`) with no login.

All settings come from **environment variables**. LifeVault does not read a `.env` file. Never commit credentials; `.env` is git-ignored as a precaution.

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `LIFEVAULT_HOST` | `127.0.0.1` | Bind address. `localhost`, `127.x.x.x` and `::1` count as local. Anything else, including `0.0.0.0`, is treated as network-exposed. |
| `LIFEVAULT_PORT` | `7860` | TCP port, `1`–`65535`. |
| `LIFEVAULT_AUTH_USER` | *(unset)* | Login username. |
| `LIFEVAULT_AUTH_PASSWORD` | *(unset)* | Login password. |
| `LIFEVAULT_ALLOWED_ROOTS` | *(unset)* | Extra **absolute** folders that may be ingested, separated by `:` (Linux/macOS). |

**Rules enforced at startup** (`app.py` exits with code 2 and an explanatory message otherwise):
- The username and password must be set **together, or not at all**.
- A non-local `LIFEVAULT_HOST` **requires** both credentials. LifeVault never falls back to unauthenticated network access.
- Invalid hosts, such as `0.0.0.0:7860`, `http://…`, `[::1]` or `127.1`, and invalid ports are rejected.
- Detailed error messages are shown in the UI only when bound to localhost. Gradio analytics are disabled.

**Enabling access from other devices (example):**
```bash
export LIFEVAULT_HOST=0.0.0.0
export LIFEVAULT_AUTH_USER=me
read -rs LIFEVAULT_AUTH_PASSWORD && export LIFEVAULT_AUTH_PASSWORD   # typed, not stored in shell history
./scripts/start.sh restart
```

> [!WARNING]
> Gradio's login **does not encrypt traffic**. Over plain HTTP, the password and your vault contents travel in clear text. On any network you don't fully trust, put LifeVault behind HTTPS (e.g. a reverse proxy with TLS) or reach it through a secure tunnel such as SSH port forwarding (`ssh -L 7860:127.0.0.1:7860 yourhost`) or a VPN, and keep `LIFEVAULT_HOST` on localhost.

### Which folders can be ingested
- Only folders inside the resolved `./my_vault` directory, or inside a `LIFEVAULT_ALLOWED_ROOTS` entry, can be ingested. Paths are fully resolved (`~`, `..`, symlinks) before checking, and containment is checked per path component. For example, `my_vault2` is **not** inside `my_vault`.
- Missing folders are rejected, not created. The one exception is the default `./my_vault`, which is still auto-created.
- `LIFEVAULT_ALLOWED_ROOTS` entries must be absolute, existing directories, and not `/`. If any entry is malformed, **all** ingestion is refused until it is fixed.
- **Symlinks:** a file or folder inside the vault that links *outside* the folder being ingested is **skipped**, and a warning is written to `logs/lifevault_debug.log`. Links that stay inside the folder work normally. If your vault itself is a symlink, its target becomes the permitted root.

### Running the isolated tests
These tests use only temporary directories, in-memory fakes and scripted LLM replies. They never touch your vault, `chat_history.md`, production ChromaDB or Ollama, and they never start the app or kill a process.
```bash
python -m venv /tmp/lv-test && /tmp/lv-test/bin/pip install pytest
# optional: gradio==6.12.0 (UI tests) and chromadb==1.5.8 (real-ChromaDB tests, temp dir only)
/tmp/lv-test/bin/python -m pytest tests/test_security.py tests/test_retrieval.py tests/test_reliability.py \
    tests/test_index_consistency.py tests/test_memory_concurrency.py tests/test_real_ollama.py -v
```
UI tests are skipped automatically if Gradio isn't installed. Real-ChromaDB tests are skipped if `chromadb` isn't installed. In every file except `tests/test_real_ollama.py`, Ollama is **always mocked**: no test talks to a model, and none of these are end-to-end tests of the running app. `tests/test_index_consistency.py` runs ingestion against a real ChromaDB created in a temporary folder, including injected storage, verification and cleanup failures.

#### Optional: real-model tests
`tests/test_real_ollama.py` talks to a **real** local Ollama server and the real `all-MiniLM-L6-v2` embedder, in a temporary folder with synthetic notes. It is skipped unless you opt in, and also skipped if the server, the model or the cached embedding model is missing (it never downloads anything):
```bash
LIFEVAULT_REAL_OLLAMA=1 python -m pytest tests/test_real_ollama.py -v                      # configured model
LIFEVAULT_REAL_OLLAMA=1 LIFEVAULT_TEST_MODEL=qwen2.5:3b python -m pytest tests/test_real_ollama.py -v
```
Model answers vary between runs; these tests only check facts stated word-for-word in the notes.

> [!CAUTION]
> `tests/test_memory_retention.py` is **not** isolated. It writes to your real `chat_history.md`, deletes all memory vectors, and re-ingests your vault. Back up `my_vault/` and `.lifevault_db/` before running it.

---

## 🧠 Reasoning Modes & Memory

The v2.0 engine has been tuned with automatic chat history tracking and optimized LLM parsing:

### 1. 📁 Ingest Vault (The Foundation)
Whenever you dump new files into `my_vault/`, navigate to the **Ingest Vault** tab and press Ingest. 
The backend breaks down your documents using `pdfplumber`, embeds them using `sentence-transformers`, strips corrupted fonts automatically, hashes content by position, and natively syncs into `ChromaDB`. Progress is streamed live in the UI.

### 2. 💬 Chat Mode: Fast Lookup 
Designed for speed. It streams tokens live; the UI estimates 5–15 s, but this has not been measured.
It retrieves the most similar document chunks and conversation memories, builds a size-budgeted context (see **Retrieval settings**), and makes a single LLM call. Use this for direct questions.

### 3. 💬 Chat Mode: Deep Reasoning (decomposition-based RAG)
Meant for multi-part questions that span several documents. It is **not** a Recursive Language Model.
1. It retrieves evidence for your whole question.
2. One LLM call splits the question into at most 4 sub-questions. Numbered or bulleted lines count, as do several numbered items on one line (`1. … 2. …`) and a line made only of complete questions (`What …? Which …?`). Surrounding prose and code fences are ignored, duplicates and restatements of your question are dropped, and a line whose boundaries are unclear is not used.
3. For each sub-question it retrieves its own evidence and makes one short LLM call. A sub-question with no evidence is answered "Not found in notes." without an LLM call.
4. A final LLM call combines the sub-answers with a short excerpt of the evidence from step 1.

If the split in step 2 is unusable (an error, prose with no list, a single unnumbered question, unclear boundaries, or just your question repeated), LifeVault answers directly in **one** call instead. It never makes an extra call to repair the split.
*(Watch the **Reasoning Trace** live. The **Evidence** panel lists exactly the excerpts that were sent to the model, showing which step used each one and its relevance score. An excerpt used by several steps is listed once. If the steps scored it differently, or one of them truncated it, each step's own value is shown, e.g. `relevance 0.41–0.72 · used in: Q1 (0.59), Q2 (0.41, truncated), synthesis (0.61)`.)*

### Retrieval settings (`lifevault.py` constants)
| Constant | Default | Meaning |
| :--- | :--- | :--- |
| `DOC_CONTEXT_CHARS` / `MEMORY_CONTEXT_CHARS` | 4000 / 2000 | Characters reserved for document excerpts and for conversation memories. Unused space passes to the other, so memories can no longer push all documents out. |
| `SUB_DOC_CONTEXT_CHARS` / `SUB_MEMORY_CONTEXT_CHARS` | 1700 / 800 | The same reservations, per Deep-mode sub-question. |
| `SYNTH_DOC_EVIDENCE_CHARS` / `SYNTH_MEMORY_EVIDENCE_CHARS` | 1000 / 500 | Evidence from step 1 included in the final Deep-mode call. |
| `MIN_DOC_RELEVANCE` | 0.0 | Documents are **never** filtered by relevance; they are only ranked and labelled. |
| `MIN_MEMORY_RELEVANCE` | 0.25 | Memories scoring below this (cosine similarity, −1 to 1) are left out. **Provisional:** not tuned on real data. Set it to `0` to turn filtering off. |

Relevance is the cosine similarity between your question and each stored embedding, computed directly from the vectors, so it doesn't depend on vector normalisation. Duplicate excerpts (identical text after whitespace and case normalisation) are shown only once.

### Persistent Memory
Every **successful** answer is saved to `my_vault/chat_history.md` and indexed as a memory, so you can ask about past conversations, e.g. "What did we talk about yesterday regarding taxes?" Error messages and empty answers are **never** saved. To wipe memory while keeping your documents, use **Clear Memory Only** in the Ingest tab.

What **Clear Memory Only** guarantees:
- It removes every conversation memory: all memory vectors **and** the contents of `chat_history.md`, including entries whose vector was never stored. Document chunks are never touched.
- If an answer is being saved at that moment, or an ingest is re-indexing `chat_history.md`, the clear waits for it (at most 30 s) and then removes it too.
- An answer to a question asked **before** the clear is never saved afterwards, even if it finishes generating later. You still see the answer; it just isn't remembered. Questions asked after the clear are remembered as usual.
- If memory stays busy for 30 s, nothing is cleared and you are asked to try again. If clearing fails, you get an error; details go only to `logs/lifevault_debug.log`. Retrying is safe.

### Answer quality: known limitations
LifeVault grounds answers in your notes, but the model can still be wrong. Always check the **Evidence** panel.
- In testing with synthetic notes, `qwen2.5:7b` answered stated facts correctly. It still occasionally:
  - added an unsupported claim (e.g. that an organisation that *uses* a sensor *manufactures* it);
  - added a false caveat;
  - appended a labelled "general knowledge" paragraph.
- `qwen2.5:3b` is faster but made more mistakes: a reversed conclusion in Deep Reasoning, and once attributing one document's facts to another.
- Deep Reasoning answers at most 4 sub-questions; any further ones are dropped.
- Saved conversation memories are part of later prompts, so earlier answers (including their formatting and citations) can influence new ones. Use **Clear Memory Only** if that becomes a problem.

### When the local model fails
LifeVault talks to Ollama through a client with a **120-second timeout** (`LLM_TIMEOUT`). A timed-out request is abandoned rather than left running in the background.

| Situation | What you see | Retried? |
| :--- | :--- | :--- |
| Ollama not running / connection refused | ❌ "The local model server (Ollama) isn't reachable…" | Yes, up to 2 more times, 1 s apart (only before any text has streamed) |
| Model not installed | ❌ "The model 'qwen2.5:7b' isn't installed… `ollama pull`…" | No |
| No response within 120 s (or a stall between streamed tokens) | ❌ "The local model didn't respond within 120 seconds…" | No |
| Other server error, unreadable or empty response, unexpected error | ❌ A short message pointing to the debug log | No |

- Error messages never include server addresses, file paths or raw backend text; those go only to `logs/lifevault_debug.log`. Prompts are never logged.
- Failed or empty responses are **never cached**, never saved as memories, and never fed back into later prompts.
- In **Deep Reasoning**, an empty or unreadable sub-answer is left out of the final step. If the server is unreachable or times out, the whole question stops with the error.
- If an answer is cut off mid-stream, the partial text is shown under the error, labelled "not saved to memory".
- **Export ↗** copies the most recent *real* answer as plain text. It skips the "Thinking…" placeholder, progress lines and error messages. Attachments appear as `[Attachment: name]`.

### Privacy note: ChromaDB telemetry
LifeVault creates its database client with `anonymized_telemetry=False`. In the pinned ChromaDB 1.5.8 the telemetry function is already a no-op, so this is a safeguard in case that changes. It does **not** make LifeVault fully offline: the embedding model is still downloaded from Hugging Face on first run if it isn't cached, and Ollama and Gradio make their own network decisions.

### How the index stays consistent
- **Identity.** Each document is identified by its path inside `my_vault`, e.g. `notes/project.md` and `archive/project.md` are different documents. Files reached through an extra allowed root use their absolute path. Source labels in answers and in the Evidence panel show this path.
- **Unchanged files** are skipped. Re-ingesting creates no duplicates.
- **Edited files.** New chunks are prepared, stored and **verified** first. Only then are the file's old chunks removed. If storing or verifying fails, the old chunks stay searchable and the ingest log says so.
- **Files that can't be read** (permission errors, read failures, missing PDF support) keep their previous chunks.
- **Files that were read successfully but are now empty** (or have too little text to index) are a real update: their old chunks are removed, and the removal is confirmed. A failed read is never treated as an empty file.
- **Deleted or renamed files** are cleaned up only when you ingest the **whole vault** (the default `./my_vault` path), and only after a **complete** scan. If any folder or entry can't be read during the scan, for example a permission error or a disconnected drive that reports an error, LifeVault still ingests the files it *can* read. It then **skips all cleanup** and lists the unreadable folders in the summary, so nothing is removed for files that merely *look* missing. Even after a complete scan, a file's chunks are removed only if it is confirmed missing from disk. If the scan finds no documents at all, cleanup is skipped. Ingesting a sub-folder or an extra root never deletes anything.
- **Memories** (`chat_history.md`) are indexed per conversation entry, so re-ingesting it adds no duplicates. Ingestion and cleanup never delete memories; only **Clear Memory Only** does.
- **Problems are reported.** You only see "✅ Done" when every file and cleanup step succeeded. Otherwise you see "⚠️ Finished with problems…", listing the affected files. Only one ingestion runs at a time.

**Chunks from older LifeVault versions are never deleted automatically.** They were stored with only the file *name*, so LifeVault can't prove which file they came from. A chunk with the same name and identical text might have come from a same-named file outside the vault. A whole-vault ingest summary reports how many old-format chunks were kept, and how many of them repeat text that is now also indexed under its current path. Search hides exact duplicates, so you see one copy. Nothing is migrated automatically, and nothing changes at startup. To drop all old-format chunks: stop LifeVault, move `.lifevault_db/` aside (keep it as your backup), start LifeVault again and ingest the whole vault. Memories are re-indexed from `my_vault/chat_history.md`.

**What the ingest summary tells you:** counts of new, updated, unchanged and emptied files; files with problems, which keep their previous chunks; obsolete chunks removed; whether cleanup ran or was skipped, and why; unreadable folders; and kept old-format chunks.

**Limits of scan checking:** LifeVault can only detect folders that *report* an error. A drive that disconnects and leaves an empty, readable folder behind looks exactly like a folder whose files were deleted. Its documents would then be cleaned up on a whole-vault ingest. If you keep part of your vault on removable or network storage, make sure it is connected before ingesting the whole vault.

**Crash safety.** ChromaDB doesn't provide transactions across these steps, so updates are *best effort*:
- A crash after storing new chunks but before removing old ones can briefly leave both. The next ingest removes the old ones.
- A crash while storing leaves only unverified extras of the new version, which the next ingest completes.
- The previous version is never deleted before the new one is verified.

### Backup and recovery
Your data is the folder `my_vault/` (your files and `chat_history.md`) and the index `.lifevault_db/`, both in the project root. The index can be rebuilt by re-ingesting `my_vault/` (and any extra `LIFEVAULT_ALLOWED_ROOTS` folders you ingested); your files cannot be rebuilt from the index, so back up both.

**Back up** (from the project root; each backup gets a new timestamped folder outside the repository, so nothing is overwritten):
```bash
./scripts/start.sh stop && ./scripts/start.sh status      # must say "Not running" — never copy the index while LifeVault runs
D=~/lifevault-backups/$(date +%F_%H%M%S)
mkdir -p "$D" && cp -a my_vault .lifevault_db "$D"/
diff -r my_vault "$D/my_vault" && diff -r .lifevault_db "$D/.lifevault_db" && echo "backup verified: $D"
```
Only trust the backup if it prints `backup verified`; a successful `cp` alone is not proof.

**Restore** a backup. Nothing is moved unless the backup exists; the current folders are then moved aside (to `$A`), never deleted or overwritten:
```bash
./scripts/start.sh stop && ls ~/lifevault-backups/          # pick the backup to restore
B=~/lifevault-backups/YYYY-MM-DD_HHMMSS                    # ← replace with that folder
A=~/lifevault-backups/before-restore-$(date +%F_%H%M%S)
[ -d "$B/my_vault" ] && [ -d "$B/.lifevault_db" ] && mkdir -p "$A" && mv my_vault .lifevault_db "$A"/ \
  && cp -a "$B/my_vault" "$B/.lifevault_db" . \
  && diff -r "$B/my_vault" my_vault && diff -r "$B/.lifevault_db" .lifevault_db && echo "restored from $B"
./scripts/start.sh start
```
To rebuild only the index, see *Chunks from older LifeVault versions* above.

---

## 🛑 Troubleshooting

> [!WARNING]  
> If the UI is returning a `Connection Refused` state, your background process likely crashed. Read `logs/app.log` for Python tracebacks.

| Issue | Resolution |
| :--- | :--- |
| **"🔴 Model offline" in header** | Ensure the Ollama daemon is actively running in the background (`ollama serve`). |
| **"🔴 Model not installed" in header** | Ollama is running but the exact configured tag (e.g. `qwen2.5:7b`) is missing — another size such as `qwen2.5:3b` doesn't count. Run the `ollama pull …` command shown. |
| **UI starts but no Answer Sets** | You haven't imported data! Confirm that `my_vault/` exists and you clicked **Ingest**. |
| **Hanging/Slow Responses** | Use `./scripts/start.sh restart` to flush the app's cache and restart it globally. |
| **"❌ The local model server (Ollama) isn't reachable"** | Start Ollama (`ollama serve`) and ask again. Nothing was saved for the failed question. |
| **"❌ … didn't respond within 120 seconds"** | The model is busy or very slow. Try again, or use ⚡ Fast Lookup. |
| **PDF parsing errors** | Some PDFs are encrypted; the engine will bypass them and explicitly name them in `logs/lifevault_debug.log`. |
| **"LifeVault configuration error" / fails to start** | Read the message in `logs/app.log`: a non-local `LIFEVAULT_HOST` needs both auth variables, and host/port must be valid. |
| **Can't open the UI from another device** | Expected: the default is localhost-only. See **Security & Network Configuration**. |
| **"⛔ Ingestion refused"** | The folder is outside `./my_vault` or doesn't exist. Add its parent to `LIFEVAULT_ALLOWED_ROOTS` and restart. |
| **`start.sh stop` says the PID file doesn't belong to LifeVault** | The PID file was stale or reused. Nothing was killed. If LifeVault really is still running, stop it manually. |

```bash
# Monitor the live debug stream down to the millisecond
tail -f logs/lifevault_debug.log
```
