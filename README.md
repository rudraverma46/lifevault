# LifeVault
**A privacy-first, locally-hosted Recursive Language Model (RLM) that agentically reasons over your personal knowledge vault.**

LifeVault solves the problem of "context window rot" inherently found in modern massive LLMs. By combining standard Retrieval-Augmented Generation (RAG) with a local agentic reasoning loop inspired by Zhang et al.'s *Recursive Language Models*, LifeVault can query gigabytes of personal PDFs or Markdown notes on consumer hardware without losing tracking accuracy.

> [!TIP]
> Everything runs **100% locally** on your machine. Your personal notes, medical history, legal files, or lecture notes never leave your hard drive.

---

## 🛠 Prerequisites

- **Machine:** A laptop/desktop. (NVIDIA or AMD GPUs will be natively accelerated by Ollama).
- **Environment:** Python 3.10+
- **LLM Engine:** [Ollama](https://ollama.com) installed globally.
- **Disk Space:** ~8 GB free (for the local `qwen2.5:7b` instruction model mapping).

---

## 🚀 Quickstart Installation

### 1. Download Local Inference Models
First, pull the core reasoning model to your machine:
```bash
ollama pull qwen2.5:7b
```
*(You can test if it works by typing `ollama run qwen2.5:7b "Hello"`)*

### 2. Set up the Environment
Clone the repository and install the backend requirements into a local virtual environment:
```bash
python -m venv LIFEVAULT_1
source LIFEVAULT_1/bin/activate
pip install -r requirements.txt
```

---

## 📂 Project Architecture

The workspace is highly modularized for production safety and logging.

```text
trial/
├── lifevault.py          # Core Engine: Handles ChromaDB, Embedding, and recursive LLM loops
├── app.py                # Frontend: Streams generator statuses dynamically using Gradio
├── requirements.txt      # PIP dependencies
├── my_vault/             # YOUR DIRECTORY: Drop .md, .txt, and .pdf files here to be parsed.
├── scripts/              
│   └── start.sh          # Background boot script
├── logs/                 # Output debug directory
│   ├── app.log           # Web-app crash & status log
│   └── lifevault_debug.log # Highly-detailed chunk indexing and Ollama timing log
├── docs/                 # Reference documents (Recursive Language Models paper)
└── .lifevault_db/        # Auto-constructed Vector DB (DO NOT DELETE)
```

---

## 🖥 Usage

### Starting the Server
You do not need to run python manually or occupy your terminal. We have provided a daemon script.

From the root directory, simply run:
```bash
./scripts/start.sh
```
This automatically boots `app.py` in the background and suppresses output to the `logs/` directory.

### Accessing the Web Application
Open your native browser and traverse to:
👉 **[http://localhost:7860](http://localhost:7860)**

---

## 🧠 Reasoning Modes Explained

The user interface exposes three primary tabs for traversing your knowledge base:

### 1. 📁 Ingest Vault (The Foundation)
Whenever you dump new files into `my_vault/`, you must click **Ingest**. 
The backend breaks down your documents using `pdfplumber`, embeds them using `sentence-transformers`, strips corrupted fonts automatically, blocks duplicates, and natively syncs into `ChromaDB`. Progress is streamed live in the UI.

### 2. ⚡ Fast Lookup (Standard RAG)
Optimized purely for speed (~10 to 15 seconds). 
Takes your literal string, fetches the 5 most mathematically similar chunks from the vault, and executes a one-shot summary. Use this for direct questions.

### 3. 🧠 Deep Reasoning (RLM Paradigm)
Optimized for highly analytical, multi-hop problems traversing completely separate documents.
1. The AI reads your prompt and generates smaller sub-questions based on what it lacks.
2. It executes an inner `while` loop, asking itself the subquestions and querying ChromaDB independently each time. 
3. It gathers all the sub-answers into a master synthetic prompt to summarize an overarching theory.
*(Note: Watch the UI **Reasoning Trace** live as it thinks!)*

---

## 🛑 Troubleshooting

> [!WARNING]  
> If the UI is returning a `Connection Refused` state, your background process likely crashed. Read `logs/app.log` for Python tracebacks.

| Issue | Resolution |
| :--- | :--- |
| **"Error communicating with local LLM"** | Ensure the Ollama daemon is actively running in the background. |
| **PDF parsing errors** | Some PDFs are encrypted or corrupted; the engine will bypass them and explicitly name them in `logs/lifevault_debug.log`. |
| **Empty Answer Sets** | Confirm that `my_vault/` exists and that completing Step 1 (Ingest Vault) occurred with >0 indexed chunks. |

### Diagnostic Logging
If Deep Reasoning ever appears stuck or slow, monitor the dedicated logger which tracks explicit HTTP request cycle times to your local LLM down to the millisecond:
```bash
tail -f logs/lifevault_debug.log
```
