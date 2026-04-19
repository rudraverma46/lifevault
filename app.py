"""
LifeVault — Gradio UI
Run with:  python app.py
Then open: http://localhost:7860
"""

import gradio as gr
from lifevault import ingest_vault, rlm_query, fast_query, VAULT_DIR, collection

# ── Custom CSS ────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0e0f14;
    --surface:   #161820;
    --border:    #2a2d3a;
    --accent:    #7c6af7;
    --accent2:   #4fc3a1;
    --text:      #e8e9f0;
    --muted:     #6b6f85;
    --evidence:  #1a1d2e;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* Header */
.lv-header {
    padding: 2.5rem 0 1.5rem;
    text-align: center;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.lv-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    letter-spacing: -0.02em;
    color: var(--text);
    margin: 0;
}
.lv-title span { color: var(--accent); }
.lv-subtitle {
    font-size: 0.9rem;
    color: var(--muted);
    margin-top: 0.4rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
}

/* Vault stats badge */
.lv-stats {
    display: inline-block;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-size: 0.8rem;
    color: var(--accent2);
    font-family: 'DM Mono', monospace;
    margin-top: 0.8rem;
}

/* Tabs */
.tab-nav button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.6rem 1.2rem !important;
}
.tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* Inputs */
textarea, input[type=text] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
}
textarea:focus, input[type=text]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(124,106,247,0.15) !important;
}

/* Buttons */
button.primary {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.4rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}
button.primary:hover { opacity: 0.85 !important; }
button.secondary {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Output panels */
.lv-answer {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    font-size: 0.97rem;
    line-height: 1.7;
    white-space: pre-wrap;
}
.lv-trajectory {
    background: var(--evidence);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent2);
    white-space: pre-wrap;
    max-height: 320px;
    overflow-y: auto;
}
.lv-evidence {
    background: var(--evidence);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: var(--muted);
    white-space: pre-wrap;
    max-height: 280px;
    overflow-y: auto;
}
label { color: var(--muted) !important; font-size: 0.82rem !important; }
.block { background: transparent !important; }
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def vault_stats() -> str:
    try:
        count = collection.count()
        return f"⬡  {count} chunks indexed"
    except Exception:
        return "⬡  0 chunks indexed"


def format_trajectory(trajectory: list) -> str:
    if not trajectory:
        return "No steps recorded."
    lines = []
    for step, detail in trajectory:
        lines.append(f"▸ {step}")
        if detail:
            # Indent detail lines
            for dl in detail.split("\n"):
                if dl.strip():
                    lines.append(f"   {dl.strip()}")
    return "\n".join(lines)


def format_evidence(evidence: list) -> str:
    if not evidence:
        return "No evidence retrieved."
    lines = []
    for i, chunk in enumerate(evidence, 1):
        lines.append(f"[{i}] {chunk['source']}")
        lines.append(f"    {chunk['text'][:220].strip()}...")
        lines.append("")
    return "\n".join(lines)


# ── UI actions ────────────────────────────────────────────────────────────────

def do_ingest(vault_path: str, progress=gr.Progress()):
    msg = ""
    for log_msg, fraction in ingest_vault(vault_path or VAULT_DIR):
        msg = log_msg
        progress(fraction, desc="Ingesting documents...")
        yield msg, vault_stats()


def do_deep_query(question: str):
    if not question.strip():
        yield "Please enter a question.", "", ""
        return
    for result in rlm_query(question):
        yield (
            result["answer"],
            format_trajectory(result["trajectory"]),
            format_evidence(result["evidence"]),
        )


def do_fast_query(question: str):
    if not question.strip():
        yield "Please enter a question.", ""
        return
    for result in fast_query(question):
        yield result["answer"], format_evidence(result["evidence"])


# ── Build UI ──────────────────────────────────────────────────────────────────

SAMPLE_DEEP = [
    "What are recurring themes in my notes?",
    "Summarize how my thinking evolved over time.",
    "What are the most common mistakes I documented?",
    "Compare different approaches I wrote about.",
]
SAMPLE_FAST = [
    "Where did I write about machine learning?",
    "Find my notes on project deadlines.",
    "What did I say about federated learning?",
]

with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="LifeVault") as demo:

    # Header
    gr.HTML("""
    <div class="lv-header">
        <h1 class="lv-title">Life<span>Vault</span></h1>
        <p class="lv-subtitle">LOCAL · PRIVATE · RECURSIVE REASONING</p>
    </div>
    """)

    stats_box = gr.HTML(vault_stats, every=10, elem_classes=["lv-stats"])

    with gr.Tabs():

        # ── Tab 1: Deep Reasoning ─────────────────────────────────────────────
        with gr.Tab("🧠  Deep Reasoning"):
            gr.HTML("<p style='color:#6b6f85;font-size:0.88rem;margin:0 0 1rem'>Decomposes your question into sub-questions and reasons recursively over your vault. Best for synthesis, patterns, and comparisons.</p>")

            with gr.Row():
                deep_q = gr.Textbox(
                    placeholder="e.g. What are recurring themes in my notes on machine learning?",
                    label="Your question",
                    lines=2,
                    scale=4,
                )
                deep_btn = gr.Button("Ask →", variant="primary", scale=1)

            gr.Examples(examples=SAMPLE_DEEP, inputs=deep_q, label="Try these")

            deep_answer = gr.Textbox(label="Answer", lines=8, elem_classes=["lv-answer"])

            with gr.Row():
                deep_trace = gr.Textbox(label="Reasoning trace", lines=10, elem_classes=["lv-trajectory"])
                deep_evid  = gr.Textbox(label="Evidence from vault", lines=10, elem_classes=["lv-evidence"])

            deep_btn.click(do_deep_query, inputs=deep_q,
                           outputs=[deep_answer, deep_trace, deep_evid])
            deep_q.submit(do_deep_query, inputs=deep_q,
                          outputs=[deep_answer, deep_trace, deep_evid])

        # ── Tab 2: Fast Lookup ────────────────────────────────────────────────
        with gr.Tab("⚡  Fast Lookup"):
            gr.HTML("<p style='color:#6b6f85;font-size:0.88rem;margin:0 0 1rem'>Direct retrieval + single LLM call. Best for quick facts and finding specific notes.</p>")

            with gr.Row():
                fast_q = gr.Textbox(
                    placeholder="e.g. Where did I write about deadlines?",
                    label="Your question",
                    lines=2,
                    scale=4,
                )
                fast_btn = gr.Button("Ask →", variant="primary", scale=1)

            gr.Examples(examples=SAMPLE_FAST, inputs=fast_q, label="Try these")

            fast_answer = gr.Textbox(label="Answer", lines=6, elem_classes=["lv-answer"])
            fast_evid   = gr.Textbox(label="Evidence from vault", lines=8, elem_classes=["lv-evidence"])

            fast_btn.click(do_fast_query, inputs=fast_q, outputs=[fast_answer, fast_evid])
            fast_q.submit(do_fast_query, inputs=fast_q, outputs=[fast_answer, fast_evid])

        # ── Tab 3: Ingest ─────────────────────────────────────────────────────
        with gr.Tab("📁  Ingest Vault"):
            gr.HTML("<p style='color:#6b6f85;font-size:0.88rem;margin:0 0 1rem'>Point LifeVault at a folder of .md, .txt, or .pdf files. Already-indexed files are skipped automatically.</p>")

            vault_path = gr.Textbox(
                value=VAULT_DIR,
                label="Vault folder path",
                placeholder="./my_vault",
            )
            ingest_btn  = gr.Button("Ingest Files", variant="primary")
            ingest_log  = gr.Textbox(label="Ingestion log", lines=12, elem_classes=["lv-trajectory"])
            stats_live  = gr.HTML(vault_stats())

            ingest_btn.click(do_ingest, inputs=vault_path, outputs=[ingest_log, stats_live])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
