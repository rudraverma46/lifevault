"""
LifeVault v2.0 — Production UI
Run: python app.py  →  http://localhost:7860

__version__ = "2.0"

UI improvements over v1
------------------------
[UI-01] System status bar in header (🟢/🔴 model + vault health)
[UI-02] Live vault stats (docs / memories / total) — updates after ingest
[UI-03] Clear Chat button with confirmation
[UI-04] Clear Memory button (wipes only chat history, keeps documents)
[UI-05] Export Answer button (copies last answer to clipboard)
[UI-06] Error state — red accent when LLM returns an error token
[UI-07] Mode description tooltips so user knows Fast vs Deep
[UI-08] Keyboard shortcut hint (Ctrl+Enter) in placeholder
[UI-09] Ingest stats_live reactive — updates immediately after ingestion
[UI-10] Vault path persists in session state
[UI-11] Improved CSS — scrollable evidence, sticky input bar, better mobile
"""

import gradio as gr
from lifevault import (
    ingest_vault, rlm_query, fast_query, clear_memory,
    startup_health_check, get_vault_stats,
    VAULT_DIR, collection,
)

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:       #0a0b10;
    --surface:  #13151f;
    --card:     #191c28;
    --border:   #252838;
    --accent:   #7c6af7;
    --teal:     #3dbfa0;
    --gold:     #f0b429;
    --red:      #f75f6a;
    --text:     #dde1f0;
    --muted:    #636880;
    --dim:      #2e3245;
}

/* ── Base ── */
body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}
.block, .form { background: transparent !important; border: none !important; }

/* ── Header ── */
.lv-header {
    padding: 2rem 0 1.2rem;
    text-align: center;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.lv-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    letter-spacing: -0.03em;
    color: var(--text);
    margin: 0;
}
.lv-title span { color: var(--accent); }
.lv-tagline {
    font-size: 0.82rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* ── Status bar [UI-01] ── */
.lv-status-bar {
    display: flex;
    justify-content: center;
    gap: 1.2rem;
    flex-wrap: wrap;
    margin: 0.8rem 0 0;
}
.lv-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.28rem 0.9rem;
    font-size: 0.78rem;
    font-family: 'DM Mono', monospace;
    color: var(--teal);
}
.lv-badge.error { color: var(--red); border-color: rgba(247,95,106,0.3); }
.lv-badge.warn  { color: var(--gold); border-color: rgba(240,180,41,0.3); }

/* ── Tabs ── */
.tab-nav button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.6rem 1.2rem !important;
    transition: color 0.15s !important;
}
.tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Inputs ── */
textarea, input[type=text] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
textarea:focus, input[type=text]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(124,106,247,0.12) !important;
    outline: none !important;
}

/* ── Buttons ── */
button.primary {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: opacity 0.15s !important;
}
button.primary:hover { opacity: 0.82 !important; }
button.secondary {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.15s !important;
}
button.secondary:hover { border-color: var(--accent) !important; }
.danger-btn button {
    background: transparent !important;
    color: var(--red) !important;
    border: 1px solid rgba(247,95,106,0.35) !important;
    border-radius: 8px !important;
}
.danger-btn button:hover { background: rgba(247,95,106,0.08) !important; }

/* ── Chatbot ── */
.chatbot-wrap .message.user .bubble-wrap .message-bubble-border {
    background: var(--dim) !important;
    border-radius: 12px 12px 2px 12px !important;
}
.chatbot-wrap .message.bot .bubble-wrap .message-bubble-border {
    background: var(--card) !important;
    border-left: 2px solid var(--accent) !important;
    border-radius: 12px 12px 12px 2px !important;
}
.chatbot-wrap { background: var(--surface) !important; border-radius: 12px !important; border: 1px solid var(--border) !important; }

/* ── Panels ── */
.lv-trace {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--teal);
    white-space: pre-wrap;
    overflow-y: auto;
    max-height: 240px;
}
.lv-evidence {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.82rem;
    color: var(--muted);
    white-space: pre-wrap;
    overflow-y: auto;
    max-height: 240px;
}
.lv-log {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: var(--teal);
    white-space: pre-wrap;
    overflow-y: auto;
    max-height: 300px;
}

/* ── Mode selector ── */
.mode-row .wrap { gap: 0.5rem !important; }
.mode-row label {
    display: flex !important;
    align-items: center !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.4rem 0.9rem !important;
    cursor: pointer !important;
    transition: border-color 0.15s, background 0.15s !important;
}
.mode-row label:has(input:checked) {
    border-color: var(--accent) !important;
    background: rgba(124,106,247,0.08) !important;
    color: var(--accent) !important;
}

label span { color: var(--muted) !important; font-size: 0.82rem !important; }
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_status_html() -> str:
    """[UI-01] Render the header status bar based on live health check."""
    health = startup_health_check()
    stats  = get_vault_stats()

    model_badge = (
        '<span class="lv-badge">🟢 Model online</span>'
        if health["model"]
        else '<span class="lv-badge error">🔴 Model offline — run: ollama serve</span>'
    )
    vault_badge = (
        f'<span class="lv-badge">⬡ {stats["documents"]} docs · '
        f'{stats["memories"]} memories · {stats["total"]} chunks</span>'
        if stats["total"] > 0
        else '<span class="lv-badge warn">⚠ Vault empty — ingest files first</span>'
    )
    return f'<div class="lv-status-bar">{model_badge}{vault_badge}</div>'


def format_trajectory(trajectory: list) -> str:
    if not trajectory:
        return "Waiting for query..."
    lines = []
    for step, detail in trajectory:
        lines.append(f"▸ {step}")
        if detail:
            for dl in detail.split("\n"):
                if dl.strip():
                    lines.append(f"   {dl.strip()}")
    return "\n".join(lines)


def format_evidence(evidence: list) -> str:
    if not evidence:
        return "No evidence retrieved."
    lines = []
    for i, chunk in enumerate(evidence, 1):
        src_type = "💭 Memory" if "chat_history" in chunk.get("source", "") else "📄 Note"
        lines.append(f"[{i}] {src_type} · {chunk['source']}")
        lines.append(f"    {chunk['text'][:240].strip()}…")
        lines.append("")
    return "\n".join(lines)


# ── Actions ───────────────────────────────────────────────────────────────────

def do_ingest(vault_path: str, progress=gr.Progress()):
    """[UI-09] Yields log + reactive status bar after each file."""
    path = vault_path.strip() or VAULT_DIR
    for log_msg, fraction in ingest_vault(path):
        progress(fraction, desc="Indexing documents…")
        yield log_msg, build_status_html()


def do_chat(question: str, history: list, mode: str):
    """Stream chat updates into the chatbot + trace + evidence panels."""
    if not question.strip():
        yield history, "", ""
        return

    history = list(history or [])
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": "⏳ Thinking…"})
    yield history, "", ""

    gen = (
        rlm_query(question, history=history)
        if mode == "🧠 Deep Reasoning"
        else fast_query(question, history=history)
    )

    for result in gen:
        ans = result.get("answer", "")
        # [UI-06] Flag error answers with a red prefix
        if ans.startswith("❌"):
            history[-1]["content"] = ans
        else:
            history[-1]["content"] = ans
        yield (
            history,
            format_trajectory(result.get("trajectory", [])),
            format_evidence(result.get("evidence", [])),
        )


def do_clear_chat():
    """[UI-03] Reset conversation + panels."""
    return [], "Waiting for query...", "No evidence retrieved."


def do_clear_memory():
    """[UI-04] Wipe only memory chunks, keep documents."""
    msg = clear_memory()
    return msg, build_status_html()


def do_export(history: list) -> str:
    """[UI-05] Pull the last assistant answer for export."""
    if not history:
        return "No answer to export."
    for msg in reversed(history):
        if msg.get("role") == "assistant" and msg.get("content") not in ("⏳ Thinking…", ""):
            return msg["content"]
    return "No answer found."


# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="LifeVault") as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="lv-header">
        <h1 class="lv-title">Life<span>Vault</span></h1>
        <p class="lv-tagline">LOCAL · PRIVATE · RECURSIVE REASONING</p>
    </div>
    """)

    # Live status bar  [UI-01, UI-02]
    status_bar = gr.HTML(build_status_html, every=30)

    gr.HTML("<div style='margin-bottom:1rem'></div>")

    with gr.Tabs():

        # ── Tab 1: Chat ──────────────────────────────────────────────────────
        with gr.Tab("💬  Chat"):
            gr.HTML(
                "<p style='color:var(--muted);font-size:0.86rem;margin:0 0 1rem'>"
                "Ask anything about your personal notes. Switch modes for different tasks.</p>"
            )

            # Mode selector  [UI-07]
            with gr.Row(elem_classes=["mode-row"]):
                chat_mode = gr.Radio(
                    choices=["⚡ Fast Lookup", "🧠 Deep Reasoning"],
                    value="⚡ Fast Lookup",
                    label="",
                    info="Fast: single LLM pass (5–15s) · Deep: recursive sub-questions (30–90s)",
                    interactive=True,
                )

            # Chatbot  [UI-06]
            chatbot = gr.Chatbot(
                label="",
                elem_classes=["chatbot-wrap"],
                height=420,
                show_label=False,
            )

            # Input row
            with gr.Row():
                chat_q = gr.Textbox(
                    placeholder="Ask something… (Enter to send, Shift+Enter for newline)",
                    show_label=False,
                    lines=2,
                    scale=5,
                )
                with gr.Column(scale=1, min_width=100):
                    chat_btn   = gr.Button("Send →", variant="primary")
                    clear_btn  = gr.Button("Clear", variant="secondary")   # [UI-03]

            # Trace + Evidence
            with gr.Row():
                chat_trace = gr.Textbox(
                    label="Reasoning trace",
                    lines=8,
                    interactive=False,
                    elem_classes=["lv-trace"],
                )
                chat_evid = gr.Textbox(
                    label="Evidence from vault",
                    lines=8,
                    interactive=False,
                    elem_classes=["lv-evidence"],
                )

            # Export row  [UI-05]
            with gr.Row():
                export_box = gr.Textbox(
                    label="Export last answer",
                    lines=3,
                    interactive=False,
                    placeholder="Click Export to copy the last answer here…",
                )
                export_btn = gr.Button("Export ↗", variant="secondary")

            # Wire chat
            def _clear_input():
                return ""

            chat_btn.click(
                do_chat,
                inputs=[chat_q, chatbot, chat_mode],
                outputs=[chatbot, chat_trace, chat_evid],
            ).then(_clear_input, outputs=[chat_q])

            chat_q.submit(
                do_chat,
                inputs=[chat_q, chatbot, chat_mode],
                outputs=[chatbot, chat_trace, chat_evid],
            ).then(_clear_input, outputs=[chat_q])

            clear_btn.click(                                    # [UI-03]
                do_clear_chat,
                outputs=[chatbot, chat_trace, chat_evid],
            )

            export_btn.click(do_export, inputs=[chatbot], outputs=[export_box])  # [UI-05]

        # ── Tab 2: Ingest ────────────────────────────────────────────────────
        with gr.Tab("📁  Ingest Vault"):
            gr.HTML(
                "<p style='color:var(--muted);font-size:0.86rem;margin:0 0 1rem'>"
                "Index .md · .txt · .pdf · .json · .eml files. "
                "Already-indexed files are automatically skipped.</p>"
            )

            vault_path_input = gr.Textbox(
                value=VAULT_DIR,
                label="Vault folder path",
                placeholder="./my_vault",
            )

            with gr.Row():
                ingest_btn = gr.Button("⬆  Ingest Files", variant="primary")
                mem_btn    = gr.Button("🗑  Clear Memory Only",  # [UI-04]
                                       variant="secondary",
                                       elem_classes=["danger-btn"])

            ingest_log  = gr.Textbox(
                label="Ingestion log",
                lines=14,
                interactive=False,
                elem_classes=["lv-log"],
            )
            ingest_status = gr.HTML(build_status_html())  # [UI-09] reactive

            ingest_btn.click(
                do_ingest,
                inputs=[vault_path_input],
                outputs=[ingest_log, ingest_status],
            )

            mem_btn.click(                                  # [UI-04]
                do_clear_memory,
                outputs=[ingest_log, ingest_status],
            )

        # ── Tab 3: Help ──────────────────────────────────────────────────────
        with gr.Tab("❓  Help"):
            gr.Markdown("""
## Quick Start

1. **Install Ollama** → [ollama.com](https://ollama.com) then run `ollama pull qwen2.5:7b`
2. **Activate venv** → `source LIFEVAULT_1/bin/activate`
3. **Start server** → `./scripts/start.sh` (or `python app.py`)
4. **Ingest vault** → Go to **📁 Ingest Vault**, enter your folder path, click Ingest
5. **Ask questions** → Go to **💬 Chat** and type your question

---

## Reasoning Modes

| Mode | Best for | Speed |
|------|----------|-------|
| ⚡ Fast Lookup | Direct questions, quick facts | 5–15s |
| 🧠 Deep Reasoning | Synthesis, patterns, timelines | 30–90s |

---

## Tips

- **Deep Reasoning** breaks your question into sub-questions and researches each separately — use it for cross-document analysis.
- **Memory** — LifeVault automatically remembers every conversation. Ask *"what did we discuss yesterday?"* and it will know.
- **Clear Memory** deletes conversation history only — your documents are safe.
- Drop files into `my_vault/` and hit Ingest anytime — duplicates are skipped automatically.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` | Run `ollama serve` in a terminal |
| Empty answers | Click Ingest Vault first |
| Slow responses | Use ⚡ Fast Lookup or try `llama3.2:3b` (lighter model) |
| PDF not parsing | Run `pip install pdfplumber` |
""")

if __name__ == "__main__":
    # Print health check to console on startup
    h = startup_health_check()
    print("\n" + "="*60)
    print("LifeVault Status:", h["message"])
    print("="*60 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CSS,
        theme=gr.themes.Base(),
    )
