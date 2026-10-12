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

import copy
import html
import math
import os
import re
import sys

from security import (
    ConfigError, PathValidationError, get_server_settings, resolve_ingest_path,
)

# [SEC-01] Validate host/port/auth BEFORE loading the engine (model + DB), so a
# bad or unsafe configuration fails fast and start.sh's liveness check sees it.
if __name__ == "__main__":
    try:
        SERVER_SETTINGS = get_server_settings()
    except ConfigError as _cfg_err:
        print(f"LifeVault configuration error: {_cfg_err}", file=sys.stderr)
        sys.exit(2)

import gradio as gr
from lifevault import (
    ingest_vault, rlm_query, fast_query, clear_memory,
    startup_health_check, get_vault_stats,
    VAULT_DIR, collection,
)
import lifevault as _engine

# ── CSS ───────────────────────────────────────────────────────────────────────
# [SEC-04] No external font requests: the DM fonts are used only if installed
# locally; otherwise the system-font fallbacks below apply.
CSS = """
:root {
    --font-sans:  'DM Sans', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    --font-mono:  'DM Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;
    --font-serif: 'DM Serif Display', Georgia, 'Times New Roman', serif;
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
    font-family: var(--font-sans) !important;
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
    font-family: var(--font-serif);
    font-size: 2.8rem;
    letter-spacing: -0.03em;
    color: var(--text);
    margin: 0;
}
.lv-title span { color: var(--accent); }
.lv-tagline {
    font-size: 0.82rem;
    color: var(--muted);
    font-family: var(--font-mono);
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
    font-family: var(--font-mono);
    color: var(--teal);
}
.lv-badge.error { color: var(--red); border-color: rgba(247,95,106,0.3); }
.lv-badge.warn  { color: var(--gold); border-color: rgba(240,180,41,0.3); }

/* ── Tabs ── */
.tab-nav button {
    font-family: var(--font-sans) !important;
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
    font-family: var(--font-sans) !important;
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
    font-family: var(--font-sans) !important;
    font-weight: 500 !important;
    transition: opacity 0.15s !important;
}
button.primary:hover { opacity: 0.82 !important; }
button.secondary {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--font-sans) !important;
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
    font-family: var(--font-mono);
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
    font-family: var(--font-mono);
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

    if health["model"]:
        model_badge = '<span class="lv-badge">🟢 Model online</span>'
    elif health.get("server"):   # [LLM-05] server up, configured tag missing
        model_badge = (f'<span class="lv-badge error">🔴 Model not installed — run: '
                       f'ollama pull {html.escape(_engine.OLLAMA_MODEL)}</span>')
    else:
        model_badge = '<span class="lv-badge error">🔴 Model offline — run: ollama serve</span>'
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


def _rel_text(r) -> str | None:
    return f"{r:.2f}" if isinstance(r, (int, float)) and not isinstance(r, bool) and math.isfinite(r) else None


def format_evidence(evidence: list) -> str:
    if not evidence:
        return "No evidence retrieved."
    lines = []
    for i, chunk in enumerate(evidence, 1):
        source = str(chunk.get("source") or "unknown")
        # [RET-04] Prefer the engine's real type; fall back to the old heuristic.
        kind = chunk.get("_type") or ("memory" if "chat_history" in source else "document")
        src_type = "💭 Memory" if kind == "memory" else "📄 Note"
        rel = _rel_text(chunk.get("relevance"))
        rel_s = f" · relevance {rel}" if rel else ""
        used = chunk.get("used_in") if isinstance(chunk.get("used_in"), list) else []
        used_s = f" · used in: {', '.join(str(u) for u in used)}" if used else ""
        trunc_s = " · truncated" if chunk.get("truncated") else ""
        # [RET-08] Deep mode: if steps saw this excerpt with different relevance
        # (or truncation), show each step's own values.
        uses = [u for u in chunk.get("uses") if isinstance(u, dict) and u.get("step")] \
            if isinstance(chunk.get("uses"), list) else []
        step_rels = [_rel_text(u.get("relevance")) for u in uses]
        if uses and (len(set(step_rels)) > 1 or any(u.get("truncated") for u in uses)):
            known = sorted({float(r) for r in step_rels if r})
            if known:
                rel_s = f" · relevance {known[0]:.2f}" + (f"–{known[-1]:.2f}" if known[-1] != known[0] else "")
            parts = []
            for u, r in zip(uses, step_rels):
                extra = ", ".join(x for x in (r, "truncated" if u.get("truncated") else None) if x)
                parts.append(f"{u['step']} ({extra})" if extra else str(u["step"]))
            used_s, trunc_s = f" · used in: {', '.join(parts)}", ""
        lines.append(f"[{i}] {src_type} · {source}{rel_s}{used_s}{trunc_s}")
        lines.append(f"    {str(chunk.get('text') or '')[:240].strip()}…")
        lines.append("")
    return "\n".join(lines)


# ── Actions ───────────────────────────────────────────────────────────────────

def do_ingest(vault_path: str, progress=gr.Progress()):
    """[UI-09] Yields log + reactive status bar after each file."""
    # [SEC-02] Reject paths outside the permitted roots with a readable
    # message (no traceback). ingest_vault re-validates as defence in depth.
    try:
        path = str(resolve_ingest_path(vault_path, default_vault=VAULT_DIR))
    except (PathValidationError, ConfigError) as e:
        yield f"⛔  Ingestion refused: {e}", build_status_html()
        return
    for log_msg, fraction in ingest_vault(path):
        progress(fraction, desc="Indexing documents…")
        yield log_msg, build_status_html()


def do_chat(question: str, history: list, mode: str):
    """Stream chat updates into the chatbot + trace + evidence panels."""
    if not question.strip():
        yield history, "", ""
        return

    # [RET-06] Snapshot PREVIOUS turns before adding the current question and
    # placeholder; the engine gets an independent deep copy, so UI updates
    # during streaming can't leak into its prompts.
    prior_turns = copy.deepcopy(list(history or []))

    history = list(history or [])
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": "⏳ Thinking…"})
    yield history, "", ""

    gen = (
        rlm_query(question, history=prior_turns)
        if mode == "🧠 Deep Reasoning"
        else fast_query(question, history=prior_turns)
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


# [EXP-01] Assistant texts that are UI/engine progress indicators, not answers.
_EXPORT_PLACEHOLDERS = {
    "⏳ Thinking…", "Thinking...", "Searching vault...", "Reasoning...",
    "Retrieving sources...", "Analysing question...", "Decomposing question...",
    "Researching sub-questions...", "Synthesising...", "Answering directly...",
}
_EXPORT_PROGRESS_RE = re.compile(r"^(Researching|Completed) \d+/\d+\.\.\.$")


def _export_text(content) -> str:
    """
    [EXP-01] Readable text for one Chatbot message's content, for export only.
    Gradio 6.12 normalises content to a list of blocks — {"type": "text",
    "text"}, {"type": "file", "file": {...}, "alt_text"}, {"type":
    "component", ...} — while messages appended in-process are plain strings.
    Non-text parts become short labels (never server paths or Python reprs).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(t for t in (_export_text(p) for p in content) if t)
    if isinstance(content, dict):
        kind = content.get("type")
        if "text" in content and kind in (None, "text"):
            return str(content["text"] or "")
        if kind == "file" or "file" in content or "path" in content:
            f = content.get("file") if isinstance(content.get("file"), dict) else content
            name = (content.get("alt_text") or f.get("orig_name")
                    or os.path.basename(str(f.get("path") or "")) or "file")
            return f"[Attachment: {name}]"
        if kind == "component":
            return f"[{content.get('component') or 'component'} content]"
        return "[non-text content]"
    return "[non-text content]"


def _is_exportable_answer(text: str) -> bool:
    """False for placeholders, progress lines and error messages."""
    t = text.strip()
    return bool(t) and t not in _EXPORT_PLACEHOLDERS and not _EXPORT_PROGRESS_RE.match(t) \
        and not t.startswith("❌")


def do_export(history: list) -> str:
    """
    [UI-05, EXP-01] Return the most recent genuine assistant answer as plain
    text. Placeholders, in-progress status lines and error messages are
    skipped (the previous real answer is exported instead). `history` is
    only read, never modified.
    """
    if not history:
        return "No answer to export."
    for msg in reversed(history):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        text = _export_text(msg.get("content")).strip()
        if _is_exportable_answer(text):
            return text
    return "No answer found."


# ── Build UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="LifeVault", analytics_enabled=False) as demo:  # [SEC-05]

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
            gr.Markdown(f"""
## Quick Start

1. **Install Ollama** → [ollama.com](https://ollama.com) then run `ollama pull {_engine.OLLAMA_MODEL}`
2. **Activate venv** → `source LIFEVAULT_1/bin/activate`
3. **Start server** → `./scripts/start.sh` (or `python app.py`)
4. **Ingest vault** → Go to **📁 Ingest Vault**, enter your folder path, click Ingest
5. **Ask questions** → Go to **💬 Chat** and type your question

---

## Model

- LifeVault uses **`{_engine.OLLAMA_MODEL}`** (`OLLAMA_MODEL` in `lifevault.py`). The header shows 🟢 only if that **exact** tag is installed; another size of the same model doesn't count.
- See which models are installed: `ollama list`.
- `qwen2.5:7b` is recommended for normal use. `qwen2.5:3b` is faster and fine for development, but in testing it made more reasoning and grounding mistakes.
- To switch models: change `OLLAMA_MODEL` in `lifevault.py`, install that exact tag with `ollama pull <tag>`, then restart LifeVault (`./scripts/start.sh restart`).

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
| Slow responses | Use ⚡ Fast Lookup. A smaller model such as `qwen2.5:3b` is faster but less reliable — see **Model** above |
| PDF not parsing | Run `pip install pdfplumber` |
""")

if __name__ == "__main__":
    # Print health check to console on startup
    h = startup_health_check()
    print("\n" + "="*60)
    print("LifeVault Status:", h["message"])
    print("="*60 + "\n")

    # [SEC-01] Bind per validated settings (default 127.0.0.1). auth is set
    # whenever credentials are configured and is mandatory off-loopback.
    # Never print SERVER_SETTINGS.auth — it holds the password.
    s = SERVER_SETTINGS
    print(f"Listening on {s.host}:{s.port} · login {'required' if s.auth_enabled else 'disabled (localhost only)'}")
    demo.launch(
        server_name=s.host,
        server_port=s.port,
        auth=s.auth,
        share=False,
        show_error=s.is_local,
        css=CSS,
        theme=gr.themes.Base(),
    )
