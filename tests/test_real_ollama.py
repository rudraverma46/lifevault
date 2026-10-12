"""
REAL-MODEL integration tests — opt-in, never part of the default fast suite.

These talk to a real local Ollama server and the real all-MiniLM-L6-v2
embedder. Each test runs the engine in pytest's tmp_path (CWD switched +
guarded) with a REAL temporary ChromaDB and synthetic documents only.

Skipped unless LIFEVAULT_REAL_OLLAMA=1. Also skipped when the server, the
model or the cached embedding model is unavailable (no downloads: the
Hugging Face hub is forced offline).

Run:
  LIFEVAULT_REAL_OLLAMA=1 [LIFEVAULT_TEST_MODEL=qwen2.5:3b] \
      python -m pytest tests/test_real_ollama.py -v
LIFEVAULT_TEST_MODEL defaults to the configured OLLAMA_MODEL. Answers are
stochastic: content assertions check only facts stated verbatim in the notes.
"""

import json
import logging
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.skipif(os.environ.get("LIFEVAULT_REAL_OLLAMA") != "1",
                                reason="real-model tests are opt-in (LIFEVAULT_REAL_OLLAMA=1)")

BOREALIS = ("Project Borealis uses a fictional orange sensor called OR-17. Its maintenance "
            "interval is 23 days. The Borealis field team works from Station Lumen.")
NORTHSTAR = ("The fictional organization Northstar uses a blue sensor called NS-4. Its "
             "maintenance interval is 41 days. Northstar calibrates its sensors every spring.")


def _installed_models() -> set[str]:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    host = host if host.startswith("http") else f"http://{host}"
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as r:
            return {m["model"] for m in json.load(r).get("models", [])}
    except Exception:
        return set()


@pytest.fixture
def lv(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("ollama")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")          # never download weights here
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    monkeypatch.delenv("LIFEVAULT_ALLOWED_ROOTS", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "lifevault", raising=False)
    assert Path.cwd().resolve() == tmp_path.resolve() != ROOT        # guard
    try:
        import lifevault
    except OSError as e:                                  # embedder not cached
        pytest.skip(f"embedding model unavailable offline: {e}")
    assert (tmp_path / ".lifevault_db").exists() and not (ROOT / ".lifevault_db").exists()
    model = os.environ.get("LIFEVAULT_TEST_MODEL", lifevault.OLLAMA_MODEL)
    if model not in _installed_models():
        _release()
        pytest.skip(f"Ollama server or model {model!r} unavailable")
    monkeypatch.setattr(lifevault, "OLLAMA_MODEL", model)
    vault = tmp_path / "my_vault"
    (vault / "projects").mkdir(parents=True)
    (vault / "orgs").mkdir()
    (vault / "projects" / "borealis.md").write_text(BOREALIS + "\n", encoding="utf-8")
    (vault / "orgs" / "northstar.md").write_text(NORTHSTAR + "\n", encoding="utf-8")
    list(lifevault.ingest_vault(str(vault)))
    yield lifevault
    time.sleep(0.5)                                       # let async memory saves finish
    _release()


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


def _final(gen) -> dict:
    out = {}
    for out in gen:
        pass
    return out


def _sources(result) -> list[str]:
    return [e["source"] for e in result.get("evidence", [])]


def test_health_check_reports_exact_model(lv):
    h = lv.startup_health_check()
    assert h["server"] is True and h["model"] is True


def test_ask_llm_returns_real_text(lv):
    text = lv.ask_llm("Reply with exactly the word: ready")
    assert text.strip() and not text.startswith(lv.UI_ERROR_PREFIX)


def test_fast_query_is_grounded_in_the_right_document(lv):
    res = _final(lv.fast_query("What is the name of the sensor used by Project Borealis, "
                               "and what is its maintenance interval?", history=[]))
    assert not res.get("error"), res["answer"]
    assert _sources(res)[0] == "projects/borealis.md"
    assert "OR-17" in res["answer"] and "23" in res["answer"]


def test_deep_query_completes_with_both_documents(lv):
    res = _final(lv.rlm_query("Compare the sensors used by Project Borealis and Northstar: "
                              "name and maintenance interval for each.", history=[]))
    assert not res.get("error"), res["answer"]
    assert {"projects/borealis.md", "orgs/northstar.md"} <= set(_sources(res))
    assert all(e.get("used_in") for e in res["evidence"])
    assert "OR-17" in res["answer"] and "NS-4" in res["answer"]


def test_unsupported_question_invents_no_price(lv):
    res = _final(lv.fast_query("How much does the OR-17 sensor cost?", history=[]))
    assert not res.get("error"), res["answer"]
    assert not re.search(r"[$€£₹]\s?\d|\d+(\.\d+)?\s?(USD|EUR|dollars|euros)", res["answer"], re.I)


def test_missing_model_fails_gracefully_and_saves_nothing(lv, monkeypatch, tmp_path):
    monkeypatch.setattr(lv, "OLLAMA_MODEL", "lifevault-test-missing-model:none")
    res = _final(lv.fast_query("What sensor does Northstar use?", history=[]))
    assert res.get("error") and res["answer"].startswith(lv.UI_ERROR_PREFIX)
    # The message names the CONFIGURED model (built at import), not this override.
    assert "isn't installed" in res["answer"]
    assert "Traceback" not in res["answer"] and str(tmp_path) not in res["answer"]
    time.sleep(0.5)
    assert not (tmp_path / "my_vault" / "chat_history.md").exists()
    assert not lv._query_cache
