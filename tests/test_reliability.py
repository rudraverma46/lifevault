"""
Isolated tests for Phase 3 (Tasks A–D): export formatting, ChromaDB telemetry
configuration, dead-code removal, and Ollama reliability.

Safety guarantees (by construction):
  * lifevault.py is imported only with sentence_transformers and ollama
    replaced by fakes and the CWD switched to pytest's tmp_path (guarded).
    The fake ollama.Client is scripted per test — no Ollama server, no network.
  * Network failures use REAL httpx exception classes and a real pydantic
    ValidationError (both already installed with Gradio), so classification
    is tested against the same types ollama 0.6.1 raises.
  * TestTelemetry uses the REAL chromadb package (if installed) against
    ./.lifevault_db inside tmp_path only.
  * app.py is imported (building the UI) but never launched.

Run:  python -m pytest tests/test_reliability.py -v
"""

import json
import logging
import math
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

httpx = pytest.importorskip("httpx")


# ══════════════════════════════════════════════════════════════════════════════
# Fakes
# ══════════════════════════════════════════════════════════════════════════════
class FakeResponseError(Exception):
    """Same constructor/attributes as ollama 0.6.1 ResponseError."""

    def __init__(self, error: str, status_code: int = -1):
        super().__init__(error)
        self.error = error
        self.status_code = status_code

    def __str__(self):
        return f"{self.error} (status code: {self.status_code})"


class Script:
    """Per-test script for the fake ollama.Client.chat()."""

    def __init__(self):
        self.calls: list[dict] = []
        self.steps: list = []          # each: response, iterable (stream), or Exception
        self.list_result = {"models": [{"model": "qwen2.5:7b"}]}

    def next(self, kwargs):
        self.calls.append(kwargs)
        step = self.steps.pop(0) if self.steps else {"message": {"content": "default answer"}}
        if isinstance(step, BaseException):
            raise step
        if kwargs.get("stream"):
            return _stream(step)
        return step


def _stream(items):
    for it in items:
        if isinstance(it, BaseException):
            raise it
        yield it


def tok(t):
    return {"message": {"role": "assistant", "content": t}}


def _fake_ollama(script: Script):
    o = types.ModuleType("ollama")
    o.ResponseError = FakeResponseError
    o.RequestError = type("RequestError", (Exception,), {})
    o.created_clients = []

    class Client:
        def __init__(self, host=None, **kwargs):
            self.kwargs = kwargs
            o.created_clients.append(self)

        def chat(self, **kwargs):
            return script.next(kwargs)

        def list(self):
            if isinstance(script.list_result, BaseException):
                raise script.list_result
            return script.list_result

    o.Client = Client
    o.list = lambda: script.list_result

    def _module_chat(*a, **k):
        raise AssertionError("module-level ollama.chat must not be used")

    o.chat = _module_chat
    return o


def _fake_st():
    st = types.ModuleType("sentence_transformers")

    class _Vec(list):
        def tolist(self):
            return list(self)

    class SentenceTransformer:
        def __init__(self, *a, **k):
            pass

        def encode(self, text):
            return _Vec([1.0, 0.0, 0.0])

    st.SentenceTransformer = SentenceTransformer
    return st


class _FakeCollection:
    def __init__(self):
        self.items = {}

    def count(self):
        return len(self.items)

    def get(self, ids=None, where=None, **_):
        if ids is not None:
            return {"ids": [i for i in ids if i in self.items]}
        return {"ids": [i for i, v in self.items.items()
                        if not where or all(v["meta"].get(k) == w for k, w in where.items())]}

    def add(self, ids, embeddings, documents, metadatas):
        for i, e, d, m in zip(ids, embeddings, documents, metadatas):
            self.items[i] = {"emb": list(e), "doc": d, "meta": m}

    def query(self, query_embeddings, n_results, include=None, where=None):
        pool = [(i, v) for i, v in self.items.items()
                if not where or all(v["meta"].get(k) == w for k, w in where.items())][:n_results]
        return {"ids": [[i for i, _ in pool]], "documents": [[v["doc"] for _, v in pool]],
                "metadatas": [[v["meta"] for _, v in pool]], "distances": [[0.0 for _ in pool]],
                "embeddings": [[v["emb"] for _, v in pool]]}


def _import_engine(tmp_path, monkeypatch, script, real_chroma=False):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LIFEVAULT_ALLOWED_ROOTS", raising=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", _fake_st())
    monkeypatch.setitem(sys.modules, "ollama", _fake_ollama(script))
    if not real_chroma:
        col = _FakeCollection()
        chromadb = types.ModuleType("chromadb")
        chromadb.PersistentClient = lambda path=None, **_: types.SimpleNamespace(
            get_or_create_collection=lambda name, **__: col)
        monkeypatch.setitem(sys.modules, "chromadb", chromadb)
    monkeypatch.delitem(sys.modules, "lifevault", raising=False)
    assert Path.cwd().resolve() == tmp_path.resolve() != ROOT  # guard
    import lifevault

    # No real retry waits — patch the engine's delay, NOT the global time module.
    monkeypatch.setattr(lifevault, "LLM_RETRY_DELAY", 0.0)
    return lifevault


def _teardown():
    lg = logging.getLogger("LifeVault")
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()
    sys.modules.pop("lifevault", None)


@pytest.fixture
def script():
    return Script()


@pytest.fixture
def lv(tmp_path, monkeypatch, script):
    mod = _import_engine(tmp_path, monkeypatch, script)
    yield mod
    _teardown()


def _validation_error():
    import pydantic

    class M(pydantic.BaseModel):
        x: int

    try:
        M(x="not-an-int")
    except pydantic.ValidationError as e:
        return e
    raise AssertionError("expected ValidationError")


def _req():
    return httpx.Request("POST", "http://127.0.0.1:11434/api/chat")


# ══════════════════════════════════════════════════════════════════════════════
# A. Export
# ══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def app_mod(lv, monkeypatch):
    pytest.importorskip("gradio")
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.delitem(sys.modules, "app", raising=False)
    import app

    yield app
    sys.modules.pop("app", None)


def blocks(*parts):
    return [{"text": p, "type": "text"} for p in parts]


class TestExport:
    @pytest.mark.parametrize("content,expected", [
        ("plain answer", "plain answer"),
        (blocks("hello"), "hello"),
        (blocks("line 1", "line 2"), "line 1\nline 2"),
        ({"text": "dict text", "type": "text"}, "dict text"),
        ([{"type": "file", "file": {"path": "/tmp/gradio/abc/report.pdf", "orig_name": "report.pdf"},
           "alt_text": None}], "[Attachment: report.pdf]"),
        ([{"type": "file", "file": {"path": "/srv/tmp/x/photo.png"}}], "[Attachment: photo.png]"),
        ([{"type": "component", "component": "image", "value": {}, "constructor_args": {}, "props": {}}],
         "[image content]"),
        (None, ""),
        ([], ""),
        (12345, "[non-text content]"),
        ({"weird": True}, "[non-text content]"),
    ])
    def test_export_text_formats(self, app_mod, content, expected):
        out = app_mod._export_text(content)
        assert out == expected
        assert "{'" not in out and "/srv/" not in out and "/tmp/" not in out

    def test_exports_last_genuine_answer_from_gradio_blocks(self, app_mod):
        h = [{"role": "user", "content": blocks("q1")},
             {"role": "assistant", "content": blocks("First answer", "continued")},
             {"role": "user", "content": blocks("q2")},
             {"role": "assistant", "content": blocks("Second answer")}]
        assert app_mod.do_export(h) == "Second answer"

    @pytest.mark.parametrize("last", [
        "⏳ Thinking…", "Thinking...", "Researching 2/4...", "Completed 1/3...", "Synthesising...",
        "❌ The local model server (Ollama) isn't reachable.", "   ", None, blocks("⏳ Thinking…"),
    ])
    def test_placeholders_and_errors_skipped(self, app_mod, last):
        h = [{"role": "assistant", "content": blocks("Real answer")},
             {"role": "user", "content": "q"},
             {"role": "assistant", "content": last}]
        assert app_mod.do_export(h) == "Real answer"

    def test_role_respected_and_history_not_modified(self, app_mod):
        import copy
        h = [{"role": "user", "content": blocks("user text only")},
             "garbage", {"content": "no role"}]
        before = copy.deepcopy(h)
        assert app_mod.do_export(h) == "No answer found."
        assert h == before
        assert app_mod.do_export([]) == "No answer to export."
        assert app_mod.do_export(None) == "No answer to export."

    def test_gradio_round_trip_format(self, app_mod):
        """Use Gradio's own postprocess→preprocess to produce the real format."""
        gr = pytest.importorskip("gradio")
        c = gr.Chatbot()
        data = c.postprocess([{"role": "user", "content": "q"},
                              {"role": "assistant", "content": "Exported **text**"}])
        history = c.preprocess(data)
        assert isinstance(history[1]["content"], list)  # confirms the block format
        assert app_mod.do_export(history) == "Exported **text**"


# ══════════════════════════════════════════════════════════════════════════════
# B. Telemetry (real ChromaDB, temp dir)
# ══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def real_lv(tmp_path, monkeypatch, script):
    pytest.importorskip("chromadb")
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "True")  # explicit setting must still win
    mod = _import_engine(tmp_path, monkeypatch, script, real_chroma=True)
    yield mod
    try:
        from chromadb.api.client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
    _teardown()


class TestTelemetry:
    def test_engine_client_has_telemetry_disabled(self, real_lv, tmp_path):
        assert real_lv.chroma.get_settings().anonymized_telemetry is False
        assert (tmp_path / ".lifevault_db").exists() and not (ROOT / ".lifevault_db").exists()

    def test_engine_creates_exactly_one_client(self):
        src = (ROOT / "lifevault.py").read_text(encoding="utf-8")
        assert src.count("chromadb.PersistentClient(") == 2      # settings branch + stand-in branch
        assert "anonymized_telemetry=False" in src
        for f in ("app.py", "security.py"):
            assert "PersistentClient" not in (ROOT / f).read_text(encoding="utf-8")

    def test_installed_chroma_capture_is_noop(self):
        chromadb = pytest.importorskip("chromadb")
        import inspect
        from chromadb.telemetry.product.posthog import Posthog
        body = inspect.getsource(Posthog.capture)
        assert chromadb.__version__ == "1.5.8"
        assert body.strip().endswith("pass")


# ══════════════════════════════════════════════════════════════════════════════
# C. Dead code
# ══════════════════════════════════════════════════════════════════════════════
class TestDeadCodeRemoved:
    def test_safe_context_gone_everywhere(self, lv):
        assert not hasattr(lv, "_safe_context") and not hasattr(lv, "MAX_CONTEXT_CHARS")
        for f in list(ROOT.glob("*.py")) + list((ROOT / "tests").glob("*.py")):
            if f.name == "test_reliability.py":
                continue
            assert "_safe_context(" not in f.read_text(encoding="utf-8"), f


# ══════════════════════════════════════════════════════════════════════════════
# D. Ollama reliability — ask_llm
# ══════════════════════════════════════════════════════════════════════════════
class TestAskLLM:
    def test_success_strips_and_caches(self, lv, script):
        script.steps = [{"message": {"content": "  hello  "}}]
        assert lv.ask_llm("p") == "hello"
        assert lv.ask_llm("p") == "hello"
        assert len(script.calls) == 1                      # second call served from cache

    def test_uses_timeout_bounded_client_not_module_chat(self, lv, script):
        script.steps = [tok("x")]
        lv.ask_llm("p")
        import ollama
        assert ollama.created_clients and ollama.created_clients[0].kwargs["timeout"] == lv.LLM_TIMEOUT

    @pytest.mark.parametrize("content", ["", "   \n", None])
    def test_empty_response_raises_and_is_not_cached(self, lv, script, content):
        script.steps = [{"message": {"content": content}}, tok("now fine")]
        with pytest.raises(lv.LLMError) as ei:
            lv.ask_llm("p")
        assert ei.value.kind == "empty"
        assert lv._query_cache == {}
        assert lv.ask_llm("p") == "now fine"               # not served from a cached failure
        assert len(script.calls) == 2

    @pytest.mark.parametrize("bad", [
        {"foo": 1}, {"message": None}, {"message": {"content": 5}}, "not a mapping",
    ])
    def test_malformed_payload(self, lv, script, bad):
        script.steps = [bad]
        with pytest.raises(lv.LLMError) as ei:
            lv.ask_llm("p")
        assert ei.value.kind == "malformed" and len(script.calls) == 1

    @pytest.mark.parametrize("exc_factory", [
        lambda: json.JSONDecodeError("Expecting value", "<html>", 0),
        _validation_error,
    ])
    def test_unparseable_response_is_malformed(self, lv, script, exc_factory):
        script.steps = [exc_factory()]
        with pytest.raises(lv.LLMError) as ei:
            lv.ask_llm("p")
        assert ei.value.kind == "malformed"

    @pytest.mark.parametrize("exc_factory", [
        lambda: ConnectionError("Failed to connect to Ollama at http://secret-host:11434"),
        lambda: httpx.ConnectError("[Errno 111] Connection refused", request=_req()),
        lambda: httpx.RemoteProtocolError("peer closed connection", request=_req()),
    ])
    def test_unavailable_is_retried_then_raised(self, lv, script, exc_factory):
        script.steps = [exc_factory() for _ in range(lv.MAX_RETRIES + 1)]
        with pytest.raises(lv.LLMError) as ei:
            lv.ask_llm("p")
        assert ei.value.kind == "unavailable"
        assert len(script.calls) == lv.MAX_RETRIES + 1     # bounded attempts
        assert "secret-host" not in ei.value.user_message and "secret-host" not in str(ei.value)
        assert lv._query_cache == {}

    def test_unavailable_then_recovers(self, lv, script):
        script.steps = [ConnectionError("down"), tok("back")]
        assert lv.ask_llm("p") == "back" and len(script.calls) == 2

    @pytest.mark.parametrize("exc_factory", [
        lambda: httpx.ReadTimeout("timed out", request=_req()),
        lambda: httpx.ConnectTimeout("timed out", request=_req()),
        lambda: TimeoutError("timed out"),
    ])
    def test_timeout_not_retried(self, lv, script, exc_factory):
        script.steps = [exc_factory(), tok("never reached")]
        with pytest.raises(lv.LLMError) as ei:
            lv.ask_llm("p")
        assert ei.value.kind == "timeout" and len(script.calls) == 1
        assert str(lv.LLM_TIMEOUT) in ei.value.user_message

    @pytest.mark.parametrize("err,kind", [
        (FakeResponseError('{"error":"model \'qwen2.5:7b\' not found"}', 404), "model_not_found"),
        (FakeResponseError("model 'qwen2.5:7b' not found, try pulling it first", 400), "model_not_found"),
        (FakeResponseError("internal server error at /var/lib/ollama", 500), "http"),
        (FakeResponseError("in-stream failure"), "http"),
        (httpx.HTTPStatusError("bad", request=_req(), response=httpx.Response(502, request=_req())), "http"),
    ])
    def test_http_errors(self, lv, script, err, kind):
        script.steps = [err]
        with pytest.raises(lv.LLMError) as ei:
            lv.ask_llm("p")
        assert ei.value.kind == kind and len(script.calls) == 1
        assert "/var/lib" not in ei.value.user_message and "status code" not in ei.value.user_message

    def test_unexpected_error_is_logged_with_traceback(self, lv, script, caplog):
        script.steps = [ZeroDivisionError("bug")]
        logger = logging.getLogger("LifeVault")
        logger.addHandler(caplog.handler)
        try:
            with pytest.raises(lv.LLMError) as ei:
                lv.ask_llm("p")
        finally:
            logger.removeHandler(caplog.handler)
        assert ei.value.kind == "unexpected"
        rec = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert rec and rec[0].exc_info is not None          # not silently swallowed

    def test_prompt_never_logged(self, lv, script, caplog):
        script.steps = [ConnectionError("x")] * 3
        logger = logging.getLogger("LifeVault")
        logger.addHandler(caplog.handler)
        try:
            with pytest.raises(lv.LLMError):
                lv.ask_llm("TOP-SECRET-PROMPT-CONTENT")
        finally:
            logger.removeHandler(caplog.handler)
        assert all("TOP-SECRET" not in r.getMessage() for r in caplog.records)

    def test_unknown_kind_maps_to_unexpected(self, lv):
        assert lv.LLMError("bogus").kind == "unexpected"


class TestAskLLMStream:
    def test_success(self, lv, script):
        script.steps = [[tok("Hel"), tok("lo"), {"message": {"content": ""}, "done": True}]]
        assert "".join(lv.ask_llm_stream("p")) == "Hello"
        assert script.calls[0]["stream"] is True

    @pytest.mark.parametrize("items", [[], [tok("")], [tok("  "), tok("\n")],
                                        [{"message": {"content": None}}]])
    def test_empty_stream_raises(self, lv, script, items):
        script.steps = [items]
        with pytest.raises(lv.LLMError) as ei:
            list(lv.ask_llm_stream("p"))
        assert ei.value.kind == "empty"

    def test_mid_stream_failure_after_tokens_not_retried(self, lv, script):
        script.steps = [[tok("partial "), httpx.RemoteProtocolError("dropped", request=_req())],
                        [tok("SHOULD NOT BE USED")]]
        got = []
        with pytest.raises(lv.LLMError) as ei:
            for t in lv.ask_llm_stream("p"):
                got.append(t)
        assert got == ["partial "] and ei.value.kind == "unavailable" and len(script.calls) == 1

    def test_connect_failure_before_first_token_retried(self, lv, script):
        script.steps = [httpx.ConnectError("refused", request=_req()), [tok("ok")]]
        assert "".join(lv.ask_llm_stream("p")) == "ok" and len(script.calls) == 2

    def test_in_stream_response_error_and_malformed_chunk(self, lv, script):
        script.steps = [[tok("a"), FakeResponseError("model crashed")]]
        with pytest.raises(lv.LLMError) as ei:
            list(lv.ask_llm_stream("p"))
        assert ei.value.kind == "http"
        script.steps = [[{"nope": 1}]]
        with pytest.raises(lv.LLMError) as ei:
            list(lv.ask_llm_stream("p"))
        assert ei.value.kind == "malformed"

    def test_stream_timeout(self, lv, script):
        script.steps = [[tok("a"), httpx.ReadTimeout("slow", request=_req())]]
        with pytest.raises(lv.LLMError) as ei:
            list(lv.ask_llm_stream("p"))
        assert ei.value.kind == "timeout"


# ══════════════════════════════════════════════════════════════════════════════
# D. Engine behaviour on failures (caching, memory, history, UI flag)
# ══════════════════════════════════════════════════════════════════════════════
def _seed(lv):
    lv.collection.add(ids=["d1"], embeddings=[[1.0, 0.0, 0.0]],
                      documents=["The cat is called Shadow. " * 5],
                      metadatas=[{"source": "cat.md", "type": "document"}])


@pytest.fixture
def saved(lv, monkeypatch):
    rec = []
    orig = lv.save_chat_memory
    monkeypatch.setattr(lv, "save_chat_memory", lambda q, a, epoch=None: (rec.append((q, a)), orig(q, a, epoch=epoch)))
    return rec


def _mem_file(tmp_path):
    return tmp_path / "my_vault" / "chat_history.md"


class TestEngineFailures:
    def test_fast_error_flagged_not_saved_not_in_history(self, lv, script, saved, tmp_path):
        _seed(lv)
        script.steps = [ConnectionError("down")] * 3
        out = list(lv.fast_query("cat?", history=[]))
        last = out[-1]
        assert last.get("error") is True and last["answer"].startswith(lv.UI_ERROR_PREFIX)
        assert "isn't reachable" in last["answer"] and saved == []
        hist = [{"role": "user", "content": "cat?"}, {"role": "assistant", "content": last["answer"]}]
        assert "reachable" not in lv._safe_history(hist)    # never fed to later prompts

    def test_fast_mid_stream_failure_keeps_partial_but_flags_it(self, lv, script, saved):
        _seed(lv)
        script.steps = [[tok("The cat is "), httpx.ReadTimeout("slow", request=_req())]]
        last = list(lv.fast_query("cat?", history=[]))[-1]
        assert last["error"] is True and last["answer"].startswith(lv.UI_ERROR_PREFIX)
        assert "The cat is" in last["answer"] and "not saved" in last["answer"]
        assert saved == []

    def test_fast_success_still_saved(self, lv, script, saved):
        _seed(lv)
        script.steps = [[tok("Shadow.")]]
        last = list(lv.fast_query("cat?", history=[]))[-1]
        assert last["answer"] == "Shadow." and "error" not in last and saved == [("cat?", "Shadow.")]

    def test_deep_server_down_fails_fast(self, lv, script, saved):
        _seed(lv)
        script.steps = [ConnectionError("down")] * 3
        last = list(lv.rlm_query("cat?", history=[]))[-1]
        assert last["error"] is True and len(script.calls) == lv.MAX_RETRIES + 1 and saved == []

    def test_deep_empty_decomposition_falls_back_to_direct_answer(self, lv, script, saved):
        _seed(lv)
        script.steps = [tok("  "), tok("Direct: Shadow.")]
        last = list(lv.rlm_query("cat?", history=[]))[-1]
        assert last["answer"] == "Direct: Shadow." and "error" not in last
        assert saved == [("cat?", "Direct: Shadow.")]

    def test_deep_empty_sub_answer_omitted_from_synthesis(self, lv, script, saved):
        _seed(lv)
        script.steps = [tok("1. What is the cat called?\n2. Who named the cat?"),
                        tok(""), tok("It is Shadow."), tok("FINAL")]
        last = list(lv.rlm_query("cat?", history=[]))[-1]
        synth = script.calls[-1]["messages"][0]["content"]
        assert last["answer"] == "FINAL"
        assert "What is the cat called" not in synth.split("Research findings")[1].split("Direct evidence")[0]
        assert "It is Shadow." in synth
        assert any("unavailable" in s for s, _ in last["trajectory"])

    def test_deep_timeout_in_sub_call_aborts_without_saving(self, lv, script, saved):
        _seed(lv)
        script.steps = [tok("1. What is the cat called?"), httpx.ReadTimeout("slow", request=_req())]
        last = list(lv.rlm_query("cat?", history=[]))[-1]
        assert last["error"] is True and "didn't respond" in last["answer"] and saved == []
        assert len(script.calls) == 2                       # no synthesis after a fatal error

    def test_deep_synthesis_failure(self, lv, script, saved):
        _seed(lv)
        script.steps = [tok("1. What is the cat called?"), tok("Shadow."),
                        FakeResponseError("boom", 500)]
        last = list(lv.rlm_query("cat?", history=[]))[-1]
        assert last["error"] is True and saved == []

    def test_failed_generation_never_cached(self, lv, script):
        _seed(lv)
        script.steps = [ConnectionError("x")] * 3 + [tok("1. What is the cat called?"),
                                                       tok("Shadow."), tok("FINAL")]
        assert list(lv.rlm_query("cat?", history=[]))[-1]["error"] is True
        assert all(v.strip() for v in lv._query_cache.values())
        assert list(lv.rlm_query("cat?", history=[]))[-1]["answer"] == "FINAL"

    @pytest.mark.parametrize("answer", ["", "   ", None, "❌ The local model server isn't reachable."])
    def test_save_chat_memory_refuses_errors_and_empty(self, lv, tmp_path, answer):
        lv.save_chat_memory("q", answer)
        import time as _t
        _t.sleep(0.2)
        assert not _mem_file(tmp_path).exists()
        assert lv.collection.count() == 0

    def test_save_chat_memory_still_saves_real_answers(self, lv, tmp_path):
        lv.save_chat_memory("q", "a real answer")
        import time as _t
        text = ""
        for _ in range(100):  # background thread: wait for the content, not just the file
            if _mem_file(tmp_path).exists():
                text = _mem_file(tmp_path).read_text(encoding="utf-8")
                if "a real answer" in text:
                    break
            _t.sleep(0.05)
        assert "a real answer" in text


class TestHealthCheck:
    def test_unavailable_server_reported_not_raised(self, lv, script):
        script.list_result = ConnectionError("refused")
        h = lv.startup_health_check()
        assert h["ok"] is False and h["model"] is False

    def test_health_check_uses_bounded_client(self, lv, script):
        lv.startup_health_check()
        import ollama
        assert any(c.kwargs.get("timeout") == lv.HEALTH_TIMEOUT for c in ollama.created_clients)

    # [LLM-05] The configured tag must be installed exactly — Ollama answers
    # 404 for "qwen2.5:7b" when only another tag of the family is present.
    def test_exact_configured_tag_is_available(self, lv, script):
        script.list_result = {"models": [{"model": "qwen2.5:3b"}, {"model": "qwen2.5:7b"}]}
        assert lv.startup_health_check()["model"] is True

    @pytest.mark.parametrize("installed", [["qwen2.5:3b"], ["qwen2.5:7b-instruct"],
                                           ["qwen2.5:14b", "qwen2.5-coder:7b"], []])
    def test_other_tags_of_same_family_are_not_available(self, lv, script, installed):
        script.list_result = {"models": [{"model": m} for m in installed]}
        h = lv.startup_health_check()
        assert h["model"] is False and h["ok"] is False
        assert f"ollama pull {lv.OLLAMA_MODEL}" in h["message"]

    def test_status_bar_distinguishes_missing_model_from_server_down(self, app_mod, lv, script):
        script.list_result = {"models": [{"model": "qwen2.5:3b"}]}
        html = app_mod.build_status_html()
        assert "not installed" in html and f"ollama pull {lv.OLLAMA_MODEL}" in html
        assert "ollama serve" not in html and "Model online" not in html
        script.list_result = ConnectionError("refused on 127.0.0.1:11434")
        html = app_mod.build_status_html()
        assert "ollama serve" in html and "not installed" not in html
        assert "127.0.0.1" not in html and "refused" not in html
        script.list_result = {"models": [{"model": lv.OLLAMA_MODEL}]}
        assert "Model online" in app_mod.build_status_html()

    def test_help_tab_model_guidance_matches_configuration(self, app_mod, lv):
        import gradio as gr
        helps = [b.value for b in app_mod.demo.blocks.values()
                 if isinstance(b, gr.Markdown) and "## Quick Start" in str(b.value)]
        assert len(helps) == 1
        text = helps[0]
        assert f"ollama pull {lv.OLLAMA_MODEL}" in text      # the configured, exact tag
        assert "ollama list" in text and "OLLAMA_MODEL" in text and "qwen2.5:3b" in text
        assert "llama3.2" not in text                         # stale, unconfigured suggestion

    def test_untagged_name_matches_latest(self, lv, script, monkeypatch):
        monkeypatch.setattr(lv, "OLLAMA_MODEL", "llama3")
        script.list_result = {"models": [{"model": "llama3:latest"}]}
        assert lv.startup_health_check()["model"] is True


class TestUIErrorHandling:
    def test_do_chat_shows_error_and_export_skips_it(self, app_mod, lv, script):
        _seed(lv)
        script.steps = [[tok("Shadow.")], ConnectionError("down"), ConnectionError("down"),
                        ConnectionError("down")]
        hist = []
        for h, _, _ in app_mod.do_chat("cat?", hist, "⚡ Fast Lookup"):
            hist = h
        assert hist[-1]["content"] == "Shadow."
        for h, trace, _ in app_mod.do_chat("again?", hist, "⚡ Fast Lookup"):
            hist = h
        assert hist[-1]["content"].startswith("❌ ") and "Model error" in trace
        assert app_mod.do_export(hist) == "Shadow."
