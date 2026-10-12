"""
Isolated security tests for LifeVault.

Safety guarantees (by construction):
  * Every filesystem operation happens under pytest's tmp_path.
  * get_server_settings / resolve_ingest_path receive explicit env dicts and
    base_dir — the real environment and ./my_vault are never consulted.
  * lifevault.py / app.py are imported ONLY with chromadb, sentence_transformers
    and ollama replaced by in-memory fakes and the CWD switched to tmp_path, so
    no real ChromaDB, embedding model, Ollama server, vault or chat history is
    touched. A guard asserts this before import.
  * start.sh is exercised as a COPY inside tmp_path; only `status` and `stop`
    are run, and only against throw-away processes that the script must
    REFUSE to kill. No process is killed by these tests.

Run:  python -m pytest tests/test_security.py -v
"""

import logging
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from security import (  # noqa: E402  (stdlib-only module — safe to import)
    ConfigError,
    PathValidationError,
    get_server_settings,
    is_loopback_host,
    partition_contained,
    resolve_ingest_path,
)

SECRET = "s3cr3t-Pa55word-do-not-leak"
NET = {"LIFEVAULT_HOST": "0.0.0.0"}
CREDS = {"LIFEVAULT_AUTH_USER": "alice", "LIFEVAULT_AUTH_PASSWORD": SECRET}

needs_symlinks = pytest.mark.skipif(
    not hasattr(os, "symlink") or os.name == "nt", reason="POSIX symlinks required"
)


# ══════════════════════════════════════════════════════════════════════════════
# A. Server settings / authentication
# ══════════════════════════════════════════════════════════════════════════════
class TestServerSettings:
    def test_localhost_defaults(self):
        s = get_server_settings({})
        assert (s.host, s.port, s.is_local, s.auth) == ("127.0.0.1", 7860, True, None)
        assert not s.auth_enabled

    @pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "localhost", "LOCALHOST", "::1"])
    def test_loopback_hosts_need_no_auth(self, host):
        s = get_server_settings({"LIFEVAULT_HOST": host})
        assert s.is_local and s.auth is None and s.host == host

    def test_explicit_host_and_port(self):
        s = get_server_settings({"LIFEVAULT_HOST": " 127.0.0.1 ", "LIFEVAULT_PORT": " 8080 "})
        assert (s.host, s.port) == ("127.0.0.1", 8080)

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "myhost.lan", "example"])
    def test_network_binding_without_credentials_refused(self, host):
        with pytest.raises(ConfigError) as ei:
            get_server_settings({"LIFEVAULT_HOST": host})
        msg = str(ei.value)
        assert "LIFEVAULT_AUTH_USER" in msg and "LIFEVAULT_AUTH_PASSWORD" in msg
        assert "HTTPS" in msg  # transport caveat is surfaced

    def test_network_binding_with_both_credentials(self):
        s = get_server_settings({**NET, **CREDS, "LIFEVAULT_PORT": "9000"})
        assert s.auth == ("alice", SECRET)
        assert s.auth_enabled and not s.is_local and s.port == 9000

    def test_localhost_with_credentials_enables_auth(self):
        s = get_server_settings(CREDS)
        assert s.is_local and s.auth == ("alice", SECRET)

    @pytest.mark.parametrize("base", [{}, NET])
    def test_missing_username(self, base):
        with pytest.raises(ConfigError) as ei:
            get_server_settings({**base, "LIFEVAULT_AUTH_PASSWORD": SECRET})
        assert "LIFEVAULT_AUTH_USER" in str(ei.value)
        assert SECRET not in str(ei.value)

    @pytest.mark.parametrize("base", [{}, NET])
    def test_missing_password(self, base):
        with pytest.raises(ConfigError) as ei:
            get_server_settings({**base, "LIFEVAULT_AUTH_USER": "alice"})
        assert "LIFEVAULT_AUTH_PASSWORD" in str(ei.value)

    def test_blank_username_counts_as_missing(self):
        with pytest.raises(ConfigError):
            get_server_settings({**NET, "LIFEVAULT_AUTH_USER": "   ", "LIFEVAULT_AUTH_PASSWORD": SECRET})

    def test_empty_password_counts_as_missing(self):
        with pytest.raises(ConfigError):
            get_server_settings({**NET, "LIFEVAULT_AUTH_USER": "alice", "LIFEVAULT_AUTH_PASSWORD": ""})

    @pytest.mark.parametrize(
        "port", ["0", "65536", "-1", "+80", "8_080", "80.0", "abc", "", "  ", "99999999", "0x50", "８０"]
    )
    def test_invalid_ports(self, port):
        with pytest.raises(ConfigError) as ei:
            get_server_settings({"LIFEVAULT_PORT": port})
        assert "LIFEVAULT_PORT" in str(ei.value)

    @pytest.mark.parametrize("port", ["1", "65535", "07860"])
    def test_boundary_ports(self, port):
        assert get_server_settings({"LIFEVAULT_PORT": port}).port == int(port)

    @pytest.mark.parametrize(
        "host",
        ["", "   ", "http://0.0.0.0", "0.0.0.0:7860", "[::1]", "host name", "a/b",
         "127.1", "0177.0.0.1", "0", "-bad.host", "bad-.host", "x" * 300, "local host", "*"],
    )
    def test_invalid_hosts(self, host):
        with pytest.raises(ConfigError) as ei:
            get_server_settings({"LIFEVAULT_HOST": host})
        assert "LIFEVAULT_HOST" in str(ei.value)

    def test_invalid_host_checked_before_credentials(self):
        # A malformed host never yields settings, even with credentials present.
        with pytest.raises(ConfigError):
            get_server_settings({"LIFEVAULT_HOST": "0.0.0.0:80", **CREDS})

    def test_password_never_in_repr_or_errors(self):
        s = get_server_settings({**NET, **CREDS})
        assert SECRET not in repr(s) and SECRET not in str(s)
        for env in ({**NET, "LIFEVAULT_AUTH_PASSWORD": SECRET},
                    {"LIFEVAULT_PORT": "bad", **CREDS},
                    {"LIFEVAULT_HOST": "bad host", **CREDS}):
            with pytest.raises(ConfigError) as ei:
                get_server_settings(env)
            assert SECRET not in str(ei.value)

    def test_loopback_detection_is_conservative(self):
        assert is_loopback_host("127.255.0.1")
        assert not is_loopback_host("0.0.0.0")
        assert not is_loopback_host("localhost.evil.com")
        assert not is_loopback_host("127.0.0.1.nip.io")


# ══════════════════════════════════════════════════════════════════════════════
# B. Ingestion path validation
# ══════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def tree(tmp_path):
    """tmp/{my_vault/{sub,notes.md}, other/, my_vault2/, notes/, notes_private/}"""
    (tmp_path / "my_vault" / "sub").mkdir(parents=True)
    (tmp_path / "my_vault" / "notes.md").write_text("x")
    for d in ("other", "my_vault2", "notes", "notes_private"):
        (tmp_path / d).mkdir()
    return tmp_path


def rip(path, base, env=None):
    return resolve_ingest_path(path, env={} if env is None else env, base_dir=base)


class TestResolveIngestPath:
    def test_blank_means_default_vault(self, tree):
        assert rip("", tree) == (tree / "my_vault").resolve()
        assert rip(None, tree) == (tree / "my_vault").resolve()
        assert rip("   ", tree) == (tree / "my_vault").resolve()

    @pytest.mark.parametrize("p", ["./my_vault", "my_vault", "my_vault/sub", "my_vault/sub/../sub"])
    def test_valid_relative_paths_inside_vault(self, tree, p):
        assert rip(p, tree) == (tree / p).resolve()

    def test_valid_absolute_path_inside_vault(self, tree):
        assert rip(str(tree / "my_vault" / "sub"), tree) == (tree / "my_vault" / "sub").resolve()

    def test_pathlike_accepted(self, tree):
        assert rip(tree / "my_vault", tree) == (tree / "my_vault").resolve()

    def test_default_vault_is_auto_created(self, tmp_path):
        assert not (tmp_path / "my_vault").exists()
        assert rip("", tmp_path) == (tmp_path / "my_vault").resolve()
        assert (tmp_path / "my_vault").is_dir()

    def test_valid_path_inside_explicit_allowed_root(self, tree):
        env = {"LIFEVAULT_ALLOWED_ROOTS": str(tree / "other")}
        assert rip(str(tree / "other"), tree, env) == (tree / "other").resolve()
        (tree / "other" / "deep").mkdir()
        assert rip(str(tree / "other" / "deep"), tree, env) == (tree / "other" / "deep").resolve()

    def test_multiple_allowed_roots_and_empty_segments(self, tree):
        env = {"LIFEVAULT_ALLOWED_ROOTS": f"{os.pathsep}{tree/'other'}{os.pathsep}{os.pathsep}{tree/'notes'}{os.pathsep}"}
        assert rip(str(tree / "notes"), tree, env) == (tree / "notes").resolve()
        assert rip(str(tree / "other"), tree, env) == (tree / "other").resolve()

    @pytest.mark.parametrize("p", ["other", "/", "..", "/etc"])
    def test_outside_permitted_roots_rejected(self, tree, p):
        target = p if os.path.isabs(p) else str(tree / p)
        with pytest.raises(PathValidationError) as ei:
            rip(target, tree)
        assert "outside the permitted" in str(ei.value)

    @pytest.mark.parametrize("p", ["my_vault/../other", "my_vault/sub/../../other", "my_vault/../../"])
    def test_parent_traversal_rejected(self, tree, p):
        with pytest.raises(PathValidationError):
            rip(p, tree)

    def test_common_prefix_sibling_of_vault_rejected(self, tree):
        with pytest.raises(PathValidationError):
            rip(str(tree / "my_vault2"), tree)

    def test_common_prefix_sibling_of_allowed_root_rejected(self, tree):
        env = {"LIFEVAULT_ALLOWED_ROOTS": str(tree / "notes")}
        with pytest.raises(PathValidationError):
            rip(str(tree / "notes_private"), tree, env)

    @needs_symlinks
    def test_symlink_inside_vault_pointing_outside_rejected(self, tree):
        (tree / "secret_target").mkdir()
        os.symlink(tree / "secret_target", tree / "my_vault" / "link")
        with pytest.raises(PathValidationError) as ei:
            rip("my_vault/link", tree)
        assert "secret_target" not in str(ei.value)  # resolved target not disclosed

    @needs_symlinks
    def test_symlink_inside_vault_pointing_inside_allowed(self, tree):
        os.symlink(tree / "my_vault" / "sub", tree / "my_vault" / "alias")
        assert rip("my_vault/alias", tree) == (tree / "my_vault" / "sub").resolve()

    @needs_symlinks
    def test_allowed_root_given_via_symlink_is_resolved(self, tree):
        os.symlink(tree / "other", tree / "other_link")
        env = {"LIFEVAULT_ALLOWED_ROOTS": str(tree / "other_link")}
        assert rip(str(tree / "other"), tree, env) == (tree / "other").resolve()

    @needs_symlinks
    def test_symlinked_default_vault_is_followed(self, tmp_path):
        (tmp_path / "real_vault").mkdir()
        os.symlink(tmp_path / "real_vault", tmp_path / "my_vault")
        assert rip("", tmp_path) == (tmp_path / "real_vault").resolve()

    @needs_symlinks
    def test_dangling_default_vault_link_not_created(self, tmp_path):
        os.symlink(tmp_path / "nowhere", tmp_path / "my_vault")
        with pytest.raises(PathValidationError):
            rip("", tmp_path)
        assert not (tmp_path / "nowhere").exists()

    @needs_symlinks
    def test_symlink_loop_rejected(self, tree):
        loop = tree / "my_vault" / "loop"
        os.symlink(loop, loop)
        with pytest.raises(PathValidationError) as ei:
            rip("my_vault/loop", tree)
        assert "resolved" in str(ei.value)

    def test_nonexistent_path_rejected_and_not_created(self, tree):
        with pytest.raises(PathValidationError) as ei:
            rip("my_vault/new_folder", tree)
        assert "does not exist" in str(ei.value)
        assert not (tree / "my_vault" / "new_folder").exists()

    def test_nonexistent_outside_path_not_created(self, tree):
        with pytest.raises(PathValidationError):
            rip(str(tree / "brand_new"), tree)
        assert not (tree / "brand_new").exists()

    def test_file_instead_of_directory_rejected(self, tree):
        with pytest.raises(PathValidationError) as ei:
            rip("my_vault/notes.md", tree)
        assert "not a directory" in str(ei.value)

    @pytest.mark.parametrize("bad", ["my_vault\x00/../other", "\x00"])
    def test_nul_byte_rejected(self, tree, bad):
        with pytest.raises(PathValidationError):
            rip(bad, tree)

    def test_non_text_rejected(self, tree):
        with pytest.raises(PathValidationError):
            rip(123, tree)

    def test_overlong_path_rejected(self, tree):
        with pytest.raises(PathValidationError):
            rip("my_vault/" + "a" * 5000, tree)

    @pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root bypasses permissions")
    def test_permission_denied_is_handled(self, tree):
        locked = tree / "my_vault" / "locked"
        (locked / "inner").mkdir(parents=True)
        locked.chmod(0)
        try:
            with pytest.raises(PathValidationError) as ei:
                rip("my_vault/locked/inner", tree)
            assert "permission" in str(ei.value).lower() or "accessible" in str(ei.value) \
                or "exist" in str(ei.value)
        finally:
            locked.chmod(0o700)

    # ── malformed LIFEVAULT_ALLOWED_ROOTS ────────────────────────────────────
    @pytest.mark.parametrize(
        "value_fn",
        [
            lambda t: "relative/dir",
            lambda t: str(t / "missing"),
            lambda t: str(t / "my_vault" / "notes.md"),
            lambda t: "/",
            lambda t: f"{t/'other'}{os.pathsep}relative",
            lambda t: "bad\x00root",
        ],
        ids=["relative", "missing", "file", "fs-root", "one-bad-entry", "nul"],
    )
    def test_malformed_allowed_roots_fail_closed(self, tree, value_fn):
        env = {"LIFEVAULT_ALLOWED_ROOTS": value_fn(tree)}
        with pytest.raises(ConfigError) as ei:
            rip("", tree, env)  # even the default vault is refused until fixed
        assert "LIFEVAULT_ALLOWED_ROOTS" in str(ei.value)


class TestPartitionContained:
    @needs_symlinks
    def test_escaping_links_are_skipped(self, tree):
        vault = (tree / "my_vault").resolve()
        (tree / "other" / "secret.md").write_text("secret")
        os.symlink(tree / "other" / "secret.md", vault / "leak.md")
        os.symlink(tree / "other", vault / "leakdir")
        os.symlink(tree / "gone.md", vault / "broken.md")
        os.symlink(vault / "notes.md", vault / "alias.md")

        found = sorted(vault.rglob("*.md"))
        # Python < 3.13 may or may not descend into symlinked dirs; add explicitly.
        found.append(vault / "leakdir" / "secret.md")
        inside, skipped = partition_contained(found, vault)

        names_in = {p.name for p in inside}
        reasons = {p.name if p.parent.name != "leakdir" else "leakdir/" + p.name: r for p, r in skipped}
        assert names_in == {"notes.md", "alias.md"}
        assert "outside" in reasons["leak.md"]
        assert "outside" in reasons["leakdir/secret.md"]
        assert "broken" in reasons["broken.md"]


# ══════════════════════════════════════════════════════════════════════════════
# C. Engine / UI integration — with fakes, inside tmp_path only
# ══════════════════════════════════════════════════════════════════════════════
class _Vec(list):
    def tolist(self):
        return list(self)


class _FakeCollection:
    def __init__(self):
        self.items: dict[str, dict] = {}

    def get(self, ids=None, where=None, **_):
        if ids is not None:
            return {"ids": [i for i in ids if i in self.items]}
        return {"ids": [i for i, v in self.items.items()
                        if not where or all(v["meta"].get(k) == w for k, w in where.items())]}

    def add(self, ids, embeddings, documents, metadatas):
        for i, d, m in zip(ids, documents, metadatas):
            self.items[i] = {"doc": d, "meta": m}

    def count(self):
        return len(self.items)


def _fake_modules(collection):
    chromadb = types.ModuleType("chromadb")
    chromadb.PersistentClient = lambda path=None, **_: types.SimpleNamespace(
        get_or_create_collection=lambda name, **__: collection)

    st = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, *a, **k):
            pass

        def encode(self, text):
            return _Vec([0.0] * 4)

    st.SentenceTransformer = SentenceTransformer

    ollama = types.ModuleType("ollama")
    ollama.list = lambda: {"models": []}

    def _no_network(*a, **k):
        raise RuntimeError("ollama is stubbed in tests")

    ollama.chat = _no_network
    return {"chromadb": chromadb, "sentence_transformers": st, "ollama": ollama}


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """Import lifevault with all heavy deps faked and CWD = tmp_path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LIFEVAULT_ALLOWED_ROOTS", raising=False)
    col = _FakeCollection()
    for name, mod in _fake_modules(col).items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "lifevault", raising=False)
    # Guard: never import the engine against the real project directory.
    assert Path.cwd().resolve() == tmp_path.resolve() != ROOT
    import lifevault

    assert lifevault.collection is col
    yield lifevault, col, tmp_path
    lg = logging.getLogger("LifeVault")
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()
    sys.modules.pop("lifevault", None)


LONG = "This is a sufficiently long line of note text for chunking purposes. " * 3


def _drain(gen):
    return [msg for msg, _ in gen]


class TestEngineIntegration:
    def test_default_vault_ingest_still_works(self, engine):
        lv, col, tmp = engine
        (tmp / "my_vault").mkdir()
        (tmp / "my_vault" / "a.md").write_text(LONG)
        logs = _drain(lv.ingest_vault())
        assert "Done" in logs[-1]
        assert {v["meta"]["source"] for v in col.items.values()} == {"a.md"}

    def test_default_vault_auto_created(self, engine):
        lv, col, tmp = engine
        logs = _drain(lv.ingest_vault())
        assert (tmp / "my_vault").is_dir()
        assert "No supported files" in logs[-1]

    def test_outside_path_refused_and_nothing_indexed(self, engine):
        lv, col, tmp = engine
        (tmp / "elsewhere").mkdir()
        (tmp / "elsewhere" / "b.md").write_text(LONG)
        logs = _drain(lv.ingest_vault(str(tmp / "elsewhere")))
        assert len(logs) == 1 and "Ingestion refused" in logs[0]
        assert col.count() == 0

    def test_missing_path_not_created(self, engine):
        lv, col, tmp = engine
        logs = _drain(lv.ingest_vault(str(tmp / "my_vault" / "nope")))
        assert "Ingestion refused" in logs[0]
        assert not (tmp / "my_vault" / "nope").exists()

    @needs_symlinks
    def test_symlink_escape_skipped_with_warning(self, engine, caplog):
        lv, col, tmp = engine
        (tmp / "my_vault").mkdir()
        (tmp / "outside").mkdir()
        (tmp / "my_vault" / "ok.md").write_text(LONG)
        (tmp / "outside" / "secret.md").write_text(LONG)
        os.symlink(tmp / "outside" / "secret.md", tmp / "my_vault" / "leak.md")
        logger = logging.getLogger("LifeVault")
        logger.addHandler(caplog.handler)
        try:
            _drain(lv.ingest_vault())
        finally:
            logger.removeHandler(caplog.handler)
        assert {v["meta"]["source"] for v in col.items.values()} == {"ok.md"}
        assert any("leak.md" in r.getMessage() and "outside" in r.getMessage()
                   for r in caplog.records if r.levelno == logging.WARNING)


@pytest.fixture
def app_module(engine, monkeypatch):
    pytest.importorskip("gradio")
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.delitem(sys.modules, "app", raising=False)
    import app  # builds the Blocks UI; does NOT launch (no __main__)

    yield app, engine
    sys.modules.pop("app", None)


class TestAppIntegration:
    def test_do_ingest_rejects_with_readable_message(self, app_module):
        app, (lv, col, tmp) = app_module
        (tmp / "elsewhere").mkdir()
        out = list(app.do_ingest(str(tmp / "elsewhere")))
        assert len(out) == 1
        log, status_html = out[0]
        assert log.startswith("⛔") and "Traceback" not in log
        assert "outside the permitted" in log
        assert col.count() == 0

    def test_blocks_analytics_disabled(self, app_module):
        app, _ = app_module
        assert app.demo.analytics_enabled is False

    def test_no_external_font_import(self, app_module):
        app, _ = app_module
        assert "googleapis" not in app.CSS and "@import" not in app.CSS


# ══════════════════════════════════════════════════════════════════════════════
# D. start.sh — run as a COPY in tmp; never kills anything
# ══════════════════════════════════════════════════════════════════════════════
pytestmark_sh = pytest.mark.skipif(
    shutil.which("bash") is None or not Path("/proc/self/cmdline").exists(),
    reason="needs bash and Linux /proc",
)


@pytest.fixture
def sh_root(tmp_path):
    root = tmp_path / "lv_copy"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "start.sh", root / "scripts" / "start.sh")
    (root / "app.py").write_text("import time\ntime.sleep(4)\n")
    return root


def _sh(root, *args, env=None):
    e = {k: v for k, v in os.environ.items() if not k.startswith("LIFEVAULT_")}
    e.update(env or {})
    return subprocess.run(["bash", str(root / "scripts" / "start.sh"), *args],
                          capture_output=True, text=True, env=e, timeout=30)


def _spawn(cwd, *argv):
    return subprocess.Popen([sys.executable, *argv], cwd=cwd,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _write_pid(root, content):
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / "lifevault.pid").write_text(content)


@pytestmark_sh
class TestStartScript:
    def test_status_without_pid_file(self, sh_root):
        assert "Not running" in _sh(sh_root, "status").stdout

    @pytest.mark.parametrize("content", ["abc", "-1", "0", "12 34", "", "1;echo injected"])
    def test_malformed_pid_file_never_kills(self, sh_root, content):
        _write_pid(sh_root, content)
        r = _sh(sh_root, "stop")
        assert "not killing" in r.stdout or "does not belong to this LifeVault" in r.stdout \
            or "malformed" in r.stdout
        assert not (sh_root / "logs" / "lifevault.pid").exists()

    def test_stale_pid(self, sh_root):
        p = _spawn(sh_root, "-c", "pass")
        p.wait()
        _write_pid(sh_root, str(p.pid))
        r = _sh(sh_root, "stop")
        assert "stale" in r.stdout or "does not belong to this LifeVault" in r.stdout

    def test_unrelated_process_is_not_killed(self, sh_root):
        p = _spawn(sh_root, "-c", "import time; time.sleep(4)")
        try:
            _write_pid(sh_root, str(p.pid))
            r = _sh(sh_root, "stop")
            assert "does not belong to this LifeVault" in r.stdout
            assert p.poll() is None, "unrelated process must still be alive"
            assert "Not running" in _sh(sh_root, "status").stdout
        finally:
            p.wait(timeout=15)  # let it exit naturally — never killed

    def test_app_py_from_another_directory_is_not_killed(self, sh_root, tmp_path):
        other = tmp_path / "other_checkout"
        other.mkdir()
        (other / "app.py").write_text("import time\ntime.sleep(4)\n")
        p = _spawn(other, "app.py")
        try:
            _write_pid(sh_root, str(p.pid))
            r = _sh(sh_root, "stop")
            assert "does not belong to this LifeVault" in r.stdout
            assert p.poll() is None
        finally:
            p.wait(timeout=15)

    @pytest.mark.parametrize(
        "env, url",
        [({}, "http://127.0.0.1:7860"),
         ({"LIFEVAULT_HOST": "0.0.0.0", "LIFEVAULT_PORT": "9000"}, "http://localhost:9000"),
         ({"LIFEVAULT_HOST": "::1"}, "http://[::1]:7860")],
    )
    def test_status_recognises_own_app_and_shows_configured_url(self, sh_root, env, url):
        p = _spawn(sh_root, "app.py")
        try:
            time.sleep(0.3)
            _write_pid(sh_root, str(p.pid))
            r = _sh(sh_root, "status", env=env)
            assert f"Running (PID {p.pid})" in r.stdout and url in r.stdout
            assert p.poll() is None
        finally:
            p.wait(timeout=15)  # status never signals; process exits by itself
