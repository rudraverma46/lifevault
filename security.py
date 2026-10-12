"""
LifeVault — security configuration helpers
==========================================
Standard library only. This module must stay free of heavy imports
(ChromaDB, sentence-transformers, Ollama, Gradio) so it can be imported and
tested without initialising the engine or touching the real vault/database.

Responsibilities
----------------
[SEC-01] get_server_settings()  — host / port / auth from environment variables.
         Localhost by default; refuses to bind beyond loopback without auth.
[SEC-02] resolve_ingest_path()  — confines ingestion to permitted root folders,
         resolving '..' and symlinks before any containment check.
[SEC-03] partition_contained()  — drops files (e.g. symlinks) whose real
         location escapes the folder being ingested.

Environment variables
---------------------
LIFEVAULT_HOST           Bind address (default 127.0.0.1).
LIFEVAULT_PORT           TCP port 1–65535 (default 7860).
LIFEVAULT_AUTH_USER      Login username (required for non-loopback hosts).
LIFEVAULT_AUTH_PASSWORD  Login password (required for non-loopback hosts).
LIFEVAULT_ALLOWED_ROOTS  Extra absolute folders permitted for ingestion,
                         separated by os.pathsep (':' on Linux/macOS).

Credentials are never included in exception messages, repr() output or logs.
"""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
DEFAULT_VAULT_DIR = "./my_vault"

ENV_HOST = "LIFEVAULT_HOST"
ENV_PORT = "LIFEVAULT_PORT"
ENV_AUTH_USER = "LIFEVAULT_AUTH_USER"
ENV_AUTH_PASSWORD = "LIFEVAULT_AUTH_PASSWORD"
ENV_ALLOWED_ROOTS = "LIFEVAULT_ALLOWED_ROOTS"

# RFC 1123 hostname: dot-separated labels of letters/digits/hyphens,
# each 1–63 chars, not starting or ending with a hyphen.
_HOST_LABEL = r"(?!-)[A-Za-z0-9-]{1,63}(?<!-)"
_HOSTNAME_RE = re.compile(rf"^{_HOST_LABEL}(\.{_HOST_LABEL})*$")
_PORT_RE = re.compile(r"^[0-9]{1,5}$")


class ConfigError(ValueError):
    """Invalid or unsafe LifeVault configuration (environment variables)."""


class PathValidationError(ValueError):
    """An ingestion path was rejected. Message is safe to show to the user."""


# ── SERVER SETTINGS  [SEC-01] ─────────────────────────────────────────────────
@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int
    is_local: bool
    # repr=False keeps the password out of tracebacks, logs and debug prints.
    auth: tuple[str, str] | None = field(default=None, repr=False)

    @property
    def auth_enabled(self) -> bool:
        return self.auth is not None


def is_loopback_host(host: str) -> bool:
    """
    True only for 'localhost' or a loopback IP literal (127.0.0.0/8, ::1).
    Any other hostname is treated as non-local, even if it might resolve to
    loopback, so that ambiguous configurations require authentication.
    """
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_host(raw: str) -> str:
    host = raw.strip()
    if not host:
        raise ConfigError(
            f"{ENV_HOST} is set but empty. Unset it to use {DEFAULT_HOST}, "
            f"or set it to an IP address or hostname."
        )
    if len(host) > 253:
        raise ConfigError(f"{ENV_HOST} is too long to be a valid host.")
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    if _HOSTNAME_RE.match(host):
        # All-numeric names such as "127.1" or "0177.0.0.1" are interpreted
        # inconsistently by resolvers (shorthand/octal IPv4) — reject them.
        if all(label.isdigit() for label in host.split(".")):
            raise ConfigError(
                f"{ENV_HOST}={host!r} is not a valid IP address. Use the full "
                f"dotted form, e.g. 127.0.0.1."
            )
        return host
    raise ConfigError(
        f"{ENV_HOST}={host!r} is not a valid IP address or hostname. "
        f"Use e.g. 127.0.0.1, 0.0.0.0, ::1 or a plain hostname — no scheme, "
        f"port, brackets or path."
    )


def _validate_port(raw: str) -> int:
    value = raw.strip()
    if not _PORT_RE.match(value):
        raise ConfigError(
            f"{ENV_PORT} must be a whole number between 1 and 65535."
        )
    port = int(value)
    if not 1 <= port <= 65535:
        raise ConfigError(
            f"{ENV_PORT} must be a whole number between 1 and 65535."
        )
    return port


def get_server_settings(env: Mapping[str, str] | None = None) -> ServerSettings:
    """
    Build validated server settings from environment variables.

    Rules:
      * Default host 127.0.0.1, default port 7860.
      * Credentials must be supplied as a pair, or not at all.
      * A non-loopback host REQUIRES both credentials — there is no
        fallback to unauthenticated network access.
    Raises ConfigError with an actionable message (never containing secrets).
    """
    env = os.environ if env is None else env

    raw_host = env.get(ENV_HOST)
    host = DEFAULT_HOST if raw_host is None else _validate_host(raw_host)

    raw_port = env.get(ENV_PORT)
    port = DEFAULT_PORT if raw_port is None else _validate_port(raw_port)

    user = env.get(ENV_AUTH_USER) or ""
    password = env.get(ENV_AUTH_PASSWORD) or ""
    has_user = bool(user.strip())
    has_password = bool(password)

    if has_user != has_password:
        missing = ENV_AUTH_PASSWORD if has_user else ENV_AUTH_USER
        raise ConfigError(
            f"Incomplete credentials: {missing} is not set. Set both "
            f"{ENV_AUTH_USER} and {ENV_AUTH_PASSWORD}, or neither."
        )

    local = is_loopback_host(host)
    auth = (user, password) if has_user and has_password else None

    if not local and auth is None:
        raise ConfigError(
            f"Refusing to start: {ENV_HOST}={host} exposes LifeVault beyond this "
            f"machine, but no login is configured. Set both {ENV_AUTH_USER} and "
            f"{ENV_AUTH_PASSWORD}, or unset {ENV_HOST} to stay on {DEFAULT_HOST}. "
            f"Note: the login does not encrypt traffic — use HTTPS or a secure "
            f"tunnel on untrusted networks."
        )

    return ServerSettings(host=host, port=port, is_local=local, auth=auth)


# ── INGESTION PATHS  [SEC-02, SEC-03] ─────────────────────────────────────────
def _expand(raw: str, base_dir: Path) -> Path:
    if "\x00" in raw:
        raise PathValidationError("Path contains an invalid character.")
    p = Path(raw).expanduser()
    return p if p.is_absolute() else base_dir / p


def _resolve_existing_dir(p: Path, what: str) -> Path:
    """Resolve symlinks/'..' strictly; the target must be an existing directory."""
    try:
        resolved = p.resolve(strict=True)
    except FileNotFoundError:
        raise PathValidationError(f"{what} does not exist.") from None
    except PermissionError:
        raise PathValidationError(f"{what} is not accessible (permission denied).") from None
    except (OSError, RuntimeError):  # symlink loops, ENAMETOOLONG, etc.
        raise PathValidationError(f"{what} could not be resolved.") from None
    try:
        is_dir = resolved.is_dir()
    except OSError:
        raise PathValidationError(f"{what} is not accessible.") from None
    if not is_dir:
        raise PathValidationError(f"{what} is not a directory.")
    return resolved


def get_allowed_roots(
    default_vault: str = DEFAULT_VAULT_DIR,
    env: Mapping[str, str] | None = None,
    base_dir: Path | None = None,
) -> list[Path]:
    """
    Return the resolved list of permitted ingestion roots:
    the default vault plus every entry in LIFEVAULT_ALLOWED_ROOTS.

    Extra roots must be absolute (or '~'-prefixed), exist, be directories, and
    must not be the filesystem root. Empty entries (e.g. a trailing ':') are
    ignored. Any other malformed entry raises ConfigError — the whole
    configuration is rejected rather than partially applied.
    """
    env = os.environ if env is None else env
    base = Path.cwd() if base_dir is None else Path(base_dir)

    roots: list[Path] = []
    default_path = _expand(default_vault, base)
    # The default vault may not exist yet; resolve non-strictly for comparison.
    try:
        roots.append(default_path.resolve(strict=False))
    except (OSError, RuntimeError):
        raise ConfigError("Default vault path could not be resolved.") from None

    raw = env.get(ENV_ALLOWED_ROOTS, "")
    for i, entry in enumerate(raw.split(os.pathsep), start=1):
        entry = entry.strip()
        if not entry:
            continue
        label = f"{ENV_ALLOWED_ROOTS} entry #{i}"
        if "\x00" in entry:
            raise ConfigError(f"{label} contains an invalid character.")
        p = Path(entry).expanduser()
        if not p.is_absolute():
            raise ConfigError(f"{label} must be an absolute path.")
        try:
            resolved = _resolve_existing_dir(p, label)
        except PathValidationError as e:
            raise ConfigError(str(e)) from None
        if resolved == Path(resolved.anchor):
            raise ConfigError(f"{label} must not be the filesystem root.")
        roots.append(resolved)
    return roots


def is_within(path: Path, root: Path) -> bool:
    """Component-wise containment (not string prefix): /a/vault2 ∉ /a/vault."""
    return path == root or root in path.parents


def resolve_ingest_path(
    user_path: str | os.PathLike | None,
    default_vault: str = DEFAULT_VAULT_DIR,
    env: Mapping[str, str] | None = None,
    base_dir: Path | None = None,
) -> Path:
    """
    Validate a folder requested for ingestion and return its resolved Path.

    * Blank input means the default vault.
    * Relative paths are taken relative to base_dir (default: current dir,
      matching how VAULT_DIR is interpreted elsewhere).
    * '~', '..' and symlinks are fully resolved BEFORE the containment check.
    * The result must be an existing directory inside a permitted root.
    * Only the default vault may be created if missing (preserves v2.0
      behaviour); any other missing path is rejected.

    Raises PathValidationError (user-facing, no resolved-target disclosure)
    or ConfigError (malformed LIFEVAULT_ALLOWED_ROOTS).
    """
    base = Path.cwd() if base_dir is None else Path(base_dir)

    if user_path is None:
        user_path = ""
    if isinstance(user_path, os.PathLike):
        user_path = os.fspath(user_path)
    if not isinstance(user_path, str):
        raise PathValidationError("Path must be text.")

    roots = get_allowed_roots(default_vault, env=env, base_dir=base)
    default_root = roots[0]

    default_path = _expand(default_vault, base)
    raw = user_path.strip() or default_vault
    candidate = _expand(raw, base)

    # Auto-create only the default vault, exactly as v2.0 did.
    try:
        wants_default = candidate.resolve(strict=False) == default_root
    except (OSError, RuntimeError):
        raise PathValidationError("Folder could not be resolved.") from None
    if wants_default and not os.path.exists(default_path):
        if os.path.islink(default_path):
            # Never create a directory through a dangling symlink.
            raise PathValidationError("Default vault path is a broken link.")
        try:
            default_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise PathValidationError("Default vault folder could not be created.") from None

    resolved = _resolve_existing_dir(candidate, "Folder")

    # Re-resolve the default root strictly now that it may exist (symlinked vault).
    try:
        roots[0] = default_root.resolve(strict=True)
    except (OSError, RuntimeError):
        pass

    if not any(is_within(resolved, r) for r in roots):
        raise PathValidationError(
            "Folder is outside the permitted ingestion folders. Use the vault "
            f"folder, or add the folder to {ENV_ALLOWED_ROOTS} and restart."
        )
    return resolved


def partition_contained(
    files: Iterable[Path], root: Path
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """
    Split discovered files into (inside, skipped) relative to the resolved
    ingestion root. A file is kept only if its fully resolved location lies
    inside `root`. Symlinks escaping the tree, broken links and unresolvable
    entries are returned in `skipped` with a short reason.
    """
    inside: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for f in files:
        try:
            real = Path(f).resolve(strict=True)
        except FileNotFoundError:
            skipped.append((f, "broken link or missing file"))
            continue
        except (OSError, RuntimeError):
            skipped.append((f, "could not be resolved"))
            continue
        if is_within(real, root):
            inside.append(f)
        else:
            skipped.append((f, "link points outside the ingestion folder"))
    return inside, skipped
