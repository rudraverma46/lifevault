#!/usr/bin/env bash
# LifeVault — background boot script
# Usage: ./scripts/start.sh [start|stop|restart|status]
#
# Host/port/login come from LIFEVAULT_HOST, LIFEVAULT_PORT,
# LIFEVAULT_AUTH_USER, LIFEVAULT_AUTH_PASSWORD (validated by app.py/security.py).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$ROOT_DIR/logs/lifevault.pid"
LOG_FILE="$ROOT_DIR/logs/app.log"

mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/my_vault"

# URL shown to the user (display only — app.py does the real validation).
app_url() {
  local host="${LIFEVAULT_HOST:-127.0.0.1}" port="${LIFEVAULT_PORT:-7860}"
  case "$host" in
    0.0.0.0|::) host="localhost" ;;   # all interfaces → reachable locally as localhost
  esac
  case "$host" in
    *:*) host="[$host]" ;;             # bracket IPv6 literals
  esac
  echo "http://$host:$port"
}

# [SEC-06] Print the PID from the PID file only if it is well-formed, alive,
# and is this checkout's `python app.py` — never trust the file blindly
# (stale or reused PIDs must not lead to killing unrelated processes).
lifevault_pid() {
  [ -f "$PID_FILE" ] || return 1
  local pid
  pid="$(head -c 32 "$PID_FILE" 2>/dev/null)"
  # Trim only leading/trailing whitespace; "12 34" must stay malformed.
  pid="${pid#"${pid%%[![:space:]]*}"}"
  pid="${pid%"${pid##*[![:space:]]}"}"
  [[ "$pid" =~ ^[1-9][0-9]{0,9}$ ]] || return 2
  kill -0 "$pid" 2>/dev/null || return 3

  if [ -r "/proc/$pid/cmdline" ]; then
    # Linux: inspect argv exactly and confirm the working directory.
    local -a argv=()
    mapfile -d '' -t argv < "/proc/$pid/cmdline" 2>/dev/null || return 4
    [ "${#argv[@]}" -ge 2 ] || return 4
    case "$(basename -- "${argv[0]}")" in python*) ;; *) return 4 ;; esac
    local found=0 arg
    for arg in "${argv[@]:1}"; do
      if [ "$arg" = "app.py" ] || [ "$arg" = "$ROOT_DIR/app.py" ]; then found=1; fi
    done
    [ "$found" -eq 1 ] || return 4
    local cwd
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null)" || return 4
    [ "$cwd" = "$(readlink -f "$ROOT_DIR")" ] || return 4
  else
    # Other POSIX (e.g. macOS): best effort via ps; cwd cannot be checked.
    local args
    args="$(ps -p "$pid" -o args= 2>/dev/null)" || return 4
    case "$args" in
      *python*" app.py"|*python*" $ROOT_DIR/app.py") ;;
      *) return 4 ;;
    esac
  fi
  echo "$pid"
}

case "${1:-start}" in
  start)
    if pid="$(lifevault_pid)"; then
      echo "LifeVault is already running (PID $pid)."
      echo "Open: $(app_url)"
      exit 0
    fi
    rm -f "$PID_FILE"   # stale/malformed/foreign PID file — safe to discard
    echo "Starting LifeVault..."
    cd "$ROOT_DIR"
    nohup ./LIFEVAULT_1/bin/python app.py >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 2
    if pid="$(lifevault_pid)"; then
      echo "✅  LifeVault running (PID $pid)"
      echo "    Open: $(app_url)"
      echo "    Logs: tail -f $LOG_FILE"
    else
      echo "❌  LifeVault failed to start. Check: $LOG_FILE"
      rm -f "$PID_FILE"
      exit 1
    fi
    ;;
  stop)
    if [ ! -f "$PID_FILE" ]; then
      echo "LifeVault is not running."
      exit 0
    fi
    if pid="$(lifevault_pid)"; then
      kill "$pid" 2>/dev/null && echo "Stopped (PID $pid)." || echo "Could not signal PID $pid."
      # Wait briefly for a clean exit so 'restart' can rebind the port.
      for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
    else
      case $? in
        2) echo "PID file is malformed — not killing anything." ;;
        3) echo "Process not found (stale PID file)." ;;
        *) echo "PID file does not belong to this LifeVault app — not killing it." ;;
      esac
    fi
    rm -f "$PID_FILE"
    ;;
  restart)
    "$0" stop
    sleep 1
    "$0" start
    ;;
  status)
    if pid="$(lifevault_pid)"; then
      echo "🟢  Running (PID $pid) → $(app_url)"
    else
      echo "🔴  Not running"
    fi
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac
