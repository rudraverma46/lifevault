#!/usr/bin/env bash
# LifeVault — background boot script
# Usage: ./scripts/start.sh [stop|restart|status]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$ROOT_DIR/logs/lifevault.pid"
LOG_FILE="$ROOT_DIR/logs/app.log"

mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/my_vault"

case "${1:-start}" in
  start)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "LifeVault is already running (PID $(cat "$PID_FILE"))."
      echo "Open: http://localhost:7860"
      exit 0
    fi
    echo "Starting LifeVault..."
    cd "$ROOT_DIR"
    nohup ./LIFEVAULT_1/bin/python app.py >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 2
    if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "✅  LifeVault running (PID $(cat "$PID_FILE"))"
      echo "    Open: http://localhost:7860"
      echo "    Logs: tail -f $LOG_FILE"
    else
      echo "❌  LifeVault failed to start. Check: $LOG_FILE"
      rm -f "$PID_FILE"
      exit 1
    fi
    ;;
  stop)
    if [ -f "$PID_FILE" ]; then
      kill "$(cat "$PID_FILE")" 2>/dev/null && echo "Stopped." || echo "Process not found."
      rm -f "$PID_FILE"
    else
      echo "LifeVault is not running."
    fi
    ;;
  restart)
    $0 stop
    sleep 1
    $0 start
    ;;
  status)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "🟢  Running (PID $(cat "$PID_FILE")) → http://localhost:7860"
    else
      echo "🔴  Not running"
    fi
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac
