#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
MODE="${TRADING_AGENT_MODE:-paper}"
PYTHON_BIN="${TRADING_AGENT_PYTHON:-$SCRIPT_DIR/.venv/bin/python}"

mkdir -p "$LOG_DIR"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python runtime not found: $PYTHON_BIN" >&2
  exit 1
fi

case "$MODE" in
  live)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/main.py" --live
    ;;
  paper|dry-run|dry_run)
    exec "$PYTHON_BIN" "$SCRIPT_DIR/main.py" --dry-run
    ;;
  *)
    echo "Unsupported TRADING_AGENT_MODE: $MODE" >&2
    exit 1
    ;;
esac
