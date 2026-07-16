#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
PYTHON_BIN="${TRADING_AGENT_PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
PORT="${TRADING_DASHBOARD_PORT:-8080}"

if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

mkdir -p "$LOG_DIR"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1
export PORT

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python runtime not found: $PYTHON_BIN" >&2
  exit 1
fi

# The local macOS service must stay single-process. A long-lived Gunicorn
# master can crash replacement workers when Objective-C-backed dependencies
# have initialized threads, leaving the dashboard master alive but unusable.
exec "$PYTHON_BIN" "$SCRIPT_DIR/dashboard/app.py"
