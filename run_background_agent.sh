#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
MODE="${TRADING_AGENT_MODE:-paper}"
PYTHON_BIN="${TRADING_AGENT_PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
KEEP_AWAKE="${TRADING_AGENT_KEEP_AWAKE:-1}"
CAFFEINATE_FLAGS="${TRADING_AGENT_CAFFEINATE_FLAGS:--dimsu}"

if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

DATA_ROOT="${DATA_DIR:-$SCRIPT_DIR}"
CONTROL_FILE="$DATA_ROOT/control.json"
KILL_FILE="$DATA_ROOT/KILL"

mkdir -p "$LOG_DIR"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python runtime not found: $PYTHON_BIN" >&2
  exit 1
fi

while true; do
  CONTROL_ACTIVE="0"
  if [[ -f "$CONTROL_FILE" ]]; then
    CONTROL_ACTIVE="$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path
path = Path(r'''$CONTROL_FILE''')
try:
    data = json.loads(path.read_text())
    print("1" if ((data.get("kill") or {}).get("active")) else "0")
except Exception:
    print("0")
PY
)"
  fi

  if [[ "$CONTROL_ACTIVE" == "1" || -f "$KILL_FILE" ]]; then
    echo "Kill control is active; waiting for it to be cleared..." >&2
    sleep 15
    continue
  fi
  break
done

case "$MODE" in
  live)
    AGENT_ARGS=("$PYTHON_BIN" "$SCRIPT_DIR/main.py" --live)
    ;;
  paper|dry-run|dry_run)
    AGENT_ARGS=("$PYTHON_BIN" "$SCRIPT_DIR/main.py" --dry-run)
    ;;
  *)
    echo "Unsupported TRADING_AGENT_MODE: $MODE" >&2
    exit 1
    ;;
esac

if [[ "$KEEP_AWAKE" == "1" ]] && command -v caffeinate >/dev/null 2>&1; then
  echo "Sleep prevention enabled via caffeinate ${CAFFEINATE_FLAGS}" >&2
  exec caffeinate ${=CAFFEINATE_FLAGS} "${AGENT_ARGS[@]}"
fi

exec "${AGENT_ARGS[@]}"
