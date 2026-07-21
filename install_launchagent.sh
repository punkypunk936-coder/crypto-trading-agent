#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_LABEL="com.manvinder.crypto-trading-agent"
DASHBOARD_LABEL="com.manvinder.crypto-trading-dashboard"
LEGACY_LABEL="com.punky.trading-agent"
RUNTIME_DIR="$HOME/Library/Application Support/crypto_trading_agent_runtime"
AGENT_PLIST_SRC="$SCRIPT_DIR/launchd/$AGENT_LABEL.plist"
AGENT_PLIST_DEST="$HOME/Library/LaunchAgents/$AGENT_LABEL.plist"
DB_PLIST_SRC="$SCRIPT_DIR/launchd/$DASHBOARD_LABEL.plist"
DB_PLIST_DEST="$HOME/Library/LaunchAgents/$DASHBOARD_LABEL.plist"
LOG_DIR="$RUNTIME_DIR/logs"

select_python_bootstrap() {
  local explicit="${PYTHON_BOOTSTRAP_BIN:-}"
  local candidates=()
  if [[ -n "$explicit" ]]; then
    candidates+=("$explicit")
  fi
  candidates+=(python3.14 python3.13 python3.12 python3.11 python3.10 python3)

  local candidate=""
  local resolved=""
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      resolved="$candidate"
    elif command -v "$candidate" >/dev/null 2>&1; then
      resolved="$(command -v "$candidate")"
    else
      continue
    fi
    if "$resolved" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
    then
      printf '%s\n' "$resolved"
      return 0
    fi
  done
  return 1
}

PYTHON_BOOTSTRAP_BIN="$(select_python_bootstrap)" || {
  echo "Python 3.10+ is required to install the trading agent runtime." >&2
  exit 1
}

mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$LOG_DIR"
mkdir -p "$RUNTIME_DIR"

rsync -a \
  --exclude ".git" \
  --exclude ".env" \
  --exclude ".agent.pid" \
  --exclude ".venv" \
  --exclude ".venv.py39.backup.1774870801" \
  --exclude "__pycache__" \
  --exclude "logs" \
  --exclude "checkpoints.db" \
  --exclude "state.json" \
  --exclude "trades_log.csv" \
  --exclude "trade_memory.json" \
  --exclude "decision_dataset.jsonl" \
  --exclude "feature_store.jsonl" \
  --exclude "trade_dataset.jsonl" \
  --exclude "precision_lab_report.json" \
  --exclude "playbook_distiller_report.json" \
  --exclude "control.json" \
  --exclude "dashboard_snapshot.json" \
  --exclude "KILL" \
  --exclude "netlify-dashboard/node_modules" \
  "$SCRIPT_DIR/" "$RUNTIME_DIR/"

if [[ -x "$RUNTIME_DIR/.venv/bin/python" ]] && ! "$RUNTIME_DIR/.venv/bin/python" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
  rm -rf "$RUNTIME_DIR/.venv"
fi

if [[ ! -x "$RUNTIME_DIR/.venv/bin/python" ]]; then
  "$PYTHON_BOOTSTRAP_BIN" -m venv "$RUNTIME_DIR/.venv"
fi
"$RUNTIME_DIR/.venv/bin/python" -m pip install -q -r "$RUNTIME_DIR/requirements.txt"

: > "$LOG_DIR/launchd.stdout.log"
: > "$LOG_DIR/launchd.stderr.log"
: > "$LOG_DIR/dashboard.stdout.log"
: > "$LOG_DIR/dashboard.stderr.log"

cp "$AGENT_PLIST_SRC" "$AGENT_PLIST_DEST"
cp "$DB_PLIST_SRC" "$DB_PLIST_DEST"

launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/$LEGACY_LABEL.plist" 2>/dev/null || true
launchctl disable "gui/$(id -u)/$LEGACY_LABEL" 2>/dev/null || true

for LABEL in "$AGENT_LABEL" "$DASHBOARD_LABEL"; do
  launchctl enable "gui/$(id -u)/$LABEL"
done
launchctl bootout "gui/$(id -u)" "$AGENT_PLIST_DEST" 2>/dev/null || true
launchctl bootout "gui/$(id -u)" "$DB_PLIST_DEST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$AGENT_PLIST_DEST"
launchctl bootstrap "gui/$(id -u)" "$DB_PLIST_DEST"
launchctl kickstart -k "gui/$(id -u)/$AGENT_LABEL"
launchctl kickstart -k "gui/$(id -u)/$DASHBOARD_LABEL"

echo "Installed and started $AGENT_LABEL"
echo "Installed and started $DASHBOARD_LABEL"
echo "Status: launchctl print gui/$(id -u)/$AGENT_LABEL"
echo "Status: launchctl print gui/$(id -u)/$DASHBOARD_LABEL"
echo "Logs: $LOG_DIR/launchd.stdout.log, $LOG_DIR/launchd.stderr.log, $LOG_DIR/dashboard.stdout.log, $LOG_DIR/dashboard.stderr.log"
echo "Runtime copy: $RUNTIME_DIR"
