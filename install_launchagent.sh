#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LABEL="com.manvinder.crypto-trading-agent"
RUNTIME_DIR="$HOME/Library/Application Support/crypto_trading_agent_runtime"
PLIST_SRC="$SCRIPT_DIR/launchd/$LABEL.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$RUNTIME_DIR/logs"

mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$LOG_DIR"
mkdir -p "$RUNTIME_DIR"

rsync -a \
  --exclude "__pycache__" \
  --exclude "logs" \
  --exclude "checkpoints.db" \
  --exclude "state.json" \
  --exclude "trades_log.csv" \
  --exclude "netlify-dashboard/node_modules" \
  "$SCRIPT_DIR/" "$RUNTIME_DIR/"

if [[ ! -x "$RUNTIME_DIR/.venv/bin/python" ]]; then
  /usr/bin/python3 -m venv "$RUNTIME_DIR/.venv"
fi
"$RUNTIME_DIR/.venv/bin/pip" install -q -r "$RUNTIME_DIR/requirements.txt"

cp "$PLIST_SRC" "$PLIST_DEST"

launchctl bootout "gui/$(id -u)" "$PLIST_DEST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DEST"
launchctl enable "gui/$(id -u)/$LABEL"
launchctl kickstart -k "gui/$(id -u)/$LABEL"

echo "Installed and started $LABEL"
echo "Status: launchctl print gui/$(id -u)/$LABEL"
echo "Logs: $LOG_DIR/launchd.stdout.log and $LOG_DIR/launchd.stderr.log"
echo "Runtime copy: $RUNTIME_DIR"
