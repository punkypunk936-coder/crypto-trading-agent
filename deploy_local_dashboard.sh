#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_DIR="${LOCAL_DASHBOARD_RUNTIME_DIR:-$HOME/Library/Application Support/crypto_trading_agent_runtime}"
DASHBOARD_LABEL="${LOCAL_DASHBOARD_LABEL:-com.manvinder.crypto-trading-dashboard}"
PORT="${TRADING_DASHBOARD_PORT:-8080}"
VERIFY_URL="${LOCAL_DASHBOARD_VERIFY_URL:-http://127.0.0.1:$PORT/}"

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

if launchctl print "gui/$(id -u)/$DASHBOARD_LABEL" >/dev/null 2>&1; then
  launchctl kickstart -k "gui/$(id -u)/$DASHBOARD_LABEL"
else
  echo "Dashboard launchd service is not loaded: $DASHBOARD_LABEL" >&2
  echo "Run ./install_launchagent.sh once if this is a fresh machine." >&2
fi

for attempt in {1..20}; do
  if curl -fsS "$VERIFY_URL" >/tmp/local_dashboard_verify.html 2>/dev/null; then
    echo "Local dashboard updated: $VERIFY_URL"
    if grep -q "renderMarketMapSummary" /tmp/local_dashboard_verify.html; then
      echo "Verified served dashboard bundle."
    fi
    exit 0
  fi
  sleep 1
done

echo "Local dashboard sync completed, but verification timed out: $VERIFY_URL" >&2
exit 1
