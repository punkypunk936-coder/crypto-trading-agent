#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_LABEL="com.manvinder.crypto-trading-agent"
DASHBOARD_LABEL="com.manvinder.crypto-trading-dashboard"
RUNTIME_DIR="$HOME/Library/Application Support/crypto_trading_agent_runtime"

echo "=== agent launchd status ==="
launchctl print "gui/$(id -u)/$AGENT_LABEL" || true
echo ""
echo "=== dashboard launchd status ==="
launchctl print "gui/$(id -u)/$DASHBOARD_LABEL" || true
echo ""
echo "=== sleep prevention ==="
pmset -g batt 2>/dev/null || true
pgrep -af "caffeinate.*crypto_trading_agent_runtime/.venv/bin/python" || echo "No trading-agent caffeinate wrapper found"
echo ""
echo "=== stdout tail ==="
tail -n 40 "$RUNTIME_DIR/logs/launchd.stdout.log" 2>/dev/null || true
echo ""
echo "=== stderr tail ==="
tail -n 40 "$RUNTIME_DIR/logs/launchd.stderr.log" 2>/dev/null || true
echo ""
echo "=== dashboard stdout tail ==="
tail -n 40 "$RUNTIME_DIR/logs/dashboard.stdout.log" 2>/dev/null || true
echo ""
echo "=== dashboard stderr tail ==="
tail -n 40 "$RUNTIME_DIR/logs/dashboard.stderr.log" 2>/dev/null || true
