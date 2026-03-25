#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LABEL="com.manvinder.crypto-trading-agent"
RUNTIME_DIR="$HOME/Library/Application Support/crypto_trading_agent_runtime"

echo "=== launchd status ==="
launchctl print "gui/$(id -u)/$LABEL" || true
echo ""
echo "=== stdout tail ==="
tail -n 40 "$RUNTIME_DIR/logs/launchd.stdout.log" 2>/dev/null || true
echo ""
echo "=== stderr tail ==="
tail -n 40 "$RUNTIME_DIR/logs/launchd.stderr.log" 2>/dev/null || true
