#!/bin/bash
# ──────────────────────────────────────────────────────────
#  setup_daemon.sh — Install the trading agent as a
#  macOS background service that runs 24/7
#
#  Usage:
#    chmod +x setup_daemon.sh
#    ./setup_daemon.sh
# ──────────────────────────────────────────────────────────

set -euo pipefail

AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_NAME="com.punky.trading-agent.plist"
PLIST_SRC="$AGENT_DIR/$PLIST_NAME"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"
DOMAIN="gui/$(id -u)"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   🤖  Trading Agent — Daemon Setup                    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Create logs directory FIRST (launchd needs it before it starts)
mkdir -p "$AGENT_DIR/logs"
touch "$AGENT_DIR/logs/launchd_stdout.log"
touch "$AGENT_DIR/logs/launchd_stderr.log"

# Create LaunchAgents directory if needed
mkdir -p "$HOME/Library/LaunchAgents"

# Unload existing if present (try both old and new method)
echo "→ Removing any existing agent..."
launchctl bootout "$DOMAIN/$PLIST_NAME" 2>/dev/null || true
launchctl unload "$PLIST_DST" 2>/dev/null || true
sleep 1

# Copy plist
echo "→ Installing LaunchAgent plist..."
cp "$PLIST_SRC" "$PLIST_DST"

# Fix permissions (required for launchd)
chmod 644 "$PLIST_DST"

# Load using the modern method first, fall back to legacy
echo "→ Starting agent..."
if launchctl bootstrap "$DOMAIN" "$PLIST_DST" 2>/dev/null; then
    echo "→ Loaded via bootstrap (modern macOS)"
elif launchctl load "$PLIST_DST" 2>/dev/null; then
    echo "→ Loaded via load (legacy macOS)"
else
    echo ""
    echo "⚠️  LaunchAgent failed. Using nohup fallback..."
    echo "→ Starting agent with nohup (survives Terminal close)..."
    cd "$AGENT_DIR"
    nohup bash run_forever.sh --dry-run >> logs/launchd_stdout.log 2>> logs/launchd_stderr.log &
    AGENT_PID=$!
    echo "$AGENT_PID" > "$AGENT_DIR/.agent.pid"
    echo "→ Agent started with PID $AGENT_PID"
    echo ""
    echo "✅ Agent is running in the background!"
    echo ""
    echo "  📊 View live logs:    tail -f $AGENT_DIR/logs/agent_$(date +%Y-%m-%d).log"
    echo "  🛑 Stop agent:        kill \$(cat $AGENT_DIR/.agent.pid)"
    echo "  🔍 Check status:      ps aux | grep run_forever"
    echo ""
    echo "  ⚠️  Note: with nohup, the agent will NOT auto-start on reboot."
    echo "  You'll need to run this script again after restarting your Mac."
    echo ""
    exit 0
fi

sleep 2

# Verify
if launchctl list 2>/dev/null | grep -q "com.punky.trading-agent"; then
    echo ""
    echo "✅ Agent is now running in the background!"
    echo ""
    echo "  📊 View live logs:    tail -f $AGENT_DIR/logs/agent_$(date +%Y-%m-%d).log"
    echo "  🛑 Stop agent:        launchctl bootout $DOMAIN/$PLIST_NAME"
    echo "  ▶️  Restart agent:     launchctl kickstart -k $DOMAIN/$PLIST_NAME"
    echo "  🔍 Check status:      launchctl list | grep punky"
    echo ""
    echo "  The agent will auto-start on every Mac reboot."
    echo "  If it crashes, it auto-restarts with backoff."
    echo ""
else
    echo ""
    echo "⚠️  LaunchAgent may not have registered. Falling back to nohup..."
    cd "$AGENT_DIR"
    nohup bash run_forever.sh --dry-run >> logs/launchd_stdout.log 2>> logs/launchd_stderr.log &
    AGENT_PID=$!
    echo "$AGENT_PID" > "$AGENT_DIR/.agent.pid"
    echo ""
    echo "✅ Agent started with PID $AGENT_PID (nohup fallback)"
    echo ""
    echo "  📊 View live logs:    tail -f $AGENT_DIR/logs/agent_$(date +%Y-%m-%d).log"
    echo "  🛑 Stop agent:        kill \$(cat $AGENT_DIR/.agent.pid)"
    echo ""
fi
