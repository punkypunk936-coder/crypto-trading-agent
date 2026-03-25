#!/bin/bash
# ──────────────────────────────────────────────────────────
#  run_forever.sh — Keeps the trading agent alive 24/7
#
#  Features:
#    • Auto-restarts on crash with exponential backoff
#    • Logs everything to logs/ with daily rotation
#    • Caps restarts to prevent runaway loops
#    • Sends a macOS notification on crash (optional)
#
#  Usage:
#    chmod +x run_forever.sh
#    ./run_forever.sh              # dry-run (default)
#    ./run_forever.sh --live       # live trading
# ──────────────────────────────────────────────────────────

set -euo pipefail

AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$AGENT_DIR"

# ── Config ─────────────────────────────────────────────
MODE="${1:---dry-run}"              # --dry-run or --live
MAX_RESTARTS=50                     # max restarts before giving up
BASE_BACKOFF=5                      # starting backoff in seconds
MAX_BACKOFF=300                     # cap at 5 minutes
LOG_DIR="$AGENT_DIR/logs"

# ── Setup ──────────────────────────────────────────────
mkdir -p "$LOG_DIR"

RESTART_COUNT=0
BACKOFF=$BASE_BACKOFF

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/runner.log"
}

notify_crash() {
    # macOS notification (silent fail if not on mac)
    osascript -e "display notification \"Agent crashed (#$RESTART_COUNT). Restarting in ${BACKOFF}s...\" with title \"Trading Agent\" sound name \"Basso\"" 2>/dev/null || true
}

# ── Main Loop ──────────────────────────────────────────
log "═══════════════════════════════════════════════════"
log "  Trading Agent Runner — Mode: $MODE"
log "  Max restarts: $MAX_RESTARTS"
log "  PID: $$"
log "═══════════════════════════════════════════════════"

while true; do
    TODAY=$(date '+%Y-%m-%d')
    LOG_FILE="$LOG_DIR/agent_${TODAY}.log"

    log "Starting agent (attempt $((RESTART_COUNT + 1))/$MAX_RESTARTS)..."

    # Run the agent, tee output to both console and dated log file
    START_TIME=$(date +%s)

    python3 main.py $MODE 2>&1 | tee -a "$LOG_FILE" || true

    EXIT_CODE=${PIPESTATUS[0]:-$?}
    END_TIME=$(date +%s)
    UPTIME=$((END_TIME - START_TIME))

    log "Agent exited with code $EXIT_CODE after ${UPTIME}s"

    # If it ran for more than 10 minutes, reset backoff (was a healthy run)
    if [ "$UPTIME" -gt 600 ]; then
        RESTART_COUNT=0
        BACKOFF=$BASE_BACKOFF
        log "Was running >10min — resetting backoff"
    else
        RESTART_COUNT=$((RESTART_COUNT + 1))
    fi

    # Check restart limit
    if [ "$RESTART_COUNT" -ge "$MAX_RESTARTS" ]; then
        log "╔══════════════════════════════════════════════════╗"
        log "║  FATAL: Max restarts ($MAX_RESTARTS) reached.   ║"
        log "║  Agent is NOT restarting. Check logs.            ║"
        log "╚══════════════════════════════════════════════════╝"
        osascript -e "display notification \"Agent stopped after $MAX_RESTARTS crashes. Check logs!\" with title \"Trading Agent STOPPED\" sound name \"Sosumi\"" 2>/dev/null || true
        exit 1
    fi

    notify_crash
    log "Restarting in ${BACKOFF}s... (backoff)"
    sleep "$BACKOFF"

    # Exponential backoff with cap
    BACKOFF=$((BACKOFF * 2))
    if [ "$BACKOFF" -gt "$MAX_BACKOFF" ]; then
        BACKOFF=$MAX_BACKOFF
    fi
done
