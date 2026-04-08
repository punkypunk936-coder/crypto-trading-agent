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
PYTHON_BIN="$AGENT_DIR/.venv/bin/python3"
PIP_BIN="$AGENT_DIR/.venv/bin/pip"

select_python_bootstrap() {
    local explicit="${PYTHON_BOOTSTRAP_BIN:-}"
    local candidates=()
    if [ -n "$explicit" ]; then
        candidates+=("$explicit")
    fi
    candidates+=(python3.14 python3.13 python3.12 python3.11 python3.10 python3)

    local candidate=""
    local resolved=""
    for candidate in "${candidates[@]}"; do
        if [ -x "$candidate" ]; then
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
    echo "Python 3.10+ is required to run the trading agent." >&2
    exit 1
}

# ── Load .env so secrets are available to Python ───────
if [ -f "$AGENT_DIR/.env" ]; then
    set -a                          # auto-export every variable we source
    # shellcheck disable=SC1090
    source "$AGENT_DIR/.env"
    set +a
fi

DATA_ROOT="${DATA_DIR:-$AGENT_DIR}"
CONTROL_FILE="$DATA_ROOT/control.json"
KILL_FILE="$DATA_ROOT/KILL"

# ── Setup ──────────────────────────────────────────────
mkdir -p "$LOG_DIR"

if [ -x "$PYTHON_BIN" ] && ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
    rm -rf "$AGENT_DIR/.venv"
fi

if [ ! -x "$PYTHON_BIN" ]; then
    "$PYTHON_BOOTSTRAP_BIN" -m venv "$AGENT_DIR/.venv"
fi

if ! "$PYTHON_BIN" -c "import lighter" >/dev/null 2>&1; then
    "$PIP_BIN" install -q -r "$AGENT_DIR/requirements.txt"
fi

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
    CONTROL_ACTIVE="0"
    if [ -f "$CONTROL_FILE" ]; then
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

    if [ "$CONTROL_ACTIVE" = "1" ] || [ -f "$KILL_FILE" ]; then
        log "Kill control is active; waiting before restarting the agent..."
        sleep "$BASE_BACKOFF"
        continue
    fi

    TODAY=$(date '+%Y-%m-%d')
    LOG_FILE="$LOG_DIR/agent_${TODAY}.log"

    log "Starting agent (attempt $((RESTART_COUNT + 1))/$MAX_RESTARTS)..."

    # Run the agent, tee output to both console and dated log file
    START_TIME=$(date +%s)

    "$PYTHON_BIN" main.py $MODE 2>&1 | tee -a "$LOG_FILE" || true

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
