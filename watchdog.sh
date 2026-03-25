#!/bin/bash

# Crypto Trading Agent Watchdog
# Monitors agent process and state.json for signs of failure
# Restarts agent if it's dead or state.json is stale
# Runs every 5 minutes via cron

set -e

AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_SCRIPT="${AGENT_DIR}/run_forever.sh"
STATE_FILE="${AGENT_DIR}/state.json"
LOG_FILE="${AGENT_DIR}/logs/watchdog.log"
MAX_STATE_AGE_SECONDS=600  # 10 minutes

# Ensure logs directory exists
mkdir -p "${AGENT_DIR}/logs"

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "${LOG_FILE}"
}

# Check if the agent process is running
check_process() {
    if pgrep -f "run_forever.sh" > /dev/null 2>&1; then
        return 0  # Process is running
    else
        return 1  # Process is not running
    fi
}

# Check if state.json exists and is recent
check_state_freshness() {
    if [ ! -f "${STATE_FILE}" ]; then
        log_message "WARN: state.json does not exist"
        return 1  # State file missing
    fi

    # Get the modification time of state.json
    state_mod_time=$(stat -f %m "${STATE_FILE}" 2>/dev/null || stat -c %Y "${STATE_FILE}" 2>/dev/null || echo 0)
    current_time=$(date +%s)
    state_age=$((current_time - state_mod_time))

    if [ ${state_age} -gt ${MAX_STATE_AGE_SECONDS} ]; then
        log_message "WARN: state.json is stale (${state_age}s old, max: ${MAX_STATE_AGE_SECONDS}s)"
        return 1  # State is stale
    else
        log_message "OK: state.json is fresh (${state_age}s old)"
        return 0  # State is fresh
    fi
}

# Send macOS notification
send_notification() {
    local title="$1"
    local message="$2"

    if command -v osascript &> /dev/null; then
        osascript -e "display notification \"${message}\" with title \"${title}\"" 2>/dev/null || true
    fi
}

# Restart the agent
restart_agent() {
    log_message "ACTION: Restarting agent..."

    # Kill any existing processes
    pkill -f "run_forever.sh" 2>/dev/null || true
    sleep 2

    # Start the agent in background
    cd "${AGENT_DIR}"
    nohup "${AGENT_SCRIPT}" > "${AGENT_DIR}/logs/agent_restart.log" 2>&1 &
    local new_pid=$!

    log_message "ACTION: Agent restarted with PID ${new_pid}"
    send_notification "Trading Agent Restarted" "Process restarted with PID ${new_pid}"
}

# Main watchdog logic
main() {
    log_message "=== Watchdog cycle started ==="

    if ! check_process; then
        log_message "ALERT: Agent process is not running!"
        send_notification "Trading Agent Alert" "Process is dead - restarting"
        restart_agent
    elif ! check_state_freshness; then
        log_message "ALERT: Agent state is stale - process may be frozen!"
        send_notification "Trading Agent Alert" "State is stale - restarting"
        restart_agent
    else
        log_message "OK: Agent process and state are healthy"
    fi

    log_message "=== Watchdog cycle completed ==="
}

main
