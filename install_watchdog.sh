#!/bin/bash

# Crypto Trading Agent Watchdog Installer
# Adds the watchdog.sh script to crontab to run every 5 minutes
# Checks for existing entry to avoid duplicates

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCHDOG_SCRIPT="${SCRIPT_DIR}/watchdog.sh"

# Ensure watchdog.sh is executable
if [ ! -x "${WATCHDOG_SCRIPT}" ]; then
    chmod +x "${WATCHDOG_SCRIPT}"
    echo "Made watchdog.sh executable"
fi

# Define the cron job entry
CRON_ENTRY="*/5 * * * * ${WATCHDOG_SCRIPT}"

# Get the current crontab
CURRENT_CRONTAB=$(crontab -l 2>/dev/null || true)

# Check if the watchdog is already installed
if echo "${CURRENT_CRONTAB}" | grep -q "${WATCHDOG_SCRIPT}"; then
    echo "✓ Watchdog is already installed in crontab"
    echo "Entry: ${CRON_ENTRY}"
    exit 0
fi

# Add the new cron job
echo "${CURRENT_CRONTAB}" | crontab -
(echo "${CURRENT_CRONTAB}"; echo "${CRON_ENTRY}") | crontab -

echo "✓ Watchdog successfully installed to crontab"
echo "Entry: ${CRON_ENTRY}"
echo ""
echo "The watchdog will run every 5 minutes and monitor:"
echo "  - Agent process status (run_forever.sh)"
echo "  - state.json freshness (max 10 minutes old)"
echo "  - Auto-restart on failure with macOS notification"
echo ""
echo "View logs at: ${SCRIPT_DIR}/logs/watchdog.log"
