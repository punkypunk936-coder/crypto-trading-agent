#!/bin/zsh
set -euo pipefail

AGENT_LABEL="com.manvinder.crypto-trading-agent"
DASHBOARD_LABEL="com.manvinder.crypto-trading-dashboard"
AGENT_PLIST_DEST="$HOME/Library/LaunchAgents/$AGENT_LABEL.plist"
DASHBOARD_PLIST_DEST="$HOME/Library/LaunchAgents/$DASHBOARD_LABEL.plist"

launchctl bootout "gui/$(id -u)" "$AGENT_PLIST_DEST" 2>/dev/null || true
launchctl bootout "gui/$(id -u)" "$DASHBOARD_PLIST_DEST" 2>/dev/null || true
echo "Stopped $AGENT_LABEL"
echo "Stopped $DASHBOARD_LABEL"
