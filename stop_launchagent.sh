#!/bin/zsh
set -euo pipefail

LABEL="com.manvinder.crypto-trading-agent"
PLIST_DEST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$(id -u)" "$PLIST_DEST" 2>/dev/null || true
echo "Stopped $LABEL"
