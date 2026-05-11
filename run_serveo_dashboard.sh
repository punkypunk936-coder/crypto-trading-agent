#!/bin/zsh
set -euo pipefail

RUNTIME_DIR="${RUNTIME_DIR:-$HOME/Library/Application Support/crypto_trading_agent_runtime}"
ENV_FILE="$RUNTIME_DIR/hosted_dashboard.env"
URL_FILE="$RUNTIME_DIR/serveo_dashboard.url"
PORT="${PORT:-8091}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  . "$ENV_FILE"
  set +a
fi

mkdir -p "$RUNTIME_DIR/logs"

/usr/bin/ssh \
  -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=30 \
  -R "80:localhost:${PORT}" \
  serveo.net 2>&1 | while IFS= read -r line; do
    print -r -- "$line"
    clean="$(print -r -- "$line" | /usr/bin/sed -E 's/\x1B\[[0-9;]*[A-Za-z]//g')"
    if [[ "$clean" == *"Forwarding HTTP traffic from https://"* ]]; then
      url="${clean##*Forwarding HTTP traffic from }"
      url="${url%%[[:space:]]*}"
      if [[ "$url" == https://* ]]; then
        print -r -- "$url" > "$URL_FILE"
      fi
    fi
  done
