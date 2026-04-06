#!/bin/bash
# ──────────────────────────────────────────────────────────
#  CRYPTO TRADING AGENT — Paper Trading (Safe Mode)
#  Double-click this file to start.
#  No real money is used. Perfect for testing.
# ──────────────────────────────────────────────────────────

# Move into the folder this script lives in
cd "$(dirname "$0")"
PYTHON_BIN=".venv/bin/python3"
PIP_BIN=".venv/bin/pip"

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
  echo "❌ Python 3.10+ is required."
  read -n 1
  exit 1
}

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   🤖  CRYPTO TRADING AGENT  — PAPER TRADING  🟡  ║"
echo "║   No real money. Safe to test.                   ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Install / update dependencies automatically
echo "→ Checking dependencies..."
if [ -x "$PYTHON_BIN" ] && ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
  rm -rf .venv
fi
if [ ! -x "$PYTHON_BIN" ]; then
  "$PYTHON_BOOTSTRAP_BIN" -m venv .venv
fi
"$PIP_BIN" install -r requirements.txt -q
echo "✅ Dependencies ready."
echo ""

# Run in dry-run (paper trading) mode
echo "→ Starting agent in PAPER TRADING mode..."
echo "   Press Ctrl+C at any time to stop."
echo ""
"$PYTHON_BIN" main.py --dry-run

echo ""
echo "Agent stopped. Press any key to close this window."
read -n 1
