#!/bin/bash
# ──────────────────────────────────────────────────────────
#  CRYPTO TRADING AGENT — Live Trading
#  Double-click this file to start with REAL money.
#  Make sure your .env file has your wallet keys first!
# ──────────────────────────────────────────────────────────

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
echo "║   🤖  CRYPTO TRADING AGENT  — LIVE TRADING  🔴   ║"
echo "║   Real money mode. Ensure .env is configured.    ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Check .env exists
if [ ! -f ".env" ]; then
  echo "❌  ERROR: .env file not found!"
  echo ""
  echo "   You need to create a .env file with your wallet keys."
  echo "   Open .env.example, fill in your details, and save it as .env"
  echo ""
  echo "Press any key to close."
  read -n 1
  exit 1
fi

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

echo "⚠️  LIVE MODE — real orders will be placed."
echo "   Press Ctrl+C at any time to stop."
echo ""
read -p "   Type YES to confirm and start: " confirm
echo ""

if [ "$confirm" = "YES" ]; then
  "$PYTHON_BIN" main.py --live
else
  echo "Cancelled. Run again when ready."
fi

echo ""
echo "Agent stopped. Press any key to close this window."
read -n 1
