#!/bin/bash
# ──────────────────────────────────────────────────────────
#  CRYPTO TRADING AGENT — Live Trading
#  Double-click this file to start with REAL money.
#  Make sure your .env file has your wallet keys first!
# ──────────────────────────────────────────────────────────

cd "$(dirname "$0")"

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
pip3 install -r requirements.txt -q
echo "✅ Dependencies ready."
echo ""

echo "⚠️  LIVE MODE — real orders will be placed."
echo "   Press Ctrl+C at any time to stop."
echo ""
read -p "   Type YES to confirm and start: " confirm
echo ""

if [ "$confirm" = "YES" ]; then
  python3 main.py --live
else
  echo "Cancelled. Run again when ready."
fi

echo ""
echo "Agent stopped. Press any key to close this window."
read -n 1
