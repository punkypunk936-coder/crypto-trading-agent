#!/bin/bash
# ──────────────────────────────────────────────────────────
#  CRYPTO TRADING AGENT — Paper Trading (Safe Mode)
#  Double-click this file to start.
#  No real money is used. Perfect for testing.
# ──────────────────────────────────────────────────────────

# Move into the folder this script lives in
cd "$(dirname "$0")"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   🤖  CRYPTO TRADING AGENT  — PAPER TRADING  🟡  ║"
echo "║   No real money. Safe to test.                   ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Install / update dependencies automatically
echo "→ Checking dependencies..."
pip3 install -r requirements.txt -q
echo "✅ Dependencies ready."
echo ""

# Run in dry-run (paper trading) mode
echo "→ Starting agent in PAPER TRADING mode..."
echo "   Press Ctrl+C at any time to stop."
echo ""
python3 main.py --dry-run

echo ""
echo "Agent stopped. Press any key to close this window."
read -n 1
