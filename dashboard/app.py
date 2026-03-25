"""
dashboard/app.py — Trading agent dashboard.

Works in two modes:
  LOCAL:  reads state.json + trades_log.csv written by the agent
  REMOTE: receives state via POST /api/push (agent pushes each cycle)

Start locally:   python3 dashboard/app.py
Deploy (Railway): set PORT env var, agent pushes to your Railway URL
"""

import json
import os
import csv
import threading
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, jsonify, request, abort

BASE    = Path(__file__).parent.parent
STATE   = BASE / "state.json"
LOG     = BASE / "trades_log.csv"

# Secret token for push endpoint (set DASHBOARD_TOKEN env var for security)
PUSH_TOKEN = os.environ.get("DASHBOARD_TOKEN", "")

app = Flask(__name__, template_folder="templates", static_folder="static")

# In-memory store for remote-pushed state
_remote_state = {"state": None, "trades": [], "stats": {}}
_lock = threading.Lock()


def _load_state_local() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return {
        "status": "offline", "last_cycle": None, "cycle_number": 0,
        "portfolio_usd": 0, "available_usd": 0, "positions": [],
        "signals": {}, "pending_orders": [], "sentiment": {}, "mode": "unknown",
    }


def _load_trades_local() -> list:
    if not LOG.exists():
        return []
    try:
        with open(LOG, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _calc_stats(trades: list) -> dict:
    if not trades:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl": 0, "avg_win": 0, "avg_loss": 0, "best": 0, "worst": 0}
    closed = [t for t in trades if t.get("exit_price") and float(t.get("exit_price", 0)) > 0]
    if not closed:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl": 0, "avg_win": 0, "avg_loss": 0, "best": 0, "worst": 0}
    pnls   = [float(t.get("pnl_usd", 0)) for t in closed]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {
        "total":     len(closed),
        "wins":      len(wins),
        "losses":    len(losses),
        "win_rate":  round(len(wins) / len(closed) * 100, 1),
        "total_pnl": round(sum(pnls), 2),
        "avg_win":   round(sum(wins)   / len(wins)   if wins   else 0, 2),
        "avg_loss":  round(sum(losses) / len(losses) if losses else 0, 2),
        "best":      round(max(pnls), 2),
        "worst":     round(min(pnls), 2),
    }


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/state")
def api_state():
    # If remote state has been pushed, use that
    with _lock:
        if _remote_state["state"] is not None:
            return jsonify({
                "state":       _remote_state["state"],
                "trades":      _remote_state["trades"],
                "stats":       _remote_state["stats"],
                "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

    # Otherwise read local files
    state  = _load_state_local()
    trades = _load_trades_local()
    stats  = _calc_stats(trades)
    return jsonify({
        "state":       state,
        "trades":      trades[-50:][::-1],
        "stats":       stats,
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/api/push", methods=["POST"])
def api_push():
    """Agent pushes state here each cycle (for remote/Railway deployment)."""
    # Token check
    if PUSH_TOKEN:
        token = request.headers.get("X-Token", "")
        if token != PUSH_TOKEN:
            abort(403, "Invalid token")

    data = request.get_json(silent=True)
    if not data or "state" not in data:
        abort(400, "Missing state in payload")

    trades = data.get("trades", [])
    stats  = _calc_stats(trades)

    with _lock:
        _remote_state["state"]  = data["state"]
        _remote_state["trades"] = trades[-50:][::-1]
        _remote_state["stats"]  = stats

    return jsonify({"ok": True, "cycle": data["state"].get("cycle_number", 0)})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\n  Trading Agent Dashboard")
    print(f"  Open in your browser: http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
