"""
paths.py — Centralised file paths for the trading agent.

On your Mac (local): files sit next to main.py (current behaviour).
On Railway (cloud):  set DATA_DIR=/data  → all state files go to the
                     Railway persistent volume, so they survive restarts
                     and redeploys with zero data loss.

Usage everywhere:
    from paths import DATA_DIR, CHECKPOINTS_DB, TRADE_MEMORY, STATE_JSON, TRADES_CSV, KILL_FILE
"""

import os
from pathlib import Path

# ── Root of the codebase (where main.py lives) ────────────────────────────────
CODE_ROOT = Path(__file__).parent

# ── Data directory ────────────────────────────────────────────────────────────
# Local:   same folder as the code (original behaviour)
# Railway: set DATA_DIR=/data in Railway env vars → persistent volume
_data_env = os.environ.get("DATA_DIR", "")
DATA_DIR = Path(_data_env) if _data_env else CODE_ROOT

# Create it if running in cloud and it doesn't exist yet
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Persistent state files ────────────────────────────────────────────────────
CHECKPOINTS_DB = DATA_DIR / "checkpoints.db"      # SQLite: open positions, orders
TRADE_MEMORY   = DATA_DIR / "trade_memory.json"   # RL learning history
STATE_JSON     = DATA_DIR / "state.json"           # Live dashboard state
TRADES_CSV     = DATA_DIR / "trades_log.csv"       # Full closed-trade log
KILL_FILE      = DATA_DIR / "KILL"                 # Touch to gracefully stop agent
CONTROL_JSON   = DATA_DIR / "control.json"         # Dashboard/operator control state
DASHBOARD_SNAPSHOT_JSON = DATA_DIR / "dashboard_snapshot.json"  # Canonical dashboard payload
TRADE_DATASET_JSONL = DATA_DIR / "trade_dataset.jsonl"          # Structured closed-trade dataset for learning
DAILY_MARKET_MAP_JSON = DATA_DIR / "daily_market_map.json"      # Operator-owned daily key-level / thesis map
TRADE_REVIEWS_JSON = DATA_DIR / "trade_reviews.json"            # Operator review labels for closed trades
MACRO_EVENTS_JSON = DATA_DIR / "macro_events.json"              # Optional macro event calendar for narrative gating
DASHBOARD_STATE_SYNC_REPO = DATA_DIR / ".dashboard_state_sync"  # Local clone/worktree for hosted dashboard fallback sync

# ── Logs directory ────────────────────────────────────────────────────────────
LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
