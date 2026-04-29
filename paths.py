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
DECISION_DATASET_JSONL = DATA_DIR / "decision_dataset.jsonl"    # Per-cycle decision dataset, including FLAT decisions
FEATURE_STORE_JSONL = DATA_DIR / "feature_store.jsonl"          # Model-ready feature rows derived from decisions and closed trades
DECISION_REVIEW_REPORT_JSON = DATA_DIR / "decision_review_report.json"  # Labeled review of executed vs missed decisions
CHALLENGER_MODEL_JSON = DATA_DIR / "challenger_model_report.json"       # Champion/challenger expectancy report
DAILY_MARKET_MAP_JSON = DATA_DIR / "daily_market_map.json"      # Operator-owned daily key-level / thesis map
TRADE_REVIEWS_JSON = DATA_DIR / "trade_reviews.json"            # Operator review labels for closed trades
MACRO_EVENTS_JSON = DATA_DIR / "macro_events.json"              # Optional macro event calendar for narrative gating
DASHBOARD_STATE_SYNC_REPO = DATA_DIR / ".dashboard_state_sync"  # Local clone/worktree for hosted dashboard fallback sync
MARKET_CAP_UNIVERSE_JSON = DATA_DIR / "market_cap_universe.json"  # Cached Hyperliquid scout universe from market-cap filter
ASSET_DOSSIERS_JSON = DATA_DIR / "asset_dossiers.json"          # Living per-asset trading dossiers
MISSED_MOVE_REPORT_JSON = DATA_DIR / "missed_move_report.json"  # Rich review of obvious winners the bot skipped
LLM_REFEREE_REPORT_JSON = DATA_DIR / "llm_referee_report.json"  # Latest structured LLM referee verdicts
PLAYBOOK_DISTILLER_REPORT_JSON = DATA_DIR / "playbook_distiller_report.json"  # Rolling rewrite of what is working by asset/regime
PROACTIVE_TRADER_REPORT_JSON = DATA_DIR / "proactive_trader_report.json"  # Research brain: scout book, theses, read-through, starter basket, forecast calibration
THESIS_LEDGER_JSONL = DATA_DIR / "thesis_ledger.jsonl"      # Persistent structured pre-trade theses
FORECAST_LEDGER_JSONL = DATA_DIR / "forecast_ledger.jsonl"  # Persistent probability forecasts and later outcomes

# ── Logs directory ────────────────────────────────────────────────────────────
LOGS_DIR = DATA_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
