"""
checkpoint.py — State persistence and crash recovery system for the Trading Agent.

Features:
  - Automatic checkpointing of positions, pending orders, and cycle state
  - Recovery on startup after crashes
  - SQLite-backed for durability
  - Automatic cleanup of old checkpoints

Usage:
  from checkpoint import CheckpointManager
  
  checkpoint = CheckpointManager()
  checkpoint.save(agent_state)  # Save current state
  recovered = checkpoint.load() # Restore after crash
"""

import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from contextlib import contextmanager

from logger import get_logger

log = get_logger("checkpoint")


@dataclass
class AgentCheckpoint:
    """Complete snapshot of agent state for recovery."""
    version: int = 1
    timestamp: str = ""
    cycle_number: int = 0
    portfolio_usd: float = 0.0
    available_usd: float = 0.0
    positions_json: str = "{}"
    pending_orders_json: str = "{}"
    reentry_watches_json: str = "{}"
    daily_pnl_usd: float = 0.0
    daily_trades: int = 0
    last_trade_date: str = ""
    exchange_states_json: str = "{}"


class CheckpointManager:
    """Manages persistent state checkpoints for crash recovery."""
    
    def __init__(self, db_path: Optional[str] = None, max_checkpoints: int = 50):
        if db_path is None:
            from paths import CHECKPOINTS_DB
            db_path = str(CHECKPOINTS_DB)
        self.db_path = Path(db_path)
        self.max_checkpoints = max_checkpoints
        self._init_database()
        
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    cycle_number INTEGER NOT NULL,
                    portfolio_usd REAL NOT NULL,
                    available_usd REAL NOT NULL,
                    positions_json TEXT NOT NULL,
                    pending_orders_json TEXT NOT NULL,
                    reentry_watches_json TEXT NOT NULL,
                    daily_pnl_usd REAL NOT NULL DEFAULT 0.0,
                    daily_trades INTEGER NOT NULL DEFAULT 0,
                    last_trade_date TEXT NOT NULL DEFAULT '',
                    exchange_states_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_checkpoints_time 
                ON checkpoints(created_at DESC)
            """)
            conn.commit()
            log.info(f"Checkpoint database initialized: {self.db_path}")
    
    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def save(self, cycle_number: int, portfolio_usd: float, available_usd: float,
             positions: Dict[str, Any], pending_orders: Dict[str, Any],
             reentry_watches: Dict[str, Any], risk_manager: Any,
             exchange_states: Optional[Dict] = None) -> bool:
        """Save a new checkpoint."""
        try:
            checkpoint = AgentCheckpoint(
                version=1,
                timestamp=datetime.now().isoformat(),
                cycle_number=cycle_number,
                portfolio_usd=portfolio_usd,
                available_usd=available_usd,
                positions_json=json.dumps(self._serialize_positions(positions)),
                pending_orders_json=json.dumps(self._serialize_pending_orders(pending_orders)),
                reentry_watches_json=json.dumps(self._serialize_watches(reentry_watches)),
                daily_pnl_usd=getattr(risk_manager, 'daily_pnl_usd', 0.0),
                daily_trades=getattr(risk_manager, 'daily_trades', 0),
                last_trade_date=getattr(risk_manager, 'last_trade_date', ''),
                exchange_states_json=json.dumps(exchange_states or {})
            )
            
            with self._connection() as conn:
                conn.execute("""
                    INSERT INTO checkpoints 
                    (version, timestamp, cycle_number, portfolio_usd, available_usd,
                     positions_json, pending_orders_json, reentry_watches_json,
                     daily_pnl_usd, daily_trades, last_trade_date, exchange_states_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    checkpoint.version, checkpoint.timestamp, checkpoint.cycle_number,
                    checkpoint.portfolio_usd, checkpoint.available_usd,
                    checkpoint.positions_json, checkpoint.pending_orders_json,
                    checkpoint.reentry_watches_json, checkpoint.daily_pnl_usd,
                    checkpoint.daily_trades, checkpoint.last_trade_date,
                    checkpoint.exchange_states_json, time.time()
                ))
                conn.commit()
                
            self._cleanup_old_checkpoints()
            log.debug(f"Checkpoint saved: cycle={cycle_number}, positions={len(positions)}")
            return True
            
        except Exception as e:
            log.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load(self, max_age_seconds: float = 300) -> Optional[Dict[str, Any]]:
        """Load the most recent valid checkpoint."""
        try:
            cutoff_time = time.time() - max_age_seconds
            
            with self._connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM checkpoints 
                    WHERE created_at > ?
                    ORDER BY created_at DESC 
                    LIMIT 1
                """, (cutoff_time,))
                
                row = cursor.fetchone()
                
            if not row:
                log.info("No recent checkpoint found for recovery")
                return None
            
            checkpoint = {
                'version': row[1], 'timestamp': row[2], 'cycle_number': row[3],
                'portfolio_usd': row[4], 'available_usd': row[5],
                'positions': json.loads(row[6]), 'pending_orders': json.loads(row[7]),
                'reentry_watches': json.loads(row[8]), 'daily_pnl_usd': row[9],
                'daily_trades': row[10], 'last_trade_date': row[11],
                'exchange_states': json.loads(row[12]),
                'age_seconds': time.time() - row[13]
            }
            
            log.info(f"Checkpoint loaded: cycle={checkpoint['cycle_number']}, "
                    f"positions={len(checkpoint['positions'])}, "
                    f"age={checkpoint['age_seconds']:.0f}s")
            return checkpoint
            
        except Exception as e:
            log.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _serialize_positions(self, positions: Dict[str, Any]) -> Dict:
        """Convert position objects to serializable dicts."""
        result = {}
        for coin, pos in positions.items():
            if hasattr(pos, '__dict__'):
                result[coin] = {
                    'direction': getattr(pos, 'direction', None),
                    'entry_price': getattr(pos, 'entry_price', 0.0),
                    'size_usd': getattr(pos, 'size_usd', 0.0),
                    'size_coin': getattr(pos, 'size_coin', 0.0),
                    'stop_loss': getattr(pos, 'stop_loss', 0.0),
                    'take_profit': getattr(pos, 'take_profit', 0.0),
                    'trailing_stop_price': getattr(pos, 'trailing_stop_price', None),
                    'highest_price': getattr(pos, 'highest_price', None),
                    'lowest_price': getattr(pos, 'lowest_price', None),
                    'opened_at': getattr(pos, 'opened_at', datetime.now().isoformat()),
                    'exchange': getattr(pos, 'exchange', ''),
                    'leverage': getattr(pos, 'leverage', 1.0),
                    'margin_usd': getattr(pos, 'margin_usd', 0.0),
                    'metadata': getattr(pos, 'metadata', {}) or {},
                }
            else:
                result[coin] = pos
        return result
    
    def _serialize_pending_orders(self, orders: Dict[str, Any]) -> Dict:
        """Convert pending order objects to serializable dicts."""
        result = {}
        for coin, order in orders.items():
            if hasattr(order, '__dict__'):
                result[coin] = {
                    'coin': getattr(order, 'coin', coin),
                    'direction': getattr(order, 'direction', None),
                    'limit_price': getattr(order, 'limit_price', 0.0),
                    'size_coin': getattr(order, 'size_coin', 0.0),
                    'size_usd': getattr(order, 'size_usd', 0.0),
                    'stop_loss': getattr(order, 'stop_loss', 0.0),
                    'take_profit': getattr(order, 'take_profit', 0.0),
                    'signal_score': getattr(order, 'signal_score', 0.0),
                    'exchange': getattr(order, 'exchange', ''),
                    'leverage': getattr(order, 'leverage', 1),
                    'margin_usd': getattr(order, 'margin_usd', 0.0),
                    'exchange_order_id': getattr(order, 'exchange_order_id', ''),
                    'cycles_waiting': getattr(order, 'cycles_waiting', 0),
                    'reprice_count': getattr(order, 'reprice_count', 0),
                    'max_cycles': getattr(order, 'max_cycles', 15),
                    'reason': getattr(order, 'reason', ''),
                    'placed_at': getattr(order, 'placed_at', time.time()),
                    'metadata': getattr(order, 'metadata', {}) or {},
                }
            else:
                result[coin] = order
        return result
    
    def _serialize_watches(self, watches: Dict[str, Any]) -> Dict:
        """Convert re-entry watch objects to serializable dicts."""
        result = {}
        for coin, watch in watches.items():
            if hasattr(watch, '__dict__'):
                result[coin] = {
                    'coin': getattr(watch, 'coin', coin),
                    'direction': getattr(watch, 'direction', None),
                    'entry_price': getattr(watch, 'entry_price', 0.0),
                    'tp_price': getattr(watch, 'tp_price', 0.0),
                    'reentry_price': getattr(watch, 'reentry_price', 0.0),
                    'stop_price': getattr(watch, 'stop_price', 0.0),
                    'size_usd': getattr(watch, 'size_usd', 0.0),
                    'signal_score': getattr(watch, 'signal_score', 0.0),
                    'cycles': getattr(watch, 'cycles', 0),
                    'max_cycles': getattr(watch, 'max_cycles', 20),
                }
            else:
                result[coin] = watch
        return result

    def get_checkpoint_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return recent checkpoints for inspection/debugging."""
        with self._connection() as conn:
            rows = conn.execute("""
                SELECT cycle_number, portfolio_usd, available_usd, created_at
                FROM checkpoints
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return [
            {
                "cycle": row[0],
                "portfolio": row[1],
                "available": row[2],
                "created_at": row[3],
            }
            for row in rows
        ]
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to prevent disk bloat."""
        try:
            with self._connection() as conn:
                conn.execute("""
                    DELETE FROM checkpoints 
                    WHERE id NOT IN (
                        SELECT id FROM checkpoints 
                        ORDER BY created_at DESC 
                        LIMIT ?
                    )
                """, (self.max_checkpoints,))
                week_ago = time.time() - (7 * 24 * 60 * 60)
                conn.execute("DELETE FROM checkpoints WHERE created_at < ?", (week_ago,))
                conn.commit()
        except Exception as e:
            log.warning(f"Checkpoint cleanup failed: {e}")
    
    def has_recent_checkpoint(self, max_age_seconds: float = 300) -> bool:
        """Check if a recent checkpoint exists."""
        return self.load(max_age_seconds) is not None


# Singleton instance
checkpoint_manager = CheckpointManager()


def save_checkpoint(agent) -> bool:
    """Convenience function to save checkpoint from agent instance."""
    return checkpoint_manager.save(
        cycle_number=getattr(agent, '_cycle', 0),
        portfolio_usd=getattr(agent, '_last_portfolio_usd', 0.0),
        available_usd=getattr(agent, '_last_available_usd', 0.0),
        positions=getattr(agent.risk, 'positions', {}),
        pending_orders=getattr(agent.order_mgr, 'pending_orders', {}),
        reentry_watches=getattr(agent.order_mgr, 'reentry_watches', {}),
        risk_manager=agent.risk,
        exchange_states={}
    )


def load_checkpoint(max_age_seconds: float = 300) -> Optional[Dict[str, Any]]:
    """Convenience function to load checkpoint."""
    return checkpoint_manager.load(max_age_seconds)
