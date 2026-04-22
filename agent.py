"""
agent.py — Main trading agent orchestrator.

Each cycle:
  1. Read portfolio equity from exchange(s)
  2. Fetch latest sentiment (Fear & Greed)
  3. Monitor existing LONG/SHORT positions for SL/TP/trailing exits
  4. Tick the OrderManager (check re-entry watches + expire stale limit orders)
  5. Poll pending limit orders for fills
  6. For each coin: fetch candles → compute technical + advanced + regime signals
     → generate LONG / SHORT / FLAT decision → risk-check → execute
     → if score is borderline (42–62): confirm with chart screenshot analyst
  7. Summary

No-emotion rule: once a TP is hit → the agent immediately plans re-entry at a
fib retracement level without hesitation. No second-guessing, no manual override.

Chart confirmation rule:
  • Score ≥ 63 or ≤ 37 → trade on indicators alone (high conviction)
  • Score 38–62        → also run visual chart analyst for confirmation
    - If chart agrees  → trade with normal sizing
    - If chart says WAIT → skip trade this cycle
    - If chart disagrees (opposite direction) → definitely skip
"""

import math
import time
import sys
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Dict
from paths import (
    ASSET_DOSSIERS_JSON,
    CHALLENGER_MODEL_JSON,
    CONTROL_JSON,
    DATA_DIR,
    DAILY_MARKET_MAP_JSON,
    DASHBOARD_SNAPSHOT_JSON,
    DECISION_REVIEW_REPORT_JSON,
    KILL_FILE,
    LLM_REFEREE_REPORT_JSON,
    MISSED_MOVE_REPORT_JSON,
    PLAYBOOK_DISTILLER_REPORT_JSON,
    STATE_JSON,
    TRADE_REVIEWS_JSON,
    TRADES_CSV,
)
from datetime import datetime

import os
import analog_engine
import asset_dossier
import challenger_model
from config import Config
import data_reliability
import decision_dataset
import decision_review_lab
import execution_coach
import feature_store
import llm_referee
from logger import get_logger
from data.market_data import completed_candle_frame, fetch_candles, get_current_price
import hosted_state_sync
import market_map
import missed_move_lab
import playbook_distiller
import portfolio_guard
import trade_dataset
import trade_logger
import trade_review
from asset_state_machine import build_asset_state
from indicators.technical import compute_signals
from indicators.advanced  import compute_advanced_signals
from indicators.sentiment import get_fear_greed_score, sentiment_summary
from indicators.regimes   import compute_regimes
from indicators.chart_analyst import read_chart, ChartVerdict
from indicators.mtf  import compute_mtf, MTFAnalysis
from indicators.news import get_news_signal
from indicators.candlestick_patterns import compute_candlestick_patterns
from indicators.orderbook_levels import (
    configure_background_orderbook_feed,
    get_orderbook_levels,
    prime_background_orderbook_feed,
    start_background_orderbook_feed,
    stop_background_orderbook_feed,
)
from indicators.trade_memory import trade_memory
from indicators.funding_oi_cvd import get_funding_oi_cvd
from exchanges.hyperliquid_markets import (
    hyperliquid_supports_shorts,
    hyperliquid_instrument_type,
    hyperliquid_market_is_active,
    is_hyperliquid_supported,
)
from narrative import get_narrative_signal
from strategy.aggressive_strategy import AggressiveStrategy
from strategy.order_manager import OrderManager, PendingOrder, ReEntryWatch
from risk.risk_manager import RiskManager, OrderRequest, OpenPosition
from exchanges.base import BaseExchange
from notifications import build_notifier
from checkpoint import checkpoint_manager, load_checkpoint
from runtime_power import get_power_status
from dashboard.snapshot import build_dashboard_snapshot, default_control, merge_dataset_into_trades
from circuit_breaker import (
    circuit_breaker_registry,
    get_exchange_circuit,
    get_price_feed_circuit,
    get_indicator_circuit,
    retry_with_backoff,
    CircuitBreakerError,
)

log = get_logger("agent")


class TradingAgent:
    def __init__(self, cfg: Config, exchanges: List[BaseExchange]):
        self.cfg           = cfg
        self.exchanges     = exchanges
        self.risk          = RiskManager(cfg.trading)
        self.strategy      = AggressiveStrategy(cfg.trading, cfg.indicators)
        self.order_mgr     = OrderManager()
        self.notifier      = build_notifier(cfg)
        self._cycle        = 0
        self._running      = False
        self._last_portfolio_usd = 0.0
        self._last_available_usd = 0.0
        self._last_signals: Dict[str, dict] = {}
        self._last_power_status: Dict[str, object] = {}
        self._orderbook_history: Dict[str, List[dict]] = {}
        self._tradable_coins = [coin.upper() for coin in cfg.trading.coins]
        self._tradable_coin_set = set(self._tradable_coins)
        self._dynamic_analysis_coins = [
            str(coin).upper()
            for coin in (getattr(cfg.trading, "dynamic_analysis_coins", []) or [])
            if coin
        ]
        seen_analysis: List[str] = []
        analysis_sources = list(getattr(cfg.trading, "analysis_coins", []) or []) + list(self._dynamic_analysis_coins)
        for coin in analysis_sources or self._tradable_coins:
            coin_upper = coin.upper()
            if coin_upper not in seen_analysis:
                seen_analysis.append(coin_upper)
        for coin in self._tradable_coins:
            if coin not in seen_analysis:
                seen_analysis.insert(0, coin)
        self._analysis_coins = seen_analysis
        self._price_circuits = {
            coin: get_price_feed_circuit(f"primary_{coin.lower()}")
            for coin in self._analysis_coins
        }
        self._sentiment_circuit = get_indicator_circuit("sentiment")
        self._mtf_circuit = get_indicator_circuit("mtf")
        self._news_circuit = get_indicator_circuit("news")
        self._orderbook_circuit = get_indicator_circuit("orderbook_levels")
        self._exchange_circuits = {
            ex.name: get_exchange_circuit(ex.name) for ex in exchanges
        }
        self._memory = trade_memory          # reinforcement learning module
        self._analog_engine = analog_engine.HistoricalAnalogEngine(cfg.trading)
        self._llm_referee = llm_referee.LLMReferee(cfg.trading)
        self._last_learning_report_refresh_ts = 0.0

        # ── Signal streak: require N consecutive cycles before entering ──────
        # Prevents entering on the very first noisy crossing of a threshold.
        # coin → {"action": "LONG"|"SHORT", "count": int}
        self._signal_streak: Dict[str, dict] = {}

        # ── Post-reversal cooldown: don't re-enter immediately after a close ─
        # After signal_reversal close, we wait 1 full cycle before re-evaluating.
        # Prevents the classic whipsaw: close LONG → immediately open SHORT.
        # coin → timestamp of last signal_reversal close
        self._reversal_cooldown: Dict[str, float] = {}

        # ── FLAT-while-positioned tracker ────────────────────────────────────
        # Counts consecutive FLAT cycles while a position is open.
        # If conviction disappears for N cycles, position is stale → close it.
        # coin → consecutive flat cycle count
        self._flat_streak: Dict[str, int] = {}
        self._precision_entry_history: List[dict] = []

        self._bootstrap_precision_entry_history()
        self._attempt_recovery()
        self._reconcile_with_exchange()

    # ── Control ───────────────────────────────────────────────

    def start(self):
        """Run the agent loop. Press Ctrl+C to stop."""
        log.info("=" * 64)
        log.info(f"  Hyperliquid Trading Agent  |  Mode: "
                 f"{'DRY RUN 🟡' if self.cfg.is_dry_run else 'LIVE 🔴'}")
        log.info(f"  Trade coins: {self._tradable_coins}")
        log.info(f"  Watchlist : {self._analysis_coins}")
        log.info(f"  Exchanges : {[e.name for e in self.exchanges]}")
        log.info(f"  Leverage  : {self.cfg.trading.leverage}×")
        if getattr(self.cfg.trading, "dynamic_trade_planning", True):
            log.info("  Trade plan: dynamic ATR + structure-based SL/TP")
        else:
            log.info(f"  SL / TP   : {self.cfg.trading.stop_loss_pct*100:.0f}% / "
                     f"{self.cfg.trading.take_profit_pct*100:.0f}%")
        log.info(f"  Trailing  : {self.cfg.trading.trailing_stop_pct*100:.0f}%")
        log.info(f"  Limit orders: "
                 f"{'YES' if any(e.supports_limit_orders() for e in self.exchanges) else 'NO (market fallback)'}")
        log.info("=" * 64)
        self._log_circuit_status()
        self._start_background_orderbook_feed()
        self._running = True
        try:
            while self._running:
                self._cycle += 1
                log.info(f"\n{'─'*50}\n  Cycle #{self._cycle}\n{'─'*50}")
                try:
                    self._run_cycle()
                except Exception as e:
                    log.error(f"Cycle #{self._cycle} failed: {e}", exc_info=True)
                    self.notifier.error_alert(f"Cycle #{self._cycle} error: {e}")
                self._save_checkpoint()
                secs = self.cfg.trading.check_interval_seconds
                log.info(f"Sleeping {secs}s…")
                time.sleep(secs)
        except KeyboardInterrupt:
            log.info("\nStopped by user (Ctrl+C)")
            self._save_checkpoint()
            self._print_final_summary()
        finally:
            self._stop_background_orderbook_feed()

    def stop(self):
        self._running = False

    def _start_background_orderbook_feed(self) -> None:
        if not (
            getattr(self.cfg.trading, "use_orderbook_levels", True)
            and getattr(self.cfg.trading, "orderbook_feed_enabled", True)
        ):
            return
        tradable_coin_set = set(getattr(self, "_tradable_coin_set", set(getattr(self, "_tradable_coins", []))))
        dynamic_analysis_coins = [
            str(coin).upper()
            for coin in getattr(self, "_dynamic_analysis_coins", []) or []
            if coin
        ]
        scout_limit = int(getattr(self.cfg.trading, "dynamic_market_cap_feed_limit", 16) or 16)
        scout_feed = [coin for coin in dynamic_analysis_coins if coin not in tradable_coin_set][:scout_limit]
        feed_coins = list(getattr(self, "_tradable_coins", []))
        if not feed_coins:
            feed_coins = list(getattr(self, "_analysis_coins", []))
        for coin in scout_feed:
            if coin not in feed_coins:
                feed_coins.append(coin)
        if getattr(self.cfg.trading, "enforce_active_venue_markets", True):
            filtered = []
            for coin in feed_coins:
                if is_hyperliquid_supported(coin) and not hyperliquid_market_is_active(coin):
                    continue
                filtered.append(coin)
            feed_coins = filtered
        if not feed_coins:
            log.info("No active venue-backed symbols available for the background orderbook feed")
            return
        configure_background_orderbook_feed(
            feed_coins,
            depth_limit=getattr(self.cfg.trading, "orderbook_depth_limit", 120),
            poll_interval_seconds=getattr(self.cfg.trading, "orderbook_feed_poll_seconds", 3.0),
            history_size=getattr(self.cfg.trading, "orderbook_feed_history_size", 120),
        )
        try:
            prime_background_orderbook_feed()
        except Exception as exc:
            log.debug("Orderbook feed prime skipped: %s", exc)
        try:
            start_background_orderbook_feed()
        except Exception as exc:
            log.warning("Failed to start background orderbook feed: %s", exc)

    @staticmethod
    def _stop_background_orderbook_feed() -> None:
        try:
            stop_background_orderbook_feed()
        except Exception as exc:
            log.debug("Orderbook feed stop skipped: %s", exc)

    def _bootstrap_precision_entry_history(self) -> None:
        if not getattr(self.cfg.trading, "precision_mode_enabled", False):
            return

        history: List[dict] = []
        try:
            rows = decision_dataset.load_decisions(limit=1000)
        except Exception as exc:
            log.debug("Precision entry bootstrap skipped: %s", exc)
            self._precision_entry_history = history
            return

        for row in rows:
            stage = str(row.get("stage", "") or "").lower()
            if stage not in {"market_entry_opened", "limit_entry_placed"}:
                continue
            if not (bool(row.get("executed", False)) or bool(row.get("pending_limit", False))):
                continue
            action = str(row.get("final_action", row.get("candidate_action", "")) or "").upper()
            if action not in {"LONG", "SHORT"}:
                continue
            snap = dict(row.get("signal_snapshot") or {})
            thesis = dict(snap.get("thesis") or {})
            history.append({
                "ts": float(row.get("recorded_at_ts", 0.0) or 0.0),
                "coin": str(row.get("coin", "") or "").upper(),
                "action": action,
                "family": self._precision_family_key(
                    str(row.get("coin", "") or "").upper(),
                    action,
                    str(thesis.get("archetype", snap.get("thesis_archetype", "UNKNOWN")) or "UNKNOWN").upper(),
                ),
                "mode": "limit" if bool(row.get("pending_limit", False)) else "market",
            })
        self._precision_entry_history = history[-250:]

    @staticmethod
    def _precision_family_key(coin: str, action: str, archetype: str) -> str:
        return f"{str(coin or '').upper()}:{str(action or '').upper()}:{str(archetype or 'UNKNOWN').upper()}"

    def _prune_precision_entry_history(self) -> None:
        if not self._precision_entry_history:
            return
        cutoff = time.time() - (72 * 3600)
        self._precision_entry_history = [
            item for item in self._precision_entry_history
            if float(item.get("ts", 0.0) or 0.0) >= cutoff
        ][-250:]

    def _record_precision_entry(self, coin: str, signal, *, mode: str) -> None:
        if not getattr(self.cfg.trading, "precision_mode_enabled", False):
            return
        action = str(getattr(signal, "action", "") or "").upper()
        if action not in {"LONG", "SHORT"}:
            return
        thesis = dict(getattr(signal, "thesis", {}) or {})
        self._precision_entry_history.append({
            "ts": time.time(),
            "coin": str(coin or "").upper(),
            "action": action,
            "family": self._precision_family_key(
                coin,
                action,
                str(thesis.get("archetype", "UNKNOWN") or "UNKNOWN").upper(),
            ),
            "mode": mode,
        })
        self._prune_precision_entry_history()

    def _check_precision_entry_cadence(self, coin: str, signal) -> tuple[bool, str]:
        if not getattr(self.cfg.trading, "precision_mode_enabled", False):
            return True, ""
        action = str(getattr(signal, "action", "") or "").upper()
        if action not in {"LONG", "SHORT"}:
            return True, ""

        self._prune_precision_entry_history()
        now = time.time()
        coin_upper = str(coin or "").upper()
        thesis = dict(getattr(signal, "thesis", {}) or {})
        family = self._precision_family_key(
            coin_upper,
            action,
            str(thesis.get("archetype", "UNKNOWN") or "UNKNOWN").upper(),
        )

        max_per_day = int(getattr(self.cfg.trading, "precision_max_new_entries_per_day", 2) or 2)
        today = datetime.fromtimestamp(now).date()
        entries_today = sum(
            1 for item in self._precision_entry_history
            if datetime.fromtimestamp(float(item.get("ts", 0.0) or 0.0)).date() == today
        )
        if entries_today >= max_per_day:
            return False, f"precision mode already used {entries_today}/{max_per_day} new entries today"

        same_coin_cooldown = int(getattr(self.cfg.trading, "precision_same_coin_cooldown_minutes", 360) or 360) * 60
        recent_same_coin = [
            item for item in self._precision_entry_history
            if item.get("coin") == coin_upper and (now - float(item.get("ts", 0.0) or 0.0)) < same_coin_cooldown
        ]
        if recent_same_coin:
            minutes_left = max(1, int(math.ceil((same_coin_cooldown - (now - float(recent_same_coin[-1].get("ts", 0.0) or 0.0))) / 60.0)))
            return False, f"{coin_upper} is on a precision cooldown for another ~{minutes_left}m"

        family_cooldown = int(getattr(self.cfg.trading, "precision_same_family_cooldown_minutes", 720) or 720) * 60
        recent_same_family = [
            item for item in self._precision_entry_history
            if item.get("family") == family and (now - float(item.get("ts", 0.0) or 0.0)) < family_cooldown
        ]
        if recent_same_family:
            minutes_left = max(1, int(math.ceil((family_cooldown - (now - float(recent_same_family[-1].get("ts", 0.0) or 0.0))) / 60.0)))
            return False, f"{family.replace(':', ' ')} is cooling down for another ~{minutes_left}m"

        return True, ""

    def _refresh_asset_state(
        self,
        coin: str,
        *,
        stage: str = "",
        current_position: str | None = None,
        pending_limit: bool = False,
    ) -> dict:
        if coin not in self._last_signals:
            return {}
        snap = self._last_signals[coin]
        lifecycle = build_asset_state(
            snap,
            stage=stage or str(snap.get("decision_stage") or "analysis"),
            current_position=current_position if current_position is not None else (self.risk.position_direction(coin) or ""),
            pending_limit=bool(pending_limit or self.order_mgr.has_pending(coin)),
        )
        snap.update({
            "decision_stage": lifecycle.get("stage", stage or "analysis"),
            "asset_state": lifecycle.get("state", "OBSERVING"),
            "asset_state_label": lifecycle.get("label", "Observing"),
            "next_unblock_reason": lifecycle.get("next_unblock_reason", ""),
        })
        return lifecycle

    def _load_json_file(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _get_asset_dossier_entry(self, coin: str) -> dict:
        payload = self._load_json_file(ASSET_DOSSIERS_JSON)
        return dict((payload.get("assets") or {}).get(coin) or {})

    def _get_missed_move_context(self, coin: str) -> dict:
        payload = self._load_json_file(MISSED_MOVE_REPORT_JSON)
        recent = [
            dict(item or {})
            for item in list(payload.get("recent_missed_moves") or [])
            if str((item or {}).get("coin") or "").strip().upper() == coin
        ][:3]
        top_assets = {
            str((item or {}).get("coin") or "").strip().upper(): int((item or {}).get("misses") or 0)
            for item in list(payload.get("top_missed_assets") or [])
        }
        return {
            "miss_count": int(top_assets.get(coin, 0)),
            "recent_examples": recent,
        }

    def _apply_llm_referee(self, coin: str, signal, *, current_position: str | None = None) -> bool:
        snap = dict(self._last_signals.get(coin) or {})
        if not self._llm_referee.should_review(coin, snap, current_position=current_position or ""):
            if not self._llm_referee.enabled():
                self._last_signals[coin]["llm_referee"] = {
                    "enabled": False,
                    "used": False,
                    "verdict": "DISABLED",
                    "summary": "OpenAI referee is disabled or OPENAI_API_KEY is missing",
                }
            return False

        dossier = self._get_asset_dossier_entry(coin)
        missed_context = self._get_missed_move_context(coin)
        verdict = self._llm_referee.review_setup(
            coin,
            snap,
            dossier=dossier,
            missed_move_context=missed_context,
        )
        self._last_signals[coin]["llm_referee"] = verdict
        summary = str(verdict.get("summary") or "").strip()
        why_now = str(verdict.get("why_now") or "").strip()
        next_unblock = str(verdict.get("next_unblock") or "").strip()
        if summary:
            self._last_signals[coin]["llm_referee_summary"] = summary
        if why_now:
            self._last_signals[coin]["llm_referee_why_now"] = why_now
        if next_unblock and not str(self._last_signals[coin].get("next_unblock_reason") or "").strip():
            self._last_signals[coin]["next_unblock_reason"] = next_unblock

        blocking_verdicts = {
            str(item or "").strip().upper()
            for item in list(getattr(self.cfg.trading, "llm_referee_block_on_verdicts", []) or [])
        }
        verdict_name = str(verdict.get("verdict") or "").strip().upper()
        if verdict_name in blocking_verdicts:
            reason = summary or "OpenAI referee blocked the setup"
            log.info(f"[{coin}] 🧠 OpenAI referee blocks {signal.action}: {reason}")
            signal.action = "FLAT"
            signal.flat_reason = reason
            signal.reason = reason
            self._sync_signal_snapshot(coin, signal)
            return True

        if why_now:
            combined = why_now if not summary else f"{summary} • {why_now}"
            self._last_signals[coin]["decision_reason"] = combined
            signal.reason = combined
        self._sync_signal_snapshot(coin, signal)
        return False

    def _maybe_refresh_learning_reports(self) -> None:
        if not (
            getattr(self.cfg.trading, "decision_review_enabled", True)
            or getattr(self.cfg.trading, "challenger_model_enabled", True)
            or getattr(self.cfg.trading, "missed_move_lab_enabled", True)
            or getattr(self.cfg.trading, "playbook_distiller_enabled", True)
        ):
            return

        refresh_seconds = max(
            1800.0,
            float(getattr(self.cfg.trading, "challenger_refresh_hours", 6.0) or 6.0) * 3600.0,
        )
        now = time.time()
        if self._last_learning_report_refresh_ts and (now - self._last_learning_report_refresh_ts) < refresh_seconds:
            return

        target_r = float(getattr(self.cfg.trading, "decision_review_target_r", 0.25) or 0.25)
        horizon_minutes = int(getattr(self.cfg.trading, "decision_review_horizon_minutes", 720) or 720)
        interval = str(getattr(self.cfg.trading, "decision_review_interval", "5m") or "5m")
        dedupe_minutes = int(getattr(self.cfg.trading, "decision_review_dedupe_minutes", 30) or 30)

        try:
            if getattr(self.cfg.trading, "decision_review_enabled", True):
                decision_review_lab.build_and_save_report(
                    data_dir=DATA_DIR,
                    target_r=target_r,
                    horizon_minutes=horizon_minutes,
                    interval=interval,
                    dedupe_minutes=dedupe_minutes,
                )
            if getattr(self.cfg.trading, "missed_move_lab_enabled", True):
                missed_move_lab.build_and_save_report(
                    data_dir=DATA_DIR,
                    target_r=target_r,
                    horizon_minutes=horizon_minutes,
                    interval=interval,
                    dedupe_minutes=dedupe_minutes,
                )
            if getattr(self.cfg.trading, "challenger_model_enabled", True):
                challenger_model.build_and_save_report(
                    self.cfg,
                    data_dir=DATA_DIR,
                )
            if getattr(self.cfg.trading, "playbook_distiller_enabled", True):
                playbook_distiller.build_and_save_report(
                    self.cfg,
                    data_dir=DATA_DIR,
                )
            self._last_learning_report_refresh_ts = now
        except Exception as exc:
            log.debug("Learning report refresh skipped: %s", exc)

    # ── Recovery and reconciliation ──────────────────────────

    def _attempt_recovery(self):
        checkpoint = load_checkpoint(max_age_seconds=3600)
        if not checkpoint:
            log.info("No recent checkpoint found - starting fresh")
            return

        log.info("=" * 64)
        log.info("  RECOVERING FROM CHECKPOINT")
        log.info("=" * 64)

        self._cycle = checkpoint.get("cycle_number", 0)
        self._last_portfolio_usd = checkpoint.get("portfolio_usd", 0.0)
        self._last_available_usd = checkpoint.get("available_usd", 0.0)
        self.risk.daily_pnl_usd = checkpoint.get("daily_pnl_usd", 0.0)
        self.risk.daily_trades = checkpoint.get("daily_trades", 0)
        self.risk.last_trade_date = checkpoint.get("last_trade_date", "")
        active_universe = {coin.upper() for coin in self.cfg.trading.coins}

        for coin, pos in checkpoint.get("positions", {}).items():
            if coin.upper() not in active_universe:
                log.warning(f"[{coin}] Skipping restored position outside active trade universe")
                continue
            entry_price = float(pos.get("entry_price", 0.0) or 0.0)
            direction = pos.get("direction", "")
            trail = pos.get("trailing_stop_price")
            if not trail and entry_price > 0:
                trail = (
                    entry_price * (1 - self.cfg.trading.trailing_stop_pct)
                    if direction == "LONG"
                    else entry_price * (1 + self.cfg.trading.trailing_stop_pct)
                )
            opened_at = self._coerce_timestamp(pos.get("opened_at"))
            self.risk.restore_position(
                OpenPosition(
                    coin=coin,
                    direction=direction,
                    entry_price=entry_price,
                    size_usd=float(pos.get("size_usd", 0.0) or 0.0),
                    size_coin=float(pos.get("size_coin", 0.0) or 0.0),
                    stop_loss=float(pos.get("stop_loss", 0.0) or 0.0),
                    take_profit=float(pos.get("take_profit", 0.0) or 0.0),
                    trailing_stop_price=float(trail or 0.0),
                    opened_at=opened_at,
                    exchange=pos.get("exchange", ""),
                    metadata=pos.get("metadata", {}) or {},
                )
            )
            trade_logger.restore_open(
                coin=coin,
                direction=direction,
                entry_price=entry_price,
                size_usd=float(pos.get("size_usd", 0.0) or 0.0),
                stop_loss=float(pos.get("stop_loss", 0.0) or 0.0),
                take_profit=float(pos.get("take_profit", 0.0) or 0.0),
                leverage=self.cfg.trading.leverage,
                opened_at=datetime.utcfromtimestamp(opened_at).strftime("%Y-%m-%d %H:%M"),
            )

        for coin, order in checkpoint.get("pending_orders", {}).items():
            if coin.upper() not in active_universe:
                log.warning(f"[{coin}] Skipping restored pending order outside active trade universe")
                continue
            self.order_mgr.restore_pending_order(
                PendingOrder(
                    coin=coin,
                    direction=order.get("direction", ""),
                    limit_price=float(order.get("limit_price", 0.0) or 0.0),
                    size_coin=float(order.get("size_coin", 0.0) or 0.0),
                    size_usd=float(order.get("size_usd", 0.0) or 0.0),
                    stop_loss=float(order.get("stop_loss", 0.0) or 0.0),
                    take_profit=float(order.get("take_profit", 0.0) or 0.0),
                    signal_score=float(order.get("signal_score", 0.0) or 0.0),
                    exchange=order.get("exchange", ""),
                    exchange_order_id=order.get("exchange_order_id", ""),
                    cycles_waiting=int(order.get("cycles_waiting", 0) or 0),
                    reprice_count=int(order.get("reprice_count", 0) or 0),
                    reason=order.get("reason", "re_entry"),
                    placed_at=float(order.get("placed_at", time.time()) or time.time()),
                    metadata=order.get("metadata", {}) or {},
                )
            )

        for coin, watch in checkpoint.get("reentry_watches", {}).items():
            if coin.upper() not in active_universe:
                log.warning(f"[{coin}] Skipping restored re-entry watch outside active trade universe")
                continue
            self.order_mgr.restore_watch(
                ReEntryWatch(
                    coin=coin,
                    direction=watch.get("direction", ""),
                    entry_price=float(watch.get("entry_price", 0.0) or 0.0),
                    tp_price=float(watch.get("tp_price", 0.0) or 0.0),
                    reentry_price=float(watch.get("reentry_price", 0.0) or 0.0),
                    stop_price=float(watch.get("stop_price", 0.0) or 0.0),
                    size_usd=float(watch.get("size_usd", 0.0) or 0.0),
                    signal_score=float(watch.get("signal_score", 0.0) or 0.0),
                    cycles=int(watch.get("cycles", 0) or 0),
                    max_cycles=int(watch.get("max_cycles", 15) or 15),
                )
            )

        log.info(
            f"Recovered cycle #{self._cycle} with "
            f"{len(self.risk.positions)} position(s), "
            f"{len(self.order_mgr.pending_orders)} pending order(s), "
            f"{len(self.order_mgr.reentry_watches)} watch(es)"
        )

    def _reconcile_with_exchange(self):
        if not self.exchanges or all(ex.is_dry_run() for ex in self.exchanges):
            log.info("Skipping exchange reconciliation in dry-run mode")
            return

        reconciled: Dict[str, OpenPosition] = {}
        for ex in self.exchanges:
            state = self._get_account_state_safe(ex)
            if not state:
                continue
            for raw in state.positions:
                coin = raw.get("coin")
                if not coin:
                    continue
                if coin in reconciled:
                    raise RuntimeError(
                        f"Duplicate live position for {coin} across exchanges is not supported safely"
                    )
                entry_price = float(raw.get("entry_price", 0.0) or 0.0)
                size_coin = abs(float(raw.get("size", 0.0) or 0.0))
                size_usd = abs(size_coin * entry_price)
                direction = raw.get("direction", "")
                stop_loss = (
                    entry_price * (1 - self.cfg.trading.stop_loss_pct)
                    if direction == "LONG"
                    else entry_price * (1 + self.cfg.trading.stop_loss_pct)
                )
                take_profit = (
                    entry_price * (1 + self.cfg.trading.take_profit_pct)
                    if direction == "LONG"
                    else entry_price * (1 - self.cfg.trading.take_profit_pct)
                )
                trailing_stop_price = (
                    entry_price * (1 - self.cfg.trading.trailing_stop_pct)
                    if direction == "LONG"
                    else entry_price * (1 + self.cfg.trading.trailing_stop_pct)
                )
                reconciled[coin] = OpenPosition(
                    coin=coin,
                    direction=direction,
                    entry_price=entry_price,
                    size_usd=size_usd,
                    size_coin=size_coin,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop_price=trailing_stop_price,
                    exchange=ex.name,
                )
                trade_logger.restore_open(
                    coin=coin,
                    direction=direction,
                    entry_price=entry_price,
                    size_usd=size_usd,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=self.cfg.trading.leverage,
                )
        if reconciled:
            self.risk.replace_positions(reconciled)

    def _coerce_timestamp(self, value) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value:
            try:
                return float(value)
            except ValueError:
                try:
                    return datetime.fromisoformat(value).timestamp()
                except ValueError:
                    pass
        return time.time()

    def _refresh_power_status(self) -> dict:
        status = get_power_status().as_dict()
        self._last_power_status = status
        return status

    def _enforce_power_safety(self) -> bool:
        status = self._refresh_power_status()
        if self.cfg.is_dry_run or not status.get("available"):
            return True

        reason = None
        on_ac = status.get("on_ac_power")
        battery_pct = status.get("battery_pct")
        if self.cfg.trading.require_ac_power_for_live and on_ac is False:
            reason = (
                "Live trading requires AC power on the local Mac"
                + (f" (battery {battery_pct}%)" if battery_pct is not None else "")
            )
        elif (
            battery_pct is not None
            and battery_pct < self.cfg.trading.minimum_battery_pct_for_live
        ):
            reason = (
                "Battery dropped below the live-trading minimum "
                f"({battery_pct}% < {self.cfg.trading.minimum_battery_pct_for_live}%)"
            )

        if not reason:
            return True

        log.critical(reason)
        self.notifier.error_alert(reason)
        if self.cfg.trading.stop_live_on_power_loss:
            self.emergency_close_all(reason)
            self._running = False
        return False

    # ── Core cycle ────────────────────────────────────────────

    def _run_cycle(self):
        # 0. Kill switch check
        kill_file = KILL_FILE
        if kill_file.exists():
            reason = kill_file.read_text().strip() or "manual kill switch"
            log.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
            self.emergency_close_all(reason)
            kill_file.unlink(missing_ok=True)
            self._running = False
            return

        if not self._enforce_power_safety():
            return

        # 1. Portfolio equity
        portfolio_usd = self._get_portfolio_usd()
        if portfolio_usd is None:
            log.error("Cannot read portfolio value - using last known value")
            portfolio_usd = self._last_portfolio_usd or 0.0
        self._last_portfolio_usd = portfolio_usd
        self._last_available_usd = self.risk.available_capital(portfolio_usd)
        self.risk.update_peak_portfolio(portfolio_usd)
        log.info(f"Portfolio equity: ${portfolio_usd:,.2f}")

        if (not self.cfg.is_dry_run and
                self.cfg.trading.reconcile_every_n_cycles > 0 and
                self._cycle % self.cfg.trading.reconcile_every_n_cycles == 0):
            self._reconcile_with_exchange()

        # 2. Sentiment
        sentiment = self._get_sentiment_safe()
        if not sentiment:
            log.warning("Sentiment unavailable - using neutral")
            sentiment = {'raw_score': 50, 'label': 'Neutral', 'signal_score': 50, 'is_extreme': False}
        log.info(sentiment_summary(sentiment))

        # 3. Exit monitoring (SL / TP / trailing)
        current_prices = self._fetch_all_prices()
        self._check_and_execute_exits(current_prices, portfolio_usd)

        # 4. OrderManager tick — advance re-entry watches, expire stale limits
        om_actions = self.order_mgr.tick(current_prices)
        for action in om_actions:
            self._handle_order_manager_action(action, portfolio_usd)

        # 5. Poll pending limit orders for fills
        self._poll_pending_limits(current_prices)
        self._manage_pending_limits(current_prices, portfolio_usd)

        # 6. Tick trade memory cooldowns (once per cycle)
        self._memory.tick_cooldowns()
        log.info(self._memory.summary())

        # 7. Analyse each coin for new positions
        self._last_signals = {}
        for coin in self._analysis_coins:
            # Skip if we already have a pending limit order for this coin
            if self.order_mgr.has_pending(coin):
                log.info(f"[{coin}] Skipping analysis — limit order pending")
                pending = self.order_mgr.pending_orders.get(coin)
                if pending is not None:
                    entry_context = dict(getattr(pending, "metadata", {}).get("entry_context", {}) or {})
                    self._last_signals[coin] = {
                        "action": str(getattr(pending, "direction", "FLAT") or "FLAT").upper(),
                        "decision": str(getattr(pending, "direction", "FLAT") or "FLAT").upper(),
                        "score": float(getattr(pending, "signal_score", 50.0) or 50.0),
                        "confidence": str(entry_context.get("confidence") or "MEDIUM").upper(),
                        "price": float(getattr(pending, "limit_price", 0.0) or 0.0),
                        "analysis_price": float(getattr(pending, "limit_price", 0.0) or 0.0),
                        "live_price": float(current_prices.get(coin) or getattr(pending, "limit_price", 0.0) or 0.0),
                        "decision_reason": f"Resting {pending.direction} limit is waiting for a fill.",
                        "reason": f"Resting {pending.direction} limit is waiting for a fill.",
                        "flat_reason": "",
                        "instrument_type": str(entry_context.get("instrument_type") or hyperliquid_instrument_type(coin) or "crypto"),
                        "execution_mode": "tradable",
                        "planned_stop_loss": float(getattr(pending, "stop_loss", 0.0) or 0.0),
                        "planned_take_profit": float(getattr(pending, "take_profit", 0.0) or 0.0),
                        "planned_risk_reward_ratio": float(entry_context.get("planned_risk_reward_ratio") or 0.0),
                        "trade_plan": dict(entry_context.get("trade_plan") or getattr(pending, "metadata", {}).get("trade_plan", {}) or {}),
                        "execution_quality_summary": "A qualifying passive entry is already resting on the book.",
                    }
                    self._refresh_asset_state(coin, stage="entry_limit_already_pending", pending_limit=True)
                continue
            try:
                self._analyse_coin(coin, sentiment, portfolio_usd)
            except Exception as e:
                log.error(f"[{coin}] Unexpected error: {e}", exc_info=True)

        # 8. Summary + write state.json for dashboard
        log.info("\n" + self.risk.summary(portfolio_usd))
        log.info(self.order_mgr.summary())
        self._maybe_refresh_learning_reports()
        self._write_state(portfolio_usd, sentiment)

        # 8. Heartbeat notification every 6 cycles
        if self._cycle % 6 == 0:
            self.notifier.heartbeat(portfolio_usd, len(self.risk.positions))

    # ── Coin analysis ─────────────────────────────────────────

    def _analyse_coin(self, coin: str, sentiment: dict, portfolio_usd: float):
        log.info(f"[{coin}] Analysing…")
        icfg = self.cfg.indicators

        if (
            getattr(self.cfg.trading, "enforce_active_venue_markets", True)
            and is_hyperliquid_supported(coin)
            and not hyperliquid_market_is_active(coin)
        ):
            log.info(f"[{coin}] Hyperliquid market is currently inactive — keeping it analysis-only")
            return

        # Fetch OHLCV
        df = retry_with_backoff(
            fetch_candles,
            max_retries=2,
            base_delay=0.5,
            coin=coin,
            interval=self.cfg.trading.candle_interval,
            lookback=self.cfg.trading.lookback_periods,
        )
        if df is None:
            if is_hyperliquid_supported(coin):
                log.warning(f"[{coin}] No recent Hyperliquid candle data — skipping")
            else:
                log.warning(f"[{coin}] No candle data — skipping")
            return

        analysis_df = df
        if getattr(self.cfg.trading, "use_closed_candles_for_conviction", True):
            analysis_df = completed_candle_frame(df)
            if analysis_df is None or analysis_df.empty:
                log.warning(f"[{coin}] No completed candles available — skipping")
                return

        live_price = float(df["close"].iloc[-1]) if len(df) else 0.0
        analysis_price = float(analysis_df["close"].iloc[-1]) if len(analysis_df) else live_price
        if getattr(self.cfg.trading, "use_closed_candles_for_conviction", True):
            log.info(
                f"[{coin}] Conviction on completed {self.cfg.trading.candle_interval} candles "
                f"@ {analysis_price:.2f} | live price {live_price:.2f}"
            )

        # Classic indicators
        tech = compute_signals(analysis_df, coin, icfg, self.cfg.trading)
        tech.closed_price = analysis_price
        tech.live_price = live_price or tech.price
        tech.price = tech.live_price

        # Advanced / structure indicators
        advanced = compute_advanced_signals(analysis_df, coin)

        # Market regime indicators
        try:
            regimes = compute_regimes(analysis_df, coin)
            log.info(f"[{coin}] Regime: {regimes.dominant_regime} "
                     f"(mom={regimes.momentum_score:.0f} trend={regimes.trend_score:.0f} "
                     f"mr={regimes.mean_rev_score:.0f} vol={regimes.volatility_score:.0f} "
                     f"abs={regimes.absorption_score:.0f} cat={regimes.catalyst_score:.0f})")
        except Exception as e:
            log.warning(f"[{coin}] Regime compute failed: {e}")
            regimes = None

        # Candlestick pattern analysis (pure OHLCV, no API)
        try:
            candle_patterns = compute_candlestick_patterns(analysis_df, coin)
        except Exception as e:
            log.warning(f"[{coin}] Candlestick patterns failed: {e}")
            candle_patterns = None

        # News sentiment (fetch, cached 10 min)
        news_signal = None
        if self.cfg.trading.use_news:
            try:
                news_signal = self._news_circuit.call(
                    get_news_signal,
                    coin,
                    self.cfg.trading.cryptopanic_auth_token,
                )
            except Exception as e:
                log.warning(f"[{coin}] News fetch failed: {e}")

        narrative_signal = None
        if getattr(self.cfg.trading, "use_narrative_gate", True):
            try:
                narrative_signal = get_narrative_signal(
                    coin,
                    news_signal=news_signal,
                    risk_window_minutes=getattr(self.cfg.trading, "narrative_event_risk_window_minutes", 90),
                    post_event_cooldown_minutes=getattr(self.cfg.trading, "narrative_post_event_cooldown_minutes", 45),
                )
            except Exception as e:
                log.debug(f"[{coin}] Narrative gate skipped: {e}")

        # Trade memory adjustment
        current_pos = self.risk.position_direction(coin)
        prelim_dir  = "LONG" if tech.rsi_score >= 50 else "SHORT"
        # Extract current regime context to pass into RL for regime-aware learning
        _dom_regime  = regimes.dominant_regime if regimes else "MIXED"
        _msb_struct  = (advanced.msb.structure_trend if advanced and advanced.valid else "RANGING")
        memory_adj   = self._memory.get_score_adjustment(
            coin, prelim_dir, tech.rsi_score,
            market_regime   = _msb_struct,
            dominant_regime = _dom_regime,
        )

        # Determine instrument type for this coin
        instrument_type = self.cfg.trading.instrument_types.get(
            coin,
            hyperliquid_instrument_type(coin, "crypto"),
        )

        # Funding Rate / OI / CVD — order-flow intelligence
        # Only computed for crypto perps (not indexes — no funding on SP500 Yahoo data)
        funding_oi_signal = None
        if instrument_type == "crypto":
            try:
                funding_oi_signal = get_funding_oi_cvd(coin, analysis_df)
            except Exception as e:
                log.debug(f"[{coin}] FundingOI skipped: {e}")

        # Live orderbook + higher-timeframe key levels
        orderbook_signal = None
        supports_orderbook = self._supports_orderbook_context(coin)
        if supports_orderbook and getattr(self.cfg.trading, "use_orderbook_levels", True):
            try:
                orderbook_signal = self._orderbook_circuit.call(
                    get_orderbook_levels,
                    coin,
                    current_price=tech.price,
                    depth_limit=getattr(self.cfg.trading, "orderbook_depth_limit", 120),
                    daily_lookback=getattr(self.cfg.trading, "orderbook_daily_lookback", 120),
                    cache_ttl_seconds=getattr(self.cfg.trading, "orderbook_cache_ttl_seconds", 25),
                    feed_max_age_seconds=getattr(self.cfg.trading, "orderbook_feed_max_snapshot_age_seconds", 45.0),
                    feed_breakout_samples=getattr(self.cfg.trading, "orderbook_feed_breakout_samples", 2),
                    guard_distance_pct=getattr(self.cfg.trading, "orderbook_guard_distance_pct", 1.25),
                    reaction_distance_pct=getattr(self.cfg.trading, "orderbook_reaction_distance_pct", 0.45),
                )
            except Exception as e:
                log.debug(f"[{coin}] Orderbook levels skipped: {e}")
        self._track_orderbook_snapshot(coin, orderbook_signal)

        market_map_signal = None
        if getattr(self.cfg.trading, "use_daily_market_map", True):
            try:
                market_map_signal = market_map.get_market_map_signal(
                    coin,
                    current_price=tech.price,
                    closed_price=getattr(tech, "closed_price", tech.price),
                )
            except Exception as e:
                log.debug(f"[{coin}] Market map skipped: {e}")

        # Generate signal  (LONG / SHORT / FLAT)
        signal = self.strategy.generate_signal(
            tech, advanced, sentiment, current_pos, regimes,
            news_signal=news_signal,
            candle_patterns=candle_patterns,
            memory_adjustment=memory_adj,
            instrument_type=instrument_type,
            funding_oi_signal=funding_oi_signal,
            orderbook_signal=orderbook_signal,
            market_map_signal=market_map_signal,
            narrative_signal=narrative_signal,
        )

        log.info(
            f"[{coin}] Signal={signal.action} score={signal.score:.1f} "
            f"confidence={signal.confidence}"
        )

        rl_stats = self._memory.get_stats().get(coin, {})
        rl_total_trades = int(rl_stats.get("total", 0) or 0)
        rl_pattern_boost = self._memory.get_pattern_boost(coin, signal.action, signal.score)
        rl_win_rate_for_sizing = (
            rl_stats.get("win_rate", 50.0)
            if rl_total_trades >= 3 else 50.0
        )
        long_guard = self._memory.get_directional_guard(coin, "LONG")
        short_guard = self._memory.get_directional_guard(coin, "SHORT")
        top_root_causes = sorted(
            (rl_stats.get("root_causes") or {}).items(),
            key=lambda item: (-item[1], item[0]),
        )[:3]

        # Store signal for dashboard + memory (enriched with new intelligence)
        trade_plan = dict(getattr(signal, "trade_plan", {}) or {})
        thesis = dict(getattr(signal, "thesis", {}) or {})
        self._last_signals[coin] = {
            "action":          signal.action,
            "decision":        signal.action,
            "score":           signal.score,
            "confidence":      signal.confidence,
            "price":           signal.price,
            "analysis_price":  analysis_price,
            "live_price":      live_price,
            "using_closed_candles": bool(getattr(self.cfg.trading, "use_closed_candles_for_conviction", True)),
            "reason":          signal.reason,
            "flat_reason":     signal.flat_reason,
            "decision_reason": signal.reason or signal.flat_reason or "",
            "instrument_type": instrument_type,
            "mtf_bias":        "FLAT",   # updated below if MTF runs
            # Candlestick patterns
            "candle_score":    candle_patterns.score    if candle_patterns and candle_patterns.valid else 50.0,
            "candle_patterns": candle_patterns.patterns if candle_patterns and candle_patterns.valid else [],
            "candle_trend":    candle_patterns.trend_3  if candle_patterns and candle_patterns.valid else "FLAT",
            # News intelligence
            "news_score":     news_signal.score         if news_signal and news_signal.valid else 50.0,
            "news_velocity":  news_signal.velocity      if news_signal and news_signal.valid else "LOW",
            "news_headline":  news_signal.top_headlines[0][:80] if news_signal and news_signal.valid and news_signal.top_headlines else "",
            "news_articles":  news_signal.article_count if news_signal and news_signal.valid else 0,
            "news_catalyst_score": getattr(news_signal, "catalyst_score", 0.0) if news_signal and news_signal.valid else 0.0,
            "news_catalyst_summary": getattr(news_signal, "catalyst_summary", "") if news_signal and news_signal.valid else "",
            "narrative_summary": getattr(narrative_signal, "summary", "") if narrative_signal else "",
            "narrative_event_risk_active": bool(getattr(narrative_signal, "event_risk_active", False)) if narrative_signal else False,
            "narrative_event_name": getattr(narrative_signal, "event_name", "") if narrative_signal else "",
            "narrative_event_importance": getattr(narrative_signal, "event_importance", "NONE") if narrative_signal else "NONE",
            "narrative_minutes_to_event": getattr(narrative_signal, "minutes_to_event", None) if narrative_signal else None,
            "narrative_headline_bias": getattr(narrative_signal, "headline_bias", "NEUTRAL") if narrative_signal else "NEUTRAL",
            "narrative_score_adjustment": getattr(narrative_signal, "score_adjustment", 0.0) if narrative_signal else 0.0,
            "narrative_uncertainty_delta": getattr(narrative_signal, "uncertainty_delta", 0.0) if narrative_signal else 0.0,
            # Memory / learning
            "memory_adj":      memory_adj,
            "memory_cooldown": self._memory._cooldown.get(coin, 0),
            "rl_total_trades": rl_total_trades,
            "rl_win_rate":     rl_stats.get("win_rate"),
            "rl_long_wr":      rl_stats.get("long_wr"),
            "rl_short_wr":     rl_stats.get("short_wr"),
            "rl_last_5_results": rl_stats.get("last_5_results", []),
            "rl_pattern_boost": rl_pattern_boost,
            "rl_latest_failure_summary": rl_stats.get("latest_failure_summary", ""),
            "rl_top_root_causes": [cause for cause, _ in top_root_causes],
            "rl_long_guard": long_guard,
            "rl_short_guard": short_guard,
            # Regime context — stored for RL record_trade on close
            "market_regime":   _msb_struct,
            "dominant_regime": _dom_regime,
            "volatility_label": advanced.atr.volatility_label if advanced and advanced.valid else "NORMAL",
            "msb_type":        advanced.msb.msb_type if advanced and advanced.valid else "NONE",
            "structure_trend": advanced.msb.structure_trend if advanced and advanced.valid else "RANGING",
            "swing_high":      advanced.msb.last_swing_high if advanced and advanced.valid else 0.0,
            "swing_low":       advanced.msb.last_swing_low if advanced and advanced.valid else 0.0,
            "atr_pct":         advanced.atr.atr_pct if advanced and advanced.valid else 0.0,
            # Funding / OI / CVD
            "funding_rate":    funding_oi_signal.funding_rate      if funding_oi_signal and funding_oi_signal.valid else 0.0,
            "funding_label":   funding_oi_signal.funding_label     if funding_oi_signal and funding_oi_signal.valid else "N/A",
            "oi_change_pct":   funding_oi_signal.oi_change_pct     if funding_oi_signal and funding_oi_signal.valid else 0.0,
            "cvd_divergence":  funding_oi_signal.cvd_divergence    if funding_oi_signal and funding_oi_signal.valid else "NONE",
            "foc_score":       funding_oi_signal.composite_score   if funding_oi_signal and funding_oi_signal.valid else 50.0,
            # Orderbook + key levels
            "orderbook_score": orderbook_signal.score if orderbook_signal and orderbook_signal.valid else 50.0,
            "orderbook_imbalance": orderbook_signal.imbalance_ratio if orderbook_signal and orderbook_signal.valid else 0.0,
            "orderbook_imbalance_mean": orderbook_signal.imbalance_mean if orderbook_signal and orderbook_signal.valid else 0.0,
            "orderbook_imbalance_trend": orderbook_signal.imbalance_trend if orderbook_signal and orderbook_signal.valid else 0.0,
            "orderbook_imbalance_volatility": orderbook_signal.imbalance_volatility if orderbook_signal and orderbook_signal.valid else 0.0,
            "orderbook_interaction": orderbook_signal.level_interaction if orderbook_signal and orderbook_signal.valid else "BETWEEN_LEVELS",
            "orderbook_breakout_state": orderbook_signal.breakout_state if orderbook_signal and orderbook_signal.valid else "NONE",
            "orderbook_intracycle_breakout_state": orderbook_signal.intracycle_breakout_state if orderbook_signal and orderbook_signal.valid else "NONE",
            "orderbook_support": orderbook_signal.nearest_support if orderbook_signal and orderbook_signal.valid else 0.0,
            "orderbook_support_distance_pct": orderbook_signal.nearest_support_distance_pct if orderbook_signal and orderbook_signal.valid else 0.0,
            "orderbook_resistance": orderbook_signal.nearest_resistance if orderbook_signal and orderbook_signal.valid else 0.0,
            "orderbook_resistance_distance_pct": orderbook_signal.nearest_resistance_distance_pct if orderbook_signal and orderbook_signal.valid else 0.0,
            "orderbook_support_strength": orderbook_signal.nearest_support_strength if orderbook_signal and orderbook_signal.valid else 0.0,
            "orderbook_resistance_strength": orderbook_signal.nearest_resistance_strength if orderbook_signal and orderbook_signal.valid else 0.0,
            "orderbook_support_wall_persistence": orderbook_signal.support_wall_persistence if orderbook_signal and orderbook_signal.valid else 0,
            "orderbook_resistance_wall_persistence": orderbook_signal.resistance_wall_persistence if orderbook_signal and orderbook_signal.valid else 0,
            "orderbook_feed_age_seconds": orderbook_signal.feed_age_seconds if orderbook_signal and orderbook_signal.valid else 0.0,
            "orderbook_feed_snapshot_count": orderbook_signal.feed_snapshot_count if orderbook_signal and orderbook_signal.valid else 0,
            "orderbook_favor_longs": orderbook_signal.favor_longs if orderbook_signal and orderbook_signal.valid else False,
            "orderbook_favor_shorts": orderbook_signal.favor_shorts if orderbook_signal and orderbook_signal.valid else False,
            "orderbook_block_longs": orderbook_signal.block_longs if orderbook_signal and orderbook_signal.valid else False,
            "orderbook_block_shorts": orderbook_signal.block_shorts if orderbook_signal and orderbook_signal.valid else False,
            "daily_breakout_level": orderbook_signal.daily_breakout_level if orderbook_signal and orderbook_signal.valid else 0.0,
            "daily_breakdown_level": orderbook_signal.daily_breakdown_level if orderbook_signal and orderbook_signal.valid else 0.0,
            "orderbook_valid": bool(orderbook_signal and orderbook_signal.valid),
            "market_map_available": bool(market_map_signal and getattr(market_map_signal, "valid", False)),
            "market_map_bias": getattr(market_map_signal, "bias", "NEUTRAL") if market_map_signal else "NEUTRAL",
            "market_map_summary": getattr(market_map_signal, "summary", "") if market_map_signal else "",
            "market_map_score_adjustment": getattr(market_map_signal, "score_adjustment", 0.0) if market_map_signal else 0.0,
            "market_map_favor_longs": getattr(market_map_signal, "favor_longs", False) if market_map_signal else False,
            "market_map_favor_shorts": getattr(market_map_signal, "favor_shorts", False) if market_map_signal else False,
            "market_map_block_longs": getattr(market_map_signal, "block_longs", False) if market_map_signal else False,
            "market_map_block_shorts": getattr(market_map_signal, "block_shorts", False) if market_map_signal else False,
            "market_map_reclaim_confirmed": bool(getattr(market_map_signal, "above_reclaim_levels", [])) if market_map_signal else False,
            "market_map_live_reclaim": bool(getattr(market_map_signal, "live_above_reclaim_levels", [])) if market_map_signal else False,
            "market_map_reclaim_lost": (
                bool(getattr(market_map_signal, "above_reclaim_levels", []))
                and not bool(getattr(market_map_signal, "live_above_reclaim_levels", []))
            ) if market_map_signal else False,
            "market_map_breakdown_confirmed": bool(getattr(market_map_signal, "below_breakdown_levels", [])) if market_map_signal else False,
            "market_map_live_breakdown": bool(getattr(market_map_signal, "live_below_breakdown_levels", [])) if market_map_signal else False,
            "market_map_daily_close": getattr(market_map_signal, "daily_close", 0.0) if market_map_signal else 0.0,
            "market_map_nearest_support": getattr(market_map_signal, "nearest_support", 0.0) if market_map_signal else 0.0,
            "market_map_nearest_resistance": getattr(market_map_signal, "nearest_resistance", 0.0) if market_map_signal else 0.0,
            "market_map_notes": getattr(market_map_signal, "notes", "") if market_map_signal else "",
            "planned_stop_loss": signal.stop_loss_price,
            "planned_take_profit": signal.take_profit_price,
            "planned_risk_pct": trade_plan.get("risk_pct", 0.0),
            "planned_reward_pct": trade_plan.get("reward_pct", 0.0),
            "planned_risk_reward_ratio": trade_plan.get("risk_reward_ratio", 0.0),
            "planned_stop_atr_multiple": trade_plan.get("stop_atr_multiple", 0.0),
            "planned_target_atr_multiple": trade_plan.get("target_atr_multiple", 0.0),
            "planned_target_r_multiple": trade_plan.get("target_r_multiple", 0.0),
            "stop_basis":      trade_plan.get("stop_basis", ""),
            "target_basis":    trade_plan.get("target_basis", ""),
            "price_action_summary": trade_plan.get("price_action_summary", ""),
            "thesis_candidate_action": thesis.get("candidate_action", signal.action),
            "thesis_state":    thesis.get("state", "NO_TRADE"),
            "thesis_permitted": thesis.get("permitted", signal.action in ("LONG", "SHORT")),
            "thesis_quality":  thesis.get("quality", signal.confidence),
            "thesis_alignment_points": thesis.get("alignment_points", 0.0),
            "thesis_conflict_points": thesis.get("conflict_points", 0.0),
            "thesis_conviction_score": thesis.get("conviction_score", signal.score),
            "thesis_summary":  thesis.get("summary", ""),
            "thesis_reasons":  thesis.get("reasons", []),
            "thesis_blockers": thesis.get("blockers", []),
            "thesis":          thesis,
            "expectancy_probability": getattr(signal, "expectancy", {}).get("probability", 0.50),
            "expectancy_expected_r": getattr(signal, "expectancy", {}).get("expected_r", 0.0),
            "expectancy_uncertainty": getattr(signal, "expectancy", {}).get("uncertainty", 0.50),
            "expectancy_score": getattr(signal, "expectancy", {}).get("score", signal.score),
            "expectancy_summary": getattr(signal, "expectancy", {}).get("summary", ""),
            "expectancy_reasons": getattr(signal, "expectancy", {}).get("reasons", []),
            "expectancy_blockers": getattr(signal, "expectancy", {}).get("blockers", []),
            "expectancy": getattr(signal, "expectancy", {}),
            "execution_plan": getattr(signal, "execution_plan", {}),
            "trade_plan":      trade_plan,
            "execution_quality": {},
            "execution_quality_score": 0.0,
            "execution_quality_summary": "",
            "execution_coach": {},
            "execution_coach_used": False,
            "execution_coach_verdict": "",
            "execution_coach_summary": "",
            "execution_coach_urgency_score": 0.0,
            "execution_coach_stretch_bps": 0.0,
            "estimated_slippage_bps": 0.0,
            "execution_persistence_cycles": 0,
            "execution_mode":  "tradable" if coin in self._tradable_coin_set else "observation_only",
            "decision_stage": "analysis",
            "streak_confirmation_remaining": 0,
            "data_reliability": {},
            "data_reliability_score": 0.0,
            "data_reliability_quality": "UNKNOWN",
            "data_reliability_summary": "",
            "portfolio_guard": {},
            "portfolio_theme": "",
            "portfolio_guard_summary": "",
            "portfolio_guard_size_multiplier": 1.0,
            "asset_state": "OBSERVING",
            "asset_state_label": "Observing",
            "next_unblock_reason": "",
            "llm_referee": {},
            "llm_referee_summary": "",
            "llm_referee_why_now": "",
        }
        self._refresh_asset_state(coin, stage="analysis", current_position=current_pos)
        self._apply_analog_context(
            coin,
            signal,
            current_pos,
            orderbook_signal=orderbook_signal,
            market_map_signal=market_map_signal,
        )

        mtf_allows_signal = True
        if self.cfg.trading.use_mtf and (current_pos or signal.action != "FLAT"):
            mtf_allows_signal = self._check_mtf_safe(coin, signal)

        reliability = data_reliability.assess_reliability(self.cfg.trading, self._last_signals.get(coin, {}))
        self._last_signals[coin]["data_reliability"] = reliability
        self._last_signals[coin]["data_reliability_score"] = reliability.get("score", 0.0)
        self._last_signals[coin]["data_reliability_quality"] = reliability.get("quality", "UNKNOWN")
        self._last_signals[coin]["data_reliability_summary"] = reliability.get("summary", "")
        self._last_signals[coin]["data_reliability_blockers"] = reliability.get("blockers", [])
        self._last_signals[coin]["data_reliability_issues"] = reliability.get("issues", [])
        self._last_signals[coin]["data_reliability_price_gap_pct"] = reliability.get("price_gap_pct", 0.0)
        if (
            getattr(self.cfg.trading, "data_reliability_enabled", True)
            and not current_pos
            and signal.action in {"LONG", "SHORT"}
            and not bool(reliability.get("permitted", True))
        ):
            reason = str(reliability.get("summary", "") or "data quality is not reliable enough yet")
            log.info(f"[{coin}] 🛰️ Data-reliability gate blocks {signal.action}: {reason}")
            signal.action = "FLAT"
            signal.flat_reason = reason
            signal.reason = reason
            self._sync_signal_snapshot(coin, signal)
            self._record_decision_snapshot(
                coin,
                portfolio_usd=portfolio_usd,
                stage="data_reliability_block",
                signal=signal,
                current_position=current_pos,
                blocked=True,
            )
            return

        if current_pos:
            invalidation_reason = self._detect_position_invalidation(coin, current_pos, signal)
            if invalidation_reason:
                log.info(
                    f"[{coin}] 🧭 Thesis invalidated — closing {current_pos} via {invalidation_reason}"
                )
                self._close_position(coin, invalidation_reason, signal.price)
                self._flat_streak.pop(coin, None)
                self._signal_streak.pop(coin, None)
                self._record_decision_snapshot(
                    coin,
                    portfolio_usd=portfolio_usd,
                    stage="position_invalidation_close",
                    signal=signal,
                    current_position=current_pos,
                    blocked=True,
                )
                return

        if signal.action == "FLAT":
            if current_pos:
                flat_count = self._flat_streak.get(coin, 0) + 1
                self._flat_streak[coin] = flat_count
                decay = self._assess_conviction_decay(coin, current_pos, signal)
                self._last_signals[coin]["conviction_decay"] = decay
                if decay.get("should_exit", False):
                    log.info(
                        f"[{coin}] 🏳️ Conviction decay {decay.get('score', 0):.0f} "
                        f"— closing {current_pos}. {decay.get('summary', '')}"
                    )
                    self._close_position(coin, "conviction_lost", signal.price)
                    self._flat_streak.pop(coin, None)
                    self._signal_streak.pop(coin, None)
                    self._record_decision_snapshot(
                        coin,
                        portfolio_usd=portfolio_usd,
                        stage="conviction_exit",
                        signal=signal,
                        current_position=current_pos,
                        blocked=True,
                    )
                else:
                    log.info(
                        f"[{coin}] 🏳️ FLAT while {current_pos} open — "
                        f"decay {decay.get('score', 0):.0f}. {decay.get('summary', '')}"
                    )
                    self._record_decision_snapshot(
                        coin,
                        portfolio_usd=portfolio_usd,
                        stage="hold_existing_position",
                        signal=signal,
                        current_position=current_pos,
                    )
            else:
                self._flat_streak.pop(coin, None)
                self._record_decision_snapshot(
                    coin,
                    portfolio_usd=portfolio_usd,
                    stage="flat_no_trade",
                    signal=signal,
                    blocked=True,
                )
            return

        # ── Multi-timeframe confirmation ────────────────────────
        if mtf_allows_signal is False:
            self._record_decision_snapshot(
                coin,
                portfolio_usd=portfolio_usd,
                stage="mtf_block",
                signal=signal,
                current_position=current_pos,
                blocked=True,
            )
            return

        if (
            supports_orderbook
            and getattr(self.cfg.trading, "use_orderbook_levels", True)
            and (
                getattr(self.cfg.trading, "require_orderbook_for_crypto_entries", True)
                if instrument_type == "crypto"
                else getattr(self.cfg.trading, "require_orderbook_for_supported_entries", True)
            )
            and (not orderbook_signal or not orderbook_signal.valid)
        ):
            log.info(f"[{coin}] 🧱 Missing valid orderbook context — fail-closed, skipping entry")
            self._record_decision_snapshot(
                coin,
                portfolio_usd=portfolio_usd,
                stage="missing_orderbook_context",
                signal=signal,
                current_position=current_pos,
                blocked=True,
            )
            return

        if (
            self.cfg.trading.use_news
            and getattr(self.cfg.trading, "strict_confirmation_fail_closed", True)
            and (not news_signal or not news_signal.valid)
        ):
            log.info(f"[{coin}] 📰 Missing valid news confirmation — fail-closed, skipping entry")
            self._record_decision_snapshot(
                coin,
                portfolio_usd=portfolio_usd,
                stage="missing_news_confirmation",
                signal=signal,
                current_position=current_pos,
                blocked=True,
            )
            return

        # ── News extreme-event gate (secondary safety net) ───────
        # Note: news is already factored into the signal score above.
        # This gate only blocks if news is catastrophically against the signal.
        if self.cfg.trading.use_news and news_signal:
            if not self._check_news_extreme(coin, signal, news_signal):
                self._record_decision_snapshot(
                    coin,
                    portfolio_usd=portfolio_usd,
                    stage="extreme_news_block",
                    signal=signal,
                    current_position=current_pos,
                    blocked=True,
                )
                return

        if not self._check_narrative_gate(coin, signal, narrative_signal):
            self._record_decision_snapshot(
                coin,
                portfolio_usd=portfolio_usd,
                stage="narrative_gate_block",
                signal=signal,
                current_position=current_pos,
                blocked=True,
            )
            return

        # ── Chart confirmation on borderline signals ────────────
        if self.cfg.trading.use_chart_confirmation:
            chart_verdict = self._get_chart_confirmation(coin, signal.score)
            if not chart_verdict or not chart_verdict.valid:
                if getattr(self.cfg.trading, "strict_confirmation_fail_closed", True):
                    log.info(f"[{coin}] 👁️  Chart confirmation unavailable — fail-closed, skipping entry")
                    self._record_decision_snapshot(
                        coin,
                        portfolio_usd=portfolio_usd,
                        stage="chart_confirmation_unavailable",
                        signal=signal,
                        current_position=current_pos,
                        blocked=True,
                    )
                    return
            elif chart_verdict.action == "WAIT":
                log.info(f"[{coin}] 👁️  Chart analyst says WAIT "
                         f"({chart_verdict.confidence}) — skipping this cycle. "
                         f"Reason: {chart_verdict.reasoning[:80]}")
                self._record_decision_snapshot(
                    coin,
                    portfolio_usd=portfolio_usd,
                    stage="chart_wait",
                    signal=signal,
                    current_position=current_pos,
                    blocked=True,
                )
                return
            elif chart_verdict.action != signal.action:
                log.info(
                    f"[{coin}] 👁️  Chart analyst disagrees: "
                    f"indicators={signal.action} chart={chart_verdict.action} "
                    f"— skipping. Reason: {chart_verdict.reasoning[:80]}"
                )
                self._record_decision_snapshot(
                    coin,
                    portfolio_usd=portfolio_usd,
                    stage="chart_disagrees",
                    signal=signal,
                    current_position=current_pos,
                    blocked=True,
                )
                return
            else:
                log.info(
                    f"[{coin}] 👁️  Chart analyst CONFIRMS {signal.action} "
                    f"({chart_verdict.confidence}) — proceeding."
                )

        # ── RL directional guardrails ─────────────────────────
        if signal.action in ("LONG", "SHORT"):
            guard = long_guard if signal.action == "LONG" else short_guard
            self._last_signals[coin]["rl_active_guard"] = guard
            self._last_signals[coin]["rl_threshold_boost"] = guard.get("threshold_boost", 0.0)
            self._last_signals[coin]["rl_pause_cycles"] = guard.get("pause_cycles", 0)
            self._last_signals[coin]["rl_guard_reasons"] = guard.get("reasons", [])
            self._last_signals[coin]["rl_hard_block"] = guard.get("hard_block", False)
            self._last_signals[coin]["rl_hard_block_reason"] = guard.get("hard_block_reason", "")

            if guard.get("hard_block", False):
                block_reason = str(guard.get("hard_block_reason", "") or f"RL embargo on {signal.action}")
                log.info(f"[{coin}] 🚫 {block_reason}")
                signal.action = "FLAT"
                signal.flat_reason = block_reason
                signal.reason = block_reason
            elif guard.get("pause_cycles", 0) > 0:
                pause_reason = (
                    f"RL pause on {signal.action}: {guard['pause_cycles']} cycles left"
                    + (f" ({', '.join(guard['reasons'])})" if guard.get("reasons") else "")
                )
                log.info(f"[{coin}] ⏸️ {pause_reason}")
                signal.action = "FLAT"
                signal.flat_reason = pause_reason
                signal.reason = pause_reason
            else:
                threshold_boost = float(guard.get("threshold_boost", 0.0) or 0.0)
                if threshold_boost > 0:
                    if signal.action == "LONG":
                        required_score = self.cfg.trading.signal_long_threshold + threshold_boost
                        self._last_signals[coin]["rl_required_score"] = required_score
                        if signal.score < required_score:
                            tighten_reason = (
                                f"RL tightened LONG threshold to ≥{required_score:.0f}"
                                + (f" ({', '.join(guard['reasons'])})" if guard.get("reasons") else "")
                            )
                            log.info(f"[{coin}] 🧠 {tighten_reason}")
                            signal.action = "FLAT"
                            signal.flat_reason = tighten_reason
                            signal.reason = tighten_reason
                    else:
                        required_score = self.cfg.trading.signal_short_threshold - threshold_boost
                        self._last_signals[coin]["rl_required_score"] = required_score
                        if signal.score > required_score:
                            tighten_reason = (
                                f"RL tightened SHORT threshold to ≤{required_score:.0f}"
                                + (f" ({', '.join(guard['reasons'])})" if guard.get("reasons") else "")
                            )
                            log.info(f"[{coin}] 🧠 {tighten_reason}")
                            signal.action = "FLAT"
                            signal.flat_reason = tighten_reason
                            signal.reason = tighten_reason

            self._sync_signal_snapshot(coin, signal)

        if signal.action in ("LONG", "SHORT"):
            review_guard = trade_review.get_directional_feedback(coin, signal.action)
            self._last_signals[coin]["operator_review_guard"] = review_guard
            self._last_signals[coin]["operator_review_reasons"] = review_guard.get("reasons", [])
            self._last_signals[coin]["operator_review_score_adjustment"] = review_guard.get("score_adjustment", 0.0)
            if review_guard.get("hard_block", False):
                block_reason = str(review_guard.get("reason", "") or f"operator review blocks {signal.action}")
                log.info(f"[{coin}] 🧾 {block_reason}")
                signal.action = "FLAT"
                signal.flat_reason = block_reason
                signal.reason = block_reason
            else:
                review_adj = float(review_guard.get("score_adjustment", 0.0) or 0.0)
                if review_adj != 0.0:
                    effective_score = signal.score + review_adj
                    long_threshold = float(self.cfg.trading.signal_long_threshold)
                    short_threshold = float(self.cfg.trading.signal_short_threshold)
                    if signal.action == "LONG" and effective_score < long_threshold:
                        review_reason = (
                            f"operator review trims LONG to {effective_score:.1f} "
                            f"(< {long_threshold:.0f})"
                        )
                        log.info(f"[{coin}] 🧾 {review_reason}")
                        signal.action = "FLAT"
                        signal.flat_reason = review_reason
                        signal.reason = review_reason
                    elif signal.action == "SHORT" and effective_score > short_threshold:
                        review_reason = (
                            f"operator review lifts SHORT to {effective_score:.1f} "
                            f"(> {short_threshold:.0f})"
                        )
                        log.info(f"[{coin}] 🧾 {review_reason}")
                        signal.action = "FLAT"
                        signal.flat_reason = review_reason
                        signal.reason = review_reason

            self._sync_signal_snapshot(coin, signal)

        if signal.action in ("LONG", "SHORT"):
            if self._apply_llm_referee(coin, signal, current_position=current_pos):
                self._record_decision_snapshot(
                    coin,
                    portfolio_usd=portfolio_usd,
                    stage="llm_referee_block",
                    signal=signal,
                    current_position=current_pos,
                    blocked=True,
                )
                return

        if signal.action == "FLAT":
            log.info(f"[{coin}] Guardrails keep this setup flat for now")
            self._record_decision_snapshot(
                coin,
                portfolio_usd=portfolio_usd,
                stage="guardrails_flat",
                signal=signal,
                current_position=current_pos,
                blocked=True,
            )
            return

        if coin not in self._tradable_coin_set:
            log.info(f"[{coin}] Observation-only asset — signal tracked, no execution on the live venue")
            self._record_decision_snapshot(
                coin,
                portfolio_usd=portfolio_usd,
                stage="observation_only",
                signal=signal,
            )
            return

        # ── Loss-based circuit breaker check ────────────────
        if self.risk.is_trading_halted():
            log.info(f"[{coin}] ⏸️  Skipping — loss circuit breaker active")
            self._record_decision_snapshot(
                coin,
                portfolio_usd=portfolio_usd,
                stage="loss_circuit_breaker",
                signal=signal,
                current_position=current_pos,
                blocked=True,
            )
            return

        if signal.action == "SHORT" and not current_pos and not hyperliquid_supports_shorts(coin):
            reason = f"{coin} is running as a long-only Hyperliquid spot market — short entries are disabled"
            log.info(f"[{coin}] 🛡️ {reason}")
            signal.action = "FLAT"
            signal.flat_reason = reason
            signal.reason = reason
            self._sync_signal_snapshot(coin, signal)
            self._record_decision_snapshot(
                coin,
                portfolio_usd=portfolio_usd,
                stage="long_only_short_block",
                signal=signal,
                current_position=current_pos,
                blocked=True,
            )
            return

        # Cancel any re-entry watch if we're opening a fresh signal
        if self.order_mgr.has_watch(coin):
            self.order_mgr.cancel_reentry_watch(coin, reason="new signal overrides watch")

        # ── Post-reversal cooldown: skip this cycle if we just closed ──────────
        # Prevents the whipsaw pattern: close LONG → immediately open SHORT.
        # After any signal_reversal close we sit out one full cycle (~2 min).
        reversal_ts = self._reversal_cooldown.get(coin, 0)
        if reversal_ts and (time.time() - reversal_ts) < self.cfg.trading.check_interval_seconds:
            log.info(
                f"[{coin}] ⏸ Post-reversal cooldown — sitting out this cycle "
                f"(closed {(time.time()-reversal_ts):.0f}s ago)"
            )
            self._signal_streak.pop(coin, None)   # reset streak too
            self._record_decision_snapshot(
                coin,
                portfolio_usd=portfolio_usd,
                stage="post_reversal_cooldown",
                signal=signal,
                current_position=current_pos,
                blocked=True,
            )
            return

        # ── Anti-whipsaw: guard signal reversals ─────────────────────────────
        if current_pos and current_pos != signal.action:
            # Check minimum hold time
            pos = self.risk.positions.get(coin)
            if pos and pos.opened_at:
                hold_minutes = (time.time() - pos.opened_at) / 60.0
                # Indexes need longer hold — they move slower than crypto
                if instrument_type == "index":
                    min_hold = self.cfg.trading.index_min_hold_minutes
                elif instrument_type == "equity":
                    min_hold = getattr(self.cfg.trading, "equity_min_hold_minutes", self.cfg.trading.min_hold_minutes)
                else:
                    min_hold = self.cfg.trading.min_hold_minutes
                if hold_minutes < min_hold:
                    log.info(
                        f"[{coin}] ⏳ Anti-whipsaw: position held only "
                        f"{hold_minutes:.0f}m (min={min_hold:.0f}m) — "
                        f"blocking reversal {current_pos}→{signal.action}"
                    )
                    self._record_decision_snapshot(
                        coin,
                        portfolio_usd=portfolio_usd,
                        stage="anti_whipsaw_hold_block",
                        signal=signal,
                        current_position=current_pos,
                        blocked=True,
                    )
                    return

            # Check reversal conviction — needs stronger signal than fresh entry
            reversal_boost = self.cfg.trading.reversal_threshold_boost
            if signal.action == "LONG":
                required_score = self.cfg.trading.signal_long_threshold + reversal_boost
                if signal.score < required_score:
                    log.info(
                        f"[{coin}] ⚡ Reversal blocked: SHORT→LONG needs score "
                        f"≥{required_score:.0f} but got {signal.score:.1f}"
                    )
                    self._record_decision_snapshot(
                        coin,
                        portfolio_usd=portfolio_usd,
                        stage="reversal_conviction_block",
                        signal=signal,
                        current_position=current_pos,
                        blocked=True,
                    )
                    return
            elif signal.action == "SHORT":
                required_score = self.cfg.trading.signal_short_threshold - reversal_boost
                if signal.score > required_score:
                    log.info(
                        f"[{coin}] ⚡ Reversal blocked: LONG→SHORT needs score "
                        f"≤{required_score:.0f} but got {signal.score:.1f}"
                    )
                    self._record_decision_snapshot(
                        coin,
                        portfolio_usd=portfolio_usd,
                        stage="reversal_conviction_block",
                        signal=signal,
                        current_position=current_pos,
                        blocked=True,
                    )
                    return

            log.info(f"[{coin}] Closing {current_pos} (score={signal.score:.1f}) "
                     f"— will re-evaluate next cycle before entering {signal.action}")
            self._close_position(coin, "signal_reversal", signal.price)
            # ✋ CRITICAL: do NOT open the opposite position immediately.
            # Set cooldown — next cycle will re-evaluate with fresh data.
            self._reversal_cooldown[coin] = time.time()
            self._signal_streak.pop(coin, None)
            self._flat_streak.pop(coin, None)
            self._record_decision_snapshot(
                coin,
                portfolio_usd=portfolio_usd,
                stage="signal_reversal_close",
                signal=signal,
                current_position=current_pos,
                blocked=True,
            )
            return

        # ── Directional signal: reset flat streak ──────────────────────────
        self._flat_streak.pop(coin, None)

        # ── Signal streak: require 2 consecutive agreeing cycles to enter ─────
        # Filters out single-candle noise crossings of the threshold.
        # Only applies to NEW entries (not managing existing positions).
        if not current_pos:
            streak = self._signal_streak.get(coin, {"action": None, "count": 0})
            if streak["action"] == signal.action:
                streak["count"] += 1
            else:
                streak = {"action": signal.action, "count": 1}
            self._signal_streak[coin] = streak

            required_streak = self.cfg.trading.signal_streak_required
            if streak["count"] < required_streak:
                self._last_signals[coin]["streak_confirmation_remaining"] = max(0, required_streak - streak["count"])
                log.info(
                    f"[{coin}] 🔁 Signal streak: {signal.action} confirmed "
                    f"{streak['count']}/{required_streak} cycles — waiting for confirmation"
                )
                self._record_decision_snapshot(
                    coin,
                    portfolio_usd=portfolio_usd,
                    stage="signal_streak_wait",
                    signal=signal,
                    blocked=True,
                )
                return
            else:
                log.info(
                    f"[{coin}] ✅ Signal streak reached {streak['count']}/{required_streak} "
                    f"— proceeding with {signal.action} entry"
                )
                self._last_signals[coin]["streak_confirmation_remaining"] = 0
                self._signal_streak.pop(coin, None)  # reset after entry

            cadence_allowed, cadence_reason = self._check_precision_entry_cadence(coin, signal)
            if not cadence_allowed:
                log.info(f"[{coin}] 🎯 Precision cadence block — {cadence_reason}")
                signal.action = "FLAT"
                signal.flat_reason = cadence_reason
                signal.reason = cadence_reason
                self._record_decision_snapshot(
                    coin,
                    portfolio_usd=portfolio_usd,
                    stage="precision_cadence_block",
                    signal=signal,
                    blocked=True,
                )
                return

        # ── Pull RL stats to inform position sizing ────────────────────────
        # Risk-check & size the order (conviction + RL win-rate aware)
        order = self.risk.compute_order(
            coin              = coin,
            direction         = signal.action,
            signal_score      = signal.score,
            current_price     = float((getattr(signal, "execution_plan", {}) or {}).get("entry_price", signal.price) or signal.price),
            stop_loss_price   = signal.stop_loss_price,
            take_profit_price = signal.take_profit_price,
            portfolio_usd     = portfolio_usd,
            rl_win_rate       = rl_win_rate_for_sizing,
            rl_pattern_boost  = rl_pattern_boost,
            expectancy_score  = float((getattr(signal, "expectancy", {}) or {}).get("score", signal.score) or signal.score),
            win_probability   = float((getattr(signal, "expectancy", {}) or {}).get("probability", 0.50) or 0.50),
            expected_r        = float((getattr(signal, "expectancy", {}) or {}).get("expected_r", 0.0) or 0.0),
            uncertainty       = float((getattr(signal, "expectancy", {}) or {}).get("uncertainty", 0.50) or 0.50),
            thesis_conviction = float((getattr(signal, "thesis", {}) or {}).get("conviction_score", signal.score) or signal.score),
        )

        if not order.approved:
            log.info(f"[{coin}] Rejected: {order.rejection_reason}")
            self._record_decision_snapshot(
                coin,
                portfolio_usd=portfolio_usd,
                stage="risk_rejected",
                signal=signal,
                current_position=current_pos,
                blocked=True,
            )
            return

        portfolio_theme_guard = portfolio_guard.assess_correlation(
            self.cfg.trading,
            coin=coin,
            direction=signal.action,
            instrument_type=str(self._last_signals.get(coin, {}).get("instrument_type", instrument_type) or instrument_type),
            portfolio_usd=float(portfolio_usd or 0.0),
            proposed_size_usd=float(getattr(order, "size_usd", 0.0) or 0.0),
            open_positions=list(self.risk.positions.values()),
            pending_orders=list(self.order_mgr.pending_orders.values()),
        )
        self._last_signals[coin]["portfolio_guard"] = portfolio_theme_guard
        self._last_signals[coin]["portfolio_theme"] = portfolio_theme_guard.get("theme", "")
        self._last_signals[coin]["portfolio_guard_summary"] = portfolio_theme_guard.get("summary", "")
        self._last_signals[coin]["portfolio_guard_size_multiplier"] = portfolio_theme_guard.get("size_multiplier", 1.0)
        self._last_signals[coin]["portfolio_guard_related_coins"] = portfolio_theme_guard.get("related_coins", [])
        self._last_signals[coin]["portfolio_guard_blockers"] = portfolio_theme_guard.get("blockers", [])
        self._last_signals[coin]["portfolio_guard_warnings"] = portfolio_theme_guard.get("warnings", [])

        if getattr(self.cfg.trading, "portfolio_correlation_guard_enabled", True):
            if not portfolio_theme_guard.get("permitted", True):
                reason = str(portfolio_theme_guard.get("summary", "") or "portfolio concentration is already too high")
                log.info(f"[{coin}] 🧺 Portfolio guard blocks entry: {reason}")
                signal.action = "FLAT"
                signal.flat_reason = reason
                signal.reason = reason
                self._sync_signal_snapshot(coin, signal)
                self._record_decision_snapshot(
                    coin,
                    portfolio_usd=portfolio_usd,
                    stage="portfolio_correlation_block",
                    signal=signal,
                    current_position=current_pos,
                    blocked=True,
                )
                return

            size_multiplier = float(portfolio_theme_guard.get("size_multiplier", 1.0) or 1.0)
            if size_multiplier < 0.999:
                original_size_usd = float(getattr(order, "size_usd", 0.0) or 0.0)
                trimmed_size_usd = original_size_usd * size_multiplier
                if trimmed_size_usd < float(self.cfg.trading.min_trade_usd or 0.0):
                    reason = (
                        f"{portfolio_theme_guard.get('theme', 'theme')} exposure trim would shrink the trade "
                        f"below the ${self.cfg.trading.min_trade_usd:.0f} minimum"
                    )
                    log.info(f"[{coin}] 🧺 Portfolio trim keeps trade flat: {reason}")
                    signal.action = "FLAT"
                    signal.flat_reason = reason
                    signal.reason = reason
                    self._sync_signal_snapshot(coin, signal)
                    self._record_decision_snapshot(
                        coin,
                        portfolio_usd=portfolio_usd,
                        stage="portfolio_correlation_block",
                        signal=signal,
                        current_position=current_pos,
                        blocked=True,
                    )
                    return
                order.size_usd = trimmed_size_usd
                order.size_coin = trimmed_size_usd / max(float(getattr(order, "price", signal.price) or signal.price), 1e-9)
                self._last_signals[coin]["portfolio_guard_summary"] = (
                    f"{portfolio_theme_guard.get('summary', '')} Size trimmed to ${trimmed_size_usd:.2f}."
                ).strip()

        execution_quality = self._assess_execution_quality(coin, signal.action, order, orderbook_signal)
        self._last_signals[coin]["execution_quality"] = execution_quality
        self._last_signals[coin]["execution_quality_score"] = execution_quality.get("score", 0.0)
        self._last_signals[coin]["execution_quality_summary"] = execution_quality.get("summary", "")
        self._last_signals[coin]["estimated_slippage_bps"] = execution_quality.get("estimated_slippage_bps", 0.0)
        self._last_signals[coin]["execution_persistence_cycles"] = execution_quality.get("persistence_cycles", 0)
        coached_execution = execution_coach.decide_execution(
            self.cfg.trading,
            coin=coin,
            signal_snapshot=self._last_signals.get(coin, {}),
            order=order,
            execution_quality=execution_quality,
            orderbook_signal=orderbook_signal,
        )
        self._last_signals[coin]["execution_coach"] = coached_execution
        self._last_signals[coin]["execution_coach_used"] = bool(coached_execution.get("enabled", False))
        self._last_signals[coin]["execution_coach_verdict"] = coached_execution.get("verdict", "")
        self._last_signals[coin]["execution_coach_summary"] = coached_execution.get("summary", "")
        self._last_signals[coin]["execution_coach_urgency_score"] = coached_execution.get("urgency_score", 0.0)
        self._last_signals[coin]["execution_coach_stretch_bps"] = coached_execution.get("stretch_bps", 0.0)
        coached_plan = dict(coached_execution.get("execution_plan") or {})
        if coached_plan:
            signal.execution_plan = coached_plan
            self._last_signals[coin]["execution_plan"] = coached_plan
        self._sync_signal_snapshot(coin, signal)
        if str(coached_execution.get("verdict") or "").upper() == "SKIP":
            reason = str(coached_execution.get("summary") or execution_quality.get("summary") or "execution coach skipped the entry").strip()
            log.info(f"[{coin}] 🎯 Execution coach keeps entry flat: {reason}")
            signal.action = "FLAT"
            signal.flat_reason = reason
            signal.reason = reason
            self._sync_signal_snapshot(coin, signal)
            self._record_decision_snapshot(
                coin,
                portfolio_usd=portfolio_usd,
                stage="execution_coach_skip",
                signal=signal,
                current_position=current_pos,
                blocked=True,
            )
            return

        execution_plan = dict(getattr(signal, "execution_plan", {}) or {})
        plan_mode = str(execution_plan.get("mode", "market") or "market").lower()
        if plan_mode in {"limit", "maker_limit"}:
            if self.order_mgr.has_pending(coin):
                log.info(f"[{coin}] Planned entry limit already pending — waiting for fill/cancel")
                self._record_decision_snapshot(
                    coin,
                    portfolio_usd=portfolio_usd,
                    stage="entry_limit_already_pending",
                    signal=signal,
                    current_position=current_pos,
                    blocked=True,
                    pending_limit=True,
                )
                return
            limit_price = float(execution_plan.get("limit_price", 0.0) or 0.0)
            if limit_price > 0:
                limit_result = self._place_limit_order(
                    coin,
                    signal.action,
                    limit_price,
                    order.size_usd,
                    order.stop_loss,
                    order.take_profit,
                    signal.score,
                    reason="initial_limit",
                    entry_context=self._build_entry_context(coin, signal, order, entry_type="initial_limit"),
                    trade_plan=dict(getattr(signal, "trade_plan", {}) or {}),
                    maker_only=(plan_mode == "maker_limit"),
                )
                log.info(
                    f"[{coin}] 📋 Planned {plan_mode} entry @ ${limit_price:.2f} "
                    f"({execution_plan.get('reason', 'execution plan')})"
                )
                if limit_result.get("success"):
                    self._record_precision_entry(coin, signal, mode="limit")
                self._record_decision_snapshot(
                    coin,
                    portfolio_usd=portfolio_usd,
                    stage="limit_entry_placed" if limit_result.get("pending") else ("limit_entry_opened" if limit_result.get("filled") else "limit_entry_failed"),
                    signal=signal,
                    current_position=current_pos,
                    executed=bool(limit_result.get("filled")),
                    blocked=not bool(limit_result.get("success")),
                    pending_limit=bool(limit_result.get("pending")),
                )
                return

        executed = self._execute_order(coin, signal, order)
        if executed:
            self._record_precision_entry(coin, signal, mode="market")
        self._record_decision_snapshot(
            coin,
            portfolio_usd=portfolio_usd,
            stage="market_entry_opened" if executed else "market_entry_failed",
            signal=signal,
            current_position=current_pos,
            executed=bool(executed),
            blocked=not bool(executed),
        )

    # ── Execution ─────────────────────────────────────────────

    def _track_orderbook_snapshot(self, coin: str, orderbook_signal) -> None:
        if not orderbook_signal or not getattr(orderbook_signal, "valid", False):
            return
        history = self._orderbook_history.setdefault(coin, [])
        history.append({
            "ts": time.time(),
            "breakout_state": getattr(orderbook_signal, "breakout_state", "NONE"),
            "level_interaction": getattr(orderbook_signal, "level_interaction", "BETWEEN_LEVELS"),
            "support": round(float(getattr(orderbook_signal, "nearest_support", 0.0) or 0.0), 4),
            "resistance": round(float(getattr(orderbook_signal, "nearest_resistance", 0.0) or 0.0), 4),
            "spread_bps": float(getattr(orderbook_signal, "spread_bps", 0.0) or 0.0),
            "bid_notional": float(getattr(orderbook_signal, "bid_notional", 0.0) or 0.0),
            "ask_notional": float(getattr(orderbook_signal, "ask_notional", 0.0) or 0.0),
        })
        max_points = max(4, int(getattr(self.cfg.trading, "min_orderbook_persistence_cycles", 2)) + 3)
        if len(history) > max_points:
            del history[:-max_points]

    def _orderbook_persistence_cycles(self, coin: str, direction: str, orderbook_signal) -> int:
        history = list(self._orderbook_history.get(coin, []))
        if not history or not orderbook_signal or not getattr(orderbook_signal, "valid", False):
            return 0

        direction = direction.upper()
        current_support = round(float(getattr(orderbook_signal, "nearest_support", 0.0) or 0.0), 4)
        current_resistance = round(float(getattr(orderbook_signal, "nearest_resistance", 0.0) or 0.0), 4)
        current_breakout = str(getattr(orderbook_signal, "breakout_state", "NONE") or "NONE")
        current_interaction = str(getattr(orderbook_signal, "level_interaction", "BETWEEN_LEVELS") or "BETWEEN_LEVELS")

        count = 0
        for snap in reversed(history):
            if str(snap.get("breakout_state", "NONE")) != current_breakout:
                break
            if str(snap.get("level_interaction", "BETWEEN_LEVELS")) != current_interaction:
                break
            if direction == "LONG":
                if current_support <= 0 or abs(float(snap.get("support", 0.0)) - current_support) > max(current_support * 0.0025, 1e-6):
                    break
            else:
                if current_resistance <= 0 or abs(float(snap.get("resistance", 0.0)) - current_resistance) > max(current_resistance * 0.0025, 1e-6):
                    break
            count += 1
        return count

    def _sync_signal_snapshot(self, coin: str, signal) -> None:
        if coin not in self._last_signals:
            return

        trade_plan = dict(getattr(signal, "trade_plan", {}) or {})
        thesis = dict(getattr(signal, "thesis", {}) or {})
        expectancy = dict(getattr(signal, "expectancy", {}) or {})
        execution_plan = dict(getattr(signal, "execution_plan", {}) or {})
        snap = self._last_signals[coin]
        snap.update({
            "action": signal.action,
            "decision": signal.action,
            "score": float(getattr(signal, "score", snap.get("score", 50.0)) or snap.get("score", 50.0)),
            "confidence": getattr(signal, "confidence", snap.get("confidence", "LOW")),
            "reason": getattr(signal, "reason", snap.get("reason", "")),
            "flat_reason": getattr(signal, "flat_reason", snap.get("flat_reason", "")),
            "decision_reason": getattr(signal, "reason", "") or getattr(signal, "flat_reason", "") or "",
            "planned_stop_loss": getattr(signal, "stop_loss_price", snap.get("planned_stop_loss", 0.0)),
            "planned_take_profit": getattr(signal, "take_profit_price", snap.get("planned_take_profit", 0.0)),
            "planned_risk_pct": trade_plan.get("risk_pct", snap.get("planned_risk_pct", 0.0)),
            "planned_reward_pct": trade_plan.get("reward_pct", snap.get("planned_reward_pct", 0.0)),
            "planned_risk_reward_ratio": trade_plan.get("risk_reward_ratio", snap.get("planned_risk_reward_ratio", 0.0)),
            "planned_stop_atr_multiple": trade_plan.get("stop_atr_multiple", snap.get("planned_stop_atr_multiple", 0.0)),
            "planned_target_atr_multiple": trade_plan.get("target_atr_multiple", snap.get("planned_target_atr_multiple", 0.0)),
            "planned_target_r_multiple": trade_plan.get("target_r_multiple", snap.get("planned_target_r_multiple", 0.0)),
            "stop_basis": trade_plan.get("stop_basis", snap.get("stop_basis", "")),
            "target_basis": trade_plan.get("target_basis", snap.get("target_basis", "")),
            "price_action_summary": trade_plan.get("price_action_summary", snap.get("price_action_summary", "")),
            "thesis_candidate_action": thesis.get("candidate_action", snap.get("thesis_candidate_action", signal.action)),
            "thesis_state": thesis.get("state", snap.get("thesis_state", "NO_TRADE")),
            "thesis_permitted": thesis.get("permitted", snap.get("thesis_permitted", False)),
            "thesis_quality": thesis.get("quality", snap.get("thesis_quality", snap.get("confidence", "LOW"))),
            "thesis_alignment_points": thesis.get("alignment_points", snap.get("thesis_alignment_points", 0.0)),
            "thesis_conflict_points": thesis.get("conflict_points", snap.get("thesis_conflict_points", 0.0)),
            "thesis_conviction_score": thesis.get("conviction_score", snap.get("thesis_conviction_score", signal.score)),
            "thesis_summary": thesis.get("summary", snap.get("thesis_summary", "")),
            "thesis_reasons": thesis.get("reasons", snap.get("thesis_reasons", [])),
            "thesis_blockers": thesis.get("blockers", snap.get("thesis_blockers", [])),
            "thesis": thesis,
            "expectancy_probability": expectancy.get("probability", snap.get("expectancy_probability", 0.50)),
            "expectancy_expected_r": expectancy.get("expected_r", snap.get("expectancy_expected_r", 0.0)),
            "expectancy_uncertainty": expectancy.get("uncertainty", snap.get("expectancy_uncertainty", 0.50)),
            "expectancy_score": expectancy.get("score", snap.get("expectancy_score", signal.score)),
            "expectancy_summary": expectancy.get("summary", snap.get("expectancy_summary", "")),
            "expectancy_reasons": expectancy.get("reasons", snap.get("expectancy_reasons", [])),
            "expectancy_blockers": expectancy.get("blockers", snap.get("expectancy_blockers", [])),
            "expectancy": expectancy,
            "execution_plan": execution_plan,
            "trade_plan": trade_plan,
        })
        self._refresh_asset_state(coin, stage=str(snap.get("decision_stage") or "analysis"))

    def _apply_analog_context(
        self,
        coin: str,
        signal,
        current_position: str | None,
        *,
        orderbook_signal=None,
        market_map_signal=None,
    ):
        snap = dict(self._last_signals.get(coin, {}) or {})
        candidate_action = signal.action if signal.action in {"LONG", "SHORT"} else str(
            (getattr(signal, "thesis", {}) or {}).get("candidate_action", snap.get("thesis_candidate_action", "FLAT"))
        ).upper()
        analog = self._analog_engine.evaluate(coin, candidate_action, snap)

        if coin in self._last_signals:
            self._last_signals[coin].update({
                "analog_candidate_action": candidate_action,
                "analog_verdict": analog.get("verdict", "INSUFFICIENT"),
                "analog_sample_size": analog.get("sample_size", 0),
                "analog_avg_similarity": analog.get("avg_similarity", 0.0),
                "analog_reliability": analog.get("reliability", 0.0),
                "analog_win_rate": analog.get("win_rate", 0.0),
                "analog_avg_pnl_pct": analog.get("avg_pnl_pct", 0.0),
                "analog_avg_captured_r": analog.get("avg_captured_r", 0.0),
                "analog_supportive": analog.get("supportive", False),
                "analog_adverse": analog.get("adverse", False),
                "analog_hard_block": analog.get("hard_block", False),
                "analog_score_adjustment": analog.get("score_adjustment", 0.0),
                "analog_probability_adjustment": analog.get("probability_adjustment", 0.0),
                "analog_expected_r_adjustment": analog.get("expected_r_adjustment", 0.0),
                "analog_uncertainty_adjustment": analog.get("uncertainty_adjustment", 0.0),
                "analog_summary": analog.get("summary", ""),
                "analog_top_matches": analog.get("top_matches", []),
            })

        if signal.action in {"LONG", "SHORT"}:
            blended_expectancy = self._analog_engine.blend_expectancy(
                getattr(signal, "expectancy", {}) or {},
                analog,
                same_direction_position=bool(current_position and current_position == signal.action),
            )
            signal.expectancy = blended_expectancy
            score_delta = float(analog.get("score_adjustment", 0.0) or 0.0)
            if score_delta:
                signal.score = max(0.0, min(100.0, float(signal.score) + score_delta))
            if isinstance(getattr(signal, "thesis", None), dict):
                signal.thesis["analog_summary"] = analog.get("summary", "")
                signal.thesis["analog_verdict"] = analog.get("verdict", "INSUFFICIENT")
            if not blended_expectancy.get("permitted", True):
                flat_reason = str(blended_expectancy.get("summary", "") or analog.get("summary", "") or "historical analogs do not support the setup")
                log.info(f"[{coin}] 🧠 Analog gate keeps {signal.action} flat: {flat_reason}")
                signal.action = "FLAT"
                signal.flat_reason = flat_reason
                signal.reason = flat_reason
            else:
                self._apply_precision_analog_guard(
                    coin,
                    signal,
                    orderbook_signal=orderbook_signal,
                    market_map_signal=market_map_signal,
                )

        self._sync_signal_snapshot(coin, signal)
        return analog

    def _apply_precision_analog_guard(self, coin: str, signal, *, orderbook_signal=None, market_map_signal=None) -> None:
        if not getattr(self.cfg.trading, "precision_mode_enabled", False):
            return
        if signal.action not in {"LONG", "SHORT"}:
            return

        snap = dict(self._last_signals.get(coin, {}) or {})
        sample_size = int(snap.get("analog_sample_size", 0) or 0)
        reliability = float(snap.get("analog_reliability", 0.0) or 0.0)
        win_rate = float(snap.get("analog_win_rate", 0.0) or 0.0)
        hard_block = bool(snap.get("analog_hard_block", False))
        adverse = bool(snap.get("analog_adverse", False))
        summary = str(snap.get("analog_summary", "") or "").strip()

        min_samples = int(getattr(self.cfg.trading, "precision_min_analog_samples", 3) or 3)
        min_reliability = float(getattr(self.cfg.trading, "precision_min_analog_reliability", 0.50) or 0.50)
        min_win_rate = float(getattr(self.cfg.trading, "precision_min_analog_win_rate", 0.60) or 0.60)

        if hard_block:
            reason = summary or "historical analogs hard-block the setup"
        elif sample_size >= min_samples and reliability >= min_reliability and (adverse or win_rate < min_win_rate):
            reason = summary or (
                f"analog history only wins {win_rate * 100:.0f}% with "
                f"{sample_size} close matches"
            )
        else:
            allowed, precision_reason = self.strategy._passes_precision_mode(
                coin=coin,
                action=signal.action,
                confidence=getattr(signal, "confidence", "LOW"),
                thesis=getattr(signal, "thesis", {}) or {},
                expectancy=getattr(signal, "expectancy", {}) or {},
                trade_plan=getattr(signal, "trade_plan", {}) or {},
                orderbook_signal=orderbook_signal,
                market_map_signal=market_map_signal,
            )
            if allowed:
                return
            reason = precision_reason or "setup did not clear precision mode"

        log.info(f"[{coin}] 🎯 Precision gate keeps {signal.action} flat: {reason}")
        signal.action = "FLAT"
        signal.flat_reason = reason
        signal.reason = reason

    def _record_decision_snapshot(
        self,
        coin: str,
        *,
        portfolio_usd: float,
        stage: str,
        signal=None,
        current_position: str | None = None,
        executed: bool = False,
        blocked: bool = False,
        pending_limit: bool = False,
    ) -> None:
        if not getattr(self.cfg.trading, "decision_dataset_enabled", True):
            return

        snap = dict(self._last_signals.get(coin, {}) or {})
        if signal is not None:
            snap.setdefault("action", getattr(signal, "action", "FLAT"))
            snap.setdefault("decision", getattr(signal, "action", "FLAT"))
            snap.setdefault("decision_reason", getattr(signal, "reason", "") or getattr(signal, "flat_reason", "") or "")
            snap.setdefault("flat_reason", getattr(signal, "flat_reason", ""))
            snap.setdefault("score", getattr(signal, "score", 50.0))
            snap.setdefault("confidence", getattr(signal, "confidence", "LOW"))
        if not snap:
            return
        snap["decision_stage"] = stage
        lifecycle = self._refresh_asset_state(
            coin,
            stage=stage,
            current_position=current_position,
            pending_limit=pending_limit,
        )
        resolved_stage = stage
        if (
            lifecycle.get("state") == "MAJOR_CATALYST_WATCH"
            and stage in {"analysis", "flat_no_trade", "guardrails_flat", "observation_only"}
        ):
            resolved_stage = "major_catalyst_watch"
            snap["decision_stage"] = resolved_stage
            lifecycle = self._refresh_asset_state(
                coin,
                stage=resolved_stage,
                current_position=current_position,
                pending_limit=pending_limit,
            )
            snap = dict(self._last_signals.get(coin, {}) or {})

        record = {
            "cycle_number": self._cycle,
            "coin": coin,
            "stage": resolved_stage,
            "candidate_action": snap.get("thesis_candidate_action", snap.get("action", "FLAT")),
            "final_action": snap.get("action", "FLAT"),
            "decision_reason": snap.get("decision_reason", ""),
            "has_position": bool(current_position),
            "current_position": current_position or "",
            "tradable": coin in self._tradable_coin_set,
            "execution_mode": snap.get("execution_mode", "observation_only"),
            "executed": bool(executed),
            "blocked": bool(blocked),
            "pending_limit": bool(pending_limit),
            "asset_state": lifecycle.get("state", snap.get("asset_state", "OBSERVING")),
            "next_unblock_reason": snap.get("next_unblock_reason", ""),
            "portfolio_usd": round(float(portfolio_usd or 0.0), 2),
            "available_usd": round(float(self.risk.available_capital(float(portfolio_usd or 0.0))), 2),
            "signal_snapshot": snap,
        }
        try:
            decision_dataset.append_decision(record)
            if getattr(self.cfg.trading, "feature_store_enabled", True):
                feature_store.append_decision_feature_row(record)
        except Exception as exc:
            log.debug(f"[{coin}] Decision dataset append skipped: {exc}")

    def _assess_execution_quality(self, coin: str, direction: str, order, orderbook_signal) -> dict:
        direction = direction.upper()
        quality = {
            "permitted": True,
            "score": 70.0,
            "summary": "Execution conditions are acceptable",
            "blockers": [],
            "warnings": [],
            "spread_bps": 0.0,
            "side_notional_usd": 0.0,
            "estimated_slippage_bps": 0.0,
            "depth_multiple": 0.0,
            "persistence_cycles": 0,
            "prefer_passive_entry": False,
            "passive_limit_price": 0.0,
            "passive_summary": "",
        }

        if not getattr(self.cfg.trading, "require_execution_quality", True):
            quality["summary"] = "Execution-quality gate disabled"
            return quality

        if not orderbook_signal or not getattr(orderbook_signal, "valid", False):
            quality["permitted"] = False
            quality["score"] = 0.0
            quality["summary"] = "Orderbook execution quality is unavailable"
            quality["blockers"].append("no valid orderbook snapshot")
            return quality

        spread_bps = float(getattr(orderbook_signal, "spread_bps", 0.0) or 0.0)
        side_notional = float(
            getattr(orderbook_signal, "bid_notional" if direction == "LONG" else "ask_notional", 0.0) or 0.0
        )
        size_usd = float(getattr(order, "size_usd", 0.0) or 0.0)
        depth_multiple = side_notional / max(size_usd, 1e-9) if size_usd > 0 else 0.0
        estimated_slippage_bps = (spread_bps * 0.5) + ((size_usd / max(side_notional, 1e-9)) * 10_000.0)
        persistence_cycles = self._orderbook_persistence_cycles(coin, direction, orderbook_signal)

        quality.update({
            "spread_bps": round(spread_bps, 3),
            "side_notional_usd": round(side_notional, 2),
            "estimated_slippage_bps": round(estimated_slippage_bps, 3),
            "depth_multiple": round(depth_multiple, 3),
            "persistence_cycles": persistence_cycles,
        })

        max_spread = float(getattr(self.cfg.trading, "max_execution_spread_bps", 12.0) or 12.0)
        min_depth_multiple = float(getattr(self.cfg.trading, "min_execution_depth_multiple", 10.0) or 10.0)
        max_slippage = float(getattr(self.cfg.trading, "max_execution_slippage_bps", 18.0) or 18.0)
        min_persistence = int(getattr(self.cfg.trading, "min_orderbook_persistence_cycles", 2) or 2)
        breakout_state = str(getattr(orderbook_signal, "breakout_state", "NONE") or "NONE")
        level_interaction = str(getattr(orderbook_signal, "level_interaction", "BETWEEN_LEVELS") or "BETWEEN_LEVELS")
        confirmed_break = breakout_state in {
            "CONFIRMED_BULLISH_BREAKOUT",
            "CONFIRMED_BEARISH_BREAKDOWN",
            "PERSISTENT_BULLISH_BREAKOUT",
            "PERSISTENT_BEARISH_BREAKDOWN",
        }
        direction_blocked = (
            direction == "LONG" and getattr(orderbook_signal, "block_longs", False)
        ) or (
            direction == "SHORT" and getattr(orderbook_signal, "block_shorts", False)
        )

        if direction_blocked:
            quality["blockers"].append("orderbook/key levels are blocking this direction")
        if spread_bps > max_spread:
            quality["blockers"].append(f"spread too wide ({spread_bps:.1f}bps > {max_spread:.1f}bps)")
        if depth_multiple < min_depth_multiple:
            quality["blockers"].append(
                f"book depth too thin ({depth_multiple:.1f}x size < {min_depth_multiple:.1f}x)"
            )
        if estimated_slippage_bps > max_slippage:
            quality["blockers"].append(
                f"estimated slippage too high ({estimated_slippage_bps:.1f}bps > {max_slippage:.1f}bps)"
            )
        if level_interaction == "RANGE_COMPRESSION" and not confirmed_break:
            quality["blockers"].append("orderbook still shows range compression")
        if persistence_cycles < min_persistence and not confirmed_break:
            quality["blockers"].append(
                f"key level has only persisted {persistence_cycles}/{min_persistence} cycles"
            )

        if quality["blockers"]:
            quality["permitted"] = False
            quality["score"] = max(0.0, 70.0 - 14.0 * len(quality["blockers"]))
            quality["summary"] = "; ".join(quality["blockers"][:3])
            passive_rescue_enabled = bool(getattr(self.cfg.trading, "execution_passive_rescue_enabled", True))
            rescue_only_execution_friction = (
                passive_rescue_enabled
                and not direction_blocked
                and level_interaction != "RANGE_COMPRESSION"
                and persistence_cycles >= max(1, min_persistence - 1)
            )
            rescue_spread = float(
                getattr(self.cfg.trading, "execution_passive_rescue_max_spread_bps", 28.0) or 28.0
            )
            rescue_depth = float(
                getattr(self.cfg.trading, "execution_passive_rescue_min_depth_multiple", 2.5) or 2.5
            )
            rescue_slippage = float(
                getattr(self.cfg.trading, "execution_passive_rescue_max_slippage_bps", 85.0) or 85.0
            )
            execution_friction_only = all(
                "spread too wide" in blocker
                or "book depth too thin" in blocker
                or "estimated slippage too high" in blocker
                for blocker in quality["blockers"]
            )
            best_bid = float(getattr(orderbook_signal, "best_bid", 0.0) or 0.0)
            best_ask = float(getattr(orderbook_signal, "best_ask", 0.0) or 0.0)
            if (
                rescue_only_execution_friction
                and execution_friction_only
                and spread_bps <= rescue_spread
                and depth_multiple >= rescue_depth
                and estimated_slippage_bps <= rescue_slippage
                and ((direction == "LONG" and best_bid > 0) or (direction == "SHORT" and best_ask > 0))
            ):
                passive_limit_price = best_bid if direction == "LONG" else best_ask
                quality["prefer_passive_entry"] = True
                quality["passive_limit_price"] = round(passive_limit_price, 6)
                quality["passive_summary"] = (
                    f"market sweep is too loose, but a maker {direction.lower()} can rest near "
                    f"${passive_limit_price:,.2f} while waiting for cleaner fills"
                )
                quality["summary"] = quality["passive_summary"]
        else:
            depth_bonus = min(15.0, max(0.0, (depth_multiple - min_depth_multiple) * 1.5))
            persistence_bonus = min(10.0, max(0, persistence_cycles - min_persistence) * 3.0)
            spread_penalty = min(10.0, spread_bps / max(max_spread, 1e-9) * 6.0)
            slippage_penalty = min(10.0, estimated_slippage_bps / max(max_slippage, 1e-9) * 6.0)
            quality["score"] = round(max(0.0, min(100.0, 74.0 + depth_bonus + persistence_bonus - spread_penalty - slippage_penalty)), 2)
            quality["summary"] = (
                f"spread {spread_bps:.1f}bps, est slippage {estimated_slippage_bps:.1f}bps, "
                f"depth {depth_multiple:.1f}x, persistence {persistence_cycles}c"
            )
        return quality

    def _supports_orderbook_context(self, coin: str) -> bool:
        if not is_hyperliquid_supported(coin):
            return False
        if not getattr(self.cfg.trading, "enforce_active_venue_markets", True):
            return True
        return hyperliquid_market_is_active(coin)

    def _check_narrative_gate(self, coin: str, signal, narrative_signal) -> bool:
        if not getattr(self.cfg.trading, "use_narrative_gate", True):
            return True
        if not narrative_signal or not getattr(narrative_signal, "valid", False):
            return True

        expectancy = dict(getattr(signal, "expectancy", {}) or {})
        expectancy_score = float(expectancy.get("score", signal.score) or signal.score)
        probability = float(expectancy.get("probability", 0.50) or 0.50)

        if signal.action == "LONG" and getattr(narrative_signal, "block_longs", False):
            log.info(f"[{coin}] 🧭 Narrative gate blocks LONG: {getattr(narrative_signal, 'summary', '')}")
            return False
        if signal.action == "SHORT" and getattr(narrative_signal, "block_shorts", False):
            log.info(f"[{coin}] 🧭 Narrative gate blocks SHORT: {getattr(narrative_signal, 'summary', '')}")
            return False

        if getattr(narrative_signal, "event_risk_active", False):
            min_score = float(getattr(self.cfg.trading, "narrative_event_block_min_expectancy_score", 72.0) or 72.0)
            min_probability = float(getattr(self.cfg.trading, "narrative_event_block_min_probability", 0.60) or 0.60)
            if expectancy_score < min_score or probability < min_probability:
                log.info(
                    f"[{coin}] 🗓️ Narrative event risk active — "
                    f"need expectancy ≥{min_score:.0f} and p ≥{min_probability*100:.0f}% "
                    f"(have {expectancy_score:.0f}, {probability*100:.0f}%)"
                )
                return False
        return True

    def _assess_conviction_decay(self, coin: str, current_pos: str, signal) -> dict:
        pos = self.risk.positions.get(coin)
        if not pos:
            return {"should_exit": False, "score": 0.0, "summary": "no open position"}

        sig = dict(self._last_signals.get(coin, {}) or {})
        entry_ctx = dict((pos.metadata or {}).get("entry_context", {}) or {})
        hold_minutes = (time.time() - pos.opened_at) / 60.0 if pos.opened_at else 0.0
        planned_tp = float(entry_ctx.get("planned_take_profit") or pos.take_profit or 0.0)
        reward_distance = abs(planned_tp - pos.entry_price) if planned_tp > 0 else 0.0
        live_price = float(sig.get("live_price", signal.price) or signal.price or pos.entry_price)
        favorable_move = self._signed_move(pos.direction, pos.entry_price, live_price)
        tp_progress = favorable_move / max(reward_distance, 1e-9) if reward_distance > 0 else 0.0
        flat_cycles = self._flat_streak.get(coin, 0)

        expectancy = dict(sig.get("expectancy", {}) or getattr(signal, "expectancy", {}) or {})
        thesis = dict(sig.get("thesis", {}) or getattr(signal, "thesis", {}) or {})
        expectancy_score = float(expectancy.get("score", 50.0) or 50.0)
        uncertainty = float(expectancy.get("uncertainty", 0.50) or 0.50)
        thesis_permitted = bool(thesis.get("permitted", signal.action in {"LONG", "SHORT"}))
        structure_trend = str(sig.get("structure_trend", "RANGING") or "RANGING").upper()
        breakout_state = str(sig.get("orderbook_breakout_state", "NONE") or "NONE").upper()

        score = 0.0
        reasons: list[str] = []
        flat_weight = float(getattr(self.cfg.trading, "conviction_decay_flat_cycle_weight", 7.0) or 7.0)
        micro_weight = float(getattr(self.cfg.trading, "conviction_decay_microstructure_weight", 14.0) or 14.0)
        structure_weight = float(getattr(self.cfg.trading, "conviction_decay_structure_weight", 12.0) or 12.0)
        expectancy_weight = float(getattr(self.cfg.trading, "conviction_decay_expectancy_weight", 16.0) or 16.0)

        if signal.action == "FLAT":
            score += flat_cycles * flat_weight
            reasons.append(f"signal flat for {flat_cycles} cycles")

        if not thesis_permitted:
            score += expectancy_weight
            reasons.append("thesis no longer qualifies")

        if expectancy_score < float(getattr(self.cfg.trading, "expectancy_min_score", 56.0) or 56.0):
            deficit = max(0.0, float(getattr(self.cfg.trading, "expectancy_min_score", 56.0)) - expectancy_score)
            score += min(22.0, deficit * 0.9)
            reasons.append(f"expectancy slipped to {expectancy_score:.0f}")

        if uncertainty >= float(getattr(self.cfg.trading, "expectancy_max_uncertainty", 0.42) or 0.42):
            score += (uncertainty - float(getattr(self.cfg.trading, "expectancy_max_uncertainty", 0.42))) * 40.0
            reasons.append("uncertainty expanded materially")

        structure_against = (
            (pos.direction == "LONG" and structure_trend == "DOWNTREND")
            or (pos.direction == "SHORT" and structure_trend == "UPTREND")
        )
        breakout_against = (
            (pos.direction == "LONG" and breakout_state in {"CONFIRMED_BEARISH_BREAKDOWN", "PERSISTENT_BEARISH_BREAKDOWN"})
            or (pos.direction == "SHORT" and breakout_state in {"CONFIRMED_BULLISH_BREAKOUT", "PERSISTENT_BULLISH_BREAKOUT"})
        )
        if structure_against:
            score += structure_weight
            reasons.append("higher structure flipped against the trade")
        if breakout_against:
            score += micro_weight
            reasons.append("orderbook breakout now points the other way")

        if hold_minutes >= float(getattr(self.cfg.trading, "time_stop_minutes", 360.0) or 360.0) and tp_progress < float(getattr(self.cfg.trading, "time_stop_min_tp_progress", 0.25) or 0.25):
            score += 12.0
            reasons.append("time stop is approaching with poor progress")

        if tp_progress >= 0.60:
            score -= 12.0
            reasons.append("trade has already travelled far enough to avoid panic-exit")
        elif favorable_move > 0:
            score -= min(8.0, tp_progress * 8.0)

        score = max(0.0, min(100.0, score))
        exit_threshold = float(getattr(self.cfg.trading, "conviction_decay_exit_threshold", 58.0) or 58.0)
        hold_threshold = float(getattr(self.cfg.trading, "conviction_decay_hold_threshold", 36.0) or 36.0)
        summary = "; ".join(reasons[:3]) if reasons else "conviction is stable"
        if score >= exit_threshold:
            return {"should_exit": True, "score": round(score, 2), "summary": summary}
        if score >= hold_threshold:
            return {"should_exit": False, "score": round(score, 2), "summary": f"watch closely: {summary}"}
        return {"should_exit": False, "score": round(score, 2), "summary": summary}

    def _detect_position_invalidation(self, coin: str, current_pos: str, signal) -> str:
        pos = self.risk.positions.get(coin)
        if not pos:
            return ""

        sig = dict(self._last_signals.get(coin, {}) or {})
        thesis = dict(sig.get("thesis", {}) or {})
        hold_minutes = (time.time() - pos.opened_at) / 60.0 if pos.opened_at else 0.0
        entry_ctx = dict((pos.metadata or {}).get("entry_context", {}) or {})
        planned_stop = float(entry_ctx.get("planned_stop_loss") or pos.stop_loss or 0.0)
        planned_tp = float(entry_ctx.get("planned_take_profit") or pos.take_profit or 0.0)
        risk_per_unit = abs(pos.entry_price - planned_stop) if planned_stop > 0 else 0.0
        reward_per_unit = abs(planned_tp - pos.entry_price) if planned_tp > 0 else 0.0
        live_price = float(sig.get("live_price", signal.price) or signal.price or pos.entry_price)
        favorable_move = self._signed_move(pos.direction, pos.entry_price, live_price)
        adverse_move = max(0.0, -favorable_move)
        adverse_r = adverse_move / max(risk_per_unit, 1e-9) if risk_per_unit > 0 else 0.0
        tp_progress = favorable_move / max(reward_per_unit, 1e-9) if reward_per_unit > 0 else 0.0

        structure_trend = str(sig.get("structure_trend", "") or "").upper()
        mtf_bias = str(sig.get("mtf_bias", "FLAT") or "FLAT").upper()
        breakout_state = str(sig.get("orderbook_breakout_state", "NONE") or "NONE").upper()
        thesis_permitted = bool(thesis.get("permitted", signal.action in {"LONG", "SHORT"}))
        thesis_state = str(thesis.get("state", "") or "").upper()
        thesis_conflicts = float(thesis.get("conflict_points", 0.0) or 0.0)

        early_minutes = float(getattr(self.cfg.trading, "early_invalidation_minutes", 90.0) or 90.0)
        early_adverse_r = float(getattr(self.cfg.trading, "early_invalidation_adverse_r", 0.55) or 0.55)
        htf_min_minutes = float(getattr(self.cfg.trading, "htf_invalidation_min_minutes", 60.0) or 60.0)
        time_stop_minutes = float(getattr(self.cfg.trading, "time_stop_minutes", 360.0) or 360.0)
        time_stop_min_progress = float(getattr(self.cfg.trading, "time_stop_min_tp_progress", 0.25) or 0.25)

        structure_against = (
            (pos.direction == "LONG" and structure_trend == "DOWNTREND") or
            (pos.direction == "SHORT" and structure_trend == "UPTREND")
        )
        breakout_against = (
            (pos.direction == "LONG" and breakout_state in {"CONFIRMED_BEARISH_BREAKDOWN", "PERSISTENT_BEARISH_BREAKDOWN"}) or
            (pos.direction == "SHORT" and breakout_state in {"CONFIRMED_BULLISH_BREAKOUT", "PERSISTENT_BULLISH_BREAKOUT"})
        )
        mtf_against = (
            (pos.direction == "LONG" and mtf_bias == "BEARISH") or
            (pos.direction == "SHORT" and mtf_bias == "BULLISH")
        )

        if hold_minutes <= early_minutes and adverse_r >= early_adverse_r and not thesis_permitted:
            return "micro_invalidation"
        if (structure_against or breakout_against) and (not thesis_permitted or thesis_state == "NO_TRADE" or thesis_conflicts >= 2):
            return "structure_invalidation"
        if hold_minutes >= htf_min_minutes and mtf_against and (not thesis_permitted or thesis_conflicts >= 1):
            return "htf_invalidation"
        if hold_minutes >= time_stop_minutes and tp_progress < time_stop_min_progress and (not thesis_permitted or thesis_state == "NO_TRADE"):
            return "time_stop"
        return ""

    def _build_closed_trade_dataset_record(
        self,
        coin: str,
        pos,
        exit_price: float,
        exit_reason: str,
        hold_minutes: float,
        entry_ctx: dict,
        last_sig: dict,
        csv_trade: dict | None,
    ) -> dict:
        entry_price = float(pos.entry_price or 0.0)
        if entry_price <= 0:
            return {}
        if pos.direction == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price * 100.0
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100.0
        pnl_usd = pnl_pct / 100.0 * float(pos.size_usd or 0.0)

        return {
            "trade_id": (csv_trade or {}).get("trade_id"),
            "coin": coin,
            "direction": pos.direction,
            "exchange": pos.exchange,
            "opened_at_ts": pos.opened_at,
            "closed_at_ts": time.time(),
            "hold_minutes": round(hold_minutes, 2),
            "entry_price": round(entry_price, 6),
            "exit_price": round(float(exit_price or 0.0), 6),
            "size_usd": round(float(pos.size_usd or 0.0), 2),
            "size_coin": round(float(pos.size_coin or 0.0), 8),
            "pnl_usd": round(pnl_usd, 4),
            "pnl_pct": round(pnl_pct, 4),
            "exit_reason": exit_reason,
            "outcome": "WIN" if pnl_usd > 0 else "LOSS" if pnl_usd < 0 else "BREAKEVEN",
            "signal_score": entry_ctx.get("score", last_sig.get("score", 50.0)),
            "thesis": dict(entry_ctx.get("thesis", {}) or last_sig.get("thesis", {}) or {}),
            "trade_plan": dict(entry_ctx.get("trade_plan", {}) or {}),
            "execution_quality": dict(entry_ctx.get("execution_quality", {}) or {}),
            "entry_context": dict(entry_ctx or {}),
            "exit_context": {
                "signal_action": last_sig.get("action", "FLAT"),
                "signal_score": last_sig.get("score", 50.0),
                "thesis_state": last_sig.get("thesis_state", ""),
                "thesis_summary": last_sig.get("thesis_summary", ""),
                "expectancy_score": last_sig.get("expectancy_score", last_sig.get("score", 50.0)),
                "expectancy_probability": last_sig.get("expectancy_probability", 0.50),
                "expectancy_expected_r": last_sig.get("expectancy_expected_r", 0.0),
                "expectancy_uncertainty": last_sig.get("expectancy_uncertainty", 0.50),
                "mtf_bias": last_sig.get("mtf_bias", "FLAT"),
                "market_regime": last_sig.get("market_regime", "RANGING"),
                "dominant_regime": last_sig.get("dominant_regime", "MIXED"),
                "orderbook_breakout_state": last_sig.get("orderbook_breakout_state", "NONE"),
                "orderbook_interaction": last_sig.get("orderbook_interaction", "BETWEEN_LEVELS"),
            },
            "plan_outcome": {
                "captured_r_multiple": entry_ctx.get("captured_r_multiple"),
                "tp_progress_ratio": entry_ctx.get("tp_progress_ratio"),
                "remaining_to_tp_pct": entry_ctx.get("remaining_to_tp_pct"),
                "remaining_to_sl_pct": entry_ctx.get("remaining_to_sl_pct"),
                "stop_pressure_ratio": entry_ctx.get("stop_pressure_ratio"),
            },
        }

    def _build_entry_context(self, coin: str, signal, order=None, entry_type: str = "signal_entry") -> dict:
        sig = dict(getattr(self, "_last_signals", {}).get(coin, {}) or {})
        trade_plan = dict(sig.get("trade_plan", {}) or getattr(signal, "trade_plan", {}) or {})
        planned_stop = float(
            getattr(order, "stop_loss", trade_plan.get("stop_loss", sig.get("planned_stop_loss", 0.0))) or 0.0
        )
        planned_take_profit = float(
            getattr(order, "take_profit", trade_plan.get("take_profit", sig.get("planned_take_profit", 0.0))) or 0.0
        )
        trade_plan.update({
            "entry_price": round(float(getattr(order, "price", trade_plan.get("entry_price", sig.get("price", 0.0))) or 0.0), 6),
            "stop_loss": round(planned_stop, 6) if planned_stop else 0.0,
            "take_profit": round(planned_take_profit, 6) if planned_take_profit else 0.0,
            "risk_pct": sig.get("planned_risk_pct", trade_plan.get("risk_pct", 0.0)),
            "reward_pct": sig.get("planned_reward_pct", trade_plan.get("reward_pct", 0.0)),
            "risk_reward_ratio": sig.get("planned_risk_reward_ratio", trade_plan.get("risk_reward_ratio", 0.0)),
            "stop_atr_multiple": sig.get("planned_stop_atr_multiple", trade_plan.get("stop_atr_multiple", 0.0)),
            "target_atr_multiple": sig.get("planned_target_atr_multiple", trade_plan.get("target_atr_multiple", 0.0)),
            "target_r_multiple": sig.get("planned_target_r_multiple", trade_plan.get("target_r_multiple", 0.0)),
            "stop_basis": sig.get("stop_basis", trade_plan.get("stop_basis", "")),
            "target_basis": sig.get("target_basis", trade_plan.get("target_basis", "")),
            "price_action_summary": sig.get("price_action_summary", trade_plan.get("price_action_summary", "")),
        })
        return {
            "entry_type": entry_type,
            "action": getattr(signal, "action", sig.get("action", "")),
            "score": getattr(signal, "score", sig.get("score", 50.0)),
            "confidence": getattr(signal, "confidence", sig.get("confidence", "LOW")),
            "reason": getattr(signal, "reason", sig.get("reason", "")),
            "flat_reason": getattr(signal, "flat_reason", sig.get("flat_reason", "")),
            "mtf_bias": sig.get("mtf_bias", "FLAT"),
            "market_regime": sig.get("market_regime", "RANGING"),
            "dominant_regime": sig.get("dominant_regime", "MIXED"),
            "instrument_type": sig.get("instrument_type", "crypto"),
            "news_score": sig.get("news_score", 50.0),
            "news_velocity": sig.get("news_velocity", "LOW"),
            "candle_score": sig.get("candle_score", 50.0),
            "candle_trend": sig.get("candle_trend", "FLAT"),
            "foc_score": sig.get("foc_score", 50.0),
            "funding_label": sig.get("funding_label", "N/A"),
            "memory_adj": sig.get("memory_adj", 0.0),
            "rl_total_trades": sig.get("rl_total_trades", 0),
            "rl_win_rate": sig.get("rl_win_rate"),
            "rl_pattern_boost": sig.get("rl_pattern_boost", 0.0),
            "execution_mode": sig.get("execution_mode", "observation_only"),
            "decision_stage": sig.get("decision_stage", "analysis"),
            "asset_state": sig.get("asset_state", "OBSERVING"),
            "asset_state_label": sig.get("asset_state_label", "Observing"),
            "next_unblock_reason": sig.get("next_unblock_reason", ""),
            "planned_stop_loss": planned_stop,
            "planned_take_profit": planned_take_profit,
            "planned_risk_pct": sig.get("planned_risk_pct", trade_plan.get("risk_pct", 0.0)),
            "planned_reward_pct": sig.get("planned_reward_pct", trade_plan.get("reward_pct", 0.0)),
            "planned_risk_reward_ratio": sig.get("planned_risk_reward_ratio", trade_plan.get("risk_reward_ratio", 0.0)),
            "planned_stop_atr_multiple": sig.get("planned_stop_atr_multiple", trade_plan.get("stop_atr_multiple", 0.0)),
            "planned_target_atr_multiple": sig.get("planned_target_atr_multiple", trade_plan.get("target_atr_multiple", 0.0)),
            "planned_target_r_multiple": sig.get("planned_target_r_multiple", trade_plan.get("target_r_multiple", 0.0)),
            "stop_basis": sig.get("stop_basis", trade_plan.get("stop_basis", "")),
            "target_basis": sig.get("target_basis", trade_plan.get("target_basis", "")),
            "price_action_summary": sig.get("price_action_summary", trade_plan.get("price_action_summary", "")),
            "orderbook_score": sig.get("orderbook_score", 50.0),
            "orderbook_imbalance": sig.get("orderbook_imbalance", 0.0),
            "orderbook_imbalance_mean": sig.get("orderbook_imbalance_mean", 0.0),
            "orderbook_imbalance_trend": sig.get("orderbook_imbalance_trend", 0.0),
            "orderbook_imbalance_volatility": sig.get("orderbook_imbalance_volatility", 0.0),
            "orderbook_interaction": sig.get("orderbook_interaction", "BETWEEN_LEVELS"),
            "orderbook_breakout_state": sig.get("orderbook_breakout_state", "NONE"),
            "orderbook_intracycle_breakout_state": sig.get("orderbook_intracycle_breakout_state", "NONE"),
            "orderbook_support": sig.get("orderbook_support", 0.0),
            "orderbook_support_distance_pct": sig.get("orderbook_support_distance_pct", 0.0),
            "orderbook_resistance": sig.get("orderbook_resistance", 0.0),
            "orderbook_resistance_distance_pct": sig.get("orderbook_resistance_distance_pct", 0.0),
            "orderbook_support_wall_persistence": sig.get("orderbook_support_wall_persistence", 0),
            "orderbook_resistance_wall_persistence": sig.get("orderbook_resistance_wall_persistence", 0),
            "orderbook_feed_age_seconds": sig.get("orderbook_feed_age_seconds", 0.0),
            "orderbook_feed_snapshot_count": sig.get("orderbook_feed_snapshot_count", 0),
            "daily_breakout_level": sig.get("daily_breakout_level", 0.0),
            "daily_breakdown_level": sig.get("daily_breakdown_level", 0.0),
            "market_map_bias": sig.get("market_map_bias", "NEUTRAL"),
            "market_map_summary": sig.get("market_map_summary", ""),
            "market_map_score_adjustment": sig.get("market_map_score_adjustment", 0.0),
            "market_map_notes": sig.get("market_map_notes", ""),
            "narrative_summary": sig.get("narrative_summary", ""),
            "narrative_event_risk_active": sig.get("narrative_event_risk_active", False),
            "narrative_event_name": sig.get("narrative_event_name", ""),
            "narrative_event_importance": sig.get("narrative_event_importance", "NONE"),
            "narrative_minutes_to_event": sig.get("narrative_minutes_to_event"),
            "narrative_headline_bias": sig.get("narrative_headline_bias", "NEUTRAL"),
            "data_reliability": sig.get("data_reliability", {}),
            "data_reliability_score": sig.get("data_reliability_score", 0.0),
            "data_reliability_quality": sig.get("data_reliability_quality", "UNKNOWN"),
            "data_reliability_summary": sig.get("data_reliability_summary", ""),
            "portfolio_guard": sig.get("portfolio_guard", {}),
            "portfolio_theme": sig.get("portfolio_theme", ""),
            "portfolio_guard_summary": sig.get("portfolio_guard_summary", ""),
            "portfolio_guard_size_multiplier": sig.get("portfolio_guard_size_multiplier", 1.0),
            "operator_review_guard": sig.get("operator_review_guard", {}),
            "trade_plan": trade_plan,
            "thesis": sig.get("thesis", {}),
            "expectancy": sig.get("expectancy", {}),
            "expectancy_probability": sig.get("expectancy_probability", 0.50),
            "expectancy_expected_r": sig.get("expectancy_expected_r", 0.0),
            "expectancy_uncertainty": sig.get("expectancy_uncertainty", 0.50),
            "expectancy_score": sig.get("expectancy_score", sig.get("score", 50.0)),
            "expectancy_summary": sig.get("expectancy_summary", ""),
            "execution_plan": sig.get("execution_plan", {}),
            "execution_quality": sig.get("execution_quality", {}),
            "execution_quality_score": sig.get("execution_quality_score", 0.0),
            "execution_quality_summary": sig.get("execution_quality_summary", ""),
            "execution_coach": sig.get("execution_coach", {}),
            "execution_coach_used": sig.get("execution_coach_used", False),
            "execution_coach_verdict": sig.get("execution_coach_verdict", ""),
            "execution_coach_summary": sig.get("execution_coach_summary", ""),
            "execution_coach_urgency_score": sig.get("execution_coach_urgency_score", 0.0),
            "execution_coach_stretch_bps": sig.get("execution_coach_stretch_bps", 0.0),
            "estimated_slippage_bps": sig.get("estimated_slippage_bps", 0.0),
            "execution_persistence_cycles": sig.get("execution_persistence_cycles", 0),
            "conviction_tier": getattr(order, "conviction_tier", ""),
            "conviction_pct": getattr(order, "conviction_pct", 0.0),
        }

    @staticmethod
    def _signed_move(direction: str, start: float, end: float) -> float:
        if start <= 0 or end <= 0:
            return 0.0
        return (end - start) if direction == "LONG" else (start - end)

    def _reanchor_trade_plan_to_fill(self, coin: str, signal, order, fill_price: float) -> None:
        trade_plan = dict(getattr(signal, "trade_plan", {}) or {})
        risk_per_unit = float(trade_plan.get("risk_per_unit", 0.0) or 0.0)
        reward_per_unit = float(trade_plan.get("reward_per_unit", 0.0) or 0.0)
        if fill_price <= 0 or risk_per_unit <= 0 or reward_per_unit <= 0:
            return

        if signal.action == "LONG":
            order.stop_loss = max(0.0, fill_price - risk_per_unit)
            order.take_profit = max(0.0, fill_price + reward_per_unit)
        else:
            order.stop_loss = fill_price + risk_per_unit
            order.take_profit = max(0.0, fill_price - reward_per_unit)

        trade_plan.update({
            "entry_price": round(fill_price, 6),
            "stop_loss": round(order.stop_loss, 6),
            "take_profit": round(order.take_profit, 6),
            "risk_pct": round(abs(fill_price - order.stop_loss) / max(fill_price, 1e-9) * 100, 3),
            "reward_pct": round(abs(order.take_profit - fill_price) / max(fill_price, 1e-9) * 100, 3),
            "risk_reward_ratio": round(
                abs(order.take_profit - fill_price) / max(abs(fill_price - order.stop_loss), 1e-9), 3
            ),
        })
        signal.trade_plan = trade_plan
        if coin in self._last_signals:
            self._last_signals[coin]["planned_stop_loss"] = order.stop_loss
            self._last_signals[coin]["planned_take_profit"] = order.take_profit
            self._last_signals[coin]["planned_risk_pct"] = trade_plan.get("risk_pct", 0.0)
            self._last_signals[coin]["planned_reward_pct"] = trade_plan.get("reward_pct", 0.0)
            self._last_signals[coin]["planned_risk_reward_ratio"] = trade_plan.get("risk_reward_ratio", 0.0)
            self._last_signals[coin]["trade_plan"] = dict(trade_plan)

    def _annotate_exit_against_plan(self, direction: str, entry_ctx: dict, entry: float, exit_price: float) -> dict:
        ctx = dict(entry_ctx or {})
        if entry <= 0 or exit_price <= 0:
            return ctx

        try:
            stop_price = float(ctx.get("planned_stop_loss") or 0.0)
        except Exception:
            stop_price = 0.0
        try:
            take_profit = float(ctx.get("planned_take_profit") or 0.0)
        except Exception:
            take_profit = 0.0

        favorable_move = self._signed_move(direction, entry, exit_price)
        tp_distance = self._signed_move(direction, entry, take_profit) if take_profit > 0 else 0.0
        if direction == "LONG":
            sl_distance = max(0.0, entry - stop_price) if stop_price > 0 else 0.0
            remaining_to_sl = max(0.0, exit_price - stop_price) if stop_price > 0 else 0.0
            remaining_to_tp = max(0.0, take_profit - exit_price) if take_profit > 0 else 0.0
        else:
            sl_distance = max(0.0, stop_price - entry) if stop_price > 0 else 0.0
            remaining_to_sl = max(0.0, stop_price - exit_price) if stop_price > 0 else 0.0
            remaining_to_tp = max(0.0, exit_price - take_profit) if take_profit > 0 else 0.0

        ctx["realized_move_pct"] = round(favorable_move / entry * 100, 3)
        if sl_distance > 0:
            ctx["captured_r_multiple"] = round(favorable_move / sl_distance, 3)
            ctx["stop_pressure_ratio"] = round(max(0.0, -favorable_move) / sl_distance, 3)
        if tp_distance > 0:
            ctx["tp_progress_ratio"] = round(favorable_move / tp_distance, 3)
        if take_profit > 0:
            ctx["remaining_to_tp_pct"] = round(remaining_to_tp / entry * 100, 3)
        if stop_price > 0:
            ctx["remaining_to_sl_pct"] = round(remaining_to_sl / entry * 100, 3)
        return ctx

    def _execute_order(self, coin, signal, order):
        exchanges = self._eligible_exchanges(coin)
        if not exchanges:
            log.error(f"[{coin}] No exchange supports this symbol")
            return False

        for ex in exchanges:
            ex.set_leverage(coin, self.cfg.trading.leverage)

            result = None
            for attempt in range(1, 4):
                if signal.action == "LONG":
                    result = ex.market_buy(coin, order.size_coin)
                else:   # SHORT
                    result = ex.market_sell(coin, order.size_coin)

                if result.success:
                    break
                else:
                    if attempt < 3:
                        log.warning(f"[{coin}] Attempt {attempt}/3 failed: {result.error}. "
                                   f"Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        log.error(f"[{coin}] ❌ {ex.name} order failed after 3 attempts: {result.error}")
                        self.notifier.error_alert(
                            f"{signal.action} failed on {ex.name} for {coin}: {result.error}"
                        )

            if result and result.success:
                fill_price = result.filled_price or signal.price
                order.price = fill_price
                self._reanchor_trade_plan_to_fill(coin, signal, order, fill_price)
                time.sleep(1)
                verified = self._verify_position_on_exchange(ex, coin, should_exist=True)
                if verified is not True:
                    recovered = self._reconcile_and_check_coin(coin, should_exist=True)
                    if not recovered:
                        log.critical(
                            f"[{coin}] CRITICAL: order succeeded on {ex.name} but the position "
                            f"could not be verified or recovered from reconciliation"
                        )
                        self.notifier.error_alert(
                            f"CRITICAL: {coin} {signal.action} order succeeded on {ex.name} but verification failed"
                        )
                        return False
                    pos = self.risk.positions.get(coin)
                    if pos:
                        if order.is_scale_in:
                            trade_logger.update_open(
                                coin=coin,
                                entry_price=pos.entry_price,
                                size_usd=pos.size_usd,
                                stop_loss=pos.stop_loss,
                                take_profit=pos.take_profit,
                            )
                        else:
                            trade_logger.restore_open(
                                coin=coin,
                                direction=signal.action,
                                entry_price=pos.entry_price,
                                size_usd=pos.size_usd,
                                stop_loss=pos.stop_loss,
                                take_profit=pos.take_profit,
                                leverage=self.cfg.trading.leverage,
                                signal_score=signal.score,
                            )
                    log.warning(f"[{coin}] Verification recovered from reconciliation on {ex.name}")
                else:
                    if order.is_scale_in:
                        self.risk.record_scale_in_fill(order, exchange=ex.name)
                        pos = self.risk.positions.get(coin)
                        if pos:
                            trade_logger.update_open(
                                coin=coin,
                                entry_price=pos.entry_price,
                                size_usd=pos.size_usd,
                                stop_loss=pos.stop_loss,
                                take_profit=pos.take_profit,
                            )
                    else:
                        self.risk.record_open(
                            order,
                            exchange=ex.name,
                            metadata={"entry_context": self._build_entry_context(coin, signal, order, entry_type="market_entry")},
                        )
                        trade_logger.log_open(
                            coin         = coin,
                            direction    = signal.action,
                            entry_price  = fill_price,
                            size_usd     = order.size_usd,
                            stop_loss    = order.stop_loss,
                            take_profit  = order.take_profit,
                            signal_score = signal.score,
                            leverage     = self.cfg.trading.leverage,
                        )
                    log.info(f"[{coin}] Position verified on {ex.name}")

                total_size_usd = None
                if order.is_scale_in:
                    pos = self.risk.positions.get(coin)
                    if pos:
                        total_size_usd = pos.size_usd
                self.notifier.trade_opened(
                    coin     = coin,
                    direction= signal.action,
                    price    = fill_price,
                    size_usd = order.size_usd,
                    sl       = order.stop_loss,
                    tp       = order.take_profit,
                    score    = signal.score,
                    exchange = ex.name,
                    is_scale_in = bool(order.is_scale_in),
                    total_size_usd = total_size_usd,
                )
                log.info(
                    f"[{coin}] ✅ {'scale-in added' if order.is_scale_in else signal.action + ' opened'} on {ex.name}: "
                    f"{order.size_coin:.6f} @ ${fill_price:.2f} "
                    f"| SL ${order.stop_loss:.2f} TP ${order.take_profit:.2f}"
                )
                return True
            elif not result:
                log.error(f"[{coin}] ❌ No result returned from {ex.name}")
                self.notifier.error_alert(
                    f"{signal.action} failed on {ex.name} for {coin}: No result returned"
                )
        return False

    def _place_limit_order(self, coin: str, direction: str, limit_price: float,
                           size_usd: float, sl: float, tp: float,
                           score: float, reason: str = "re_entry",
                           entry_context: Optional[dict] = None,
                           trade_plan: Optional[dict] = None,
                           maker_only: bool = False,
                           extra_metadata: Optional[dict] = None) -> dict:
        """Place a limit order on the first eligible exchange and register or book any fill."""
        exchanges = [ex for ex in self._eligible_exchanges(coin) if ex.supports_limit_orders()]
        if not exchanges:
            log.error(f"[{coin}] No exchange supports limit orders for this symbol")
            return {"success": False, "pending": False, "filled": False}
        for ex in exchanges:
            ex.set_leverage(coin, self.cfg.trading.leverage)
            size_coin = size_usd / limit_price if limit_price > 0 else 0

            if direction == "LONG":
                result = ex.limit_buy(coin, size_coin, limit_price, maker_only=maker_only)
            else:
                result = ex.limit_sell(coin, size_coin, limit_price, maker_only=maker_only)

            if result.success:
                fill_price = float(result.filled_price or limit_price or 0.0)
                filled_size_coin = max(0.0, float(result.filled_size or 0.0))
                remaining_size_coin = max(0.0, size_coin - filled_size_coin)
                remaining_size_usd = max(0.0, remaining_size_coin * limit_price)
                same_direction_position = self.risk.has_position(coin) and self.risk.position_direction(coin) == direction

                if filled_size_coin > 0 and fill_price > 0:
                    filled_order = OrderRequest(
                        coin=coin,
                        direction=direction,
                        size_usd=filled_size_coin * fill_price,
                        size_coin=filled_size_coin,
                        price=fill_price,
                        stop_loss=sl,
                        take_profit=tp,
                        leverage=self.cfg.trading.leverage,
                        approved=True,
                    )
                    entry_payload = {
                        "entry_context": (
                            dict(entry_context or {})
                            or self._build_entry_context(
                                coin,
                                type("PendingSignal", (), {
                                    "action": direction,
                                    "score": score,
                                    "confidence": "MEDIUM",
                                    "reason": reason,
                                    "flat_reason": "",
                                    "trade_plan": dict(trade_plan or {}),
                                })(),
                                filled_order,
                                entry_type=reason,
                            )
                        )
                    }
                    if same_direction_position:
                        self.risk.record_scale_in_fill(filled_order, exchange=ex.name)
                        pos = self.risk.positions.get(coin)
                        if pos:
                            trade_logger.update_open(
                                coin=coin,
                                entry_price=pos.entry_price,
                                size_usd=pos.size_usd,
                                stop_loss=pos.stop_loss,
                                take_profit=pos.take_profit,
                            )
                    else:
                        self.risk.record_open(filled_order, exchange=ex.name, metadata=entry_payload)
                        trade_logger.log_open(
                            coin=coin,
                            direction=direction,
                            entry_price=fill_price,
                            size_usd=filled_order.size_usd,
                            stop_loss=sl,
                            take_profit=tp,
                            signal_score=score,
                            leverage=self.cfg.trading.leverage,
                        )
                    self.notifier.trade_opened(
                        coin=coin,
                        direction=direction,
                        price=fill_price,
                        size_usd=filled_order.size_usd,
                        sl=sl,
                        tp=tp,
                        score=score,
                        exchange=ex.name,
                        is_scale_in=bool(same_direction_position),
                        total_size_usd=(self.risk.positions.get(coin).size_usd if same_direction_position and self.risk.positions.get(coin) else None),
                    )
                    log.info(
                        f"[{coin}] ✅ Limit {'scale-in' if same_direction_position else direction} immediately filled on {ex.name}: "
                        f"{filled_size_coin:.6f} @ ${fill_price:.2f}"
                    )

                if remaining_size_coin <= 1e-9 or not result.order_id:
                    return {"success": True, "pending": False, "filled": filled_size_coin > 0}

                pending = PendingOrder(
                    coin              = coin,
                    direction         = direction,
                    limit_price       = limit_price,
                    size_coin         = remaining_size_coin,
                    size_usd          = remaining_size_usd,
                    stop_loss         = sl,
                    take_profit       = tp,
                    signal_score      = score,
                    exchange          = ex.name,
                    exchange_order_id = result.order_id,
                    reprice_count     = int((extra_metadata or {}).get("reprice_count", 0) or 0),
                    reason            = reason,
                    metadata          = {
                        "entry_context": (
                            dict(entry_context or {})
                            or self._build_entry_context(
                                coin,
                                type("PendingSignal", (), {
                                    "action": direction,
                                    "score": score,
                                    "confidence": "MEDIUM",
                                    "reason": reason,
                                    "flat_reason": "",
                                    "trade_plan": (
                                        dict(trade_plan or {})
                                        or {
                                            "entry_price": limit_price,
                                            "stop_loss": sl,
                                            "take_profit": tp,
                                            "risk_pct": round(abs(limit_price - sl) / max(limit_price, 1e-9) * 100, 3),
                                            "reward_pct": round(abs(tp - limit_price) / max(limit_price, 1e-9) * 100, 3),
                                            "risk_reward_ratio": round(
                                                abs(tp - limit_price) / max(abs(limit_price - sl), 1e-9), 3
                                            ),
                                            "stop_basis": "limit_plan",
                                            "target_basis": "limit_plan",
                                        }
                                    ),
                                })(),
                                entry_type=reason,
                            )
                        )
                    },
                )
                if extra_metadata:
                    pending.metadata.update(dict(extra_metadata or {}))
                self.order_mgr.register_limit_order(pending)
                log.info(
                    f"[{coin}] 📋 Limit {direction} placed @ ${limit_price:.2f} "
                    f"SL=${sl:.2f} TP=${tp:.2f} (reason={reason})"
                )
                return {"success": True, "pending": True, "filled": filled_size_coin > 0}
            else:
                log.error(f"[{coin}] Limit order failed on {ex.name}: {result.error}")
        return {"success": False, "pending": False, "filled": False}

    # ── Order manager action handler ──────────────────────────

    def _handle_order_manager_action(self, action: dict, portfolio_usd: float):
        """Handle actions emitted by OrderManager.tick()."""
        if action["type"] == "place_limit":
            coin      = action["coin"]
            direction = action["direction"]
            price     = action["price"]
            size_usd  = min(action.get("size_usd", self.cfg.trading.max_trade_usd),
                            self.cfg.trading.max_trade_usd)
            sl        = action["sl"]
            tp        = action["tp"]
            score     = action.get("score", 60.0)
            reason    = action.get("reason", "re_entry")

            # Final sanity check: don't open if we already have this position
            if self.risk.has_position(coin):
                log.info(f"[{coin}] Skipping limit order — position already open")
                return

            self._place_limit_order(
                coin, direction, price, size_usd, sl, tp, score, reason,
                entry_context=action.get("entry_context"),
                trade_plan=action.get("trade_plan"),
            )

        elif action["type"] == "cancel_limit":
            coin     = action["coin"]
            order_id = action.get("order_id", "")
            target = self._get_exchange_by_name(action.get("exchange", ""))
            exchanges = [target] if target else self.exchanges
            for ex in exchanges:
                if ex and ex.cancel_order(coin, order_id):
                    break
            self.order_mgr.mark_cancelled(coin)

    # ── Poll pending limit orders ─────────────────────────────

    def _poll_pending_limits(self, current_prices: dict):
        """Check if any pending limit orders have been filled."""
        for coin, pending in list(self.order_mgr.pending_orders.items()):
            ex = self._get_exchange_by_name(pending.exchange) or (self.exchanges[0] if self.exchanges else None)
            if not ex:
                continue
            had_same_direction_position = self.risk.has_position(coin) and self.risk.position_direction(coin) == pending.direction
            try:
                circuit = self._exchange_circuits.get(ex.name)
                if circuit:
                    status = circuit.call(ex.get_order_status, coin, pending.exchange_order_id or "")
                else:
                    status = ex.get_order_status(coin, pending.exchange_order_id or "")
            except CircuitBreakerError as e:
                log.warning(f"[{coin}] Cannot poll {ex.name}: {e}")
                continue
            except Exception as e:
                log.error(f"[{coin}] Failed to poll order on {ex.name}: {e}")
                continue
            if status.filled:
                fill_price = status.filled_price or pending.limit_price
                verified = self._verify_position_on_exchange(ex, coin, should_exist=True)
                recovered = False
                if verified is not True:
                    recovered = self._reconcile_and_check_coin(coin, should_exist=True)
                if verified is not True and not recovered:
                    log.critical(
                        f"[{coin}] Limit order filled on {ex.name} but the resulting position "
                        f"could not be verified or reconciled"
                    )
                    self.notifier.error_alert(
                        f"CRITICAL: {coin} limit fill on {ex.name} could not be verified"
                    )
                    self.order_mgr.mark_filled(coin, fill_price)
                    continue

                log.info(f"[{coin}] Limit order FILLED @ ${fill_price:.2f}")
                order = OrderRequest(
                    coin        = coin,
                    direction   = pending.direction,
                    size_usd    = pending.size_usd,
                    size_coin   = pending.size_coin,
                    price       = fill_price,
                    stop_loss   = pending.stop_loss,
                    take_profit = pending.take_profit,
                    leverage    = self.cfg.trading.leverage,
                    approved    = True,
                )
                if recovered:
                    pos = self.risk.positions.get(coin)
                    if pos:
                        if had_same_direction_position:
                            trade_logger.update_open(
                                coin=coin,
                                entry_price=pos.entry_price,
                                size_usd=pos.size_usd,
                                stop_loss=pos.stop_loss,
                                take_profit=pos.take_profit,
                            )
                        else:
                            trade_logger.restore_open(
                                coin=coin,
                                direction=pending.direction,
                                entry_price=pos.entry_price,
                                size_usd=pos.size_usd,
                                stop_loss=pos.stop_loss,
                                take_profit=pos.take_profit,
                                leverage=self.cfg.trading.leverage,
                                signal_score=pending.signal_score,
                            )
                else:
                    pending_signal = type("PendingSignal", (), {
                        "action": pending.direction,
                        "score": pending.signal_score,
                        "confidence": "MEDIUM",
                        "reason": pending.reason or "Limit/re-entry order filled",
                        "flat_reason": "",
                        "trade_plan": (
                            (pending.metadata or {}).get("entry_context", {}) or {}
                        ).get("trade_plan", {}),
                    })()
                    entry_context = (
                        (pending.metadata or {}).get("entry_context")
                        or self._build_entry_context(coin, pending_signal, order, entry_type=pending.reason or "limit_entry")
                    )
                    entry_context = dict(entry_context or {})
                    entry_context["planned_stop_loss"] = pending.stop_loss
                    entry_context["planned_take_profit"] = pending.take_profit
                    trade_plan = dict(entry_context.get("trade_plan", {}) or {})
                    trade_plan.update({
                        "entry_price": round(fill_price, 6),
                        "stop_loss": round(pending.stop_loss, 6),
                        "take_profit": round(pending.take_profit, 6),
                    })
                    entry_context["trade_plan"] = trade_plan
                    if had_same_direction_position:
                        self.risk.record_scale_in_fill(order, exchange=ex.name)
                        pos = self.risk.positions.get(coin)
                        if pos:
                            trade_logger.update_open(
                                coin=coin,
                                entry_price=pos.entry_price,
                                size_usd=pos.size_usd,
                                stop_loss=pos.stop_loss,
                                take_profit=pos.take_profit,
                            )
                    else:
                        self.risk.record_open(
                            order,
                            exchange=ex.name,
                            metadata={"entry_context": entry_context},
                        )
                        trade_logger.log_open(
                            coin         = coin,
                            direction    = pending.direction,
                            entry_price  = fill_price,
                            size_usd     = pending.size_usd,
                            stop_loss    = pending.stop_loss,
                            take_profit  = pending.take_profit,
                            signal_score = pending.signal_score,
                            leverage     = self.cfg.trading.leverage,
                        )
                self.notifier.trade_opened(
                    coin     = coin,
                    direction= pending.direction,
                    price    = fill_price,
                    size_usd = pending.size_usd,
                    sl       = pending.stop_loss,
                    tp       = pending.take_profit,
                    score    = pending.signal_score,
                    exchange = ex.name,
                    is_scale_in = bool(had_same_direction_position),
                    total_size_usd=(self.risk.positions.get(coin).size_usd if had_same_direction_position and self.risk.positions.get(coin) else None),
                )
                self.order_mgr.mark_filled(coin, fill_price)
            elif status.cancelled:
                log.info(f"[{coin}] Limit order cancelled by exchange")
                self.order_mgr.mark_cancelled(coin)

    def _pending_entry_context(self, pending: PendingOrder) -> dict:
        return dict((getattr(pending, "metadata", {}) or {}).get("entry_context", {}) or {})

    def _build_pending_signal(self, pending: PendingOrder, *, live_price: float, reason: str) -> SimpleNamespace:
        entry_context = self._pending_entry_context(pending)
        trade_plan = dict(entry_context.get("trade_plan", {}) or {})
        expectancy = dict(entry_context.get("expectancy", {}) or {})
        thesis = dict(entry_context.get("thesis", {}) or {})
        return SimpleNamespace(
            action=str(pending.direction or "FLAT").upper(),
            score=float(getattr(pending, "signal_score", 50.0) or 50.0),
            confidence=str(entry_context.get("confidence") or "MEDIUM").upper(),
            price=float(live_price or pending.limit_price or 0.0),
            stop_loss_price=float(getattr(pending, "stop_loss", 0.0) or 0.0),
            take_profit_price=float(getattr(pending, "take_profit", 0.0) or 0.0),
            reason=reason,
            flat_reason="",
            trade_plan=trade_plan,
            execution_plan={"mode": "market", "reason": reason},
            expectancy=expectancy,
            thesis=thesis,
        )

    def _cancel_pending_limit(self, pending: PendingOrder, reason: str) -> bool:
        coin = str(pending.coin or "").upper()
        ex = self._get_exchange_by_name(pending.exchange) or (self.exchanges[0] if self.exchanges else None)
        if ex and pending.exchange_order_id:
            try:
                ex.cancel_order(coin, pending.exchange_order_id)
            except Exception as exc:
                log.warning(f"[{coin}] Pending cancel encountered an error: {exc}")
        self.order_mgr.mark_cancelled(coin)
        log.info(f"[{coin}] Pending limit cancelled: {reason}")
        return True

    def _reprice_pending_limit(
        self,
        pending: PendingOrder,
        *,
        new_price: float,
        live_price: float,
        reason: str,
    ) -> bool:
        coin = str(pending.coin or "").upper()
        if new_price <= 0:
            return False
        entry_context = self._pending_entry_context(pending)
        entry_context["reason"] = reason
        if not self._cancel_pending_limit(pending, reason):
            return False
        result = self._place_limit_order(
            coin,
            pending.direction,
            new_price,
            pending.size_usd,
            pending.stop_loss,
            pending.take_profit,
            pending.signal_score,
            reason="reprice",
            entry_context=entry_context,
            trade_plan=dict(entry_context.get("trade_plan", {}) or {}),
            maker_only=True,
            extra_metadata={
                "reprice_count": int(getattr(pending, "reprice_count", 0) or 0) + 1,
                "prior_limit_price": float(getattr(pending, "limit_price", 0.0) or 0.0),
                "pending_management_reason": reason,
            },
        )
        if result.get("success"):
            log.info(
                f"[{coin}] Pending entry repriced from ${pending.limit_price:,.4f} to ${new_price:,.4f} "
                f"after {pending.cycles_waiting} cycles"
            )
        return bool(result.get("success"))

    def _escalate_pending_limit_to_market(
        self,
        pending: PendingOrder,
        *,
        live_price: float,
        reason: str,
        orderbook_signal=None,
    ) -> bool:
        coin = str(pending.coin or "").upper()
        order = OrderRequest(
            coin=coin,
            direction=pending.direction,
            size_usd=pending.size_usd,
            size_coin=(pending.size_usd / max(float(live_price or pending.limit_price or 0.0), 1e-9)),
            price=float(live_price or pending.limit_price or 0.0),
            stop_loss=pending.stop_loss,
            take_profit=pending.take_profit,
            leverage=self.cfg.trading.leverage,
            approved=True,
        )
        synthetic_signal = self._build_pending_signal(pending, live_price=live_price, reason=reason)
        if orderbook_signal and getattr(orderbook_signal, "valid", False):
            synthetic_signal.execution_plan = {
                "mode": "market",
                "reason": reason,
                "entry_price": float(live_price or pending.limit_price or 0.0),
            }
        self._cancel_pending_limit(pending, reason)
        executed = self._execute_order(coin, synthetic_signal, order)
        if executed:
            self._record_precision_entry(coin, synthetic_signal, mode="market_escalation")
            log.info(f"[{coin}] Pending limit escalated to market: {reason}")
        return bool(executed)

    def _manage_pending_limits(self, current_prices: dict, portfolio_usd: float) -> None:
        if not getattr(self.cfg.trading, "execution_pending_management_enabled", True):
            return

        for coin, pending in list(self.order_mgr.pending_orders.items()):
            if self.risk.has_position(coin):
                continue

            live_price = float(current_prices.get(coin) or pending.limit_price or 0.0)
            if live_price <= 0:
                continue

            orderbook_signal = None
            if self._supports_orderbook_context(coin):
                try:
                    orderbook_signal = get_orderbook_levels(
                        coin,
                        current_price=live_price,
                        depth_limit=getattr(self.cfg.trading, "orderbook_depth_limit", 120),
                        daily_lookback=getattr(self.cfg.trading, "orderbook_daily_lookback", 120),
                        cache_ttl_seconds=getattr(self.cfg.trading, "orderbook_cache_ttl_seconds", 25),
                        guard_distance_pct=getattr(self.cfg.trading, "orderbook_guard_distance_pct", 1.25),
                        reaction_distance_pct=getattr(self.cfg.trading, "orderbook_reaction_distance_pct", 0.45),
                        feed_max_age_seconds=getattr(self.cfg.trading, "orderbook_feed_max_snapshot_age_seconds", 45.0),
                        feed_breakout_samples=getattr(self.cfg.trading, "orderbook_feed_breakout_samples", 2),
                    )
                except Exception as exc:
                    log.debug(f"[{coin}] Pending orderbook refresh skipped: {exc}")

            if (
                getattr(self.cfg.trading, "execution_pending_cancel_on_stop_breach", True)
                and (
                    (pending.direction == "LONG" and live_price <= pending.stop_loss)
                    or (pending.direction == "SHORT" and live_price >= pending.stop_loss)
                )
            ):
                self._cancel_pending_limit(pending, "setup invalidated before fill")
                self._record_decision_snapshot(
                    coin,
                    portfolio_usd=portfolio_usd,
                    stage="pending_limit_cancelled",
                    signal=self._build_pending_signal(pending, live_price=live_price, reason="setup invalidated before fill"),
                    blocked=True,
                )
                continue

            breakout_state = str(getattr(orderbook_signal, "breakout_state", "NONE") or "NONE").upper()
            opposite_breakout = (
                pending.direction == "LONG"
                and breakout_state in {"CONFIRMED_BEARISH_BREAKDOWN", "PERSISTENT_BEARISH_BREAKDOWN"}
            ) or (
                pending.direction == "SHORT"
                and breakout_state in {"CONFIRMED_BULLISH_BREAKOUT", "PERSISTENT_BULLISH_BREAKOUT"}
            )
            if (
                getattr(self.cfg.trading, "execution_pending_cancel_on_opposite_breakout", True)
                and opposite_breakout
            ):
                self._cancel_pending_limit(pending, f"opposite orderbook breakout invalidated the setup ({breakout_state})")
                self._record_decision_snapshot(
                    coin,
                    portfolio_usd=portfolio_usd,
                    stage="pending_limit_cancelled",
                    signal=self._build_pending_signal(pending, live_price=live_price, reason=f"opposite breakout {breakout_state}"),
                    blocked=True,
                )
                continue

            should_reprice = (
                getattr(self.cfg.trading, "execution_pending_reprice_enabled", True)
                and pending.reason in {"initial_limit", "passive_rescue", "reprice"}
                and pending.cycles_waiting >= int(getattr(self.cfg.trading, "execution_pending_reprice_after_cycles", 2) or 2)
                and int(getattr(pending, "reprice_count", 0) or 0) < int(getattr(self.cfg.trading, "execution_pending_max_reprices", 3) or 3)
                and orderbook_signal
                and getattr(orderbook_signal, "valid", False)
            )
            if should_reprice:
                target_price = float(
                    getattr(orderbook_signal, "best_bid", 0.0) if pending.direction == "LONG"
                    else getattr(orderbook_signal, "best_ask", 0.0)
                )
                threshold_bps = float(getattr(self.cfg.trading, "execution_pending_reprice_threshold_bps", 8.0) or 8.0)
                drift_bps = abs(target_price - pending.limit_price) / max(pending.limit_price, 1e-9) * 10_000.0 if target_price > 0 else 0.0
                if target_price > 0 and drift_bps >= threshold_bps:
                    if self._reprice_pending_limit(
                        pending,
                        new_price=target_price,
                        live_price=live_price,
                        reason=f"top of book drifted {drift_bps:.1f}bps while the thesis stayed intact",
                    ):
                        self._record_decision_snapshot(
                            coin,
                            portfolio_usd=portfolio_usd,
                            stage="pending_limit_repriced",
                            signal=self._build_pending_signal(pending, live_price=live_price, reason="pending order repriced"),
                            pending_limit=True,
                        )
                        continue

            should_escalate = (
                getattr(self.cfg.trading, "execution_pending_market_escalation_enabled", True)
                and pending.reason in {"initial_limit", "passive_rescue", "reprice"}
                and pending.cycles_waiting >= int(getattr(self.cfg.trading, "execution_pending_market_escalation_after_cycles", 3) or 3)
                and orderbook_signal
                and getattr(orderbook_signal, "valid", False)
            )
            if should_escalate:
                breakout_ok = breakout_state in {
                    "CONFIRMED_BULLISH_BREAKOUT",
                    "PERSISTENT_BULLISH_BREAKOUT",
                    "CONFIRMED_BEARISH_BREAKDOWN",
                    "PERSISTENT_BEARISH_BREAKDOWN",
                }
                if (
                    not getattr(self.cfg.trading, "execution_pending_market_escalation_breakout_only", True)
                    or breakout_ok
                ):
                    order = OrderRequest(
                        coin=coin,
                        direction=pending.direction,
                        size_usd=pending.size_usd,
                        size_coin=(pending.size_usd / max(live_price, 1e-9)),
                        price=live_price,
                        stop_loss=pending.stop_loss,
                        take_profit=pending.take_profit,
                        leverage=self.cfg.trading.leverage,
                        approved=True,
                    )
                    execution_quality = self._assess_execution_quality(coin, pending.direction, order, orderbook_signal)
                    if (
                        execution_quality.get("permitted", False)
                        and float(execution_quality.get("spread_bps", 999.0) or 999.0)
                        <= float(getattr(self.cfg.trading, "execution_pending_market_escalation_max_spread_bps", 10.0) or 10.0)
                        and float(execution_quality.get("estimated_slippage_bps", 999.0) or 999.0)
                        <= float(getattr(self.cfg.trading, "execution_pending_market_escalation_max_slippage_bps", 16.0) or 16.0)
                        and float(execution_quality.get("score", 0.0) or 0.0)
                        >= float(getattr(self.cfg.trading, "execution_pending_market_escalation_min_quality_score", 72.0) or 72.0)
                    ):
                        if self._escalate_pending_limit_to_market(
                            pending,
                            live_price=live_price,
                            reason="breakout is running away and fill quality is finally clean",
                            orderbook_signal=orderbook_signal,
                        ):
                            self._record_decision_snapshot(
                                coin,
                                portfolio_usd=portfolio_usd,
                                stage="pending_limit_market_escalation",
                                signal=self._build_pending_signal(pending, live_price=live_price, reason="pending limit escalated to market"),
                                executed=True,
                            )
                            continue

    # ── Exit handling ─────────────────────────────────────────

    def _check_and_execute_exits(self, current_prices: dict, portfolio_usd: float):
        exits = self.risk.check_exits(current_prices)
        for info in exits:
            coin   = info["coin"]
            reason = info["reason"]
            price  = info["price"]
            log.info(f"[{coin}] Exit triggered: {reason} @ ${price:.2f}")
            self._close_position(coin, reason, price)

            # No-emotion re-entry: if TP was hit, schedule a limit re-entry
            if info.get("was_take_profit"):
                exit_entry_context = dict((info.get("metadata") or {}).get("entry_context", {}) or {})
                log.info(f"[{coin}] TP hit — scheduling re-entry watch "
                         f"(fib retracement from ${info['entry_price']:.2f} → ${info['tp_price']:.2f})")
                self.order_mgr.schedule_reentry(
                    coin         = coin,
                    direction    = info["direction"],
                    entry_price  = info["entry_price"],
                    tp_price     = info["tp_price"],
                    size_usd     = min(info["size_usd"], self.cfg.trading.max_trade_usd),
                    signal_score = 65.0,   # assume medium-high conviction for re-entry
                    trade_plan   = dict(exit_entry_context.get("trade_plan", {}) or {}),
                    entry_context= exit_entry_context,
                )

    def _close_position(self, coin: str, reason: str, price: float):
        pos   = self.risk.positions.get(coin)
        entry = pos.entry_price if pos else price
        open_trade_snapshot = trade_logger.get_open_trade(coin)

        exit_price = price
        exchanges = self._eligible_exchanges(coin)
        if pos and pos.exchange:
            preferred = self._get_exchange_by_name(pos.exchange)
            exchanges = ([preferred] if preferred and coin in preferred.supported_coins() else []) + [
                ex for ex in exchanges if not preferred or ex.name != preferred.name
            ]

        close_successful = False
        verified_close = False
        for ex in exchanges:
            result = None
            for attempt in range(1, 4):
                result = ex.close_position(coin)
                if result.success:
                    close_successful = True
                    break
                else:
                    if attempt < 3:
                        log.warning(f"[{coin}] Close attempt {attempt}/3 failed: {result.error}. "
                                   f"Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        log.error(f"[{coin}] Close failed on {ex.name} after 3 attempts: {result.error}")

            if close_successful and result:
                exit_price = result.filled_price or price
                log.info(f"[{coin}] Closed on {ex.name} ({reason})")

                time.sleep(1)
                verification = self._verify_position_on_exchange(ex, coin, should_exist=False)
                verified_close = verification is True or self._reconcile_and_check_coin(coin, should_exist=False)
                if verified_close:
                    log.info(f"[{coin}] Close verified on {ex.name}")
                else:
                    log.critical(
                        f"[{coin}] CRITICAL: close succeeded on {ex.name} but the position "
                        f"still appears open or could not be reconciled"
                    )
                    self.notifier.error_alert(
                        f"CRITICAL: {coin} close succeeded on {ex.name} but verification failed"
                    )
                break

        if not close_successful:
            self.notifier.error_alert(f"CRITICAL: failed to close {coin} after all retries")
            return

        if not verified_close:
            return

        if pos:
            direction = pos.direction
            if direction == "LONG":
                pnl_pct = (exit_price - entry) / entry
            else:
                pnl_pct = (entry - exit_price) / entry
            pnl_usd = pnl_pct * pos.size_usd
            self.notifier.trade_closed(coin, direction, entry, exit_price, pnl_usd, reason)

        was_tp = reason == "take_profit"
        self.risk.record_close(coin, exit_price, reason, was_take_profit=was_tp)
        csv_trade = trade_logger.log_close(coin, exit_price=exit_price, exit_reason=reason)

        # ── Record outcome to trade memory for self-learning ─────────────────
        if pos:
            hold_minutes = (time.time() - pos.opened_at) / 60.0 if pos.opened_at else 0.0
            last_sig     = self._last_signals.get(coin, {})
            entry_ctx    = dict((getattr(pos, "metadata", {}) or {}).get("entry_context", {}) or {})
            entry_ctx    = self._annotate_exit_against_plan(pos.direction, entry_ctx, entry, exit_price)
            dataset_record = self._build_closed_trade_dataset_record(
                coin,
                pos,
                exit_price,
                reason,
                hold_minutes,
                entry_ctx,
                last_sig,
                csv_trade or open_trade_snapshot,
            )
            if dataset_record:
                trade_dataset.append_closed_trade(dataset_record)
                if getattr(self.cfg.trading, "feature_store_enabled", True):
                    try:
                        feature_store.append_closed_trade_feature_row(dataset_record)
                    except Exception as exc:
                        log.debug(f"[{coin}] Closed-trade feature row skipped: {exc}")
            trend_ctx    = entry_ctx.get("mtf_bias") or last_sig.get("mtf_bias", "FLAT")
            # Pull regime context so RL can learn which regimes are profitable
            market_regime   = entry_ctx.get("market_regime") or last_sig.get("market_regime",   "RANGING")
            dominant_regime = entry_ctx.get("dominant_regime") or last_sig.get("dominant_regime", "MIXED")
            self._memory.record_trade(
                coin             = coin,
                direction        = pos.direction,
                signal_score     = entry_ctx.get("score") or last_sig.get("score", 50.0),
                entry_price      = entry,
                exit_price       = exit_price,
                exit_reason      = reason,
                hold_minutes     = hold_minutes,
                trend_context    = trend_ctx,
                market_regime    = market_regime,
                dominant_regime  = dominant_regime,
                volatility_label = entry_ctx.get("volatility_label", "NORMAL"),
                entry_context    = entry_ctx,
            )

    def emergency_close_all(self, reason: str):
        """
        Emergency close-all kill switch.
        Immediately closes all open positions in self.risk.positions.
        """
        log.critical(f"EMERGENCY CLOSE-ALL TRIGGERED: {reason}")
        self.notifier.error_alert(f"EMERGENCY: Closing all positions. Reason: {reason}")

        open_coins = list(self.risk.positions.keys())
        if not open_coins:
            log.info("No open positions to close")
            return

        log.critical(f"Closing {len(open_coins)} open position(s): {open_coins}")
        for coin in open_coins:
            try:
                log.critical(f"[{coin}] Emergency closing position...")
                self._close_position(coin, f"emergency_close_all:{reason}", 0.0)
            except Exception as e:
                log.critical(f"[{coin}] Emergency close failed with exception: {e}")
                self.notifier.error_alert(f"CRITICAL: Failed to emergency close {coin}: {e}")

        log.critical("Emergency close-all complete")

    # ── Helpers ───────────────────────────────────────────────

    # ── Chart confirmation ────────────────────────────────────

    def _get_chart_confirmation(
        self, coin: str, indicator_score: float
    ) -> Optional[ChartVerdict]:
        """
        Run visual chart analysis for borderline signals.
        Only fires when indicator_score is in the "I'm not sure" range.
        Strong signals skip this to avoid unnecessary API calls.
        """
        cfg = self.cfg.trading
        lo  = cfg.chart_confirm_score_low    # default 38
        hi  = cfg.chart_confirm_score_high   # default 62

        if not (lo <= indicator_score <= hi):
            # High conviction — trust the indicators, skip visual check
            log.debug(f"[{coin}] Score {indicator_score:.1f} is high-conviction "
                      f"— skipping chart visual check")
            return None

        # Try to get a chart URL from config overrides
        chart_url = (cfg.chart_urls or {}).get(coin)

        # Option A: Use the screener (headless browser auto-capture)
        if cfg.use_chart_screener:
            try:
                from indicators.chart_screener import screen_coin
                return screen_coin(coin, url=chart_url,
                                   save_screenshots=cfg.save_chart_screenshots)
            except Exception as e:
                log.warning(f"[{coin}] Chart screener failed: {e}")

        # Option B: Use a pre-saved screenshot if the user dropped one in
        screenshot_path = f"screenshots/{coin}_latest.png"
        import os
        if os.path.exists(screenshot_path):
            log.info(f"[{coin}] Using saved screenshot: {screenshot_path}")
            return read_chart(coin=coin, image_path=screenshot_path)

        log.debug(f"[{coin}] No chart image source available — skipping visual check")
        return None

    def _get_portfolio_usd(self) -> Optional[float]:
        total_equity = 0.0
        any_state = False
        for ex in self.exchanges:
            state = self._get_account_state_safe(ex)
            if state:
                any_state = True
                total_equity += state.total_equity_usd
        return total_equity if any_state else None

    def _fetch_all_prices(self) -> dict:
        prices = {}
        for coin in self._tradable_coins:
            try:
                p = self._price_circuits[coin].call(get_current_price, coin)
                if p:
                    prices[coin] = p
            except CircuitBreakerError:
                log.warning(f"[{coin}] Price feed circuit open")
            except Exception as e:
                log.warning(f"[{coin}] Price fetch failed: {e}")
        return prices

    def _get_account_state_safe(self, ex: BaseExchange):
        try:
            circuit = self._exchange_circuits.get(ex.name)
            if circuit:
                return circuit.call(ex.get_account_state)
            return ex.get_account_state()
        except CircuitBreakerError as e:
            log.warning(f"[{ex.name}] Circuit open - skipping account state: {e}")
        except Exception as e:
            log.error(f"[{ex.name}] Failed to get account state: {e}")
        return None

    def _get_sentiment_safe(self) -> Optional[dict]:
        try:
            return self._sentiment_circuit.call(get_fear_greed_score)
        except CircuitBreakerError:
            log.warning("Sentiment circuit open - using default")
        except Exception as e:
            log.warning(f"Sentiment fetch failed: {e}")
        return None

    def _check_mtf_safe(self, coin: str, signal) -> bool:
        try:
            mtf = self._mtf_circuit.call(compute_mtf, coin)
            # Persist MTF bias for trade memory context
            if coin in self._last_signals:
                self._last_signals[coin]["mtf_bias"] = mtf.combined_bias or "FLAT"
                self._last_signals[coin]["mtf_status"] = "ok"
            if signal.action == "LONG" and not mtf.allow_long:
                log.info(f"[{coin}] 🕐 MTF blocks LONG — {mtf.reason}")
                return False
            if signal.action == "SHORT" and not mtf.allow_short:
                log.info(f"[{coin}] 🕐 MTF blocks SHORT — {mtf.reason}")
                return False
            log.info(f"[{coin}] MTF combined={mtf.combined_bias} adj={mtf.score_adjustment:+.0f}")
            return True
        except CircuitBreakerError:
            log.warning(f"[{coin}] MTF circuit open")
            if coin in self._last_signals:
                self._last_signals[coin]["mtf_bias"] = "UNAVAILABLE"
                self._last_signals[coin]["mtf_status"] = "circuit_open"
            return not getattr(self.cfg.trading, "strict_confirmation_fail_closed", True)
        except Exception as e:
            log.warning(f"[{coin}] MTF failed: {e}")
            if coin in self._last_signals:
                self._last_signals[coin]["mtf_bias"] = "UNAVAILABLE"
                self._last_signals[coin]["mtf_status"] = "error"
            return not getattr(self.cfg.trading, "strict_confirmation_fail_closed", True)

    def _check_news_safe(self, coin: str, signal) -> bool:
        """Legacy wrapper — kept for compatibility. Use _check_news_extreme instead."""
        try:
            news = self._news_circuit.call(
                get_news_signal,
                coin,
                self.cfg.trading.cryptopanic_auth_token,
            )
            return self._check_news_extreme(coin, signal, news)
        except CircuitBreakerError:
            log.warning(f"[{coin}] News circuit open")
            return not getattr(self.cfg.trading, "strict_confirmation_fail_closed", True)
        except Exception as e:
            log.warning(f"[{coin}] News signal failed: {e}")
            return not getattr(self.cfg.trading, "strict_confirmation_fail_closed", True)

    def _check_news_extreme(self, coin: str, signal, news) -> bool:
        """
        Secondary news gate — only blocks on catastrophically contrary news.
        Normal news is already baked into the signal score; this is the last
        line of defence against 'hack detected' / 'exchange collapse' events.
        """
        if not news or not news.valid:
            return True
        if news.is_extreme:
            if signal.action == "LONG" and news.score < 25:
                log.info(
                    f"[{coin}] 📰 EXTREME bearish news (score={news.score:.0f}) "
                    f"— hard-blocking LONG. "
                    f"Headline: {news.top_headlines[0][:70] if news.top_headlines else ''}"
                )
                return False
            if signal.action == "SHORT" and news.score > 75:
                log.info(
                    f"[{coin}] 📰 EXTREME bullish news (score={news.score:.0f}) "
                    f"— hard-blocking SHORT. "
                    f"Headline: {news.top_headlines[0][:70] if news.top_headlines else ''}"
                )
                return False
        return True

    def _save_checkpoint(self):
        exchange_states = {}
        for ex in self.exchanges:
            exchange_states[ex.name] = {
                "circuit_state": self._exchange_circuits[ex.name].state.value,
            }
        checkpoint_manager.save(
            cycle_number=self._cycle,
            portfolio_usd=self._last_portfolio_usd,
            available_usd=self._last_available_usd,
            positions=self.risk.positions,
            pending_orders=self.order_mgr.pending_orders,
            reentry_watches=self.order_mgr.reentry_watches,
            risk_manager=self.risk,
            exchange_states=exchange_states,
        )

    def _log_circuit_status(self):
        states = circuit_breaker_registry.get_states()
        if not states:
            return
        log.info("Circuit breaker status:")
        for name, state in states.items():
            log.info(f"  {name}: {state}")

    def _get_exchange_by_name(self, name: str) -> Optional[BaseExchange]:
        for ex in self.exchanges:
            if ex.name == name:
                return ex
        return None

    def _eligible_exchanges(self, coin: str) -> List[BaseExchange]:
        return [ex for ex in self.exchanges if coin in ex.supported_coins()]

    def _verify_position_on_exchange(self, ex: BaseExchange, coin: str, should_exist: bool) -> Optional[bool]:
        state = self._get_account_state_safe(ex)
        if not state:
            return None
        exists = any(pos.get("coin") == coin for pos in state.positions)
        return exists == should_exist

    def _reconcile_and_check_coin(self, coin: str, should_exist: bool) -> bool:
        try:
            self._reconcile_with_exchange()
        except Exception as exc:
            log.critical(f"[{coin}] Reconciliation after verification failure crashed: {exc}")
            return False
        return self.risk.has_position(coin) if should_exist else not self.risk.has_position(coin)

    # ── Dashboard state writer ────────────────────────────────

    def _write_state(self, portfolio_usd: float, sentiment: dict):
        """Write current agent state to state.json for the dashboard."""
        import os
        from pathlib import Path

        positions_out = []
        for coin, p in self.risk.positions.items():
            price = get_current_price(coin) or p.entry_price
            if p.direction == "LONG":
                upnl = (price - p.entry_price) / p.entry_price * p.size_usd
            else:
                upnl = (p.entry_price - price) / p.entry_price * p.size_usd
            positions_out.append({
                "coin":          coin,
                "direction":     p.direction,
                "entry_price":   p.entry_price,
                "current_price": price,
                "stop_loss":     p.stop_loss,
                "take_profit":   p.take_profit,
                "trailing_stop": p.trailing_stop_price,
                "size_usd":      p.size_usd,
                "unrealised_pnl":round(upnl, 2),
                "opened_at":     getattr(p, "opened_at", None),
            })

        pending_out = []
        for coin, o in self.order_mgr.pending_orders.items():
            pending_out.append({
                "coin":        coin,
                "direction":   o.direction,
                "limit_price": o.limit_price,
                "size_usd":    o.size_usd,
                "cycles_waiting": o.cycles_waiting,
                "reprice_count": getattr(o, "reprice_count", 0),
                "reason": getattr(o, "reason", ""),
                "max_cycles":  15,
            })

        # Enrich positions with hold-time and anti-whipsaw data
        import time as _time
        for p_out in positions_out:
            coin = p_out["coin"]
            pos  = self.risk.positions.get(coin)
            if pos and pos.opened_at:
                hold_mins = (_time.time() - pos.opened_at) / 60.0
                itype     = self.cfg.trading.instrument_types.get(coin, "crypto")
                min_hold  = (self.cfg.trading.index_min_hold_minutes
                             if itype == "index"
                             else self.cfg.trading.min_hold_minutes)
                p_out["hold_minutes"]       = round(hold_mins, 0)
                p_out["min_hold_minutes"]   = min_hold
                p_out["reversal_locked"]    = hold_mins < min_hold
                p_out["reversal_unlock_in"] = max(0, round(min_hold - hold_mins, 0))
                # Conviction tier from risk manager
                sig = getattr(self, "_last_signals", {}).get(coin, {})
                p_out["conviction_tier"]    = sig.get("conviction_tier", "")
                p_out["candle_patterns"]    = sig.get("candle_patterns", [])
                p_out["news_score"]         = sig.get("news_score", 50.0)
                p_out["memory_cooldown"]    = sig.get("memory_cooldown", 0)
                entry_ctx = dict((pos.metadata or {}).get("entry_context", {}) or {})
                entry_thesis = dict(entry_ctx.get("thesis", {}) or {})
                p_out["entry_logic"] = (
                    entry_ctx.get("reason")
                    or entry_thesis.get("summary")
                    or sig.get("thesis_summary", "")
                )
                p_out["current_logic"] = (
                    sig.get("decision_reason")
                    or sig.get("thesis_summary", "")
                    or ""
                )

        state = {
            "status":        "running",
            "mode":          "dry_run" if self.cfg.is_dry_run else "live",
            "leverage":      self.cfg.trading.leverage,
            "last_cycle":    __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cycle_number":  self._cycle,
            "portfolio_usd": round(portfolio_usd, 2),
            "available_usd": round(self.risk.available_capital(portfolio_usd), 2),
            "positions":     positions_out,
            "pending_orders":pending_out,
            "signals":       getattr(self, "_last_signals", {}),
            "sentiment":     sentiment,
            "daily_pnl_usd":     round(getattr(self.risk, "daily_pnl_usd", 0.0), 2),
            "daily_trades":      getattr(self.risk, "daily_trades", 0),
            "daily_pnl_history": getattr(self.risk, "daily_pnl_history", {}),
            "circuit_health": circuit_breaker_registry.health_check(),
            "loss_circuit_breaker": self.risk.circuit_breaker_status(),
            "trade_memory":  self._memory.get_stats(),
            "power":         self._last_power_status,
            "config": {
                "long_threshold":        self.cfg.trading.signal_long_threshold,
                "short_threshold":       self.cfg.trading.signal_short_threshold,
                "min_hold_minutes":      self.cfg.trading.min_hold_minutes,
                "index_min_hold_minutes": self.cfg.trading.index_min_hold_minutes,
                "reversal_boost":        self.cfg.trading.reversal_threshold_boost,
                "check_interval_seconds": self.cfg.trading.check_interval_seconds,
                "use_daily_market_map":  getattr(self.cfg.trading, "use_daily_market_map", True),
                "coins":                 self._tradable_coins,
                "analysis_coins":        self._analysis_coins,
                "dynamic_analysis_coins": self._dynamic_analysis_coins,
                "dynamic_market_cap_min_usd": float(
                    getattr(self.cfg.trading, "dynamic_market_cap_min_usd", 1_000_000_000.0) or 1_000_000_000.0
                ),
                "instrument_types":      self.cfg.trading.instrument_types,
                "asset_categories":      getattr(self.cfg.trading, "asset_category_map", {}),
                "asset_category_labels": getattr(self.cfg.trading, "asset_category_labels", {}),
            },
        }

        state_path = STATE_JSON
        try:
            state_path.write_text(json.dumps(state, indent=2))
        except Exception as e:
            log.debug(f"state.json write failed: {e}")

        trades_data = []
        log_path = TRADES_CSV
        if log_path.exists():
            try:
                trades_data = trade_logger.read_closed_trades()
            except Exception as e:
                log.debug(f"trades_log.csv read failed: {e}")

        history_data_dir = trade_dataset.resolve_richest_history_data_dir()
        decision_history_dir = decision_dataset.resolve_richest_decision_data_dir(history_data_dir)
        trade_dataset_records = []
        try:
            trade_dataset_records = trade_dataset.load_closed_trades(
                limit=max(200, len(trades_data) + 20),
                data_dir=history_data_dir,
            )
        except Exception as e:
            log.debug(f"trade_dataset.jsonl read failed: {e}")
        decision_dataset_records = []
        try:
            decision_dataset_records = decision_dataset.load_decisions(limit=25000, data_dir=decision_history_dir)
        except Exception as e:
            log.debug(f"decision_dataset.jsonl read failed: {e}")
        enriched_trade_records = merge_dataset_into_trades(trades_data, trade_dataset_records)

        market_map_data = market_map.build_effective_market_map(
            self._analysis_coins,
            current_prices={
                coin: float((sig or {}).get("live_price") or (sig or {}).get("price") or 0.0)
                for coin, sig in (getattr(self, "_last_signals", {}) or {}).items()
            },
            closed_prices={
                coin: float((sig or {}).get("analysis_price") or (sig or {}).get("price") or 0.0)
                for coin, sig in (getattr(self, "_last_signals", {}) or {}).items()
            },
        )
        review_data = trade_review.load_reviews()
        decision_review_data = {}
        if DECISION_REVIEW_REPORT_JSON.exists():
            try:
                decision_review_data = json.loads(DECISION_REVIEW_REPORT_JSON.read_text())
            except Exception as e:
                log.debug(f"decision_review_report.json read failed: {e}")
        missed_move_report_data = {}
        if MISSED_MOVE_REPORT_JSON.exists():
            try:
                missed_move_report_data = json.loads(MISSED_MOVE_REPORT_JSON.read_text())
            except Exception as e:
                log.debug(f"missed_move_report.json read failed: {e}")
        challenger_report_data = {}
        if CHALLENGER_MODEL_JSON.exists():
            try:
                challenger_report_data = json.loads(CHALLENGER_MODEL_JSON.read_text())
            except Exception as e:
                log.debug(f"challenger_model_report.json read failed: {e}")
        playbook_distiller_report_data = {}
        if PLAYBOOK_DISTILLER_REPORT_JSON.exists():
            try:
                playbook_distiller_report_data = json.loads(PLAYBOOK_DISTILLER_REPORT_JSON.read_text())
            except Exception as e:
                log.debug(f"playbook_distiller_report.json read failed: {e}")
        llm_referee_report_data = self._llm_referee.default_report()
        if LLM_REFEREE_REPORT_JSON.exists():
            try:
                llm_referee_report_data = json.loads(LLM_REFEREE_REPORT_JSON.read_text())
            except Exception as e:
                log.debug(f"llm_referee_report.json read failed: {e}")
        asset_dossier_data = {}
        try:
            if getattr(self.cfg.trading, "asset_dossier_enabled", True):
                asset_dossier_data = asset_dossier.build_and_save_report(
                    state=state,
                    trades=enriched_trade_records,
                    market_map=market_map_data,
                    missed_move_report=missed_move_report_data,
                    llm_referee_report=llm_referee_report_data,
                    playbook_distiller_report=playbook_distiller_report_data,
                )
        except Exception as e:
            log.debug(f"asset_dossiers.json write failed: {e}")
            if ASSET_DOSSIERS_JSON.exists():
                try:
                    asset_dossier_data = json.loads(ASSET_DOSSIERS_JSON.read_text())
                except Exception:
                    asset_dossier_data = {}

        control_data = default_control()
        control_path = CONTROL_JSON
        if control_path.exists():
            try:
                control_data = json.loads(control_path.read_text())
            except Exception as e:
                log.debug(f"control.json read failed: {e}")

        snapshot = build_dashboard_snapshot(
            state,
            trades_data,
            control_data,
            market_map=market_map_data,
            trade_reviews=review_data,
            trade_dataset_records=trade_dataset_records,
            decision_dataset_records=decision_dataset_records,
            decision_review_report=decision_review_data,
            challenger_report=challenger_report_data,
            missed_move_report=missed_move_report_data,
            asset_dossiers=asset_dossier_data,
            llm_referee_report=llm_referee_report_data,
            playbook_distiller_report=playbook_distiller_report_data,
        )
        try:
            DASHBOARD_SNAPSHOT_JSON.write_text(json.dumps(snapshot, indent=2))
        except Exception as e:
            log.debug(f"dashboard_snapshot.json write failed: {e}")

        # Push the exact local dashboard snapshot to the hosted dashboard.
        remote_url = os.environ.get("DASHBOARD_URL", "")
        remote_push_ok = False
        remote_push_used_fallback = False
        if remote_url:
            try:
                import ssl
                import urllib.request

                payload = json.dumps({
                    "snapshot": snapshot,
                    "state": state,
                    "trades": trades_data,
                    "control": control_data,
                    "market_map": market_map_data,
                    "trade_reviews": review_data,
                    "decision_review_report": decision_review_data,
                    "challenger_report": challenger_report_data,
                    "missed_move_report": missed_move_report_data,
                    "asset_dossiers": asset_dossier_data,
                    "llm_referee_report": llm_referee_report_data,
                    "playbook_distiller_report": playbook_distiller_report_data,
                }).encode()
                req = urllib.request.Request(
                    remote_url.rstrip("/") + "/api/push",
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-Token": os.environ.get("DASHBOARD_TOKEN", ""),
                    },
                    method="POST"
                )
                ctx = ssl.create_default_context()
                try:
                    import certifi
                    ctx.load_verify_locations(certifi.where())
                except ImportError:
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                resp = urllib.request.urlopen(req, timeout=15, context=ctx)
                body = resp.read().decode() if hasattr(resp, "read") else ""
                remote_payload = {}
                if body:
                    try:
                        remote_payload = json.loads(body)
                    except Exception:
                        remote_payload = {"raw": body}
                remote_push_ok = bool(getattr(resp, "status", 200) < 400)
                if isinstance(remote_payload, dict) and remote_payload.get("ok") is False:
                    remote_push_ok = False
                remote_push_used_fallback = bool(
                    isinstance(remote_payload, dict)
                    and (
                        remote_payload.get("fallback")
                        or str(remote_payload.get("storage") or "").lower() == "fallback"
                    )
                )

                # Check for remote kill signal
                try:
                    kill_url = remote_url.rstrip("/") + "/api/state"
                    kill_req = urllib.request.Request(kill_url, method="GET")
                    kill_resp = urllib.request.urlopen(kill_req, timeout=10, context=ctx)
                    kill_data = json.loads(kill_resp.read().decode())
                    kill_sig = ((kill_data.get("control") or {}).get("kill") or {})
                    if kill_sig and kill_sig.get("active"):
                        kill_reason = kill_sig.get("reason", "remote dashboard kill")
                        log.critical(f"🚨 REMOTE KILL SIGNAL: {kill_reason}")
                        kill_file = KILL_FILE
                        kill_file.write_text(kill_reason)
                        try:
                            ack_payload = json.dumps({
                                "active": False,
                                "reason": f"Agent acknowledged kill: {kill_reason}",
                            }).encode()
                            ack_req = urllib.request.Request(
                                remote_url.rstrip("/") + "/api/kill",
                                data=ack_payload,
                                headers={
                                    "Content-Type": "application/json",
                                    "X-Token": os.environ.get("DASHBOARD_TOKEN", ""),
                                },
                                method="POST",
                            )
                            urllib.request.urlopen(ack_req, timeout=10, context=ctx)
                        except Exception:
                            pass
                except Exception:
                    pass  # kill signal check is best-effort
            except Exception as e:
                detail = str(e)
                if hasattr(e, "read"):
                    try:
                        body = e.read().decode()
                        if body:
                            detail = f"{detail} | {body[:220]}"
                    except Exception:
                        pass
                log.debug(f"Remote dashboard push failed: {detail}")

        if (not remote_push_ok) or remote_push_used_fallback:
            hosted_state_sync.publish_snapshot(
                snapshot,
                state=state,
                trades=trades_data,
                control=control_data,
                market_map=market_map_data,
                trade_reviews=review_data,
                decision_review_report=decision_review_data,
                challenger_report=challenger_report_data,
                missed_move_report=missed_move_report_data,
                asset_dossiers=asset_dossier_data,
                llm_referee_report=llm_referee_report_data,
                playbook_distiller_report=playbook_distiller_report_data,
            )

    def _print_final_summary(self):
        log.info("\n" + "=" * 64)
        log.info("  SESSION SUMMARY")
        log.info("=" * 64)
        if self.risk.trade_log:
            total_pnl = sum(t["pnl_usd"] for t in self.risk.trade_log)
            wins  = sum(1 for t in self.risk.trade_log if t["pnl_usd"] > 0)
            total = len(self.risk.trade_log)
            log.info(f"  Trades: {total}  |  Win rate: {wins/total*100:.1f}%  |  "
                     f"Total PnL: ${total_pnl:+,.2f}")
            for t in self.risk.trade_log:
                tag = "✅" if t["pnl_usd"] >= 0 else "❌"
                log.info(
                    f"  {tag} {t['coin']} {t['direction']:6s} "
                    f"entry=${t['entry']:.2f} exit=${t['exit']:.2f} "
                    f"PnL={t['pnl_pct']*100:+.2f}% (${t['pnl_usd']:+.2f}) [{t['reason']}]"
                )
        else:
            log.info("  No completed trades this session.")
        log.info("=" * 64)
