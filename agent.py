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

import time
import sys
from pathlib import Path
from typing import List, Optional, Dict
from paths import STATE_JSON, TRADES_CSV, KILL_FILE
from datetime import datetime

import os
from config import Config
from logger import get_logger
from data.market_data import fetch_candles, get_current_price
import trade_logger
from indicators.technical import compute_signals
from indicators.advanced  import compute_advanced_signals
from indicators.sentiment import get_fear_greed_score, sentiment_summary
from indicators.regimes   import compute_regimes
from indicators.chart_analyst import read_chart, ChartVerdict
from indicators.mtf  import compute_mtf, MTFAnalysis
from indicators.news import get_news_signal
from indicators.candlestick_patterns import compute_candlestick_patterns
from indicators.trade_memory import trade_memory
from indicators.funding_oi_cvd import get_funding_oi_cvd
from strategy.aggressive_strategy import AggressiveStrategy
from strategy.order_manager import OrderManager, PendingOrder, ReEntryWatch
from risk.risk_manager import RiskManager, OrderRequest, OpenPosition
from exchanges.base import BaseExchange
from notifications import build_notifier
from checkpoint import checkpoint_manager, load_checkpoint
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
        self._price_circuits = {
            coin: get_price_feed_circuit(f"primary_{coin.lower()}")
            for coin in cfg.trading.coins
        }
        self._sentiment_circuit = get_indicator_circuit("sentiment")
        self._mtf_circuit = get_indicator_circuit("mtf")
        self._news_circuit = get_indicator_circuit("news")
        self._exchange_circuits = {
            ex.name: get_exchange_circuit(ex.name) for ex in exchanges
        }
        self._memory = trade_memory          # reinforcement learning module

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

        self._attempt_recovery()
        self._reconcile_with_exchange()

    # ── Control ───────────────────────────────────────────────

    def start(self):
        """Run the agent loop. Press Ctrl+C to stop."""
        log.info("=" * 64)
        log.info(f"  Crypto Perps Agent  |  Mode: "
                 f"{'DRY RUN 🟡' if self.cfg.is_dry_run else 'LIVE 🔴'}")
        log.info(f"  Coins     : {self.cfg.trading.coins}")
        log.info(f"  Exchanges : {[e.name for e in self.exchanges]}")
        log.info(f"  Leverage  : {self.cfg.trading.leverage}×")
        log.info(f"  SL / TP   : {self.cfg.trading.stop_loss_pct*100:.0f}% / "
                 f"{self.cfg.trading.take_profit_pct*100:.0f}%")
        log.info(f"  Trailing  : {self.cfg.trading.trailing_stop_pct*100:.0f}%")
        log.info(f"  Limit orders: "
                 f"{'YES' if any(e.supports_limit_orders() for e in self.exchanges) else 'NO (market fallback)'}")
        log.info("=" * 64)
        self._log_circuit_status()
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

    def stop(self):
        self._running = False

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

        for coin, pos in checkpoint.get("positions", {}).items():
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
                    reason=order.get("reason", "re_entry"),
                    placed_at=float(order.get("placed_at", time.time()) or time.time()),
                )
            )

        for coin, watch in checkpoint.get("reentry_watches", {}).items():
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

        # 1. Portfolio equity
        portfolio_usd = self._get_portfolio_usd()
        if portfolio_usd is None:
            log.error("Cannot read portfolio value - using last known value")
            portfolio_usd = self._last_portfolio_usd or 0.0
        self._last_portfolio_usd = portfolio_usd
        self._last_available_usd = self.risk.available_capital(portfolio_usd)
        self.risk.update_peak_portfolio(portfolio_usd)
        log.info(f"Portfolio equity: ${portfolio_usd:,.2f}")

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

        # 6. Tick trade memory cooldowns (once per cycle)
        self._memory.tick_cooldowns()
        log.info(self._memory.summary())

        # 7. Analyse each coin for new positions
        self._last_signals = {}
        for coin in self.cfg.trading.coins:
            # Skip if we already have a pending limit order for this coin
            if self.order_mgr.has_pending(coin):
                log.info(f"[{coin}] Skipping analysis — limit order pending")
                continue
            try:
                self._analyse_coin(coin, sentiment, portfolio_usd)
            except Exception as e:
                log.error(f"[{coin}] Unexpected error: {e}", exc_info=True)

        # 8. Summary + write state.json for dashboard
        log.info("\n" + self.risk.summary(portfolio_usd))
        log.info(self.order_mgr.summary())
        self._write_state(portfolio_usd, sentiment)

        # 8. Heartbeat notification every 6 cycles
        if self._cycle % 6 == 0:
            self.notifier.heartbeat(portfolio_usd, len(self.risk.positions))

    # ── Coin analysis ─────────────────────────────────────────

    def _analyse_coin(self, coin: str, sentiment: dict, portfolio_usd: float):
        log.info(f"[{coin}] Analysing…")
        icfg = self.cfg.indicators

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
            log.warning(f"[{coin}] No candle data — skipping")
            return

        # Classic indicators
        tech = compute_signals(df, coin, icfg, self.cfg.trading)

        # Advanced / structure indicators
        advanced = compute_advanced_signals(df, coin)

        # Market regime indicators
        try:
            regimes = compute_regimes(df, coin)
            log.info(f"[{coin}] Regime: {regimes.dominant_regime} "
                     f"(mom={regimes.momentum_score:.0f} trend={regimes.trend_score:.0f} "
                     f"mr={regimes.mean_rev_score:.0f} vol={regimes.volatility_score:.0f} "
                     f"abs={regimes.absorption_score:.0f} cat={regimes.catalyst_score:.0f})")
        except Exception as e:
            log.warning(f"[{coin}] Regime compute failed: {e}")
            regimes = None

        # Candlestick pattern analysis (pure OHLCV, no API)
        try:
            candle_patterns = compute_candlestick_patterns(df, coin)
        except Exception as e:
            log.warning(f"[{coin}] Candlestick patterns failed: {e}")
            candle_patterns = None

        # News sentiment (fetch, cached 10 min)
        news_signal = None
        if self.cfg.trading.use_news:
            try:
                news_signal = self._news_circuit.call(get_news_signal, coin)
            except Exception as e:
                log.warning(f"[{coin}] News fetch failed: {e}")

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
        instrument_type = self.cfg.trading.instrument_types.get(coin, "crypto")

        # Funding Rate / OI / CVD — order-flow intelligence
        # Only computed for crypto perps (not indexes — no funding on SP500 Yahoo data)
        funding_oi_signal = None
        if instrument_type == "crypto":
            try:
                funding_oi_signal = get_funding_oi_cvd(coin, df)
            except Exception as e:
                log.debug(f"[{coin}] FundingOI skipped: {e}")

        # Generate signal  (LONG / SHORT / FLAT)
        signal = self.strategy.generate_signal(
            tech, advanced, sentiment, current_pos, regimes,
            news_signal=news_signal,
            candle_patterns=candle_patterns,
            memory_adjustment=memory_adj,
            instrument_type=instrument_type,
            funding_oi_signal=funding_oi_signal,
        )

        log.info(
            f"[{coin}] Signal={signal.action} score={signal.score:.1f} "
            f"confidence={signal.confidence}"
        )

        # Store signal for dashboard + memory (enriched with new intelligence)
        self._last_signals[coin] = {
            "action":          signal.action,
            "score":           signal.score,
            "confidence":      signal.confidence,
            "price":           signal.price,
            "reason":          signal.reason,
            "flat_reason":     signal.flat_reason,
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
            # Memory / learning
            "memory_adj":      memory_adj,
            "memory_cooldown": self._memory._cooldown.get(coin, 0),
            # Regime context — stored for RL record_trade on close
            "market_regime":   _msb_struct,
            "dominant_regime": _dom_regime,
            # Funding / OI / CVD
            "funding_rate":    funding_oi_signal.funding_rate      if funding_oi_signal and funding_oi_signal.valid else 0.0,
            "funding_label":   funding_oi_signal.funding_label     if funding_oi_signal and funding_oi_signal.valid else "N/A",
            "oi_change_pct":   funding_oi_signal.oi_change_pct     if funding_oi_signal and funding_oi_signal.valid else 0.0,
            "cvd_divergence":  funding_oi_signal.cvd_divergence    if funding_oi_signal and funding_oi_signal.valid else "NONE",
            "foc_score":       funding_oi_signal.composite_score   if funding_oi_signal and funding_oi_signal.valid else 50.0,
        }

        if signal.action == "FLAT":
            # ── FLAT-while-positioned: close position if conviction is gone ──
            # If the market has turned sideways while we're holding, and the
            # signal stays FLAT for N consecutive cycles, the original thesis
            # no longer holds — exit cleanly rather than waiting for SL.
            if current_pos:
                flat_count = self._flat_streak.get(coin, 0) + 1
                self._flat_streak[coin] = flat_count
                max_flat = getattr(self.cfg.trading, 'max_flat_cycles_with_position', 3)
                if flat_count >= max_flat:
                    log.info(
                        f"[{coin}] 🏳️  Conviction gone: FLAT for {flat_count} cycles "
                        f"while {current_pos} open — closing (no thesis = no trade)"
                    )
                    self._close_position(coin, "conviction_lost", signal.price)
                    self._flat_streak.pop(coin, None)
                    self._signal_streak.pop(coin, None)
                else:
                    log.info(
                        f"[{coin}] 🏳️  FLAT signal ({flat_count}/{max_flat} cycles) "
                        f"— holding {current_pos} but watching conviction"
                    )
            else:
                self._flat_streak.pop(coin, None)
            return

        # ── Multi-timeframe confirmation ────────────────────────
        mtf = None
        if self.cfg.trading.use_mtf:
            mtf = self._check_mtf_safe(coin, signal)
            if mtf is False:
                return

        # ── News extreme-event gate (secondary safety net) ───────
        # Note: news is already factored into the signal score above.
        # This gate only blocks if news is catastrophically against the signal.
        if self.cfg.trading.use_news and news_signal:
            if not self._check_news_extreme(coin, signal, news_signal):
                return

        # ── Chart confirmation on borderline signals ────────────
        if self.cfg.trading.use_chart_confirmation:
            chart_verdict = self._get_chart_confirmation(coin, signal.score)
            if chart_verdict and chart_verdict.valid:
                if chart_verdict.action == "WAIT":
                    log.info(f"[{coin}] 👁️  Chart analyst says WAIT "
                             f"({chart_verdict.confidence}) — skipping this cycle. "
                             f"Reason: {chart_verdict.reasoning[:80]}")
                    return
                elif chart_verdict.action != signal.action:
                    log.info(
                        f"[{coin}] 👁️  Chart analyst disagrees: "
                        f"indicators={signal.action} chart={chart_verdict.action} "
                        f"— skipping. Reason: {chart_verdict.reasoning[:80]}"
                    )
                    return
                else:
                    log.info(
                        f"[{coin}] 👁️  Chart analyst CONFIRMS {signal.action} "
                        f"({chart_verdict.confidence}) — proceeding."
                    )

        # ── Loss-based circuit breaker check ────────────────
        if self.risk.is_trading_halted():
            log.info(f"[{coin}] ⏸️  Skipping — loss circuit breaker active")
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
                else:
                    min_hold = self.cfg.trading.min_hold_minutes
                if hold_minutes < min_hold:
                    log.info(
                        f"[{coin}] ⏳ Anti-whipsaw: position held only "
                        f"{hold_minutes:.0f}m (min={min_hold:.0f}m) — "
                        f"blocking reversal {current_pos}→{signal.action}"
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
                    return
            elif signal.action == "SHORT":
                required_score = self.cfg.trading.signal_short_threshold - reversal_boost
                if signal.score > required_score:
                    log.info(
                        f"[{coin}] ⚡ Reversal blocked: LONG→SHORT needs score "
                        f"≤{required_score:.0f} but got {signal.score:.1f}"
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
                log.info(
                    f"[{coin}] 🔁 Signal streak: {signal.action} confirmed "
                    f"{streak['count']}/{required_streak} cycles — waiting for confirmation"
                )
                return
            else:
                log.info(
                    f"[{coin}] ✅ Signal streak reached {streak['count']}/{required_streak} "
                    f"— proceeding with {signal.action} entry"
                )
                self._signal_streak.pop(coin, None)  # reset after entry

        # ── Pull RL stats to inform position sizing ────────────────────────
        rl_stats        = self._memory.get_stats().get(coin, {})
        rl_win_rate     = rl_stats.get("win_rate", 50.0) if rl_stats.get("total", 0) >= 3 else 50.0
        rl_pat_boost    = self._memory.get_pattern_boost(coin, signal.action, signal.score)

        # Risk-check & size the order (conviction + RL win-rate aware)
        order = self.risk.compute_order(
            coin              = coin,
            direction         = signal.action,
            signal_score      = signal.score,
            current_price     = signal.price,
            stop_loss_price   = signal.stop_loss_price,
            take_profit_price = signal.take_profit_price,
            portfolio_usd     = portfolio_usd,
            rl_win_rate       = rl_win_rate,
            rl_pattern_boost  = rl_pat_boost,
        )

        if not order.approved:
            log.info(f"[{coin}] Rejected: {order.rejection_reason}")
            return

        self._execute_order(coin, signal, order)

    # ── Execution ─────────────────────────────────────────────

    def _execute_order(self, coin, signal, order):
        for ex in self.exchanges:
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
                self.risk.record_open(order, exchange=ex.name)
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
                self.notifier.trade_opened(
                    coin     = coin,
                    direction= signal.action,
                    price    = fill_price,
                    size_usd = order.size_usd,
                    sl       = order.stop_loss,
                    tp       = order.take_profit,
                    score    = signal.score,
                    exchange = ex.name,
                )
                log.info(
                    f"[{coin}] ✅ {signal.action} opened on {ex.name}: "
                    f"{order.size_coin:.6f} @ ${fill_price:.2f} "
                    f"| SL ${order.stop_loss:.2f} TP ${order.take_profit:.2f}"
                )

                # Verify position exists on exchange
                time.sleep(1)
                try:
                    account_state = ex.get_account_state()
                    position_found = any(pos.get('coin') == coin for pos in account_state.positions)
                    if not position_found:
                        log.critical(f"[{coin}] CRITICAL: Order reported success but position "
                                    f"not found on exchange after verification!")
                        self.notifier.error_alert(
                            f"CRITICAL: {coin} {signal.action} order verified but position missing on {ex.name}"
                        )
                    else:
                        log.info(f"[{coin}] Position verified on {ex.name}")
                except Exception as e:
                    log.critical(f"[{coin}] CRITICAL: Could not verify position on {ex.name}: {e}")
                    self.notifier.error_alert(
                        f"CRITICAL: Could not verify {coin} position on {ex.name}: {e}"
                    )
                return
            elif not result:
                log.error(f"[{coin}] ❌ No result returned from {ex.name}")
                self.notifier.error_alert(
                    f"{signal.action} failed on {ex.name} for {coin}: No result returned"
                )

    def _place_limit_order(self, coin: str, direction: str, limit_price: float,
                           size_usd: float, sl: float, tp: float,
                           score: float, reason: str = "re_entry"):
        """Place a limit order on the first available exchange and register it."""
        for ex in self.exchanges:
            ex.set_leverage(coin, self.cfg.trading.leverage)
            size_coin = size_usd / limit_price if limit_price > 0 else 0

            if direction == "LONG":
                result = ex.limit_buy(coin, size_coin, limit_price)
            else:
                result = ex.limit_sell(coin, size_coin, limit_price)

            if result.success:
                pending = PendingOrder(
                    coin              = coin,
                    direction         = direction,
                    limit_price       = limit_price,
                    size_coin         = size_coin,
                    size_usd          = size_usd,
                    stop_loss         = sl,
                    take_profit       = tp,
                    signal_score      = score,
                    exchange          = ex.name,
                    exchange_order_id = result.order_id,
                    reason            = reason,
                )
                self.order_mgr.register_limit_order(pending)
                log.info(
                    f"[{coin}] 📋 Limit {direction} placed @ ${limit_price:.2f} "
                    f"SL=${sl:.2f} TP=${tp:.2f} (reason={reason})"
                )
                return
            else:
                log.error(f"[{coin}] Limit order failed on {ex.name}: {result.error}")

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

            self._place_limit_order(coin, direction, price, size_usd,
                                    sl, tp, score, reason)

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
                self.risk.record_open(order, exchange=ex.name)
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
                )
                self.order_mgr.mark_filled(coin, fill_price)
            elif status.cancelled:
                log.info(f"[{coin}] Limit order cancelled by exchange")
                self.order_mgr.mark_cancelled(coin)

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
                log.info(f"[{coin}] TP hit — scheduling re-entry watch "
                         f"(fib retracement from ${info['entry_price']:.2f} → ${info['tp_price']:.2f})")
                self.order_mgr.schedule_reentry(
                    coin         = coin,
                    direction    = info["direction"],
                    entry_price  = info["entry_price"],
                    tp_price     = info["tp_price"],
                    size_usd     = min(info["size_usd"], self.cfg.trading.max_trade_usd),
                    signal_score = 65.0,   # assume medium-high conviction for re-entry
                )

    def _close_position(self, coin: str, reason: str, price: float):
        pos   = self.risk.positions.get(coin)
        entry = pos.entry_price if pos else price

        exit_price = price
        exchanges = self.exchanges
        if pos and pos.exchange:
            preferred = self._get_exchange_by_name(pos.exchange)
            exchanges = ([preferred] if preferred else []) + [
                ex for ex in self.exchanges if not preferred or ex.name != preferred.name
            ]

        close_successful = False
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

                # Verify position is actually gone
                time.sleep(1)
                try:
                    account_state = ex.get_account_state()
                    position_still_open = any(pos.get('coin') == coin for pos in account_state.positions)
                    if position_still_open:
                        log.critical(f"[{coin}] CRITICAL: Close order succeeded but position "
                                    f"still exists on exchange after verification!")
                        self.notifier.error_alert(
                            f"CRITICAL: {coin} close order verified but position still open on {ex.name}"
                        )
                    else:
                        log.info(f"[{coin}] Close verified on {ex.name}")
                except Exception as e:
                    log.critical(f"[{coin}] CRITICAL: Could not verify close on {ex.name}: {e}")
                    self.notifier.error_alert(
                        f"CRITICAL: Could not verify {coin} close on {ex.name}: {e}"
                    )
                break

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
        trade_logger.log_close(coin, exit_price=exit_price, exit_reason=reason)

        # ── Record outcome to trade memory for self-learning ─────────────────
        if pos:
            hold_minutes = (time.time() - pos.opened_at) / 60.0 if pos.opened_at else 0.0
            last_sig     = self._last_signals.get(coin, {})
            trend_ctx    = last_sig.get("mtf_bias", "FLAT")
            # Pull regime context so RL can learn which regimes are profitable
            market_regime   = last_sig.get("market_regime",   "RANGING")
            dominant_regime = last_sig.get("dominant_regime", "MIXED")
            self._memory.record_trade(
                coin             = coin,
                direction        = pos.direction,
                signal_score     = last_sig.get("score", 50.0),
                entry_price      = entry,
                exit_price       = exit_price,
                exit_reason      = reason,
                hold_minutes     = hold_minutes,
                trend_context    = trend_ctx,
                market_regime    = market_regime,
                dominant_regime  = dominant_regime,
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
        for coin in self.cfg.trading.coins:
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
            if signal.action == "LONG" and not mtf.allow_long:
                log.info(f"[{coin}] 🕐 MTF blocks LONG — {mtf.reason}")
                return False
            if signal.action == "SHORT" and not mtf.allow_short:
                log.info(f"[{coin}] 🕐 MTF blocks SHORT — {mtf.reason}")
                return False
            log.info(f"[{coin}] MTF combined={mtf.combined_bias} adj={mtf.score_adjustment:+.0f}")
            return True
        except CircuitBreakerError:
            log.warning(f"[{coin}] MTF circuit open - allowing signal")
            return True
        except Exception as e:
            log.warning(f"[{coin}] MTF failed: {e}")
            return True

    def _check_news_safe(self, coin: str, signal) -> bool:
        """Legacy wrapper — kept for compatibility. Use _check_news_extreme instead."""
        try:
            news = self._news_circuit.call(get_news_signal, coin)
            return self._check_news_extreme(coin, signal, news)
        except CircuitBreakerError:
            log.warning(f"[{coin}] News circuit open - allowing signal")
            return True
        except Exception as e:
            log.warning(f"[{coin}] News signal failed: {e}")
            return True

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

    # ── Dashboard state writer ────────────────────────────────

    def _write_state(self, portfolio_usd: float, sentiment: dict):
        """Write current agent state to state.json for the dashboard."""
        import json, os
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
            "config": {
                "long_threshold":        self.cfg.trading.signal_long_threshold,
                "short_threshold":       self.cfg.trading.signal_short_threshold,
                "min_hold_minutes":      self.cfg.trading.min_hold_minutes,
                "index_min_hold_minutes": self.cfg.trading.index_min_hold_minutes,
                "reversal_boost":        self.cfg.trading.reversal_threshold_boost,
                "coins":                 self.cfg.trading.coins,
                "instrument_types":      self.cfg.trading.instrument_types,
            },
        }

        state_path = STATE_JSON
        try:
            state_path.write_text(json.dumps(state, indent=2))
        except Exception as e:
            log.debug(f"state.json write failed: {e}")

        # Push to remote dashboard (Netlify) if configured
        remote_url = os.environ.get("DASHBOARD_URL", "")
        if remote_url:
            try:
                import urllib.request
                trades_data = []
                log_path = TRADES_CSV
                if log_path.exists():
                    import csv as csv_mod
                    with open(log_path, newline="") as f:
                        trades_data = list(csv_mod.DictReader(f))
                payload = json.dumps({"state": state, "trades": trades_data}).encode()
                req = urllib.request.Request(
                    remote_url.rstrip("/") + "/api/push",
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-Token": os.environ.get("DASHBOARD_TOKEN", ""),
                    },
                    method="POST"
                )
                import ssl
                ctx = ssl.create_default_context()
                try:
                    import certifi
                    ctx.load_verify_locations(certifi.where())
                except ImportError:
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                resp = urllib.request.urlopen(req, timeout=15, context=ctx)

                # Check for remote kill signal
                try:
                    kill_url = remote_url.rstrip("/") + "/api/state"
                    kill_req = urllib.request.Request(kill_url, method="GET")
                    kill_resp = urllib.request.urlopen(kill_req, timeout=10, context=ctx)
                    kill_data = json.loads(kill_resp.read().decode())
                    kill_sig = kill_data.get("state", {}).get("_kill_signal")
                    if kill_sig and kill_sig.get("active"):
                        kill_reason = kill_sig.get("reason", "remote dashboard kill")
                        log.critical(f"🚨 REMOTE KILL SIGNAL: {kill_reason}")
                        kill_file = KILL_FILE
                        kill_file.write_text(kill_reason)
                except Exception:
                    pass  # kill signal check is best-effort
            except Exception as e:
                log.debug(f"Remote dashboard push failed: {e}")

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
