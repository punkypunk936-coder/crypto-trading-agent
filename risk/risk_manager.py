"""
risk/risk_manager.py
Position sizing, SL/TP/trailing-stop tracking, and portfolio exposure control.

Perps-native: positions are either LONG, SHORT, or FLAT (no "holds").
SL: 10%  |  TP: 50%  |  Trailing stop: 12%
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List
from logger import get_logger

log = get_logger("risk")


@dataclass
class OpenPosition:
    coin: str
    direction: str          # "LONG" or "SHORT"
    entry_price: float
    size_usd: float
    size_coin: float
    stop_loss:  float
    take_profit: float
    trailing_stop_price: float = 0.0
    opened_at: float = field(default_factory=time.time)
    exchange: str = ""


@dataclass
class OrderRequest:
    coin: str
    direction: str          # "LONG" or "SHORT"
    size_usd: float
    size_coin: float
    price: float
    stop_loss: float
    take_profit: float
    leverage: int           = 1
    approved: bool          = True
    rejection_reason: str   = ""
    conviction_tier: str    = ""
    conviction_pct: float   = 0.0


class RiskManager:
    def __init__(self, trading_cfg):
        self.cfg = trading_cfg
        self.positions: Dict[str, OpenPosition] = {}
        self.trade_log: List[dict] = []
        self.daily_pnl_usd: float = 0.0
        self.daily_trades: int = 0
        self.last_trade_date: str = ""
        # ── IST daily P&L history (for dashboard daily tab) ──────────────────
        # Stored as {"YYYY-MM-DD": pnl_usd} keyed on IST date (UTC+5:30)
        self.daily_pnl_history: Dict[str, float] = {}
        # Set by record_close so agent.py can decide if re-entry should be scheduled
        self._last_close_was_tp: bool = False
        self._last_close_info: dict = {}

        # ── Loss-based circuit breaker ──────────────────────────
        self.consecutive_losses: int = 0
        self.max_consecutive_losses: int = 5       # halt after 5 losses in a row
        self.daily_loss_limit_usd: float = -500.0  # halt if daily P&L drops below -$500
        self.drawdown_halt_pct: float = 0.15       # halt if portfolio drops 15% from peak
        self.peak_portfolio_usd: float = 0.0       # tracks high-water mark
        self.trading_halted: bool = False
        self.halt_reason: str = ""
        self.halt_time: float = 0.0
        self.cooldown_seconds: int = 3600          # 1-hour cooldown before auto-resume

    # ── Portfolio state ────────────────────────────────────────

    def total_exposure_usd(self) -> float:
        return sum(p.size_usd for p in self.positions.values())

    def available_capital(self, portfolio_usd: float) -> float:
        used     = self.total_exposure_usd()
        max_dep  = portfolio_usd * self.cfg.max_total_exposure_pct
        return max(0.0, max_dep - used)

    def has_position(self, coin: str) -> bool:
        return coin in self.positions

    def position_direction(self, coin: str) -> Optional[str]:
        p = self.positions.get(coin)
        return p.direction if p else None

    # ── Order sizing ───────────────────────────────────────────

    def compute_order(
        self,
        coin: str,
        direction: str,          # "LONG" or "SHORT"
        signal_score: float,
        current_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        portfolio_usd: float,
        rl_win_rate: float = 50.0,    # 0-100 historical win rate for this coin/direction
        rl_pattern_boost: float = 0.0, # extra multiplier from RL pattern recognition
    ) -> OrderRequest:
        """
        Conviction-aware order sizing.
        Size scales continuously with signal strength AND RL track record:
          • Higher score  → larger position (more signals agree)
          • Better RL win rate on this coin/direction → further size boost
          • Known winning pattern → additional multiplier
        Returns approved=False if any rule is violated.
        """
        # ── Maximum size for this trade ───────────────────
        max_trade_usd = portfolio_usd * self.cfg.max_position_pct

        # ── Continuous conviction sizing (replaces hard tiers) ──────────
        # conviction: 0 = score right at threshold (e.g. 65), 35 = score 100/0
        # We map conviction linearly across the 10pt–35pt range.
        conviction = abs(signal_score - 50.0)  # 0-50 scale

        if conviction >= 40:          # score ≥90 or ≤10 — extreme conviction
            conviction_tier    = "EXTREME"
            confidence_factor  = 1.00
        elif conviction >= 30:        # score ≥80 or ≤20 — high
            conviction_tier    = "HIGH"
            confidence_factor  = 0.80
        elif conviction >= 20:        # score ≥70 or ≤30 — medium
            conviction_tier    = "MEDIUM"
            confidence_factor  = 0.60
        elif conviction >= 15:        # score ≥65 or ≤35 — base+
            conviction_tier    = "BASE+"
            confidence_factor  = 0.45
        else:                         # score 60-64 / 35-40 — minimum
            conviction_tier    = "BASE"
            confidence_factor  = 0.30

        # ── RL win-rate boost: double down on proven setups ──────────────
        # If RL shows this coin+direction has a strong track record, size up.
        # This is the "doubling down on what works" mechanism.
        rl_boost = 0.0
        if rl_win_rate >= 75:     # 75%+ win rate → +25% size
            rl_boost = 0.25
        elif rl_win_rate >= 65:   # 65%+ win rate → +15% size
            rl_boost = 0.15
        elif rl_win_rate >= 55:   # 55%+ win rate → +8% size
            rl_boost = 0.08
        elif rl_win_rate <= 30:   # bad history → -20% size (be cautious)
            rl_boost = -0.20
        elif rl_win_rate <= 40:   # below average → -10% size
            rl_boost = -0.10

        # RL pattern boost: further reward recognized winning patterns
        total_factor = min(confidence_factor * (1.0 + rl_boost + rl_pattern_boost), 1.10)

        size_usd = min(max_trade_usd * total_factor, self.cfg.max_trade_usd)

        # Cap at available capital
        avail    = self.available_capital(portfolio_usd)
        size_usd = min(size_usd, avail)

        # ── Minimum trade check ───────────────────────────
        if size_usd < self.cfg.min_trade_usd:
            return OrderRequest(
                coin=coin, direction=direction,
                size_usd=0, size_coin=0,
                price=current_price,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                approved=False,
                rejection_reason=(
                    f"Trade size ${size_usd:.2f} < minimum ${self.cfg.min_trade_usd} "
                    f"(portfolio=${portfolio_usd:.0f}, available=${avail:.0f})"
                ),
            )

        # ── Duplicate position check ──────────────────────
        existing = self.position_direction(coin)
        if existing == direction:
            return OrderRequest(
                coin=coin, direction=direction,
                size_usd=0, size_coin=0,
                price=current_price,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                approved=False,
                rejection_reason=f"Already {existing} on {coin} — no duplicate"
            )

        size_coin = size_usd / current_price if current_price > 0 else 0.0

        rl_boost_str = f" RL+{rl_boost*100:+.0f}% pat+{rl_pattern_boost*100:.0f}%" if (rl_boost or rl_pattern_boost) else ""
        log.info(
            f"[{coin}] ✅ Order approved: {direction} "
            f"${size_usd:.2f} ({size_coin:.6f} coins) "
            f"SL=${stop_loss_price:.2f} TP=${take_profit_price:.2f} "
            f"({self.cfg.stop_loss_pct*100:.0f}% SL / {self.cfg.take_profit_pct*100:.0f}% TP) "
            f"| Conviction: {conviction_tier} ({conviction:.0f}/50) "
            f"→ {total_factor*100:.0f}% size{rl_boost_str}"
        )

        return OrderRequest(
            coin            = coin,
            direction       = direction,
            size_usd        = size_usd,
            size_coin       = size_coin,
            price           = current_price,
            stop_loss       = stop_loss_price,
            take_profit     = take_profit_price,
            leverage        = self.cfg.leverage,
            approved        = True,
            conviction_tier = conviction_tier,
            conviction_pct  = confidence_factor * 100,
        )

    # ── Position lifecycle ─────────────────────────────────────

    def record_open(self, order: OrderRequest, exchange: str = ""):
        # Calculate initial trailing stop price
        if order.direction == "LONG":
            trail_price = order.price * (1 - self.cfg.trailing_stop_pct)
        else:
            trail_price = order.price * (1 + self.cfg.trailing_stop_pct)

        pos = OpenPosition(
            coin               = order.coin,
            direction          = order.direction,
            entry_price        = order.price,
            size_usd           = order.size_usd,
            size_coin          = order.size_coin,
            stop_loss          = order.stop_loss,
            take_profit        = order.take_profit,
            trailing_stop_price= trail_price,
            exchange           = exchange,
        )
        self.positions[order.coin] = pos
        log.info(
            f"[{order.coin}] 📈 Position OPENED: {order.direction} "
            f"entry=${order.price:.2f} "
            f"SL=${order.stop_loss:.2f} ({self.cfg.stop_loss_pct*100:.0f}%) "
            f"TP=${order.take_profit:.2f} ({self.cfg.take_profit_pct*100:.0f}%)"
        )

    def restore_position(self, position: OpenPosition):
        """Restore a position into the in-memory risk state after restart."""
        self.positions[position.coin] = position
        log.info(
            f"[{position.coin}] Restored position: {position.direction} "
            f"entry=${position.entry_price:.2f} size=${position.size_usd:.2f} "
            f"exchange={position.exchange or 'unknown'}"
        )

    def replace_positions(self, positions: Dict[str, OpenPosition]):
        """Replace all tracked positions with exchange-reconciled truth."""
        self.positions = dict(positions)
        if positions:
            log.info(f"Reconciled {len(positions)} live position(s) from exchange state")
        else:
            log.info("Reconciled exchange state: no live positions")

    def record_close(self, coin: str, exit_price: float, reason: str = "signal",
                     was_take_profit: bool = False):
        pos = self.positions.pop(coin, None)
        if not pos:
            return
        # Flag so the agent can schedule a re-entry watch
        self._last_close_was_tp = was_take_profit
        self._last_close_info = {
            "coin":        coin,
            "direction":   pos.direction,
            "entry_price": pos.entry_price,
            "tp_price":    pos.take_profit,
            "size_usd":    pos.size_usd,
        }
        if pos.direction == "LONG":
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        pnl_usd = pnl_pct * pos.size_usd
        self._update_daily_stats(pnl_usd)
        result  = "✅ WIN" if pnl_usd > 0 else "❌ LOSS"
        log.info(
            f"[{coin}] {result} closed: {pos.direction} "
            f"entry=${pos.entry_price:.2f} → exit=${exit_price:.2f} "
            f"PnL={pnl_pct*100:+.2f}% (${pnl_usd:+.2f}) | reason={reason}"
        )
        self.trade_log.append({
            "coin":      coin,
            "direction": pos.direction,
            "entry":     pos.entry_price,
            "exit":      exit_price,
            "size_usd":  pos.size_usd,
            "pnl_pct":   pnl_pct,
            "pnl_usd":   pnl_usd,
            "reason":    reason,
            "closed_at": time.time(),
        })

    def _update_daily_stats(self, pnl_usd: float):
        # Use IST (UTC+5:30) so "today" resets at midnight India time
        from datetime import timezone, timedelta
        ist_tz  = timezone(timedelta(hours=5, minutes=30))
        today   = datetime.now(ist_tz).strftime("%Y-%m-%d")
        utc_day = datetime.utcnow().strftime("%Y-%m-%d")

        if self.last_trade_date != utc_day:
            self.daily_pnl_usd = 0.0
            self.daily_trades  = 0
            self.last_trade_date = utc_day

        self.daily_pnl_usd += pnl_usd
        self.daily_trades  += 1

        # ── IST daily history ──────────────────────────────
        # Keep last 30 days of daily P&L keyed on IST date
        self.daily_pnl_history[today] = round(
            self.daily_pnl_history.get(today, 0.0) + pnl_usd, 2
        )
        # Prune to last 30 days
        if len(self.daily_pnl_history) > 30:
            oldest = sorted(self.daily_pnl_history.keys())[0]
            del self.daily_pnl_history[oldest]

        # ── Update consecutive loss counter ─────────────────
        if pnl_usd > 0:
            self.consecutive_losses = 0          # reset on any win
        else:
            self.consecutive_losses += 1

        # ── Check if we should halt trading ─────────────────
        if self.consecutive_losses >= self.max_consecutive_losses:
            self._halt_trading(
                f"🛑 {self.consecutive_losses} consecutive losses — "
                f"cooling down for safety"
            )
        elif self.daily_pnl_usd <= self.daily_loss_limit_usd:
            self._halt_trading(
                f"🛑 Daily loss limit hit: ${self.daily_pnl_usd:+.2f} "
                f"(limit: ${self.daily_loss_limit_usd:.2f})"
            )

    # ── Loss-based circuit breaker ───────────────────────────

    def _halt_trading(self, reason: str):
        if not self.trading_halted:
            self.trading_halted = True
            self.halt_reason = reason
            self.halt_time = time.time()
            log.warning(reason)
            log.warning(
                f"⏸️  New trades PAUSED for {self.cooldown_seconds // 60} minutes. "
                f"Open positions will still be monitored for SL/TP exits."
            )

    def update_peak_portfolio(self, portfolio_usd: float):
        """Call once per cycle with current portfolio value to track drawdown."""
        if portfolio_usd > self.peak_portfolio_usd:
            self.peak_portfolio_usd = portfolio_usd

        if self.peak_portfolio_usd > 0:
            drawdown = (self.peak_portfolio_usd - portfolio_usd) / self.peak_portfolio_usd
            if drawdown >= self.drawdown_halt_pct:
                self._halt_trading(
                    f"🛑 Portfolio drawdown {drawdown*100:.1f}% from peak "
                    f"${self.peak_portfolio_usd:,.0f} → ${portfolio_usd:,.0f} "
                    f"(limit: {self.drawdown_halt_pct*100:.0f}%)"
                )

    def is_trading_halted(self) -> bool:
        """
        Check if the loss-based circuit breaker is active.
        Auto-resumes after cooldown period expires.
        """
        if not self.trading_halted:
            return False

        # Auto-resume after cooldown
        elapsed = time.time() - self.halt_time
        if elapsed >= self.cooldown_seconds:
            log.info(
                f"✅ Cooldown expired ({self.cooldown_seconds // 60} min). "
                f"Resuming trading. Consecutive losses reset."
            )
            self.trading_halted = False
            self.halt_reason = ""
            self.consecutive_losses = 0
            return False

        remaining = int(self.cooldown_seconds - elapsed)
        log.info(
            f"⏸️  Trading halted: {self.halt_reason} "
            f"({remaining}s remaining in cooldown)"
        )
        return True

    def force_resume(self):
        """Manual override to resume trading early (for the operator)."""
        if self.trading_halted:
            log.info("🔓 Manual resume: trading circuit breaker cleared.")
            self.trading_halted = False
            self.halt_reason = ""
            self.consecutive_losses = 0

    def circuit_breaker_status(self) -> dict:
        """Return current circuit breaker state for the dashboard."""
        return {
            "halted": self.trading_halted,
            "reason": self.halt_reason,
            "consecutive_losses": self.consecutive_losses,
            "max_consecutive_losses": self.max_consecutive_losses,
            "daily_pnl_usd": round(self.daily_pnl_usd, 2),
            "daily_loss_limit_usd": self.daily_loss_limit_usd,
            "peak_portfolio_usd": round(self.peak_portfolio_usd, 2),
            "drawdown_halt_pct": self.drawdown_halt_pct,
            "cooldown_seconds": self.cooldown_seconds,
            "time_remaining": max(0, int(
                self.cooldown_seconds - (time.time() - self.halt_time)
            )) if self.trading_halted else 0,
        }

    # ── SL / TP / trailing stop monitoring ────────────────────

    def check_exits(self, current_prices: Dict[str, float]) -> List[dict]:
        """
        Scan all open positions against current prices.
        Returns a list of exit instructions for positions that hit SL/TP/trail.
        """
        exits = []
        for coin, pos in list(self.positions.items()):
            price = current_prices.get(coin)
            if not price:
                continue

            should_close = False
            reason       = ""

            if pos.direction == "LONG":
                if price <= pos.stop_loss:
                    should_close, reason = True, "stop_loss"
                elif price >= pos.take_profit:
                    should_close, reason = True, "take_profit"
                elif self.cfg.trailing_stop_enabled:
                    # Ratchet the trailing stop up as price rises
                    new_trail = price * (1 - self.cfg.trailing_stop_pct)
                    if new_trail > pos.trailing_stop_price:
                        log.debug(f"[{coin}] Trailing stop ratcheted: "
                                  f"${pos.trailing_stop_price:.2f} → ${new_trail:.2f}")
                        pos.trailing_stop_price = new_trail
                    if price <= pos.trailing_stop_price:
                        should_close, reason = True, "trailing_stop"

            else:   # SHORT
                if price >= pos.stop_loss:
                    should_close, reason = True, "stop_loss"
                elif price <= pos.take_profit:
                    should_close, reason = True, "take_profit"
                elif self.cfg.trailing_stop_enabled:
                    # Ratchet trailing stop down as price falls
                    new_trail = price * (1 + self.cfg.trailing_stop_pct)
                    if new_trail < pos.trailing_stop_price or pos.trailing_stop_price == 0:
                        log.debug(f"[{coin}] Trailing stop ratcheted: "
                                  f"${pos.trailing_stop_price:.2f} → ${new_trail:.2f}")
                        pos.trailing_stop_price = new_trail
                    if price >= pos.trailing_stop_price and pos.trailing_stop_price > 0:
                        should_close, reason = True, "trailing_stop"

            if should_close:
                exits.append({
                    "coin":            coin,
                    "direction":       pos.direction,
                    "reason":          reason,
                    "price":           price,
                    "exchange":        pos.exchange,
                    "was_take_profit": reason == "take_profit",
                    "entry_price":     pos.entry_price,
                    "tp_price":        pos.take_profit,
                    "size_usd":        pos.size_usd,
                })

        return exits

    # ── Performance summary ───────────────────────────────────

    def summary(self, portfolio_usd: float) -> str:
        lines = [
            f"Portfolio: ${portfolio_usd:,.2f}  |  "
            f"Deployed: ${self.total_exposure_usd():,.2f}  |  "
            f"Available: ${self.available_capital(portfolio_usd):,.2f}"
        ]
        if self.positions:
            lines.append("Open positions:")
            for coin, p in self.positions.items():
                lines.append(
                    f"  {coin:5s} {p.direction:6s} "
                    f"entry=${p.entry_price:.2f} "
                    f"SL=${p.stop_loss:.2f} TP=${p.take_profit:.2f} "
                    f"trail=${p.trailing_stop_price:.2f}"
                )
        else:
            lines.append("No open positions (FLAT)")

        if self.trade_log:
            wins  = sum(1 for t in self.trade_log if t["pnl_usd"] > 0)
            total = len(self.trade_log)
            total_pnl = sum(t["pnl_usd"] for t in self.trade_log)
            lines.append(
                f"Closed: {total} trades  |  Win rate: {wins}/{total} "
                f"({wins/total*100:.1f}%)  |  Total PnL: ${total_pnl:+,.2f}"
            )
        return "\n".join(lines)
