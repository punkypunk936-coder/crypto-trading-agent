"""
risk/risk_manager.py
Position sizing, SL/TP/trailing-stop tracking, and portfolio exposure control.

Perps-native: positions are either LONG, SHORT, or FLAT (no "holds").
SL: 10%  |  TP: 50%  |  Trailing stop: 12%
"""

import time
import math
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
    leverage: float = 1.0
    margin_usd: float = 0.0
    metadata: dict = field(default_factory=dict)


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
    margin_usd: float       = 0.0
    leverage_note: str      = ""
    approved: bool          = True
    rejection_reason: str   = ""
    conviction_tier: str    = ""
    conviction_pct: float   = 0.0
    is_scale_in: bool       = False


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

    def _cfg_float(self, name: str, default: float) -> float:
        try:
            return float(getattr(self.cfg, name, default))
        except Exception:
            return float(default)

    def _cfg_int(self, name: str, default: int) -> int:
        try:
            return int(getattr(self.cfg, name, default))
        except Exception:
            return int(default)

    def select_adaptive_leverage(
        self,
        *,
        conviction_tier: str,
        conviction: float,
        probability: float,
        expectancy_r: float,
        uncertainty: float,
        rl_win_rate: float,
        rl_pattern_boost: float,
        starter_multiplier: float = 1.0,
        event_starter: bool = False,
        scale_in: bool = False,
        scalp: bool = False,
    ) -> tuple[int, str]:
        """Choose per-order leverage from conviction, learned edge, and uncertainty."""
        configured = max(1, self._cfg_int("leverage", 1))
        min_lev = max(1, self._cfg_int("min_leverage", 1))
        max_lev = max(min_lev, self._cfg_int("max_leverage", configured))
        if not bool(getattr(self.cfg, "adaptive_leverage_enabled", False)):
            fixed = max(min_lev, min(configured, max_lev))
            return fixed, f"fixed {fixed}x"

        base_lev = max(min_lev, min(self._cfg_int("base_leverage", configured), max_lev))
        tier = str(conviction_tier or "").upper()
        notes: List[str] = []

        if tier.startswith("EXTREME"):
            lev = base_lev + 2
            notes.append("extreme conviction")
        elif tier.startswith("HIGH"):
            lev = base_lev + 1
            notes.append("high conviction")
        elif tier.startswith("MEDIUM"):
            lev = base_lev
            notes.append("medium conviction")
        elif tier.startswith("BASE+"):
            lev = max(min_lev, base_lev)
            notes.append("base+ conviction")
        else:
            lev = min_lev
            notes.append("base conviction")

        if probability >= 0.68 and expectancy_r >= 0.35 and uncertainty <= 0.34:
            lev += 2
            notes.append("A+ forecast")
        elif probability >= 0.62 and expectancy_r >= 0.22 and uncertainty <= 0.42:
            lev += 1
            notes.append("forecast edge")

        if rl_win_rate >= 72.0:
            lev += 1
            notes.append("RL validated")
        elif rl_win_rate <= 30.0:
            lev -= 2
            notes.append("poor RL history")
        elif rl_win_rate <= 40.0:
            lev -= 1
            notes.append("weak RL history")

        if rl_pattern_boost >= 0.08 and rl_win_rate >= 58.0:
            lev += 1
            notes.append("winning pattern")
        elif rl_pattern_boost < 0:
            lev -= 1
            notes.append("pattern penalty")

        max_allowed = max_lev
        if scale_in:
            max_allowed = min(max_allowed, max(1, self._cfg_int("scale_in_max_leverage", 4)))
            notes.append("scale-in cap")
        if scalp:
            max_allowed = min(max_allowed, max(1, self._cfg_int("scalp_max_leverage", 3)))
            notes.append("scalp cap")
        if event_starter:
            max_allowed = min(max_allowed, max(1, self._cfg_int("event_starter_max_leverage", 3)))
            notes.append("event-risk cap")
        elif starter_multiplier < 0.999:
            max_allowed = min(max_allowed, max(1, self._cfg_int("starter_max_leverage", 2)))
            notes.append("starter cap")

        hard_cap = max_allowed
        leverage_floor = min_lev
        if tier.startswith("EXTREME"):
            leverage_floor = max(leverage_floor, self._cfg_int("extreme_conviction_min_leverage", 4))
        elif tier.startswith("HIGH"):
            leverage_floor = max(leverage_floor, self._cfg_int("high_conviction_min_leverage", 3))
        if event_starter:
            leverage_floor = max(leverage_floor, self._cfg_int("event_starter_min_leverage", 2))
        elif starter_multiplier < 0.999:
            leverage_floor = max(leverage_floor, self._cfg_int("starter_min_leverage", 1))
        if scalp:
            leverage_floor = max(leverage_floor, self._cfg_int("scalp_min_leverage", 1))
        leverage_floor = min(leverage_floor, hard_cap)

        if uncertainty >= 0.60 or probability < 0.53:
            max_allowed = min(max_allowed, 1)
            notes.append("low clarity")
        elif uncertainty >= 0.50 or probability < 0.58 or expectancy_r < 0.10:
            max_allowed = min(max_allowed, 2)
            notes.append("clarity cap")

        floor_can_override_clarity = bool(
            getattr(self.cfg, "conviction_leverage_floor_overrides_clarity_cap", True)
        )
        floor_is_earned = (
            (tier.startswith(("HIGH", "EXTREME")) and probability >= 0.56 and uncertainty <= 0.58 and expectancy_r >= 0.08)
            or (event_starter and probability >= 0.54 and uncertainty <= 0.66)
            or (scalp and probability >= 0.52 and uncertainty <= 0.66)
            or (starter_multiplier < 0.999 and probability >= 0.54 and uncertainty <= 0.62 and expectancy_r >= 0.06)
        )
        if leverage_floor > max_allowed and floor_can_override_clarity and floor_is_earned:
            max_allowed = leverage_floor
            notes.append("conviction floor")

        effective_floor = leverage_floor if leverage_floor <= max_allowed else min_lev
        lev = max(effective_floor, min(int(round(lev)), max_allowed))
        note = f"{lev}x: " + ", ".join(notes[:4])
        return lev, note

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
        expectancy_score: float | None = None,
        win_probability: float | None = None,
        expected_r: float | None = None,
        uncertainty: float | None = None,
        thesis_conviction: float | None = None,
        sizing_multiplier: float | None = None,
        event_starter: bool = False,
        scalp: bool = False,
    ) -> OrderRequest:
        """
        Conviction-aware order sizing.
        Size scales continuously with signal strength AND RL track record:
          • Higher score  → larger position (more signals agree)
          • Better RL win rate on this coin/direction → further size boost
          • Known winning pattern → additional multiplier
          • Extreme conviction without learned edge gets dampened to avoid euphoria
        Returns approved=False if any rule is violated.
        """
        # ── Maximum margin budget for this trade ───────────────────
        max_margin_usd = min(
            portfolio_usd * self.cfg.max_position_pct,
            self._cfg_float("max_trade_usd", portfolio_usd * self.cfg.max_position_pct),
        )

        # ── Continuous conviction sizing ────────────────────────────────
        # conviction: score distance from neutral (50).
        # We normalise from the minimum actionable threshold outward so
        # capital scales smoothly instead of jumping across a few hard tiers.
        expectancy_anchor = float(expectancy_score if expectancy_score is not None else signal_score)
        conviction_anchor = float(thesis_conviction if thesis_conviction is not None else expectancy_anchor)
        conviction = abs(conviction_anchor - 50.0)  # 0-50 scale
        entry_conviction_floor = min(
            abs(self.cfg.signal_long_threshold - 50.0),
            abs(self.cfg.signal_short_threshold - 50.0),
        )
        conviction_span = max(1.0, 50.0 - entry_conviction_floor)
        conviction_norm = max(0.0, min(1.0, (conviction - entry_conviction_floor) / conviction_span))
        base_floor = max(0.10, min(0.60, float(getattr(self.cfg, "conviction_size_floor", 0.30))))
        curve = max(0.25, float(getattr(self.cfg, "conviction_size_curve", 0.85)))
        confidence_factor = base_floor + (1.0 - base_floor) * math.pow(conviction_norm, curve)

        probability = float(win_probability if win_probability is not None else 0.50)
        probability = max(0.05, min(0.95, probability))
        expectancy_r = float(expected_r if expected_r is not None else 0.0)
        uncertainty_level = float(uncertainty if uncertainty is not None else 0.35)
        uncertainty_level = max(0.0, min(1.0, uncertainty_level))
        probability_factor = 0.85 + max(-0.15, min(0.22, (probability - 0.50) * 1.30))
        expectancy_factor = 0.90 + max(-0.20, min(0.28, expectancy_r * 0.32))
        uncertainty_factor = 1.0 - min(0.35, uncertainty_level * 0.42)
        confidence_factor *= probability_factor * expectancy_factor * uncertainty_factor

        if conviction >= 40:          # score ≥90 or ≤10 — extreme conviction
            conviction_tier    = "EXTREME"
        elif conviction >= 30:        # score ≥80 or ≤20 — high
            conviction_tier    = "HIGH"
        elif conviction >= 20:        # score ≥70 or ≤30 — medium
            conviction_tier    = "MEDIUM"
        elif conviction >= 15:        # score ≥65 or ≤35 — base+
            conviction_tier    = "BASE+"
        else:                         # score 60-64 / 35-40 — minimum
            conviction_tier    = "BASE"

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

        # ── Anti-euphoria dampener ───────────────────────────────────────
        # Very strong raw conviction is not enough on its own; if the RL layer
        # has not yet validated that setup family, shave size rather than
        # letting a single flashy score consume max capital.
        euphoria_penalty = 0.0
        euphoria_threshold = float(getattr(self.cfg, "euphoria_conviction_threshold", 38.0))
        rl_guard_floor = float(getattr(self.cfg, "euphoria_rl_guard_win_rate", 58.0))
        max_euphoria_penalty = float(getattr(self.cfg, "max_euphoria_penalty", 0.20))
        if conviction >= euphoria_threshold:
            excess_norm = max(0.0, min(1.0, (conviction - euphoria_threshold) / max(1.0, 50.0 - euphoria_threshold)))
            if rl_win_rate < rl_guard_floor:
                history_gap = (rl_guard_floor - rl_win_rate) / max(rl_guard_floor, 1.0)
                euphoria_penalty = max_euphoria_penalty * excess_norm * max(0.35, history_gap)
            if probability < 0.58 or expectancy_r < 0.20:
                euphoria_penalty += min(0.10, excess_norm * 0.12)
            if rl_pattern_boost < 0:
                euphoria_penalty += min(abs(rl_pattern_boost) * 0.50, max_euphoria_penalty * 0.50)
            elif rl_pattern_boost > 0 and rl_win_rate >= rl_guard_floor:
                euphoria_penalty *= 0.50
            euphoria_penalty = min(max_euphoria_penalty, euphoria_penalty)

        # RL pattern boost: further reward recognized winning patterns
        total_factor = confidence_factor * (1.0 + rl_boost + rl_pattern_boost - euphoria_penalty)
        total_factor = max(base_floor * 0.80, min(total_factor, 1.10))

        starter_multiplier = float(sizing_multiplier if sizing_multiplier is not None else 1.0)
        starter_multiplier = max(0.10, min(starter_multiplier, 1.0))
        total_factor *= starter_multiplier

        leverage, leverage_note = self.select_adaptive_leverage(
            conviction_tier=conviction_tier,
            conviction=conviction,
            probability=probability,
            expectancy_r=expectancy_r,
            uncertainty=uncertainty_level,
            rl_win_rate=rl_win_rate,
            rl_pattern_boost=rl_pattern_boost,
            starter_multiplier=starter_multiplier,
            event_starter=event_starter,
            scale_in=False,
            scalp=scalp,
        )

        margin_usd = min(max_margin_usd * total_factor, self._cfg_float("max_trade_usd", max_margin_usd))
        max_notional_cfg = self._cfg_float(
            "max_trade_notional_usd",
            self._cfg_float("max_trade_usd", margin_usd) * max(leverage, 1),
        )
        max_notional_portfolio = portfolio_usd * self._cfg_float("max_levered_position_pct", self.cfg.max_position_pct)
        max_notional_usd = max(0.0, min(max_notional_cfg, max_notional_portfolio))
        size_usd = min(margin_usd * max(leverage, 1), max_notional_usd)

        # Cap at available capital
        avail    = self.available_capital(portfolio_usd)
        size_usd = min(size_usd, avail)
        margin_usd = size_usd / max(leverage, 1)

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

        # ── Duplicate position / scale-in check ──────────────────────
        existing = self.position_direction(coin)
        if existing == direction:
            pos = self.positions.get(coin)

            # Scale into a PROFITABLE winner when conviction is HIGH or EXTREME
            if conviction_tier in ("HIGH", "EXTREME") and pos is not None:
                if pos.direction == "LONG":
                    in_profit = current_price > pos.entry_price * 1.005  # 0.5% buffer
                else:
                    in_profit = current_price < pos.entry_price * 0.995

                if in_profit:
                    # Add up to 50% of normal size, never more than the existing leg
                    scale_usd = min(size_usd * 0.50, pos.size_usd, avail)
                    if scale_usd < self.cfg.min_trade_usd:
                        return OrderRequest(
                            coin=coin, direction=direction,
                            size_usd=0, size_coin=0,
                            price=current_price,
                            stop_loss=stop_loss_price,
                            take_profit=take_profit_price,
                            approved=False,
                            rejection_reason=(
                                f"Scale-in size ${scale_usd:.2f} < "
                                f"minimum ${self.cfg.min_trade_usd} — "
                                f"skipping HYPE-style add"
                            ),
                        )
                    scale_leverage, scale_leverage_note = self.select_adaptive_leverage(
                        conviction_tier=conviction_tier,
                        conviction=conviction,
                        probability=probability,
                        expectancy_r=expectancy_r,
                        uncertainty=uncertainty_level,
                        rl_win_rate=rl_win_rate,
                        rl_pattern_boost=rl_pattern_boost,
                        starter_multiplier=starter_multiplier,
                        event_starter=event_starter,
                        scale_in=True,
                        scalp=scalp,
                    )
                    scale_coin = scale_usd / current_price if current_price > 0 else 0.0
                    scale_margin = scale_usd / max(scale_leverage, 1)
                    log.info(
                        f"[{coin}] 📈 SCALE-IN approved: {direction} +${scale_usd:.2f} notional "
                        f"(${scale_margin:.2f} margin @ {scale_leverage}x) "
                        f"(50%% add to profitable {pos.direction} @ "
                        f"entry=${pos.entry_price:.2f} cur=${current_price:.2f}) "
                        f"| Conviction: {conviction_tier}"
                    )
                    return OrderRequest(
                        coin            = coin,
                        direction       = direction,
                        size_usd        = scale_usd,
                        size_coin       = scale_coin,
                        price           = current_price,
                        stop_loss       = stop_loss_price,
                        take_profit     = take_profit_price,
                        leverage        = scale_leverage,
                        margin_usd      = scale_margin,
                        leverage_note   = scale_leverage_note,
                        approved        = True,
                        conviction_tier = conviction_tier + "_SCALEIN",
                        conviction_pct  = total_factor * 100,
                        is_scale_in     = True,
                    )
                else:
                    return OrderRequest(
                        coin=coin, direction=direction,
                        size_usd=0, size_coin=0,
                        price=current_price,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                        approved=False,
                        rejection_reason=(
                            f"Already {existing} on {coin} — "
                            f"position not yet profitable enough to scale into"
                        ),
                    )
            else:
                return OrderRequest(
                    coin=coin, direction=direction,
                    size_usd=0, size_coin=0,
                    price=current_price,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price,
                    approved=False,
                    rejection_reason=f"Already {existing} on {coin} — conviction too low to scale in"
                )

        size_coin = size_usd / current_price if current_price > 0 else 0.0

        sizing_parts = [f"conv={confidence_factor*100:.0f}%"]
        if starter_multiplier < 0.999:
            sizing_parts.append(f"starter={starter_multiplier*100:.0f}%")
            conviction_tier = f"{conviction_tier}_STARTER"
        if scalp:
            sizing_parts.append("style=scalp")
            if "_SCALP" not in conviction_tier:
                conviction_tier = f"{conviction_tier}_SCALP"
        sizing_parts.append(f"p={probability*100:.0f}%")
        sizing_parts.append(f"ev={expectancy_r:+.2f}R")
        sizing_parts.append(f"u={uncertainty_level*100:.0f}%")
        if rl_boost:
            sizing_parts.append(f"rl={rl_boost*100:+.0f}%")
        if rl_pattern_boost:
            sizing_parts.append(f"pattern={rl_pattern_boost*100:+.0f}%")
        if euphoria_penalty:
            sizing_parts.append(f"euphoria={euphoria_penalty*100:.0f}%")
        sizing_parts.append(f"lev={leverage}x")
        sizing_parts.append(f"margin=${margin_usd:.0f}")
        sizing_summary = " ".join(sizing_parts)
        log.info(
            f"[{coin}] ✅ Order approved: {direction} "
            f"${size_usd:.2f} notional (${margin_usd:.2f} margin @ {leverage}x) "
            f"({size_coin:.6f} coins) "
            f"SL=${stop_loss_price:.2f} TP=${take_profit_price:.2f} "
            f"({abs(current_price - stop_loss_price) / max(current_price, 1e-9) * 100:.1f}% SL / "
            f"{abs(take_profit_price - current_price) / max(current_price, 1e-9) * 100:.1f}% TP) "
            f"| Conviction: {conviction_tier} ({conviction:.0f}/50) "
            f"→ {total_factor*100:.0f}% size [{sizing_summary}]"
        )

        return OrderRequest(
            coin            = coin,
            direction       = direction,
            size_usd        = size_usd,
            size_coin       = size_coin,
            price           = current_price,
            stop_loss       = stop_loss_price,
            take_profit     = take_profit_price,
            leverage        = leverage,
            margin_usd      = margin_usd,
            leverage_note   = leverage_note,
            approved        = True,
            conviction_tier = conviction_tier,
            conviction_pct  = total_factor * 100,
        )

    # ── Position lifecycle ─────────────────────────────────────

    def record_open(self, order: OrderRequest, exchange: str = "", metadata: Optional[dict] = None):
        # Calculate initial trailing stop price
        if order.direction == "LONG":
            trail_price = order.price * (1 - self.cfg.trailing_stop_pct)
        else:
            trail_price = order.price * (1 + self.cfg.trailing_stop_pct)

        leverage = max(1, int(getattr(order, "leverage", None) or self.cfg.leverage or 1))
        margin_usd = float(getattr(order, "margin_usd", 0.0) or 0.0)
        if margin_usd <= 0:
            margin_usd = float(order.size_usd or 0.0) / max(leverage, 1)
        metadata_payload = dict(metadata or {})
        metadata_payload.setdefault("leverage", leverage)
        metadata_payload.setdefault("margin_usd", margin_usd)
        metadata_payload.setdefault("leverage_note", getattr(order, "leverage_note", ""))

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
            leverage           = float(leverage),
            margin_usd         = margin_usd,
            metadata           = metadata_payload,
        )
        self.positions[order.coin] = pos
        log.info(
            f"[{order.coin}] 📈 Position OPENED: {order.direction} "
            f"entry=${order.price:.2f} notional=${order.size_usd:.2f} "
            f"margin=${margin_usd:.2f} lev={leverage}x "
            f"SL=${order.stop_loss:.2f} ({abs(order.price - order.stop_loss) / max(order.price, 1e-9) * 100:.1f}%) "
            f"TP=${order.take_profit:.2f} ({abs(order.take_profit - order.price) / max(order.price, 1e-9) * 100:.1f}%)"
        )

    def restore_position(self, position: OpenPosition):
        """Restore a position into the in-memory risk state after restart."""
        if float(getattr(position, "margin_usd", 0.0) or 0.0) <= 0:
            leverage = max(1.0, float(getattr(position, "leverage", 1.0) or 1.0))
            position.margin_usd = float(getattr(position, "size_usd", 0.0) or 0.0) / leverage
        self.positions[position.coin] = position
        log.info(
            f"[{position.coin}] Restored position: {position.direction} "
            f"entry=${position.entry_price:.2f} notional=${position.size_usd:.2f} "
            f"margin=${getattr(position, 'margin_usd', 0.0):.2f} lev={getattr(position, 'leverage', 1.0)}x "
            f"exchange={position.exchange or 'unknown'}"
        )

    def replace_positions(self, positions: Dict[str, OpenPosition]):
        """Replace all tracked positions with exchange-reconciled truth."""
        self.positions = dict(positions)
        if positions:
            log.info(f"Reconciled {len(positions)} live position(s) from exchange state")
        else:
            log.info("Reconciled exchange state: no live positions")

    def record_scale_in_fill(self, order: OrderRequest, exchange: str = ""):
        """Merge a confirmed scale-in fill into an existing position."""
        pos = self.positions.get(order.coin)
        if not pos:
            self.record_open(order, exchange=exchange)
            return

        if pos.direction != order.direction:
            raise ValueError(f"Cannot scale into {order.coin}: position direction mismatch")

        new_size_usd = pos.size_usd + order.size_usd
        new_size_coin = pos.size_coin + order.size_coin
        if new_size_coin <= 0 or new_size_usd <= 0:
            raise ValueError(f"Cannot scale into {order.coin}: invalid aggregate size")

        weighted_entry = (
            (pos.entry_price * pos.size_coin) + (order.price * order.size_coin)
        ) / new_size_coin

        pos.entry_price = weighted_entry
        pos.size_usd = new_size_usd
        pos.size_coin = new_size_coin
        existing_margin = float(getattr(pos, "margin_usd", 0.0) or 0.0)
        if existing_margin <= 0:
            existing_margin = float(pos.size_usd - order.size_usd) / max(float(getattr(pos, "leverage", 1.0) or 1.0), 1.0)
        added_margin = float(getattr(order, "margin_usd", 0.0) or 0.0)
        if added_margin <= 0:
            added_margin = float(order.size_usd or 0.0) / max(int(getattr(order, "leverage", 1) or 1), 1)
        pos.margin_usd = existing_margin + added_margin
        pos.leverage = round(new_size_usd / max(pos.margin_usd, 1e-9), 2)
        pos.stop_loss = order.stop_loss
        pos.take_profit = order.take_profit
        pos.exchange = exchange or pos.exchange
        pos.trailing_stop_price = (
            weighted_entry * (1 - self.cfg.trailing_stop_pct)
            if pos.direction == "LONG"
            else weighted_entry * (1 + self.cfg.trailing_stop_pct)
        )
        log.info(
            f"[{order.coin}] 📈 Scale-in filled: new entry=${weighted_entry:.2f} "
            f"notional=${new_size_usd:.2f} margin=${pos.margin_usd:.2f} "
            f"effective_lev={pos.leverage}x exchange={pos.exchange or 'unknown'}"
        )

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
            "margin_usd": getattr(pos, "margin_usd", 0.0),
            "leverage": getattr(pos, "leverage", 1.0),
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
                    "metadata":        dict(pos.metadata or {}),
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
