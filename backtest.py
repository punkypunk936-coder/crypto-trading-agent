"""
backtest.py — Local backtesting engine for the trading strategy.

Usage
─────
  # Test BTC over last 60 days
  python3 backtest.py --coin BTC --days 60

  # Test all coins, last 30 days, save results
  python3 backtest.py --days 30

  # Custom date range
  python3 backtest.py --coin ETH --start 2024-10-01 --end 2024-12-31

  # More capital, different sizing
  python3 backtest.py --balance 5000 --trade-size 50

Output
──────
  • Printed summary table in terminal
  • backtest_results.xlsx — full trade log + equity curve per coin

How it works
────────────
  1. Fetches historical 1H candles from Hyperliquid public API
  2. Walks through each candle in chronological order
  3. At each step: computes all indicators on the candles BEFORE that point
     (no lookahead — we only use past data)
  4. Generates a signal (LONG/SHORT/FLAT)
  5. Simulates entry/exit with SL and TP (no slippage — conservative assumption)
  6. Tracks portfolio equity, drawdown, win rate

Limitations
───────────
  • No multi-timeframe in backtest (1H only — same as live signals)
  • No regime signals (requires enough history per window — included where possible)
  • Sentiment held constant at neutral (50)
  • No limit orders / re-entry simulation (market fills assumed)
"""

import sys
import argparse
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import List, Optional
import pandas as pd
import numpy as np

# ── Load agent modules ────────────────────────────────────────
from config import config
from data.market_data import completed_candle_frame, fetch_candles
from indicators.technical import compute_signals
from indicators.advanced  import compute_advanced_signals
from indicators.candlestick_patterns import compute_candlestick_patterns
from indicators.mtf import _combine as combine_mtf_bias
from indicators.mtf import _compute_bias as compute_timeframe_bias
from indicators.regimes import compute_regimes
from strategy.aggressive_strategy import AggressiveStrategy
from logger import get_logger

log = get_logger("backtest")

NEUTRAL_SENTIMENT = {
    "signal_score": 50.0,
    "raw_score":    50,
    "label":        "Neutral",
    "is_extreme":   False,
}


def _ensure_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "timestamp" in work.columns:
        ts = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
        if ts.notna().any():
            work.index = ts
            return work
    work.index = pd.date_range(end=datetime.utcnow(), periods=len(work), freq="h")
    return work


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    work = _ensure_timestamp_index(df)
    agg = (
        work[["open", "high", "low", "close", "volume"]]
        .resample(rule)
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna()
        .reset_index(drop=True)
    )
    return agg


def _mtf_from_window(coin: str, window: pd.DataFrame):
    one_h = completed_candle_frame(window)
    four_h = completed_candle_frame(_resample_ohlcv(window, "4h"))
    twelve_h = completed_candle_frame(_resample_ohlcv(window, "12h"))
    if one_h is None or len(one_h) < 20:
        return None
    b1 = compute_timeframe_bias(one_h, "1h")
    b4 = compute_timeframe_bias(four_h, "4h") if four_h is not None and len(four_h) >= 20 else None
    b12 = compute_timeframe_bias(twelve_h, "12h") if twelve_h is not None and len(twelve_h) >= 20 else None
    return combine_mtf_bias(coin, b1, b4, b12)


def _local_level_proxy(window: pd.DataFrame, price: float):
    if window is None or window.empty:
        return None
    recent = window.tail(min(len(window), 48))
    if len(recent) < 12:
        return None

    resistance = float(recent["high"].tail(24).max())
    support = float(recent["low"].tail(24).min())
    resistance_distance = abs(resistance - price) / max(price, 1e-9) * 100 if resistance > 0 else 0.0
    support_distance = abs(price - support) / max(price, 1e-9) * 100 if support > 0 else 0.0
    breakout_state = "NONE"
    if resistance > 0 and price > resistance * 1.001:
        breakout_state = "CONFIRMED_BULLISH_BREAKOUT"
    elif support > 0 and price < support * 0.999:
        breakout_state = "CONFIRMED_BEARISH_BREAKDOWN"

    level_interaction = "BETWEEN_LEVELS"
    range_compression = support_distance <= 0.55 and resistance_distance <= 0.55
    if range_compression and breakout_state == "NONE":
        level_interaction = "RANGE_COMPRESSION"
    elif support_distance <= 0.45:
        level_interaction = "AT_SUPPORT"
    elif resistance_distance <= 0.45:
        level_interaction = "AT_RESISTANCE"
    elif support_distance <= 1.25:
        level_interaction = "ABOVE_SUPPORT"
    elif resistance_distance <= 1.25:
        level_interaction = "BELOW_RESISTANCE"
    elif breakout_state == "CONFIRMED_BULLISH_BREAKOUT":
        level_interaction = "ABOVE_BREAKOUT"
    elif breakout_state == "CONFIRMED_BEARISH_BREAKDOWN":
        level_interaction = "BELOW_BREAKDOWN"

    score = 50.0
    if level_interaction in {"AT_SUPPORT", "ABOVE_SUPPORT"}:
        score += 8.0
    if level_interaction in {"AT_RESISTANCE", "BELOW_RESISTANCE"}:
        score -= 8.0
    if breakout_state == "CONFIRMED_BULLISH_BREAKOUT":
        score += 10.0
    elif breakout_state == "CONFIRMED_BEARISH_BREAKDOWN":
        score -= 10.0
    if level_interaction == "RANGE_COMPRESSION":
        score = 50.0 + (score - 50.0) * 0.35

    return SimpleNamespace(
        valid=True,
        score=score,
        level_interaction=level_interaction,
        breakout_state=breakout_state,
        block_longs=level_interaction in {"AT_RESISTANCE", "BELOW_RESISTANCE", "RANGE_COMPRESSION"} and breakout_state != "CONFIRMED_BULLISH_BREAKOUT",
        block_shorts=level_interaction in {"AT_SUPPORT", "ABOVE_SUPPORT", "RANGE_COMPRESSION"} and breakout_state != "CONFIRMED_BEARISH_BREAKDOWN",
        favor_longs=breakout_state == "CONFIRMED_BULLISH_BREAKOUT" or level_interaction in {"AT_SUPPORT", "ABOVE_SUPPORT"},
        favor_shorts=breakout_state == "CONFIRMED_BEARISH_BREAKDOWN" or level_interaction in {"AT_RESISTANCE", "BELOW_RESISTANCE"},
        nearest_support=support,
        nearest_resistance=resistance,
        nearest_support_distance_pct=support_distance,
        nearest_resistance_distance_pct=resistance_distance,
        nearest_support_strength=0.8,
        nearest_resistance_strength=0.8,
        support_levels=[],
        resistance_levels=[],
        imbalance_ratio=0.0,
        bid_notional=1_000_000.0,
        ask_notional=1_000_000.0,
        spread_bps=1.0,
        daily_breakout_level=resistance,
        daily_breakdown_level=support,
    )


# ─────────────────────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────────────────────

class Backtester:
    def __init__(
        self,
        coin: str,
        df: pd.DataFrame,
        starting_balance: float = 10_000.0,
        trade_size_usd: float = 20.0,
        stop_loss_pct: float = 0.10,
        take_profit_pct: float = 0.50,
        leverage: int = 3,
    ):
        self.coin = coin
        self.df = df.reset_index(drop=True)
        self.balance = starting_balance
        self.starting_bal = starting_balance
        self.trade_size = trade_size_usd
        self.sl_pct = stop_loss_pct
        self.tp_pct = take_profit_pct
        self.leverage = leverage
        self.strategy = AggressiveStrategy(config.trading, config.indicators)
        self.trades: List[dict] = []
        self.equity_curve: List[float] = []
        self.flat_streak = 0

    def _fill_price(self, direction: str, reference_price: float, *, for_exit: bool = False) -> float:
        slip_bps = float(getattr(config.trading, "backtest_market_slippage_bps", 4.0) or 4.0) / 10_000.0
        if direction == "LONG":
            return reference_price * (1 + slip_bps)  # buys fill slightly worse
        return reference_price * (1 - slip_bps)      # sells fill slightly worse

    def _close_position(self, position: dict, exit_price: float, exit_reason: str, closed_at) -> None:
        pnl = self._calc_pnl(position, exit_price)
        self.balance += position["size_usd"] + pnl
        self.trades.append({
            "coin": self.coin,
            "direction": position["direction"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "size_usd": position["size_usd"],
            "pnl_usd": round(pnl, 2),
            "pnl_pct": round(pnl / position["size_usd"] * 100, 2),
            "exit_reason": exit_reason,
            "opened_at": position["opened_at"],
            "closed_at": str(closed_at),
            "balance_after": round(self.balance, 2),
        })

    def run(self, min_candles: int = 80) -> dict:
        df = self.df
        n = len(df)
        position = None

        log.info(f"[{self.coin}] Running walk-forward backtest on {n} candles…")

        for i in range(min_candles, n):
            window = df.iloc[:i].copy()
            analysis_window = completed_candle_frame(window)
            if analysis_window is None or len(analysis_window) < 50:
                continue

            current_row = df.iloc[i]
            entry_reference_price = float(current_row.get("open", current_row["close"]))
            mark_price = float(current_row["close"])
            hi = float(current_row["high"])
            lo = float(current_row["low"])
            ts = current_row.get("timestamp", str(i))

            self.equity_curve.append(self._equity(position, mark_price))

            try:
                tech = compute_signals(analysis_window, self.coin, config.indicators, config.trading)
                advanced = compute_advanced_signals(analysis_window, self.coin)
                regimes = compute_regimes(analysis_window, self.coin)
                candle_patterns = compute_candlestick_patterns(analysis_window, self.coin)
                mtf = _mtf_from_window(self.coin, window)
                orderbook_signal = _local_level_proxy(analysis_window, float(analysis_window["close"].iloc[-1]))
            except Exception:
                continue

            if not tech.valid:
                continue

            tech.closed_price = float(analysis_window["close"].iloc[-1])
            tech.live_price = entry_reference_price
            tech.price = entry_reference_price

            signal = self.strategy.generate_signal(
                tech,
                advanced,
                NEUTRAL_SENTIMENT,
                position["direction"] if position else None,
                regimes,
                news_signal=None,
                candle_patterns=candle_patterns,
                memory_adjustment=0.0,
                instrument_type=config.trading.instrument_types.get(self.coin, "crypto"),
                funding_oi_signal=None,
                orderbook_signal=orderbook_signal,
            )

            if mtf:
                if signal.action == "LONG" and not mtf.allow_long:
                    signal.action = "FLAT"
                    signal.flat_reason = f"MTF blocked LONG ({mtf.reason})"
                elif signal.action == "SHORT" and not mtf.allow_short:
                    signal.action = "FLAT"
                    signal.flat_reason = f"MTF blocked SHORT ({mtf.reason})"

            # ── Check exits first using the current candle range ───────────
            if position:
                exit_reason = None
                exit_price = mark_price
                if position["direction"] == "LONG":
                    if lo <= position["sl"]:
                        exit_reason = "stop_loss"
                        exit_price = self._fill_price("SHORT", position["sl"], for_exit=True)
                    elif hi >= position["tp"]:
                        exit_reason = "take_profit"
                        exit_price = self._fill_price("SHORT", position["tp"], for_exit=True)
                else:
                    if hi >= position["sl"]:
                        exit_reason = "stop_loss"
                        exit_price = self._fill_price("LONG", position["sl"], for_exit=True)
                    elif lo <= position["tp"]:
                        exit_reason = "take_profit"
                        exit_price = self._fill_price("LONG", position["tp"], for_exit=True)

                if exit_reason:
                    self._close_position(position, exit_price, exit_reason, ts)
                    position = None
                    self.flat_streak = 0
                    continue

            if position:
                hold_minutes = max(60.0, (i - position["opened_index"]) * 60.0)
                planned_tp = abs(position["tp"] - position["entry_price"])
                favorable_move = (
                    mark_price - position["entry_price"]
                    if position["direction"] == "LONG"
                    else position["entry_price"] - mark_price
                )
                tp_progress = favorable_move / max(planned_tp, 1e-9) if planned_tp > 0 else 0.0

                if signal.action == "FLAT":
                    self.flat_streak += 1
                else:
                    self.flat_streak = 0

                if self.flat_streak >= config.trading.max_flat_cycles_with_position:
                    exit_price = self._fill_price(
                        "SHORT" if position["direction"] == "LONG" else "LONG",
                        entry_reference_price,
                        for_exit=True,
                    )
                    self._close_position(position, exit_price, "conviction_lost", ts)
                    position = None
                    self.flat_streak = 0
                    continue

                if (
                    hold_minutes >= config.trading.time_stop_minutes
                    and tp_progress < config.trading.time_stop_min_tp_progress
                    and signal.action == "FLAT"
                ):
                    exit_price = self._fill_price(
                        "SHORT" if position["direction"] == "LONG" else "LONG",
                        entry_reference_price,
                        for_exit=True,
                    )
                    self._close_position(position, exit_price, "time_stop", ts)
                    position = None
                    self.flat_streak = 0
                    continue

                if (
                    signal.action in {"LONG", "SHORT"}
                    and signal.action != position["direction"]
                    and hold_minutes >= config.trading.min_hold_minutes
                ):
                    exit_price = self._fill_price(
                        "SHORT" if position["direction"] == "LONG" else "LONG",
                        entry_reference_price,
                        for_exit=True,
                    )
                    self._close_position(position, exit_price, "signal_reversal", ts)
                    position = None
                    self.flat_streak = 0
                    continue

            if position is not None or signal.action == "FLAT":
                continue

            size_usd = min(self.trade_size, self.balance * 0.20)
            if size_usd < 1:
                continue

            fill_price = self._fill_price(signal.action, entry_reference_price, for_exit=False)
            sl = float(signal.stop_loss_price or 0.0)
            tp = float(signal.take_profit_price or 0.0)
            if signal.action == "LONG":
                if sl <= 0:
                    sl = fill_price * (1 - self.sl_pct)
                if tp <= 0:
                    tp = fill_price * (1 + self.tp_pct)
            else:
                if sl <= 0:
                    sl = fill_price * (1 + self.sl_pct)
                if tp <= 0:
                    tp = fill_price * (1 - self.tp_pct)

            self.balance -= size_usd
            position = {
                "direction": signal.action,
                "entry_price": fill_price,
                "sl": sl,
                "tp": tp,
                "size_usd": size_usd,
                "opened_at": str(ts),
                "opened_index": i,
                "score": signal.score,
            }

        if position:
            last_price = self._fill_price(
                "SHORT" if position["direction"] == "LONG" else "LONG",
                float(df.iloc[-1]["close"]),
                for_exit=True,
            )
            self._close_position(position, last_price, "end_of_data", "end")

        return self._summary()

    def _calc_pnl(self, pos: dict, exit_price: float) -> float:
        if pos["direction"] == "LONG":
            pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
        else:
            pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]
        return pnl_pct * pos["size_usd"] * self.leverage

    def _equity(self, pos: Optional[dict], price: float) -> float:
        if pos:
            return self.balance + pos["size_usd"] + self._calc_pnl(pos, price)
        return self.balance

    def _summary(self) -> dict:
        t = self.trades
        if not t:
            return {"coin": self.coin, "trades": 0}

        pnls   = [tr["pnl_usd"] for tr in t]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        eq     = self.equity_curve
        peak   = self.starting_bal
        max_dd = 0.0
        for e in eq:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100
            if dd > max_dd:
                max_dd = dd

        total_return = (self.balance - self.starting_bal) / self.starting_bal * 100

        return {
            "coin":          self.coin,
            "start_balance": self.starting_bal,
            "end_balance":   round(self.balance, 2),
            "total_return":  round(total_return, 2),
            "trades":        len(t),
            "wins":          len(wins),
            "losses":        len(losses),
            "win_rate":      round(len(wins) / len(t) * 100, 1) if t else 0,
            "total_pnl":     round(sum(pnls), 2),
            "avg_win":       round(sum(wins)   / len(wins)   if wins   else 0, 2),
            "avg_loss":      round(sum(losses) / len(losses) if losses else 0, 2),
            "best_trade":    round(max(pnls), 2),
            "worst_trade":   round(min(pnls), 2),
            "max_drawdown":  round(max_dd, 2),
            "profit_factor": round(sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0, 2),
            "trade_log":     t,
        }


# ─────────────────────────────────────────────────────────────
# Excel report
# ─────────────────────────────────────────────────────────────

def save_excel(results: List[dict], path: str = "backtest_results.xlsx"):
    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.utils import get_column_letter
    except ImportError:
        log.error("openpyxl not installed: pip install openpyxl")
        return

    wb = openpyxl.Workbook()

    # ── Sheet 1: Summary ──────────────────────────────────────
    ws = wb.active
    ws.title = "Summary"
    headers = ["Coin", "Trades", "Win Rate %", "Total PnL $", "Return %",
               "Avg Win $", "Avg Loss $", "Best $", "Worst $",
               "Max DD %", "Profit Factor", "End Balance $"]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1a1a2e")

    green_fill = PatternFill("solid", fgColor="003300")
    red_fill   = PatternFill("solid", fgColor="330000")

    for r in results:
        row = [
            r["coin"], r["trades"],
            r["win_rate"], r["total_pnl"], r["total_return"],
            r["avg_win"], r["avg_loss"], r["best_trade"], r["worst_trade"],
            r["max_drawdown"], r["profit_factor"], r["end_balance"],
        ]
        ws.append(row)
        fill = green_fill if r["total_pnl"] > 0 else red_fill
        for cell in ws[ws.max_row]:
            cell.fill = fill

    for col in ws.columns:
        ws.column_dimensions[get_column_letter(col[0].column)].width = 14

    # ── Sheet 2+: Per-coin trade log ──────────────────────────
    for r in results:
        if not r.get("trade_log"):
            continue
        ws2 = wb.create_sheet(title=f"{r['coin']} Trades")
        cols = ["direction", "entry_price", "exit_price", "size_usd",
                "pnl_usd", "pnl_pct", "exit_reason", "opened_at",
                "closed_at", "balance_after"]
        ws2.append(cols)
        for cell in ws2[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1a1a2e")

        for t in r["trade_log"]:
            ws2.append([t.get(c, "") for c in cols])
            fill = green_fill if t["pnl_usd"] > 0 else red_fill
            for cell in ws2[ws2.max_row]:
                cell.fill = fill

        for col in ws2.columns:
            ws2.column_dimensions[get_column_letter(col[0].column)].width = 14

    wb.save(path)
    log.info(f"Backtest results saved: {path}")


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest the trading strategy")
    parser.add_argument("--coin",       default=None,    help="Coin to test (default: all)")
    parser.add_argument("--days",       type=int, default=30, help="Days of history (default: 30)")
    parser.add_argument("--balance",    type=float, default=10_000, help="Starting balance USD")
    parser.add_argument("--trade-size", type=float, default=20,     help="Trade size USD")
    parser.add_argument("--no-excel",   action="store_true",        help="Skip Excel report")
    args = parser.parse_args()

    coins   = [args.coin.upper()] if args.coin else config.trading.coins
    lookback= args.days * 24   # hours → candle count (1H candles)

    all_results = []
    print(f"\n  {'─'*60}")
    print(f"  BACKTEST  |  {args.days} days  |  ${args.balance:,.0f} start  |  ${args.trade_size}/trade")
    print(f"  Coins: {coins}")
    print(f"  {'─'*60}\n")

    for coin in coins:
        print(f"  Fetching {coin} candles ({lookback} periods)…")
        df = fetch_candles(coin=coin, interval="1h", lookback=lookback)
        if df is None or len(df) < 60:
            print(f"  ⚠️  Not enough {coin} data — skipping\n")
            continue

        bt  = Backtester(
            coin              = coin,
            df                = df,
            starting_balance  = args.balance,
            trade_size_usd    = args.trade_size,
            stop_loss_pct     = config.trading.stop_loss_pct,
            take_profit_pct   = config.trading.take_profit_pct,
            leverage          = config.trading.leverage,
        )
        res = bt.run()
        all_results.append(res)

        tag = "✅" if res.get("total_pnl", 0) > 0 else "❌"
        print(f"  {tag} {coin:5s}  Trades={res['trades']:3d}  "
              f"Win={res['win_rate']:5.1f}%  "
              f"PnL=${res['total_pnl']:+,.2f}  "
              f"Return={res['total_return']:+.1f}%  "
              f"MaxDD={res['max_drawdown']:.1f}%")

    print(f"\n  {'─'*60}")
    if all_results:
        total_pnl = sum(r.get("total_pnl", 0) for r in all_results)
        print(f"  Combined PnL: ${total_pnl:+,.2f}")
        if not args.no_excel:
            save_excel(all_results, "backtest_results.xlsx")
            print(f"  Report saved: backtest_results.xlsx")
    print(f"  {'─'*60}\n")


if __name__ == "__main__":
    main()
