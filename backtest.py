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
from typing import List, Optional
import pandas as pd
import numpy as np

# ── Load agent modules ────────────────────────────────────────
from config import config
from data.market_data import fetch_candles
from indicators.technical import compute_signals
from indicators.advanced  import compute_advanced_signals
from strategy.aggressive_strategy import AggressiveStrategy
from logger import get_logger

log = get_logger("backtest")

NEUTRAL_SENTIMENT = {
    "signal_score": 50.0,
    "raw_score":    50,
    "label":        "Neutral",
    "is_extreme":   False,
}


# ─────────────────────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────────────────────

class Backtester:
    def __init__(self, coin: str, df: pd.DataFrame,
                 starting_balance: float = 10_000.0,
                 trade_size_usd: float   = 20.0,
                 stop_loss_pct: float    = 0.10,
                 take_profit_pct: float  = 0.50,
                 leverage: int           = 3):
        self.coin           = coin
        self.df             = df.reset_index(drop=True)
        self.balance        = starting_balance
        self.starting_bal   = starting_balance
        self.trade_size     = trade_size_usd
        self.sl_pct         = stop_loss_pct
        self.tp_pct         = take_profit_pct
        self.leverage       = leverage
        self.strategy       = AggressiveStrategy(config.trading, config.indicators)
        self.trades: List[dict] = []
        self.equity_curve: List[float] = []

    def run(self, min_candles: int = 50) -> dict:
        """
        Walk through candles chronologically.
        At each step we have candles[0:i] — no future data.
        """
        df      = self.df
        n       = len(df)
        position= None   # {"direction", "entry_price", "sl", "tp", "size_usd", "opened_at"}

        log.info(f"[{self.coin}] Running backtest on {n} candles…")

        for i in range(min_candles, n):
            window = df.iloc[:i].copy()
            price  = float(df.iloc[i]["close"])
            ts     = df.iloc[i].get("timestamp", str(i))

            self.equity_curve.append(self._equity(position, price))

            # ── Check exits on current candle ──────────────────
            if position:
                exit_reason = None
                hi = float(df.iloc[i]["high"])
                lo = float(df.iloc[i]["low"])

                if position["direction"] == "LONG":
                    if lo <= position["sl"]:
                        exit_reason = "stop_loss"
                        exit_price  = position["sl"]
                    elif hi >= position["tp"]:
                        exit_reason = "take_profit"
                        exit_price  = position["tp"]
                else:   # SHORT
                    if hi >= position["sl"]:
                        exit_reason = "stop_loss"
                        exit_price  = position["sl"]
                    elif lo <= position["tp"]:
                        exit_reason = "take_profit"
                        exit_price  = position["tp"]

                if exit_reason:
                    pnl = self._calc_pnl(position, exit_price)
                    self.balance += position["size_usd"] + pnl
                    self.trades.append({
                        "coin":        self.coin,
                        "direction":   position["direction"],
                        "entry_price": position["entry_price"],
                        "exit_price":  exit_price,
                        "size_usd":    position["size_usd"],
                        "pnl_usd":     round(pnl, 2),
                        "pnl_pct":     round(pnl / position["size_usd"] * 100, 2),
                        "exit_reason": exit_reason,
                        "opened_at":   position["opened_at"],
                        "closed_at":   str(ts),
                        "balance_after": round(self.balance, 2),
                    })
                    position = None

            # ── Generate signal ────────────────────────────────
            if position is None:
                try:
                    tech     = compute_signals(window, self.coin, config.indicators, config.trading)
                    advanced = compute_advanced_signals(window, self.coin)
                    if not tech.valid:
                        continue

                    current_dir = position["direction"] if position else None
                    signal = self.strategy.generate_signal(
                        tech, advanced, NEUTRAL_SENTIMENT, current_dir
                    )
                except Exception:
                    continue

                if signal.action == "FLAT":
                    continue

                size_usd = min(self.trade_size, self.balance * 0.20)
                if size_usd < 1:
                    continue

                self.balance -= size_usd

                if signal.action == "LONG":
                    sl = price * (1 - self.sl_pct)
                    tp = price * (1 + self.tp_pct)
                else:
                    sl = price * (1 + self.sl_pct)
                    tp = price * (1 - self.tp_pct)

                position = {
                    "direction":   signal.action,
                    "entry_price": price,
                    "sl":          sl,
                    "tp":          tp,
                    "size_usd":    size_usd,
                    "opened_at":   str(ts),
                }

        # Close any remaining open position at last price
        if position:
            last_price = float(df.iloc[-1]["close"])
            pnl = self._calc_pnl(position, last_price)
            self.balance += position["size_usd"] + pnl
            self.trades.append({
                "coin":        self.coin,
                "direction":   position["direction"],
                "entry_price": position["entry_price"],
                "exit_price":  last_price,
                "size_usd":    position["size_usd"],
                "pnl_usd":     round(pnl, 2),
                "pnl_pct":     round(pnl / position["size_usd"] * 100, 2),
                "exit_reason": "end_of_data",
                "opened_at":   position["opened_at"],
                "closed_at":   "end",
                "balance_after": round(self.balance, 2),
            })

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
