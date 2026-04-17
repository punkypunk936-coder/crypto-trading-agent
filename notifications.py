"""
notifications.py
Optional Telegram alerts for every trade event.
Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your .env to activate.
"""

import requests
from logger import get_logger

log = get_logger("notifications")


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id   = chat_id
        self.base_url  = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    def send(self, message: str) -> bool:
        try:
            resp = requests.post(self.base_url, json={
                "chat_id":    self.chat_id,
                "text":       message,
                "parse_mode": "Markdown",
            }, timeout=5)
            return resp.status_code == 200
        except Exception as e:
            log.warning(f"Telegram notification failed: {e}")
            return False

    def trade_opened(self, coin: str, direction: str, price: float,
                     size_usd: float, sl: float, tp: float,
                     score: float, exchange: str,
                     is_scale_in: bool = False,
                     total_size_usd: float | None = None):
        emoji = "🟢" if direction == "LONG" else "🔴"
        sl_pct = abs(price - sl) / price * 100 if price else 0.0
        tp_pct = abs(tp - price) / price * 100 if price else 0.0
        rr = tp_pct / sl_pct if sl_pct > 0 else 0.0
        headline = f"{direction} scale-in added — {coin}" if is_scale_in else f"{direction} opened — {coin}"
        msg = (
            f"{emoji} *{headline}*\n"
            f"Exchange: {exchange}\n"
            f"Entry:  ${price:,.2f}\n"
            f"Size:   ${size_usd:,.2f}\n"
            f"Stop:   ${sl:,.2f}  (-{sl_pct:.0f}%)\n"
            f"Target: ${tp:,.2f}  (+{tp_pct:.0f}%)\n"
            f"R:R:    {rr:.2f}\n"
            f"Signal: {score:.1f}/100"
        )
        if is_scale_in and total_size_usd and total_size_usd > 0:
            msg += f"\nTotal position: ${total_size_usd:,.2f}"
        self.send(msg)

    def trade_closed(self, coin: str, direction: str, entry: float,
                     exit_price: float, pnl_usd: float, reason: str):
        emoji = "✅" if pnl_usd >= 0 else "❌"
        if direction == "LONG":
            pnl_pct = (exit_price - entry) / entry * 100
        else:
            pnl_pct = (entry - exit_price) / entry * 100
        if reason.startswith("emergency_close_all"):
            reason_label = "🚨 Emergency close-all"
        else:
            reason_label = {
                "take_profit":     "🎯 Take Profit hit",
                "stop_loss":       "🛑 Stop Loss hit",
                "trailing_stop":   "📉 Trailing Stop hit",
                "signal_reversal": "🔄 Signal reversed",
                "conviction_lost": "🧠 Conviction faded",
                "manual_test":     "🧪 Manual close",
            }.get(reason, reason)
        msg = (
            f"{emoji} *{coin} {direction} closed*\n"
            f"{reason_label}\n"
            f"Entry: ${entry:,.2f} → Exit: ${exit_price:,.2f}\n"
            f"PnL: {pnl_pct:+.2f}% (${pnl_usd:+,.2f})"
        )
        self.send(msg)

    def error_alert(self, message: str):
        self.send(f"⚠️ *Agent Error*\n{message}")

    def heartbeat(self, portfolio_usd: float, open_positions: int):
        self.send(
            f"💓 *Heartbeat*\n"
            f"Portfolio: ${portfolio_usd:,.2f}\n"
            f"Open positions: {open_positions}"
        )


class NoOpNotifier:
    """Used when Telegram is not configured — all methods are silent no-ops."""
    def send(self, *a, **kw):           pass
    def trade_opened(self, *a, **kw):   pass
    def trade_closed(self, *a, **kw):   pass
    def error_alert(self, *a, **kw):    pass
    def heartbeat(self, *a, **kw):      pass


def build_notifier(cfg):
    if cfg.notifications.enabled:
        log.info("Telegram notifications enabled ✅")
        return TelegramNotifier(
            cfg.notifications.telegram_bot_token,
            cfg.notifications.telegram_chat_id,
        )
    log.info("Running without Telegram notifications")
    return NoOpNotifier()
