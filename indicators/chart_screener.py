"""
indicators/chart_screener.py — Auto-capture TradingView charts for visual analysis.

What this does
──────────────
Uses playwright (headless browser) to open TradingView chart URLs for each
coin, wait for the candles to load, then take a screenshot and pass it to
chart_analyst.py for a WAIT / LONG / SHORT verdict.

This runs ONCE per agent cycle (not on every single coin — only on coins
where the indicator score is borderline, i.e. 45–65, where visual confirmation
is most useful).

Requirements (install once):
  pip install playwright --break-system-packages
  playwright install chromium

TradingView chart URLs (public, no login needed):
  BTC  → https://www.tradingview.com/chart/?symbol=BINANCE:BTCUSDT
  ETH  → https://www.tradingview.com/chart/?symbol=BINANCE:ETHUSDT
  SOL  → https://www.tradingview.com/chart/?symbol=BINANCE:SOLUSDT
  HYPE → https://www.tradingview.com/chart/?symbol=BINANCE:HYPEUSDT

You can override these with your own Lighter chart links in config.py:
  chart_urls = { "BTC": "https://lighter.xyz/trade/btc-perp" }
"""

from __future__ import annotations

import os
import time
import tempfile
from typing import Optional, Dict

from logger import get_logger
from indicators.chart_analyst import ChartVerdict, read_chart

log = get_logger("chart_screener")


# Default TradingView chart URLs — no login required, works headless
DEFAULT_CHART_URLS: Dict[str, str] = {
    "BTC":  "https://www.tradingview.com/chart/?symbol=BINANCE:BTCUSDT&interval=240",
    "ETH":  "https://www.tradingview.com/chart/?symbol=BINANCE:ETHUSDT&interval=240",
    "SOL":  "https://www.tradingview.com/chart/?symbol=BINANCE:SOLUSDT&interval=240",
    "HYPE": "https://www.tradingview.com/chart/?symbol=BINANCE:HYPEUSDT&interval=240",
}

# How long to wait for chart candles to fully render (seconds)
CHART_LOAD_WAIT = 8


def capture_chart_screenshot(
    coin: str,
    url: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Optional[bytes]:
    """
    Open a TradingView chart in a headless browser and take a screenshot.

    Returns raw PNG bytes, or None if playwright is unavailable.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.warning(
            "playwright not installed — chart screenshots disabled.\n"
            "  Install with: pip install playwright --break-system-packages\n"
            "  Then: playwright install chromium"
        )
        return None

    chart_url = url or DEFAULT_CHART_URLS.get(coin)
    if not chart_url:
        log.warning(f"[{coin}] No chart URL configured")
        return None

    log.info(f"[{coin}] Capturing chart from {chart_url}")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page    = browser.new_page(viewport={"width": 1280, "height": 720})

            # Block ads, trackers and cookie consent dialogs
            page.route("**/*consent*", lambda route: route.abort())
            page.route("**/*cookie*", lambda route: route.abort())
            page.route("**/*analytics*", lambda route: route.abort())

            page.goto(chart_url, wait_until="networkidle", timeout=30_000)
            time.sleep(CHART_LOAD_WAIT)   # wait for candles to render

            # Dismiss any popups (TradingView cookie banners etc.)
            for selector in ["[data-name='cookie-policy-dialog'] button",
                             "[class*='closeButton']",
                             "[aria-label='Close']"]:
                try:
                    btn = page.query_selector(selector)
                    if btn:
                        btn.click()
                        time.sleep(0.5)
                except Exception:
                    pass

            screenshot_bytes = page.screenshot(full_page=False)
            browser.close()

        if save_path:
            with open(save_path, "wb") as f:
                f.write(screenshot_bytes)
            log.info(f"[{coin}] Chart saved to {save_path}")

        log.info(f"[{coin}] Chart screenshot captured ({len(screenshot_bytes)//1024} KB)")
        return screenshot_bytes

    except Exception as e:
        log.error(f"[{coin}] Screenshot failed: {e}")
        return None


def screen_coin(
    coin: str,
    url: Optional[str] = None,
    save_screenshots: bool = False,
) -> ChartVerdict:
    """
    Capture a chart screenshot for `coin` and return a ChartVerdict.
    This is the main function called by the agent.

    If playwright is unavailable, returns WAIT with a note.
    """
    save_path = None
    if save_screenshots:
        os.makedirs("screenshots", exist_ok=True)
        save_path = f"screenshots/{coin}_{int(time.time())}.png"

    img_bytes = capture_chart_screenshot(coin, url=url, save_path=save_path)

    if img_bytes is None:
        log.warning(f"[{coin}] Chart screener unavailable — returning WAIT")
        return ChartVerdict(
            coin=coin, action="WAIT", confidence="LOW", valid=False,
            error="chart screener unavailable (playwright not installed)"
        )

    return read_chart(coin=coin, image_bytes=img_bytes)


def screen_borderline_coins(
    coins: list,
    indicator_scores: Dict[str, float],
    chart_urls: Optional[Dict[str, str]] = None,
    borderline_range: tuple = (42.0, 62.0),
    save_screenshots: bool = False,
) -> Dict[str, ChartVerdict]:
    """
    Only screen coins where the indicator score is in the borderline range
    (where visual confirmation adds the most value).

    Coins with a very strong score (>62) or very weak score (<38) are
    handled by the indicator system alone — no need to waste API calls.

    Parameters
    ----------
    coins              : list of coin tickers
    indicator_scores   : {coin: score} from the strategy
    chart_urls         : optional override URLs per coin
    borderline_range   : (low, high) score range to trigger visual check
    save_screenshots   : save PNGs to screenshots/ folder

    Returns dict of {coin: ChartVerdict} for borderline coins only.
    """
    verdicts: Dict[str, ChartVerdict] = {}
    lo, hi = borderline_range

    for coin in coins:
        score = indicator_scores.get(coin, 50.0)
        if lo <= score <= hi:
            log.info(f"[{coin}] Score {score:.1f} is borderline — requesting visual check")
            url = (chart_urls or {}).get(coin)
            verdicts[coin] = screen_coin(coin, url=url,
                                         save_screenshots=save_screenshots)
        else:
            log.debug(f"[{coin}] Score {score:.1f} outside borderline range — skipping visual")

    return verdicts


# ── Standalone manual run ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    coin_arg = sys.argv[1].upper() if len(sys.argv) > 1 else "BTC"
    print(f"\nScreening {coin_arg} (this will open a headless browser)…\n")
    v = screen_coin(coin_arg, save_screenshots=True)
    print(f"VERDICT    : {v.action}")
    print(f"CONFIDENCE : {v.confidence}")
    if v.action != "WAIT":
        print(f"ENTRY ZONE : ${v.entry_low:,.2f} – ${v.entry_high:,.2f}")
        print(f"STOP LOSS  : ${v.stop_loss:,.2f}")
        print(f"TAKE PROFIT: ${v.take_profit:,.2f}")
    print(f"REASONING  : {v.reasoning}")
