"""
indicators/chart_analyst.py — Visual chart reading using Claude's vision API.

What this does
──────────────
Sends a chart screenshot (or auto-captured TradingView URL screenshot) to
Claude claude-haiku-4-5 (fast + cheap) with a structured prompt that forces
a clear, unambiguous trading verdict:

  VERDICT: WAIT | LONG | SHORT
  CONFIDENCE: HIGH | MEDIUM | LOW
  ENTRY ZONE: $xxx – $xxx
  STOP LOSS: $xxx
  TAKE PROFIT: $xxx
  REASONING: ...

This is used as a CONFIRMATION LAYER on top of the indicator-based signals.
The agent only trades if both the indicator score AND the chart reading agree.

Usage
─────
  from indicators.chart_analyst import read_chart, ChartVerdict

  verdict = read_chart(image_path="screenshot.png", coin="BTC")
  if verdict.action == "LONG" and signal.action == "LONG":
      # Confirmed — execute

  # Or: pass a raw PIL image
  verdict = read_chart(image=pil_image, coin="ETH")
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import re

from logger import get_logger

log = get_logger("chart_analyst")


# ── Prompt that forces a structured, no-fluff trading verdict ─────────────

CHART_PROMPT = """You are a professional cryptocurrency perps trader.
Analyse this price chart and give a clear, concise trading verdict.

RULES:
- Think in LONG / SHORT / WAIT only. No hedging, no "maybe".
- If you are not confident enough to trade, say WAIT.
- Focus on: trend direction, structure (HH/HL or LH/LL), EMA position, key S/R.
- Be direct. No lengthy explanations. Max 3 sentences of reasoning.

Return your answer in EXACTLY this format (copy the template, fill in values):

VERDICT: <LONG|SHORT|WAIT>
CONFIDENCE: <HIGH|MEDIUM|LOW>
ENTRY_ZONE: $<low>-$<high>  (or N/A if WAIT)
STOP_LOSS: $<price>          (or N/A if WAIT)
TAKE_PROFIT: $<price>        (or N/A if WAIT)
REASONING: <3 sentences max>

Coin: {coin}
"""


@dataclass
class ChartVerdict:
    """Parsed output from the chart analyst."""
    coin: str
    action: str          # "LONG", "SHORT", "WAIT"
    confidence: str      # "HIGH", "MEDIUM", "LOW"
    entry_low: float     = 0.0
    entry_high: float    = 0.0
    stop_loss: float     = 0.0
    take_profit: float   = 0.0
    reasoning: str       = ""
    raw_response: str    = ""
    valid: bool          = False
    error: str           = ""

    @property
    def entry_mid(self) -> float:
        if self.entry_low > 0 and self.entry_high > 0:
            return (self.entry_low + self.entry_high) / 2
        return self.entry_low or self.entry_high


def read_chart(
    coin: str,
    image_path: Optional[str] = None,
    image: Optional[object] = None,    # PIL.Image accepted
    image_bytes: Optional[bytes] = None,
) -> ChartVerdict:
    """
    Analyse a chart image and return a structured trading verdict.

    Provide exactly one of: image_path, image (PIL), or image_bytes.
    Falls back gracefully to WAIT if the API is unavailable.
    """
    # ── Get image bytes ────────────────────────────────────────
    raw_bytes = _get_image_bytes(image_path, image, image_bytes)
    if raw_bytes is None:
        return ChartVerdict(coin=coin, action="WAIT", confidence="LOW",
                            valid=False, error="No image provided")

    # ── Encode to base64 ───────────────────────────────────────
    b64 = base64.standard_b64encode(raw_bytes).decode("utf-8")

    # ── Call Claude vision ─────────────────────────────────────
    try:
        import anthropic   # pip install anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            log.warning("ANTHROPIC_API_KEY not set — chart analyst disabled")
            return ChartVerdict(coin=coin, action="WAIT", confidence="LOW",
                                valid=False,
                                error="ANTHROPIC_API_KEY not set in .env")

        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",   # fast and cheap
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/png",
                            "data":       b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": CHART_PROMPT.format(coin=coin),
                    }
                ]
            }]
        )

        raw = response.content[0].text
        verdict = _parse_response(raw, coin)
        log.info(
            f"[{coin}] Chart analyst: {verdict.action} ({verdict.confidence}) — "
            f"{verdict.reasoning[:80]}"
        )
        return verdict

    except ImportError:
        log.warning("anthropic package not installed. "
                    "Run: pip install anthropic --break-system-packages")
        return ChartVerdict(coin=coin, action="WAIT", confidence="LOW",
                            valid=False, error="anthropic SDK not installed")
    except Exception as e:
        log.error(f"[{coin}] Chart analyst error: {e}")
        return ChartVerdict(coin=coin, action="WAIT", confidence="LOW",
                            valid=False, error=str(e))


def _get_image_bytes(image_path, image, image_bytes) -> Optional[bytes]:
    """Convert any of the three accepted input forms to raw bytes."""
    if image_bytes is not None:
        return image_bytes

    if image_path is not None:
        p = Path(image_path)
        if not p.exists():
            log.error(f"Chart image not found: {image_path}")
            return None
        return p.read_bytes()

    if image is not None:
        # PIL.Image support
        import io
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    return None


def _parse_response(raw: str, coin: str) -> ChartVerdict:
    """Extract structured fields from the model's text response."""
    lines = raw.strip().splitlines()

    def _get(key: str) -> str:
        for line in lines:
            if line.upper().startswith(key.upper() + ":"):
                return line.split(":", 1)[1].strip()
        return ""

    def _price(s: str) -> float:
        nums = re.findall(r"[\d,]+\.?\d*", s.replace(",", ""))
        return float(nums[0]) if nums else 0.0

    action     = _get("VERDICT").upper()
    confidence = _get("CONFIDENCE").upper()
    entry_raw  = _get("ENTRY_ZONE")
    sl_raw     = _get("STOP_LOSS")
    tp_raw     = _get("TAKE_PROFIT")
    reasoning  = _get("REASONING")

    # Normalise action
    if "LONG" in action:
        action = "LONG"
    elif "SHORT" in action:
        action = "SHORT"
    else:
        action = "WAIT"

    # Parse entry zone  "$69,500-$70,000"
    entry_prices = re.findall(r"[\d,]+\.?\d*", entry_raw.replace(",", ""))
    entry_low    = float(entry_prices[0]) if len(entry_prices) > 0 else 0.0
    entry_high   = float(entry_prices[1]) if len(entry_prices) > 1 else entry_low

    return ChartVerdict(
        coin         = coin,
        action       = action,
        confidence   = confidence if confidence in ("HIGH", "MEDIUM", "LOW") else "LOW",
        entry_low    = entry_low,
        entry_high   = entry_high,
        stop_loss    = _price(sl_raw),
        take_profit  = _price(tp_raw),
        reasoning    = reasoning,
        raw_response = raw,
        valid        = action in ("LONG", "SHORT", "WAIT"),
    )


# ── Manual test: python -m indicators.chart_analyst ───────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m indicators.chart_analyst <coin> <image_path>")
        print("Example: python -m indicators.chart_analyst BTC chart.png")
        sys.exit(1)

    coin_arg  = sys.argv[1].upper()
    img_arg   = sys.argv[2]

    print(f"\nAnalysing {coin_arg} chart: {img_arg}\n")
    v = read_chart(coin=coin_arg, image_path=img_arg)

    print(f"VERDICT    : {v.action}")
    print(f"CONFIDENCE : {v.confidence}")
    print(f"ENTRY ZONE : ${v.entry_low:,.2f} – ${v.entry_high:,.2f}")
    print(f"STOP LOSS  : ${v.stop_loss:,.2f}")
    print(f"TAKE PROFIT: ${v.take_profit:,.2f}")
    print(f"REASONING  : {v.reasoning}")
    if v.error:
        print(f"ERROR      : {v.error}")
