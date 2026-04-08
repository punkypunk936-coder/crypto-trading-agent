"""
runtime_power.py — macOS power-state helpers for local live-trading safety.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class PowerStatus:
    available: bool
    on_ac_power: Optional[bool] = None
    battery_pct: Optional[int] = None
    charging: Optional[bool] = None
    source: str = ""
    raw: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def get_power_status() -> PowerStatus:
    if sys.platform != "darwin":
        return PowerStatus(available=False, raw="unsupported platform")

    try:
        result = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return PowerStatus(available=False, raw=str(exc))

    raw = (result.stdout or "").strip()
    if result.returncode != 0 or not raw:
        return PowerStatus(available=False, raw=raw or result.stderr or "pmset failed")

    lower = raw.lower()
    on_ac = "ac power" in lower
    source = "AC Power" if on_ac else "Battery Power" if "battery power" in lower else ""

    pct_match = re.search(r"(\d+)%", raw)
    battery_pct = int(pct_match.group(1)) if pct_match else None

    charging = None
    if "discharging" in lower or "not charging" in lower:
        charging = False
    elif "charging" in lower:
        charging = True

    return PowerStatus(
        available=True,
        on_ac_power=on_ac,
        battery_pct=battery_pct,
        charging=charging,
        source=source,
        raw=raw,
    )
