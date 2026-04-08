"""
logger.py — Centralized, colour-coded logging for the trading agent.
Every module imports `log` from here.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


# ── Colour codes for terminal output ─────────────────────
COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
    "RESET":    "\033[0m",
}


class ColourFormatter(logging.Formatter):
    FMT = "[%(asctime)s] %(levelname)-8s %(name)-20s %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record):
        colour = COLOURS.get(record.levelname, "")
        reset  = COLOURS["RESET"]
        formatter = logging.Formatter(
            f"{colour}{self.FMT}{reset}", datefmt=self.DATE_FMT
        )
        return formatter.format(record)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger
    logger.setLevel(logging.DEBUG)

    # Console handler (coloured)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColourFormatter())
    logger.addHandler(ch)

    # File handler (plain text, rotated daily)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    fh = logging.FileHandler(LOG_DIR / f"agent_{today}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)-20s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)
    logger.propagate = False
    return logger


# Default logger used by most modules
log = get_logger("agent")
