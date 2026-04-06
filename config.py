"""
config.py — Central configuration for the Crypto Trading Agent
All settings live here. Secrets are loaded from your .env file.
"""

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────
# Exchange Credentials  (loaded from .env — never hardcode!)
# ─────────────────────────────────────────────────────────
@dataclass
class ExchangeConfig:
    # Hyperliquid  ─ uses an EVM private key
    hl_private_key: str        = field(default_factory=lambda: os.getenv("HL_PRIVATE_KEY", ""))
    hl_account_address: str    = field(default_factory=lambda: os.getenv("HL_ACCOUNT_ADDRESS", ""))
    hl_use_mainnet: bool       = True          # set False for testnet

    # Lighter
    # LIGHTER_PRIVATE_KEY remains supported as a legacy alias for the L1 wallet key.
    lighter_l1_private_key: str = field(
        default_factory=lambda: os.getenv("LIGHTER_L1_PRIVATE_KEY", "") or os.getenv("LIGHTER_PRIVATE_KEY", "")
    )
    lighter_api_private_key: str = field(default_factory=lambda: os.getenv("LIGHTER_API_PRIVATE_KEY", ""))
    lighter_account_index: str   = field(default_factory=lambda: os.getenv("LIGHTER_ACCOUNT_INDEX", ""))
    lighter_api_key_index: int   = field(default_factory=lambda: int(os.getenv("LIGHTER_API_KEY_INDEX", "1")))
    lighter_api_base_url: str    = field(
        default_factory=lambda: os.getenv("LIGHTER_API_BASE_URL", "https://mainnet.zklighter.elliot.ai")
    )
    lighter_web3_url: str        = field(
        default_factory=lambda: os.getenv("LIGHTER_WEB3_URL", "https://arb1.arbitrum.io/rpc")
    )

    # Enable / disable each exchange independently
    use_hyperliquid: bool      = False
    use_lighter: bool          = True    # Primary exchange — key in .env


# ─────────────────────────────────────────────────────────
# Trading Parameters
# ─────────────────────────────────────────────────────────
@dataclass
class TradingConfig:
    # Coins / instruments the agent will trade
    # Primary tradable universe on the current Lighter runtime.
    coins: List[str]            = field(default_factory=lambda: ["BTC", "ETH", "SOL"])

    # Broader watchlist / learning universe.
    # These assets are analysed and shown in the dashboard even when they are
    # not tradable on the primary execution venue.
    analysis_coins: List[str]   = field(default_factory=lambda: [
        "BTC", "ETH", "SOL", "HYPE", "SP500", "TAO", "BRENT", "WTI"
    ])

    # ── Instrument type classification ───────────────────────────────────────
    # "crypto"  → standard crypto perp logic
    # "index"   → macro / non-crypto instrument (SP500, BRENT, WTI, etc.) —
    #             Yahoo-backed market data, macro/news-driven, smoother momentum,
    #             higher minimum hold time
    instrument_types: dict      = field(default_factory=lambda: {
        "BTC":   "crypto",
        "ETH":   "crypto",
        "SOL":   "crypto",
        "HYPE":  "crypto",
        "TAO":   "crypto",
        "SP500": "index",
        "BRENT": "index",
        "WTI":   "index",
        "CL":    "index",
    })

    # Indexes need a longer minimum hold — they move slower than crypto
    index_min_hold_minutes: float = 360.0   # 6h for indexes (vs 4h for crypto)

    # ── Aggressive strategy thresholds ──────────────────
    # RSI: go long when below / short when above
    rsi_long_threshold: float    = 42.0
    rsi_short_threshold: float   = 58.0

    # Overall signal score (0–100). Enter when score is outside neutral zone.
    signal_long_threshold: float  = 65.0   # Tighter: require stronger conviction for longs
    signal_short_threshold: float = 35.0   # Tighter: require stronger conviction for shorts

    # ── Anti-whipsaw settings ────────────────────────────────────
    # Minimum time a position must be held before a signal_reversal is allowed.
    # Prevents the agent flip-flopping on borderline signals every few hours.
    min_hold_minutes: float       = 240.0  # 4 hours minimum hold

    # A signal_reversal requires a STRONGER signal than a fresh entry.
    # e.g. if signal_long_threshold=65, reversal from SHORT→LONG needs ≥73
    reversal_threshold_boost: float = 8.0  # extra points required to reverse

    # Minimum signal strength delta for re-entry (avoids flip-flopping)
    min_signal_delta: float       = 5.0

    # ── Signal streak (persistence filter) ───────────────────────
    # How many consecutive cycles must agree on direction before entering.
    # Filters single-candle noise crossings. 2 = 4 min confirmation at 2min cycles.
    signal_streak_required: int   = 2

    # ── FLAT-while-positioned guard ───────────────────────────────
    # If signal stays FLAT for this many consecutive cycles while holding a
    # position, the original thesis is gone → close the trade.
    max_flat_cycles_with_position: int = 3  # ~6 min at 2min cycles

    # ── Position sizing ─────────────────────────────────
    max_position_pct: float       = 0.05   # Max 5% of portfolio per trade (was 2%)
    max_total_exposure_pct: float = 0.40   # Max 40% total deployed at once (was 20%)
    leverage: int                 = 2      # Safer default for small live capital
    conviction_size_floor: float  = 0.30   # Minimum size factor at threshold conviction
    conviction_size_curve: float  = 0.85   # <1 ramps size faster, >1 slower
    euphoria_conviction_threshold: float = 38.0
    euphoria_rl_guard_win_rate: float = 58.0
    max_euphoria_penalty: float   = 0.20   # Max size haircut when conviction outruns learned edge

    # ── Risk management ─────────────────────────────────
    # Wide TP/SL to let winning trades run on perps
    stop_loss_pct: float          = 0.10   # 10% stop-loss (wider = survives volatility)
    take_profit_pct: float        = 0.50   # 50% take-profit (let big moves run)
    dynamic_trade_planning: bool  = True   # ATR + structure-based SL/TP planning
    base_stop_atr_multiple: float = 1.35
    min_stop_atr_multiple: float  = 0.90
    max_stop_atr_multiple: float  = 3.25
    base_target_r_multiple: float = 2.00
    min_target_r_multiple: float  = 1.35
    max_target_r_multiple: float  = 3.50
    trailing_stop_enabled: bool   = True
    trailing_stop_pct: float      = 0.12   # 12% trailing stop (locks profits as price runs)
    use_orderbook_levels: bool    = True   # Live L2 + daily key-level intelligence for BTC/ETH/SOL
    orderbook_depth_limit: int    = 120
    orderbook_daily_lookback: int = 120
    orderbook_cache_ttl_seconds: int = 25
    orderbook_guard_distance_pct: float = 1.25
    orderbook_reaction_distance_pct: float = 0.45
    orderbook_level_min_strength: float = 0.55
    orderbook_score_influence: float = 0.35
    orderbook_override_score: float = 82.0

    # ── Timing ──────────────────────────────────────────
    check_interval_seconds: int  = 120    # Run cycle every 2 minutes
    candle_interval: str         = "1h"   # Candle size for indicator maths
    lookback_periods: int        = 100    # Candles to load each cycle
    reconcile_every_n_cycles: int = 10    # Reconcile exchange truth every N cycles in live mode

    # ── Live runtime guardrails ─────────────────────────
    require_ac_power_for_live: bool = True
    minimum_battery_pct_for_live: int = 35
    stop_live_on_power_loss: bool = True

    # ── Trade size limits ───────────────────────────────
    min_trade_usd: float         = 100.0   # Small-capital friendly minimum
    max_trade_usd: float         = 600.0   # Cap single-trade notional — rises with conviction

    # ── Dry-run (paper trading — no real orders sent) ───
    dry_run: bool                = True   # Explicit --live required for real orders

    # ── Universe management ──────────────────────────────
    # When enabled, any watchlist asset that the active venue can actually
    # execute is promoted into the live tradeable universe automatically.
    auto_promote_analysis_coins: bool = True

    # ── Multi-timeframe analysis ─────────────────────────
    # Fetches 4H and 12H candles to determine the higher-timeframe trend.
    # Blocks LONG trades when 4H+12H are both bearish, and vice versa.
    use_mtf: bool                = True

    # ── News sentiment ───────────────────────────────────
    # Fetches live news from CryptoPanic. Blocks trades when headlines
    # are strongly against the signal direction.
    # Especially important for HYPE (Hyperliquid protocol news).
    use_news: bool               = False
    cryptopanic_auth_token: str  = field(default_factory=lambda: os.getenv("CRYPTOPANIC_AUTH_TOKEN", ""))

    # ── Visual chart confirmation ────────────────────────
    # When enabled, borderline signals (score 38–62) are confirmed by
    # sending a chart screenshot to Claude's vision API before trading.
    # Strong signals (score <38 or >62) skip this to avoid extra API calls.
    use_chart_confirmation: bool = False   # Enable only when optional deps + API key are configured
    use_chart_screener: bool     = False   # True = auto-capture with playwright
    save_chart_screenshots: bool = False   # Save PNGs to screenshots/ folder
    chart_confirm_score_low: float  = 38.0 # Only check below this score (short zone)
    chart_confirm_score_high: float = 62.0 # Only check above this score (long zone)
    # Optional custom TradingView or Lighter chart URLs per coin
    # chart_urls: dict = {"BTC": "https://www.tradingview.com/chart/?symbol=BINANCE:BTCUSDT"}
    chart_urls: dict             = field(default_factory=dict)


# ─────────────────────────────────────────────────────────
# Indicator Parameters (for technical analysis)
# ─────────────────────────────────────────────────────────
@dataclass
class IndicatorConfig:
    # RSI
    rsi_period: int              = 14

    # MACD
    macd_fast: int               = 12
    macd_slow: int               = 26
    macd_signal: int             = 9

    # Bollinger Bands
    bb_period: int               = 20
    bb_std: float                = 2.0

    # EMA crossover
    ema_fast: int                = 9
    ema_slow: int                = 21

    # Volume: how many candles to average for volume baseline
    volume_ma_period: int        = 20

    # ── Advanced indicator parameters ───────────────────
    # Fibonacci lookback (candles)
    fib_lookback: int            = 60
    # MSB swing strength (candles each side for pivot detection)
    msb_strength: int            = 3
    msb_lookback: int            = 50
    # Order block parameters
    ob_lookback: int             = 40
    ob_impulse_candles: int      = 3
    # FVG lookback
    fvg_lookback: int            = 30
    # ATR period
    atr_period: int              = 14

    # ── Signal weights (MUST sum to 100) ─────────────
    # Classic technical + news + candles (43 pts total)
    weight_rsi: float            =  8.0
    weight_macd: float           =  8.0
    weight_bb: float             =  6.0
    weight_ema: float            =  6.0
    weight_sentiment: float      =  3.0   # Fear & Greed index
    weight_news: float           =  6.0   # Live news sentiment (NEW)
    weight_candles: float        =  6.0   # Candlestick patterns (NEW)
    # Advanced / structure-based (35 pts total)
    weight_fib: float            =  9.0
    weight_msb: float            = 15.0   # MSB is the most reliable structure signal
    weight_ob: float             =  7.0
    weight_fvg: float            =  4.0
    # Market regime signals (22 pts total)
    weight_regime_momentum: float   = 5.0   # Price acceleration (ROC)
    weight_regime_trend: float      = 5.0   # ADX directional strength
    weight_regime_mean_rev: float   = 4.0   # Z-score overextension
    weight_regime_vol_exp: float    = 3.0   # BB squeeze → breakout timing
    weight_regime_absorption: float = 3.0   # Wick / absorption analysis
    weight_regime_catalyst: float   = 2.0   # Volume anomaly + small move


# ─────────────────────────────────────────────────────────
# Optional Notifications (Telegram)
# ─────────────────────────────────────────────────────────
@dataclass
class NotificationConfig:
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str   = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    @property
    def enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


# ─────────────────────────────────────────────────────────
# Master Config Object
# ─────────────────────────────────────────────────────────
class Config:
    def __init__(self):
        self.exchange      = ExchangeConfig()
        self.trading       = TradingConfig()
        self.indicators    = IndicatorConfig()
        self.notifications = NotificationConfig()

    @property
    def is_dry_run(self) -> bool:
        return self.trading.dry_run

    def validate(self):
        """Raise early if critical settings are missing."""
        errors = []
        if self.exchange.use_hyperliquid and not self.exchange.hl_private_key:
            errors.append("HL_PRIVATE_KEY is missing — add it to your .env file")
        if self.exchange.use_hyperliquid and not self.exchange.hl_account_address:
            errors.append("HL_ACCOUNT_ADDRESS is missing — add it to your .env file")
        if self.exchange.use_lighter and not self.exchange.lighter_l1_private_key:
            errors.append("LIGHTER_L1_PRIVATE_KEY (or legacy LIGHTER_PRIVATE_KEY) is missing — add it to your .env file")
        if self.exchange.use_lighter and not self.exchange.lighter_api_private_key:
            errors.append("LIGHTER_API_PRIVATE_KEY is missing — run the Lighter bootstrap first")
        if self.exchange.use_lighter and not self.exchange.lighter_account_index:
            errors.append("LIGHTER_ACCOUNT_INDEX is missing — run the Lighter bootstrap first")
        if self.exchange.use_lighter and not self.exchange.lighter_web3_url:
            errors.append("LIGHTER_WEB3_URL is missing — add it to your .env file")
        ind = self.indicators
        weight_sum = (ind.weight_rsi + ind.weight_macd + ind.weight_bb +
                      ind.weight_ema + ind.weight_sentiment +
                      ind.weight_news + ind.weight_candles +
                      ind.weight_fib + ind.weight_msb +
                      ind.weight_ob + ind.weight_fvg +
                      ind.weight_regime_momentum + ind.weight_regime_trend +
                      ind.weight_regime_mean_rev + ind.weight_regime_vol_exp +
                      ind.weight_regime_absorption + ind.weight_regime_catalyst)
        if abs(weight_sum - 100.0) > 0.01:
            errors.append(f"Indicator weights must sum to 100 (got {weight_sum})")
        if errors:
            raise ValueError("Config errors:\n" + "\n".join(f"  • {e}" for e in errors))


# Singleton — import this everywhere
config = Config()
