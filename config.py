"""
config.py — Central configuration for the Crypto Trading Agent
All settings live here. Secrets are loaded from your .env file.
"""

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

from exchanges.hyperliquid_markets import TRADEXYZ_ASSET_METADATA

load_dotenv()


_BASE_EXECUTION_COINS = ["BTC", "ETH", "SOL", "HYPE", "MON", "TAO", "SP500", "XAU"]

_BASE_INSTRUMENT_TYPES = {
    "BTC": "crypto",
    "ETH": "crypto",
    "SOL": "crypto",
    "HYPE": "crypto",
    "MON": "crypto",
    "TAO": "crypto",
    "SP500": "index",
    "XAU": "index",
    "BRENT": "index",
    "WTI": "index",
    "CL": "index",
}

_BASE_ASSET_CATEGORY_MAP = {
    "BTC": ["crypto"],
    "ETH": ["crypto"],
    "SOL": ["crypto"],
    "HYPE": ["crypto"],
    "MON": ["crypto"],
    "TAO": ["crypto"],
    "SP500": ["indices_macro"],
    "XAU": ["indices_macro"],
}

_ASSET_CATEGORY_LABELS = {
    "crypto": "Coins",
    "indices_macro": "Indices & Macro",
    "mag7": "Mag7",
    "semis_memory": "Semis & Memory",
    "neoclouds": "Neoclouds",
    "ai_infra": "AI Infra",
    "crypto_equities": "Crypto Equities",
    "asia_macro": "Asia Macro",
    "commodities_metals": "Metals",
    "energy": "Energy",
    "agriculture": "Agriculture",
    "fx_rates": "FX & Rates",
    "uranium": "Uranium",
    "volatility": "Volatility",
    "consumer": "Consumer",
    "financials": "Financials",
    "biotech_glp1": "Biotech & GLP-1",
    "meme_momentum": "Meme Momentum",
    "growth": "Growth",
    "other_stocks": "Other Stocks",
}

_THEME_BY_CATEGORY = {
    "crypto": "CRYPTO_BETA",
    "indices_macro": "US_MACRO_BETA",
    "mag7": "MEGA_CAP_TECH",
    "semis_memory": "SEMIS_MEMORY",
    "neoclouds": "NEOCLOUDS",
    "ai_infra": "AI_INFRA",
    "crypto_equities": "CRYPTO_EQUITIES",
    "asia_macro": "ASIA_MACRO",
    "commodities_metals": "COMMODITIES_METALS",
    "energy": "ENERGY_COMPLEX",
    "agriculture": "AGRICULTURE",
    "fx_rates": "FX_RATES",
    "uranium": "URANIUM",
    "volatility": "VOLATILITY",
    "consumer": "CONSUMER_GROWTH",
    "financials": "FINANCIALS",
    "biotech_glp1": "BIOTECH_GLP1",
    "meme_momentum": "MEME_MOMENTUM",
    "growth": "US_GROWTH",
    "other_stocks": "OTHER_STOCKS",
}


def _unique_coins(*groups) -> List[str]:
    seen = set()
    out: List[str] = []
    for values in groups:
        for value in values or []:
            coin = str(value or "").upper().strip()
            if coin and coin not in seen:
                seen.add(coin)
                out.append(coin)
    return out


def _default_analysis_coins() -> List[str]:
    return _unique_coins(_BASE_EXECUTION_COINS, TRADEXYZ_ASSET_METADATA.keys())


def _default_execution_coins() -> List[str]:
    return _default_analysis_coins()


def _default_instrument_types() -> dict:
    instrument_types = dict(_BASE_INSTRUMENT_TYPES)
    for coin, meta in TRADEXYZ_ASSET_METADATA.items():
        instrument_types[str(coin).upper()] = str(meta.get("instrument_type") or "equity").strip().lower()
    return instrument_types


def _default_asset_category_map() -> dict:
    category_map = {coin: list(categories) for coin, categories in _BASE_ASSET_CATEGORY_MAP.items()}
    for coin, meta in TRADEXYZ_ASSET_METADATA.items():
        categories = [
            str(category or "").strip().lower()
            for category in list(meta.get("categories") or ["other_stocks"])
            if str(category or "").strip()
        ]
        category_map[str(coin).upper()] = categories or ["other_stocks"]
    return category_map


def _default_portfolio_theme_map() -> dict:
    theme_map = {
        "BTC": "CRYPTO_CORE",
        "ETH": "CRYPTO_CORE",
        "SOL": "CRYPTO_HIGH_BETA",
        "HYPE": "CRYPTO_HIGH_BETA",
        "MON": "CRYPTO_HIGH_BETA",
        "TAO": "CRYPTO_HIGH_BETA",
        "SP500": "US_MACRO_BETA",
        "XAU": "DEFENSIVE_HARD_ASSET",
        "BRENT": "ENERGY_COMPLEX",
        "WTI": "ENERGY_COMPLEX",
    }
    for coin, categories in _default_asset_category_map().items():
        primary_category = str((categories or ["other_stocks"])[0] or "other_stocks")
        theme_map.setdefault(coin, _THEME_BY_CATEGORY.get(primary_category, primary_category.upper()))
    return theme_map


# ─────────────────────────────────────────────────────────
# Exchange Credentials  (loaded from .env — never hardcode!)
# ─────────────────────────────────────────────────────────
@dataclass
class ExchangeConfig:
    # Hyperliquid  ─ uses an EVM private key
    hl_private_key: str        = field(default_factory=lambda: os.getenv("HL_PRIVATE_KEY", ""))
    hl_account_address: str    = field(default_factory=lambda: os.getenv("HL_ACCOUNT_ADDRESS", ""))
    hl_use_mainnet: bool       = True          # set False for testnet
    hl_spot_execution_enabled: bool = field(
        default_factory=lambda: os.getenv("HL_SPOT_EXECUTION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    )

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
    use_hyperliquid: bool      = True
    use_lighter: bool          = False


# ─────────────────────────────────────────────────────────
# Trading Parameters
# ─────────────────────────────────────────────────────────
@dataclass
class TradingConfig:
    # Coins / instruments the agent will trade
    # Baseline execution universe. Additional watchlist symbols that are
    # supported on the active venue are auto-promoted at startup.
    coins: List[str]            = field(default_factory=_default_execution_coins)

    # Broader watchlist / learning universe.
    # TradeXYZ-backed symbols are analysed by default and auto-promoted into
    # the executable universe whenever the connected venue exposes them.
    analysis_coins: List[str]   = field(default_factory=_default_analysis_coins)

    # ── Instrument type classification ───────────────────────────────────────
    # "crypto"  → standard crypto perp logic
    # "index"   → macro / non-crypto instrument (SP500, BRENT, WTI, etc.) —
    #             Yahoo-backed market data, macro/news-driven, smoother momentum,
    #             higher minimum hold time
    instrument_types: dict      = field(default_factory=_default_instrument_types)

    # Indexes need a longer minimum hold — they move slower than crypto
    index_min_hold_minutes: float = 360.0   # 6h for indexes (vs 4h for crypto)
    equity_min_hold_minutes: float = 240.0  # 4h for spot equities

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
    max_flat_cycles_with_position: int = 3  # Legacy input into conviction-decay logic, not a hard close alone

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
    expectancy_min_probability: float = 0.54
    expectancy_min_expected_r: float = 0.18
    expectancy_max_uncertainty: float = 0.42
    expectancy_min_score: float = 56.0
    expectancy_same_direction_min_score: float = 52.0
    expectancy_orderbook_bonus: float = 0.05
    expectancy_market_map_bonus: float = 0.04
    expectancy_news_bonus: float = 0.04
    trailing_stop_enabled: bool   = True
    trailing_stop_pct: float      = 0.12   # 12% trailing stop (locks profits as price runs)
    use_orderbook_levels: bool    = True   # Live L2 + daily key-level intelligence for venue-backed assets
    orderbook_depth_limit: int    = 120
    orderbook_daily_lookback: int = 120
    orderbook_cache_ttl_seconds: int = 25
    orderbook_feed_enabled: bool = True
    orderbook_feed_poll_seconds: float = 3.0
    orderbook_feed_history_size: int = 120
    orderbook_feed_max_snapshot_age_seconds: float = 45.0
    orderbook_feed_breakout_samples: int = 2
    orderbook_guard_distance_pct: float = 1.25
    orderbook_reaction_distance_pct: float = 0.45
    orderbook_level_min_strength: float = 0.55
    orderbook_score_influence: float = 0.35
    orderbook_override_score: float = 82.0
    require_orderbook_for_crypto_entries: bool = True
    require_orderbook_for_supported_entries: bool = True
    require_execution_quality: bool = True
    max_execution_spread_bps: float = 12.0
    min_execution_depth_multiple: float = 10.0
    max_execution_slippage_bps: float = 18.0
    min_orderbook_persistence_cycles: int = 2
    execution_planning_enabled: bool = True
    execution_limit_retest_distance_pct: float = 0.40
    execution_passive_entry_offset_bps: float = 4.0
    execution_breakout_market_probability: float = 0.64
    execution_breakout_market_expectancy_score: float = 64.0
    execution_limit_timeout_cycles: int = 6
    execution_pending_management_enabled: bool = True
    execution_pending_cancel_on_stop_breach: bool = True
    execution_pending_cancel_on_opposite_breakout: bool = True
    execution_pending_reprice_enabled: bool = True
    execution_pending_reprice_after_cycles: int = 2
    execution_pending_reprice_threshold_bps: float = 8.0
    execution_pending_max_reprices: int = 3
    execution_pending_market_escalation_enabled: bool = True
    execution_pending_market_escalation_after_cycles: int = 3
    execution_pending_market_escalation_breakout_only: bool = True
    execution_pending_market_escalation_max_spread_bps: float = 10.0
    execution_pending_market_escalation_max_slippage_bps: float = 16.0
    execution_pending_market_escalation_min_quality_score: float = 72.0

    # ── Timing ──────────────────────────────────────────
    check_interval_seconds: int  = 120    # Run cycle every 2 minutes
    candle_interval: str         = "1h"   # Candle size for indicator maths
    lookback_periods: int        = 100    # Candles to load each cycle
    use_closed_candles_for_conviction: bool = True  # Build conviction from completed candles, not the live bar
    reconcile_every_n_cycles: int = 10    # Reconcile exchange truth every N cycles in live mode

    # ── Thesis / abstention guardrails ──────────────────
    # The raw score may suggest direction, but the thesis layer decides whether
    # the setup is actually good enough to deserve a trade.
    thesis_min_alignment_points: int = 4
    thesis_max_conflict_points: int = 1
    thesis_min_score_buffer: float = 3.0
    thesis_min_risk_reward_ratio: float = 1.75
    thesis_block_on_range_conditions: bool = True
    strict_confirmation_fail_closed: bool = True
    support_defense_long_enabled: bool = True
    support_defense_long_score_floor: float = 24.0
    support_defense_max_support_distance_pct: float = 0.40
    support_defense_min_orderbook_score: float = 62.0
    support_defense_min_imbalance: float = 0.10
    support_defense_breakout_score_floor: float = 36.0
    support_defense_map_override_enabled: bool = True
    support_defense_expectancy_bonus: float = 0.05

    # ── Early conviction entries ─────────────────────────
    # Lets the agent take smaller, thesis-backed starter positions before the
    # full reclaim / breakout confirmation is obvious, instead of missing the
    # move while waiting for precision mode perfection.
    conviction_entry_enabled: bool = True
    conviction_entry_score_buffer: float = 6.0
    conviction_entry_min_news_score: float = 58.0
    conviction_entry_min_catalyst_score: float = 3.0
    conviction_entry_min_alignment_points: float = 3.0
    conviction_entry_max_conflict_points: float = 1.5
    conviction_entry_min_probability: float = 0.56
    conviction_entry_min_expectancy_score: float = 54.0
    conviction_entry_max_uncertainty: float = 0.46
    conviction_entry_size_multiplier: float = 0.45
    conviction_entry_precision_override_enabled: bool = True
    conviction_entry_bypass_signal_streak: bool = True
    conviction_entry_bypass_precision_cadence: bool = True
    conviction_entry_event_score_buffer: float = 22.0
    conviction_entry_event_min_news_score: float = 60.0
    conviction_entry_event_min_catalyst_score: float = 4.0
    conviction_entry_event_min_alignment_points: float = 1.0
    conviction_entry_event_max_conflict_points: float = 3.75
    conviction_entry_event_min_probability: float = 0.51
    conviction_entry_event_min_expectancy_score: float = 46.0
    conviction_entry_event_max_uncertainty: float = 0.62
    conviction_entry_event_size_multiplier: float = 0.30
    conviction_entry_event_max_size_multiplier: float = 0.46

    # ── Official event intelligence feeds ────────────────────
    official_event_feed_enabled: bool = True
    official_event_feed_cache_seconds: int = 1800
    official_ir_calendar_sync_enabled: bool = True
    sec_filing_feed_enabled: bool = True
    sec_filing_feed_lookback_days: int = 21
    options_implied_move_feed_enabled: bool = True
    analyst_revision_feed_enabled: bool = True

    # ── Thesis invalidation ladder ───────────────────────
    early_invalidation_minutes: float = 90.0
    early_invalidation_adverse_r: float = 0.55
    htf_invalidation_min_minutes: float = 60.0
    time_stop_minutes: float = 360.0
    time_stop_min_tp_progress: float = 0.25
    conviction_decay_exit_threshold: float = 58.0
    conviction_decay_hold_threshold: float = 36.0
    conviction_decay_flat_cycle_weight: float = 7.0
    conviction_decay_microstructure_weight: float = 14.0
    conviction_decay_structure_weight: float = 12.0
    conviction_decay_expectancy_weight: float = 16.0
    use_daily_market_map: bool = True
    market_map_guard_distance_pct: float = 1.10
    market_map_score_influence: float = 1.00
    market_map_countertrend_penalty: float = 6.0
    market_map_alignment_boost: float = 4.0

    # ── Live runtime guardrails ─────────────────────────
    require_ac_power_for_live: bool = True
    minimum_battery_pct_for_live: int = 35
    stop_live_on_power_loss: bool = True
    require_notifications_for_live: bool = True
    live_promotion_gate_enabled: bool = True
    live_promotion_lookback_closed_trades: int = 40
    live_promotion_min_closed_trades: int = 20
    live_promotion_min_win_rate: float = 0.58
    live_promotion_min_avg_pnl_pct: float = 0.10
    live_promotion_min_profit_factor: float = 1.15
    live_promotion_min_precision_samples: int = 6
    live_promotion_min_precision_win_rate: float = 0.60
    live_promotion_report_max_age_hours: float = 12.0
    live_promotion_precision_target_r: float = 0.25
    live_promotion_precision_horizon_minutes: int = 720
    live_promotion_precision_interval: str = "5m"
    live_promotion_precision_dedupe_minutes: int = 30

    # ── Trade size limits ───────────────────────────────
    min_trade_usd: float         = 100.0   # Small-capital friendly minimum
    max_trade_usd: float         = 600.0   # Cap single-trade notional — rises with conviction

    # ── Dry-run (paper trading — no real orders sent) ───
    dry_run: bool                = True   # Explicit --live required for real orders

    # ── Universe management ──────────────────────────────
    # When enabled, any watchlist asset that the active venue can actually
    # execute is promoted into the live tradeable universe automatically.
    auto_promote_analysis_coins: bool = True
    auto_promote_supported_stocks: bool = True
    enforce_active_venue_markets: bool = True
    dynamic_analysis_coins: List[str] = field(default_factory=list)
    dynamic_analysis_auto_promote: bool = True
    promote_analysis_before_activity: bool = True
    dynamic_market_cap_watchlist_enabled: bool = True
    dynamic_market_cap_min_usd: float = 1_000_000_000.0
    dynamic_market_cap_pages: int = 3
    dynamic_market_cap_cache_hours: float = 6.0
    dynamic_market_cap_active_only: bool = True
    dynamic_market_cap_max_coins: int = 60
    dynamic_market_cap_feed_limit: int = 16

    # ── Multi-timeframe analysis ─────────────────────────
    # Fetches 4H and 12H candles to determine the higher-timeframe trend.
    # Blocks LONG trades when 4H+12H are both bearish, and vice versa.
    use_mtf: bool                = True

    # ── News sentiment ───────────────────────────────────
    # Fetches live news from CryptoPanic. Blocks trades when headlines
    # are strongly against the signal direction.
    # Especially important for HYPE (Hyperliquid protocol news).
    use_news: bool               = True
    cryptopanic_auth_token: str  = field(default_factory=lambda: os.getenv("CRYPTOPANIC_AUTH_TOKEN", ""))
    use_narrative_gate: bool     = True
    narrative_event_risk_window_minutes: int = 90
    narrative_post_event_cooldown_minutes: int = 45
    narrative_event_block_min_expectancy_score: float = 72.0
    narrative_event_block_min_probability: float = 0.60
    narrative_headline_alignment_bonus: float = 0.06
    narrative_headline_conflict_penalty: float = 0.08
    narrative_event_uncertainty_add: float = 0.14
    narrative_high_impact_keywords: List[str] = field(default_factory=lambda: [
        "cpi", "pce", "nfp", "nonfarm payrolls", "fomc", "powell",
        "fed", "ecb", "boj", "rate decision", "inflation", "gdp",
    ])

    # ── Decision intelligence / AI prep ───────────────────────
    decision_dataset_enabled: bool = True
    feature_store_enabled: bool = True
    analog_engine_enabled: bool = True
    analog_history_limit: int = 1500
    analog_min_samples: int = 5
    analog_hard_block_min_samples: int = 8
    analog_similarity_floor: float = 0.58
    analog_min_reliability: float = 0.42
    analog_same_coin_bonus: float = 0.08
    analog_same_instrument_bonus: float = 0.04
    analog_max_examples: int = 5
    analog_supportive_win_rate: float = 0.57
    analog_adverse_win_rate: float = 0.43
    analog_hard_block_win_rate: float = 0.35
    analog_positive_expected_r: float = 0.18
    analog_negative_expected_r: float = -0.10
    analog_score_adjustment_cap: float = 4.0
    analog_probability_adjustment_cap: float = 0.06
    analog_expected_r_adjustment_cap: float = 0.12
    analog_uncertainty_adjustment_cap: float = 0.08

    # ── Precision mode: trade less, but only on elite setups ───────────────
    # This layer exists to optimize top-trade precision, not activity.
    precision_mode_enabled: bool = True
    precision_min_confidence: str = "HIGH"
    precision_min_thesis_quality: str = "HIGH"
    precision_min_expectancy_probability: float = 0.92
    precision_min_expected_r: float = 0.35
    precision_max_uncertainty: float = 0.20
    precision_min_risk_reward_ratio: float = 2.10
    precision_require_confirmed_breakout: bool = True
    precision_allow_support_defense_longs: bool = True
    precision_require_market_map_alignment: bool = False
    precision_min_long_orderbook_score: float = 68.0
    precision_max_short_orderbook_score: float = 38.0
    precision_min_analog_samples: int = 4
    precision_min_analog_reliability: float = 0.55
    precision_min_analog_win_rate: float = 0.65
    precision_same_family_cooldown_minutes: int = 720
    precision_same_coin_cooldown_minutes: int = 360
    precision_max_new_entries_per_day: int = 2
    precision_coin_direction_embargoes: List[str] = field(default_factory=lambda: [
        "SP500:SHORT",
        "TAO:LONG",
    ])

    # ── Asset state machine / next-unblock reasoning ───────────────────────
    asset_state_machine_enabled: bool = True

    # ── Data reliability guard ──────────────────────────────────────────────
    data_reliability_enabled: bool = True
    data_reliability_min_score: float = 58.0
    data_reliability_max_live_analysis_gap_pct: float = 0.90
    data_reliability_min_news_articles: int = 1
    data_reliability_min_orderbook_snapshots: int = 3
    data_reliability_max_reference_deviation_pct: float = 2.0

    # ── Portfolio correlation guard ─────────────────────────────────────────
    portfolio_correlation_guard_enabled: bool = True
    portfolio_theme_max_positions: int = 2
    portfolio_theme_event_starter_extra_slots: int = 2
    portfolio_theme_max_same_direction_exposure_pct: float = 0.18
    portfolio_theme_warning_exposure_pct: float = 0.10
    portfolio_correlation_soft_penalty: float = 0.65
    portfolio_correlation_secondary_penalty: float = 0.82
    portfolio_correlation_event_starter_extra_penalty: float = 0.40
    portfolio_theme_map: dict = field(default_factory=_default_portfolio_theme_map)
    event_risk_budget_enabled: bool = True
    event_risk_budget_max_portfolio_pct: float = 0.10
    event_risk_budget_max_theme_pct: float = 0.08
    event_risk_budget_max_single_pct: float = 0.02
    event_risk_budget_soft_penalty_pct: float = 0.65
    event_risk_budget_min_trade_usd: float = 100.0
    event_risk_budget_strict_caps: bool = True

    asset_category_map: dict = field(default_factory=_default_asset_category_map)
    asset_category_labels: dict = field(default_factory=lambda: dict(_ASSET_CATEGORY_LABELS))

    # ── Smarter execution tactics ───────────────────────────────────────────
    execution_passive_rescue_enabled: bool = True
    execution_passive_rescue_max_spread_bps: float = 28.0
    execution_passive_rescue_min_depth_multiple: float = 2.5
    execution_passive_rescue_max_slippage_bps: float = 85.0
    execution_coach_enabled: bool = True
    execution_coach_aggressive_max_stretch_bps: float = 10.0
    execution_coach_passive_hold_distance_bps: float = 18.0
    execution_coach_max_chase_bps: float = 32.0
    execution_coach_skip_stretch_bps: float = 48.0
    execution_coach_min_quality_score: float = 74.0
    execution_coach_min_breakout_probability: float = 0.66
    execution_coach_min_breakout_expectancy_score: float = 66.0
    execution_coach_reprice_passive_orders: bool = True

    # ── Missed-trade review / champion-challenger scaffolding ──────────────
    decision_review_enabled: bool = True
    decision_review_target_r: float = 0.25
    decision_review_horizon_minutes: int = 720
    decision_review_interval: str = "5m"
    decision_review_dedupe_minutes: int = 30
    challenger_model_enabled: bool = True
    challenger_min_labeled_decisions: int = 25
    challenger_min_win_rate_edge: float = 0.04
    challenger_refresh_hours: float = 6.0
    asset_dossier_enabled: bool = True
    asset_dossier_refresh_hours: float = 6.0
    missed_move_lab_enabled: bool = True
    llm_referee_enabled: bool = field(
        default_factory=lambda: os.getenv("OPENAI_REFEREE_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    )
    llm_referee_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_referee_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    llm_referee_model: str = field(default_factory=lambda: os.getenv("OPENAI_REFEREE_MODEL", "gpt-5.4"))
    llm_referee_timeout_seconds: float = 18.0
    llm_referee_cache_minutes: int = 45
    llm_referee_max_setups_per_cycle: int = 2
    llm_referee_min_expectancy_probability: float = 0.56
    llm_referee_min_score_distance: float = 10.0
    llm_referee_block_on_verdicts: List[str] = field(default_factory=lambda: ["BLOCK"])
    llm_referee_review_on_asset_states: List[str] = field(default_factory=lambda: [
        "ARMED",
        "WAITING_CONFIRMATION",
        "EXECUTABLE",
        "PASSIVE_ENTRY",
        "READY_LONG",
        "READY_SHORT",
    ])
    playbook_distiller_enabled: bool = True
    playbook_distiller_refresh_hours: float = 24.0
    playbook_distiller_lookback_days: int = 28
    playbook_distiller_min_samples: int = 3
    playbook_distiller_min_win_rate: float = 0.55
    playbook_distiller_max_losing_win_rate: float = 0.45
    proactive_trader_enabled: bool = True
    thesis_ledger_enabled: bool = True
    morning_scout_book_enabled: bool = True
    read_through_engine_enabled: bool = True
    starter_basket_optimizer_enabled: bool = True
    forecast_calibration_enabled: bool = True
    morning_scout_max_names: int = 14
    proactive_starter_basket_max_names: int = 6
    proactive_starter_min_conviction: float = 58.0
    proactive_forecast_horizon_hours: float = 24.0
    proactive_forecast_max_open: int = 60
    proactive_report_max_theses: int = 24
    proactive_starter_execution_enabled: bool = field(
        default_factory=lambda: os.getenv("PROACTIVE_STARTER_EXECUTION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    )
    proactive_starter_execution_max_per_cycle: int = 3
    proactive_starter_execution_min_score: float = 60.0
    proactive_starter_execution_cooldown_minutes: float = 240.0
    proactive_starter_execution_hard_block_stages: List[str] = field(default_factory=lambda: [
        "data_reliability_block",
        "execution_coach_skip",
        "llm_referee_block",
        "long_only_short_block",
        "loss_circuit_breaker",
        "portfolio_correlation_block",
    ])

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
    backtest_market_slippage_bps: float = 4.0


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
        if (
            not self.trading.dry_run
            and self.trading.require_notifications_for_live
            and not self.notifications.enabled
        ):
            errors.append("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required for live trading")
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
