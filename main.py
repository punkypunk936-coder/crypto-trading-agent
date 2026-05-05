"""
main.py — Entry point for the Crypto Trading Agent.

Run modes:
  python main.py              → paper trading by default
  python main.py --live       → live trading (requires credentials in .env)
  python main.py --dry-run    → paper trading (safe, no real orders)
  python main.py --status     → show account state and exit
  python main.py --help       → show this help
"""

import sys
import argparse
import importlib.util
import os
import subprocess
import tempfile

import checkpoint as checkpoint_module
from config import config
from logger import log
from promotion_gate import evaluate_live_promotion
from exchanges.hyperliquid_client import HyperliquidClient
from exchanges.hyperliquid_markets import (
    get_hyperliquid_market_catalog,
    hyperliquid_instrument_type,
    get_hyperliquid_market_activity,
    get_hyperliquid_supported_coins,
    hyperliquid_market_is_active,
    hyperliquid_supports_shorts,
)
from exchanges.lighter_client     import LighterClient, bootstrap_lighter_api
from exchanges.dry_run            import DryRunExchange
from agent import TradingAgent
from data.market_data import fetch_candles, get_current_price
from market_universe import build_hyperliquid_market_cap_watchlist
from runtime_power import get_power_status
from paths import (
    CHECKPOINTS_DB,
    CONTROL_JSON,
    DATA_DIR,
    KILL_FILE,
    STATE_JSON,
    TRADE_MEMORY,
    TRADES_CSV,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crypto Trading Agent — Hyperliquid-first",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dry-run           Test the strategy with paper money
  python main.py --live              Run live (reads .env for credentials)
  python main.py --status            Show portfolio and exit
  python main.py --coins BTC ETH     Override coins to trade
        """
    )
    parser.add_argument("--dry-run",  action="store_true",
                        help="Paper trading mode — no real orders sent")
    parser.add_argument("--live", action="store_true",
                        help="Explicitly enable live trading")
    parser.add_argument("--status",   action="store_true",
                        help="Print account status and exit")
    parser.add_argument("--preflight", action="store_true",
                        help="Validate runtime, configuration, market-data, and exchange readiness")
    parser.add_argument("--paper-reset", action="store_true",
                        help="Clear dry-run state files for a fresh paper-trading run")
    parser.add_argument("--lighter-bootstrap", action="store_true",
                        help="Create/register a Lighter API key for the configured wallet")
    parser.add_argument("--coins",    nargs="+",
                        help="Override coins to trade, e.g. --coins BTC ETH")
    parser.add_argument("--interval", type=int,
                        help="Override check interval in seconds")
    parser.add_argument("--balance",  type=float, default=10_000.0,
                        help="Starting balance for dry-run mode (default: 10000)")
    return parser.parse_args()


def configured_supported_coins(*, dry_run_mode: bool | None = None) -> list[str]:
    supported = set()
    if config.exchange.use_hyperliquid:
        include_spot = (
            bool(dry_run_mode)
            or bool(getattr(config.exchange, "hl_spot_execution_enabled", False))
        )
        supported.update(
            get_hyperliquid_supported_coins(
                include_spot=include_spot,
                live_tradeable_only=not include_spot,
                # Discover the broad venue universe here; activity is checked
                # later only for the symbols we actually care about.
                active_only=False,
            )
        )
    if config.exchange.use_lighter:
        supported.update(LighterClient.supported_market_symbols())
    return sorted(supported)


def _normalise_coin_list(values) -> list[str]:
    seen = set()
    out = []
    for coin in values or []:
        coin_upper = str(coin).upper()
        if coin_upper and coin_upper not in seen:
            seen.add(coin_upper)
            out.append(coin_upper)
    return out


def _theme_from_categories(categories: list[str], instrument_type: str) -> str:
    primary = str((categories or [])[0] if categories else "").strip().lower()
    theme_by_category = {
        "crypto": "CRYPTO_BETA",
        "indices_macro": "US_MACRO_BETA",
        "pre_ipo": "PRE_IPO_EVENT",
        "mag7": "MEGA_CAP_TECH",
        "semis_memory": "SEMIS_MEMORY",
        "neoclouds": "NEOCLOUDS",
        "ai_infra": "AI_INFRA",
        "crypto_equities": "CRYPTO_EQUITIES",
        "asia_macro": "ASIA_MACRO",
        "latam_macro": "LATAM_MACRO",
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
        "software": "SOFTWARE_GROWTH",
        "other_stocks": "OTHER_STOCKS",
    }
    if primary in theme_by_category:
        return theme_by_category[primary]
    if str(instrument_type or "").lower() == "crypto":
        return "CRYPTO_BETA"
    if str(instrument_type or "").lower() == "index":
        return "US_MACRO_BETA"
    return (primary or "OTHER_STOCKS").upper()


def _sync_config_market_metadata(coin: str, spec: dict | None = None) -> None:
    coin_upper = str(coin or "").upper().strip()
    if not coin_upper:
        return
    spec = dict(spec or get_hyperliquid_market_catalog().get(coin_upper) or {})
    instrument_type = str(
        spec.get("instrument_type")
        or config.trading.instrument_types.get(coin_upper)
        or hyperliquid_instrument_type(coin_upper, "crypto")
    ).strip().lower() or "crypto"
    raw_categories = spec.get("categories")
    if isinstance(raw_categories, str):
        raw_categories = [raw_categories]
    categories = [
        str(category or "").strip().lower()
        for category in list(raw_categories or [])
        if str(category or "").strip()
    ]
    if not categories:
        existing = (getattr(config.trading, "asset_category_map", {}) or {}).get(coin_upper, [])
        if isinstance(existing, str):
            existing = [existing]
        categories = [
            str(category or "").strip().lower()
            for category in list(existing or [])
            if str(category or "").strip()
        ]
    if not categories:
        if instrument_type == "crypto":
            categories = ["crypto"]
        elif instrument_type == "index":
            categories = ["indices_macro"]
        else:
            categories = ["other_stocks"]

    config.trading.instrument_types[coin_upper] = instrument_type
    config.trading.asset_category_map[coin_upper] = categories
    config.trading.portfolio_theme_map.setdefault(
        coin_upper,
        _theme_from_categories(categories, instrument_type),
    )


def _sync_supported_stock_universe(*, dry_run_mode: bool | None = None) -> list[str]:
    if not config.exchange.use_hyperliquid or not getattr(config.trading, "auto_promote_supported_stocks", True):
        return []

    supported = configured_supported_coins(dry_run_mode=dry_run_mode)
    if not supported:
        return []

    catalog = get_hyperliquid_market_catalog()
    promoted: list[str] = []
    for coin in supported:
        spec = dict(catalog.get(str(coin).upper()) or {})
        instrument_type = str(
            spec.get("instrument_type")
            or config.trading.instrument_types.get(str(coin).upper(), "")
            or hyperliquid_instrument_type(coin, "crypto")
        ).strip().lower()
        if instrument_type not in {"equity", "index"}:
            continue
        coin_upper = str(coin).upper()
        _sync_config_market_metadata(coin_upper, spec)
        promoted.append(coin_upper)
    return _normalise_coin_list(promoted)


def apply_dynamic_analysis_universe() -> list[str]:
    previous_dynamic = {str(coin).upper() for coin in getattr(config.trading, "dynamic_analysis_coins", []) or []}
    base_analysis = [
        coin for coin in _normalise_coin_list(getattr(config.trading, "analysis_coins", []) or [])
        if coin not in previous_dynamic
    ]
    dynamic_analysis: list[str] = []
    if config.exchange.use_hyperliquid and getattr(config.trading, "dynamic_market_cap_watchlist_enabled", False):
        payload = build_hyperliquid_market_cap_watchlist(
            min_market_cap_usd=float(getattr(config.trading, "dynamic_market_cap_min_usd", 1_000_000_000.0) or 1_000_000_000.0),
            pages=int(getattr(config.trading, "dynamic_market_cap_pages", 3) or 3),
            cache_hours=float(getattr(config.trading, "dynamic_market_cap_cache_hours", 6.0) or 6.0),
            active_only=bool(getattr(config.trading, "dynamic_market_cap_active_only", True)),
            max_coins=int(getattr(config.trading, "dynamic_market_cap_max_coins", 60) or 60),
        )
        dynamic_analysis = _normalise_coin_list(payload.get("coins", []) or [])
        if dynamic_analysis:
            log.info(
                "Expanded Hyperliquid scout universe (>$%s market cap): %s",
                f"{int(float(getattr(config.trading, 'dynamic_market_cap_min_usd', 1_000_000_000.0))):,}",
                ", ".join(dynamic_analysis),
            )
        for coin in dynamic_analysis:
            _sync_config_market_metadata(coin)

    supported_stock_watchlist = _sync_supported_stock_universe(dry_run_mode=config.trading.dry_run)

    config.trading.dynamic_analysis_coins = dynamic_analysis
    merged = list(base_analysis)
    for coin in supported_stock_watchlist:
        if coin not in merged:
            merged.append(coin)
    for coin in dynamic_analysis:
        if coin not in merged:
            merged.append(coin)
    config.trading.analysis_coins = merged
    return dynamic_analysis


def enforce_trade_universe(
    exchanges: list | None = None,
    *,
    supported_override: set[str] | list[str] | None = None,
) -> list[str]:
    if supported_override is not None:
        supported = {str(coin).upper() for coin in supported_override if coin}
    elif exchanges:
        supported = set()
        for ex in exchanges:
            supported.update(ex.supported_coins())
    else:
        supported = set(configured_supported_coins(dry_run_mode=config.trading.dry_run))

    if not supported:
        raise ValueError("No supported trading symbols are available for the configured venues")

    enforce_active = bool(getattr(config.trading, "enforce_active_venue_markets", True))
    promote_before_activity = bool(getattr(config.trading, "promote_analysis_before_activity", True))

    def _is_active_for_execution(coin: str) -> bool:
        if not enforce_active:
            return True
        if config.exchange.use_hyperliquid and coin in supported:
            return hyperliquid_market_is_active(coin)
        return True

    active = _normalise_coin_list(config.trading.coins)
    unsupported = sorted(set(active) - supported)
    if unsupported:
        raise ValueError(
            "Configured symbols are not tradable on the active venue(s): "
            + ", ".join(unsupported)
        )
    inactive_configured = [coin for coin in active if coin in supported and not _is_active_for_execution(coin)]
    if inactive_configured:
        if promote_before_activity:
            log.info(
                "Arming configured symbols for execution even while the venue is still warming up: "
                + ", ".join(inactive_configured)
            )
        else:
            log.info(
                "Configured symbols stay observation-only until the venue prints fresh activity: "
                + ", ".join(inactive_configured)
            )

    promoted = []
    armed_pending_activity = []
    if getattr(config.trading, "auto_promote_analysis_coins", False):
        analysis = _normalise_coin_list(getattr(config.trading, "analysis_coins", []) or [])
        if not getattr(config.trading, "dynamic_analysis_auto_promote", False):
            dynamic_set = {str(coin).upper() for coin in getattr(config.trading, "dynamic_analysis_coins", []) or []}
            analysis = [coin for coin in analysis if coin not in dynamic_set]
        for coin in analysis:
            if coin in supported and coin not in active and (_is_active_for_execution(coin) or promote_before_activity):
                promoted.append(coin)
                if not _is_active_for_execution(coin):
                    armed_pending_activity.append(coin)
        if promoted:
            log.info(
                "Promoting watchlist symbols into the tradeable universe on active venue(s): "
                + ", ".join(promoted)
            )
        if armed_pending_activity:
            log.info(
                "These promoted symbols are executable immediately, but the venue is still cold so the live data/execution gates must still clear: "
                + ", ".join(armed_pending_activity)
            )
        deferred = [
            coin for coin in analysis
            if coin not in active and coin not in supported
        ]
        if deferred:
            log.info(
                "Keeping watchlist symbols in observation mode until a connected venue supports them: "
                + ", ".join(deferred)
            )

    config.trading.coins = [
        coin for coin in active
        if coin in supported and (_is_active_for_execution(coin) or promote_before_activity)
    ] + promoted
    return config.trading.coins


def build_exchanges(args) -> list:
    """Construct and connect exchange clients based on config and args."""
    exchanges = []
    apply_dynamic_analysis_universe()

    dry_run = args.dry_run or config.trading.dry_run
    if dry_run:
        supported_symbols = configured_supported_coins(dry_run_mode=True) or ["BTC", "ETH", "SOL"]
        shortable_map = {coin: hyperliquid_supports_shorts(coin) for coin in supported_symbols}
        config.trading.coins = [coin.upper() for coin in config.trading.coins]
        enforce_trade_universe(supported_override=set(supported_symbols))
        log.info("🟡  DRY RUN mode — using paper trading exchange")
        ex = DryRunExchange(
            starting_balance_usd=args.balance,
            supported_symbols=supported_symbols,
            shortable_map=shortable_map,
        )
        ex.connect()
        exchanges.append(ex)
        return exchanges

    # ── Hyperliquid ───────────────────────────────────────
    if config.exchange.use_hyperliquid:
        if not config.exchange.hl_private_key:
            log.error("HL_PRIVATE_KEY is empty. Add it to .env or run with --dry-run")
            sys.exit(1)
        hl = HyperliquidClient(
            private_key     = config.exchange.hl_private_key,
            account_address = config.exchange.hl_account_address,
            mainnet         = config.exchange.hl_use_mainnet,
            allow_spot_execution = getattr(config.exchange, "hl_spot_execution_enabled", False),
        )
        if hl.connect():
            exchanges.append(hl)
        else:
            log.error("Failed to connect to Hyperliquid")

    # ── Lighter ───────────────────────────────────────────
    if config.exchange.use_lighter:
        if not config.exchange.lighter_l1_private_key:
            log.error("LIGHTER_L1_PRIVATE_KEY is empty. Add it to .env or run with --dry-run")
            sys.exit(1)
        lt = LighterClient(
            l1_private_key = config.exchange.lighter_l1_private_key,
            api_private_key = config.exchange.lighter_api_private_key,
            account_index = config.exchange.lighter_account_index,
            api_key_index = config.exchange.lighter_api_key_index,
            api_base_url = config.exchange.lighter_api_base_url,
            web3_url = config.exchange.lighter_web3_url,
        )
        if lt.connect():
            exchanges.append(lt)
        else:
            log.error("Failed to connect to Lighter")

    if not exchanges:
        log.error("No exchanges connected. Cannot start agent.")
        sys.exit(1)

    enforce_trade_universe(exchanges)
    return exchanges


def run_preflight(args) -> int:
    failures: list[str] = []
    warnings: list[str] = []
    apply_dynamic_analysis_universe()
    infos: list[str] = []

    def ok(message: str):
        print(f"[OK] {message}")

    def warn(message: str):
        warnings.append(message)
        print(f"[WARN] {message}")

    def fail(message: str):
        failures.append(message)
        print(f"[FAIL] {message}")

    print("Crypto Trading Agent preflight")
    print("")

    if sys.version_info >= (3, 10):
        ok(f"Python {sys.version.split()[0]}")
    else:
        fail(f"Python {sys.version.split()[0]} is too old; Python 3.10+ is required")

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=DATA_DIR, delete=True):
            pass
        ok(f"DATA_DIR writable: {DATA_DIR}")
    except Exception as exc:
        fail(f"DATA_DIR is not writable ({DATA_DIR}): {exc}")

    try:
        preflight_dry_run = bool(getattr(args, "dry_run", False) or config.trading.dry_run)
        active = enforce_trade_universe(
            supported_override=set(
                configured_supported_coins(dry_run_mode=preflight_dry_run)
            )
        )
        ok("Active trading symbols: " + ", ".join(active))
    except ValueError as exc:
        fail(str(exc))
        active = [coin.upper() for coin in config.trading.coins]

    analysis_coins = []
    for coin in getattr(config.trading, "analysis_coins", []) or []:
        coin_upper = coin.upper()
        if coin_upper not in analysis_coins:
            analysis_coins.append(coin_upper)
    if analysis_coins:
        ok("Analysis/watchlist symbols: " + ", ".join(analysis_coins))
    if config.exchange.use_hyperliquid and getattr(config.trading, "enforce_active_venue_markets", True):
        inactive_analysis = []
        for coin in analysis_coins:
            activity = get_hyperliquid_market_activity(coin)
            if activity.get("reason") == "unsupported":
                continue
            if not activity.get("active", False):
                inactive_analysis.append(f"{coin} ({activity.get('reason', 'inactive')})")
        if inactive_analysis:
            warn(
                "Inactive Hyperliquid markets will stay analysis-only until fresh venue activity returns: "
                + ", ".join(inactive_analysis)
            )

    if os.environ.get("DASHBOARD_URL", "") and not os.environ.get("DASHBOARD_TOKEN", ""):
        fail("DASHBOARD_URL is set but DASHBOARD_TOKEN is missing")
    elif os.environ.get("DASHBOARD_TOKEN", "") and not os.environ.get("DASHBOARD_URL", ""):
        warn("DASHBOARD_TOKEN is set but DASHBOARD_URL is empty; remote dashboard push is disabled")
    elif os.environ.get("DASHBOARD_URL", "") and os.environ.get("DASHBOARD_TOKEN", ""):
        ok("Remote dashboard push is configured")
    else:
        ok("Remote dashboard push not configured (local dashboard only)")

    if config.exchange.use_hyperliquid:
        if importlib.util.find_spec("hyperliquid") is not None:
            ok("hyperliquid-python-sdk import available")
        else:
            fail("hyperliquid-python-sdk is not installed")
    else:
        ok("Hyperliquid venue disabled")

    if config.exchange.use_lighter:
        lighter_sdk = importlib.util.find_spec("lighter") is not None
        if lighter_sdk:
            ok("lighter-sdk import available")
        else:
            fail("lighter-sdk is not installed")
    else:
        ok("Lighter venue disabled")

    flask_installed = importlib.util.find_spec("flask") is not None
    if flask_installed:
        ok("Flask dashboard dependency available")
    else:
        fail("Flask is not installed; local dashboard service cannot start")

    if config.trading.use_chart_confirmation:
        if importlib.util.find_spec("anthropic") is None:
            fail("Chart confirmation is enabled but the anthropic package is missing")
        elif not os.environ.get("ANTHROPIC_API_KEY", ""):
            fail("Chart confirmation is enabled but ANTHROPIC_API_KEY is missing")
        else:
            ok("Chart confirmation dependencies are configured")
    else:
        ok("Chart confirmation disabled")

    if config.trading.use_news:
        if config.trading.cryptopanic_auth_token:
            ok("News integration enabled with CryptoPanic token")
        else:
            warn("News integration enabled without CRYPTOPANIC_AUTH_TOKEN — using public CryptoPanic feed")
    else:
        ok("News integration disabled")

    probe_coin = config.trading.coins[0] if config.trading.coins else "BTC"
    try:
        candles = fetch_candles(probe_coin, interval=config.trading.candle_interval, lookback=5)
        if candles is None or candles.empty:
            fail(f"Market data probe failed for {probe_coin}")
        else:
            ok(f"Market data probe succeeded for {probe_coin} ({len(candles)} candles)")
    except Exception as exc:
        fail(f"Market data probe failed for {probe_coin}: {exc}")

    try:
        price = get_current_price(probe_coin)
        if price:
            ok(f"Current price probe succeeded for {probe_coin}: ${price:,.2f}")
        else:
            fail(f"Current price probe returned no price for {probe_coin}")
    except Exception as exc:
        fail(f"Current price probe failed for {probe_coin}: {exc}")

    try:
        checkpoint = checkpoint_module.load_checkpoint(max_age_seconds=7 * 24 * 3600)
        if checkpoint:
            unsupported_positions = sorted(
                coin for coin in checkpoint.get("positions", {}) if coin.upper() not in set(active)
            )
            unsupported_orders = sorted(
                coin for coin in checkpoint.get("pending_orders", {}) if coin.upper() not in set(active)
            )
            unsupported_watches = sorted(
                coin for coin in checkpoint.get("reentry_watches", {}) if coin.upper() not in set(active)
            )
            if unsupported_positions:
                warn(
                    "Checkpoint contains positions outside the active trade universe: "
                    + ", ".join(unsupported_positions)
                    + " (they will be skipped on recovery)"
                )
            if unsupported_orders:
                warn(
                    "Checkpoint contains pending orders outside the active trade universe: "
                    + ", ".join(unsupported_orders)
                    + " (they will be skipped on recovery)"
                )
            if unsupported_watches:
                warn(
                    "Checkpoint contains re-entry watches outside the active trade universe: "
                    + ", ".join(unsupported_watches)
                    + " (they will be skipped on recovery)"
                )
    except Exception as exc:
        warn(f"Checkpoint inspection skipped: {exc}")

    live_mode = args.live or not config.trading.dry_run
    power = get_power_status()
    if power.available:
        source = power.source or ("AC Power" if power.on_ac_power else "Battery Power")
        battery_text = f"{power.battery_pct}%" if power.battery_pct is not None else "unknown"
        ok(f"Power status: {source} (battery {battery_text})")
        if live_mode:
            if config.trading.require_ac_power_for_live and power.on_ac_power is False:
                fail("Live trading requires AC power on the local Mac")
            if (
                power.battery_pct is not None
                and power.battery_pct < config.trading.minimum_battery_pct_for_live
            ):
                fail(
                    "Battery is below the live-trading minimum "
                    f"({power.battery_pct}% < {config.trading.minimum_battery_pct_for_live}%)"
                )
    else:
        warn("Power status unavailable; local live-trading power guard could not be verified")

    if live_mode:
        if config.trading.require_notifications_for_live and not config.notifications.enabled:
            fail("Telegram alerts are required for live trading; set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        elif not config.notifications.enabled:
            warn("Telegram alerts are not configured; live open/close notifications will be silent")

    if config.exchange.use_hyperliquid:
        hyperliquid_ready = all([
            config.exchange.hl_private_key,
            config.exchange.hl_account_address,
        ])
        if live_mode and getattr(config.exchange, "hl_spot_execution_enabled", False):
            ok("Hyperliquid live spot execution enabled for active long-only equities")
        if live_mode:
            if not hyperliquid_ready:
                fail("Hyperliquid live credentials are incomplete; add HL_PRIVATE_KEY and HL_ACCOUNT_ADDRESS")
            else:
                try:
                    client = HyperliquidClient(
                        private_key=config.exchange.hl_private_key,
                        account_address=config.exchange.hl_account_address,
                        mainnet=config.exchange.hl_use_mainnet,
                        allow_spot_execution=getattr(config.exchange, "hl_spot_execution_enabled", False),
                    )
                    if client.connect():
                        state = client.get_account_state()
                        if state:
                            ok(f"Hyperliquid live connectivity succeeded; equity ${state.total_equity_usd:,.2f}")
                        else:
                            fail("Hyperliquid connected but account state could not be read")
                    else:
                        fail("Hyperliquid live connectivity failed")
                except Exception as exc:
                    fail(f"Hyperliquid live connectivity failed: {exc}")
        else:
            if hyperliquid_ready:
                ok("Hyperliquid live credentials are present")
            else:
                warn("Hyperliquid live credentials are incomplete; this is fine for dry-run, but live trading is blocked until they are added")

    if config.exchange.use_lighter:
        lighter_ready = all([
            config.exchange.lighter_l1_private_key,
            config.exchange.lighter_api_private_key,
            config.exchange.lighter_account_index,
        ])
        if live_mode:
            if not lighter_ready:
                fail("Lighter live credentials are incomplete; run --lighter-bootstrap after creating/funding the account")
            else:
                try:
                    client = LighterClient(
                        l1_private_key=config.exchange.lighter_l1_private_key,
                        api_private_key=config.exchange.lighter_api_private_key,
                        account_index=config.exchange.lighter_account_index,
                        api_key_index=config.exchange.lighter_api_key_index,
                        api_base_url=config.exchange.lighter_api_base_url,
                        web3_url=config.exchange.lighter_web3_url,
                    )
                    if client.connect():
                        state = client.get_account_state()
                        if state:
                            ok(f"Lighter live connectivity succeeded; equity ${state.total_equity_usd:,.2f}")
                        else:
                            fail("Lighter connected but account state could not be read")
                    else:
                        fail("Lighter live connectivity failed")
                except Exception as exc:
                    fail(f"Lighter live connectivity failed: {exc}")
        else:
            if lighter_ready:
                ok("Lighter live credentials are present")
            else:
                warn("Lighter live credentials are incomplete; this is fine for dry-run, but live trading is blocked until bootstrap is completed")

    if live_mode and getattr(config.trading, "live_promotion_gate_enabled", False):
        try:
            gate = evaluate_live_promotion(config, DATA_DIR)
            if gate["passed"]:
                trade_metrics = gate.get("trade_metrics", {})
                precision_metrics = gate.get("precision_metrics", {})
                ok(
                    "Live promotion gate passed: "
                    f"{trade_metrics.get('closed_trades', 0)} closed trades, "
                    f"{trade_metrics.get('win_rate', 0.0) * 100:.1f}% WR, "
                    f"precision replay {precision_metrics.get('overall_win_rate', 0.0) * 100:.1f}%"
                )
            else:
                fail("Live promotion gate blocked trading: " + "; ".join(gate.get("blockers", [])))
        except Exception as exc:
            fail(f"Live promotion gate evaluation failed: {exc}")

    print("")
    if failures:
        print("Preflight failed.")
        return 1

    print("Preflight passed.")
    if warnings:
        print("Warnings:")
        for item in warnings:
            print(f"  - {item}")
    return 0


def paper_reset() -> int:
    try:
        running = subprocess.run(
            ["pgrep", "-f", "crypto_trading_agent.*main.py"],
            capture_output=True,
            text=True,
            check=False,
        )
        if running.returncode == 0 and running.stdout.strip():
            print("Refusing to reset paper state while an agent process is running. Stop launchd or the active session first.")
            return 1
    except Exception:
        pass

    removed = []
    for path in [CHECKPOINTS_DB, TRADE_MEMORY, TRADES_CSV, STATE_JSON, KILL_FILE, CONTROL_JSON]:
        try:
            if path.exists():
                path.unlink()
                removed.append(str(path))
        except Exception as exc:
            print(f"Failed to remove {path}: {exc}")
            return 1

    print("Paper-trading state cleared.")
    for item in removed:
        print(f"  removed: {item}")
    if not removed:
        print("  nothing to clear")
    return 0


def show_status(exchanges: list):
    """Print account state for all connected exchanges and exit."""
    for ex in exchanges:
        state = ex.get_account_state()
        if not state:
            log.error(f"[{ex.name}] Could not retrieve account state")
            continue
        print(f"\n── {ex.name} ──────────────────────────────────")
        print(f"  Total equity   : ${state.total_equity_usd:,.2f}")
        print(f"  Available USD  : ${state.available_usd:,.2f}")
        if state.positions:
            print("  Open positions:")
            for p in state.positions:
                print(
                    f"    {p['coin']:5s} {p['direction']:6s} "
                    f"size={p['size']:.6f}  "
                    f"entry=${p['entry_price']:.2f}  "
                    f"uPnL=${p.get('unrealised_pnl', 0):+.2f}"
                )
        else:
            print("  No open positions")


def main():
    args = parse_args()

    if args.dry_run and args.live:
        log.error("Choose either --dry-run or --live, not both")
        sys.exit(1)

    # ── Apply CLI overrides to config ─────────────────────
    if args.coins:
        config.trading.coins = args.coins
    if args.interval:
        config.trading.check_interval_seconds = args.interval
    if args.live:
        config.trading.dry_run = False
    elif args.dry_run:
        config.trading.dry_run = True

    if args.paper_reset:
        sys.exit(paper_reset())

    if args.preflight:
        sys.exit(run_preflight(args))

    if args.lighter_bootstrap:
        result = bootstrap_lighter_api(
            l1_private_key=config.exchange.lighter_l1_private_key,
            api_base_url=config.exchange.lighter_api_base_url,
            api_key_index=config.exchange.lighter_api_key_index,
        )
        if not result.get("ok"):
            log.error(result.get("error", "Lighter bootstrap failed"))
            sys.exit(1)
        print("")
        print("Lighter bootstrap succeeded. Add these lines to your .env:")
        print(f"LIGHTER_ACCOUNT_INDEX={result['account_index']}")
        print(f"LIGHTER_API_KEY_INDEX={result['api_key_index']}")
        print(f"LIGHTER_API_PRIVATE_KEY={result['api_private_key']}")
        print(f"LIGHTER_API_BASE_URL={config.exchange.lighter_api_base_url}")
        print("")
        return

    # ── Print welcome banner ──────────────────────────────
    print("""
╔══════════════════════════════════════════════════════╗
║         🤖  CRYPTO TRADING AGENT  🤖                  ║
║     Hyperliquid-First  |  Aggressive Strategy         ║
╚══════════════════════════════════════════════════════╝
""")

    # ── Connect exchanges ─────────────────────────────────
    try:
        exchanges = build_exchanges(args)
    except ValueError as exc:
        log.error(str(exc))
        sys.exit(1)

    # ── Status-only mode ──────────────────────────────────
    if args.status:
        show_status(exchanges)
        return

    # ── Validate config ───────────────────────────────────
    if not config.trading.dry_run:
        try:
            config.validate()
        except ValueError as e:
            log.error(str(e))
            sys.exit(1)
        if getattr(config.trading, "live_promotion_gate_enabled", False):
            gate = evaluate_live_promotion(config, DATA_DIR)
            if not gate["passed"]:
                log.error("Live promotion gate blocked trading: " + "; ".join(gate.get("blockers", [])))
                sys.exit(1)
            trade_metrics = gate.get("trade_metrics", {})
            precision_metrics = gate.get("precision_metrics", {})
            log.info(
                "Live promotion gate passed: "
                f"{trade_metrics.get('closed_trades', 0)} trades, "
                f"{trade_metrics.get('win_rate', 0.0) * 100:.1f}% WR, "
                f"precision replay {precision_metrics.get('overall_win_rate', 0.0) * 100:.1f}%"
            )

    # ── Launch agent ──────────────────────────────────────
    agent = TradingAgent(config, exchanges)
    agent.start()


if __name__ == "__main__":
    main()
