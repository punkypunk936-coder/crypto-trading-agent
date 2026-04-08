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
from exchanges.hyperliquid_client import HyperliquidClient
from exchanges.lighter_client     import LighterClient, bootstrap_lighter_api
from exchanges.dry_run            import DryRunExchange
from agent import TradingAgent
from data.market_data import fetch_candles, get_current_price
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
        description="Crypto Trading Agent — Hyperliquid + Lighter",
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


def configured_supported_coins() -> list[str]:
    supported = set()
    if config.exchange.use_lighter:
        supported.update(LighterClient.supported_market_symbols())
    if config.exchange.use_hyperliquid:
        supported.update(HyperliquidClient("", "").supported_coins())
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


def enforce_trade_universe(exchanges: list | None = None) -> list[str]:
    if exchanges:
        supported = set()
        for ex in exchanges:
            supported.update(ex.supported_coins())
    else:
        supported = set(configured_supported_coins())

    if not supported:
        raise ValueError("No supported trading symbols are available for the configured venues")

    active = _normalise_coin_list(config.trading.coins)
    unsupported = sorted(set(active) - supported)
    if unsupported:
        raise ValueError(
            "Configured symbols are not tradable on the active venue(s): "
            + ", ".join(unsupported)
        )

    promoted = []
    if getattr(config.trading, "auto_promote_analysis_coins", False):
        analysis = _normalise_coin_list(getattr(config.trading, "analysis_coins", []) or [])
        for coin in analysis:
            if coin in supported and coin not in active:
                promoted.append(coin)
        if promoted:
            log.info(
                "Promoting watchlist symbols into the tradeable universe on active venue(s): "
                + ", ".join(promoted)
            )
        deferred = [coin for coin in analysis if coin not in supported and coin not in active]
        if deferred:
            log.info(
                "Keeping watchlist symbols in observation mode until a connected venue supports them: "
                + ", ".join(deferred)
            )

    config.trading.coins = [coin for coin in active if coin in supported] + promoted
    return config.trading.coins


def build_exchanges(args) -> list:
    """Construct and connect exchange clients based on config and args."""
    exchanges = []

    dry_run = args.dry_run or config.trading.dry_run
    if dry_run:
        supported_symbols = configured_supported_coins() or ["BTC", "ETH", "SOL"]
        config.trading.coins = [coin.upper() for coin in config.trading.coins]
        enforce_trade_universe()
        log.info("🟡  DRY RUN mode — using paper trading exchange")
        ex = DryRunExchange(
            starting_balance_usd=args.balance,
            supported_symbols=supported_symbols,
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
        active = enforce_trade_universe()
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

    if os.environ.get("DASHBOARD_URL", "") and not os.environ.get("DASHBOARD_TOKEN", ""):
        fail("DASHBOARD_URL is set but DASHBOARD_TOKEN is missing")
    elif os.environ.get("DASHBOARD_TOKEN", "") and not os.environ.get("DASHBOARD_URL", ""):
        warn("DASHBOARD_TOKEN is set but DASHBOARD_URL is empty; remote dashboard push is disabled")
    elif os.environ.get("DASHBOARD_URL", "") and os.environ.get("DASHBOARD_TOKEN", ""):
        ok("Remote dashboard push is configured")
    else:
        ok("Remote dashboard push not configured (local dashboard only)")

    lighter_sdk = importlib.util.find_spec("lighter") is not None
    if lighter_sdk:
        ok("lighter-sdk import available")
    else:
        fail("lighter-sdk is not installed")

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
            fail("News integration is enabled but CRYPTOPANIC_AUTH_TOKEN is missing")
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
║    Hyperliquid + Lighter  |  Aggressive Strategy      ║
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

    # ── Launch agent ──────────────────────────────────────
    agent = TradingAgent(config, exchanges)
    agent.start()


if __name__ == "__main__":
    main()
