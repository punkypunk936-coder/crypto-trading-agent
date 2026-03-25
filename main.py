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

from config import config
from logger import log
from exchanges.hyperliquid_client import HyperliquidClient
from exchanges.lighter_client     import LighterClient
from exchanges.dry_run            import DryRunExchange
from agent import TradingAgent


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
    parser.add_argument("--coins",    nargs="+",
                        help="Override coins to trade, e.g. --coins BTC ETH")
    parser.add_argument("--interval", type=int,
                        help="Override check interval in seconds")
    parser.add_argument("--balance",  type=float, default=10_000.0,
                        help="Starting balance for dry-run mode (default: 10000)")
    return parser.parse_args()


def build_exchanges(args) -> list:
    """Construct and connect exchange clients based on config and args."""
    exchanges = []

    dry_run = args.dry_run or config.trading.dry_run
    if dry_run:
        log.info("🟡  DRY RUN mode — using paper trading exchange")
        ex = DryRunExchange(starting_balance_usd=args.balance)
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
        if not config.exchange.lighter_private_key:
            log.error("LIGHTER_PRIVATE_KEY is empty. Add it to .env or run with --dry-run")
            sys.exit(1)
        lt = LighterClient(
            private_key = config.exchange.lighter_private_key,
            web3_url    = config.exchange.lighter_web3_url,
        )
        if lt.connect():
            exchanges.append(lt)
        else:
            log.error("Failed to connect to Lighter")

    if not exchanges:
        log.error("No exchanges connected. Cannot start agent.")
        sys.exit(1)

    return exchanges


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

    # ── Print welcome banner ──────────────────────────────
    print("""
╔══════════════════════════════════════════════════════╗
║         🤖  CRYPTO TRADING AGENT  🤖                  ║
║    Hyperliquid + Lighter  |  Aggressive Strategy      ║
╚══════════════════════════════════════════════════════╝
""")

    # ── Connect exchanges ─────────────────────────────────
    exchanges = build_exchanges(args)

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
