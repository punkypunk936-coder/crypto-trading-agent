import type { Context, Config } from "@netlify/functions";
import { getStore } from "@netlify/blobs";

export default async (req: Request, context: Context) => {
  const store = getStore({ name: "trading-state", consistency: "strong" });

  const snapshotBlob = await store.get("dashboard-snapshot", { type: "json" });
  if (snapshotBlob && typeof snapshotBlob === "object" && (snapshotBlob as any).state) {
    return new Response(JSON.stringify(snapshotBlob), {
      headers: { "Content-Type": "application/json", "Cache-Control": "no-store, max-age=0" },
    });
  }

  const stateBlob = await store.get("current-state", { type: "json" });
  const tradesBlob = await store.get("trades", { type: "json" });
  const policyHealthBlob = await store.get("policy-health-report", { type: "json" });

  const state = stateBlob || {
    status: "offline",
    last_cycle: null,
    cycle_number: 0,
    portfolio_usd: 0,
    available_usd: 0,
    positions: [],
    signals: {},
    pending_orders: [],
    sentiment: {},
    mode: "unknown",
  };

  const trades = tradesBlob || [];

  // Calculate stats from trades
  const closed = trades.filter(
    (t: any) => t.exit_price && parseFloat(t.exit_price) > 0
  );
  const pnls = closed.map((t: any) => parseFloat(t.pnl_usd || 0));
  const wins = pnls.filter((p: number) => p > 0);
  const losses = pnls.filter((p: number) => p <= 0);

  const stats = closed.length
    ? {
        total: closed.length,
        wins: wins.length,
        losses: losses.length,
        win_rate: Math.round((wins.length / closed.length) * 1000) / 10,
        total_pnl: Math.round(pnls.reduce((a: number, b: number) => a + b, 0) * 100) / 100,
        avg_win: wins.length ? Math.round((wins.reduce((a: number, b: number) => a + b, 0) / wins.length) * 100) / 100 : 0,
        avg_loss: losses.length ? Math.round((losses.reduce((a: number, b: number) => a + b, 0) / losses.length) * 100) / 100 : 0,
        best: pnls.length ? Math.round(Math.max(...pnls) * 100) / 100 : 0,
        worst: pnls.length ? Math.round(Math.min(...pnls) * 100) / 100 : 0,
      }
    : {
        total: 0, wins: 0, losses: 0, win_rate: 0,
        total_pnl: 0, avg_win: 0, avg_loss: 0, best: 0, worst: 0,
      };

  return new Response(
    JSON.stringify({
      state,
      trades: trades.slice(-50).reverse(),
      stats,
      policy_health_report:
        policyHealthBlob && typeof policyHealthBlob === "object" ? policyHealthBlob : {},
      server_time: new Date().toISOString().slice(0, 19).replace("T", " "),
    }),
    { headers: { "Content-Type": "application/json", "Cache-Control": "no-store, max-age=0" } }
  );
};

export const config: Config = {
  path: "/api/state",
};
