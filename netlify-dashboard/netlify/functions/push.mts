import type { Context, Config } from "@netlify/functions";
import { getStore } from "@netlify/blobs";

export default async (req: Request, context: Context) => {
  if (req.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  // Token check
  const expectedToken = Netlify.env.get("DASHBOARD_TOKEN") || "";
  if (expectedToken) {
    const token = req.headers.get("X-Token") || "";
    if (token !== expectedToken) {
      return new Response("Forbidden", { status: 403 });
    }
  }

  const data = await req.json();
  if (!data || !data.state) {
    return new Response("Missing state in payload", { status: 400 });
  }

  const store = getStore({ name: "trading-state", consistency: "strong" });

  const snapshot =
    data.snapshot && typeof data.snapshot === "object" && data.snapshot.state
      ? data.snapshot
      : {
          state: data.state,
          trades: Array.isArray(data.trades) ? data.trades : [],
          control: data.control && typeof data.control === "object" ? data.control : {},
          market_map: data.market_map && typeof data.market_map === "object" ? data.market_map : {},
          trade_reviews:
            data.trade_reviews && typeof data.trade_reviews === "object" ? data.trade_reviews : {},
          policy_health_report:
            data.policy_health_report && typeof data.policy_health_report === "object" ? data.policy_health_report : {},
          server_time: new Date().toISOString().slice(0, 19).replace("T", " "),
        };

  await store.setJSON("dashboard-snapshot", snapshot);
  await store.setJSON("current-state", snapshot.state || data.state);

  if (Array.isArray(snapshot.trades)) {
    await store.setJSON("trades", snapshot.trades);
  } else if (data.trades && Array.isArray(data.trades)) {
    await store.setJSON("trades", data.trades);
  }
  await store.setJSON("control", snapshot.control || data.control || {});
  await store.setJSON("market-map", snapshot.market_map || data.market_map || {});
  await store.setJSON("trade-reviews", snapshot.trade_reviews || data.trade_reviews || {});
  await store.setJSON("policy-health-report", snapshot.policy_health_report || data.policy_health_report || {});

  return new Response(
    JSON.stringify({ ok: true, cycle: snapshot?.state?.cycle_number || data.state.cycle_number || 0 }),
    { headers: { "Content-Type": "application/json" } }
  );
};

export const config: Config = {
  path: "/api/push",
};
