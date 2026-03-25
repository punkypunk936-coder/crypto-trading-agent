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

  await store.setJSON("current-state", data.state);

  if (data.trades && Array.isArray(data.trades)) {
    await store.setJSON("trades", data.trades);
  }

  return new Response(
    JSON.stringify({ ok: true, cycle: data.state.cycle_number || 0 }),
    { headers: { "Content-Type": "application/json" } }
  );
};

export const config: Config = {
  path: "/api/push",
};
