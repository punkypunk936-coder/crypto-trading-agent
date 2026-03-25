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

  const data = await req.json().catch(() => ({}));
  const reason = data.reason || "Dashboard kill switch";

  const store = getStore({ name: "trading-state", consistency: "strong" });
  await store.setJSON("kill-signal", {
    active: true,
    reason,
    timestamp: new Date().toISOString(),
  });

  return new Response(
    JSON.stringify({ ok: true, message: "Kill signal set" }),
    { headers: { "Content-Type": "application/json" } }
  );
};

export const config: Config = {
  path: "/api/kill",
};
