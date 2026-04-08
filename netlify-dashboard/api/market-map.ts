import {
  MARKET_MAP_PATH,
  SNAPSHOT_PATH,
  defaultMarketMap,
  json,
  normalizeMarketMap,
  readJson,
  unauthorized,
  writeJson,
  marketMapSummary,
} from "../lib/dashboard-store";

function upsertCoin(current: any, payload: any) {
  const marketMap = normalizeMarketMap(current);
  const coin = String(payload?.coin || "").toUpperCase().trim();
  if (!coin) {
    return marketMap;
  }
  const next = { ...(marketMap.coins || {}) };
  if (payload?.delete) {
    delete next[coin];
  } else {
    next[coin] = {
      ...(next[coin] || {}),
      bias: String(payload?.bias || next[coin]?.bias || "NEUTRAL").toUpperCase(),
      confidence: String(payload?.confidence || next[coin]?.confidence || "MEDIUM").toUpperCase(),
      supports: Array.isArray(payload?.supports) ? payload.supports : String(payload?.supports || next[coin]?.supports || "").split(",").map((v) => Number.parseFloat(String(v).trim())).filter((v) => v > 0),
      resistances: Array.isArray(payload?.resistances) ? payload.resistances : String(payload?.resistances || next[coin]?.resistances || "").split(",").map((v) => Number.parseFloat(String(v).trim())).filter((v) => v > 0),
      daily_close_long_above: Array.isArray(payload?.daily_close_long_above) ? payload.daily_close_long_above : String(payload?.daily_close_long_above || next[coin]?.daily_close_long_above || "").split(",").map((v) => Number.parseFloat(String(v).trim())).filter((v) => v > 0),
      daily_close_short_below: Array.isArray(payload?.daily_close_short_below) ? payload.daily_close_short_below : String(payload?.daily_close_short_below || next[coin]?.daily_close_short_below || "").split(",").map((v) => Number.parseFloat(String(v).trim())).filter((v) => v > 0),
      demand_zone: payload?.demand_zone || next[coin]?.demand_zone || { low: 0, high: 0 },
      supply_zone: payload?.supply_zone || next[coin]?.supply_zone || { low: 0, high: 0 },
      notes: String(payload?.notes || next[coin]?.notes || ""),
      trade_mode: String(payload?.trade_mode || next[coin]?.trade_mode || ""),
      updated_at: new Date().toISOString().slice(0, 19).replace("T", " "),
    };
  }
  return normalizeMarketMap({
    ...marketMap,
    global_notes: payload?.global_notes ?? marketMap.global_notes,
    updated_at: new Date().toISOString().slice(0, 19).replace("T", " "),
    coins: next,
  });
}

export async function GET() {
  const payload = await readJson(MARKET_MAP_PATH, defaultMarketMap());
  return json(normalizeMarketMap(payload));
}

export async function POST(request: Request) {
  if (unauthorized(request)) {
    return new Response("Forbidden", { status: 403 });
  }
  const data = await request.json().catch(() => null);
  if (!data) {
    return new Response("Missing payload", { status: 400 });
  }
  const current = await readJson(MARKET_MAP_PATH, defaultMarketMap());
  const payload = data?.coin ? upsertCoin(current, data) : normalizeMarketMap(data);
  await writeJson(MARKET_MAP_PATH, payload);
  const snapshot = await readJson(SNAPSHOT_PATH, null);
  if (snapshot && typeof snapshot === "object") {
    snapshot.market_map = payload;
    snapshot.market_map_summary = marketMapSummary(payload);
    await writeJson(SNAPSHOT_PATH, snapshot);
  }
  return json({ ok: true, market_map: payload });
}
