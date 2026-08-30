import { json, readJson, writeJson } from "../lib/dashboard-store";

const LEDGER_PATH = "dashboard/ask-forecast-ledger.json";
const MAX_RECORDS = 2000;
const MAX_SETTLEMENTS = 24;
const RESOLUTION_SOURCE = "venue_candle_path_v2";

function finite(value: unknown, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function normalize(source: any) {
  const ticker = String(source?.ticker || source?.coin || "").trim().toUpperCase();
  const direction = String(source?.direction || "").trim().toUpperCase();
  const analysedMs = Date.parse(String(source?.analysedAt || source?.analysed_at || ""));
  const target = finite(source?.target);
  const invalidation = finite(source?.invalidation);
  if (!ticker || !["LONG", "SHORT"].includes(direction) || !Number.isFinite(analysedMs) || target <= 0 || invalidation <= 0) {
    return null;
  }
  const trustedOutcome = [0, 1].includes(source?.outcome) && source?.resolutionSource === RESOLUTION_SOURCE;
  const current = finite(source?.current);
  const entry = finite(source?.entry, current);
  const entryIsLive = direction === "LONG" ? current >= entry : current <= entry;
  const activatedAt = source?.activatedAt || (entryIsLive ? new Date(analysedMs).toISOString() : null);
  const status = trustedOutcome
    ? "resolved"
    : ["pending_entry", "active", "superseded", "expired_untriggered"].includes(String(source?.status || ""))
      ? String(source.status)
      : activatedAt ? "active" : "pending_entry";
  return {
    id: String(source?.id || `${ticker}-${direction}-${analysedMs}`).slice(0, 180),
    ticker,
    direction,
    decision: String(source?.decision || "").slice(0, 80),
    query: String(source?.query || "").slice(0, 240),
    source: String(source?.source || "ask_punky").slice(0, 80),
    analysedAt: new Date(analysedMs).toISOString(),
    horizonHours: Math.max(1, Math.min(168, finite(source?.horizonHours, 24))),
    venueSymbol: String(source?.venueSymbol || `xyz:${ticker}`).trim(),
    current,
    entry,
    entryMode: String(source?.entryMode || (direction === "LONG" ? "above" : "below")),
    target,
    invalidation,
    rr: finite(source?.rr),
    rrBucket: String(source?.rrBucket || ""),
    assetBucket: String(source?.assetBucket || ""),
    setupType: String(source?.setupType || (direction === "LONG" ? "long_above_level" : "short_below_level")),
    probability: Math.max(0.01, Math.min(0.99, finite(source?.probability, 0.5))),
    evidenceQuality: Math.max(0, Math.min(100, finite(source?.evidenceQuality))),
    limitedHistory: Boolean(source?.limitedHistory),
    outcome: trustedOutcome ? Number(source.outcome) : null,
    outcomeReason: trustedOutcome || ["superseded", "expired_untriggered"].includes(status) ? String(source?.outcomeReason || "") : "",
    resolvedAt: trustedOutcome || ["superseded", "expired_untriggered"].includes(status) ? source?.resolvedAt || null : null,
    resolutionSource: trustedOutcome ? RESOLUTION_SOURCE : "",
    activatedAt,
    status,
  };
}

async function fetchCandles(record: any, nowMs: number) {
  const startedMs = Date.parse(record.analysedAt);
  const expiresMs = startedMs + (finite(record.horizonHours, 24) * 60 * 60 * 1000);
  const endMs = Math.min(nowMs, expiresMs);
  if (!(startedMs > 0) || endMs <= startedMs) return [];
  const response = await fetch("https://api.hyperliquid.xyz/info", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      type: "candleSnapshot",
      req: {
        coin: record.venueSymbol,
        interval: "15m",
        startTime: startedMs - (15 * 60 * 1000),
        endTime: endMs + (15 * 60 * 1000),
      },
    }),
  });
  if (!response.ok) return [];
  const payload = await response.json();
  return (Array.isArray(payload) ? payload : []).sort((a, b) => finite(a?.t) - finite(b?.t));
}

function settle(record: any, bars: any[], nowMs: number) {
  if ([0, 1].includes(record.outcome) || ["superseded", "expired_untriggered"].includes(record.status)) return record;
  const startedMs = Date.parse(record.analysedAt);
  const expiresMs = startedMs + (finite(record.horizonHours, 24) * 60 * 60 * 1000);
  const relevant = bars.filter((bar) => finite(bar?.t) >= startedMs && finite(bar?.t) <= expiresMs);
  if (!relevant.length) return record;
  let activatedAt = Date.parse(String(record.activatedAt || ""));
  for (const bar of relevant) {
    const high = finite(bar?.h);
    const low = finite(bar?.l);
    if (!Number.isFinite(activatedAt)) {
      const entryHit = record.direction === "LONG" ? high >= record.entry : low <= record.entry;
      if (!entryHit) continue;
      activatedAt = finite(bar?.T || bar?.t);
      record = { ...record, activatedAt: new Date(activatedAt).toISOString(), status: "active" };
    }
    const targetHit = record.direction === "LONG" ? high >= record.target : low <= record.target;
    const invalidHit = record.direction === "LONG" ? low <= record.invalidation : high >= record.invalidation;
    if (!targetHit && !invalidHit) continue;
    const ambiguous = targetHit && invalidHit;
    return {
      ...record,
      outcome: ambiguous || invalidHit ? 0 : 1,
      outcomeReason: ambiguous ? "target_and_invalidation_same_candle" : invalidHit ? "invalidation_first" : "target_first",
      resolvedAt: new Date(finite(bar?.T || bar?.t)).toISOString(),
      resolutionSource: RESOLUTION_SOURCE,
      status: "resolved",
    };
  }
  if (nowMs >= expiresMs && relevant.length >= 2) {
    if (!Number.isFinite(activatedAt)) {
      return {
        ...record,
        outcome: null,
        outcomeReason: "entry_never_reached",
        resolvedAt: new Date(expiresMs).toISOString(),
        resolutionSource: "",
        status: "expired_untriggered",
      };
    }
    return {
      ...record,
      outcome: 0,
      outcomeReason: "horizon_expired_before_target",
      resolvedAt: new Date(expiresMs).toISOString(),
      resolutionSource: RESOLUTION_SOURCE,
      status: "resolved",
    };
  }
  return record;
}

async function settlePending(records: any[]) {
  const nowMs = Date.now();
  const pending = records
    .map((record, index) => ({ record, index }))
    .filter(({ record }) => (
      ![0, 1].includes(record.outcome)
      && !["superseded", "expired_untriggered"].includes(record.status)
    ))
    .slice(0, MAX_SETTLEMENTS);
  const settled = await Promise.all(pending.map(async ({ record, index }) => ({
    index,
    record: settle(record, await fetchCandles(record, nowMs).catch(() => []), nowMs),
  })));
  const next = records.slice();
  for (const item of settled) next[item.index] = item.record;
  return next;
}

function summary(records: any[]) {
  const resolved = records.filter((record) => [0, 1].includes(record.outcome));
  const wins = resolved.reduce((total, record) => total + Number(record.outcome === 1), 0);
  const families: Record<string, any> = {};
  const setupFamilies: Record<string, any> = {};
  for (const record of resolved) {
    const key = `${record.ticker}:${record.direction}`;
    families[key] ||= { samples: 0, wins: 0 };
    families[key].samples += 1;
    families[key].wins += Number(record.outcome === 1);
    const setupKey = String(record.setupType || "unknown");
    setupFamilies[setupKey] ||= { samples: 0, wins: 0 };
    setupFamilies[setupKey].samples += 1;
    setupFamilies[setupKey].wins += Number(record.outcome === 1);
  }
  for (const value of [...Object.values(families), ...Object.values(setupFamilies)] as any[]) {
    value.hit_rate = Math.round((value.wins / Math.max(1, value.samples)) * 10000) / 10000;
  }
  return {
    records: records.length,
    resolved: resolved.length,
    wins,
    hit_rate: resolved.length ? Math.round((wins / resolved.length) * 10000) / 10000 : null,
    families,
    setup_families: setupFamilies,
    pending_entry: records.filter((record) => record.status === "pending_entry").length,
    active: records.filter((record) => record.status === "active" && ![0, 1].includes(record.outcome)).length,
    resolution_source: RESOLUTION_SOURCE,
  };
}

async function processRecords(incoming: any[] = []) {
  const stored = await readJson(LEDGER_PATH, []);
  const byId = new Map<string, any>();
  for (const raw of Array.isArray(stored) ? stored : []) {
    const record = normalize(raw);
    if (record) byId.set(record.id, record);
  }
  for (const raw of incoming.slice(0, 500)) {
    const record = normalize(raw);
    if (!record) continue;
    const prior = byId.get(record.id);
    if (prior && [0, 1].includes(prior.outcome)) {
      record.outcome = prior.outcome;
      record.outcomeReason = prior.outcomeReason;
      record.resolvedAt = prior.resolvedAt;
      record.resolutionSource = prior.resolutionSource;
    }
    byId.set(record.id, { ...(prior || {}), ...record });
  }
  let records = [...byId.values()]
    .sort((a, b) => Date.parse(a.analysedAt) - Date.parse(b.analysedAt))
    .slice(-MAX_RECORDS);
  records = await settlePending(records);
  await writeJson(LEDGER_PATH, records);
  return records;
}

export async function GET() {
  try {
    const records = await processRecords();
    return json({ ok: true, records: records.slice(-500), summary: summary(records) }, {
      headers: { "Cache-Control": "no-store, no-cache, max-age=0" },
    });
  } catch (error) {
    return json({ ok: false, error: error instanceof Error ? error.message : "Ask ledger unavailable" }, { status: 503 });
  }
}

export async function POST(request: Request) {
  try {
    const payload = await request.json().catch(() => ({}));
    const incoming = Array.isArray(payload?.records) ? payload.records : [];
    const records = await processRecords(incoming);
    return json({ ok: true, records: records.slice(-500), summary: summary(records) });
  } catch (error) {
    return json({ ok: false, error: error instanceof Error ? error.message : "Ask ledger unavailable" }, { status: 503 });
  }
}
