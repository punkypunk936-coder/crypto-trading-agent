import { get, put } from "@vercel/blob";

declare const process: {
  env: Record<string, string | undefined>;
};

export const SNAPSHOT_PATH = "dashboard/dashboard_snapshot.json";
export const STATE_PATH = "dashboard/current-state.json";
export const TRADES_PATH = "dashboard/trades.json";
export const CONTROL_PATH = "dashboard/control.json";
export const MARKET_MAP_PATH = "dashboard/daily_market_map.json";
export const TRADE_REVIEWS_PATH = "dashboard/trade_reviews.json";
export const DECISION_REVIEW_PATH = "dashboard/decision_review_report.json";
export const CHALLENGER_REPORT_PATH = "dashboard/challenger_model_report.json";
export const PLAYBOOK_DISTILLER_PATH = "dashboard/playbook_distiller_report.json";
export const POLICY_HEALTH_REPORT_PATH = "dashboard/policy_health_report.json";
export const FALLBACK_REPO_OWNER = "punkypunk936-coder";
export const FALLBACK_REPO_NAME = "crypto-trading-agent";
export const FALLBACK_REPO_TAG = "dashboard-state-live";
export const NETLIFY_FALLBACK_API_BASE = "https://punky-crypto-agent-dash.netlify.app/api";

const DEFAULT_STATE = {
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

export function defaultState() {
  return structuredClone(DEFAULT_STATE);
}

export function defaultControl() {
  return {
    kill: {
      active: false,
      reason: "",
      requested_at: null,
      acknowledged_at: null,
    },
  };
}

export function defaultMarketMap() {
  return {
    date: null,
    updated_at: null,
    global_notes: "",
    coins: {},
  };
}

function zoneAround(level: number) {
  if (!(level > 0)) {
    return { low: 0, high: 0 };
  }
  const pad = Math.max(level * 0.0025, 0.01);
  return {
    low: Math.max(0, Math.round((level - pad) * 1_000_000) / 1_000_000),
    high: Math.round((level + pad) * 1_000_000) / 1_000_000,
  };
}

function autoTradeMode(bias: string) {
  if (bias === "BULLISH") {
    return "Buy pullbacks into support/demand; press harder after reclaim closes confirm.";
  }
  if (bias === "BEARISH") {
    return "Sell rallies into supply/resistance; press harder after breakdown closes confirm.";
  }
  return "Range the edges only; wait for reclaim/breakdown confirmation before pressing.";
}

export function marketMapFromState(state: any) {
  const safeState = state && typeof state === "object" ? state : {};
  const signals = safeState.signals && typeof safeState.signals === "object" ? safeState.signals : {};
  const coins: Record<string, any> = {};

  for (const [coin, rawSignal] of Object.entries(signals)) {
    const sig = rawSignal && typeof rawSignal === "object" ? rawSignal as Record<string, any> : {};
    const support = Number.parseFloat(String(sig.market_map_nearest_support || 0)) || 0;
    const resistance = Number.parseFloat(String(sig.market_map_nearest_resistance || 0)) || 0;
    const breakout = Number.parseFloat(String(sig.daily_breakout_level || 0)) || 0;
    const breakdown = Number.parseFloat(String(sig.daily_breakdown_level || 0)) || 0;
    const available = Boolean(sig.market_map_available) || support > 0 || resistance > 0 || breakout > 0 || breakdown > 0;
    if (!available) {
      continue;
    }

    const bias = String(sig.market_map_bias || "NEUTRAL").toUpperCase();
    coins[String(coin)] = {
      bias,
      confidence: String(sig.confidence || "MEDIUM").toUpperCase(),
      supports: support > 0 ? [support] : [],
      resistances: resistance > 0 ? [resistance] : [],
      daily_close_long_above: breakout > 0 ? [breakout] : [],
      daily_close_short_below: breakdown > 0 ? [breakdown] : [],
      demand_zone: zoneAround(support),
      supply_zone: zoneAround(resistance),
      notes: String(sig.market_map_notes || ""),
      trade_mode: autoTradeMode(bias),
      summary: String(sig.market_map_summary || ""),
      source: "AUTO",
      auto_generated: true,
      updated_at: safeState.last_cycle || null,
    };
  }

  return normalizeMarketMap({
    date: safeState.last_cycle ? String(safeState.last_cycle).slice(0, 10) : null,
    updated_at: safeState.last_cycle || null,
    global_notes: "",
    coins,
  });
}

export function normalizeMarketMap(marketMap: any) {
  const next = defaultMarketMap();
  if (marketMap && typeof marketMap === "object") {
    next.date = marketMap.date ?? null;
    next.updated_at = marketMap.updated_at ?? null;
    next.global_notes = String(marketMap.global_notes || "");
    next.coins = marketMap.coins && typeof marketMap.coins === "object" ? marketMap.coins : {};
  }
  return next;
}

export function defaultTradeReviews() {
  return {
    updated_at: null,
    reviews: {},
  };
}

export function normalizeTradeReviews(tradeReviews: any) {
  const next = defaultTradeReviews();
  if (tradeReviews && typeof tradeReviews === "object") {
    next.updated_at = tradeReviews.updated_at ?? null;
    next.reviews = tradeReviews.reviews && typeof tradeReviews.reviews === "object" ? tradeReviews.reviews : {};
  }
  return next;
}

function safeFloat(value: any) {
  const num = Number.parseFloat(String(value ?? 0));
  return Number.isFinite(num) ? num : 0;
}

function pickLevel(values: any, prefer: "min" | "max" = "min", fallback: any = null) {
  const numbers = (Array.isArray(values) ? values : [])
    .map((value) => safeFloat(value))
    .filter((value) => value > 0);
  if (!numbers.length && fallback !== null && fallback !== undefined) {
    const fallbackNumber = safeFloat(fallback);
    if (fallbackNumber > 0) {
      numbers.push(fallbackNumber);
    }
  }
  if (!numbers.length) {
    return null;
  }
  return prefer === "min" ? Math.min(...numbers) : Math.max(...numbers);
}

function primaryReason(text: any) {
  const parts = String(text || "")
    .split("·")
    .map((part) => part.trim())
    .filter(Boolean);
  const preferred = parts.find((part) => {
    const lower = part.toLowerCase();
    return !(
      lower.startsWith("score ") ||
      lower.startsWith("map:") ||
      lower.startsWith("breakout state:") ||
      lower.startsWith("key levels:")
    );
  });
  return preferred || parts[0] || "";
}

function mapBlurb(text: any) {
  let cleaned = String(text || "").trim();
  if (!cleaned) {
    return "";
  }
  const replacements: Record<string, string> = {
    "auto bullish map": "bullish daily view",
    "auto bearish map": "bearish daily view",
    "auto neutral map": "neutral daily view",
    "daily reclaim confirmed": "reclaim is confirmed",
    "daily breakdown confirmed": "breakdown is still active",
    "price is sitting in mapped demand": "price is in demand",
    "price is sitting in mapped supply": "price is in supply",
    "price is pressing mapped resistance": "price is at resistance",
    "price is testing mapped support": "price is testing support",
  };
  for (const [source, target] of Object.entries(replacements)) {
    cleaned = cleaned.replaceAll(source, target);
  }
  return cleaned;
}

export function actionBoard(state: any, marketMap: any) {
  const signals = state?.signals && typeof state.signals === "object" ? state.signals : {};
  const positions = Array.isArray(state?.positions) ? state.positions : [];
  const positionsByCoin = Object.fromEntries(
    positions
      .map((position) => [String(position?.coin || "").toUpperCase(), position && typeof position === "object" ? position : {}])
      .filter(([coin]) => coin),
  );
  const tracked = new Set<string>();
  const configCoins = Array.isArray(state?.config?.coins) ? state.config.coins : [];
  for (const coin of configCoins) tracked.add(String(coin || "").toUpperCase());
  for (const coin of Object.keys(signals)) tracked.add(String(coin || "").toUpperCase());
  for (const coin of Object.keys(positionsByCoin)) tracked.add(String(coin || "").toUpperCase());

  const entries = marketMap?.coins && typeof marketMap.coins === "object" ? marketMap.coins : {};
  const items: any[] = [];
  const order: Record<string, number> = {
    OPEN_LONG: 0,
    OPEN_SHORT: 0,
    READY_LONG: 1,
    READY_SHORT: 1,
    WAIT_RECLAIM: 2,
    WATCH_LONG: 2,
    WAIT_BREAKDOWN: 2,
    WATCH_SHORT: 2,
    NO_SETUP: 3,
  };

  for (const coin of [...tracked].filter(Boolean).sort()) {
    const sig = signals[coin] && typeof signals[coin] === "object" ? signals[coin] : {};
    const pos = positionsByCoin[coin];
    const mapEntry = entries[coin] && typeof entries[coin] === "object" ? entries[coin] : {};
    const tradable = (sig.execution_mode || "observation_only") === "tradable" || Boolean(pos);
    const bias = String(sig.market_map_bias || mapEntry.bias || "NEUTRAL").toUpperCase();
    const support = pickLevel(mapEntry.supports, "max", sig.market_map_nearest_support);
    const resistance = pickLevel(mapEntry.resistances, "min", sig.market_map_nearest_resistance);
    const longTrigger = pickLevel(mapEntry.daily_close_long_above, "min", resistance);
    const shortTrigger = pickLevel(mapEntry.daily_close_short_below, "max", support);
    const currentLogic = String(
      pos?.current_logic ||
      pos?.entry_logic ||
      sig.decision_reason ||
      sig.flat_reason ||
      "",
    ).trim();
    const blocker = primaryReason(sig.flat_reason || sig.decision_reason || "");
    const summaryText = mapBlurb(sig.market_map_summary || mapEntry.summary || mapEntry.notes || "");
    const confidence = String(sig.confidence || "LOW").toUpperCase();
    const score = safeFloat(sig.score || 50);
    const action = String(sig.action || "FLAT").toUpperCase();

    let status = "NO_SETUP";
    let label = "No setup";
    let headline = blocker || "No clean edge right now.";
    let trigger = "Wait for structure and order-flow to agree.";

    if (pos) {
      const direction = String(pos.direction || "").toUpperCase() || action || "LONG";
      status = `OPEN_${direction}`;
      label = `In ${direction}`;
      headline = primaryReason(currentLogic) || "Trade is live and being managed.";
      const stop = safeFloat(pos.stop_loss);
      const target = safeFloat(pos.take_profit);
      trigger =
        stop > 0 && target > 0
          ? `Stop ${stop.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} • Target ${target.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
          : "Trade is already open.";
    } else if (action === "LONG") {
      status = "READY_LONG";
      label = "Long ready";
      headline = primaryReason(sig.decision_reason || currentLogic) || "Long setup is ready.";
      const live = safeFloat(sig.live_price) || safeFloat(sig.price);
      trigger = `Entry is live now around ${live.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    } else if (action === "SHORT") {
      status = "READY_SHORT";
      label = "Short ready";
      headline = primaryReason(sig.decision_reason || currentLogic) || "Short setup is ready.";
      const live = safeFloat(sig.live_price) || safeFloat(sig.price);
      trigger = `Entry is live now around ${live.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    } else if (bias === "BULLISH" && Boolean(sig.market_map_block_longs) && longTrigger) {
      status = "WAIT_RECLAIM";
      label = "Wait for reclaim";
      headline = summaryText
        ? `Daily bias is bullish, but ${summaryText}.`
        : "Daily bias is bullish, but the long is still blocked until price reclaims resistance.";
      trigger = `Long only after a reclaim above ${longTrigger.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    } else if (bias === "BEARISH" && Boolean(sig.market_map_block_shorts) && shortTrigger) {
      status = "WAIT_BREAKDOWN";
      label = "Wait for breakdown";
      headline = summaryText
        ? `Daily bias is bearish, but ${summaryText}.`
        : "Daily bias is bearish, but the short is still blocked until price breaks support.";
      trigger = `Short only after a breakdown below ${shortTrigger.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    } else if (bias === "BULLISH") {
      status = "WATCH_LONG";
      label = "Bullish watch";
      headline = summaryText
        ? `Higher-timeframe bias is bullish, and ${summaryText}.`
        : "Higher-timeframe bias is bullish, but the entry is not ready.";
      trigger = longTrigger
        ? `Best long trigger is above ${longTrigger.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
        : "Wait for cleaner long confirmation.";
    } else if (bias === "BEARISH") {
      status = "WATCH_SHORT";
      label = "Bearish watch";
      headline = summaryText
        ? `Higher-timeframe bias is bearish, and ${summaryText}.`
        : "Higher-timeframe bias is bearish, but the entry is not ready.";
      trigger = shortTrigger
        ? `Best short trigger is below ${shortTrigger.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
        : "Wait for cleaner short confirmation.";
    }

    let risk = summaryText || "";
    if (["WATCH_LONG", "WAIT_RECLAIM", "READY_LONG", "OPEN_LONG"].includes(status) && support) {
      risk = `Risk if it loses ${support.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    } else if (["WATCH_SHORT", "WAIT_BREAKDOWN", "READY_SHORT", "OPEN_SHORT"].includes(status) && resistance) {
      risk = `Risk if it reclaims ${resistance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }

    items.push({
      coin,
      tradable,
      bias,
      status,
      label,
      headline,
      trigger,
      risk,
      map_summary: summaryText,
      confidence,
      score: Math.round(score * 10) / 10,
      pnl_usd: pos ? safeFloat(pos.unrealised_pnl) : 0,
    });
  }

  items.sort((left, right) => {
    const leftRank = [
      left.tradable ? 0 : 1,
      order[String(left.status || "NO_SETUP")] ?? 9,
      -(Math.abs(safeFloat(left.score) - 50)),
      String(left.coin || ""),
    ];
    const rightRank = [
      right.tradable ? 0 : 1,
      order[String(right.status || "NO_SETUP")] ?? 9,
      -(Math.abs(safeFloat(right.score) - 50)),
      String(right.coin || ""),
    ];
    return leftRank < rightRank ? -1 : leftRank > rightRank ? 1 : 0;
  });

  return {
    updated_at: state?.last_cycle || null,
    lead: items.length ? items[0] : null,
    items,
  };
}

export function mergeReviewsIntoTrades(trades: any[], tradeReviews: any) {
  const reviews = normalizeTradeReviews(tradeReviews).reviews || {};
  return (Array.isArray(trades) ? trades : []).map((trade) => {
    const item = trade && typeof trade === "object" ? { ...trade } : {};
    const review = reviews[String(item.trade_id || "")];
    if (review) {
      item.review = review;
    }
    return item;
  });
}

export function marketMapSummary(marketMap: any) {
  const coins = normalizeMarketMap(marketMap).coins || {};
  const summary = {
    count: 0,
    bullish: 0,
    bearish: 0,
    neutral: 0,
    manual_count: 0,
    auto_count: 0,
    updated_at: normalizeMarketMap(marketMap).updated_at,
  };
  for (const entry of Object.values(coins as Record<string, any>)) {
    summary.count += 1;
    const bias = String((entry as any)?.bias || "NEUTRAL").toUpperCase();
    if (bias === "BULLISH") summary.bullish += 1;
    else if (bias === "BEARISH") summary.bearish += 1;
    else summary.neutral += 1;
    const source = String((entry as any)?.source || "").toUpperCase();
    if (Boolean((entry as any)?.auto_generated) || source === "AUTO") summary.auto_count += 1;
    else summary.manual_count += 1;
  }
  return summary;
}

export function humanizeExitReason(reason: any) {
  const mapping: Record<string, string> = {
    take_profit: "target was reached",
    stop_loss: "hard invalidation was hit",
    conviction_lost: "conviction faded after entry",
    signal_reversal: "signal reversed against the trade",
    micro_invalidation: "micro invalidation triggered early",
    structure_invalidation: "structure invalidation triggered",
    htf_invalidation: "higher-timeframe invalidation triggered",
    time_stop: "time stop cut the trade",
  };
  const key = String(reason || "").trim().toLowerCase();
  return mapping[key] || key.replaceAll("_", " ").trim() || "no close logic recorded";
}

export function agentLesson(trade: any) {
  const direction = String(trade?.direction || "").toUpperCase();
  const exitReason = String(trade?.exit_reason || "").toLowerCase();
  const pnl = Number.parseFloat(trade?.pnl_usd || 0) || 0;

  if (pnl > 0) {
    if (exitReason === "take_profit") {
      return "The thesis followed through cleanly. Similar structure can stay tradeable when the same alignment shows up.";
    }
    if (exitReason === "conviction_lost" || exitReason === "time_stop") {
      return "The move worked, but momentum faded before the full target. Take cleaner partials when follow-through stalls.";
    }
    return "This setup paid. Keep favoring trades where structure, levels, and invalidation stay this coherent.";
  }

  if (exitReason === "stop_loss") {
    if (direction === "SHORT") {
      return "Avoid shorting straight into defended support and demand.";
    }
    if (direction === "LONG") {
      return "Avoid longing straight into heavy overhead resistance.";
    }
    return "The invalidation was hit quickly. Demand cleaner alignment before taking this setup again.";
  }
  if (exitReason === "conviction_lost" || exitReason === "time_stop") {
    return "The thesis never developed enough follow-through. Wait for stronger structure before committing capital.";
  }
  if (["signal_reversal", "structure_invalidation", "htf_invalidation", "micro_invalidation"].includes(exitReason)) {
    return "Structure turned against the trade. Respect invalidation faster when the higher timeframe disagrees.";
  }
  return "Only re-take this pattern when the market map, structure, and order-flow line up more cleanly.";
}

export function enrichTradesForLearning(trades: any[]) {
  return (Array.isArray(trades) ? trades : []).map((trade) => {
    const item = trade && typeof trade === "object" ? { ...trade } : {};
    item.open_logic = item.open_logic || item.reason || "No opening logic recorded";
    item.close_logic = item.close_logic || humanizeExitReason(item.exit_reason);
    item.agent_lesson = item.agent_lesson || agentLesson(item);
    return item;
  });
}

export function learningSummary(trades: any[]) {
  const safeTrades = enrichTradesForLearning(trades);
  const recent = [...safeTrades].slice(-8).reverse();
  const lessons = recent.map((trade) => {
    const pnl = Number.parseFloat(trade?.pnl_usd || 0) || 0;
    return {
      trade_id: trade?.trade_id ?? null,
      coin: trade?.coin ?? null,
      direction: trade?.direction ?? null,
      pnl_usd: pnl,
      result: pnl > 0 ? "WIN" : pnl < 0 ? "LOSS" : "FLAT",
      open_logic: trade?.open_logic || "",
      close_logic: trade?.close_logic || "",
      lesson: trade?.agent_lesson || "",
    };
  });
  return {
    count: lessons.length,
    wins: lessons.filter((lesson) => lesson.result === "WIN").length,
    losses: lessons.filter((lesson) => lesson.result === "LOSS").length,
    latest: lessons.length ? lessons[0] : null,
    recent_lessons: lessons,
  };
}

export function reviewSummary(trades: any[], tradeReviews: any) {
  const reviews = Object.values(normalizeTradeReviews(tradeReviews).reviews || {}) as Array<Record<string, any>>;
  const mergedTrades = mergeReviewsIntoTrades(trades, tradeReviews);
  const verdicts: Record<string, number> = {};
  const thesis_quality: Record<string, number> = {};
  const execution_quality: Record<string, number> = {};
  for (const review of reviews) {
    const verdict = String(review.verdict || "");
    const thesis = String(review.thesis_quality || "");
    const execution = String(review.execution_quality || "");
    if (verdict) verdicts[verdict] = (verdicts[verdict] || 0) + 1;
    if (thesis) thesis_quality[thesis] = (thesis_quality[thesis] || 0) + 1;
    if (execution) execution_quality[execution] = (execution_quality[execution] || 0) + 1;
  }
  const reviewed = mergedTrades.filter((trade) => trade?.review).length;
  return {
    count: reviews.length,
    coverage_pct: mergedTrades.length ? Math.round((reviewed / mergedTrades.length) * 1000) / 10 : 0,
    verdicts,
    thesis_quality,
    execution_quality,
    updated_at: reviews.length ? normalizeTradeReviews(tradeReviews).updated_at : null,
  };
}

export function normalizeControl(control: any) {
  const next = defaultControl();
  if (control && typeof control === "object" && control.kill && typeof control.kill === "object") {
    next.kill = {
      active: Boolean(control.kill.active),
      reason: String(control.kill.reason || ""),
      requested_at: control.kill.requested_at ?? null,
      acknowledged_at: control.kill.acknowledged_at ?? null,
    };
  }
  if (!next.kill.active) {
    next.kill.acknowledged_at = null;
  }
  return next;
}

export function requireStorage() {
  if (!process.env.BLOB_READ_WRITE_TOKEN) {
    throw new Error("BLOB_READ_WRITE_TOKEN is not configured");
  }
}

export async function readJson(pathname: string, fallback: any) {
  requireStorage();
  const result = await get(pathname, { access: "private" });
  if (!result || result.statusCode !== 200 || !result.stream) {
    return fallback;
  }

  const text = await new Response(result.stream).text();
  if (!text.trim()) {
    return fallback;
  }

  try {
    return JSON.parse(text);
  } catch {
    return fallback;
  }
}

export async function writeJson(pathname: string, value: unknown) {
  requireStorage();
  await put(pathname, JSON.stringify(value), {
    access: "private",
    addRandomSuffix: false,
    allowOverwrite: true,
    contentType: "application/json",
  });
}

export function githubFallbackUrl(pathname: string) {
  const owner = process.env.DASHBOARD_STATE_GITHUB_OWNER || FALLBACK_REPO_OWNER;
  const repo = process.env.DASHBOARD_STATE_GITHUB_REPO || FALLBACK_REPO_NAME;
  const tag = githubFallbackRef();
  return `https://raw.githubusercontent.com/${owner}/${repo}/${encodeURIComponent(tag)}/${pathname}`;
}

function githubFallbackRef() {
  return process.env.DASHBOARD_STATE_GIT_REF || process.env.DASHBOARD_STATE_GIT_TAG || FALLBACK_REPO_TAG;
}

function githubJsonHeaders(): Record<string, string> {
  const token = process.env.DASHBOARD_STATE_GITHUB_TOKEN || process.env.GITHUB_TOKEN || "";
  const headers: Record<string, string> = { Accept: "application/vnd.github+json" };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

function githubRawHeaders(): Record<string, string> {
  const token = process.env.DASHBOARD_STATE_GITHUB_TOKEN || process.env.GITHUB_TOKEN || "";
  const headers: Record<string, string> = {};
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

function encodeGithubPath(pathname: string) {
  return pathname.split("/").map((part) => encodeURIComponent(part)).join("/");
}

function withCacheBuster(url: string) {
  return `${url}${url.includes("?") ? "&" : "?"}ts=${Date.now()}`;
}

export function githubContentsFallbackUrl(pathname: string) {
  const owner = process.env.DASHBOARD_STATE_GITHUB_OWNER || FALLBACK_REPO_OWNER;
  const repo = process.env.DASHBOARD_STATE_GITHUB_REPO || FALLBACK_REPO_NAME;
  const ref = githubFallbackRef();
  return `https://api.github.com/repos/${owner}/${repo}/contents/${encodeGithubPath(pathname)}?ref=${encodeURIComponent(ref)}`;
}

async function parseJsonResponse(response: Response) {
  const text = await response.text();
  if (!text.trim()) {
    return undefined;
  }
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

async function readGithubContentsJson(pathname: string) {
  try {
    const response = await fetch(withCacheBuster(githubContentsFallbackUrl(pathname)), {
      cache: "no-store",
      headers: githubJsonHeaders(),
    });
    if (!response.ok) {
      return undefined;
    }

    const meta = await response.json().catch(() => null);
    const downloadUrl = typeof meta?.download_url === "string" ? meta.download_url : "";
    if (downloadUrl) {
      const rawResponse = await fetch(withCacheBuster(downloadUrl), {
        cache: "no-store",
        headers: githubRawHeaders(),
      });
      if (rawResponse.ok) {
        return await parseJsonResponse(rawResponse);
      }
    }

    const content = typeof meta?.content === "string" ? meta.content.replace(/\s/g, "") : "";
    const decoder = (globalThis as any).atob;
    if (content && typeof decoder === "function") {
      try {
        return JSON.parse(decoder(content));
      } catch {
        return undefined;
      }
    }
    return undefined;
  } catch {
    return undefined;
  }
}

export async function readGitFallbackJson(pathname: string, fallback: any) {
  const contentsValue = await readGithubContentsJson(pathname);
  if (contentsValue !== undefined) {
    return contentsValue;
  }

  try {
    const response = await fetch(withCacheBuster(githubFallbackUrl(pathname)), {
      cache: "no-store",
      headers: githubRawHeaders(),
    });
    if (!response.ok) {
      return fallback;
    }

    const rawValue = await parseJsonResponse(response);
    return rawValue === undefined ? fallback : rawValue;
  } catch {
    return fallback;
  }
}

export async function readNetlifyStateFallback() {
  try {
    const response = await fetch(`${NETLIFY_FALLBACK_API_BASE}/state?ts=${Date.now()}`, {
      cache: "no-store",
      headers: {
        "Cache-Control": "no-cache",
      },
    });
    if (!response.ok) {
      return null;
    }
    const payload = await response.json().catch(() => null);
    if (payload && typeof payload === "object" && (payload as any).state) {
      return payload;
    }
    return null;
  } catch {
    return null;
  }
}

export async function forwardNetlifyPush(payload: unknown, token: string) {
  try {
    const response = await fetch(`${NETLIFY_FALLBACK_API_BASE}/push`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { "X-Token": token } : {}),
      },
      body: JSON.stringify(payload),
      cache: "no-store",
    });
    const text = await response.text();
    let data: any = null;
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
    return { ok: response.ok, status: response.status, data };
  } catch (error) {
    return {
      ok: false,
      status: 502,
      data: error instanceof Error ? error.message : "Netlify fallback push failed",
    };
  }
}

export function unauthorized(request: Request) {
  const expectedToken = process.env.DASHBOARD_TOKEN || "";
  if (!expectedToken) {
    return false;
  }
  return (request.headers.get("X-Token") || "") !== expectedToken;
}

export function json(payload: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(payload), {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
}

export function calcStats(trades: any[]) {
  const safeTrades = Array.isArray(trades) ? trades : [];
  const closed = safeTrades.filter((trade) => {
    try {
      return trade?.exit_price && Number.parseFloat(trade.exit_price) > 0;
    } catch {
      return false;
    }
  });

  if (!closed.length) {
    return {
      total: 0,
      wins: 0,
      losses: 0,
      win_rate: 0,
      total_pnl: 0,
      avg_win: 0,
      avg_loss: 0,
      best: 0,
      worst: 0,
    };
  }

  const pnls = closed.map((trade) => {
    try {
      return Number.parseFloat(trade.pnl_usd || 0);
    } catch {
      return 0;
    }
  });
  const wins = pnls.filter((pnl) => pnl > 0);
  const losses = pnls.filter((pnl) => pnl <= 0);

  return {
    total: closed.length,
    wins: wins.length,
    losses: losses.length,
    win_rate: Math.round((wins.length / closed.length) * 1000) / 10,
    total_pnl: Math.round(pnls.reduce((sum, pnl) => sum + pnl, 0) * 100) / 100,
    avg_win: wins.length
      ? Math.round((wins.reduce((sum, pnl) => sum + pnl, 0) / wins.length) * 100) / 100
      : 0,
    avg_loss: losses.length
      ? Math.round((losses.reduce((sum, pnl) => sum + pnl, 0) / losses.length) * 100) / 100
      : 0,
    best: pnls.length ? Math.round(Math.max(...pnls) * 100) / 100 : 0,
    worst: pnls.length ? Math.round(Math.min(...pnls) * 100) / 100 : 0,
  };
}

export function decisionSummary(state: any) {
  const signals = state?.signals && typeof state.signals === "object" ? state.signals : {};
  const summary = {
    long_count: 0,
    short_count: 0,
    flat_count: 0,
    tradable_count: 0,
    tradable_active_count: 0,
    lead: null as any,
  };
  let leadRank: [number, number, number] = [-1, -1, -1];

  for (const [coin, rawSignal] of Object.entries(signals)) {
    const sig = rawSignal && typeof rawSignal === "object" ? rawSignal as Record<string, any> : {};
    let action = String(sig.action || "FLAT").toUpperCase();
    if (!["LONG", "SHORT", "FLAT"].includes(action)) {
      action = "FLAT";
    }
    if (action === "LONG") summary.long_count += 1;
    else if (action === "SHORT") summary.short_count += 1;
    else summary.flat_count += 1;

    const executionMode = sig.execution_mode || "observation_only";
    const isTradable = executionMode === "tradable";
    if (isTradable) {
      summary.tradable_count += 1;
    }
    if (isTradable && action !== "FLAT") {
      summary.tradable_active_count += 1;
    }

    let strength = 0;
    try {
      strength = Math.abs(Number.parseFloat(sig.score ?? 50) - 50);
    } catch {
      strength = 0;
    }

    const rank: [number, number, number] = [
      action !== "FLAT" ? 1 : 0,
      isTradable ? 1 : 0,
      strength,
    ];
    if (
      rank[0] > leadRank[0] ||
      (rank[0] === leadRank[0] && rank[1] > leadRank[1]) ||
      (rank[0] === leadRank[0] && rank[1] === leadRank[1] && rank[2] > leadRank[2])
    ) {
      leadRank = rank;
      summary.lead = {
        coin,
        action,
        score: sig.score ?? 50,
        confidence: sig.confidence ?? "LOW",
        execution_mode: executionMode,
        reason: sig.decision_reason || sig.reason || sig.flat_reason || "",
      };
    }
  }

  return summary;
}

export function augmentState(state: any) {
  const safeState = state && typeof state === "object" ? { ...defaultState(), ...state } : defaultState();
  safeState.positions_count = Array.isArray(safeState.positions) ? safeState.positions.length : 0;
  safeState.decision_summary = decisionSummary(safeState);
  return safeState;
}

export function runtimeStatus(state: any) {
  const lastCycle = state?.last_cycle;
  const interval = Number.parseInt(state?.config?.check_interval_seconds || 120, 10);
  let ageSeconds: number | null = null;
  let stale = false;

  if (typeof lastCycle === "string") {
    const parsed = Date.parse(lastCycle.replace(" ", "T"));
    if (!Number.isNaN(parsed)) {
      ageSeconds = Math.max(0, Math.floor((Date.now() - parsed) / 1000));
      stale = ageSeconds > Math.max(interval * 2, 240);
    }
  }

  return {
    stale,
    state_age_seconds: ageSeconds,
  };
}

export function serverTime() {
  return new Date().toISOString().slice(0, 19).replace("T", " ");
}

export function buildSnapshot(
  state: any,
  trades: any[],
  control?: any,
  marketMap?: any,
  tradeReviews?: any,
  timestamp?: string,
  decisionReviewReport?: any,
  challengerReport?: any,
  playbookDistillerReport?: any,
  policyHealthReport?: any,
) {
  const enrichedTrades = enrichTradesForLearning(Array.isArray(trades) ? trades : []);
  const safeTrades = mergeReviewsIntoTrades(enrichedTrades, tradeReviews);
  const shapedState = augmentState(state);
  const normalizedMarketMap = normalizeMarketMap(marketMap);
  const normalizedTradeReviews = normalizeTradeReviews(tradeReviews);
  return {
    state: shapedState,
    trades: safeTrades.slice(-50).reverse(),
    stats: calcStats(safeTrades),
    control: normalizeControl(control),
    action_board: actionBoard(shapedState, normalizedMarketMap),
    market_map: normalizedMarketMap,
    market_map_summary: marketMapSummary(normalizedMarketMap),
    trade_reviews: normalizedTradeReviews,
    review_summary: reviewSummary(safeTrades, normalizedTradeReviews),
    learning_summary: learningSummary(safeTrades),
    decision_review_report: decisionReviewReport && typeof decisionReviewReport === "object" ? decisionReviewReport : {},
    challenger_report: challengerReport && typeof challengerReport === "object" ? challengerReport : {},
    playbook_distiller_report: playbookDistillerReport && typeof playbookDistillerReport === "object" ? playbookDistillerReport : {},
    policy_health_report: policyHealthReport && typeof policyHealthReport === "object" ? policyHealthReport : {},
    runtime: runtimeStatus(shapedState),
    server_time: timestamp || serverTime(),
  };
}
