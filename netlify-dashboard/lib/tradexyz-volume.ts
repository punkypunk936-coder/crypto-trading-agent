declare const process: {
  env: Record<string, string | undefined>;
};

const HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info";
const TRADEXYZ_DEX = "xyz";
const TRADEXYZ_PREFIX = "xyz:";
const DEFAULT_START_MS = Date.parse("2024-01-01T00:00:00Z");
const WINDOW_MS = 45 * 24 * 60 * 60 * 1000;
const MIN_SPLIT_WINDOW_MS = 6 * 60 * 60 * 1000;
const FILL_RESPONSE_CAP = 2000;
const MAX_REQUESTS = 160;

function safeFloat(value: unknown) {
  const number = Number.parseFloat(String(value ?? 0));
  return Number.isFinite(number) ? number : 0;
}

function extractAddresses(payload: unknown): Set<string> {
  const matches = new Set<string>();
  if (typeof payload === "string") {
    for (const address of payload.match(/0x[a-fA-F0-9]{40}/g) || []) {
      matches.add(address.toLowerCase());
    }
    return matches;
  }
  if (Array.isArray(payload)) {
    for (const value of payload) {
      for (const address of extractAddresses(value)) {
        matches.add(address);
      }
    }
    return matches;
  }
  if (payload && typeof payload === "object") {
    for (const value of Object.values(payload)) {
      for (const address of extractAddresses(value)) {
        matches.add(address);
      }
    }
  }
  return matches;
}

function isoFromMs(timestampMs: number | null | undefined) {
  if (!timestampMs || !Number.isFinite(timestampMs)) {
    return null;
  }
  try {
    return new Date(timestampMs).toISOString();
  } catch {
    return null;
  }
}

export function validateWallet(wallet: string) {
  const text = String(wallet || "").trim();
  if (!/^0x[a-fA-F0-9]{40}$/.test(text)) {
    throw new Error("Wallet must be a valid 42-character EVM address.");
  }
  return text.toLowerCase();
}

async function postInfo(payload: Record<string, unknown>) {
  const response = await fetch(HYPERLIQUID_INFO_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const text = await response.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!response.ok) {
    throw new Error(`Hyperliquid request failed (${response.status}): ${typeof data === "string" ? data : JSON.stringify(data)}`);
  }
  return data;
}

export async function loadTradexyzUniverse() {
  const data = await postInfo({ type: "meta", dex: TRADEXYZ_DEX });
  const universe = Array.isArray(data?.universe) ? data.universe : [];
  return universe.filter((item: any) => item && typeof item === "object");
}

export async function inspectWalletIdentity(wallet: string) {
  const safeWallet = validateWallet(wallet);
  const [rolePayload, abstractionMode, dexAbstraction, subAccounts] = await Promise.all([
    postInfo({ type: "userRole", user: safeWallet }),
    postInfo({ type: "userAbstraction", user: safeWallet }),
    postInfo({ type: "userDexAbstraction", user: safeWallet }),
    postInfo({ type: "subAccounts", user: safeWallet }),
  ]);

  const role = String((rolePayload as any)?.role || "user").trim() || "user";
  const roleDetails = ((rolePayload as any)?.data && typeof (rolePayload as any).data === "object")
    ? (rolePayload as any).data
    : {};
  const linkedAddresses = new Set<string>();
  for (const address of extractAddresses(roleDetails)) linkedAddresses.add(address);
  for (const address of extractAddresses(dexAbstraction)) linkedAddresses.add(address);
  for (const address of extractAddresses(subAccounts)) linkedAddresses.add(address);
  linkedAddresses.delete(safeWallet);

  const notes: string[] = [];
  if (role.toLowerCase() === "agent") {
    const linkedUser = String((roleDetails as any)?.user || "").trim().toLowerCase();
    throw new Error(
      "This address is a Hyperliquid agent/API wallet. Use the actual user or sub-account address instead"
      + (linkedUser ? ` (linked user: ${linkedUser})` : "."),
    );
  }
  if (role.toLowerCase() === "subaccount") {
    const master = String((roleDetails as any)?.master || "").trim().toLowerCase();
    notes.push(
      "This address is a Hyperliquid sub-account. The checker stays pinned to this exact sub-account and does not intentionally roll up the master."
      + (master ? ` Master: ${master}.` : ""),
    );
  } else {
    notes.push("This lookup is strict to the exact address you entered. It does not intentionally merge linked users, sub-accounts, or agents.");
  }

  const abstractionText = String(abstractionMode || "default").trim() || "default";
  if (abstractionText.toLowerCase() !== "default") {
    notes.push(`Hyperliquid reports this address in ${abstractionText} abstraction mode. Linked abstraction addresses can surface the same Trade.xyz history at the protocol layer.`);
  }
  if (!(dexAbstraction == null || dexAbstraction === "" || dexAbstraction === "default" || (typeof dexAbstraction === "object" && Object.keys(dexAbstraction as Record<string, unknown>).length === 0))) {
    notes.push("A Hyperliquid dex-abstraction link exists for this address. If two linked addresses show the same Trade.xyz activity, that linkage is coming from Hyperliquid rather than from this checker.");
  }
  if (linkedAddresses.size) {
    notes.push(`Linked Hyperliquid addresses detected: ${[...linkedAddresses].sort().join(", ")}.`);
  }

  return {
    requested_address: safeWallet,
    queried_address: safeWallet,
    query_scope: "strict_address",
    role,
    role_details: roleDetails,
    abstraction_mode: abstractionText,
    dex_abstraction: dexAbstraction,
    linked_addresses: [...linkedAddresses].sort(),
    notes,
  };
}

async function fetchUserFillsWindow(wallet: string, startTimeMs: number, endTimeMs: number) {
  const data = await postInfo({
    type: "userFillsByTime",
    user: wallet,
    startTime: Math.trunc(startTimeMs),
    endTime: Math.trunc(endTimeMs),
    aggregateByTime: true,
    dex: TRADEXYZ_DEX,
  });
  if (!Array.isArray(data)) {
    throw new Error("Unexpected Hyperliquid fills payload.");
  }
  return data.filter((item) => item && typeof item === "object");
}

function fillKey(fill: any) {
  return [
    String(fill?.hash || ""),
    String(fill?.coin || ""),
    String(fill?.oid || ""),
    String(fill?.time || ""),
    String(fill?.px || ""),
    String(fill?.sz || ""),
    String(fill?.side || ""),
  ].join("|");
}

function isTradexyzFill(fill: any, xyzMarkets: Set<string>) {
  const coin = String(fill?.coin || "").trim();
  return coin.startsWith(TRADEXYZ_PREFIX) || xyzMarkets.has(coin);
}

export async function collectTradexyzFills(
  wallet: string,
  xyzMarkets: Set<string>,
  startTimeMs = DEFAULT_START_MS,
  endTimeMs = Date.now(),
) {
  if (!(endTimeMs > startTimeMs)) {
    throw new Error("End time must be after start time.");
  }

  const windows: Array<[number, number]> = [];
  let cursor = startTimeMs;
  while (cursor < endTimeMs) {
    const windowEnd = Math.min(cursor + WINDOW_MS - 1, endTimeMs);
    windows.push([cursor, windowEnd]);
    cursor = windowEnd + 1;
  }

  const stack = [...windows].reverse();
  const seen = new Set<string>();
  const matched: any[] = [];
  let requestCount = 0;
  let splitCount = 0;
  let truncatedWindowCount = 0;

  while (stack.length) {
    const [start, end] = stack.pop() as [number, number];
    requestCount += 1;
    if (requestCount > MAX_REQUESTS) {
      throw new Error("Trade.xyz volume lookup exceeded the safe request budget. Narrow the range and try again.");
    }

    const fills = await fetchUserFillsWindow(wallet, start, end);
    if (fills.length >= FILL_RESPONSE_CAP && (end - start) > MIN_SPLIT_WINDOW_MS) {
      const midpoint = start + Math.floor((end - start) / 2);
      splitCount += 1;
      stack.push([midpoint + 1, end], [start, midpoint]);
      continue;
    }
    if (fills.length >= FILL_RESPONSE_CAP) {
      truncatedWindowCount += 1;
    }

    for (const fill of fills) {
      if (!isTradexyzFill(fill, xyzMarkets)) continue;
      const key = fillKey(fill);
      if (seen.has(key)) continue;
      seen.add(key);
      matched.push(fill);
    }
  }

  matched.sort((a, b) => Number(b?.time || 0) - Number(a?.time || 0));
  return {
    fills: matched,
    coverage: {
      start_time_ms: startTimeMs,
      end_time_ms: endTimeMs,
      start_time: isoFromMs(startTimeMs),
      end_time: isoFromMs(endTimeMs),
      request_count: requestCount,
      split_count: splitCount,
      truncated_window_count: truncatedWindowCount,
      possible_truncation: truncatedWindowCount > 0,
    },
  };
}

export function summarizeTradexyzFills(
  wallet: string,
  fills: any[],
  universe: any[],
  coverage: Record<string, unknown>,
  identity: Record<string, unknown> = {},
) {
  const markets = new Map<string, any>();
  let totalVolume = 0;
  let buyVolume = 0;
  let sellVolume = 0;
  let firstFillMs: number | null = null;
  let lastFillMs: number | null = null;

  for (const fill of fills) {
    const coin = String(fill?.coin || "").trim();
    const px = safeFloat(fill?.px);
    const sz = Math.abs(safeFloat(fill?.sz));
    if (!coin || !(px > 0) || !(sz > 0)) continue;
    const notional = Math.abs(px * sz);
    totalVolume += notional;

    const fillTime = Number(fill?.time || 0);
    if (fillTime > 0) {
      firstFillMs = firstFillMs == null ? fillTime : Math.min(firstFillMs, fillTime);
      lastFillMs = lastFillMs == null ? fillTime : Math.max(lastFillMs, fillTime);
    }

    const side = String(fill?.side || "").toUpperCase();
    if (side === "B") buyVolume += notional;
    if (side === "A") sellVolume += notional;

    const current = markets.get(coin) || {
      coin,
      volume_usd: 0,
      buy_volume_usd: 0,
      sell_volume_usd: 0,
      fills: 0,
      first_fill_at: null,
      last_fill_at: null,
    };
    current.volume_usd += notional;
    if (side === "B") current.buy_volume_usd += notional;
    if (side === "A") current.sell_volume_usd += notional;
    current.fills += 1;
    if (fillTime > 0) {
      const iso = isoFromMs(fillTime);
      current.first_fill_at = current.first_fill_at ? (fillTime < Date.parse(current.first_fill_at) ? iso : current.first_fill_at) : iso;
      current.last_fill_at = current.last_fill_at ? (fillTime > Date.parse(current.last_fill_at) ? iso : current.last_fill_at) : iso;
    }
    markets.set(coin, current);
  }

  const marketRows = [...markets.values()]
    .map((row) => ({
      ...row,
      volume_usd: Math.round(row.volume_usd * 100) / 100,
      buy_volume_usd: Math.round(row.buy_volume_usd * 100) / 100,
      sell_volume_usd: Math.round(row.sell_volume_usd * 100) / 100,
    }))
    .sort((a, b) => b.volume_usd - a.volume_usd);

  const preview = fills.slice(0, 12).map((fill) => {
    const price = safeFloat(fill?.px);
    const size = Math.abs(safeFloat(fill?.sz));
    return {
      coin: String(fill?.coin || ""),
      time: isoFromMs(Number(fill?.time || 0)),
      side: String(fill?.side || "").toUpperCase(),
      notional_usd: Math.round(Math.abs(price * size) * 100) / 100,
      price,
      size,
    };
  });

  return {
    wallet,
    dex: TRADEXYZ_DEX,
    checked_at: new Date().toISOString(),
    identity,
    summary: {
      total_volume_usd: Math.round(totalVolume * 100) / 100,
      buy_volume_usd: Math.round(buyVolume * 100) / 100,
      sell_volume_usd: Math.round(sellVolume * 100) / 100,
      fill_count: fills.length,
      market_count: marketRows.length,
      first_fill_at: isoFromMs(firstFillMs),
      last_fill_at: isoFromMs(lastFillMs),
      tracked_markets: universe.length,
    },
    coverage,
    markets: marketRows,
    fills_preview: preview,
    tracked_markets: universe
      .map((item: any) => String(item?.name || "").trim())
      .filter((item: string) => Boolean(item)),
  };
}

export async function fetchTradexyzVolume(wallet: string) {
  const identity = await inspectWalletIdentity(wallet);
  const safeWallet = String(identity.queried_address || validateWallet(wallet));
  const universe = await loadTradexyzUniverse();
  const xyzMarkets = new Set<string>(
    universe
      .map((item: any) => String(item?.name || "").trim())
      .filter((item: string) => Boolean(item)),
  );
  const { fills, coverage } = await collectTradexyzFills(safeWallet, xyzMarkets);
  return summarizeTradexyzFills(safeWallet, fills, universe, coverage, identity);
}
