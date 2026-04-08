import {
  MARKET_MAP_PATH,
  SNAPSHOT_PATH,
  CONTROL_PATH,
  STATE_PATH,
  TRADE_REVIEWS_PATH,
  TRADES_PATH,
  buildSnapshot,
  defaultControl,
  defaultMarketMap,
  defaultState,
  defaultTradeReviews,
  json,
  readNetlifyStateFallback,
  readGitFallbackJson,
  readJson,
} from "../lib/dashboard-store";

export async function GET() {
  const cacheHeaders = { "Cache-Control": "public, s-maxage=15, stale-while-revalidate=45" };
  try {
    let snapshot = null;
    try {
      snapshot = await readJson(SNAPSHOT_PATH, null);
    } catch {
      snapshot = await readNetlifyStateFallback();
      if (!snapshot) {
        snapshot = await readGitFallbackJson(SNAPSHOT_PATH, null);
      }
    }
    if (snapshot && typeof snapshot === "object" && snapshot.state) {
      return json(snapshot, { headers: cacheHeaders });
    }

    const netlifySnapshot = await readNetlifyStateFallback();
    if (netlifySnapshot) {
      return json(netlifySnapshot, { headers: cacheHeaders });
    }

    const loaders = [
      [STATE_PATH, defaultState()],
      [TRADES_PATH, []],
      [CONTROL_PATH, defaultControl()],
      [MARKET_MAP_PATH, defaultMarketMap()],
      [TRADE_REVIEWS_PATH, defaultTradeReviews()],
    ] as const;
    const [state, trades, control, marketMap, tradeReviews] = await Promise.all(
      loaders.map(async ([path, fallback]) => {
        try {
          return await readJson(path, fallback);
        } catch {
          return await readGitFallbackJson(path, fallback);
        }
      }),
    );
    return json(buildSnapshot(state, trades, control, marketMap, tradeReviews), { headers: cacheHeaders });
  } catch (error) {
    return json(
      buildSnapshot(defaultState(), [], defaultControl(), defaultMarketMap(), defaultTradeReviews()),
      { status: 500, headers: cacheHeaders },
    );
  }
}
