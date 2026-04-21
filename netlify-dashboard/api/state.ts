import {
  CHALLENGER_REPORT_PATH,
  MARKET_MAP_PATH,
  SNAPSHOT_PATH,
  DECISION_REVIEW_PATH,
  PLAYBOOK_DISTILLER_PATH,
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
  marketMapFromState,
  readNetlifyStateFallback,
  readGitFallbackJson,
  readJson,
} from "../lib/dashboard-store";

function hasRichSnapshot(snapshot: any) {
  return Boolean(
    snapshot &&
    typeof snapshot === "object" &&
    snapshot.state &&
    snapshot.action_board &&
    snapshot.market_map_summary &&
    snapshot.learning_summary &&
    snapshot.control &&
    snapshot.trade_reviews,
  );
}

function snapshotCycle(snapshot: any) {
  const cycle = Number(snapshot?.state?.cycle_number ?? snapshot?.cycle_number ?? 0);
  return Number.isFinite(cycle) ? cycle : 0;
}

function snapshotStamp(snapshot: any) {
  const raw = String(snapshot?.server_time || snapshot?.state?.last_cycle || "").trim();
  if (!raw) {
    return 0;
  }
  const normalized = raw.includes("T") ? raw : raw.replace(" ", "T");
  const parsed = Date.parse(normalized);
  return Number.isFinite(parsed) ? parsed : 0;
}

function pickFreshestSnapshot(candidates: any[]) {
  let winner: any = null;
  let winnerRank: [number, number] = [-1, -1];
  for (const candidate of candidates) {
    if (!hasRichSnapshot(candidate)) {
      continue;
    }
    const rank: [number, number] = [snapshotCycle(candidate), snapshotStamp(candidate)];
    if (
      rank[0] > winnerRank[0] ||
      (rank[0] === winnerRank[0] && rank[1] > winnerRank[1])
    ) {
      winner = candidate;
      winnerRank = rank;
    }
  }
  return winner;
}

function pickFreshestThinFallback(candidates: any[]) {
  let winner: any = null;
  let winnerRank: [number, number] = [-1, -1];
  for (const candidate of candidates) {
    if (!candidate || typeof candidate !== "object" || !candidate.state) {
      continue;
    }
    const rank: [number, number] = [snapshotCycle(candidate), snapshotStamp(candidate)];
    if (
      rank[0] > winnerRank[0] ||
      (rank[0] === winnerRank[0] && rank[1] > winnerRank[1])
    ) {
      winner = candidate;
      winnerRank = rank;
    }
  }
  return winner;
}

export async function GET() {
  const cacheHeaders = { "Cache-Control": "public, s-maxage=15, stale-while-revalidate=45" };
  try {
    let blobSnapshot = null;
    try {
      blobSnapshot = await readJson(SNAPSHOT_PATH, null);
    } catch {
      blobSnapshot = null;
    }

    const netlifySnapshot = await readNetlifyStateFallback();
    const gitSnapshot = await readGitFallbackJson(SNAPSHOT_PATH, null);

    const freshestRichSnapshot = pickFreshestSnapshot([blobSnapshot, netlifySnapshot, gitSnapshot]);
    if (freshestRichSnapshot) {
      return json(freshestRichSnapshot, { headers: cacheHeaders });
    }

    const thinFallbackSnapshot = pickFreshestThinFallback([blobSnapshot, netlifySnapshot, gitSnapshot]);
    if (thinFallbackSnapshot && typeof thinFallbackSnapshot === "object" && thinFallbackSnapshot.state) {
      return json(
        buildSnapshot(
          thinFallbackSnapshot.state,
          thinFallbackSnapshot.trades || [],
          thinFallbackSnapshot.control || defaultControl(),
          marketMapFromState(thinFallbackSnapshot.state),
          thinFallbackSnapshot.trade_reviews || defaultTradeReviews(),
          thinFallbackSnapshot.server_time,
          thinFallbackSnapshot.decision_review_report || {},
          thinFallbackSnapshot.challenger_report || {},
          thinFallbackSnapshot.playbook_distiller_report || {},
        ),
        { headers: cacheHeaders },
      );
    }

    const loaders = [
      [STATE_PATH, defaultState()],
      [TRADES_PATH, []],
      [CONTROL_PATH, defaultControl()],
      [MARKET_MAP_PATH, defaultMarketMap()],
      [TRADE_REVIEWS_PATH, defaultTradeReviews()],
      [DECISION_REVIEW_PATH, {}],
      [CHALLENGER_REPORT_PATH, {}],
      [PLAYBOOK_DISTILLER_PATH, {}],
    ] as const;
    const [state, trades, control, marketMap, tradeReviews, decisionReviewReport, challengerReport, playbookDistillerReport] = await Promise.all(
      loaders.map(async ([path, fallback]) => {
        try {
          return await readJson(path, fallback);
        } catch {
          return await readGitFallbackJson(path, fallback);
        }
      }),
    );
    return json(
      buildSnapshot(state, trades, control, marketMap, tradeReviews, undefined, decisionReviewReport, challengerReport, playbookDistillerReport),
      { headers: cacheHeaders },
    );
  } catch (error) {
    return json(
      buildSnapshot(defaultState(), [], defaultControl(), defaultMarketMap(), defaultTradeReviews(), undefined, {}, {}, {}),
      { status: 500, headers: cacheHeaders },
    );
  }
}
