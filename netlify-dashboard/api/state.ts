import {
  CHALLENGER_REPORT_PATH,
  MARKET_MAP_PATH,
  SNAPSHOT_PATH,
  DECISION_REVIEW_PATH,
  POLICY_HEALTH_REPORT_PATH,
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

type DashboardCandidate = {
  source: string;
  snapshot: any;
};

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

function snapshotStampLabel(snapshot: any) {
  return snapshot?.server_time || snapshot?.state?.last_cycle || null;
}

function summarizeCandidates(candidates: DashboardCandidate[]) {
  return candidates.map((candidate) => ({
    source: candidate.source,
    available: Boolean(candidate.snapshot && typeof candidate.snapshot === "object"),
    rich: hasRichSnapshot(candidate.snapshot),
    cycle_number: snapshotCycle(candidate.snapshot),
    stamp: snapshotStampLabel(candidate.snapshot),
  }));
}

function withStateSource(snapshot: any, source: string, candidates: DashboardCandidate[]) {
  const runtime = snapshot?.runtime && typeof snapshot.runtime === "object" ? snapshot.runtime : {};
  return {
    ...snapshot,
    runtime: {
      ...runtime,
      dashboard_state_source: source,
      dashboard_state_candidates: summarizeCandidates(candidates),
    },
  };
}

function pickFreshestSnapshot(candidates: DashboardCandidate[]) {
  let winner: DashboardCandidate | null = null;
  let winnerRank: [number, number] = [-1, -1];
  for (const candidate of candidates) {
    const snapshot = candidate.snapshot;
    if (!hasRichSnapshot(snapshot)) {
      continue;
    }
    const rank: [number, number] = [snapshotCycle(snapshot), snapshotStamp(snapshot)];
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

function pickFreshestThinFallback(candidates: DashboardCandidate[]) {
  let winner: DashboardCandidate | null = null;
  let winnerRank: [number, number] = [-1, -1];
  for (const candidate of candidates) {
    const snapshot = candidate.snapshot;
    if (!snapshot || typeof snapshot !== "object" || !snapshot.state) {
      continue;
    }
    const rank: [number, number] = [snapshotCycle(snapshot), snapshotStamp(snapshot)];
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
  const cacheHeaders = { "Cache-Control": "no-store, max-age=0" };
  try {
    let blobSnapshot = null;
    try {
      blobSnapshot = await readJson(SNAPSHOT_PATH, null);
    } catch {
      blobSnapshot = null;
    }

    const netlifySnapshot = await readNetlifyStateFallback();
    const gitSnapshot = await readGitFallbackJson(SNAPSHOT_PATH, null);

    const candidates: DashboardCandidate[] = [
      { source: "vercel_blob", snapshot: blobSnapshot },
      { source: "netlify_fallback", snapshot: netlifySnapshot },
      { source: "github_fallback", snapshot: gitSnapshot },
    ];

    const freshestRichSnapshot = pickFreshestSnapshot(candidates);
    if (freshestRichSnapshot) {
      return json(
        withStateSource(freshestRichSnapshot.snapshot, freshestRichSnapshot.source, candidates),
        { headers: cacheHeaders },
      );
    }

    const thinFallbackSnapshot = pickFreshestThinFallback(candidates);
    if (thinFallbackSnapshot?.snapshot && typeof thinFallbackSnapshot.snapshot === "object" && thinFallbackSnapshot.snapshot.state) {
      const snapshot = thinFallbackSnapshot.snapshot;
      return json(
        withStateSource(
          buildSnapshot(
            snapshot.state,
            snapshot.trades || [],
            snapshot.control || defaultControl(),
            marketMapFromState(snapshot.state),
            snapshot.trade_reviews || defaultTradeReviews(),
            snapshot.server_time,
            snapshot.decision_review_report || {},
            snapshot.challenger_report || {},
            snapshot.playbook_distiller_report || {},
            snapshot.policy_health_report || {},
          ),
          thinFallbackSnapshot.source,
          candidates,
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
      [POLICY_HEALTH_REPORT_PATH, {}],
    ] as const;
    const [state, trades, control, marketMap, tradeReviews, decisionReviewReport, challengerReport, playbookDistillerReport, policyHealthReport] = await Promise.all(
      loaders.map(async ([path, fallback]) => {
        try {
          return await readJson(path, fallback);
        } catch {
          return await readGitFallbackJson(path, fallback);
        }
      }),
    );
    return json(
      withStateSource(
        buildSnapshot(state, trades, control, marketMap, tradeReviews, undefined, decisionReviewReport, challengerReport, playbookDistillerReport, policyHealthReport),
        "component_store",
        candidates,
      ),
      { headers: cacheHeaders },
    );
  } catch (error) {
    return json(
      buildSnapshot(defaultState(), [], defaultControl(), defaultMarketMap(), defaultTradeReviews(), undefined, {}, {}, {}, {}),
      { status: 500, headers: cacheHeaders },
    );
  }
}
