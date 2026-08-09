import {
  MANIFEST_PATH,
  SNAPSHOT_PATH,
  buildSnapshot,
  canonicalizeSnapshot,
  compareSnapshots,
  defaultControl,
  defaultMarketMap,
  defaultState,
  defaultTradeReviews,
  json,
  readGitFallbackJson,
  readGitCanonicalSnapshot,
  readJson,
  readNetlifyStateFallback,
  snapshotCycle,
  snapshotVersion,
} from "../lib/dashboard-store";

type DashboardCandidate = {
  source: string;
  snapshot: any;
};

function summarizeCandidates(candidates: DashboardCandidate[]) {
  return candidates.map(({ source, snapshot }) => ({
    source,
    available: Boolean(snapshot?.state || snapshot?.version || snapshot?.updatedAt),
    cycle_number: snapshotCycle(snapshot),
    version: snapshotVersion(snapshot),
    updatedAt: snapshot?.updatedAt || null,
    stamp: snapshot?.server_time || snapshot?.state?.last_cycle || null,
  }));
}

function withStateSource(snapshot: any, source: string, candidates: DashboardCandidate[]) {
  const canonical = canonicalizeSnapshot(snapshot);
  const runtime = canonical.runtime && typeof canonical.runtime === "object" ? canonical.runtime : {};
  return {
    ...canonical,
    runtime: {
      ...runtime,
      dashboard_state_source: source,
      dashboard_state_candidates: summarizeCandidates(candidates),
    },
  };
}

function fallbackSnapshot(snapshot: any) {
  if (!snapshot?.state) return null;
  if (snapshot.action_board && snapshot.market_map_summary && snapshot.learning_summary) {
    return canonicalizeSnapshot(snapshot);
  }
  const derived = buildSnapshot(
    snapshot.state,
    snapshot.trades || [],
    snapshot.control || defaultControl(),
    snapshot.market_map || defaultMarketMap(),
    snapshot.trade_reviews || defaultTradeReviews(),
    snapshot.server_time,
    snapshot.decision_review_report || {},
    snapshot.challenger_report || {},
    snapshot.playbook_distiller_report || {},
    snapshot.policy_health_report || {},
  );
  return canonicalizeSnapshot({ ...snapshot, ...derived });
}

export async function GET() {
  const cacheHeaders = {
    "Cache-Control": "no-store, no-cache, max-age=0, must-revalidate",
    Pragma: "no-cache",
  };
  const candidates: DashboardCandidate[] = [];

  let blobSnapshot: any = null;
  let blobManifest: any = null;
  try {
    [blobSnapshot, blobManifest] = await Promise.all([
      readJson(SNAPSHOT_PATH, null),
      readJson(MANIFEST_PATH, null),
    ]);
  } catch {
    blobSnapshot = null;
    blobManifest = null;
  }
  candidates.push({ source: "vercel_blob", snapshot: blobSnapshot || blobManifest });

  const gitManifest = await readGitFallbackJson(MANIFEST_PATH, null);
  candidates.push({ source: "github_canonical_manifest", snapshot: gitManifest });

  const blobIsCurrent = Boolean(
    blobSnapshot?.state && (!gitManifest || compareSnapshots(blobSnapshot, gitManifest) >= 0),
  );
  if (blobIsCurrent) {
    return json(withStateSource(blobSnapshot, "vercel_blob", candidates), { headers: cacheHeaders });
  }

  const gitSnapshot = await readGitCanonicalSnapshot(gitManifest);
  candidates.push({ source: "github_canonical", snapshot: gitSnapshot });
  if (gitSnapshot?.state) {
    return json(withStateSource(fallbackSnapshot(gitSnapshot), "github_canonical", candidates), { headers: cacheHeaders });
  }

  if (blobSnapshot?.state) {
    return json(withStateSource(fallbackSnapshot(blobSnapshot), "vercel_blob_degraded", candidates), { headers: cacheHeaders });
  }

  const netlifySnapshot = await readNetlifyStateFallback();
  candidates.push({ source: "netlify_emergency_fallback", snapshot: netlifySnapshot });
  if (netlifySnapshot?.state) {
    return json(withStateSource(fallbackSnapshot(netlifySnapshot), "netlify_emergency_fallback", candidates), { headers: cacheHeaders });
  }

  return json(
    withStateSource(
      buildSnapshot(defaultState(), [], defaultControl(), defaultMarketMap(), defaultTradeReviews()),
      "default_offline",
      candidates,
    ),
    { status: 503, headers: cacheHeaders },
  );
}
