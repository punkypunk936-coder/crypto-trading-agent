import {
  CHALLENGER_REPORT_PATH,
  CONTROL_PATH,
  DECISION_REVIEW_PATH,
  MARKET_MAP_PATH,
  POLICY_HEALTH_REPORT_PATH,
  PLAYBOOK_DISTILLER_PATH,
  SNAPSHOT_PATH,
  STATE_PATH,
  TRADE_REVIEWS_PATH,
  TRADES_PATH,
  buildSnapshot,
  forwardNetlifyPush,
  json,
  unauthorized,
  writeJson,
} from "../lib/dashboard-store";

export async function POST(request: Request) {
  if (unauthorized(request)) {
    return new Response("Forbidden", { status: 403 });
  }

  const data = await request.json().catch(() => null);
  if (!data || (!data.snapshot && !data.state)) {
    return new Response("Missing snapshot/state in payload", { status: 400 });
  }

  try {
    const snapshot = data.snapshot && typeof data.snapshot === "object" && data.snapshot.state
      ? data.snapshot
      : buildSnapshot(
          data.state,
          data.trades || [],
          data.control,
          data.market_map,
          data.trade_reviews,
          undefined,
          data.decision_review_report,
          data.challenger_report,
          data.playbook_distiller_report,
          data.policy_health_report,
        );

    await writeJson(SNAPSHOT_PATH, snapshot);
    await writeJson(STATE_PATH, snapshot.state || data.state || {});
    await writeJson(TRADES_PATH, Array.isArray(data.trades) ? data.trades : (snapshot.trades || []));
    await writeJson(CONTROL_PATH, snapshot.control || data.control || {});
    await writeJson(MARKET_MAP_PATH, snapshot.market_map || data.market_map || {});
    await writeJson(TRADE_REVIEWS_PATH, snapshot.trade_reviews || data.trade_reviews || {});
    await writeJson(DECISION_REVIEW_PATH, snapshot.decision_review_report || data.decision_review_report || {});
    await writeJson(CHALLENGER_REPORT_PATH, snapshot.challenger_report || data.challenger_report || {});
    await writeJson(PLAYBOOK_DISTILLER_PATH, snapshot.playbook_distiller_report || data.playbook_distiller_report || {});
    await writeJson(POLICY_HEALTH_REPORT_PATH, snapshot.policy_health_report || data.policy_health_report || {});

    return json({ ok: true, cycle: snapshot?.state?.cycle_number || data?.state?.cycle_number || 0 });
  } catch (error) {
    const snapshot = data.snapshot && typeof data.snapshot === "object" && data.snapshot.state
      ? data.snapshot
      : buildSnapshot(
          data.state,
          data.trades || [],
          data.control,
          data.market_map,
          data.trade_reviews,
          undefined,
          data.decision_review_report,
          data.challenger_report,
          data.playbook_distiller_report,
          data.policy_health_report,
        );
    const forwardPayload = {
      snapshot,
      state: snapshot?.state || data.state || {},
      trades: Array.isArray(snapshot?.trades) ? snapshot.trades : (data.trades || []),
      control: snapshot?.control || data.control || {},
      market_map: snapshot?.market_map || data.market_map || {},
      trade_reviews: snapshot?.trade_reviews || data.trade_reviews || {},
      decision_review_report: snapshot?.decision_review_report || data.decision_review_report || {},
      challenger_report: snapshot?.challenger_report || data.challenger_report || {},
      playbook_distiller_report: snapshot?.playbook_distiller_report || data.playbook_distiller_report || {},
      policy_health_report: snapshot?.policy_health_report || data.policy_health_report || {},
    };
    const forwarded = await forwardNetlifyPush(forwardPayload, request.headers.get("X-Token") || "");
    if (forwarded.ok) {
      return json(
        typeof forwarded.data === "object" && forwarded.data !== null
          ? { ...forwarded.data, fallback: "netlify", storage: "fallback" }
          : { ok: true, fallback: "netlify" },
      );
    }
    return json(
      {
        ok: false,
        error: error instanceof Error ? error.message : "Push failed",
        fallback_error: forwarded.data,
      },
      { status: 500 },
    );
  }
}
