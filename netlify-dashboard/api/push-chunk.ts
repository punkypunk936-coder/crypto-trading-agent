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
  json,
  readJson,
  unauthorized,
  writeJson,
} from "../lib/dashboard-store";

function cleanSessionId(value: unknown) {
  return String(value || "")
    .trim()
    .replace(/[^a-zA-Z0-9._-]/g, "")
    .slice(0, 96);
}

function chunkPath(sessionId: string, index: number) {
  return `dashboard/push-chunks/${sessionId}/${index}.json`;
}

async function persistPayload(data: any) {
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

  return snapshot;
}

export async function POST(request: Request) {
  if (unauthorized(request)) {
    return new Response("Forbidden", { status: 403 });
  }

  const data = await request.json().catch(() => null);
  const sessionId = cleanSessionId(data?.session_id);
  const chunkIndex = Number.parseInt(String(data?.chunk_index ?? ""), 10);
  const chunkCount = Number.parseInt(String(data?.chunk_count ?? ""), 10);
  const chunk = typeof data?.chunk === "string" ? data.chunk : "";

  if (!sessionId || !Number.isInteger(chunkIndex) || !Number.isInteger(chunkCount) || chunkCount <= 0 || chunkIndex < 0 || chunkIndex >= chunkCount || !chunk) {
    return json({ ok: false, error: "Invalid chunk payload" }, { status: 400 });
  }
  if (chunkCount > 128) {
    return json({ ok: false, error: "Too many chunks" }, { status: 400 });
  }

  try {
    await writeJson(chunkPath(sessionId, chunkIndex), { chunk });

    const pieces: string[] = [];
    const missing: number[] = [];
    for (let index = 0; index < chunkCount; index += 1) {
      if (index === chunkIndex) {
        pieces.push(chunk);
        continue;
      }
      const piece = await readJson(chunkPath(sessionId, index), null);
      if (!piece || typeof piece.chunk !== "string") {
        missing.push(index);
        continue;
      }
      pieces.push(piece.chunk);
    }

    if (missing.length > 0) {
      return json({
        ok: true,
        assembled: false,
        session_id: sessionId,
        received_index: chunkIndex,
        missing,
      }, { status: 202 });
    }

    const payloadText = pieces.join("");
    const payload = JSON.parse(payloadText);
    const snapshot = await persistPayload(payload);

    return json({
      ok: true,
      assembled: true,
      storage: "chunked",
      session_id: sessionId,
      chunks: chunkCount,
      bytes: payloadText.length,
      cycle: snapshot?.state?.cycle_number || payload?.state?.cycle_number || 0,
      server_time: snapshot?.server_time || null,
    });
  } catch (error) {
    return json(
      { ok: false, error: error instanceof Error ? error.message : "Chunk push failed" },
      { status: 500 },
    );
  }
}
