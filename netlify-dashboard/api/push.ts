import {
  CONTROL_PATH,
  SNAPSHOT_PATH,
  STATE_PATH,
  TRADES_PATH,
  buildSnapshot,
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
      : buildSnapshot(data.state, data.trades || [], data.control);

    await writeJson(SNAPSHOT_PATH, snapshot);
    await writeJson(STATE_PATH, snapshot.state || data.state || {});
    await writeJson(TRADES_PATH, Array.isArray(data.trades) ? data.trades : (snapshot.trades || []));
    await writeJson(CONTROL_PATH, snapshot.control || data.control || {});

    return json({ ok: true, cycle: snapshot?.state?.cycle_number || data?.state?.cycle_number || 0 });
  } catch (error) {
    return json(
      { ok: false, error: error instanceof Error ? error.message : "Push failed" },
      { status: 500 },
    );
  }
}
