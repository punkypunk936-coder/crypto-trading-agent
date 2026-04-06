import {
  CONTROL_PATH,
  SNAPSHOT_PATH,
  buildSnapshot,
  defaultControl,
  json,
  readJson,
  unauthorized,
  writeJson,
} from "../lib/dashboard-store";

export async function POST(request: Request) {
  if (unauthorized(request)) {
    return new Response("Forbidden", { status: 403 });
  }

  const data = await request.json().catch(() => ({}));
  const active = Boolean(data.active ?? true);
  const reason =
    String(data.reason || "Dashboard kill switch activated").trim() ||
    "Dashboard kill switch activated";
  const timestamp = new Date().toISOString().slice(0, 19).replace("T", " ");

  try {
    const [current, currentSnapshot] = await Promise.all([
      readJson(CONTROL_PATH, defaultControl()),
      readJson(SNAPSHOT_PATH, null),
    ]);
    const next = {
      ...defaultControl(),
      ...(current && typeof current === "object" ? current : {}),
      kill: {
        ...(current?.kill || defaultControl().kill),
        active,
        reason: active ? reason : "",
        requested_at: active ? timestamp : current?.kill?.requested_at || null,
        acknowledged_at: active ? current?.kill?.acknowledged_at || null : timestamp,
      },
    };

    await Promise.all([
      writeJson(CONTROL_PATH, next),
      writeJson(
        SNAPSHOT_PATH,
        currentSnapshot && typeof currentSnapshot === "object" && currentSnapshot.state
          ? {
              ...currentSnapshot,
              control: next,
              server_time: timestamp,
            }
          : buildSnapshot(null, [], next, timestamp),
      ),
    ]);
    return json({ ok: true, control: next });
  } catch (error) {
    return json(
      { ok: false, error: error instanceof Error ? error.message : "Kill update failed" },
      { status: 500 },
    );
  }
}
