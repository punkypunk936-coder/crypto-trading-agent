import {
  SNAPSHOT_PATH,
  CONTROL_PATH,
  STATE_PATH,
  TRADES_PATH,
  buildSnapshot,
  defaultControl,
  defaultState,
  json,
  readJson,
} from "../lib/dashboard-store";

export async function GET() {
  try {
    const snapshot = await readJson(SNAPSHOT_PATH, null);
    if (snapshot && typeof snapshot === "object" && snapshot.state) {
      return json(snapshot);
    }

    const [state, trades, control] = await Promise.all([
      readJson(STATE_PATH, defaultState()),
      readJson(TRADES_PATH, []),
      readJson(CONTROL_PATH, defaultControl()),
    ]);
    return json(buildSnapshot(state, trades, control));
  } catch (error) {
    return json(
      buildSnapshot(defaultState(), [], defaultControl()),
      { status: 500 },
    );
  }
}
