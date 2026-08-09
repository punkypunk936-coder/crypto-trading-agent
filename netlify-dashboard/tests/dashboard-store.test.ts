import assert from "node:assert/strict";
import test from "node:test";

import {
  canonicalizeSnapshot,
  compareSnapshots,
  snapshotFromPayload,
  snapshotManifest,
} from "../lib/dashboard-store";

test("full snapshots preserve every intelligence and execution field", () => {
  const snapshot = {
    version: 1234,
    updatedAt: "2026-08-09T05:00:00.000Z",
    state: {
      cycle_number: 77,
      signals: {
        NVDA: {
          thesis: "Durable AI compute demand",
          trigger: 190,
          entry: 191,
          invalidation: 181,
          targets: [205, 220],
          RR: 2.4,
          risk: { max_loss_usd: 40 },
          confidence: "HIGH",
          timestamp: "2026-08-09T04:59:00Z",
          supporting_market_context: { kospi: "confirming" },
        },
      },
    },
    asia_session: { regime: "risk_on" },
    global_market_context: { anchor: "NASDAQ" },
    future_field: { must_survive: true },
  };

  const canonical = snapshotFromPayload({ snapshot });
  assert.deepEqual(canonical.state.signals.NVDA, snapshot.state.signals.NVDA);
  assert.deepEqual(canonical.asia_session, snapshot.asia_session);
  assert.deepEqual(canonical.global_market_context, snapshot.global_market_context);
  assert.deepEqual(canonical.future_field, snapshot.future_field);
});

test("compact payloads retain fields unknown to the older derived schema", () => {
  const canonical = snapshotFromPayload({
    state: { cycle_number: 78, signals: {} },
    missed_move_report: { count: 3 },
    asset_dossiers: { NVDA: { thesis: "Compute leader" } },
    llm_referee_report: { verdict: "hold" },
    supporting_market_context: { spx: "range" },
  });

  assert.equal(canonical.missed_move_report.count, 3);
  assert.equal(canonical.asset_dossiers.NVDA.thesis, "Compute leader");
  assert.equal(canonical.llm_referee_report.verdict, "hold");
  assert.equal(canonical.supporting_market_context.spx, "range");
});

test("version outranks cycle and manifests carry canonical freshness", () => {
  const older = canonicalizeSnapshot({
    version: 100,
    updatedAt: "2026-08-09T04:00:00Z",
    state: { cycle_number: 999 },
  });
  const newer = canonicalizeSnapshot({
    version: 200,
    updatedAt: "2026-08-09T05:00:00Z",
    state: { cycle_number: 10 },
  });

  assert.equal(compareSnapshots(newer, older), 1);
  assert.equal(compareSnapshots(older, newer), -1);
  assert.deepEqual(snapshotManifest(newer), {
    schemaVersion: 2,
    version: 200,
    updatedAt: "2026-08-09T05:00:00Z",
    cycle_number: 10,
    server_time: null,
    snapshot_path: "dashboard/dashboard_snapshot.json",
  });
});
