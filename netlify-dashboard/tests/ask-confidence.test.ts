import assert from "node:assert/strict";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const confidence = require("../public/ask-confidence.js");

function bar(t: string, low: number, high: number) {
  const stamp = Date.parse(t);
  return { t: stamp, T: stamp + 3_599_000, o: low, l: low, h: high, c: high };
}

test("uses reward/risk as the break-even anchor", () => {
  const result = confidence.calibrate({
    rr: 2,
    score: 4.5,
    barsCount: 240,
    candleFresh: true,
    marketFresh: true,
    scheduledFresh: false,
    earningsMapped: true,
    history: { samples: 0, wins: 0 },
  });
  assert.equal(result.breakEvenPct, 33);
  assert.ok(result.probabilityPct > result.breakEvenPct);
  assert.ok(result.probabilityPct <= 68);
});

test("stale candles cannot produce a supported trade confidence", () => {
  const result = confidence.calibrate({
    rr: 2,
    score: 6,
    barsCount: 240,
    candleFresh: false,
    marketFresh: true,
    scheduledFresh: true,
    scheduledDirection: "LONG",
    direction: "LONG",
    scheduledProbability: 0.9,
    history: { samples: 40, wins: 40 },
  });
  assert.equal(result.decisionSupported, false);
  assert.ok(result.evidenceQuality < 80);
});

test("settles target first as a win and invalidation first as a loss", () => {
  const calls = [
    {
      id: "win",
      ticker: "TEST",
      direction: "LONG",
      analysedAt: "2026-08-01T00:00:00.000Z",
      horizonHours: 24,
      target: 110,
      invalidation: 95,
      probability: 0.5,
    },
    {
      id: "loss",
      ticker: "LOSS",
      direction: "SHORT",
      analysedAt: "2026-08-01T00:00:00.000Z",
      horizonHours: 24,
      target: 90,
      invalidation: 105,
      probability: 0.5,
    },
  ];
  const win = confidence.settleForecasts(calls, "TEST", [bar("2026-08-01T00:00:00.000Z", 99, 111)]);
  const loss = confidence.settleForecasts(win, "LOSS", [bar("2026-08-01T00:00:00.000Z", 99, 106)]);
  assert.equal(loss[0].outcome, 1);
  assert.equal(loss[1].outcome, 0);
});

test("same-candle ambiguity is scored conservatively as wrong", () => {
  const calls = [{
    ticker: "TEST",
    direction: "LONG",
    analysedAt: "2026-08-01T00:00:00.000Z",
    horizonHours: 24,
    target: 110,
    invalidation: 95,
    probability: 0.6,
  }];
  const settled = confidence.settleForecasts(calls, "TEST", [bar("2026-08-01T00:00:00.000Z", 94, 111)]);
  assert.equal(settled[0].outcome, 0);
  assert.equal(settled[0].outcomeReason, "target_and_invalidation_same_candle");
});

test("an unresolved forecast expires as a miss after its horizon", () => {
  const calls = [{
    ticker: "TEST",
    direction: "LONG",
    analysedAt: "2026-08-01T00:00:00.000Z",
    horizonHours: 24,
    target: 120,
    invalidation: 80,
    probability: 0.6,
  }];
  const bars = [
    bar("2026-08-01T00:00:00.000Z", 99, 101),
    bar("2026-08-02T01:00:00.000Z", 99, 101),
  ];
  const settled = confidence.settleForecasts(calls, "TEST", bars);
  assert.equal(settled[0].outcome, 0);
  assert.equal(settled[0].outcomeReason, "horizon_expired_before_target");
});

test("a conditional call cannot lose before its entry is reached", () => {
  const calls = [{
    id: "conditional-long",
    ticker: "TEST",
    direction: "LONG",
    analysedAt: "2026-08-01T00:00:00.000Z",
    horizonHours: 24,
    current: 100,
    entry: 105,
    target: 110,
    invalidation: 95,
    probability: 0.6,
    status: "pending_entry",
  }];
  const pending = confidence.settleForecasts(calls, "TEST", [bar("2026-08-01T00:00:00.000Z", 94, 104)]);
  assert.equal(pending[0].outcome, null);
  assert.equal(pending[0].status, "pending_entry");

  const active = confidence.settleForecasts(pending, "TEST", [bar("2026-08-01T01:00:00.000Z", 100, 106)]);
  assert.equal(active[0].outcome, null);
  assert.equal(active[0].status, "active");
  assert.ok(active[0].activatedAt);

  const resolved = confidence.settleForecasts(active, "TEST", [bar("2026-08-01T02:00:00.000Z", 103, 111)]);
  assert.equal(resolved[0].outcome, 1);
  assert.equal(resolved[0].outcomeReason, "target_first");
});

test("a setup that never reaches entry expires without a fake loss", () => {
  const calls = [{
    id: "never-entered",
    ticker: "TEST",
    direction: "LONG",
    analysedAt: "2026-08-01T00:00:00.000Z",
    horizonHours: 1,
    current: 100,
    entry: 105,
    target: 110,
    invalidation: 95,
    probability: 0.6,
    status: "pending_entry",
  }];
  const settled = confidence.settleForecasts(calls, "TEST", [
    bar("2026-08-01T00:00:00.000Z", 96, 104),
    bar("2026-08-01T00:59:00.000Z", 96, 104),
  ]);
  assert.equal(settled[0].outcome, null);
  assert.equal(settled[0].status, "expired_untriggered");
  assert.equal(settled[0].outcomeReason, "entry_never_reached");
});

test("performance summary separates exact ticker history from similar setups", () => {
  const records = [
    { ticker: "AMZN", direction: "LONG", setupType: "long_above_level", outcome: 1, resolutionSource: "venue_candle_path_v2" },
    { ticker: "AMZN", direction: "LONG", setupType: "long_above_level", outcome: 0, resolutionSource: "venue_candle_path_v2" },
    { ticker: "NVDA", direction: "LONG", setupType: "long_above_level", outcome: 1, resolutionSource: "venue_candle_path_v2" },
  ];
  const summary = confidence.performanceSummary(records, "AMZN", "LONG", 2, "equity", "long_above_level");
  assert.deepEqual(summary.ticker, { samples: 2, wins: 1, hitRate: 0.5, hitRatePct: 50 });
  assert.deepEqual(summary.setup, { samples: 3, wins: 2, hitRate: 2 / 3, hitRatePct: 67 });
});

test("historical misses pull future confidence down", () => {
  const optimistic = confidence.calibrate({
    rr: 2,
    score: 5,
    barsCount: 240,
    candleFresh: true,
    marketFresh: true,
    earningsMapped: true,
    history: { samples: 0, wins: 0 },
  });
  const corrected = confidence.calibrate({
    rr: 2,
    score: 5,
    barsCount: 240,
    candleFresh: true,
    marketFresh: true,
    earningsMapped: true,
    history: { samples: 12, wins: 2, hitRate: 2 / 12, brier: 0.4 },
  });
  assert.ok(corrected.probabilityPct < optimistic.probabilityPct);
});

test("new-listing intraday analysis stays capped and cannot fake certainty", () => {
  const result = confidence.calibrate({
    rr: 3,
    score: 6,
    barsCount: 61,
    candleFresh: true,
    marketFresh: true,
    scheduledFresh: true,
    scheduledDirection: "SHORT",
    direction: "SHORT",
    scheduledProbability: 0.9,
    limitedHistory: true,
    history: { samples: 30, wins: 30 },
  });
  assert.ok(result.probabilityPct <= 58);
});

test("server-settled records replace unresolved local copies", () => {
  const local = [{ id: "spcx-short", ticker: "SPCX", outcome: null }];
  const server = [{ id: "spcx-short", ticker: "SPCX", outcome: 1, resolutionSource: "venue_candle_path_v2" }];
  const merged = confidence.mergeRecords(local, server);
  assert.equal(merged.length, 1);
  assert.equal(merged[0].outcome, 1);
});
