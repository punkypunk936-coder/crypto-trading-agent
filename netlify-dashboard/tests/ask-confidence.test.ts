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
