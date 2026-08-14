(function attachPunkyConfidence(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.PunkyConfidence = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function buildPunkyConfidence() {
  "use strict";

  const STORAGE_KEY = "punky.ask-forecasts.v1";
  const HORIZON_HOURS = 24;
  const MAX_RECORDS = 500;

  function clamp(value, low, high) {
    return Math.max(low, Math.min(high, Number(value) || 0));
  }

  function finite(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function rrBucket(value) {
    const rr = finite(value, 0);
    if (rr < 1.5) return "under_1_5";
    if (rr < 2.5) return "1_5_to_2_5";
    return "over_2_5";
  }

  function normalizeRecords(value) {
    if (!Array.isArray(value)) return [];
    return value.filter((row) => row && typeof row === "object").slice(-MAX_RECORDS);
  }

  function load(storage) {
    try {
      const source = storage || (typeof localStorage !== "undefined" ? localStorage : null);
      if (!source) return [];
      return normalizeRecords(JSON.parse(source.getItem(STORAGE_KEY) || "[]"));
    } catch (_) {
      return [];
    }
  }

  function save(records, storage) {
    try {
      const target = storage || (typeof localStorage !== "undefined" ? localStorage : null);
      if (!target) return false;
      target.setItem(STORAGE_KEY, JSON.stringify(normalizeRecords(records)));
      return true;
    } catch (_) {
      return false;
    }
  }

  function barTime(bar) {
    return finite(bar && (bar.t || bar.T), 0);
  }

  function barCloseTime(bar) {
    return finite(bar && (bar.T || bar.t), 0);
  }

  function settleForecasts(records, ticker, bars) {
    const normalizedTicker = String(ticker || "").trim().toUpperCase();
    const orderedBars = (Array.isArray(bars) ? bars : [])
      .filter((bar) => bar && barTime(bar) > 0)
      .slice()
      .sort((left, right) => barTime(left) - barTime(right));
    if (!normalizedTicker || !orderedBars.length) return normalizeRecords(records);

    return normalizeRecords(records).map((source) => {
      const row = { ...source };
      if (row.outcome === 0 || row.outcome === 1) return row;
      if (String(row.ticker || "").toUpperCase() !== normalizedTicker) return row;

      const startedAt = Date.parse(String(row.analysedAt || ""));
      const horizonHours = clamp(finite(row.horizonHours, HORIZON_HOURS), 1, 168);
      const expiresAt = startedAt + (horizonHours * 60 * 60 * 1000);
      if (!Number.isFinite(startedAt) || !Number.isFinite(expiresAt)) return row;

      const relevant = orderedBars.filter((bar) => {
        const stamp = barTime(bar);
        return stamp >= startedAt && stamp <= expiresAt;
      });
      if (!relevant.length) return row;

      // Do not invent an outcome if the available chart starts well after the call.
      if (barTime(relevant[0]) > startedAt + (2 * 60 * 60 * 1000)) return row;

      const direction = String(row.direction || "").toUpperCase();
      const target = finite(row.target, NaN);
      const invalidation = finite(row.invalidation, NaN);
      if (!Number.isFinite(target) || !Number.isFinite(invalidation)) return row;

      for (const bar of relevant) {
        const high = finite(bar.h, NaN);
        const low = finite(bar.l, NaN);
        if (!Number.isFinite(high) || !Number.isFinite(low)) continue;
        const targetHit = direction === "LONG" ? high >= target : low <= target;
        const invalidHit = direction === "LONG" ? low <= invalidation : high >= invalidation;
        if (!targetHit && !invalidHit) continue;

        const ambiguous = targetHit && invalidHit;
        row.outcome = ambiguous || invalidHit ? 0 : 1;
        row.outcomeReason = ambiguous
          ? "target_and_invalidation_same_candle"
          : invalidHit ? "invalidation_first" : "target_first";
        row.resolvedAt = new Date(barCloseTime(bar) || barTime(bar)).toISOString();
        return row;
      }

      const latestClose = Math.max(...orderedBars.map(barCloseTime));
      if (latestClose >= expiresAt) {
        row.outcome = 0;
        row.outcomeReason = "horizon_expired_before_target";
        row.resolvedAt = new Date(expiresAt).toISOString();
      }
      return row;
    });
  }

  function brierScore(rows) {
    if (!rows.length) return null;
    const total = rows.reduce((sum, row) => {
      const probability = clamp(finite(row.probability, 0.5), 0.01, 0.99);
      return sum + ((probability - Number(row.outcome)) ** 2);
    }, 0);
    return total / rows.length;
  }

  function calibrationSummary(records, direction, rewardRisk, assetBucket) {
    const resolved = normalizeRecords(records).filter((row) => row.outcome === 0 || row.outcome === 1);
    const normalizedDirection = String(direction || "").toUpperCase();
    const bucket = rrBucket(rewardRisk);
    const normalizedAsset = String(assetBucket || "").toLowerCase();
    const exact = resolved.filter((row) => (
      String(row.direction || "").toUpperCase() === normalizedDirection
      && String(row.rrBucket || rrBucket(row.rr)) === bucket
      && (!normalizedAsset || String(row.assetBucket || "").toLowerCase() === normalizedAsset)
    ));
    const directional = resolved.filter((row) => String(row.direction || "").toUpperCase() === normalizedDirection);
    let sample = resolved;
    let scope = "all Ask calls";
    if (exact.length >= 4) {
      sample = exact;
      scope = "matching setup family";
    } else if (directional.length >= 5) {
      sample = directional;
      scope = normalizedDirection.toLowerCase() + " calls";
    }
    const wins = sample.reduce((sum, row) => sum + Number(row.outcome === 1), 0);
    return {
      samples: sample.length,
      wins,
      hitRate: sample.length ? wins / sample.length : null,
      brier: brierScore(sample),
      scope,
      totalResolved: resolved.length,
    };
  }

  function calibrate(input) {
    const rewardRisk = clamp(finite(input && input.rr, 0), 0.25, 8);
    const breakEven = 1 / (1 + rewardRisk);
    const strength = clamp((Math.abs(finite(input && input.score, 0)) - 1.75) / 3.75, 0, 1);
    let liveProbability = breakEven + (strength * 0.22);
    const extensionAtr = finite(input && input.extensionAtr, 0);
    if (extensionAtr > 1.25) liveProbability -= 0.025;
    if (extensionAtr > 1.75) liveProbability -= 0.045;

    const scheduledFresh = Boolean(input && input.scheduledFresh);
    const sameDirection = String(input && input.scheduledDirection || "").toUpperCase()
      === String(input && input.direction || "").toUpperCase();
    const scheduledRaw = finite(input && input.scheduledProbability, 0);
    const scheduledProbability = scheduledRaw > 1 ? scheduledRaw / 100 : scheduledRaw;
    if (scheduledFresh && sameDirection && scheduledProbability > 0 && scheduledProbability < 1) {
      const scheduledTargetEquivalent = breakEven + ((scheduledProbability - 0.5) * 0.45);
      liveProbability = (liveProbability * 0.72) + (scheduledTargetEquivalent * 0.28);
    }

    const history = input && input.history && typeof input.history === "object"
      ? input.history
      : { samples: 0, wins: 0, hitRate: null, brier: null, scope: "all Ask calls", totalResolved: 0 };
    let quality = 0;
    quality += finite(input && input.barsCount, 0) >= 120 ? 30 : finite(input && input.barsCount, 0) >= 55 ? 20 : 5;
    quality += input && input.candleFresh ? 25 : 0;
    quality += input && input.marketFresh ? 15 : 0;
    quality += scheduledFresh && sameDirection ? 15 : 0;
    quality += input && input.earningsMapped ? 5 : 0;
    quality += Math.min(10, finite(history.samples, 0) / 2);
    quality = Math.round(clamp(quality, 0, 100));

    if (input && input.eventGate) liveProbability -= 0.06;
    if (!(input && input.candleFresh)) liveProbability = breakEven;
    const reliability = 0.45 + ((quality / 100) * 0.55);
    let probability = breakEven + ((liveProbability - breakEven) * reliability);

    const samples = Math.max(0, Math.round(finite(history.samples, 0)));
    const wins = clamp(finite(history.wins, 0), 0, samples);
    if (samples > 0) {
      const priorWeight = 10;
      probability = ((probability * priorWeight) + wins) / (priorWeight + samples);
    }

    const certaintyCap = samples >= 30 ? 0.86 : samples >= 10 ? 0.78 : 0.68;
    const qualityCap = quality >= 80 ? certaintyCap : quality >= 60 ? Math.min(certaintyCap, 0.72) : 0.62;
    probability = clamp(probability, 0.08, qualityCap);
    const edge = probability - breakEven;
    const decisionSupported = Boolean(input && input.candleFresh)
      && !(input && input.eventGate)
      && edge >= 0.06;
    const confidenceLabel = edge >= 0.15 && quality >= 75
      ? "HIGH"
      : edge >= 0.07 && quality >= 55 ? "MEDIUM" : "LOW";
    const calibrationStatus = samples >= 30 ? "TESTED" : samples >= 10 ? "BUILDING" : "EARLY";

    return {
      probability,
      probabilityPct: Math.round(probability * 100),
      breakEven,
      breakEvenPct: Math.round(breakEven * 100),
      edge,
      edgePctPoints: Math.round(edge * 100),
      evidenceQuality: quality,
      confidenceLabel,
      decisionSupported,
      calibrationStatus,
      samples,
      hitRate: history.hitRate == null ? null : finite(history.hitRate, null),
      hitRatePct: history.hitRate == null ? null : Math.round(finite(history.hitRate, 0) * 100),
      brier: history.brier == null ? null : finite(history.brier, null),
      scope: String(history.scope || "all Ask calls"),
      totalResolved: Math.max(0, Math.round(finite(history.totalResolved, samples))),
      horizonHours: HORIZON_HOURS,
    };
  }

  function recordForecast(records, forecast, nowMs) {
    const rows = normalizeRecords(records);
    const ticker = String(forecast && forecast.ticker || "").trim().toUpperCase();
    const direction = String(forecast && forecast.direction || "").trim().toUpperCase();
    const analysedAt = String(forecast && forecast.analysedAt || new Date(nowMs || Date.now()).toISOString());
    const analysedMs = Date.parse(analysedAt);
    if (!ticker || !["LONG", "SHORT"].includes(direction) || !Number.isFinite(analysedMs)) return rows;
    if (!(forecast && forecast.candleFresh) || (forecast && forecast.eventGate)) return rows;

    const duplicate = rows.some((row) => (
      String(row.ticker || "").toUpperCase() === ticker
      && String(row.direction || "").toUpperCase() === direction
      && Math.abs(Date.parse(String(row.analysedAt || "")) - analysedMs) < (30 * 60 * 1000)
    ));
    if (duplicate) return rows;

    rows.push({
      id: ticker + "-" + direction + "-" + String(analysedMs),
      ticker,
      direction,
      decision: String(forecast.decision || ""),
      query: String(forecast.query || "").slice(0, 240),
      analysedAt,
      horizonHours: clamp(finite(forecast.horizonHours, HORIZON_HOURS), 1, 168),
      current: finite(forecast.current, 0),
      target: finite(forecast.target, 0),
      invalidation: finite(forecast.invalidation, 0),
      rr: finite(forecast.rr, 0),
      rrBucket: rrBucket(forecast.rr),
      assetBucket: String(forecast.assetBucket || ""),
      probability: clamp(finite(forecast.probability, 0.5), 0.01, 0.99),
      evidenceQuality: clamp(finite(forecast.evidenceQuality, 0), 0, 100),
      outcome: null,
      outcomeReason: "",
      resolvedAt: null,
    });
    return normalizeRecords(rows);
  }

  return {
    STORAGE_KEY,
    HORIZON_HOURS,
    rrBucket,
    load,
    save,
    settleForecasts,
    calibrationSummary,
    calibrate,
    recordForecast,
  };
});
