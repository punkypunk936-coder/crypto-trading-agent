import {
  SNAPSHOT_PATH,
  TRADE_REVIEWS_PATH,
  defaultTradeReviews,
  json,
  mergeReviewsIntoTrades,
  normalizeTradeReviews,
  readJson,
  reviewSummary,
  unauthorized,
  writeJson,
} from "../lib/dashboard-store";

function upsertReview(current: any, payload: any) {
  const reviews = normalizeTradeReviews(current);
  const tradeId = String(payload?.trade_id || "").trim();
  if (!tradeId) {
    return reviews;
  }
  const next = { ...(reviews.reviews || {}) };
  next[tradeId] = {
    ...(next[tradeId] || {}),
    trade_id: tradeId,
    coin: String(payload?.coin || next[tradeId]?.coin || "").toUpperCase(),
    direction: String(payload?.direction || next[tradeId]?.direction || "").toUpperCase(),
    verdict: String(payload?.verdict || next[tradeId]?.verdict || "GOOD_TRADE").toUpperCase(),
    thesis_quality: String(payload?.thesis_quality || next[tradeId]?.thesis_quality || "").toUpperCase(),
    execution_quality: String(payload?.execution_quality || next[tradeId]?.execution_quality || "").toUpperCase(),
    tags: Array.isArray(payload?.tags) ? payload.tags : String(payload?.tags || next[tradeId]?.tags || "").split(",").map((tag) => String(tag).trim()).filter(Boolean),
    notes: String(payload?.notes || next[tradeId]?.notes || ""),
    reviewed_at: new Date().toISOString().slice(0, 19).replace("T", " "),
    reviewer: String(payload?.reviewer || next[tradeId]?.reviewer || "operator"),
  };
  return normalizeTradeReviews({
    ...reviews,
    updated_at: new Date().toISOString().slice(0, 19).replace("T", " "),
    reviews: next,
  });
}

export async function GET() {
  const payload = await readJson(TRADE_REVIEWS_PATH, defaultTradeReviews());
  return json(normalizeTradeReviews(payload));
}

export async function POST(request: Request) {
  if (unauthorized(request)) {
    return new Response("Forbidden", { status: 403 });
  }
  const data = await request.json().catch(() => null);
  if (!data) {
    return new Response("Missing payload", { status: 400 });
  }
  const current = await readJson(TRADE_REVIEWS_PATH, defaultTradeReviews());
  const payload = upsertReview(current, data);
  await writeJson(TRADE_REVIEWS_PATH, payload);

  const snapshot = await readJson(SNAPSHOT_PATH, null);
  let summary = reviewSummary([], payload);
  if (snapshot && typeof snapshot === "object") {
    snapshot.trade_reviews = payload;
    snapshot.trades = mergeReviewsIntoTrades(Array.isArray(snapshot.trades) ? snapshot.trades : [], payload);
    summary = reviewSummary(snapshot.trades, payload);
    snapshot.review_summary = summary;
    await writeJson(SNAPSHOT_PATH, snapshot);
  }
  return json({ ok: true, trade_reviews: payload, review_summary: summary });
}
