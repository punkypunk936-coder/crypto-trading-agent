"""
indicators/equity_event_feeds.py - event feeds for single-stock catalysts.

The news scorer is intentionally broad. This module is narrower: it pulls the
feeds that matter before earnings or other known events, summarizes them, and
hands back a small set of synthetic headlines plus structured fields.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping, Optional

import requests

from logger import get_logger

log = get_logger("equity_event_feeds")

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "crypto-trading-agent/1.0 event-monitor contact@example.com",
)
SEC_HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
YAHOO_OPTIONS_URL = "https://query2.finance.yahoo.com/v7/finance/options/{ticker}"
YAHOO_QUOTE_SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
NASDAQ_OPTIONS_URL = "https://api.nasdaq.com/api/quote/{ticker}/option-chain"
NASDAQ_EARNINGS_FORECAST_URL = "https://api.nasdaq.com/api/analyst/{ticker}/earnings-forecast"
NASDAQ_TARGET_PRICE_URL = "https://api.nasdaq.com/api/analyst/{ticker}/targetprice"
NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}

DEFAULT_CACHE_SECONDS = 1800

SEC_FORMS_OF_INTEREST = {"8-K", "10-Q", "10-K", "6-K", "20-F", "40-F", "S-3", "S-1"}

COMMON_CIKS: dict[str, str] = {
    "AAPL": "0000320193",
    "AMD": "0000002488",
    "AMZN": "0001018724",
    "BABA": "0001577552",
    "BX": "0001393818",
    "COIN": "0001679788",
    "COST": "0000909832",
    "CRCL": "0001876042",
    "CRWV": "0001769628",
    "DKNG": "0001772757",
    "GME": "0001326380",
    "GOOGL": "0001652044",
    "HIMS": "0001773751",
    "HOOD": "0001783879",
    "INTC": "0000050863",
    "LLY": "0000059478",
    "META": "0001326801",
    "MRVL": "0001835632",
    "MSFT": "0000789019",
    "MSTR": "0001050446",
    "MU": "0000723125",
    "NFLX": "0001065280",
    "NVDA": "0001045810",
    "ORCL": "0001341439",
    "PLTR": "0001321655",
    "RIVN": "0001874178",
    "RKLB": "0001819994",
    "TSLA": "0001318605",
    "TSM": "0001046179",
    "USAR": "0000027093",
}

OFFICIAL_IR_SOURCES: dict[str, list[dict[str, str]]] = {
    "AMZN": [
        {
            "company": "Amazon",
            "url": "https://ir.aboutamazon.com/events/event-details/2026/Q1-2026-Amazoncom-Inc-Earnings-Conference-Call-/default.aspx",
            "source": "Amazon Investor Relations",
        }
    ],
    "META": [
        {
            "company": "Meta Platforms",
            "url": "https://investor.atmeta.com/investor-news/press-release-details/2026/Meta-to-Announce-First-Quarter-2026-Results/default.aspx",
            "source": "Meta Investor Relations",
        },
        {
            "company": "Meta Platforms",
            "url": "https://investor.atmeta.com/investor-events/event-details/2026/Q1-2026-Earnings-Call/default.aspx",
            "source": "Meta Investor Relations",
        },
    ],
    "GOOGL": [
        {
            "company": "Alphabet",
            "url": "https://abc.xyz/investor/events/",
            "source": "Alphabet Investor Relations",
        },
        {
            "company": "Alphabet",
            "url": "https://abc.xyz/investor/earnings/",
            "source": "Alphabet Investor Relations",
        },
    ],
    "AAPL": [{"company": "Apple", "url": "https://www.apple.com/investor/earnings-call/", "source": "Apple Investor Relations"}],
    "MSFT": [{"company": "Microsoft", "url": "https://www.microsoft.com/en-us/investor/events", "source": "Microsoft Investor Relations"}],
    "NVDA": [{"company": "NVIDIA", "url": "https://investor.nvidia.com/events-and-presentations/events-and-presentations/default.aspx", "source": "NVIDIA Investor Relations"}],
    "TSLA": [{"company": "Tesla", "url": "https://ir.tesla.com/#events", "source": "Tesla Investor Relations"}],
    "INTC": [{"company": "Intel", "url": "https://www.intc.com/news-events/ir-calendar", "source": "Intel Investor Relations"}],
    "AMD": [{"company": "AMD", "url": "https://ir.amd.com/news-events/ir-calendar", "source": "AMD Investor Relations"}],
    "MU": [{"company": "Micron", "url": "https://investors.micron.com/events-and-presentations", "source": "Micron Investor Relations"}],
    "HIMS": [
        {
            "company": "Hims & Hers",
            "url": "https://investors.hims.com/news/news-details/2026/Hims--Hers-to-Announce-First-Quarter-2026-Financial-Results-on-May-11-2026/default.aspx",
            "source": "Hims & Hers Investor Relations",
        },
        {
            "company": "Hims & Hers",
            "url": "https://investors.hims.com/events-and-presentations/events-calendar/event-details/2026/Hims--Hers-First-Quarter-2026-Earnings-Call/default.aspx",
            "source": "Hims & Hers Investor Relations",
        },
    ],
}


@dataclass
class EquityEventFeed:
    coin: str
    valid: bool = True
    headlines: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    official_event_score: float = 0.0
    official_event_summary: str = ""
    sec_event_score: float = 0.0
    sec_event_summary: str = ""
    options_implied_move_pct: float = 0.0
    options_summary: str = ""
    analyst_revision_score: float = 0.0
    analyst_revision_summary: str = ""
    source_urls: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "coin": self.coin,
            "valid": self.valid,
            "headlines": list(self.headlines),
            "tags": list(self.tags),
            "errors": list(self.errors),
            "official_event_score": self.official_event_score,
            "official_event_summary": self.official_event_summary,
            "sec_event_score": self.sec_event_score,
            "sec_event_summary": self.sec_event_summary,
            "options_implied_move_pct": self.options_implied_move_pct,
            "options_summary": self.options_summary,
            "analyst_revision_score": self.analyst_revision_score,
            "analyst_revision_summary": self.analyst_revision_summary,
            "source_urls": list(self.source_urls),
        }


_feed_cache: dict[str, dict[str, Any]] = {}
_ticker_cik_cache: dict[str, str] = {}
_ticker_cik_cache_ts = 0.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value if value is not None else default))
    except Exception:
        return default


def _raw_value(value: Any, default: float = 0.0) -> float:
    if isinstance(value, Mapping):
        return _safe_float(value.get("raw"), default)
    return _safe_float(value, default)


def _market_number(value: Any, default: float = 0.0) -> float:
    text = str(value if value is not None else "").strip()
    if not text or text in {"--", "N/A"}:
        return default
    text = text.replace("$", "").replace(",", "").replace("%", "")
    return _safe_float(text, default)


def _text_from_html(html: str) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", html or "", flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _dedupe(items: Iterable[str], *, limit: int = 12) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if not text:
            continue
        key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _format_event_day(event_day: date) -> str:
    return f"{event_day.strftime('%B')} {event_day.day}, {event_day.year}"


def _event_window(event_day: date, now: datetime) -> Optional[str]:
    today = now.date()
    days_until = (event_day - today).days
    if days_until < -2 or days_until > 45:
        return None
    if days_until == 0:
        return "today"
    if days_until == 1:
        return "tomorrow"
    if days_until > 1:
        return f"in {days_until} days"
    return "just reported"


def _calendar_headlines_from_known_events(
    coin: str,
    calendar_events: Iterable[Mapping[str, Any]] | None,
    *,
    now: datetime,
) -> tuple[list[str], list[str]]:
    headlines: list[str] = []
    urls: list[str] = []
    for event in list(calendar_events or []):
        try:
            event_day = date.fromisoformat(str(event.get("date") or ""))
        except ValueError:
            continue
        window = _event_window(event_day, now)
        if not window:
            continue
        source = str(event.get("source") or "").strip()
        company = str(event.get("company") or coin).strip()
        label = str(event.get("label") or "earnings").strip()
        timing = str(event.get("timing") or "").strip()
        focus = str(event.get("focus") or "").strip()
        source_text = f" via official {source}" if source and "investor" in source.lower() else ""
        timing_text = f" at {timing}" if timing else ""
        focus_text = f"; watch {focus}" if focus else ""
        headlines.append(
            f"{company} official IR event calendar confirms {label} {window} "
            f"on {_format_event_day(event_day)}{timing_text}{focus_text}{source_text}"
        )
        if event.get("url"):
            urls.append(str(event.get("url")))
    return headlines, urls


def _extract_calendar_dates(text: str, *, now: datetime) -> list[date]:
    month_re = (
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+([0-9]{1,2})(?:st|nd|rd|th)?(?:,\s*([0-9]{4}))?"
    )
    dates: list[date] = []
    for match in re.finditer(month_re, text or "", flags=re.IGNORECASE):
        start = max(0, match.start() - 180)
        end = min(len(text), match.end() + 180)
        context = text[start:end].lower()
        if not any(term in context for term in ("earnings", "results", "conference call", "financial results")):
            continue
        year = _safe_int(match.group(3), now.year)
        try:
            parsed = datetime.strptime(f"{match.group(1)} {match.group(2)} {year}", "%B %d %Y").date()
        except ValueError:
            continue
        if _event_window(parsed, now):
            dates.append(parsed)
    return sorted(set(dates))


def _fetch_ir_events(coin: str, *, calendar_events: Iterable[Mapping[str, Any]] | None, now: datetime) -> tuple[list[str], str, list[str], list[str]]:
    headlines, urls = _calendar_headlines_from_known_events(coin, calendar_events, now=now)
    errors: list[str] = []
    sources = OFFICIAL_IR_SOURCES.get(coin, [])
    for source in sources[:3]:
        url = str(source.get("url") or "").strip()
        if not url:
            continue
        urls.append(url)
        try:
            resp = requests.get(url, timeout=8, headers=REQUEST_HEADERS)
            resp.raise_for_status()
            text = _text_from_html(resp.text)
        except Exception as exc:
            errors.append(f"IR {coin}: {exc}")
            continue
        company = str(source.get("company") or coin).strip()
        source_name = str(source.get("source") or "Investor Relations").strip()
        for event_day in _extract_calendar_dates(text, now=now)[:2]:
            window = _event_window(event_day, now)
            if not window:
                continue
            headlines.append(
                f"{company} official IR calendar confirms earnings event {window} "
                f"on {_format_event_day(event_day)} via {source_name}"
            )
    headlines = _dedupe(headlines, limit=5)
    summary = headlines[0] if headlines else ""
    return headlines, summary, _dedupe(urls, limit=6), errors


def _refresh_ticker_cik_cache() -> None:
    global _ticker_cik_cache_ts
    if _ticker_cik_cache and time.time() - _ticker_cik_cache_ts < 86400:
        return
    resp = requests.get(SEC_TICKERS_URL, timeout=8, headers={"User-Agent": SEC_USER_AGENT})
    resp.raise_for_status()
    data = resp.json()
    mapping: dict[str, str] = {}
    for item in (data or {}).values():
        ticker = str(item.get("ticker") or "").upper()
        cik = str(item.get("cik_str") or "").strip()
        if ticker and cik:
            mapping[ticker] = cik.zfill(10)
    if mapping:
        _ticker_cik_cache.clear()
        _ticker_cik_cache.update(mapping)
        _ticker_cik_cache_ts = time.time()


def _cik_for_ticker(coin: str) -> Optional[str]:
    ticker = str(coin or "").upper()
    if ticker in COMMON_CIKS:
        return COMMON_CIKS[ticker]
    try:
        _refresh_ticker_cik_cache()
    except Exception as exc:
        log.debug(f"[{ticker}] SEC ticker CIK refresh failed: {exc}")
    return _ticker_cik_cache.get(ticker)


def _fetch_sec_filings(coin: str, *, now: datetime, lookback_days: int = 21) -> tuple[list[str], str, list[str], list[str]]:
    cik = _cik_for_ticker(coin)
    if not cik:
        return [], "", [], [f"SEC CIK unavailable for {coin}"]
    try:
        resp = requests.get(SEC_SUBMISSIONS_URL.format(cik=cik), timeout=8, headers=SEC_HEADERS)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return [], "", [], [f"SEC submissions {coin}: {exc}"]

    recent = (data or {}).get("filings", {}).get("recent", {})
    forms = list(recent.get("form") or [])
    filing_dates = list(recent.get("filingDate") or [])
    accession_numbers = list(recent.get("accessionNumber") or [])
    primary_documents = list(recent.get("primaryDocument") or [])
    accepted = list(recent.get("acceptanceDateTime") or [])
    headlines: list[str] = []
    urls: list[str] = []
    for idx, form in enumerate(forms[:80]):
        form_text = str(form or "").upper()
        if form_text not in SEC_FORMS_OF_INTEREST:
            continue
        filing_day_text = str(filing_dates[idx] if idx < len(filing_dates) else "")
        try:
            filing_day = date.fromisoformat(filing_day_text)
        except ValueError:
            continue
        days_old = (now.date() - filing_day).days
        if days_old < 0 or days_old > lookback_days:
            continue
        accession = str(accession_numbers[idx] if idx < len(accession_numbers) else "")
        doc = str(primary_documents[idx] if idx < len(primary_documents) else "")
        acc_path = accession.replace("-", "")
        if accession and doc:
            urls.append(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_path}/{doc}")
        accepted_text = str(accepted[idx] if idx < len(accepted) else filing_day_text)
        headlines.append(
            f"{coin} SEC filing monitor: Form {form_text} filed on {filing_day_text} "
            f"({days_old}d ago, accepted {accepted_text})"
        )
        if len(headlines) >= 3:
            break
    summary = headlines[0] if headlines else ""
    return headlines, summary, _dedupe(urls, limit=4), []


def _mid_price(contract: Mapping[str, Any]) -> float:
    bid = _safe_float(contract.get("bid"))
    ask = _safe_float(contract.get("ask"))
    last = _safe_float(contract.get("lastPrice"))
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return last


def _nasdaq_row_mid(row: Mapping[str, Any], prefix: str) -> float:
    bid = _market_number(row.get(f"{prefix}_Bid"))
    ask = _market_number(row.get(f"{prefix}_Ask"))
    last = _market_number(row.get(f"{prefix}_Last"))
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if ask > 0 and bid <= 0:
        return ask / 2.0
    return last


def _fetch_nasdaq_options_snapshot(coin: str) -> tuple[float, str, list[str]]:
    try:
        resp = requests.get(
            NASDAQ_OPTIONS_URL.format(ticker=coin),
            params={"assetclass": "stocks", "limit": 500},
            timeout=8,
            headers=NASDAQ_HEADERS,
        )
        resp.raise_for_status()
        data = (resp.json() or {}).get("data") or {}
    except Exception as exc:
        return 0.0, "", [f"nasdaq options {coin}: {exc}"]

    last_trade = str(data.get("lastTrade") or "")
    match = re.search(r"\$([0-9,.]+)", last_trade)
    underlying = _market_number(match.group(1)) if match else 0.0
    rows = list(((data.get("table") or {}).get("rows")) or [])
    if underlying <= 0 or not rows:
        return 0.0, "", []

    active_expiry = ""
    first_expiry = ""
    option_rows: list[Mapping[str, Any]] = []
    for row in rows:
        expiry_group = str(row.get("expirygroup") or "").strip()
        if expiry_group:
            active_expiry = expiry_group
            if first_expiry and active_expiry != first_expiry:
                break
            continue
        strike = _market_number(row.get("strike"))
        if strike <= 0:
            continue
        if not first_expiry:
            first_expiry = active_expiry or str(row.get("expiryDate") or "nearest expiry")
        if active_expiry and active_expiry != first_expiry:
            break
        option_rows.append(row)

    if not option_rows:
        return 0.0, "", []
    atm = min(option_rows, key=lambda item: abs(_market_number(item.get("strike")) - underlying))
    call_mid = _nasdaq_row_mid(atm, "c")
    put_mid = _nasdaq_row_mid(atm, "p")
    if call_mid <= 0 or put_mid <= 0:
        return 0.0, "", []
    implied_move = (call_mid + put_mid) / underlying * 100.0
    strike = _market_number(atm.get("strike"))
    summary = (
        f"{coin} options market prices about {implied_move:.1f}% implied move into {first_expiry or 'nearest expiry'} "
        f"(Nasdaq ATM {strike:.2f} straddle ${call_mid + put_mid:.2f} vs stock ${underlying:.2f})"
    )
    return round(implied_move, 2), summary, []


def _fetch_options_snapshot(coin: str, *, now: datetime) -> tuple[float, str, list[str]]:
    try:
        base = requests.get(YAHOO_OPTIONS_URL.format(ticker=coin), timeout=8, headers=REQUEST_HEADERS)
        base.raise_for_status()
        data = base.json()
        result = ((data or {}).get("optionChain", {}).get("result") or [{}])[0]
        expirations = list(result.get("expirationDates") or [])
        if expirations:
            now_ts = int(now.timestamp())
            future = sorted(ts for ts in expirations if _safe_int(ts) >= now_ts - 86400)
            expiry = future[0] if future else expirations[0]
            resp = requests.get(
                YAHOO_OPTIONS_URL.format(ticker=coin),
                params={"date": expiry},
                timeout=8,
                headers=REQUEST_HEADERS,
            )
            resp.raise_for_status()
            result = (((resp.json() or {}).get("optionChain", {}).get("result") or [{}])[0])
        quote = result.get("quote") or {}
        underlying = _safe_float(quote.get("regularMarketPrice")) or _safe_float(quote.get("postMarketPrice"))
        options = (result.get("options") or [{}])[0]
        calls = list(options.get("calls") or [])
        puts = list(options.get("puts") or [])
        if underlying <= 0 or not calls or not puts:
            return 0.0, "", []
        call = min(calls, key=lambda item: abs(_safe_float(item.get("strike")) - underlying))
        put = min(puts, key=lambda item: abs(_safe_float(item.get("strike")) - underlying))
        call_mid = _mid_price(call)
        put_mid = _mid_price(put)
        if call_mid <= 0 or put_mid <= 0:
            return 0.0, "", []
        implied_move = (call_mid + put_mid) / underlying * 100.0
        expiry_ts = _safe_int(call.get("expiration") or put.get("expiration"))
        expiry_text = datetime.fromtimestamp(expiry_ts, tz=timezone.utc).date().isoformat() if expiry_ts else "nearest expiry"
        summary = (
            f"{coin} options market prices about {implied_move:.1f}% implied move into {expiry_text} "
            f"(ATM straddle ${call_mid + put_mid:.2f} vs stock ${underlying:.2f})"
        )
        return round(implied_move, 2), summary, []
    except Exception as exc:
        yahoo_error = f"options {coin}: {exc}"
    implied_move, summary, errors = _fetch_nasdaq_options_snapshot(coin)
    if summary:
        return implied_move, summary, errors
    return 0.0, "", [yahoo_error, *errors]


def _fetch_yahoo_analyst_revisions(coin: str) -> tuple[float, str, list[str]]:
    try:
        modules = "earningsTrend,recommendationTrend,financialData"
        resp = requests.get(
            YAHOO_QUOTE_SUMMARY_URL.format(ticker=coin),
            params={"modules": modules},
            timeout=8,
            headers=REQUEST_HEADERS,
        )
        resp.raise_for_status()
        result = (((resp.json() or {}).get("quoteSummary", {}).get("result") or [{}])[0])
    except Exception as exc:
        return 0.0, "", [f"analyst revisions {coin}: {exc}"]

    earnings_trend = list((result.get("earningsTrend") or {}).get("trend") or [])
    current_period = next((row for row in earnings_trend if str(row.get("period") or "") in {"0q", "+1q"}), earnings_trend[0] if earnings_trend else {})
    eps_est = current_period.get("earningsEstimate") or {}
    rev_est = current_period.get("revenueEstimate") or {}
    eps_up = _safe_int(eps_est.get("upLast30days"))
    eps_down = _safe_int(eps_est.get("downLast30days"))
    rev_up = _safe_int(rev_est.get("upLast30days"))
    rev_down = _safe_int(rev_est.get("downLast30days"))

    rec = ((result.get("recommendationTrend") or {}).get("trend") or [{}])[0]
    buys = _safe_int(rec.get("strongBuy")) + _safe_int(rec.get("buy"))
    sells = _safe_int(rec.get("sell")) + _safe_int(rec.get("strongSell"))
    holds = _safe_int(rec.get("hold"))

    financial = result.get("financialData") or {}
    target = _raw_value(financial.get("targetMeanPrice"))
    current = _raw_value(financial.get("currentPrice"))
    target_upside_pct = ((target / current) - 1.0) * 100.0 if target > 0 and current > 0 else 0.0

    score = (
        (eps_up - eps_down) * 1.4
        + (rev_up - rev_down) * 1.0
        + min(2.5, max(-2.5, target_upside_pct / 8.0))
        + min(1.5, max(-1.5, (buys - sells) / 8.0))
    )
    score = round(max(-6.0, min(6.0, score)), 2)
    if not any([eps_up, eps_down, rev_up, rev_down, buys, sells, holds, target_upside_pct]):
        return 0.0, "", []
    bias = "positive" if score > 0.75 else "negative" if score < -0.75 else "mixed"
    summary = (
        f"{coin} analyst revision feed is {bias}: EPS revisions {eps_up} up/{eps_down} down, "
        f"revenue revisions {rev_up} up/{rev_down} down, street {buys} buy vs {sells} sell"
    )
    if target_upside_pct:
        summary += f", target upside {target_upside_pct:.1f}%"
    return score, summary, []


def _fetch_nasdaq_analyst_revisions(coin: str) -> tuple[float, str, list[str]]:
    errors: list[str] = []
    forecast_data: dict[str, Any] = {}
    target_data: dict[str, Any] = {}
    try:
        forecast_resp = requests.get(
            NASDAQ_EARNINGS_FORECAST_URL.format(ticker=coin),
            timeout=8,
            headers=NASDAQ_HEADERS,
        )
        forecast_resp.raise_for_status()
        forecast_data = (forecast_resp.json() or {}).get("data") or {}
    except Exception as exc:
        errors.append(f"nasdaq earnings forecast {coin}: {exc}")
    try:
        target_resp = requests.get(
            NASDAQ_TARGET_PRICE_URL.format(ticker=coin),
            timeout=8,
            headers=NASDAQ_HEADERS,
        )
        target_resp.raise_for_status()
        target_data = (target_resp.json() or {}).get("data") or {}
    except Exception as exc:
        errors.append(f"nasdaq target price {coin}: {exc}")

    rows = list(((forecast_data.get("quarterlyForecast") or {}).get("rows")) or [])
    quarter = rows[0] if rows else {}
    eps_up = _safe_int(quarter.get("up"))
    eps_down = _safe_int(quarter.get("down"))
    estimates = _safe_int(quarter.get("noOfEstimates"))
    fiscal_end = str(quarter.get("fiscalEnd") or "next quarter").strip()

    consensus = target_data.get("consensusOverview") or {}
    buys = _safe_int(consensus.get("buy"))
    sells = _safe_int(consensus.get("sell"))
    holds = _safe_int(consensus.get("hold"))
    target = _safe_float(consensus.get("priceTarget"))

    if not any([eps_up, eps_down, estimates, buys, sells, holds, target]):
        return 0.0, "", errors
    score = (
        (eps_up - eps_down) * 1.35
        + min(2.0, max(-2.0, (buys - sells) / 12.0))
        + (0.45 if target > 0 else 0.0)
    )
    score = round(max(-6.0, min(6.0, score)), 2)
    bias = "positive" if score > 0.75 else "negative" if score < -0.75 else "mixed"
    summary = (
        f"{coin} analyst revision feed is {bias}: Nasdaq EPS revisions {eps_up} up/{eps_down} down "
        f"for {fiscal_end}, {estimates} estimates, street {buys} buy/{holds} hold/{sells} sell"
    )
    if target > 0:
        summary += f", mean target ${target:.2f}"
    return score, summary, []


def _fetch_analyst_revisions(coin: str) -> tuple[float, str, list[str]]:
    score, summary, errors = _fetch_yahoo_analyst_revisions(coin)
    if summary:
        return score, summary, errors
    nasdaq_score, nasdaq_summary, nasdaq_errors = _fetch_nasdaq_analyst_revisions(coin)
    if nasdaq_summary:
        return nasdaq_score, nasdaq_summary, nasdaq_errors
    return 0.0, "", [*errors, *nasdaq_errors]


def _bool_cfg(trading_cfg: Any, name: str, default: bool) -> bool:
    return bool(getattr(trading_cfg, name, default))


def _int_cfg(trading_cfg: Any, name: str, default: int) -> int:
    return int(getattr(trading_cfg, name, default) or default)


def get_equity_event_feed(
    coin: str,
    *,
    trading_cfg: Any = None,
    calendar_events: Iterable[Mapping[str, Any]] | None = None,
    now: Optional[datetime] = None,
) -> EquityEventFeed:
    coin = str(coin or "").upper()
    if not coin:
        return EquityEventFeed(coin="", valid=False, errors=["missing coin"])
    if trading_cfg is not None and not _bool_cfg(trading_cfg, "official_event_feed_enabled", True):
        return EquityEventFeed(coin=coin, valid=True)

    cache_seconds = _int_cfg(trading_cfg, "official_event_feed_cache_seconds", DEFAULT_CACHE_SECONDS)
    cache_key = f"{coin}:{_utc_now().date().isoformat()}"
    cached = _feed_cache.get(cache_key)
    if cached and time.time() - float(cached.get("ts", 0.0) or 0.0) < cache_seconds:
        return cached["feed"]

    now_dt = now or _utc_now()
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)

    feed = EquityEventFeed(coin=coin)
    headlines: list[str] = []
    tags: list[str] = []
    urls: list[str] = []

    if _bool_cfg(trading_cfg, "official_ir_calendar_sync_enabled", True):
        ir_headlines, ir_summary, ir_urls, ir_errors = _fetch_ir_events(coin, calendar_events=calendar_events, now=now_dt)
        headlines.extend(ir_headlines)
        urls.extend(ir_urls)
        feed.errors.extend(ir_errors[:2])
        if ir_headlines:
            tags.append("official_ir_event")
            feed.official_event_score = 2.0
            feed.official_event_summary = ir_summary

    if _bool_cfg(trading_cfg, "sec_filing_feed_enabled", True):
        sec_headlines, sec_summary, sec_urls, sec_errors = _fetch_sec_filings(
            coin,
            now=now_dt,
            lookback_days=_int_cfg(trading_cfg, "sec_filing_feed_lookback_days", 21),
        )
        headlines.extend(sec_headlines)
        urls.extend(sec_urls)
        feed.errors.extend(sec_errors[:1])
        if sec_headlines:
            tags.append("sec_filing")
            feed.sec_event_score = 1.0 + min(1.5, len(sec_headlines) * 0.35)
            feed.sec_event_summary = sec_summary

    if _bool_cfg(trading_cfg, "options_implied_move_feed_enabled", True):
        implied_move, options_summary, options_errors = _fetch_options_snapshot(coin, now=now_dt)
        feed.errors.extend(options_errors[:1])
        if options_summary:
            headlines.append(options_summary)
            tags.append("options_implied_move")
            feed.options_implied_move_pct = implied_move
            feed.options_summary = options_summary

    if _bool_cfg(trading_cfg, "analyst_revision_feed_enabled", True):
        revision_score, revision_summary, revision_errors = _fetch_analyst_revisions(coin)
        feed.errors.extend(revision_errors[:1])
        if revision_summary:
            headlines.append(revision_summary)
            tags.append("analyst_revision")
            if revision_score >= 0.75:
                tags.append("analyst_conviction")
            feed.analyst_revision_score = revision_score
            feed.analyst_revision_summary = revision_summary

    feed.headlines = _dedupe(headlines, limit=10)
    feed.tags = _dedupe(tags, limit=8)
    feed.source_urls = _dedupe(urls, limit=8)
    feed.valid = bool(feed.headlines or not feed.errors)
    _feed_cache[cache_key] = {"ts": time.time(), "feed": feed}
    return feed
