"""
trending.py — Trending stocks discovery module
Aggregates trending/momentum tickers from multiple free sources:
Finviz screener, StockTwits, Yahoo Finance, Reddit/WSB, and unusual volume detection.
Merges results with enriched profile data for conviction scoring.

All sources have error isolation — if one fails, the rest continue.
"""

import re
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import time

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

# ─── Cache to avoid hammering sources (with size limit) ─────────────────────
_trending_cache = {}
_CACHE_TTL_5 = 300   # 5 minutes
_CACHE_TTL_10 = 600  # 10 minutes
_CACHE_MAX_SIZE = 300


def _cached(key, ttl=_CACHE_TTL_5):
    """Return cached value if fresh, else None."""
    entry = _trending_cache.get(key)
    if entry and (time.time() - entry["ts"]) < ttl:
        return entry["data"]
    return None


def _set_cache(key, data):
    # Evict stale entries if cache is too large
    if len(_trending_cache) > _CACHE_MAX_SIZE:
        now = time.time()
        stale = [k for k, v in _trending_cache.items() if (now - v["ts"]) > _CACHE_TTL_10]
        for k in stale:
            del _trending_cache[k]
        if len(_trending_cache) > _CACHE_MAX_SIZE:
            sorted_keys = sorted(_trending_cache, key=lambda k: _trending_cache[k]["ts"])
            for k in sorted_keys[:len(sorted_keys) // 2]:
                del _trending_cache[k]
    _trending_cache[key] = {"data": data, "ts": time.time()}


# Cross-module imports (deferred to avoid circular imports)
def _import_data_fetcher():
    try:
        from data_fetcher import get_stock_detail, WATCHLIST
        return get_stock_detail, WATCHLIST
    except ImportError:
        return None, []


def _import_market_scraper():
    try:
        from market_scraper import get_enriched_ticker_profile
        return get_enriched_ticker_profile
    except ImportError:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# 1. Finviz Screener — Unusual volume + big movers
# ═════════════════════════════════════════════════════════════════════════════

def get_finviz_trending():
    """
    Scrape Finviz screener for tickers with unusual volume and significant price moves.
    Returns list of ticker strings.
    """
    cached = _cached("finviz_trending", _CACHE_TTL_5)
    if cached is not None:
        return cached

    tickers = []
    urls = [
        # Big gainers with unusual volume
        "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o500,sh_relvol_o2,ta_change_u3&ft=4&o=-change",
        # Big losers with unusual volume
        "https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o500,sh_relvol_o2,ta_change_d3&ft=4&o=change",
    ]

    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")

            # Finviz screener table uses class "screener-link-primary" for ticker links
            ticker_links = soup.find_all("a", class_="screener-link-primary")
            for link in ticker_links:
                sym = link.get_text(strip=True).upper()
                if sym and re.match(r'^[A-Z]{1,5}$', sym) and sym not in tickers:
                    tickers.append(sym)

            # Fallback: look in the screener table rows
            if not ticker_links:
                table = soup.find("table", {"id": "screener-views-table"})
                if table:
                    rows = table.find_all("tr")
                    for row in rows:
                        cells = row.find_all("td")
                        if len(cells) >= 2:
                            sym_cell = cells[1]
                            a_tag = sym_cell.find("a")
                            if a_tag:
                                sym = a_tag.get_text(strip=True).upper()
                                if sym and re.match(r'^[A-Z]{1,5}$', sym) and sym not in tickers:
                                    tickers.append(sym)
        except Exception as e:
            logger.debug(f"Finviz trending scrape error: {e}")

    _set_cache("finviz_trending", tickers)
    return tickers


# ═════════════════════════════════════════════════════════════════════════════
# 2. StockTwits — Trending symbols
# ═════════════════════════════════════════════════════════════════════════════

def get_stocktwits_trending():
    """
    Fetch trending symbols from StockTwits API.
    Returns list of ticker strings.
    """
    cached = _cached("stocktwits_trending", _CACHE_TTL_5)
    if cached is not None:
        return cached

    tickers = []
    try:
        url = "https://api.stocktwits.com/api/2/trending/symbols.json"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            symbols = data.get("symbols", [])
            for sym_obj in symbols:
                sym = sym_obj.get("symbol", "").upper()
                if sym and re.match(r'^[A-Z]{1,5}$', sym):
                    tickers.append(sym)
    except Exception as e:
        logger.debug(f"StockTwits trending error: {e}")

    _set_cache("stocktwits_trending", tickers)
    return tickers


# ═════════════════════════════════════════════════════════════════════════════
# 3. Yahoo Finance — Trending tickers
# ═════════════════════════════════════════════════════════════════════════════

def get_yahoo_trending():
    """
    Fetch trending tickers from Yahoo Finance API.
    Tries multiple endpoints. Returns list of ticker strings.
    """
    cached = _cached("yahoo_trending", _CACHE_TTL_5)
    if cached is not None:
        return cached

    tickers = []

    # Method 1: Trending endpoint
    urls = [
        "https://query1.finance.yahoo.com/v1/finance/trending/US",
        "https://query2.finance.yahoo.com/v1/finance/trending/US",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers={
                **HEADERS,
                "Accept": "application/json",
            }, timeout=10)
            if r.status_code == 200:
                data = r.json()
                quotes = (
                    data.get("finance", {})
                    .get("result", [{}])[0]
                    .get("quotes", [])
                )
                for q in quotes:
                    sym = q.get("symbol", "").upper()
                    if sym and re.match(r'^[A-Z]{1,5}$', sym) and sym not in tickers:
                        tickers.append(sym)
                if tickers:
                    break
        except Exception as e:
            logger.debug(f"Yahoo trending {url}: {e}")

    # Method 2: Most active from Yahoo screener
    if not tickers:
        try:
            url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=true&lang=en-US&region=US&scrIds=most_actives&count=25"
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                data = r.json()
                quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
                for q in quotes:
                    sym = q.get("symbol", "").upper()
                    if sym and re.match(r'^[A-Z]{1,5}$', sym) and sym not in tickers:
                        tickers.append(sym)
        except Exception as e:
            logger.debug(f"Yahoo most active fallback: {e}")

    _set_cache("yahoo_trending", tickers)
    return tickers


# ═════════════════════════════════════════════════════════════════════════════
# 4. Reddit / WSB — Most-mentioned tickers
# ═════════════════════════════════════════════════════════════════════════════

def get_reddit_wsb_trending():
    """
    Fetch trending WSB tickers from free APIs.
    Tries Tradestie first, then ApeWisdom as fallback.
    Returns list of ticker strings.
    """
    cached = _cached("reddit_wsb_trending", _CACHE_TTL_10)
    if cached is not None:
        return cached

    tickers = []

    # Approach A: Tradestie free Reddit API
    try:
        url = "https://tradestie.com/api/v1/apps/reddit"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for item in data:
                sym = ""
                if isinstance(item, dict):
                    sym = item.get("ticker", item.get("symbol", "")).upper()
                elif isinstance(item, str):
                    sym = item.upper()
                if sym and re.match(r'^[A-Z]{1,5}$', sym) and sym not in tickers:
                    tickers.append(sym)
    except Exception as e:
        logger.debug(f"Tradestie Reddit API error: {e}")

    # Approach B: ApeWisdom fallback (if Tradestie returned nothing)
    if not tickers:
        try:
            url = "https://apewisdom.io/api/v1.0/filter/wallstreetbets/"
            r = requests.get(url, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                data = r.json()
                results = data.get("results", [])
                for item in results:
                    sym = item.get("ticker", item.get("name", "")).upper()
                    if sym and re.match(r'^[A-Z]{1,5}$', sym) and sym not in tickers:
                        tickers.append(sym)
        except Exception as e:
            logger.debug(f"ApeWisdom Reddit API error: {e}")

    _set_cache("reddit_wsb_trending", tickers)
    return tickers


# ═════════════════════════════════════════════════════════════════════════════
# 5. Unusual Volume Detection — yfinance-based
# ═════════════════════════════════════════════════════════════════════════════

def get_unusual_whales_trending(extra_tickers=None):
    """
    Detect unusual volume across WATCHLIST + any extra tickers.
    Flags tickers where today's volume > 3x average volume (5-day window).
    Returns list of ticker strings.
    """
    cached = _cached("unusual_vol_trending", _CACHE_TTL_5)
    if cached is not None:
        return cached

    _, watchlist = _import_data_fetcher()
    scan_list = list(watchlist) if watchlist else []

    # Add any extra tickers discovered from other sources
    if extra_tickers:
        for t in extra_tickers:
            if t not in scan_list:
                scan_list.append(t)

    if not scan_list:
        _set_cache("unusual_vol_trending", [])
        return []

    tickers = []

    try:
        # Batch download 5-day data for all tickers at once
        data = yf.download(scan_list, period="5d", group_by="ticker", progress=False, threads=True)

        for sym in scan_list:
            try:
                if len(scan_list) == 1:
                    vol_series = data["Volume"]
                else:
                    vol_series = data[sym]["Volume"]

                vol_series = vol_series.dropna()
                if len(vol_series) < 2:
                    continue

                today_vol = vol_series.iloc[-1]
                avg_vol = vol_series.iloc[:-1].mean()

                if avg_vol > 0 and today_vol > (3 * avg_vol):
                    tickers.append(sym)
            except Exception:
                continue
    except Exception as e:
        logger.debug(f"Unusual volume scan error: {e}")

    _set_cache("unusual_vol_trending", tickers)
    return tickers


# ═════════════════════════════════════════════════════════════════════════════
# 6. Master: Aggregated Trending Watchlist with Conviction Scoring
# ═════════════════════════════════════════════════════════════════════════════

def _check_earnings_within_7d(ticker):
    """Check if ticker has earnings within the next 7 days."""
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None:
            return False
        date = None
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date", [None])
            ed = ed[0] if ed else None
            date = pd.to_datetime(ed) if ed else None
        elif hasattr(cal, "index") and "Earnings Date" in cal.index:
            ed = cal.loc["Earnings Date"]
            date = pd.to_datetime(ed.iloc[0] if hasattr(ed, "iloc") else ed)
        if date is None:
            return False
        now = datetime.now()
        return now.date() <= date.date() <= (now + timedelta(days=7)).date()
    except Exception:
        return False


def get_trending_watchlist():
    """
    Master function: aggregate all trending sources, enrich with market data,
    and score by conviction.

    Returns list of dicts sorted by conviction score (desc), max 10 items.
    """
    cached = _cached("trending_watchlist", _CACHE_TTL_5)
    if cached is not None:
        return cached

    get_stock_detail, _ = _import_data_fetcher()
    get_enriched_ticker_profile = _import_market_scraper()

    # ── Step 1: Gather tickers from all sources ──────────────────────────
    source_results = {}

    source_funcs = [
        ("Finviz", get_finviz_trending),
        ("StockTwits", get_stocktwits_trending),
        ("Yahoo", get_yahoo_trending),
        ("Reddit/WSB", get_reddit_wsb_trending),
    ]

    for source_name, func in source_funcs:
        try:
            result = func()
            if result:
                source_results[source_name] = result
        except Exception as e:
            logger.debug(f"Trending source {source_name} failed: {e}")

    # Collect all unique tickers discovered so far (for unusual volume scan)
    all_discovered = set()
    for syms in source_results.values():
        all_discovered.update(syms)

    # Unusual volume — pass discovered tickers so they get scanned too
    try:
        uv_tickers = get_unusual_whales_trending(extra_tickers=list(all_discovered))
        if uv_tickers:
            source_results["UnusualVolume"] = uv_tickers
    except Exception as e:
        logger.debug(f"Unusual volume scan failed: {e}")

    # ── Step 2: Build ticker -> sources mapping ──────────────────────────
    ticker_sources = {}
    for source_name, syms in source_results.items():
        for sym in syms:
            if sym not in ticker_sources:
                ticker_sources[sym] = []
            if source_name not in ticker_sources[sym]:
                ticker_sources[sym].append(source_name)

    if not ticker_sources:
        _set_cache("trending_watchlist", [])
        return []

    # ── Step 3: Enrich each ticker and compute conviction ────────────────
    trending_list = []

    for ticker, sources in ticker_sources.items():
        source_count = len(sources)

        # Defaults
        price = None
        change_pct = None
        volume = None
        avg_volume = None
        vol_ratio = None
        iv_rank = None
        has_unusual_options = False
        earnings_within_7d = False

        # Get price/volume data from get_stock_detail or yfinance
        try:
            if get_stock_detail:
                detail = get_stock_detail(ticker)
                price = detail.get("price")
                change_pct = detail.get("pct_change")
                avg_volume = detail.get("avg_volume")
                # Check options for unusual activity
                opts = detail.get("options")
                if opts and isinstance(opts, dict):
                    has_unusual_options = bool(opts.get("unusual_activity"))
            else:
                # Fallback to raw yfinance
                t = yf.Ticker(ticker)
                fi = t.fast_info
                price = round(float(fi.last_price), 2) if fi.last_price else None
                prev_close = fi.previous_close if hasattr(fi, "previous_close") else None
                if price and prev_close and prev_close > 0:
                    change_pct = round(((price - prev_close) / prev_close) * 100, 2)
        except Exception as e:
            logger.debug(f"Trending enrich price for {ticker}: {e}")

        # Get current volume from yfinance (fast_info doesn't always have it)
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if not hist.empty:
                volume = int(hist["Volume"].iloc[-1])
                avg_vol_calc = hist["Volume"].iloc[:-1].mean()
                if avg_volume is None and avg_vol_calc > 0:
                    avg_volume = int(avg_vol_calc)
        except Exception as e:
            logger.debug(f"Trending enrich volume for {ticker}: {e}")

        # Volume ratio
        if volume and avg_volume and avg_volume > 0:
            vol_ratio = round(volume / avg_volume, 2)

        # Get full enriched profile from ALL 13 data sources
        profile = {}
        try:
            if get_enriched_ticker_profile:
                profile = get_enriched_ticker_profile(ticker) or {}
                if profile:
                    iv_rank = profile.get("iv_rank")
                    # Also check for unusual options if not already flagged
                    if not has_unusual_options:
                        pc_ratio = profile.get("put_call_ratio")
                        if pc_ratio is not None and (pc_ratio > 1.5 or pc_ratio < 0.3):
                            has_unusual_options = True
        except Exception as e:
            logger.debug(f"Trending enrich profile for {ticker}: {e}")

        # Check earnings
        try:
            earnings_within_7d = _check_earnings_within_7d(ticker)
        except Exception as e:
            logger.debug(f"Trending earnings check for {ticker}: {e}")

        # ── Conviction scoring (uses ALL data sources) ─────────────────
        conviction = source_count * 20  # base: 20 per source

        # Unusual options activity
        if has_unusual_options:
            conviction += 10

        # Earnings proximity
        if earnings_within_7d:
            conviction += 5

        # Low IV = cheap options (from Barchart/Yahoo)
        if iv_rank is not None and iv_rank < 30:
            conviction += 5

        # Intel-based boosts from all 13 sources
        # Analyst consensus (Yahoo + Finviz + Zacks)
        analyst_recom = profile.get("analyst_recom")
        if analyst_recom and analyst_recom <= 1.5:
            conviction += 8  # Strong Buy consensus
        elif analyst_recom and analyst_recom <= 2.0:
            conviction += 4

        # Insider buying (OpenInsider / SEC filings)
        insider_sent = profile.get("insider_sentiment")
        if insider_sent == "bullish":
            conviction += 8

        # Social sentiment (StockTwits)
        social_score = profile.get("social_sentiment_score")
        if social_score is not None and social_score >= 70:
            conviction += 5

        # Zacks Rank
        zacks_rank = profile.get("zacks_rank")
        if zacks_rank == 1:
            conviction += 6
        elif zacks_rank == 2:
            conviction += 3

        # Short squeeze potential (Finviz + Yahoo)
        squeeze = profile.get("short_squeeze_score")
        if squeeze is not None and squeeze >= 70:
            conviction += 8
        elif squeeze is not None and squeeze >= 50:
            conviction += 4

        # Upside to analyst target (Yahoo + Stockanalysis)
        upside = profile.get("upside_pct")
        if upside is not None and upside >= 40:
            conviction += 6
        elif upside is not None and upside >= 25:
            conviction += 3

        # Earnings beat rate (EarningsWhispers)
        beat_rate = profile.get("beat_rate")
        if beat_rate is not None and beat_rate >= 80:
            conviction += 4

        # Revenue growth (Yahoo fundamentals)
        rev_growth = profile.get("revenue_growth")
        if rev_growth is not None:
            rg = rev_growth * 100 if abs(rev_growth) < 5 else rev_growth
            if rg >= 30:
                conviction += 6
            elif rg >= 15:
                conviction += 3

        conviction = min(conviction, 100)

        if conviction >= 70:
            conviction_label = "EXTREME"
        elif conviction >= 55:
            conviction_label = "HIGH"
        elif conviction >= 40:
            conviction_label = "MEDIUM"
        else:
            conviction_label = "LOW"

        trending_list.append({
            "ticker": ticker,
            "sources": sources,
            "source_count": source_count,
            "price": price,
            "change_pct": change_pct,
            "volume": volume,
            "avg_volume": avg_volume,
            "vol_ratio": vol_ratio,
            "iv_rank": iv_rank,
            "has_unusual_options": has_unusual_options,
            "earnings_within_7d": earnings_within_7d,
            "conviction_score": conviction,
            "conviction_label": conviction_label,
            # Extra intel for display
            "analyst_rating": round(analyst_recom, 2) if analyst_recom else None,
            "insider_sentiment": insider_sent,
            "social_sentiment": round(social_score, 0) if social_score else None,
            "zacks_rank": zacks_rank,
            "short_squeeze": round(squeeze, 0) if squeeze else None,
            "target_upside": round(upside, 0) if upside else None,
            "beat_rate": round(beat_rate, 0) if beat_rate else None,
        })

    # ── Step 4: Sort by conviction score and limit to top 10 ─────────────
    trending_list.sort(key=lambda x: x["conviction_score"], reverse=True)
    result = trending_list[:10]

    _set_cache("trending_watchlist", result)
    return result
