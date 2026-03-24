"""
market_scraper.py — Multi-source market intelligence aggregator
Pulls free data from: CBOE, Barchart, Finviz, Yahoo Finance,
Stockanalysis.com, EarningsWhispers, and consolidates into a
single enriched profile per ticker.

All sources have error isolation — if one fails, the rest continue.
"""

import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
import json
import time

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

# ─── Cache to avoid hammering sources ────────────────────────────────────────
_scraper_cache = {}
_CACHE_TTL = 600  # 10 minutes


def _cached(key, ttl=_CACHE_TTL):
    """Return cached value if fresh, else None."""
    entry = _scraper_cache.get(key)
    if entry and (time.time() - entry["ts"]) < ttl:
        return entry["data"]
    return None


def _set_cache(key, data):
    _scraper_cache[key] = {"data": data, "ts": time.time()}


# ═════════════════════════════════════════════════════════════════════════════
# 1. CBOE — VIX + Total Put/Call Ratio (market-wide fear gauge)
# ═════════════════════════════════════════════════════════════════════════════

def get_cboe_pc_ratio():
    """
    Fetch CBOE total and equity put/call ratios.
    Source: CBOE publishes daily ratios.
    P/C > 1.0 = market fear (bearish), < 0.7 = complacency (bullish)
    """
    cached = _cached("cboe_pc")
    if cached:
        return cached

    result = {"total_pc": None, "equity_pc": None, "index_pc": None, "vix": None}

    try:
        # VIX from yfinance (most reliable)
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="2d")
        if not hist.empty:
            result["vix"] = round(float(hist["Close"].iloc[-1]), 2)
    except Exception as e:
        logger.debug(f"CBOE VIX: {e}")

    try:
        # CBOE total P/C ratio from their data page
        url = "https://www.cboe.com/us/options/market_statistics/daily/"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            # Look for ratio values in the page
            text = soup.get_text()
            # Try to find total P/C ratio pattern
            import re as _re
            matches = _re.findall(r'Total Put/Call.*?(\d+\.\d+)', text)
            if matches:
                result["total_pc"] = float(matches[0])
            eq_matches = _re.findall(r'Equity.*?Put/Call.*?(\d+\.\d+)', text)
            if eq_matches:
                result["equity_pc"] = float(eq_matches[0])
    except Exception as e:
        logger.debug(f"CBOE P/C scrape: {e}")

    # Fallback: calculate P/C from SPY options via yfinance
    if result["total_pc"] is None:
        try:
            spy = yf.Ticker("SPY")
            exps = spy.options
            if exps:
                chain = spy.option_chain(exps[0])
                total_call_vol = chain.calls["volume"].fillna(0).sum()
                total_put_vol = chain.puts["volume"].fillna(0).sum()
                if total_call_vol > 0:
                    result["total_pc"] = round(float(total_put_vol / total_call_vol), 2)
        except Exception as e:
            logger.debug(f"CBOE fallback SPY P/C: {e}")

    _set_cache("cboe_pc", result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 2. Barchart — Options Volume Leaders + Unusual Activity
# ═════════════════════════════════════════════════════════════════════════════

def get_barchart_options_activity(ticker):
    """
    Fetch options overview from Barchart for a specific ticker.
    Gets: IV rank, options volume, put/call volume ratio.
    """
    cached = _cached(f"barchart_{ticker}")
    if cached:
        return cached

    result = {}
    try:
        url = f"https://www.barchart.com/stocks/quotes/{ticker}/options-overview"
        r = requests.get(url, headers={
            **HEADERS,
            "Accept": "text/html,application/xhtml+xml",
            "Referer": "https://www.barchart.com/",
        }, timeout=10)

        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(separator=" ")

            # IV Rank / IV Percentile
            iv_match = re.findall(r'IV\s*(?:Rank|Percentile)\s*[:\s]*(\d+\.?\d*)\s*%?', text, re.IGNORECASE)
            if iv_match:
                result["iv_rank"] = float(iv_match[0])

            # Historical Volatility
            hv_match = re.findall(r'Historical\s*Volatility\s*[:\s]*(\d+\.?\d*)\s*%?', text, re.IGNORECASE)
            if hv_match:
                result["hist_vol"] = float(hv_match[0])

    except Exception as e:
        logger.debug(f"Barchart {ticker}: {e}")

    _set_cache(f"barchart_{ticker}", result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 3. Enhanced Finviz — Full fundamentals + technicals
# ═════════════════════════════════════════════════════════════════════════════

def get_finviz_full(ticker):
    """
    Enhanced Finviz scraper — pulls ALL available data points.
    Returns: P/E, forward P/E, PEG, market cap, revenue, profit margin,
    ROE, debt/equity, institutional ownership, plus existing fields.
    """
    cached = _cached(f"finviz_{ticker}")
    if cached:
        return cached

    result = {}
    try:
        if not re.match(r"^[A-Z0-9]{1,5}(-[A-Z])?$", ticker.upper()):
            return result

        url = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code != 200:
            return result

        soup = BeautifulSoup(r.text, "html.parser")

        # Parse all fundamentals table data
        data = {}
        rows = soup.select("tr.table-dark-row, tr.table-dark-row-2")
        for row in rows:
            cells = row.find_all("td")
            for i in range(0, len(cells) - 1, 2):
                key = cells[i].get_text(strip=True)
                val = cells[i + 1].get_text(strip=True)
                data[key] = val

        # ── Extract all useful fields ────────────────────────────────
        def _float(key, default=None):
            v = data.get(key, "-")
            if v and v != "-":
                try:
                    return float(v.replace("%", "").replace(",", "").split("/")[0].strip())
                except Exception:
                    pass
            return default

        # Analyst & Price
        result["analyst_recom"] = _float("Recom")
        result["target_price"] = _float("Target Price")

        # Valuation
        result["pe"] = _float("P/E")
        result["forward_pe"] = _float("Forward P/E")
        result["peg"] = _float("PEG")
        result["pb"] = _float("P/B")
        result["ps"] = _float("P/S")

        # Profitability
        result["profit_margin"] = _float("Profit Margin")
        result["roe"] = _float("ROE")
        result["roa"] = _float("ROA")

        # Growth
        result["eps_qq"] = data.get("EPS Q/Q", None)
        result["sales_qq"] = data.get("Sales Q/Q", None)
        result["eps_growth_5y"] = _float("EPS next 5Y")
        result["revenue_growth"] = _float("Sales Q/Q")

        # Risk
        result["short_float"] = _float("Short Float / Ratio", _float("Short Float"))
        result["debt_equity"] = _float("Debt/Eq")
        result["beta"] = _float("Beta")

        # Ownership
        result["inst_ownership"] = _float("Inst Own")
        result["insider_trans"] = data.get("Insider Trans", None)
        result["inst_trans"] = data.get("Inst Trans", None)

        # Volatility
        result["volatility"] = data.get("Volatility", None)
        result["atr"] = _float("ATR")

        # Earnings
        result["earnings_date"] = data.get("Earnings", None)

        # Performance
        result["perf_week"] = _float("Perf Week")
        result["perf_month"] = _float("Perf Month")
        result["perf_quarter"] = _float("Perf Quarter")
        result["perf_ytd"] = _float("Perf YTD")

        # Clean None values
        result = {k: v for k, v in result.items() if v is not None}

    except Exception as e:
        logger.debug(f"Finviz full {ticker}: {e}")

    _set_cache(f"finviz_{ticker}", result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 4. Stockanalysis.com — Revenue, EPS estimates, analyst ratings
# ═════════════════════════════════════════════════════════════════════════════

def get_stockanalysis_data(ticker):
    """
    Scrape stockanalysis.com for consensus estimates and fundamentals.
    Gets: revenue growth, EPS estimates, analyst consensus, price target.
    """
    cached = _cached(f"stockanalysis_{ticker}")
    if cached:
        return cached

    result = {}
    try:
        url = f"https://stockanalysis.com/stocks/{ticker.lower()}/forecast/"
        r = requests.get(url, headers={
            **HEADERS,
            "Accept": "text/html,application/xhtml+xml",
        }, timeout=10)

        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(separator=" ")

            # Analyst consensus
            consensus_match = re.findall(
                r'(?:Analyst\s*)?Consensus[:\s]*(Strong Buy|Buy|Hold|Sell|Strong Sell)',
                text, re.IGNORECASE
            )
            if consensus_match:
                result["sa_consensus"] = consensus_match[0].title()

            # Price target
            target_match = re.findall(
                r'(?:Average\s*)?Price\s*Target[:\s]*\$?([\d,]+\.?\d*)',
                text, re.IGNORECASE
            )
            if target_match:
                result["sa_target"] = float(target_match[0].replace(",", ""))

            # Number of analysts
            analyst_match = re.findall(r'(\d+)\s*(?:Wall Street\s*)?[Aa]nalysts?', text)
            if analyst_match:
                result["sa_num_analysts"] = int(analyst_match[0])

            # Revenue estimate
            rev_match = re.findall(
                r'Revenue\s*(?:Estimate|Forecast)[:\s]*\$?([\d,.]+)\s*(B|M|T)',
                text, re.IGNORECASE
            )
            if rev_match:
                val = float(rev_match[0][0].replace(",", ""))
                mult = {"B": 1e9, "M": 1e6, "T": 1e12}.get(rev_match[0][1].upper(), 1)
                result["sa_revenue_est"] = val * mult

    except Exception as e:
        logger.debug(f"Stockanalysis {ticker}: {e}")

    _set_cache(f"stockanalysis_{ticker}", result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 5. Yahoo Finance — Deep fundamentals + IV + Earnings
# ═════════════════════════════════════════════════════════════════════════════

def get_yahoo_deep_data(ticker):
    """
    Deep dive into Yahoo Finance via yfinance for data not normally pulled:
    - IV percentile (computed from historical options data)
    - Earnings surprise history
    - Institutional holders
    - Short interest ratio
    - Cash flow health
    """
    cached = _cached(f"yahoo_deep_{ticker}")
    if cached:
        return cached

    result = {}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        # Earnings & Growth
        result["earnings_date"] = None
        try:
            cal = t.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date")
                    if ed:
                        result["earnings_date"] = str(ed[0]) if isinstance(ed, list) else str(ed)
                elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                    result["earnings_date"] = str(cal.loc["Earnings Date"].iloc[0])
        except Exception:
            pass

        result["revenue_growth"] = info.get("revenueGrowth")
        result["earnings_growth"] = info.get("earningsGrowth")
        result["earnings_quarterly_growth"] = info.get("earningsQuarterlyGrowth")

        # Valuation
        result["forward_pe"] = info.get("forwardPE")
        result["trailing_pe"] = info.get("trailingPE")
        result["peg_ratio"] = info.get("pegRatio")
        result["price_to_book"] = info.get("priceToBook")
        result["ev_to_ebitda"] = info.get("enterpriseToEbitda")

        # Financial Health
        result["free_cash_flow"] = info.get("freeCashflow")
        result["total_debt"] = info.get("totalDebt")
        result["total_cash"] = info.get("totalCash")
        result["current_ratio"] = info.get("currentRatio")
        result["profit_margin"] = info.get("profitMargins")
        result["operating_margin"] = info.get("operatingMargins")

        # Ownership & Short Interest
        result["held_by_institutions"] = info.get("heldPercentInstitutions")
        result["held_by_insiders"] = info.get("heldPercentInsiders")
        result["short_ratio"] = info.get("shortRatio")
        result["short_pct_float"] = info.get("shortPercentOfFloat")

        # Analyst
        result["recommendation"] = info.get("recommendationKey")
        result["num_analysts"] = info.get("numberOfAnalystOpinions")
        result["target_mean"] = info.get("targetMeanPrice")
        result["target_high"] = info.get("targetHighPrice")
        result["target_low"] = info.get("targetLowPrice")

        # Current price for reference
        try:
            result["current_price"] = float(t.fast_info.last_price or 0)
        except Exception:
            result["current_price"] = info.get("currentPrice", 0)

        # Compute upside to target
        price = result.get("current_price", 0)
        target = result.get("target_mean")
        if price and target and price > 0:
            result["upside_pct"] = round((target - price) / price * 100, 1)

        # ── IV Rank approximation ────────────────────────────────────
        # Compute from current IV vs 1-year IV range using options data
        try:
            exps = t.options
            if exps:
                chain = t.option_chain(exps[0])
                atm_calls = chain.calls.copy()
                current = result.get("current_price", 0)
                if current > 0 and not atm_calls.empty:
                    # Find ATM call (closest strike to current price)
                    atm_calls["dist"] = abs(atm_calls["strike"] - current)
                    atm_row = atm_calls.loc[atm_calls["dist"].idxmin()]
                    current_iv = float(atm_row["impliedVolatility"]) * 100

                    # Get 1-year high/low IV from historical volatility
                    hist = t.history(period="1y")
                    if len(hist) > 20:
                        returns = hist["Close"].pct_change().dropna()
                        hist_vol = float(returns.std() * (252 ** 0.5) * 100)
                        # IV rank = where current IV sits in estimated range
                        iv_low = hist_vol * 0.6   # estimated low
                        iv_high = hist_vol * 2.0  # estimated high
                        if iv_high > iv_low:
                            iv_rank = (current_iv - iv_low) / (iv_high - iv_low) * 100
                            result["iv_rank"] = round(max(0, min(100, iv_rank)), 1)
                            result["current_iv"] = round(current_iv, 1)
        except Exception as e:
            logger.debug(f"Yahoo IV rank {ticker}: {e}")

        # Clean None values
        result = {k: v for k, v in result.items() if v is not None}

    except Exception as e:
        logger.debug(f"Yahoo deep {ticker}: {e}")

    _set_cache(f"yahoo_deep_{ticker}", result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 6. EarningsWhispers — Earnings expectations & whisper numbers
# ═════════════════════════════════════════════════════════════════════════════

def get_earnings_whispers(ticker):
    """
    Scrape EarningsWhispers for the whisper number (what smart money
    actually expects, vs the published consensus).
    """
    cached = _cached(f"ew_{ticker}")
    if cached:
        return cached

    result = {}
    try:
        url = f"https://www.earningswhispers.com/stocks/{ticker.lower()}"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(separator=" ")

            # Whisper number
            whisper_match = re.findall(
                r'Whisper\s*(?:Number|EPS)?[:\s]*\$?([\-]?[\d]+\.?\d*)',
                text, re.IGNORECASE
            )
            if whisper_match:
                result["whisper_eps"] = float(whisper_match[0])

            # Consensus EPS
            cons_match = re.findall(
                r'Consensus\s*(?:EPS|Estimate)?[:\s]*\$?([\-]?[\d]+\.?\d*)',
                text, re.IGNORECASE
            )
            if cons_match:
                result["consensus_eps"] = float(cons_match[0])

            # Expected reaction / surprise history
            surprise_match = re.findall(
                r'(?:Beat|Miss|Met)\s*(?:Rate)?[:\s]*(\d+\.?\d*)%',
                text, re.IGNORECASE
            )
            if surprise_match:
                result["beat_rate"] = float(surprise_match[0])

    except Exception as e:
        logger.debug(f"EarningsWhispers {ticker}: {e}")

    _set_cache(f"ew_{ticker}", result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 7. MASTER AGGREGATOR — Consolidate all sources into one profile
# ═════════════════════════════════════════════════════════════════════════════

def get_enriched_ticker_profile(ticker):
    """
    Pull data from ALL available sources and merge into a single
    enriched profile. Used by the options scoring algorithm to make
    smarter picks.

    Returns dict with:
    - options_flow: put/call ratio, unusual activity
    - iv_rank: where current IV sits vs 1-year range
    - fundamentals: P/E, growth, margins, debt
    - analyst: consensus, target, upside
    - earnings: date, whisper number, beat rate
    - ownership: institutional %, insider activity, short interest
    - technicals: performance, volatility, beta
    """
    profile = {"ticker": ticker, "sources_hit": 0, "sources_failed": 0}

    # ── Source 1: Yahoo Finance Deep Data (most reliable) ────────────
    try:
        yahoo = get_yahoo_deep_data(ticker)
        if yahoo:
            profile.update(yahoo)
            profile["sources_hit"] += 1
    except Exception:
        profile["sources_failed"] += 1

    # ── Source 2: Finviz Full Fundamentals ────────────────────────────
    try:
        finviz = get_finviz_full(ticker)
        if finviz:
            # Only fill in what Yahoo didn't provide
            for k, v in finviz.items():
                if k not in profile or profile[k] is None:
                    profile[k] = v
            profile["sources_hit"] += 1
    except Exception:
        profile["sources_failed"] += 1

    # ── Source 3: Stockanalysis.com ───────────────────────────────────
    try:
        sa = get_stockanalysis_data(ticker)
        if sa:
            profile.update({f"sa_{k}" if not k.startswith("sa_") else k: v for k, v in sa.items()})
            profile["sources_hit"] += 1
    except Exception:
        profile["sources_failed"] += 1

    # ── Source 4: Barchart Options Activity ───────────────────────────
    try:
        barchart = get_barchart_options_activity(ticker)
        if barchart:
            for k, v in barchart.items():
                if k not in profile or profile[k] is None:
                    profile[k] = v
            profile["sources_hit"] += 1
    except Exception:
        profile["sources_failed"] += 1

    # ── Source 5: EarningsWhispers ────────────────────────────────────
    try:
        ew = get_earnings_whispers(ticker)
        if ew:
            profile.update(ew)
            profile["sources_hit"] += 1
    except Exception:
        profile["sources_failed"] += 1

    # ── Source 6: CBOE Market-Wide Data ───────────────────────────────
    try:
        cboe = get_cboe_pc_ratio()
        if cboe:
            profile["market_vix"] = cboe.get("vix")
            profile["market_pc_ratio"] = cboe.get("total_pc")
            profile["sources_hit"] += 1
    except Exception:
        profile["sources_failed"] += 1

    # ── Compute composite intelligence score ─────────────────────────
    profile["intel_score"] = _compute_intel_score(profile)

    return profile


def _compute_intel_score(p):
    """
    Compute a 0-100 intelligence score from all aggregated data.
    This score is used to BOOST or PENALIZE options in the selection.

    Factors:
    - Analyst consensus & upside potential
    - IV rank (high IV = expensive options, low IV = cheap)
    - Institutional activity
    - Short squeeze potential
    - Earnings proximity & beat history
    - Financial health
    """
    score = 50  # Start neutral

    # ── Analyst Sentiment (+/- 15) ───────────────────────────────────
    recom = p.get("analyst_recom")  # 1=Strong Buy, 5=Strong Sell
    if recom:
        if recom <= 1.5:
            score += 15
        elif recom <= 2.0:
            score += 10
        elif recom <= 2.5:
            score += 5
        elif recom >= 4.0:
            score -= 12
        elif recom >= 3.5:
            score -= 6

    # ── Upside to Target (+/- 10) ────────────────────────────────────
    upside = p.get("upside_pct")
    if upside:
        if upside >= 30:
            score += 10
        elif upside >= 15:
            score += 6
        elif upside >= 5:
            score += 2
        elif upside <= -15:
            score -= 8
        elif upside <= -5:
            score -= 3

    # ── IV Rank (+/- 8) ──────────────────────────────────────────────
    # Low IV rank = cheap options (good for buying)
    # High IV rank = expensive options (better for selling)
    iv_rank = p.get("iv_rank")
    if iv_rank is not None:
        if iv_rank <= 25:
            score += 8   # Cheap options — great for buying
        elif iv_rank <= 40:
            score += 4
        elif iv_rank >= 80:
            score -= 6   # Expensive — avoid buying
        elif iv_rank >= 60:
            score -= 2

    # ── Short Squeeze Potential (+/- 5) ──────────────────────────────
    short_float = p.get("short_float") or p.get("short_pct_float")
    if short_float:
        sf = short_float * 100 if short_float < 1 else short_float
        if sf >= 20:
            score += 5   # High short interest — squeeze potential
        elif sf >= 10:
            score += 2

    # ── Institutional Activity (+/- 5) ───────────────────────────────
    inst = p.get("held_by_institutions")
    if inst:
        pct = inst * 100 if inst < 1 else inst
        if pct >= 70:
            score += 3   # Strong institutional backing
        elif pct <= 20:
            score -= 3   # Low institutional interest

    inst_trans = p.get("inst_trans")
    if inst_trans and isinstance(inst_trans, str):
        try:
            val = float(inst_trans.replace("%", ""))
            if val > 2:
                score += 2  # Institutions buying
            elif val < -2:
                score -= 2  # Institutions selling
        except Exception:
            pass

    # ── Financial Health (+/- 7) ─────────────────────────────────────
    profit_margin = p.get("profit_margin") or p.get("profit_margins")
    if profit_margin:
        pm = profit_margin * 100 if abs(profit_margin) < 1 else profit_margin
        if pm >= 20:
            score += 4
        elif pm >= 10:
            score += 2
        elif pm < 0:
            score -= 3

    debt_eq = p.get("debt_equity")
    if debt_eq:
        if debt_eq > 2:
            score -= 3   # High debt
        elif debt_eq < 0.5:
            score += 3   # Low debt

    # ── Earnings Proximity Boost ─────────────────────────────────────
    # Options near earnings have elevated premiums & volatility
    earnings_date = p.get("earnings_date")
    if earnings_date:
        try:
            ed = pd.to_datetime(earnings_date)
            days_to_earnings = (ed - pd.Timestamp.now()).days
            if 0 <= days_to_earnings <= 7:
                score += 3   # Earnings this week — high volatility
            elif 7 < days_to_earnings <= 21:
                score += 1
        except Exception:
            pass

    # ── Beat Rate (EarningsWhispers) ─────────────────────────────────
    beat_rate = p.get("beat_rate")
    if beat_rate:
        if beat_rate >= 75:
            score += 3   # Company consistently beats estimates
        elif beat_rate <= 40:
            score -= 2

    return max(0, min(100, score))
