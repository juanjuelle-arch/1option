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

# ─── Cache to avoid hammering sources (with size limit) ─────────────────────
_scraper_cache = {}
_CACHE_TTL = 600  # 10 minutes
_CACHE_MAX_SIZE = 500  # M-4: prevent unbounded memory growth


def _cached(key, ttl=_CACHE_TTL):
    """Return cached value if fresh, else None."""
    entry = _scraper_cache.get(key)
    if entry and (time.time() - entry["ts"]) < ttl:
        return entry["data"]
    return None


def _set_cache(key, data):
    # Evict stale entries if cache is too large
    if len(_scraper_cache) > _CACHE_MAX_SIZE:
        now = time.time()
        stale = [k for k, v in _scraper_cache.items() if (now - v["ts"]) > _CACHE_TTL]
        for k in stale:
            del _scraper_cache[k]
        # If still too large, remove oldest half
        if len(_scraper_cache) > _CACHE_MAX_SIZE:
            sorted_keys = sorted(_scraper_cache, key=lambda k: _scraper_cache[k]["ts"])
            for k in sorted_keys[:len(sorted_keys) // 2]:
                del _scraper_cache[k]
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
# 7. CNN Fear & Greed Index (market-wide sentiment gauge)
# ═════════════════════════════════════════════════════════════════════════════

def get_fear_greed_index():
    """
    Fetch CNN Fear & Greed Index score (0-100).
    0 = Extreme Fear, 100 = Extreme Greed.
    Cached for 30 minutes (market-wide, not per ticker).
    """
    cached = _cached("fear_greed", ttl=1800)
    if cached:
        return cached

    result = {}
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        r = requests.get(url, headers={
            **HEADERS,
            "Accept": "application/json",
        }, timeout=10)
        if r.status_code == 200:
            data = r.json()
            fg = data.get("fear_and_greed", {})
            score = fg.get("score")
            rating = fg.get("rating")
            if score is not None:
                result["fg_score"] = round(float(score), 1)
            if rating:
                result["fg_rating"] = rating
            # Also grab previous close for trend
            prev = fg.get("previous_close")
            if prev is not None:
                result["fg_previous"] = round(float(prev), 1)
    except Exception as e:
        logger.debug(f"CNN Fear & Greed: {e}")

    _set_cache("fear_greed", result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 8. OpenInsider — Recent insider buys/sells
# ═════════════════════════════════════════════════════════════════════════════

def get_insider_trades(ticker):
    """
    Scrape OpenInsider for insider buys/sells in the last 30 days.
    Returns net sentiment based on insider transaction patterns.
    """
    cached = _cached(f"insider_{ticker}")
    if cached:
        return cached

    result = {}
    try:
        url = (
            f"https://openinsider.com/screener?s={ticker}&o=&pl=&ph=&ll=&lh="
            f"&fd=30&fdr=&td=&tdr=&feession=&lacession=&session="
        )
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            table = soup.find("table", class_="tinytable")
            if table:
                rows = table.find_all("tr")[1:]  # skip header
                buy_count = 0
                sell_count = 0
                total_bought = 0.0
                total_sold = 0.0
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) >= 10:
                        trade_type = cells[4].get_text(strip=True).lower() if len(cells) > 4 else ""
                        # Value column (typically column index 8 or 9)
                        value_text = cells[9].get_text(strip=True) if len(cells) > 9 else ""
                        value = 0.0
                        try:
                            value = float(value_text.replace("$", "").replace(",", "").replace("+", ""))
                        except Exception:
                            pass
                        if "purchase" in trade_type or "buy" in trade_type:
                            buy_count += 1
                            total_bought += abs(value)
                        elif "sale" in trade_type or "sell" in trade_type:
                            sell_count += 1
                            total_sold += abs(value)

                result["insider_buy_count"] = buy_count
                result["insider_sell_count"] = sell_count
                result["insider_net_buys"] = buy_count - sell_count
                result["insider_total_bought"] = round(total_bought, 2)
                result["insider_total_sold"] = round(total_sold, 2)

                if buy_count > sell_count and total_bought > total_sold:
                    result["insider_sentiment"] = "bullish"
                elif sell_count > buy_count and total_sold > total_bought:
                    result["insider_sentiment"] = "bearish"
                else:
                    result["insider_sentiment"] = "neutral"

    except Exception as e:
        logger.debug(f"OpenInsider {ticker}: {e}")

    _set_cache(f"insider_{ticker}", result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 9. Zacks — Zacks Rank & EPS estimates
# ═════════════════════════════════════════════════════════════════════════════

def get_zacks_data(ticker):
    """
    Scrape Zacks for rank, EPS estimates, and earnings surprise %.
    Zacks Rank: 1=Strong Buy, 2=Buy, 3=Hold, 4=Sell, 5=Strong Sell.
    """
    cached = _cached(f"zacks_{ticker}")
    if cached:
        return cached

    result = {}
    try:
        url = f"https://www.zacks.com/stock/quote/{ticker}"
        r = requests.get(url, headers={
            **HEADERS,
            "Accept": "text/html,application/xhtml+xml",
            "Referer": "https://www.zacks.com/",
        }, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(separator=" ")

            # Zacks Rank
            rank_match = re.findall(r'Zacks\s*Rank\s*[:#]?\s*(\d)\s*[-–]?\s*(\w+\s*\w*)', text, re.IGNORECASE)
            if rank_match:
                result["zacks_rank"] = int(rank_match[0][0])
                result["zacks_rank_label"] = rank_match[0][1].strip()

            # EPS estimates current quarter
            eps_curr_match = re.findall(
                r'(?:Current\s*Qtr|F1)\s*[:\s]*\$?([\-]?[\d]+\.?\d*)',
                text, re.IGNORECASE
            )
            if eps_curr_match:
                result["zacks_eps_est_current"] = float(eps_curr_match[0])

            # EPS estimates next quarter
            eps_next_match = re.findall(
                r'(?:Next\s*Qtr|F2)\s*[:\s]*\$?([\-]?[\d]+\.?\d*)',
                text, re.IGNORECASE
            )
            if eps_next_match:
                result["zacks_eps_est_next"] = float(eps_next_match[0])

            # Last earnings surprise %
            surprise_match = re.findall(
                r'(?:Earnings\s*)?Surprise\s*[:\s]*([\-+]?[\d]+\.?\d*)\s*%',
                text, re.IGNORECASE
            )
            if surprise_match:
                result["zacks_last_surprise_pct"] = float(surprise_match[0])

    except Exception as e:
        logger.debug(f"Zacks {ticker}: {e}")

    _set_cache(f"zacks_{ticker}", result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 10. FRED Economic / Macro Data
# ═════════════════════════════════════════════════════════════════════════════

def get_fred_macro():
    """
    Fetch macro indicators: 10Y Treasury yield, 5Y Treasury, Dollar Index.
    Uses yfinance tickers as proxy for FRED data.
    Cached for 1 hour (macro data, not per ticker).
    """
    cached = _cached("fred_macro", ttl=3600)
    if cached:
        return cached

    result = {}

    # 10Y Treasury Yield (^TNX)
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="5d")
        if not hist.empty:
            result["treasury_10y"] = round(float(hist["Close"].iloc[-1]), 3)
            if len(hist) >= 2:
                result["treasury_10y_prev"] = round(float(hist["Close"].iloc[-2]), 3)
    except Exception as e:
        logger.debug(f"FRED 10Y: {e}")

    # 5Y Treasury Yield (^FVX)
    try:
        fvx = yf.Ticker("^FVX")
        hist = fvx.history(period="5d")
        if not hist.empty:
            result["treasury_5y"] = round(float(hist["Close"].iloc[-1]), 3)
    except Exception as e:
        logger.debug(f"FRED 5Y: {e}")

    # 2Y Treasury Yield (^IRX is 13-week, use 2YY=F or approximate)
    try:
        twy = yf.Ticker("2YY=F")
        hist = twy.history(period="5d")
        if not hist.empty:
            result["treasury_2y"] = round(float(hist["Close"].iloc[-1]), 3)
    except Exception as e:
        logger.debug(f"FRED 2Y: {e}")

    # Dollar Index (DX-Y.NYB)
    try:
        dx = yf.Ticker("DX-Y.NYB")
        hist = dx.history(period="5d")
        if not hist.empty:
            result["dollar_index"] = round(float(hist["Close"].iloc[-1]), 2)
    except Exception as e:
        logger.debug(f"FRED Dollar Index: {e}")

    # Yield curve spread (10Y - 2Y) for inversion detection
    t10 = result.get("treasury_10y")
    t2 = result.get("treasury_2y")
    if t10 is not None and t2 is not None:
        result["yield_curve_spread"] = round(t10 - t2, 3)

    _set_cache("fred_macro", result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 11. Social Sentiment (StockTwits)
# ═════════════════════════════════════════════════════════════════════════════

def get_social_sentiment(ticker):
    """
    Fetch social sentiment from StockTwits API (free, no auth needed).
    Counts bullish vs bearish messages to compute sentiment score.
    """
    cached = _cached(f"social_{ticker}")
    if cached:
        return cached

    result = {}
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            messages = data.get("messages", [])
            bull_count = 0
            bear_count = 0
            for msg in messages:
                sentiment = msg.get("entities", {}).get("sentiment", {})
                if sentiment:
                    basic = sentiment.get("basic")
                    if basic == "Bullish":
                        bull_count += 1
                    elif basic == "Bearish":
                        bear_count += 1

            result["social_bull_count"] = bull_count
            result["social_bear_count"] = bear_count
            total = bull_count + bear_count
            if total > 0:
                result["social_sentiment_score"] = round((bull_count / total) * 100, 1)
            else:
                result["social_sentiment_score"] = 50.0  # neutral if no data
    except Exception as e:
        logger.debug(f"StockTwits {ticker}: {e}")

    _set_cache(f"social_{ticker}", result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 12. Short Squeeze Score (computed from Finviz + Yahoo data)
# ═════════════════════════════════════════════════════════════════════════════

def get_short_squeeze_data(ticker):
    """
    Compute short squeeze potential score (0-100).
    Uses short_float and days_to_cover from Finviz/Yahoo.
    High squeeze = short_float > 15% AND days_to_cover > 3.
    """
    cached = _cached(f"squeeze_{ticker}")
    if cached:
        return cached

    result = {}
    try:
        # Pull from finviz and yahoo for short data
        finviz = get_finviz_full(ticker)
        yahoo = get_yahoo_deep_data(ticker)

        short_float = finviz.get("short_float") or yahoo.get("short_pct_float")
        short_ratio = yahoo.get("short_ratio")  # days to cover

        if short_float is not None:
            sf = short_float * 100 if short_float < 1 else short_float
            result["short_float_pct"] = round(sf, 2)
        else:
            sf = 0

        if short_ratio is not None:
            result["days_to_cover"] = round(float(short_ratio), 2)
        else:
            short_ratio = 0

        # Compute squeeze score
        squeeze_score = 0
        # Short float contribution (max 50 points)
        if sf >= 30:
            squeeze_score += 50
        elif sf >= 20:
            squeeze_score += 40
        elif sf >= 15:
            squeeze_score += 30
        elif sf >= 10:
            squeeze_score += 15
        elif sf >= 5:
            squeeze_score += 5

        # Days to cover contribution (max 50 points)
        dtc = float(short_ratio) if short_ratio else 0
        if dtc >= 7:
            squeeze_score += 50
        elif dtc >= 5:
            squeeze_score += 40
        elif dtc >= 3:
            squeeze_score += 25
        elif dtc >= 2:
            squeeze_score += 10

        result["short_squeeze_score"] = min(100, squeeze_score)

    except Exception as e:
        logger.debug(f"Short squeeze {ticker}: {e}")

    _set_cache(f"squeeze_{ticker}", result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 13. Analyst Consensus (computed from Yahoo Finance data)
# ═════════════════════════════════════════════════════════════════════════════

def get_analyst_consensus(ticker):
    """
    Compute a clean analyst consensus from Yahoo Finance recommendations.
    Strong Buy=5, Buy=4, Hold=3, Sell=2, Strong Sell=1.
    Returns weighted score and consensus label.
    """
    cached = _cached(f"analyst_consensus_{ticker}")
    if cached:
        return cached

    result = {}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        # Try to get recommendation breakdown
        try:
            recs = t.recommendations
            if recs is not None and not recs.empty:
                # Get the most recent row(s)
                latest = recs.tail(1).iloc[0] if len(recs) > 0 else None
                if latest is not None:
                    strong_buy = int(latest.get("strongBuy", 0))
                    buy = int(latest.get("buy", 0))
                    hold = int(latest.get("hold", 0))
                    sell = int(latest.get("sell", 0))
                    strong_sell = int(latest.get("strongSell", 0))

                    result["analyst_strong_buy"] = strong_buy
                    result["analyst_buy"] = buy
                    result["analyst_hold"] = hold
                    result["analyst_sell"] = sell
                    result["analyst_strong_sell"] = strong_sell

                    total = strong_buy + buy + hold + sell + strong_sell
                    if total > 0:
                        weighted = (
                            strong_buy * 5 + buy * 4 + hold * 3 + sell * 2 + strong_sell * 1
                        ) / total
                        result["analyst_score"] = round(weighted, 2)
                        result["analyst_total"] = total

                        if weighted >= 4.5:
                            result["analyst_consensus_label"] = "Strong Buy"
                        elif weighted >= 3.5:
                            result["analyst_consensus_label"] = "Buy"
                        elif weighted >= 2.5:
                            result["analyst_consensus_label"] = "Hold"
                        elif weighted >= 1.5:
                            result["analyst_consensus_label"] = "Sell"
                        else:
                            result["analyst_consensus_label"] = "Strong Sell"
        except Exception as e:
            logger.debug(f"Analyst recs detail {ticker}: {e}")

        # Fallback to recommendationKey from info
        if "analyst_score" not in result:
            rec_key = info.get("recommendationKey", "")
            score_map = {
                "strong_buy": 5.0, "buy": 4.0, "hold": 3.0,
                "sell": 2.0, "strong_sell": 1.0, "underperform": 2.0,
                "outperform": 4.0,
            }
            if rec_key in score_map:
                result["analyst_score"] = score_map[rec_key]
                result["analyst_consensus_label"] = rec_key.replace("_", " ").title()

    except Exception as e:
        logger.debug(f"Analyst consensus {ticker}: {e}")

    _set_cache(f"analyst_consensus_{ticker}", result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# MASTER AGGREGATOR — Consolidate all sources into one profile
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

    # ── Source 7: CNN Fear & Greed Index ───────────────────────────────
    try:
        fg = get_fear_greed_index()
        if fg:
            profile.update(fg)
            profile["sources_hit"] += 1
    except Exception:
        profile["sources_failed"] += 1

    # ── Source 8: OpenInsider — Insider Trades ─────────────────────────
    try:
        insider = get_insider_trades(ticker)
        if insider:
            profile.update(insider)
            profile["sources_hit"] += 1
    except Exception:
        profile["sources_failed"] += 1

    # ── Source 9: Zacks Rank & EPS Estimates ───────────────────────────
    try:
        zacks = get_zacks_data(ticker)
        if zacks:
            profile.update(zacks)
            profile["sources_hit"] += 1
    except Exception:
        profile["sources_failed"] += 1

    # ── Source 10: FRED Macro Data ─────────────────────────────────────
    try:
        macro = get_fred_macro()
        if macro:
            profile.update(macro)
            profile["sources_hit"] += 1
    except Exception:
        profile["sources_failed"] += 1

    # ── Source 11: Social Sentiment (StockTwits) ───────────────────────
    try:
        social = get_social_sentiment(ticker)
        if social:
            profile.update(social)
            profile["sources_hit"] += 1
    except Exception:
        profile["sources_failed"] += 1

    # ── Source 12: Short Squeeze Score ─────────────────────────────────
    try:
        squeeze = get_short_squeeze_data(ticker)
        if squeeze:
            profile.update(squeeze)
            profile["sources_hit"] += 1
    except Exception:
        profile["sources_failed"] += 1

    # ── Source 13: Analyst Consensus ───────────────────────────────────
    try:
        analyst = get_analyst_consensus(ticker)
        if analyst:
            profile.update(analyst)
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

    # ── Fear & Greed Index (+/- 5) ────────────────────────────────────
    fg_score = p.get("fg_score")
    if fg_score is not None:
        if fg_score <= 20:
            score += 5   # Extreme Fear = contrarian bullish
        elif fg_score <= 35:
            score += 3   # Fear = mildly contrarian bullish
        elif fg_score >= 80:
            score -= 3   # Extreme Greed = caution
        elif fg_score >= 65:
            score -= 1   # Greed = slight caution

    # ── Insider Trading Sentiment (+/- 8) ─────────────────────────────
    insider_sentiment = p.get("insider_sentiment")
    insider_net = p.get("insider_net_buys", 0)
    if insider_sentiment == "bullish" and insider_net > 0:
        score += 8   # Net insider buying = strong bullish signal
    elif insider_sentiment == "bearish" and insider_net < 0:
        score -= 5   # Net insider selling = bearish signal

    # ── Zacks Rank (+/- 8) ────────────────────────────────────────────
    zacks_rank = p.get("zacks_rank")
    if zacks_rank is not None:
        if zacks_rank == 1:
            score += 8   # Strong Buy
        elif zacks_rank == 2:
            score += 5   # Buy
        elif zacks_rank == 4:
            score -= 4   # Sell
        elif zacks_rank == 5:
            score -= 8   # Strong Sell

    # ── Social Sentiment (+/- 4) ──────────────────────────────────────
    social_score = p.get("social_sentiment_score")
    if social_score is not None:
        if social_score >= 75:
            score += 4   # Strong bullish social sentiment
        elif social_score >= 60:
            score += 2
        elif social_score <= 25:
            score -= 3   # Strong bearish social sentiment
        elif social_score <= 40:
            score -= 1

    # ── Short Squeeze Potential (+5) ──────────────────────────────────
    squeeze_score = p.get("short_squeeze_score")
    if squeeze_score is not None:
        if squeeze_score >= 70:
            score += 5   # High squeeze potential
        elif squeeze_score >= 50:
            score += 3
        elif squeeze_score >= 30:
            score += 1

    # ── Analyst Consensus (+/- 5) ─────────────────────────────────────
    analyst_score = p.get("analyst_score")
    if analyst_score is not None:
        if analyst_score >= 4.5:
            score += 5   # Strong Buy consensus
        elif analyst_score >= 3.8:
            score += 3
        elif analyst_score <= 1.5:
            score -= 5   # Strong Sell consensus
        elif analyst_score <= 2.2:
            score -= 3

    # ── Macro: Rising yields cautious on growth (-2) ──────────────────
    t10 = p.get("treasury_10y")
    t10_prev = p.get("treasury_10y_prev")
    if t10 is not None and t10_prev is not None:
        yield_change = t10 - t10_prev
        if yield_change > 0.05:
            # 10Y yield rising fast — cautious on growth stocks
            score -= 2

    return max(0, min(100, score))
