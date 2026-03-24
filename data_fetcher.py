"""
data_fetcher.py — StockPulse data layer
Sources: Yahoo Finance (yfinance + screener API), RSS news feeds
Optimized: batch downloads, screener-first approach for speed
"""

import re
import yfinance as yf
import feedparser
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import time

# Multi-source market intelligence
try:
    from market_scraper import get_enriched_ticker_profile, get_cboe_pc_ratio
except ImportError:
    get_enriched_ticker_profile = None
    get_cboe_pc_ratio = None

logger = logging.getLogger(__name__)

# ─── High-conviction watchlist (active, liquid tickers) ──────────────────────
WATCHLIST = [
    "NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "META", "AMD", "NFLX", "GOOG",
    "PLTR", "MSTR", "COIN", "HOOD", "SMCI", "ARM", "AVGO", "MU", "INTC",
    "SOFI", "RIVN", "SHOP", "NOW", "CRM", "SNOW", "NET", "PANW", "CRWD",
    "RBLX", "ABNB", "UBER", "PYPL", "BABA", "GME", "AMC",
]

SECTOR_ETFS = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF",
    "Energy": "XLE", "Consumer Disc.": "XLY", "Industrials": "XLI",
    "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE",
    "Comm. Services": "XLC",
}

NEWS_FEEDS = [
    ("Yahoo Finance",    "https://finance.yahoo.com/news/rssindex"),
    ("MarketWatch",      "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("Reuters Markets",  "https://feeds.reuters.com/reuters/businessNews"),
    ("Reuters Stocks",   "https://feeds.reuters.com/reuters/USStocksNews"),
    ("Investing.com",    "https://www.investing.com/rss/news.rss"),
    ("Seeking Alpha",    "https://seekingalpha.com/market_currents.xml"),
    ("CNBC Markets",     "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("CNBC Finance",     "https://www.cnbc.com/id/10000664/device/rss/rss.html"),
    ("Bloomberg",        "https://feeds.bloomberg.com/markets/news.rss"),
    ("WSJ Markets",      "https://feeds.content.dowjones.io/public/rss/WSJcomUSBusiness"),
    ("WSJ Economy",      "https://feeds.content.dowjones.io/public/rss/wsj_economics"),
    ("NY Times Business","https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"),
    ("NY Times Economy", "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml"),
    ("Barron's",         "https://www.barrons.com/xml/rss/3_7514.xml"),
    ("Forbes Markets",   "https://www.forbes.com/investing/feed2/"),
    ("AP Business",      "https://rsshub.app/apnews/topics/business-news"),
    ("FT Markets",       "https://www.ft.com/markets?format=rss"),
    ("Business Insider", "https://markets.businessinsider.com/rss/news"),
    ("Benzinga",         "https://www.benzinga.com/feed"),
    ("The Street",       "https://www.thestreet.com/rss/index.xml"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


# ─── Market Overview ─────────────────────────────────────────────────────────

def get_market_overview():
    tickers = {"S&P 500": "SPY", "NASDAQ": "QQQ", "DOW": "DIA", "VIX": "^VIX"}
    result = []
    try:
        syms = list(tickers.values())
        data = yf.download(syms, period="2d", interval="1d", auto_adjust=True, progress=False)
        closes = data["Close"]
        for name, sym in tickers.items():
            try:
                prices = closes[sym].dropna() if sym in closes.columns else closes.dropna()
                if len(prices) >= 2:
                    prev, curr = float(prices.iloc[-2]), float(prices.iloc[-1])
                    change = curr - prev
                    pct = (change / prev) * 100
                else:
                    curr = float(prices.iloc[-1]); change = 0; pct = 0
                result.append({"name": name, "symbol": sym, "price": round(curr, 2),
                                "change": round(change, 2), "pct": round(pct, 2), "positive": change >= 0})
            except Exception as e:
                logger.debug(f"Market overview {name}: {e}")
    except Exception as e:
        logger.error(f"Market overview: {e}")
    return result


# ─── Yahoo Finance Screener — Real Top Movers ────────────────────────────────

def get_yahoo_screener_movers():
    """Fetch real top gainers from Yahoo Finance screener API."""
    url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
    params = {"formatted": "true", "lang": "en-US", "region": "US",
              "scrIds": "day_gainers", "count": 25}
    tickers = []
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        data = r.json()
        quotes = data["finance"]["result"][0]["quotes"]
        for q in quotes:
            sym = q.get("symbol", "")
            if sym and "." not in sym:
                tickers.append(sym)
    except Exception as e:
        logger.error(f"Screener error: {e}")
    # Fallback: merge with watchlist
    combined = list(dict.fromkeys(tickers + WATCHLIST))
    return combined[:30]


# ─── Technical Signals (batch-friendly) ──────────────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig


def get_technical_signals_from_history(hist):
    """Compute RSI, MACD, MA signals from a pre-downloaded history DataFrame."""
    try:
        if hist is None or hist.empty or len(hist) < 30:
            return None
        close = hist["Close"] if "Close" in hist.columns else hist.iloc[:, 0]

        rsi_s = compute_rsi(close)
        rsi = round(float(rsi_s.iloc[-1]), 1)
        rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"

        macd_line, sig_line = compute_macd(close)
        mv, sv = float(macd_line.iloc[-1]), float(sig_line.iloc[-1])
        mv2, sv2 = float(macd_line.iloc[-2]), float(sig_line.iloc[-2])
        if mv2 < sv2 and mv > sv:
            macd_signal = "Bullish Cross"
        elif mv2 > sv2 and mv < sv:
            macd_signal = "Bearish Cross"
        elif mv > sv:
            macd_signal = "Bullish"
        else:
            macd_signal = "Bearish"

        ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        if ma50 and ma200:
            ma_signal = "Golden Cross (Bullish)" if ma50 > ma200 else "Death Cross (Bearish)"
        else:
            ma_signal = "N/A"

        return {"rsi": rsi, "rsi_signal": rsi_signal, "macd_signal": macd_signal, "ma_signal": ma_signal}
    except Exception as e:
        logger.error(f"Technical signals: {e}")
        return None


# ─── Finviz Data (analyst ratings, insider activity, short interest) ─────────

def get_finviz_data(ticker):
    """Scrape Finviz for analyst recommendation, target price, short float, insider activity."""
    try:
        if not re.match(r"^[A-Z0-9]{1,5}(-[A-Z])?$", ticker.upper()):
            return {}
        url = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code != 200:
            return {}
        soup = BeautifulSoup(r.text, "html.parser")

        # Parse fundamentals table (label-value pairs)
        data = {}
        rows = soup.select("tr.table-dark-row, tr.table-dark-row-2")
        for row in rows:
            cells = row.find_all("td")
            for i in range(0, len(cells) - 1, 2):
                key = cells[i].get_text(strip=True)
                val = cells[i + 1].get_text(strip=True)
                data[key] = val

        result = {}

        # Analyst recommendation (1.0=Strong Buy … 5.0=Strong Sell)
        recom = data.get("Recom", "-")
        if recom and recom != "-":
            try:
                result["analyst_recom"] = float(recom)
            except Exception:
                pass

        # Analyst target price
        target = data.get("Target Price", "-")
        if target and target != "-":
            try:
                result["target_price"] = float(target)
            except Exception:
                pass

        # Short float (squeeze potential)
        short = data.get("Short Float / Ratio", data.get("Short Float", "-"))
        if short and short != "-":
            try:
                result["short_float"] = float(short.split("/")[0].replace("%", "").strip())
            except Exception:
                pass

        # Insider transactions (positive % = net buying)
        insider = data.get("Insider Trans", "-")
        if insider and insider != "-":
            result["insider_trans"] = insider  # e.g. "+5.23%" or "-2.10%"

        # EPS quarter over quarter growth
        eps_qq = data.get("EPS Q/Q", "-")
        if eps_qq and eps_qq != "-":
            result["eps_qq"] = eps_qq

        # Earnings date from Finviz as backup
        earnings = data.get("Earnings", "-")
        if earnings and earnings != "-":
            result["finviz_earnings"] = earnings

        return result
    except Exception as e:
        logger.error(f"Finviz {ticker}: {e}")
        return {}


# ─── Unusual Options Detector ────────────────────────────────────────────────

def detect_unusual_options(calls_df, puts_df, current_price):
    """
    Flag unusual options activity:
    - Volume/OI ratio > 1.5 (fresh positions, not existing)
    - Notional value (volume × last_price × 100) > $250k
    Returns list of unusual call signals and put signals.
    """
    unusual_calls, unusual_puts = [], []
    for df, kind, result in [(calls_df, "CALL", unusual_calls), (puts_df, "PUT", unusual_puts)]:
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            vol  = float(row.get("volume", 0) or 0)
            oi   = float(row.get("openInterest", 1) or 1)
            last = float(row.get("lastPrice", 0) or 0)
            strike = float(row.get("strike", 0) or 0)
            if vol < 50:
                continue
            vol_oi = vol / oi if oi > 0 else vol
            notional = vol * last * 100
            if vol_oi >= 1.5 and notional >= 250_000:
                result.append({
                    "strike": round(strike, 2),
                    "volume": int(vol),
                    "oi": int(oi),
                    "vol_oi": round(vol_oi, 1),
                    "notional_m": round(notional / 1_000_000, 2),
                    "type": kind,
                })
    return unusual_calls, unusual_puts


# ─── Options Data ────────────────────────────────────────────────────────────

def _score_option(row, current_price, kind, intel_score=50):
    """
    Wall-Street-grade option scoring with multi-source intelligence.
    Evaluates each contract on 7 dimensions a professional trader cares about:
      1. Moneyness   — slightly OTM (2-8%) is the sweet spot for risk/reward
      2. Liquidity    — tight bid/ask spread, high volume + open interest
      3. Volume/OI    — ratio > 1 = fresh money flowing in (institutional signal)
      4. IV value     — moderate IV preferred (not overpriced, not dead)
      5. Notional     — bigger dollar flow = institutional conviction
      6. Premium      — filter out penny options and overpriced ones
      7. Intel boost  — multi-source intelligence score from aggregated data
    """
    try:
        strike = float(row.get("strike", 0) or 0)
        volume = float(row.get("volume", 0) or 0)
        oi     = float(row.get("openInterest", 0) or 0)
        last   = float(row.get("lastPrice", 0) or 0)
        bid    = float(row.get("bid", 0) or 0)
        ask    = float(row.get("ask", 0) or 0)
        iv     = float(row.get("impliedVolatility", 0) or 0)

        if current_price <= 0 or strike <= 0 or last <= 0:
            return -999

        # ── 1. Moneyness (0-30 pts) ─────────────────────────────────────
        # Sweet spot: slightly OTM (2-8% from current price)
        if kind == "CALL":
            otm_pct = (strike - current_price) / current_price * 100
        else:
            otm_pct = (current_price - strike) / current_price * 100

        if -2 <= otm_pct <= 2:        # Near ATM — good for high-probability
            moneyness_score = 25
        elif 2 < otm_pct <= 5:        # Slightly OTM — best risk/reward
            moneyness_score = 30
        elif 5 < otm_pct <= 10:       # Moderately OTM — still tradeable
            moneyness_score = 18
        elif 10 < otm_pct <= 15:      # Far OTM — speculative but acceptable
            moneyness_score = 8
        elif -5 <= otm_pct < -2:      # Slightly ITM — good for safer plays
            moneyness_score = 22
        elif otm_pct > 15 or otm_pct < -10:
            moneyness_score = 0        # Too far — skip
        else:
            moneyness_score = 5

        # ── 2. Liquidity (0-25 pts) ──────────────────────────────────────
        # Volume + OI + tight spread = easy to enter/exit
        liq_score = 0
        if volume >= 500:
            liq_score += 10
        elif volume >= 100:
            liq_score += 6
        elif volume >= 25:
            liq_score += 2

        if oi >= 1000:
            liq_score += 8
        elif oi >= 200:
            liq_score += 4
        elif oi >= 50:
            liq_score += 1

        # Bid-ask spread as % of mid price (tighter = better)
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2
            spread_pct = (ask - bid) / mid * 100 if mid > 0 else 100
            if spread_pct <= 3:
                liq_score += 7     # Very tight — institutional-grade
            elif spread_pct <= 8:
                liq_score += 4
            elif spread_pct <= 15:
                liq_score += 1

        # ── 3. Volume/OI ratio (0-20 pts) ────────────────────────────────
        # Vol/OI > 1 = new positions being opened (smart money signal)
        vol_oi = volume / oi if oi > 0 else 0
        if vol_oi >= 3.0:
            voi_score = 20         # Massive fresh flow — very unusual
        elif vol_oi >= 1.5:
            voi_score = 15         # Strong fresh positioning
        elif vol_oi >= 0.8:
            voi_score = 8          # Decent activity
        elif vol_oi >= 0.3:
            voi_score = 3
        else:
            voi_score = 0

        # ── 4. IV value (0-10 pts) ───────────────────────────────────────
        # Moderate IV = not overpriced. Too low = dead stock. Too high = expensive.
        iv_pct = iv * 100
        if 20 <= iv_pct <= 50:
            iv_score = 10          # Sweet spot
        elif 50 < iv_pct <= 80:
            iv_score = 6           # Elevated but ok for momentum plays
        elif 15 <= iv_pct < 20:
            iv_score = 5
        elif iv_pct > 80:
            iv_score = 2           # Very expensive premium
        else:
            iv_score = 0

        # ── 5. Notional value (0-10 pts) ─────────────────────────────────
        # Bigger $ flow = more institutional conviction
        notional = volume * last * 100
        if notional >= 5_000_000:
            not_score = 10
        elif notional >= 1_000_000:
            not_score = 7
        elif notional >= 250_000:
            not_score = 4
        elif notional >= 50_000:
            not_score = 1
        else:
            not_score = 0

        # ── 6. Premium filter (0-5 pts) ──────────────────────────────────
        # Filter out penny options (< $0.10) and extreme prices
        prem_pct = last / current_price * 100
        if 0.5 <= prem_pct <= 8:
            prem_score = 5         # Reasonable premium range
        elif 0.2 <= prem_pct < 0.5 or 8 < prem_pct <= 15:
            prem_score = 2
        else:
            prem_score = 0

        # Hard filters — disqualify bad contracts
        if last < 0.10:
            return -999            # Penny option — untradeable
        if volume < 10:
            return -999            # No liquidity
        if otm_pct > 20:
            return -999            # Way too far OTM

        # ── 7. Multi-source intelligence boost (0-15 pts) ───────────
        # intel_score is 0-100 from aggregated data (analyst, IV rank,
        # fundamentals, earnings, institutional activity)
        intel_boost = 0
        if intel_score >= 70:
            intel_boost = 15       # Strong conviction from all sources
        elif intel_score >= 60:
            intel_boost = 10
        elif intel_score >= 50:
            intel_boost = 5
        elif intel_score <= 30:
            intel_boost = -10      # Red flags from multiple sources

        total = moneyness_score + liq_score + voi_score + iv_score + not_score + prem_score + intel_boost
        return total

    except Exception:
        return -999


def get_options_data(ticker, use_intel=True):
    """
    Fetch options chain and select the best contracts using professional-grade
    scoring: moneyness, liquidity, vol/OI flow, IV, notional, premium quality.
    Set use_intel=False for faster scans (skips multi-source scraper).
    """
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return None

        # Use first 3 expiries for better selection (1-4 weeks out typically)
        all_calls = pd.DataFrame()
        all_puts  = pd.DataFrame()
        for exp in exps[:3]:
            try:
                chain = t.option_chain(exp)
                c = chain.calls.copy(); c["expiry"] = exp
                p = chain.puts.copy();  p["expiry"] = exp
                all_calls = pd.concat([all_calls, c], ignore_index=True)
                all_puts  = pd.concat([all_puts,  p], ignore_index=True)
            except Exception:
                logger.debug(f"Options chain {ticker} {exp}: failed")

        if all_calls.empty and all_puts.empty:
            return None

        total_calls = int(all_calls["volume"].fillna(0).sum())
        total_puts  = int(all_puts["volume"].fillna(0).sum())
        cp_ratio = round(total_puts / total_calls, 2) if total_calls > 0 else None

        # Get current price for moneyness calculation
        current_price = 0
        try:
            current_price = float(t.fast_info.last_price or 0)
        except Exception:
            pass

        unusual_calls, unusual_puts = detect_unusual_options(all_calls, all_puts, current_price)

        # ── Fetch multi-source intelligence for smarter scoring ──────
        intel_score = 50  # neutral default
        intel_profile = {}
        if use_intel and get_enriched_ticker_profile is not None:
            try:
                intel_profile = get_enriched_ticker_profile(ticker)
                intel_score = intel_profile.get("intel_score", 50)
                logger.debug(f"Options {ticker}: intel_score={intel_score}, sources={intel_profile.get('sources_hit', 0)}")
            except Exception as e:
                logger.debug(f"Options {ticker} intel failed: {e}")

        def fmt(row, kind):
            exp = row.get("expiry", exps[0])
            return {
                "ticker": ticker, "type": kind,
                "strike": round(float(row["strike"]), 2),
                "expiry": exp,
                "volume": int(row["volume"]) if not pd.isna(row["volume"]) else 0,
                "open_interest": int(row["openInterest"]) if not pd.isna(row["openInterest"]) else 0,
                "iv": round(float(row["impliedVolatility"]) * 100, 1) if not pd.isna(row["impliedVolatility"]) else 0,
                "last_price": round(float(row["lastPrice"]), 2) if not pd.isna(row["lastPrice"]) else 0,
                "bid": round(float(row.get("bid", 0) or 0), 2),
                "ask": round(float(row.get("ask", 0) or 0), 2),
            }

        # ── Score and rank calls ─────────────────────────────────────────
        if not all_calls.empty and current_price > 0:
            all_calls["_score"] = all_calls.apply(
                lambda r: _score_option(r, current_price, "CALL", intel_score), axis=1
            )
            best_calls = all_calls[all_calls["_score"] > 0].sort_values(
                "_score", ascending=False
            ).head(5)
            top_calls = [fmt(r, "CALL") for _, r in best_calls.iterrows()]
        else:
            # Fallback to volume sort if no price data
            top_calls = [fmt(r, "CALL") for _, r in all_calls.dropna(
                subset=["volume"]).sort_values("volume", ascending=False
            ).head(5).iterrows()]

        # ── Score and rank puts ──────────────────────────────────────────
        if not all_puts.empty and current_price > 0:
            all_puts["_score"] = all_puts.apply(
                lambda r: _score_option(r, current_price, "PUT", intel_score), axis=1
            )
            best_puts = all_puts[all_puts["_score"] > 0].sort_values(
                "_score", ascending=False
            ).head(5)
            top_puts = [fmt(r, "PUT") for _, r in best_puts.iterrows()]
        else:
            top_puts = [fmt(r, "PUT") for _, r in all_puts.dropna(
                subset=["volume"]).sort_values("volume", ascending=False
            ).head(5).iterrows()]

        return {
            "total_calls": total_calls, "total_puts": total_puts,
            "cp_ratio": cp_ratio, "top_calls": top_calls, "top_puts": top_puts,
            "expiry": exps[0],
            "unusual_calls": unusual_calls, "unusual_puts": unusual_puts,
        }
    except Exception as e:
        logger.error(f"Options {ticker}: {e}")
        return None


# ─── Top Picks — Batch Download Approach ────────────────────────────────────

def get_top_picks(earnings_list=None):
    """
    1. Get candidate tickers from Yahoo screener + watchlist + TRENDING
    2. Batch-download 1yr history for all at once (fast)
    3. Score each ticker (20 factors + trending boost)
    4. Get options data only for top 15 (to limit API calls)
    """
    if earnings_list is None:
        earnings_list = []
    earnings_tickers = {e["ticker"]: e for e in earnings_list}

    candidates = get_yahoo_screener_movers()

    # ── Inject trending tickers into candidate pool ──────────────────
    trending_map = {}  # ticker -> {sources: [...], source_count: int}
    try:
        from trending import (
            get_finviz_trending, get_yahoo_trending,
            get_reddit_wsb_trending, get_stocktwits_trending
        )
        sources_data = {
            "Finviz": get_finviz_trending(),
            "Yahoo": get_yahoo_trending(),
            "Reddit": get_reddit_wsb_trending(),
            "StockTwits": get_stocktwits_trending(),
        }
        for source_name, tickers_list in sources_data.items():
            for t in tickers_list:
                if t not in trending_map:
                    trending_map[t] = {"sources": [], "source_count": 0}
                trending_map[t]["sources"].append(source_name)
                trending_map[t]["source_count"] += 1

        # Add trending tickers to candidates (prioritize multi-source)
        trending_sorted = sorted(trending_map.keys(),
                                  key=lambda t: trending_map[t]["source_count"],
                                  reverse=True)
        added = 0
        for t in trending_sorted:
            if t not in candidates:
                candidates.append(t)
                added += 1
            if added >= 15:  # cap to avoid bloating
                break
        logger.info(f"Trending: {len(trending_map)} tickers found, {added} new added to candidates")
    except Exception as e:
        logger.debug(f"Trending injection failed: {e}")

    logger.info(f"Scoring {len(candidates)} candidates...")

    # Batch download 1yr history for all candidates
    try:
        bulk = yf.download(candidates, period="1y", interval="1d",
                           auto_adjust=True, group_by="ticker", progress=False)
    except Exception as e:
        logger.error(f"Bulk download error: {e}")
        bulk = None

    # Also batch download 10d for short-term momentum
    try:
        bulk_10d = yf.download(candidates, period="10d", interval="1d",
                               auto_adjust=True, progress=False)
        closes_10d = bulk_10d["Close"] if "Close" in bulk_10d.columns else None
    except Exception:
        closes_10d = None

    scored = []
    for ticker in candidates:
        try:
            # Get 10d price change and volume
            pct_gain = 0
            vol_surge = 1.0
            current_price = 0

            if closes_10d is not None and ticker in closes_10d.columns:
                prices_10d = closes_10d[ticker].dropna()
                if len(prices_10d) >= 2:
                    current_price = float(prices_10d.iloc[-1])
                    first_price = float(prices_10d.iloc[0])
                    if first_price > 0:
                        pct_gain = ((current_price - first_price) / first_price) * 100

            # Get 1yr history for technicals
            hist_1y = None
            if bulk is not None:
                try:
                    if ticker in bulk.columns.get_level_values(0):
                        hist_1y = bulk[ticker]
                    elif hasattr(bulk, 'columns') and len(candidates) == 1:
                        hist_1y = bulk
                except Exception:
                    pass

            tech = get_technical_signals_from_history(hist_1y)

            # Score
            rsi_score = 0.5
            macd_score = 0.5
            if tech:
                rsi_score = 1.0 if tech["rsi_signal"] == "Oversold" else 0.0 if tech["rsi_signal"] == "Overbought" else 0.5
                macd_score = 1.0 if "Bullish" in tech["macd_signal"] else 0.0 if "Bearish" in tech["macd_signal"] else 0.5

            gain_score = min(max(pct_gain / 20, 0), 1)
            composite = gain_score * 35 + vol_surge * 10 + rsi_score * 25 + macd_score * 30

            scored.append({
                "ticker": ticker,
                "company": ticker,  # Will enrich top 10 only
                "sector": "N/A",
                "price": round(current_price, 2),
                "pct_gain": round(pct_gain, 2),
                "vol_surge": round(vol_surge, 1),
                "score": round(composite, 1),
                "rsi": tech["rsi"] if tech else None,
                "rsi_signal": tech["rsi_signal"] if tech else "N/A",
                "macd_signal": tech["macd_signal"] if tech else "N/A",
                "ma_signal": tech["ma_signal"] if tech else "N/A",
                "sentiment": "Neutral",
                "total_calls": 0,
                "total_puts": 0,
                "cp_ratio": None,
                "has_earnings": ticker in earnings_tickers,
                "earnings_date": earnings_tickers[ticker]["date"] if ticker in earnings_tickers else None,
                "eps_estimate": earnings_tickers[ticker]["eps_estimate"] if ticker in earnings_tickers else None,
            })
        except Exception as e:
            logger.error(f"Score error {ticker}: {e}")

    # Filter out penny stocks, leveraged ETFs, and junk
    # A Wall Street pro doesn't recommend $1 stocks
    JUNK_PATTERNS = re.compile(r'^(TQQQ|SQQQ|UVXY|SPXS|SPXL|LABU|LABD|SOXL|SOXS|FNGU|FNGD|TZA|TNA|CRCD|YANG|YINN)', re.IGNORECASE)
    scored = [
        s for s in scored
        if s["price"] >= 10.0                   # No penny/micro stocks under $10
        and not JUNK_PATTERNS.match(s["ticker"])  # No leveraged/inverse ETFs
    ]

    # Sort and take top 15 for enrichment
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:15]

    # Fetch news once for all tickers (used for news sentiment per pick)
    logger.info("Fetching news for sentiment scanning...")
    try:
        news_articles = get_news_feed()
    except Exception:
        news_articles = []

    # Enrich top 15: company info + options + Finviz + multi-source intel + news
    logger.info("Enriching top 15 with ALL 13 data sources...")
    enriched = []

    # ── Fetch market-wide data once (shared across all tickers) ──────
    market_intel = {}
    if get_enriched_ticker_profile is not None:
        try:
            from market_scraper import get_fear_greed_index, get_fred_macro, get_cboe_pc_ratio as _cboe
            fg = get_fear_greed_index()
            market_intel["fg_score"] = fg.get("fg_score")
            market_intel["fg_rating"] = fg.get("fg_rating")
            macro = get_fred_macro()
            market_intel.update(macro)
            cboe = _cboe()
            market_intel["market_vix"] = cboe.get("vix")
            market_intel["market_pc"] = cboe.get("total_pc")
            logger.info(f"Market intel: F&G={market_intel.get('fg_score')}, VIX={market_intel.get('market_vix')}, 10Y={market_intel.get('treasury_10y')}")
        except Exception as e:
            logger.debug(f"Market-wide intel failed: {e}")

    for s in top:
        ticker = s["ticker"]
        current_price = s.get("price", 0)

        # Company info
        try:
            t = yf.Ticker(ticker)
            info = t.info
            s["company"] = info.get("shortName", ticker)
            s["sector"]  = info.get("sector", "N/A")
        except Exception:
            s["company"] = ticker

        # Options (with unusual activity detection) — fast mode for picks
        opts = get_options_data(ticker, use_intel=False)
        unusual_calls, unusual_puts = [], []
        if opts:
            s["total_calls"]   = opts["total_calls"]
            s["total_puts"]    = opts["total_puts"]
            s["cp_ratio"]      = opts["cp_ratio"]
            unusual_calls      = opts.get("unusual_calls", [])
            unusual_puts       = opts.get("unusual_puts", [])
            s["unusual_calls"] = unusual_calls
            s["unusual_puts"]  = unusual_puts

        # Finviz: analyst rating, target price, insider activity, short float
        fv = get_finviz_data(ticker)
        s["finviz"] = fv

        # ── Multi-source intelligence per ticker ─────────────────────
        intel = {}
        if get_enriched_ticker_profile is not None:
            try:
                intel = get_enriched_ticker_profile(ticker)
                s["intel_score"] = intel.get("intel_score", 50)
                s["iv_rank"] = intel.get("iv_rank")
                s["short_squeeze_score"] = intel.get("short_squeeze_score")
                s["social_sentiment"] = intel.get("social_sentiment_score")
                s["analyst_consensus"] = intel.get("analyst_consensus_label")
                s["insider_sentiment"] = intel.get("insider_sentiment")
                s["zacks_rank"] = intel.get("zacks_rank")
                s["whisper_eps"] = intel.get("whisper_eps")
                s["beat_rate"] = intel.get("beat_rate")
                s["upside_pct"] = intel.get("upside_pct")
            except Exception as e:
                logger.debug(f"Intel {ticker}: {e}")

        # ── Multi-source scoring ──────────────────────────────────────────────
        score = 0
        signals = []
        conviction_count = 0

        pct      = s.get("pct_gain", 0)
        vsrg     = s.get("vol_surge", 1)
        rsi_sig  = s.get("rsi_signal", "")
        macd_sig = s.get("macd_signal", "")
        ma_sig   = s.get("ma_signal", "")
        cp       = s.get("cp_ratio")  # put/call ratio: <1 = bullish, >1 = bearish

        # 1. Momentum — heavy penalty for significant down days
        if pct > 5:
            score += 22; signals.append(f"Strong momentum +{round(pct,1)}%"); conviction_count += 1
        elif pct > 2:
            score += 14; signals.append(f"Momentum +{round(pct,1)}%")
        elif pct > 0:
            score += 5
        elif pct < -5:
            score -= 18; signals.append(f"Sharp drop {round(pct,1)}% today")
        elif pct < -3:
            score -= 10; signals.append(f"Down {round(pct,1)}% today")
        elif pct < -1:
            score -= 4

        # 2. Volume surge
        if vsrg > 3:
            score += 12; signals.append(f"Volume surge {vsrg}x"); conviction_count += 1
        elif vsrg > 1.5:
            score += 6

        # 3. RSI
        rsi = s.get("rsi", 50) or 50
        if rsi_sig == "Oversold":
            score += 14; signals.append(f"RSI Oversold ({rsi})"); conviction_count += 1
        elif rsi_sig == "Neutral":
            score += 5
        elif rsi_sig == "Overbought":
            score -= 5

        # 4. MACD
        if "Bullish Cross" in macd_sig:
            score += 14; signals.append("MACD Bullish Cross"); conviction_count += 1
        elif "Bullish" in macd_sig:
            score += 8; signals.append("MACD Bullish")
        elif "Bearish" in macd_sig:
            score -= 6

        # 5. Moving averages
        if "Golden" in ma_sig:
            score += 10; signals.append("Golden Cross"); conviction_count += 1
        elif "Death" in ma_sig:
            score -= 8  # increased penalty

        # 6. Options flow — put/call ratio (cp < 1 = more calls = bullish)
        if cp is not None:
            if cp < 0.4:
                score += 18; signals.append("Extreme call dominance"); conviction_count += 1
            elif cp < 0.7:
                score += 10; signals.append("Bullish options flow")
            elif cp > 2.0:
                score -= 14; signals.append("Heavy put dominance (bearish)")
            elif cp > 1.5:
                score -= 8; signals.append("Put-heavy flow (bearish)")
            elif cp > 1.2:
                score -= 4

        # 7. Unusual options — compare call vs put notional
        unusual_call_notional = sum(u["notional_m"] for u in unusual_calls) if unusual_calls else 0
        unusual_put_notional  = sum(u["notional_m"] for u in unusual_puts)  if unusual_puts  else 0

        if unusual_calls and unusual_call_notional > unusual_put_notional:
            score += 16; conviction_count += 1
            signals.append(f"Unusual call sweep ${unusual_call_notional:.1f}M")
        elif unusual_puts and unusual_put_notional > unusual_call_notional * 1.5:
            score -= 12; signals.append(f"Unusual put sweep ${unusual_put_notional:.1f}M (bearish)")
        elif unusual_calls:
            score += 8; signals.append(f"Mixed unusual activity")

        # 8. Finviz signals
        if fv:
            recom = fv.get("analyst_recom")
            if recom:
                if recom <= 1.5:
                    score += 14; signals.append(f"Analysts: Strong Buy ({recom})"); conviction_count += 1
                elif recom <= 2.0:
                    score += 10; signals.append(f"Analysts: Buy ({recom})")
                elif recom >= 3.5:
                    score -= 6; signals.append(f"Analysts: Hold/Sell ({recom})")
                elif recom >= 4.0:
                    score -= 12; signals.append(f"Analysts: Sell ({recom})")

            target = fv.get("target_price")
            if target and current_price and current_price > 0:
                upside_pct = (target / current_price - 1) * 100
                if upside_pct > 30:
                    score += 12; signals.append(f"Analyst target +{round(upside_pct)}% upside"); conviction_count += 1
                elif upside_pct > 20:
                    score += 8; signals.append(f"Analyst target +{round(upside_pct)}% upside")
                elif upside_pct > 10:
                    score += 4; signals.append(f"Analyst target +{round(upside_pct)}% upside")
                elif upside_pct < -5:
                    score -= 8; signals.append(f"Below analyst target {round(upside_pct)}%")

            insider = fv.get("insider_trans", "")
            if insider and insider.startswith("+"):
                score += 10; signals.append(f"Insider buying ({insider})"); conviction_count += 1
            elif insider and insider.startswith("-"):
                try:
                    insider_pct = abs(float(insider.replace("%", "").replace("+", "")))
                    if insider_pct > 3:
                        score -= 6; signals.append(f"Heavy insider selling ({insider})")
                    else:
                        signals.append(f"Insider selling ({insider})")
                except Exception:
                    signals.append(f"Insider selling ({insider})")

            short_float = fv.get("short_float")
            if short_float and short_float > 20:
                score += 8; signals.append(f"Short squeeze: {short_float}% shorted")
            elif short_float and short_float > 10:
                score += 4; signals.append(f"High short interest: {short_float}%")

            eps_qq = fv.get("eps_qq", "")
            if eps_qq and eps_qq != "-":
                try:
                    eps_val = float(eps_qq.replace("%", ""))
                    if eps_val > 20:
                        score += 8; signals.append(f"EPS growth {eps_qq} QoQ"); conviction_count += 1
                    elif eps_val > 0:
                        score += 3
                except Exception:
                    pass

        # 9. News sentiment scan (WSJ, NYT, Reuters, Bloomberg, etc.)
        try:
            news_scan = scan_news_for_ticker(ticker, news_articles)
            if news_scan["count"] > 0:
                net = news_scan["bull"] - news_scan["bear"]
                if net >= 2:
                    score += 12; signals.append(f"Bullish press ({news_scan['count']} articles)")
                    conviction_count += 1
                elif net >= 1:
                    score += 6; signals.append(f"Positive news coverage")
                elif net <= -2:
                    score -= 12; signals.append(f"Negative press ({news_scan['count']} articles)")
                elif net <= -1:
                    score -= 6; signals.append(f"Mixed/negative news")
        except Exception as e:
            logger.debug(f"Scoring signals for {ticker}: {e}")

        # ── NEW: Multi-source intelligence signals (10-17) ───────────

        # 10. CNN Fear & Greed — contrarian indicator
        fg = market_intel.get("fg_score")
        if fg is not None:
            if fg <= 20:
                score += 8; signals.append(f"Extreme Fear ({fg}) — contrarian bullish")
                conviction_count += 1
            elif fg <= 35:
                score += 4; signals.append(f"Fear market ({fg})")
            elif fg >= 80:
                score -= 5; signals.append(f"Extreme Greed ({fg}) — caution")
            elif fg >= 65:
                score -= 2

        # 11. VIX level — elevated VIX = opportunity for value, extreme = risk
        vix = market_intel.get("market_vix")
        if vix is not None:
            if 20 <= vix <= 30:
                score += 4; signals.append(f"Elevated VIX ({vix}) — volatility opportunity")
            elif vix > 35:
                score -= 4; signals.append(f"VIX spike ({vix}) — high risk")
            elif vix < 14:
                score -= 1  # Complacency

        # 12. IV Rank — cheap options = good for buying
        iv_rank = intel.get("iv_rank")
        if iv_rank is not None:
            if iv_rank <= 20:
                score += 6; signals.append(f"Low IV Rank ({iv_rank}%) — cheap options")
                conviction_count += 1
            elif iv_rank <= 35:
                score += 3
            elif iv_rank >= 80:
                score -= 4; signals.append(f"High IV Rank ({iv_rank}%) — expensive options")

        # 13. Insider buying (SEC filings via OpenInsider)
        insider_sent = intel.get("insider_sentiment")
        if insider_sent:
            if insider_sent == "bullish":
                score += 10; signals.append("Insider NET BUYING (SEC filings)")
                conviction_count += 1
            elif insider_sent == "bearish":
                score -= 6; signals.append("Insider NET SELLING (SEC filings)")

        # 14. Short Squeeze potential
        sq_score = intel.get("short_squeeze_score")
        if sq_score is not None and sq_score > 0:
            if sq_score >= 75:
                score += 8; signals.append(f"Short squeeze alert (score {sq_score})")
                conviction_count += 1
            elif sq_score >= 50:
                score += 4; signals.append(f"Short squeeze potential ({sq_score})")
            elif sq_score >= 30:
                score += 2

        # 15. Social sentiment (StockTwits)
        social = intel.get("social_sentiment_score")
        if social is not None:
            if social >= 70:
                score += 5; signals.append(f"Social sentiment bullish ({social}%)")
            elif social >= 55:
                score += 2
            elif social <= 30:
                score -= 4; signals.append(f"Social sentiment bearish ({social}%)")
            elif social <= 40:
                score -= 1

        # 16. Zacks Rank
        zr = intel.get("zacks_rank")
        if zr is not None:
            if zr == 1:
                score += 10; signals.append("Zacks #1 Strong Buy"); conviction_count += 1
            elif zr == 2:
                score += 6; signals.append("Zacks #2 Buy")
            elif zr == 4:
                score -= 4; signals.append("Zacks #4 Sell")
            elif zr == 5:
                score -= 8; signals.append("Zacks #5 Strong Sell")

        # 17. Macro environment — yield curve + treasury
        spread = market_intel.get("yield_curve_spread")
        t10y = market_intel.get("treasury_10y")
        t10y_prev = market_intel.get("treasury_10y_prev")
        if spread is not None:
            if spread < 0:
                score -= 3; signals.append(f"Inverted yield curve ({spread}bp) — recession risk")
        if t10y and t10y_prev:
            rate_change = t10y - t10y_prev
            sector = s.get("sector", "")
            if rate_change > 0.05 and sector in ("Technology", "Consumer Cyclical"):
                score -= 3; signals.append(f"Rising rates pressure on {sector}")
            elif rate_change < -0.05:
                score += 2  # Falling rates = growth stocks benefit

        # 18. Earnings beat rate (EarningsWhispers)
        beat_rate = intel.get("beat_rate")
        if beat_rate is not None:
            if beat_rate >= 80:
                score += 5; signals.append(f"Earnings beat {beat_rate}% of time")
                conviction_count += 1
            elif beat_rate >= 65:
                score += 2
            elif beat_rate <= 35:
                score -= 4; signals.append(f"Earnings miss rate high ({100-beat_rate}%)")

        # 19. Analyst consensus (weighted from Yahoo recommendations)
        analyst_label = intel.get("analyst_consensus_label")
        analyst_score_val = intel.get("analyst_score")
        if analyst_score_val is not None:
            if analyst_score_val >= 4.2:
                score += 6; signals.append(f"Wall Street consensus: {analyst_label}")
            elif analyst_score_val >= 3.5:
                score += 3
            elif analyst_score_val <= 2.0:
                score -= 5; signals.append(f"Wall Street consensus: {analyst_label}")

        # 20. Upside to price target (multi-source average)
        upside = intel.get("upside_pct")
        if upside is not None and upside != s.get("finviz", {}).get("target_price"):
            # Only add if different from Finviz target (avoid double counting)
            if upside >= 40:
                score += 6; signals.append(f"Yahoo target +{upside}% upside")
            elif upside >= 25:
                score += 3
            elif upside <= -10:
                score -= 4; signals.append(f"Below Yahoo target {upside}%")

        # 21. TRENDING BOOST — ticker appearing across multiple social/screener sources
        trending_info = trending_map.get(ticker)
        if trending_info:
            src_count = trending_info["source_count"]
            src_names = ", ".join(trending_info["sources"])
            if src_count >= 3:
                score += 15; signals.append(f"🔥 Trending on {src_count} sources ({src_names})")
                conviction_count += 1
            elif src_count >= 2:
                score += 10; signals.append(f"📈 Trending on {src_names}")
                conviction_count += 1
            elif src_count == 1:
                score += 4; signals.append(f"Trending on {src_names}")
            s["trending_sources"] = trending_info["sources"]
            s["trending_count"] = src_count

        # Conviction level — requires BOTH strong signal count AND strong score
        score = min(max(round(score, 1), 0), 100)
        if (conviction_count >= 4 and score >= 75) or score >= 88:
            conviction = "EXTREME"
        elif (conviction_count >= 3 and score >= 60) or score >= 72:
            conviction = "HIGH"
        elif (conviction_count >= 2 and score >= 45) or score >= 55:
            conviction = "MEDIUM"
        else:
            conviction = "LOW"

        # Sentiment from put/call ratio (cp = put/call: low = bullish, high = bearish)
        if cp is not None:
            s["sentiment"] = "Bullish" if cp < 0.7 else "Bearish" if cp > 1.2 else "Neutral"
        elif "Bullish" in macd_sig:
            s["sentiment"] = "Bullish"
        elif "Bearish" in macd_sig:
            s["sentiment"] = "Bearish"

        s["score"]      = score
        s["signals"]    = signals[:6]  # top 6 signals for display
        s["conviction"] = conviction

        enriched.append(s)

    enriched.sort(key=lambda x: x["score"], reverse=True)
    return enriched[:10]


# ─── Global Top Calls & Puts ─────────────────────────────────────────────────

def get_global_top_options():
    """
    Aggregate the single best call and put from EACH ticker across the
    top 20 watchlist tickers, then rank the pool. Uses fast mode (no
    multi-source scraper) to keep scheduler fast.

    Max 1 call + 1 put per ticker shown in the final top 5.
    """
    all_calls, all_puts = [], []

    # Top 20 most active tickers — fast scan without intel scraper
    scan_tickers = list(WATCHLIST[:20])

    for ticker in scan_tickers:
        try:
            opts = get_options_data(ticker, use_intel=False)
        except Exception as e:
            logger.debug(f"Global options {ticker}: {e}")
            continue
        if not opts:
            continue
        # Take only the BEST call and BEST put per ticker (already scored & sorted)
        if opts["top_calls"]:
            all_calls.append(opts["top_calls"][0])
        if opts["top_puts"]:
            all_puts.append(opts["top_puts"][0])

    def _rank(opt):
        """
        Rank by composite: notional conviction + volume/OI freshness.
        Notional = real dollar flow (institutional signal).
        Vol/OI > 1 = new money, not just old positions.
        """
        vol = opt["volume"]
        oi  = opt.get("open_interest", 1) or 1
        notional = vol * opt["last_price"] * 100
        vol_oi = min(vol / oi, 5)  # cap to avoid outlier distortion
        return notional * 0.5 + vol * 0.2 + vol_oi * 10000 * 0.3

    all_calls.sort(key=_rank, reverse=True)
    all_puts.sort(key=_rank, reverse=True)

    # Ensure no duplicate tickers in final output
    def _dedupe(options, limit=5):
        seen = set()
        result = []
        for opt in options:
            if opt["ticker"] not in seen:
                seen.add(opt["ticker"])
                result.append(opt)
                if len(result) >= limit:
                    break
        return result

    return _dedupe(all_calls, 5), _dedupe(all_puts, 5)


# ─── Earnings Calendar ───────────────────────────────────────────────────────

# Major companies to scan for earnings (prioritized by importance)
EARNINGS_WATCHLIST = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "NFLX",
    # Semiconductors
    "AMD", "INTC", "AVGO", "MU", "ARM", "QCOM", "SMCI",
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC", "C", "V", "MA", "BLK", "COIN", "HOOD",
    # Growth tech
    "CRM", "NOW", "SNOW", "PLTR", "PANW", "CRWD", "NET", "SHOP", "MSTR",
    # Consumer & retail
    "WMT", "COST", "HD", "NKE", "SBUX", "MCD", "DIS", "AMGN",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "LLY", "MRK",
    # Energy
    "XOM", "CVX",
    # Other active
    "UBER", "PYPL", "RIVN", "BABA", "GME", "AMC", "SOFI",
]

def get_earnings_calendar():
    earnings = []
    today = datetime.now()
    weekday = today.weekday()  # 0=Mon, 6=Sun

    # Always show the UPCOMING trading week (Mon–Fri)
    if weekday == 6:        # Sunday → next Monday
        days_to_monday = 1
    elif weekday == 5:      # Saturday → next Monday
        days_to_monday = 2
    else:                   # Weekday → this Monday
        days_to_monday = -weekday

    week_start = (today + timedelta(days=days_to_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
    week_end   = week_start + timedelta(days=4)  # Friday

    for ticker in EARNINGS_WATCHLIST:
        try:
            t = yf.Ticker(ticker)
            cal = t.calendar
            if cal is None:
                continue
            date = None
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date", [None])
                ed = ed[0] if ed else None
                date = pd.to_datetime(ed) if ed else None
            elif hasattr(cal, "index") and "Earnings Date" in cal.index:
                ed = cal.loc["Earnings Date"]
                date = pd.to_datetime(ed.iloc[0] if hasattr(ed, "iloc") else ed)
            if date is None:
                continue
            if week_start.date() <= date.date() <= week_end.date():
                info = t.info
                market_cap = info.get("marketCap", 0) or 0
                earnings.append({
                    "ticker": ticker,
                    "company": info.get("shortName", ticker),
                    "date": date.strftime("%a %b %d"),
                    "eps_estimate": round(info.get("forwardEps", 0) or 0, 2) or "N/A",
                    "sector": info.get("sector", "N/A"),
                    "market_cap": market_cap,
                })
        except Exception:
            continue

    # Sort by date then market cap (most important first)
    earnings.sort(key=lambda x: (x["date"], -(x.get("market_cap") or 0)))
    # Remove market_cap from output
    for e in earnings:
        e.pop("market_cap", None)
    return earnings[:15]


# ─── Market Sentiment ────────────────────────────────────────────────────────

def get_market_sentiment():
    try:
        vix_data = yf.Ticker("^VIX").history(period="5d")
        vix = round(float(vix_data["Close"].iloc[-1]), 2) if not vix_data.empty else 20.0

        if vix < 12:   fg_score, fg_label, fg_color = 85, "Extreme Greed", "#00d4aa"
        elif vix < 16: fg_score, fg_label, fg_color = 70, "Greed",          "#7bed9f"
        elif vix < 20: fg_score, fg_label, fg_color = 55, "Neutral",        "#ffa502"
        elif vix < 28: fg_score, fg_label, fg_color = 35, "Fear",           "#ff6b81"
        else:          fg_score, fg_label, fg_color = 15, "Extreme Fear",   "#ff4757"

        # Sector performance
        sectors = []
        syms = list(SECTOR_ETFS.values())
        data = yf.download(syms, period="5d", interval="1d", auto_adjust=True, progress=False)
        closes = data["Close"]
        for name, sym in SECTOR_ETFS.items():
            try:
                prices = closes[sym].dropna() if sym in closes.columns else None
                if prices is not None and len(prices) >= 2 and float(prices.iloc[-2]) > 0:
                    pct = ((float(prices.iloc[-1]) - float(prices.iloc[-2])) / float(prices.iloc[-2])) * 100
                    sectors.append({"name": name, "pct": round(pct, 2), "positive": pct >= 0})
            except Exception:
                pass
        sectors.sort(key=lambda x: x["pct"], reverse=True)

        return {"vix": vix, "fear_greed_score": fg_score, "fear_greed_label": fg_label,
                "fear_greed_color": fg_color, "sectors": sectors}
    except Exception as e:
        logger.error(f"Sentiment: {e}")
        return {"vix": 0, "fear_greed_score": 50, "fear_greed_label": "Neutral",
                "fear_greed_color": "#ffa502", "sectors": []}


# ─── News Sentiment Scanner ───────────────────────────────────────────────────

BULLISH_KEYWORDS = ["surge", "soar", "rally", "beat", "upgrade", "buy", "bullish",
                    "record high", "strong buy", "outperform", "upside", "breakout",
                    "earnings beat", "revenue beat", "raised guidance", "buyback"]
BEARISH_KEYWORDS = ["plunge", "crash", "sell", "downgrade", "bearish", "miss",
                    "cut guidance", "layoffs", "recall", "investigation", "fraud",
                    "downside", "underperform", "earnings miss", "revenue miss"]

def scan_news_for_ticker(ticker, articles):
    """Return news sentiment for a ticker based on recent headlines."""
    matches = []
    for a in articles:
        title_lower = (a["title"] + " " + a.get("summary", "")).lower()
        if ticker.lower() in title_lower or ticker.lower() + " " in title_lower:
            bull = sum(1 for kw in BULLISH_KEYWORDS if kw in title_lower)
            bear = sum(1 for kw in BEARISH_KEYWORDS if kw in title_lower)
            matches.append({"title": a["title"], "source": a["source"], "bull": bull, "bear": bear})
    bull_total = sum(m["bull"] for m in matches)
    bear_total = sum(m["bear"] for m in matches)
    return {"count": len(matches), "bull": bull_total, "bear": bear_total, "articles": matches[:3]}


# ─── News Feed — Multiple Sources ─────────────────────────────────────────────

def get_news_feed():
    """Fetch news from multiple RSS sources + Yahoo Finance news API."""
    articles = []

    # RSS Feeds
    for source_name, url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:6]:
                try:
                    pub_dt = datetime(*entry.published_parsed[:6]) if hasattr(entry, "published_parsed") and entry.published_parsed else datetime.now()
                except Exception:
                    pub_dt = datetime.now()
                title = entry.get("title", "").strip()
                if len(title) > 10:
                    articles.append({
                        "source": source_name, "title": title,
                        "link": entry.get("link", "#"),
                        "summary": BeautifulSoup(entry.get("summary", ""), "html.parser").get_text()[:200],
                        "published": pub_dt,
                        "published_str": pub_dt.strftime("%b %d, %I:%M %p"),
                    })
        except Exception as e:
            logger.error(f"RSS {source_name}: {e}")

    # Yahoo Finance news API (additional articles)
    try:
        yf_url = "https://query1.finance.yahoo.com/v1/finance/trending/US"
        r = requests.get(yf_url, headers=HEADERS, timeout=8)
        trending = r.json().get("finance", {}).get("result", [{}])[0].get("quotes", [])
        for q in trending[:5]:
            sym = q.get("symbol", "")
            if sym:
                try:
                    news_url = f"https://query1.finance.yahoo.com/v1/finance/search?q={sym}&newsCount=3&quotesCount=0"
                    nr = requests.get(news_url, headers=HEADERS, timeout=5)
                    for item in nr.json().get("news", []):
                        title = item.get("title", "").strip()
                        if title and len(title) > 10:
                            pub_ts = item.get("providerPublishTime", 0)
                            pub_dt = datetime.fromtimestamp(pub_ts) if pub_ts else datetime.now()
                            articles.append({
                                "source": item.get("publisher", "Yahoo Finance"),
                                "title": title,
                                "link": item.get("link", "#"),
                                "summary": "",
                                "published": pub_dt,
                                "published_str": pub_dt.strftime("%b %d, %I:%M %p"),
                            })
                except Exception:
                    pass
    except Exception:
        pass

    # Deduplicate and sort
    seen, unique = set(), []
    for a in articles:
        key = a["title"][:60].lower()
        if key not in seen and len(a["title"]) > 10:
            seen.add(key)
            unique.append(a)

    unique.sort(key=lambda x: x["published"], reverse=True)
    return unique[:50]


# ─── Stock Browser ────────────────────────────────────────────────────────────

STOCK_UNIVERSE = [
    # S&P 500 — Information Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ACN", "AMD", "CSCO",
    "IBM", "INTC", "NOW", "INTU", "QCOM", "TXN", "AMAT", "LRCX", "ADI",
    "MU", "KLAC", "MCHP", "CDNS", "SNPS", "FTNT", "TDY", "CTSH",
    "IT", "EPAM", "GEN", "AKAM", "PTC", "GDDY", "GLW", "TEL",
    "STX", "WDC", "HPQ", "DELL", "HPE", "NTAP", "ZBRA", "TER", "MPWR",
    "ENPH", "SEDG", "KEYS", "LDOS", "SWKS", "QRVO", "FSLR", "PODD",
    "SMCI", "ANET", "PANW", "CRWD", "ZS", "DDOG", "NET", "SNOW", "PLTR",
    "ARM", "APP",
    # S&P 500 — Communication Services
    "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS",
    "CHTR", "WBD", "OMC", "LYV", "EA", "TTWO", "MTCH",
    "FOXA", "FOX",
    # S&P 500 — Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "CMG",
    "TJX", "ORLY", "AZO", "ROST", "GM", "F", "APTV", "LEN", "DHI",
    "PHM", "NVR", "TOL", "MHK", "LVS", "MGM", "WYNN", "MAR", "HLT",
    "CCL", "RCL", "NCLH", "YUM", "DRI", "QSR", "HAS", "MAT", "RL",
    "TPR", "CPRI", "VFC", "PVH", "G", "EXPE", "ABNB", "EBAY", "ETSY",
    "W", "RH", "BBY", "DG", "DLTR", "KR", "TSCO",
    # S&P 500 — Consumer Staples
    "WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "EL",
    "KMB", "GIS", "CPB", "HRL", "SJM", "MKC", "CAG", "LW", "TSN",
    "TAP", "STZ", "CHD", "CLX", "SPB", "SYY", "PFGC",
    "ADM", "BG",
    # S&P 500 — Health Care
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "ABT", "TMO", "DHR", "AMGN",
    "ISRG", "BSX", "MDT", "SYK", "EW", "ZBH", "BDX", "BAX", "HOLX",
    "DXCM", "PODD", "RMD", "IDXX", "IQV", "A", "MTD", "WAT", "RVTY",
    "BMY", "PFE", "GILD", "BIIB", "REGN", "VRTX", "MRNA", "HUM", "CVS",
    "CI", "HCA", "MOH", "CNC", "DVA", "VTRS", "ZTS", "IDXX", "ALGN",
    # S&P 500 — Financials
    "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW",
    "CB", "PGR", "TRV", "MET", "PRU", "AFL", "AIG", "ALL", "L", "LNC",
    "GL", "UNM", "RNR", "WRB", "HIG", "CINF", "AIZ", "AMP", "IVZ",
    "BEN", "TROW", "STT", "BK", "NTRS", "FIS", "GPN", "MA", "V",
    "PYPL", "COF", "SYF", "BFH", "MTB", "USB", "PNC", "TFC",
    "FITB", "HBAN", "CFG", "KEY", "RF", "ZION",
    "ALLY", "NDAQ", "ICE", "CME", "CBOE", "SPGI", "MCO", "MSCI",
    # S&P 500 — Industrials
    "GE", "HON", "RTX", "LMT", "NOC", "GD", "BA", "CAT", "DE", "EMR",
    "ETN", "ROK", "AME", "PH", "ITW", "DOV", "IR", "XYL", "FBIN",
    "SWK", "PNR", "MAS", "ALLE", "RRX", "GGG", "TT", "CARR", "OTIS",
    "WM", "RSG", "CTAS", "FAST", "GWW", "MSC", "CHRW", "EXPD", "UPS",
    "FDX", "DAL", "UAL", "AAL", "LUV", "ALK", "JBLU", "CSX", "NSC",
    "UNP", "CNI", "CP", "WAB", "ODFL", "XPO", "SAIA", "URI", "AGCO",
    "TDG", "HEI", "TXT", "AXON",
    # S&P 500 — Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY",
    "DVN", "FANG", "APA", "HAL", "BKR", "KMI", "WMB",
    "OKE", "LNG", "ET", "TRGP",
    # S&P 500 — Materials
    "LIN", "APD", "SHW", "ECL", "PPG", "EMN", "CE", "DD", "DOW", "LYB",
    "CF", "MOS", "NUE", "STLD", "RS", "ATI", "AA", "FCX", "NEM", "AEM",
    "GOLD", "WPM", "MP", "ALB", "CTVA", "FMC",
    # S&P 500 — Real Estate
    "AMT", "PLD", "CCI", "EQIX", "PSA", "DLR", "O", "VICI", "WPC",
    "SPG", "SLG", "BXP", "KIM", "REG", "AVB", "EQR", "MAA", "UDR",
    "CPT", "ESS", "HST", "PK", "SHO", "RHP", "INVH", "AMH", "CUBE",
    "EXR", "NSA", "VTR", "WELL", "HR", "HIW",
    # S&P 500 — Utilities
    "NEE", "SO", "DUK", "D", "SRE", "AEP", "EXC", "XEL", "WEC", "ES",
    "AWK", "CMS", "ETR", "LNT", "EVRG", "NI", "PNW", "AES", "NRG",
    "CEG", "PCG", "FE", "PPL", "CNP", "OGE",
    # ETFs
    "SPY", "QQQ", "DIA", "IWM", "GLD", "SLV", "USO", "TLT", "HYG",
    "XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLB", "XLRE",
    "XLU", "VTI", "VOO", "VGT", "ARKK", "SOXX",
    # Popular non-SPY / high-interest extras
    "COIN", "HOOD", "MSTR", "RIOT", "MARA", "CLSK",
    "RIVN", "LCID", "LYFT", "UBER", "ABNB", "SOFI", "NU", "UPST",
    "AFRM", "OPEN", "PATH", "AI", "SOUN", "IONQ", "QUBT", "RGTI",
    "ACHR", "JOBY", "LUNR", "RKLB",
    # S&P 400 Mid-Cap — Tech
    "FFIV", "JKHY", "MANH", "PCTY", "NCNO", "TOST", "ACAD",
    "CVLT", "SPSC", "QTWO", "CGNX", "NOVT", "MKSI",
    "VECO", "COHU", "ONTO", "FORM", "ACLS", "AMBA", "SITM", "ALGM",
    "RMBS", "POWI", "DIOD", "MTSI", "SLAB", "OSIS", "VIAV",
    "CALX", "ADTN", "EXTR", "CIEN", "LITE", "AAON", "ICHR",
    # S&P 400 Mid-Cap — Finance
    "FHN", "WTFC", "BOKF", "IBOC", "CVBF", "UMBF", "WSFS",
    "FFIN", "GBCI", "EFSC", "SFNC", "NBTB", "CTBI",
    "OFG", "HOPE", "BANR", "WAFD", "TCBK", "FIBK",
    "WBS", "PNFP", "TOWN", "RNST", "GSBC", "AROW",
    "EWBC", "IBCP", "NWBI", "EGBN", "CATY", "HAFC", "PFIS",
    "RDN", "ESNT", "NMIH", "MTG", "GNW", "PRG", "WRLD",
    # S&P 400 Mid-Cap — Healthcare
    "TECH", "NEOG", "MMSI", "ADUS", "AFYA", "CERT",
    "ENSG", "CCRN", "NTRA", "EXAS", "NVCR", "ACMR", "TMDX", "ATRC",
    "IRTC", "BEAT", "PRAX", "RARE", "KRYS",
    "ACVA", "ALNY", "BEAM", "CRSP", "EDIT", "NTLA",
    "SGMO", "FATE", "IOVA", "KYMR", "MGNX", "RCUS", "TPST", "XENE",
    # S&P 400 Mid-Cap — Consumer
    "CVNA", "AN", "KMX", "LAD", "ABG", "SAH", "PAG", "GPI", "RUSHA",
    "DRVN", "MNRO", "MUSA", "CASY", "ARKO", "PTLO", "SHAK",
    "DNUT", "FAT", "FRSH", "JACK", "TXRH", "CAKE", "BJRI", "RRGB",
    "PLNT", "XPOF", "BJ", "FIVE", "OLLI", "SFM",
    "PRCH", "ANGI", "IAC", "MTH", "TMHC", "GRBK", "CVCO",
    # S&P 400 Mid-Cap — Industrial
    "EXPO", "AAON", "AWR", "CWST", "ASTE", "GATX", "TRN",
    "GBX", "GNRC", "IESC", "KFY", "MAN", "KFRC", "KELYA", "TBI",
    "RHI", "HURN", "ICFI", "CRL", "MYRG", "DY",
    "PRIM", "WLDN", "GLDD", "GVA", "STRL", "MTRN", "ESAB", "GFF",
    "LPX", "UFPI", "IBP", "BLDR", "APOG",
    # S&P 400 Mid-Cap — Energy / Materials
    "CHRD", "SM", "MTDR", "REX",
    "HESM", "DKL", "CAPL", "HLX",
    "PUMP", "GTLS", "WTRG", "MSEX", "ARTNA",
    "SENEA", "MGEE", "OTTR", "SPOK", "CWCO",
    # Popular growth / trending
    "SHOP", "SPOT", "PINS", "SNAP", "RBLX", "U", "DKNG", "PENN",
    "MGM", "WYNN", "CHWY", "BARK", "PETS", "ZETA", "BRZE", "TASK",
    "HUBS", "BILL", "GLBE", "MNDY", "S", "TENB", "QLYS",
    "RPM", "DOCN", "GTLB", "CFLT", "MDB", "ESTC", "CLDT",
    "TTD", "MGNI", "PUBM", "APPS", "XPOF",
    # Biotech / Pharma growth
    "GERN", "NVAX", "SRPT", "BMRN", "ALKS", "PRGO",
    "JAZZ", "SUPN", "PCRX", "PAHC", "OMCL", "VEEV", "DOCS",
    "AMWL", "TDOC", "ALHC", "MDRX",
    "HSTM",
    # Russell 2000 / IWM small-caps (popular names)
    "SMCI", "FORM", "IRTC", "ACLS", "TIGR", "HIMS", "RXRX", "GKOS",
    "AXSM", "HRMY", "INVA", "SRRK", "CABA", "OLPX", "PRTA",
    "ARHS", "MNKD", "AGEN", "IMVT", "VERA", "PRLD", "ELVN",
    "KROS", "IMCR", "RVMD", "ARDX", "FLNC", "ERII", "HASI",
    "ARRY", "MAXN", "RUN", "SPWR", "SHLS", "STEM", "EVGO",
    "CHPT", "BLNK", "GOEV", "WKHS",
    "CLOV", "LMND", "ROOT", "DOMO", "BOX",
    "ASAN", "ALRM", "ARLO", "SONO",
    "EXPI", "HLNE", "STEP",
    "JACK", "SHAK", "PTLO", "DNUT", "FCPT", "RNGR", "CODI",
    "SAFE", "JBGS", "IIPR", "COLD", "STAG", "LXP",
    "GTY", "NLCP", "GOOD", "GAIN", "ARCC", "MAIN", "HTGC", "TPVG",
    "SLRC", "GBDC", "GSBD", "BXSL", "OBDC", "FSK", "PFLT",
    "OCSL", "CGBD", "FDUS", "HRZN", "TRIN", "TCPC",
    "CSWC", "PSEC", "GLAD", "MRCC", "WHF", "CCAP", "SSSS", "BCSF",
    # More popular small/mid caps
    "APPN", "CNXC", "PLTK",
    "AMSF", "NTST", "PSTL", "LAND", "PINE",
    "GENI", "LESL", "SFIX",
    "FIGS", "ONON", "CROX", "DECK", "CATO",
    "AEO", "ANF", "URBN", "TLYS", "ZUMZ", "BOOT",
    "HIMS", "ELF", "COTY",
    "PRTS", "CVNA", "VRM", "CPRT",
    # ETF extras
    "TQQQ", "SQQQ", "SPXL", "SPXS", "UVXY", "SVXY", "VIXY",
    "SOXL", "SOXS", "LABU", "LABD", "TECL", "TECS", "FNGU", "FNGD",
    "JEPI", "JEPQ", "SCHD", "DGRO", "VYM", "DVY", "HDV", "NOBL",
    "VIG", "QUAL", "MTUM", "VLUE", "SIZE", "USMV", "EFAV", "ACWI",
    "EEM", "EFA", "VEA", "VWO", "IEMG", "INDA", "EWZ", "EWJ", "EWC",
]

# Deduplicate preserving order
_seen = set()
STOCK_UNIVERSE = [t for t in STOCK_UNIVERSE if t not in _seen and not _seen.add(t)]

TICKER_SECTOR_MAP = {
    # Tech
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AVGO": "Tech", "ORCL": "Tech",
    "CRM": "Tech", "ACN": "Tech", "AMD": "Tech", "CSCO": "Tech", "IBM": "Tech",
    "INTC": "Tech", "NOW": "Tech", "INTU": "Tech", "QCOM": "Tech", "TXN": "Tech",
    "AMAT": "Tech", "LRCX": "Tech", "ADI": "Tech", "MU": "Tech", "KLAC": "Tech",
    "MCHP": "Tech", "CDNS": "Tech", "SNPS": "Tech", "FTNT": "Tech", "ANSS": "Tech",
    "TDY": "Tech", "CTSH": "Tech", "IT": "Tech", "EPAM": "Tech", "GEN": "Tech",
    "AKAM": "Tech", "JNPR": "Tech", "PTC": "Tech", "GDDY": "Tech", "GLW": "Tech",
    "TEL": "Tech", "STX": "Tech", "WDC": "Tech", "HPQ": "Tech", "DELL": "Tech",
    "HPE": "Tech", "NTAP": "Tech", "ZBRA": "Tech", "TER": "Tech", "MPWR": "Tech",
    "ENPH": "Tech", "SEDG": "Tech", "KEYS": "Tech", "LDOS": "Tech", "SWKS": "Tech",
    "QRVO": "Tech", "FSLR": "Tech", "SMCI": "Tech", "ANET": "Tech", "PANW": "Tech",
    "CRWD": "Tech", "ZS": "Tech", "DDOG": "Tech", "NET": "Tech", "SNOW": "Tech",
    "PLTR": "Tech", "ARM": "Tech", "APP": "Tech",
    "COIN": "Tech", "HOOD": "Tech", "MSTR": "Tech", "RIOT": "Tech", "MARA": "Tech",
    "CLSK": "Tech", "AI": "Tech", "SOUN": "Tech", "IONQ": "Tech", "QUBT": "Tech",
    "RGTI": "Tech", "PATH": "Tech",
    # Communication
    "GOOGL": "Tech", "GOOG": "Tech", "META": "Tech", "NFLX": "Tech",
    "DIS": "Consumer", "CMCSA": "Tech", "VZ": "Tech", "T": "Tech", "TMUS": "Tech",
    "CHTR": "Tech", "PARA": "Tech", "WBD": "Tech", "OMC": "Tech", "IPG": "Tech",
    "LYV": "Consumer", "EA": "Tech", "TTWO": "Tech", "MTCH": "Tech",
    "FOXA": "Tech", "FOX": "Tech",
    # Consumer Discretionary
    "AMZN": "Consumer", "TSLA": "Consumer", "HD": "Consumer", "MCD": "Consumer",
    "NKE": "Consumer", "SBUX": "Consumer", "LOW": "Consumer", "BKNG": "Consumer",
    "CMG": "Consumer", "TJX": "Consumer", "ORLY": "Consumer", "AZO": "Consumer",
    "ROST": "Consumer", "GM": "Consumer", "F": "Consumer", "APTV": "Consumer",
    "LEN": "Consumer", "DHI": "Consumer", "PHM": "Consumer", "NVR": "Consumer",
    "TOL": "Consumer", "MHK": "Consumer", "LVS": "Consumer", "MGM": "Consumer",
    "WYNN": "Consumer", "MAR": "Consumer", "HLT": "Consumer", "CCL": "Consumer",
    "RCL": "Consumer", "NCLH": "Consumer", "YUM": "Consumer", "DRI": "Consumer",
    "QSR": "Consumer", "HAS": "Consumer", "MAT": "Consumer", "RL": "Consumer",
    "TPR": "Consumer", "CPRI": "Consumer", "VFC": "Consumer", "PVH": "Consumer",
    "EXPE": "Consumer", "ABNB": "Consumer", "EBAY": "Consumer", "ETSY": "Consumer",
    "W": "Consumer", "RH": "Consumer", "BBY": "Consumer", "DG": "Consumer",
    "DLTR": "Consumer", "KR": "Consumer", "TSCO": "Consumer",
    "RIVN": "Consumer", "LCID": "Consumer", "LYFT": "Consumer", "UBER": "Consumer",
    "OPEN": "Consumer", "AFRM": "Finance", "UPST": "Finance",
    # Consumer Staples
    "WMT": "Consumer", "COST": "Consumer", "PG": "Consumer", "KO": "Consumer",
    "PEP": "Consumer", "PM": "Consumer", "MO": "Consumer", "MDLZ": "Consumer",
    "CL": "Consumer", "EL": "Consumer", "KMB": "Consumer", "GIS": "Consumer",
    "K": "Consumer", "CPB": "Consumer", "HRL": "Consumer", "SJM": "Consumer",
    "MKC": "Consumer", "CAG": "Consumer", "LW": "Consumer", "TSN": "Consumer",
    "TAP": "Consumer", "STZ": "Consumer", "CHD": "Consumer", "CLX": "Consumer",
    "SPB": "Consumer", "SYY": "Consumer", "ADM": "Consumer", "BG": "Consumer",
    # Healthcare
    "LLY": "Healthcare", "UNH": "Healthcare", "JNJ": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "ABT": "Healthcare", "TMO": "Healthcare", "DHR": "Healthcare",
    "AMGN": "Healthcare", "ISRG": "Healthcare", "BSX": "Healthcare", "MDT": "Healthcare",
    "SYK": "Healthcare", "EW": "Healthcare", "ZBH": "Healthcare", "BDX": "Healthcare",
    "BAX": "Healthcare", "HOLX": "Healthcare", "DXCM": "Healthcare", "PODD": "Healthcare",
    "RMD": "Healthcare", "IDXX": "Healthcare", "IQV": "Healthcare", "A": "Healthcare",
    "MTD": "Healthcare", "WAT": "Healthcare", "RVTY": "Healthcare", "BMY": "Healthcare",
    "PFE": "Healthcare", "GILD": "Healthcare", "BIIB": "Healthcare", "REGN": "Healthcare",
    "VRTX": "Healthcare", "MRNA": "Healthcare", "HUM": "Healthcare", "CVS": "Healthcare",
    "CI": "Healthcare", "HCA": "Healthcare", "MOH": "Healthcare", "CNC": "Healthcare",
    "DVA": "Healthcare", "VTRS": "Healthcare", "ZTS": "Healthcare", "ALGN": "Healthcare",
    # Finance
    "JPM": "Finance", "BAC": "Finance", "WFC": "Finance", "GS": "Finance",
    "MS": "Finance", "C": "Finance", "AXP": "Finance", "BLK": "Finance",
    "SCHW": "Finance", "CB": "Finance", "PGR": "Finance", "TRV": "Finance",
    "MET": "Finance", "PRU": "Finance", "AFL": "Finance", "AIG": "Finance",
    "ALL": "Finance", "HIG": "Finance", "CINF": "Finance", "AIZ": "Finance",
    "AMP": "Finance", "IVZ": "Finance", "BEN": "Finance", "TROW": "Finance",
    "STT": "Finance", "BK": "Finance", "NTRS": "Finance",
    "FIS": "Finance", "GPN": "Finance", "MA": "Finance", "V": "Finance",
    "PYPL": "Finance", "COF": "Finance", "SYF": "Finance", "BFH": "Finance",
    "MTB": "Finance", "USB": "Finance", "PNC": "Finance", "TFC": "Finance",
    "FITB": "Finance", "HBAN": "Finance", "CFG": "Finance", "KEY": "Finance",
    "RF": "Finance", "ZION": "Finance", "ALLY": "Finance",
    "NDAQ": "Finance", "ICE": "Finance", "CME": "Finance", "CBOE": "Finance",
    "SPGI": "Finance", "MCO": "Finance", "MSCI": "Finance", "BRK-B": "Finance",
    "SOFI": "Finance", "NU": "Finance", "L": "Finance", "LNC": "Finance",
    "GL": "Finance", "UNM": "Finance", "RNR": "Finance",
    "WRB": "Finance",
    # Industrial
    "GE": "Industrial", "HON": "Industrial", "RTX": "Industrial", "LMT": "Industrial",
    "NOC": "Industrial", "GD": "Industrial", "BA": "Industrial", "CAT": "Industrial",
    "DE": "Industrial", "EMR": "Industrial", "ETN": "Industrial", "ROK": "Industrial",
    "AME": "Industrial", "PH": "Industrial", "ITW": "Industrial", "DOV": "Industrial",
    "IR": "Industrial", "XYL": "Industrial", "SWK": "Industrial", "PNR": "Industrial",
    "MAS": "Industrial", "ALLE": "Industrial", "RRX": "Industrial", "GGG": "Industrial",
    "TT": "Industrial", "CARR": "Industrial", "OTIS": "Industrial", "WM": "Industrial",
    "RSG": "Industrial", "CTAS": "Industrial", "FAST": "Industrial", "GWW": "Industrial",
    "CHRW": "Industrial", "EXPD": "Industrial", "UPS": "Industrial", "FDX": "Industrial",
    "DAL": "Industrial", "UAL": "Industrial", "AAL": "Industrial", "LUV": "Industrial",
    "ALK": "Industrial", "CSX": "Industrial", "NSC": "Industrial", "UNP": "Industrial",
    "WAB": "Industrial", "ODFL": "Industrial", "XPO": "Industrial", "SAIA": "Industrial",
    "URI": "Industrial", "AGCO": "Industrial", "TDG": "Industrial", "HEI": "Industrial",
    "TXT": "Industrial", "AXON": "Industrial",
    "ACHR": "Industrial", "JOBY": "Industrial",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "EOG": "Energy",
    "SLB": "Energy", "MPC": "Energy", "PSX": "Energy", "VLO": "Energy",
    "OXY": "Energy", "PXD": "Energy", "DVN": "Energy",
    "FANG": "Energy", "APA": "Energy", "HAL": "Energy", "BKR": "Energy",
    "KMI": "Energy", "WMB": "Energy", "OKE": "Energy", "LNG": "Energy",
    "ET": "Energy", "TRGP": "Energy",
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials", "ECL": "Materials",
    "PPG": "Materials", "EMN": "Materials", "CE": "Materials", "DD": "Materials",
    "DOW": "Materials", "LYB": "Materials", "CF": "Materials", "MOS": "Materials",
    "NUE": "Materials", "STLD": "Materials", "RS": "Materials", "ATI": "Materials",
    "AA": "Materials", "FCX": "Materials", "NEM": "Materials", "AEM": "Materials",
    "GOLD": "Materials", "WPM": "Materials", "MP": "Materials", "ALB": "Materials",
    "CTVA": "Materials", "FMC": "Materials",
    # Real Estate
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
    "EQIX": "Real Estate", "PSA": "Real Estate", "DLR": "Real Estate",
    "O": "Real Estate", "VICI": "Real Estate", "WPC": "Real Estate",
    "SPG": "Real Estate", "SLG": "Real Estate", "BXP": "Real Estate",
    "KIM": "Real Estate", "REG": "Real Estate", "AVB": "Real Estate",
    "EQR": "Real Estate", "MAA": "Real Estate", "UDR": "Real Estate",
    "CPT": "Real Estate", "ESS": "Real Estate", "HST": "Real Estate",
    "INVH": "Real Estate", "AMH": "Real Estate", "CUBE": "Real Estate",
    "EXR": "Real Estate", "VTR": "Real Estate", "WELL": "Real Estate",
    # Utilities
    "NEE": "Utilities", "SO": "Utilities", "DUK": "Utilities", "D": "Utilities",
    "SRE": "Utilities", "AEP": "Utilities", "EXC": "Utilities", "XEL": "Utilities",
    "WEC": "Utilities", "ES": "Utilities", "AWK": "Utilities", "CMS": "Utilities",
    "ETR": "Utilities", "LNT": "Utilities", "EVRG": "Utilities", "NI": "Utilities",
    "PNW": "Utilities", "AES": "Utilities", "NRG": "Utilities", "CEG": "Utilities",
    "PCG": "Utilities", "FE": "Utilities", "PPL": "Utilities", "CNP": "Utilities",
    "OGE": "Utilities",
    # ETF
    "SPY": "ETF", "QQQ": "ETF", "DIA": "ETF", "IWM": "ETF", "GLD": "ETF",
    "SLV": "ETF", "USO": "ETF", "TLT": "ETF", "HYG": "ETF", "XLK": "ETF",
    "XLF": "ETF", "XLV": "ETF", "XLE": "ETF", "XLY": "ETF", "XLP": "ETF",
    "XLI": "ETF", "XLB": "ETF", "XLRE": "ETF", "XLU": "ETF", "VTI": "ETF",
    "VOO": "ETF", "VGT": "ETF", "ARKK": "ETF", "SOXX": "ETF",
    # Space/Other
    "LUNR": "Tech", "RKLB": "Tech",
    # Growth/Trending
    "SHOP": "Tech", "SPOT": "Tech", "PINS": "Tech", "SNAP": "Tech",
    "RBLX": "Tech", "U": "Tech", "DKNG": "Consumer", "PENN": "Consumer",
    "CHWY": "Consumer", "BARK": "Consumer", "PETQ": "Consumer",
    "ZETA": "Tech", "BRZE": "Tech", "TASK": "Tech", "HUBS": "Tech",
    "BILL": "Tech", "GLBE": "Tech", "MNDY": "Tech", "S": "Tech",
    "CYBR": "Tech", "TENB": "Tech", "QLYS": "Tech", "DOCN": "Tech",
    "GTLB": "Tech", "CFLT": "Tech", "MDB": "Tech", "ESTC": "Tech",
    "TTD": "Tech", "MGNI": "Tech", "IAS": "Tech", "PUBM": "Tech",
    "APPS": "Tech", "IRBT": "Tech", "LAZR": "Tech", "INVZ": "Tech",
    "OUST": "Tech", "VLDR": "Tech", "AUR": "Tech",
    # Biotech
    "GERN": "Healthcare", "NVAX": "Healthcare", "SRPT": "Healthcare",
    "BMRN": "Healthcare", "ALKS": "Healthcare", "HZNP": "Healthcare",
    "PRGO": "Healthcare", "JAZZ": "Healthcare", "SUPN": "Healthcare",
    "VEEV": "Healthcare", "DOCS": "Healthcare", "TDOC": "Healthcare",
    "HIMS": "Healthcare", "RXRX": "Healthcare", "AXSM": "Healthcare",
    "HRMY": "Healthcare", "ITCI": "Healthcare", "IMVT": "Healthcare",
    "VERA": "Healthcare", "RVMD": "Healthcare", "CRSP": "Healthcare",
    "BEAM": "Healthcare", "EDIT": "Healthcare", "NTLA": "Healthcare",
    "BLUE": "Healthcare", "FATE": "Healthcare", "IOVA": "Healthcare",
    "LMND": "Finance", "ROOT": "Finance", "ARCC": "Finance",
    "MAIN": "Finance", "HTGC": "Finance", "OBDC": "Finance",
    # EV / Clean Energy small caps
    "CHPT": "Tech", "BLNK": "Tech", "NKLA": "Consumer", "GOEV": "Consumer",
    "WKHS": "Consumer", "EVGO": "Tech", "RUN": "Tech", "SPWR": "Tech",
    "MAXN": "Tech", "FLNC": "Tech", "STEM": "Tech", "AMPS": "Tech",
    # Consumer small caps
    "ONON": "Consumer", "BIRK": "Consumer", "CROX": "Consumer",
    "DECK": "Consumer", "FIGS": "Consumer",
    "AEO": "Consumer", "ANF": "Consumer", "URBN": "Consumer",
    "BOOT": "Consumer", "ELF": "Consumer", "COTY": "Consumer",
    "CVNA": "Consumer", "CPRT": "Consumer",
    # REITs / BDCs
    "IIPR": "Real Estate", "COLD": "Real Estate", "STAG": "Real Estate",
    "LXP": "Real Estate", "SAFE": "Real Estate", "GOOD": "Finance",
    "GAIN": "Finance", "PSEC": "Finance", "GLAD": "Finance",
    # Leveraged ETFs
    "TQQQ": "ETF", "SQQQ": "ETF", "SPXL": "ETF", "SPXS": "ETF",
    "UVXY": "ETF", "SVXY": "ETF", "VIXY": "ETF", "SOXL": "ETF",
    "SOXS": "ETF", "LABU": "ETF", "LABD": "ETF", "TECL": "ETF",
    "TECS": "ETF", "FNGU": "ETF", "FNGD": "ETF",
    # Dividend / Factor ETFs
    "JEPI": "ETF", "JEPQ": "ETF", "SCHD": "ETF", "DGRO": "ETF",
    "VYM": "ETF", "DVY": "ETF", "HDV": "ETF", "NOBL": "ETF",
    "VIG": "ETF", "QUAL": "ETF", "MTUM": "ETF", "USMV": "ETF",
    # International ETFs
    "ACWI": "ETF", "EEM": "ETF", "EFA": "ETF", "VEA": "ETF",
    "VWO": "ETF", "IEMG": "ETF", "INDA": "ETF", "EWZ": "ETF",
    "EWJ": "ETF", "EWC": "ETF",
}


COMPANY_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AVGO": "Broadcom",
    "ORCL": "Oracle", "CRM": "Salesforce", "ACN": "Accenture", "AMD": "AMD",
    "CSCO": "Cisco", "IBM": "IBM", "INTC": "Intel", "NOW": "ServiceNow",
    "INTU": "Intuit", "QCOM": "Qualcomm", "TXN": "Texas Instruments",
    "AMAT": "Applied Materials", "LRCX": "Lam Research", "ADI": "Analog Devices",
    "MU": "Micron", "KLAC": "KLA Corp", "MCHP": "Microchip", "CDNS": "Cadence",
    "SNPS": "Synopsys", "FTNT": "Fortinet", "SMCI": "Super Micro",
    "ANET": "Arista Networks", "PANW": "Palo Alto", "CRWD": "CrowdStrike",
    "ZS": "Zscaler", "DDOG": "Datadog", "NET": "Cloudflare", "SNOW": "Snowflake",
    "PLTR": "Palantir", "ARM": "ARM Holdings", "APP": "AppLovin",
    "COIN": "Coinbase", "HOOD": "Robinhood", "MSTR": "MicroStrategy",
    "RIOT": "Riot Platforms", "MARA": "MARA Holdings", "CLSK": "CleanSpark",
    "AI": "C3.ai", "SOUN": "SoundHound", "IONQ": "IonQ", "QUBT": "Quantum Computing",
    "RGTI": "Rigetti Computing", "PATH": "UiPath",
    "GOOGL": "Alphabet", "GOOG": "Alphabet", "META": "Meta", "NFLX": "Netflix",
    "DIS": "Disney", "CMCSA": "Comcast", "VZ": "Verizon", "T": "AT&T",
    "TMUS": "T-Mobile", "CHTR": "Charter", "WBD": "Warner Bros",
    "LYV": "Live Nation", "EA": "EA Games", "TTWO": "Take-Two",
    "AMZN": "Amazon", "TSLA": "Tesla", "HD": "Home Depot", "MCD": "McDonald's",
    "NKE": "Nike", "SBUX": "Starbucks", "LOW": "Lowe's", "BKNG": "Booking",
    "CMG": "Chipotle", "TJX": "TJX", "ORLY": "O'Reilly Auto", "AZO": "AutoZone",
    "ROST": "Ross Stores", "GM": "General Motors", "F": "Ford", "APTV": "Aptiv",
    "LEN": "Lennar", "DHI": "D.R. Horton", "PHM": "PulteGroup",
    "LVS": "Las Vegas Sands", "MGM": "MGM Resorts", "WYNN": "Wynn Resorts",
    "MAR": "Marriott", "HLT": "Hilton", "CCL": "Carnival", "RCL": "Royal Caribbean",
    "NCLH": "Norwegian Cruise", "YUM": "Yum! Brands", "DRI": "Darden",
    "EXPE": "Expedia", "ABNB": "Airbnb", "EBAY": "eBay", "ETSY": "Etsy",
    "RH": "RH", "BBY": "Best Buy", "DG": "Dollar General",
    "DLTR": "Dollar Tree", "KR": "Kroger", "TSCO": "Tractor Supply",
    "RIVN": "Rivian", "LCID": "Lucid Motors", "LYFT": "Lyft", "UBER": "Uber",
    "AFRM": "Affirm", "UPST": "Upstart",
    "WMT": "Walmart", "COST": "Costco", "PG": "Procter & Gamble", "KO": "Coca-Cola",
    "PEP": "PepsiCo", "PM": "Philip Morris", "MO": "Altria", "MDLZ": "Mondelez",
    "CL": "Colgate", "KMB": "Kimberly-Clark", "GIS": "General Mills",
    "MKC": "McCormick", "TSN": "Tyson Foods", "STZ": "Constellation Brands",
    "SYY": "Sysco", "ADM": "Archer-Daniels",
    "LLY": "Eli Lilly", "UNH": "UnitedHealth", "JNJ": "J&J", "ABBV": "AbbVie",
    "MRK": "Merck", "ABT": "Abbott", "TMO": "Thermo Fisher", "DHR": "Danaher",
    "AMGN": "Amgen", "ISRG": "Intuitive Surgical", "BSX": "Boston Scientific",
    "MDT": "Medtronic", "SYK": "Stryker", "BDX": "Becton Dickinson",
    "DXCM": "DexCom", "RMD": "ResMed", "IDXX": "IDEXX Labs", "IQV": "IQVIA",
    "BMY": "Bristol-Myers", "PFE": "Pfizer", "GILD": "Gilead", "BIIB": "Biogen",
    "REGN": "Regeneron", "VRTX": "Vertex", "MRNA": "Moderna",
    "HUM": "Humana", "CVS": "CVS Health", "CI": "Cigna", "HCA": "HCA Healthcare",
    "ZTS": "Zoetis", "ALGN": "Align Technology",
    "JPM": "JPMorgan", "BAC": "Bank of America", "WFC": "Wells Fargo",
    "GS": "Goldman Sachs", "MS": "Morgan Stanley", "C": "Citigroup",
    "AXP": "American Express", "BLK": "BlackRock", "SCHW": "Schwab",
    "CB": "Chubb", "PGR": "Progressive", "MA": "Mastercard", "V": "Visa",
    "PYPL": "PayPal", "COF": "Capital One", "SYF": "Synchrony",
    "USB": "U.S. Bancorp", "PNC": "PNC Financial", "TFC": "Truist",
    "NDAQ": "Nasdaq", "ICE": "ICE", "CME": "CME Group", "SPGI": "S&P Global",
    "MCO": "Moody's", "MSCI": "MSCI", "BRK-B": "Berkshire Hathaway",
    "SOFI": "SoFi", "ALLY": "Ally Financial",
    "GE": "GE Aerospace", "HON": "Honeywell", "RTX": "RTX Corp",
    "LMT": "Lockheed Martin", "NOC": "Northrop Grumman", "GD": "General Dynamics",
    "BA": "Boeing", "CAT": "Caterpillar", "DE": "Deere", "EMR": "Emerson",
    "ETN": "Eaton", "ROK": "Rockwell Automation", "ITW": "Illinois Tool Works",
    "WM": "Waste Management", "RSG": "Republic Services", "CTAS": "Cintas",
    "FAST": "Fastenal", "GWW": "Grainger", "UPS": "UPS", "FDX": "FedEx",
    "DAL": "Delta Air Lines", "UAL": "United Airlines", "AAL": "American Airlines",
    "LUV": "Southwest Airlines", "CSX": "CSX", "NSC": "Norfolk Southern",
    "UNP": "Union Pacific", "ODFL": "Old Dominion", "URI": "United Rentals",
    "AXON": "Axon Enterprise",
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    "EOG": "EOG Resources", "SLB": "SLB", "MPC": "Marathon Petroleum",
    "PSX": "Phillips 66", "VLO": "Valero", "OXY": "Occidental",
    "DVN": "Devon Energy", "FANG": "Diamondback",
    "HAL": "Halliburton", "BKR": "Baker Hughes", "KMI": "Kinder Morgan",
    "WMB": "Williams Cos", "OKE": "ONEOK", "LNG": "Cheniere", "ET": "Energy Transfer",
    "LIN": "Linde", "APD": "Air Products", "SHW": "Sherwin-Williams",
    "ECL": "Ecolab", "PPG": "PPG Industries", "DOW": "Dow", "LYB": "LyondellBasell",
    "NUE": "Nucor", "FCX": "Freeport-McMoRan", "NEM": "Newmont", "ALB": "Albemarle",
    "AMT": "American Tower", "PLD": "Prologis", "CCI": "Crown Castle",
    "EQIX": "Equinix", "PSA": "Public Storage", "DLR": "Digital Realty",
    "O": "Realty Income", "VICI": "VICI Properties", "SPG": "Simon Property",
    "AVB": "AvalonBay", "EQR": "Equity Residential", "INVH": "Invitation Homes",
    "EXR": "Extra Space", "VTR": "Ventas", "WELL": "Welltower",
    "NEE": "NextEra Energy", "SO": "Southern Co", "DUK": "Duke Energy",
    "D": "Dominion Energy", "SRE": "Sempra", "AEP": "AEP", "EXC": "Exelon",
    "XEL": "Xcel Energy", "AWK": "American Water", "CEG": "Constellation Energy",
    "PCG": "PG&E",
    "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100 ETF", "DIA": "Dow Jones ETF",
    "IWM": "Russell 2000 ETF", "GLD": "Gold ETF", "SLV": "Silver ETF",
    "USO": "Oil ETF", "TLT": "Long-Term Treasury ETF",
    "XLK": "Tech ETF", "XLF": "Financials ETF", "XLV": "Healthcare ETF",
    "XLE": "Energy ETF", "XLY": "Consumer Disc ETF", "XLP": "Consumer Staples ETF",
    "XLI": "Industrials ETF", "XLB": "Materials ETF", "XLRE": "Real Estate ETF",
    "XLU": "Utilities ETF", "VTI": "Total Market ETF", "VOO": "S&P 500 ETF",
    "VGT": "Vanguard Tech ETF", "ARKK": "ARK Innovation ETF", "SOXX": "Semi ETF",
    "TQQQ": "3x Nasdaq ETF", "SQQQ": "3x Short Nasdaq ETF",
    "SOXL": "3x Semi ETF", "SOXS": "3x Short Semi ETF",
    "JEPI": "JPMorgan Income ETF", "JEPQ": "JPMorgan Nasdaq Inc ETF",
    "SCHD": "Schwab Dividend ETF",
    "SHOP": "Shopify", "SPOT": "Spotify", "PINS": "Pinterest", "SNAP": "Snap",
    "RBLX": "Roblox", "U": "Unity", "DKNG": "DraftKings", "PENN": "Penn Entertainment",
    "CHWY": "Chewy", "ZETA": "Zeta Global", "BRZE": "Braze", "HUBS": "HubSpot",
    "BILL": "Bill.com", "MNDY": "Monday.com", "S": "SentinelOne",
    "TENB": "Tenable", "QLYS": "Qualys", "DOCN": "DigitalOcean",
    "GTLB": "GitLab", "CFLT": "Confluent", "MDB": "MongoDB", "ESTC": "Elastic",
    "TTD": "The Trade Desk", "MGNI": "Magnite", "PUBM": "PubMatic",
    "HIMS": "Hims & Hers", "RXRX": "Recursion Pharma", "GKOS": "Glaukos",
    "AXSM": "Axsome Therapeutics", "IMVT": "Immunovant", "RVMD": "Revolution Medicines",
    "ARRY": "Array Technologies", "RUN": "Sunrun", "EVGO": "EVgo",
    "CHPT": "ChargePoint", "BLNK": "Blink Charging",
    "CLOV": "Clover Health", "LMND": "Lemonade", "ROOT": "Root Insurance",
    "DOMO": "Domo", "BOX": "Box", "ASAN": "Asana", "ALRM": "Alarm.com",
    "ARLO": "Arlo Technologies", "SONO": "Sonos",
    "ACHR": "Archer Aviation", "JOBY": "Joby Aviation",
    "LUNR": "Intuitive Machines", "RKLB": "Rocket Lab",
    "CVNA": "Carvana", "AN": "AutoNation", "KMX": "CarMax",
    "ONON": "On Running", "CROX": "Crocs", "DECK": "Deckers",
    "AEO": "American Eagle", "ANF": "Abercrombie", "URBN": "Urban Outfitters",
    "ELF": "e.l.f. Beauty",
    "ARCC": "Ares Capital", "MAIN": "Main Street Capital", "HTGC": "Hercules Capital",
    "STAG": "STAG Industrial", "IIPR": "Innovative Industrial",
    "VEEV": "Veeva Systems", "AMWL": "American Well", "TDOC": "Teladoc",
}


def get_stock_list():
    """Download stock data in chunks; return sorted by pct_change desc."""
    tickers = STOCK_UNIVERSE
    result = []
    chunk_size = 30

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            data = yf.download(
                chunk, period="5d", interval="1d",
                auto_adjust=True, progress=False, threads=False
            )
            if data.empty:
                continue

            # MultiIndex columns: ('Price', 'Ticker') — access via data['Close']['AAPL']
            for ticker in chunk:
                try:
                    close_col = data["Close"]
                    # single-ticker chunk returns Series; multi-ticker returns DataFrame
                    if hasattr(close_col, 'columns'):
                        prices = close_col[ticker].dropna()
                    else:
                        prices = close_col.dropna()
                    if len(prices) < 1:
                        continue
                    price = float(prices.iloc[-1])
                    pct = 0
                    if len(prices) >= 2:
                        prev = float(prices.iloc[-2])
                        pct = ((price - prev) / prev) * 100 if prev else 0
                    # Build sparkline: list of up to 5 closing prices (normalized 0-100)
                    vals = [float(v) for v in prices.tolist()]
                    mn, mx = min(vals), max(vals)
                    rng = mx - mn if mx != mn else 1
                    sparkline = [round((v - mn) / rng * 100, 1) for v in vals]
                    result.append({
                        "ticker": ticker,
                        "name": COMPANY_NAMES.get(ticker, ""),
                        "price": round(price, 2),
                        "pct_change": round(pct, 2),
                        "positive": pct >= 0,
                        "sector": TICKER_SECTOR_MAP.get(ticker, "Other"),
                        "sparkline": sparkline,
                    })
                except Exception as e:
                    logger.debug(f"Stock list {ticker}: {e}")
        except Exception as e:
            logger.warning(f"get_stock_list chunk {chunk[:3]}: {e}")
        time.sleep(0.4)  # avoid rate limiting across chunks

    result.sort(key=lambda x: x["pct_change"], reverse=True)
    logger.info(f"get_stock_list: loaded {len(result)} tickers")
    return result


def get_stock_detail(ticker):
    """Return full detail dict for a single ticker."""
    detail = {
        "ticker": ticker,
        "company": ticker,
        "sector": "N/A",
        "market_cap": None,
        "pe_ratio": None,
        "week52_high": None,
        "week52_low": None,
        "avg_volume": None,
        "beta": None,
        "price": None,
        "pct_change": None,
        "positive": True,
        "signals": None,
        "options": None,
        "finviz": None,
        "news": [],
        "updated_at": datetime.now().strftime("%b %d, %I:%M %p"),
    }

    # 1. yfinance info + fast_info fallback
    try:
        t = yf.Ticker(ticker)

        # fast_info is rate-limit resilient — use it first for price
        try:
            fi = t.fast_info
            detail["price"] = round(float(fi.last_price), 2) if fi.last_price else None
            detail["market_cap"] = fi.market_cap if hasattr(fi, "market_cap") else None
            detail["week52_high"] = fi.fifty_two_week_high if hasattr(fi, "fifty_two_week_high") else None
            detail["week52_low"] = fi.fifty_two_week_low if hasattr(fi, "fifty_two_week_low") else None
            prev_close = fi.previous_close if hasattr(fi, "previous_close") else None
            if detail["price"] and prev_close and prev_close > 0:
                pct = ((detail["price"] - prev_close) / prev_close) * 100
                detail["pct_change"] = round(pct, 2)
                detail["positive"] = pct >= 0
        except Exception as e:
            logger.debug(f"Stock detail fast_info for {ticker}: {e}")

        # Static name lookup as reliable fallback
        static_name = COMPANY_NAMES.get(ticker, "")
        if static_name:
            detail["company"] = static_name

        # Full info for company name + extras (may fail under rate limit)
        try:
            info = t.info
            name = info.get("shortName") or info.get("longName") or ""
            if name and name != ticker:
                detail["company"] = name
            detail["sector"] = info.get("sector") or TICKER_SECTOR_MAP.get(ticker, "N/A")
            detail["pe_ratio"] = info.get("trailingPE") or info.get("forwardPE")
            detail["avg_volume"] = info.get("averageVolume")
            detail["beta"] = info.get("beta")
            if not detail["market_cap"]:
                detail["market_cap"] = info.get("marketCap")
            if not detail["week52_high"]:
                detail["week52_high"] = info.get("fiftyTwoWeekHigh")
            if not detail["week52_low"]:
                detail["week52_low"] = info.get("fiftyTwoWeekLow")
        except Exception as e:
            logger.debug(f"Stock detail info for {ticker}: {e}")

        # Price from history as final fallback
        if not detail["price"]:
            hist = t.history(period="2d", interval="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                detail["price"] = round(price, 2)
                if len(hist) >= 2:
                    prev = float(hist["Close"].iloc[-2])
                    pct = ((price - prev) / prev) * 100 if prev else 0
                    detail["pct_change"] = round(pct, 2)
                    detail["positive"] = pct >= 0
    except Exception as e:
        logger.error(f"get_stock_detail info {ticker}: {e}")

    # 2. Technical signals from 1yr history
    try:
        t = yf.Ticker(ticker)
        hist_1y = t.history(period="1y", interval="1d")
        if not hist_1y.empty:
            detail["signals"] = get_technical_signals_from_history(hist_1y)
    except Exception as e:
        logger.error(f"get_stock_detail signals {ticker}: {e}")

    # 3. Options data
    try:
        detail["options"] = get_options_data(ticker)
    except Exception as e:
        logger.error(f"get_stock_detail options {ticker}: {e}")

    # 4. Finviz data
    try:
        detail["finviz"] = get_finviz_data(ticker)
    except Exception as e:
        logger.error(f"get_stock_detail finviz {ticker}: {e}")

    # 5. News: Yahoo Finance RSS for this ticker
    try:
        feed_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        feed = feedparser.parse(feed_url)
        news_items = []
        for entry in feed.entries[:10]:
            try:
                pub_dt = datetime(*entry.published_parsed[:6]) if hasattr(entry, "published_parsed") and entry.published_parsed else datetime.now()
            except Exception:
                pub_dt = datetime.now()
            title = entry.get("title", "").strip()
            if len(title) > 5:
                news_items.append({
                    "source": "Yahoo Finance",
                    "title": title,
                    "link": entry.get("link", "#"),
                    "published_str": pub_dt.strftime("%b %d, %I:%M %p"),
                })
        detail["news"] = news_items
    except Exception as e:
        logger.error(f"get_stock_detail news {ticker}: {e}")

    return detail


def get_stock_chart(ticker, period="1mo"):
    """Return OHLCV list for TradingView Lightweight Charts."""
    interval_map = {
        "1d":  "5m",
        "5d":  "15m",
        "1mo": "1d",
        "3mo": "1d",
        "6mo": "1d",
        "1y":  "1d",
        "5y":  "1wk",
    }
    interval = interval_map.get(period, "1d")
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval)
        if hist.empty:
            return []
        result = []
        use_date_str = interval in ("1d", "1wk")  # daily/weekly → YYYY-MM-DD string
        for ts, row in hist.iterrows():
            try:
                o = float(row["Open"]); h = float(row["High"])
                l = float(row["Low"]);  c = float(row["Close"])
                v = int(row["Volume"]) if not pd.isna(row["Volume"]) else 0
                if c <= 0 or pd.isna(c):
                    continue
                # Lightweight Charts v4: daily = 'YYYY-MM-DD', intraday = unix int
                if use_date_str:
                    t_val = pd.Timestamp(ts).strftime("%Y-%m-%d")
                else:
                    t_val = int(pd.Timestamp(ts).timestamp())
                result.append({
                    "time": t_val,
                    "open": round(o, 4), "high": round(h, 4),
                    "low":  round(l, 4), "close": round(c, 4),
                    "volume": v,
                })
            except Exception:
                pass
        # Deduplicate by time key (keep last)
        seen = {}
        for bar in result:
            seen[bar["time"]] = bar
        return sorted(seen.values(), key=lambda x: x["time"])
    except Exception as e:
        logger.error(f"get_stock_chart {ticker} {period}: {e}")
        return []
