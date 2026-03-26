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
from datetime import datetime, timedelta, timezone
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
        if rsi_s.isnull().all():
            return None
        rsi = round(float(rsi_s.iloc[-1]), 1)
        if np.isnan(rsi):
            return None
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
        if not re.match(r"^[A-Z0-9]{1,5}([.\-][A-Z]{1,2})?$", ticker.upper()):
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

def _score_option(row, current_price, kind, intel_score=50, intel_profile=None):
    """
    Wall-Street-grade option scoring powered by ALL 13+ data sources.
    Evaluates each contract on 11 dimensions:

    CONTRACT QUALITY (0-100 pts):
      1. Moneyness   — slightly OTM (2-8%) is the sweet spot for risk/reward
      2. Liquidity    — tight bid/ask spread, high volume + open interest
      3. Volume/OI    — ratio > 1 = fresh money flowing in (institutional signal)
      4. IV value     — moderate IV preferred (not overpriced, not dead)
      5. Notional     — bigger dollar flow = institutional conviction
      6. Premium      — filter out penny options and overpriced ones

    MULTI-SOURCE INTELLIGENCE (up to +55 / -35 pts):
      7. Composite intel score — from aggregated 13-source profile
      8. Analyst consensus     — Strong Buy boosts calls, downgrades boost puts
      9. Insider sentiment     — SEC filings from OpenInsider
     10. Social momentum       — StockTwits bullish/bearish ratio
     11. Fundamental quality   — revenue growth, margins, earnings trajectory
    """
    if intel_profile is None:
        intel_profile = {}

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
        if kind == "CALL":
            otm_pct = (strike - current_price) / current_price * 100
        else:
            otm_pct = (current_price - strike) / current_price * 100

        if -2 <= otm_pct <= 2:
            moneyness_score = 25
        elif 2 < otm_pct <= 5:
            moneyness_score = 30
        elif 5 < otm_pct <= 10:
            moneyness_score = 18
        elif 10 < otm_pct <= 15:
            moneyness_score = 8
        elif -5 <= otm_pct < -2:
            moneyness_score = 22
        elif otm_pct > 15 or otm_pct < -10:
            moneyness_score = 0
        else:
            moneyness_score = 5

        # ── 2. Liquidity (0-25 pts) ──────────────────────────────────────
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

        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2
            spread_pct = (ask - bid) / mid * 100 if mid > 0 else 100
            if spread_pct <= 3:
                liq_score += 7
            elif spread_pct <= 8:
                liq_score += 4
            elif spread_pct <= 15:
                liq_score += 1

        # ── 3. Volume/OI ratio (0-20 pts) ────────────────────────────────
        vol_oi = volume / oi if oi > 0 else 0
        if vol_oi >= 3.0:
            voi_score = 20
        elif vol_oi >= 1.5:
            voi_score = 15
        elif vol_oi >= 0.8:
            voi_score = 8
        elif vol_oi >= 0.3:
            voi_score = 3
        else:
            voi_score = 0

        # ── 4. IV value (0-10 pts) ───────────────────────────────────────
        iv_pct = iv * 100
        if 20 <= iv_pct <= 50:
            iv_score = 10
        elif 50 < iv_pct <= 80:
            iv_score = 6
        elif 15 <= iv_pct < 20:
            iv_score = 5
        elif iv_pct > 80:
            iv_score = 2
        else:
            iv_score = 0

        # ── 5. Notional value (0-10 pts) ─────────────────────────────────
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
        prem_pct = last / current_price * 100
        if 0.5 <= prem_pct <= 8:
            prem_score = 5
        elif 0.2 <= prem_pct < 0.5 or 8 < prem_pct <= 15:
            prem_score = 2
        else:
            prem_score = 0

        # Hard filters — disqualify bad contracts
        if last < 0.10:
            return -999
        if volume < 10:
            return -999
        if otm_pct > 20:
            return -999

        # ═══════════════════════════════════════════════════════════════════
        # MULTI-SOURCE INTELLIGENCE (13+ data sources)
        # ═══════════════════════════════════════════════════════════════════

        # ── 7. Composite intel score (0-15 pts / -10 penalty) ────────────
        intel_boost = 0
        if intel_score >= 70:
            intel_boost = 15
        elif intel_score >= 60:
            intel_boost = 10
        elif intel_score >= 50:
            intel_boost = 5
        elif intel_score <= 30:
            intel_boost = -10

        # ── 8. Analyst consensus — Yahoo + Finviz + Zacks (0-12 pts) ────
        # For CALLS: Strong Buy = boost. For PUTS: Sell = boost.
        analyst_boost = 0
        analyst_recom = intel_profile.get("analyst_recom")  # 1=Strong Buy, 5=Sell
        if analyst_recom:
            if kind == "CALL":
                if analyst_recom <= 1.5:
                    analyst_boost = 12   # Strong Buy → great for calls
                elif analyst_recom <= 2.0:
                    analyst_boost = 8
                elif analyst_recom >= 3.5:
                    analyst_boost = -8   # Sell rating → bad for calls
            else:  # PUT
                if analyst_recom >= 3.5:
                    analyst_boost = 10   # Sell rating → good for puts
                elif analyst_recom >= 3.0:
                    analyst_boost = 5
                elif analyst_recom <= 1.5:
                    analyst_boost = -6   # Strong Buy → bad for puts

        # Zacks Rank (1=Strong Buy, 5=Strong Sell) — from Zacks.com
        zacks = intel_profile.get("zacks_rank")
        if zacks:
            if kind == "CALL" and zacks <= 2:
                analyst_boost += 4
            elif kind == "PUT" and zacks >= 4:
                analyst_boost += 4
            elif kind == "CALL" and zacks >= 4:
                analyst_boost -= 3
            elif kind == "PUT" and zacks <= 2:
                analyst_boost -= 3

        # ── 9. Insider sentiment — OpenInsider SEC filings (0-8 pts) ─────
        insider_boost = 0
        insider_sent = intel_profile.get("insider_sentiment")
        if insider_sent:
            if kind == "CALL" and insider_sent == "bullish":
                insider_boost = 8    # Insiders buying → great for calls
            elif kind == "PUT" and insider_sent == "bearish":
                insider_boost = 6    # Insiders selling → good for puts
            elif kind == "CALL" and insider_sent == "bearish":
                insider_boost = -5   # Insiders selling → bad for calls
            elif kind == "PUT" and insider_sent == "bullish":
                insider_boost = -4   # Insiders buying → bad for puts

        # ── 10. Social sentiment — StockTwits (0-6 pts) ─────────────────
        social_boost = 0
        social_score = intel_profile.get("social_sentiment_score")
        if social_score is not None:
            if kind == "CALL" and social_score >= 70:
                social_boost = 6     # Social very bullish → calls
            elif kind == "CALL" and social_score >= 55:
                social_boost = 2
            elif kind == "PUT" and social_score <= 30:
                social_boost = 5     # Social very bearish → puts
            elif kind == "CALL" and social_score <= 30:
                social_boost = -4    # Social bearish → bad for calls
            elif kind == "PUT" and social_score >= 70:
                social_boost = -3    # Social bullish → bad for puts

        # ── 11. Fundamental quality — Yahoo + Finviz (0-14 pts) ──────────
        fundamental_boost = 0

        # Revenue growth (from Yahoo/Finviz)
        rev_growth = intel_profile.get("revenue_growth")
        if rev_growth is not None:
            rg = rev_growth * 100 if abs(rev_growth) < 5 else rev_growth
            if kind == "CALL":
                if rg >= 30:
                    fundamental_boost += 6
                elif rg >= 15:
                    fundamental_boost += 3
                elif rg < -5:
                    fundamental_boost -= 4
            else:  # PUT
                if rg < -5:
                    fundamental_boost += 4  # Revenue declining → put opportunity
                elif rg >= 30:
                    fundamental_boost -= 3  # Strong growth → bad for puts

        # Profit margins (from Yahoo/Finviz)
        profit_margin = intel_profile.get("profit_margin") or intel_profile.get("profit_margins")
        if profit_margin is not None:
            pm = profit_margin * 100 if abs(profit_margin) < 5 else profit_margin
            if kind == "CALL" and pm >= 25:
                fundamental_boost += 4   # Quality business
            elif kind == "CALL" and pm < 0:
                fundamental_boost -= 3   # Unprofitable

        # Upside to analyst target (Yahoo + Stockanalysis)
        upside = intel_profile.get("upside_pct")
        if upside is not None:
            if kind == "CALL" and upside >= 30:
                fundamental_boost += 4
            elif kind == "CALL" and upside >= 15:
                fundamental_boost += 2
            elif kind == "PUT" and upside <= -10:
                fundamental_boost += 3   # Below target → puts
            elif kind == "CALL" and upside <= -10:
                fundamental_boost -= 3

        # Short squeeze data (Finviz + Yahoo)
        squeeze = intel_profile.get("short_squeeze_score")
        if squeeze is not None and squeeze >= 60:
            if kind == "CALL":
                fundamental_boost += 4   # Squeeze potential → calls

        # Earnings proximity (EarningsWhispers + Yahoo)
        beat_rate = intel_profile.get("beat_rate")
        if beat_rate is not None:
            if kind == "CALL" and beat_rate >= 80:
                fundamental_boost += 3   # Consistent beater → calls
            elif kind == "PUT" and beat_rate <= 35:
                fundamental_boost += 3   # Consistent miss → puts

        # Fear & Greed contrarian (CNN)
        fg = intel_profile.get("fg_score")
        if fg is not None:
            if fg <= 20 and kind == "CALL":
                fundamental_boost += 3   # Extreme fear → contrarian calls
            elif fg >= 80 and kind == "PUT":
                fundamental_boost += 2   # Extreme greed → protective puts

        # IV Rank (Barchart + Yahoo computed)
        iv_rank = intel_profile.get("iv_rank")
        if iv_rank is not None:
            if iv_rank <= 25:
                fundamental_boost += 3   # Cheap options → great for buying
            elif iv_rank >= 80:
                fundamental_boost -= 4   # Expensive options → bad for buying

        total = (moneyness_score + liq_score + voi_score + iv_score +
                 not_score + prem_score + intel_boost + analyst_boost +
                 insider_boost + social_boost + fundamental_boost)
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

        # ── Fetch ALL data sources for smarter option scoring ──────────
        intel_score = 50  # neutral default
        intel_profile = {}
        if use_intel and get_enriched_ticker_profile is not None:
            try:
                intel_profile = get_enriched_ticker_profile(ticker)
                intel_score = intel_profile.get("intel_score", 50)
                logger.info(f"Options {ticker}: intel_score={intel_score}, "
                            f"sources={intel_profile.get('sources_hit', 0)}, "
                            f"analyst={intel_profile.get('analyst_recom', '-')}, "
                            f"insider={intel_profile.get('insider_sentiment', '-')}, "
                            f"social={intel_profile.get('social_sentiment_score', '-')}")
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

        # ── Score and rank calls (using ALL data sources) ────────────────
        if not all_calls.empty and current_price > 0:
            all_calls["_score"] = all_calls.apply(
                lambda r: _score_option(r, current_price, "CALL", intel_score, intel_profile), axis=1
            )
            best_calls = all_calls[all_calls["_score"] > 0].sort_values(
                "_score", ascending=False
            ).head(5)
            top_calls = [fmt(r, "CALL") for _, r in best_calls.iterrows()]
        else:
            top_calls = [fmt(r, "CALL") for _, r in all_calls.dropna(
                subset=["volume"]).sort_values("volume", ascending=False
            ).head(5).iterrows()]

        # ── Score and rank puts (using ALL data sources) ─────────────────
        if not all_puts.empty and current_price > 0:
            all_puts["_score"] = all_puts.apply(
                lambda r: _score_option(r, current_price, "PUT", intel_score, intel_profile), axis=1
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


# ─── Wall Street GARP Candidate Pool ─────────────────────────────────────────
# These are the tickers a 20-year Wall Street veteran keeps on their radar:
# AI/Semis, Big Tech, Cloud/SaaS, Cybersecurity, Fintech, Healthcare, Growth
ANALYST_CANDIDATES = [
    # AI & Semiconductors — the defining secular theme
    "NVDA", "AMD", "AVGO", "TSM", "MRVL", "ARM", "QCOM", "SMCI", "MU", "INTC",
    # Big Tech — cash machines
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX",
    # Cloud / SaaS / Data
    "CRM", "SNOW", "DDOG", "NET", "NOW", "SHOP",
    # Cybersecurity — essential spend
    "PANW", "CRWD",
    # Fintech / Payments
    "V", "MA", "SOFI", "HOOD", "AFRM", "SQ", "PYPL", "COIN",
    # Growth / Platforms
    "PLTR", "UBER", "APP", "RBLX", "DUOL", "ABNB",
    # Healthcare / Biotech
    "LLY", "ISRG", "DXCM",
    # Energy / Solar
    "FSLR", "ENPH",
    # Industrials / Defense
    "GE", "LMT", "RTX",
    # Bitcoin proxy
    "MSTR",
    # Wild cards from watchlist
    "TSLA", "BABA", "RIVN",
]


# ─── Top Picks — Wall Street GARP Scoring Engine ────────────────────────────

def get_top_picks(earnings_list=None):
    """
    Wall Street 1-Year GARP (Growth At Reasonable Price) scoring engine.

    Methodology (20-year veteran approach):
    ───────────────────────────────────────
    Phase 1: Build candidate pool (ANALYST_CANDIDATES + Yahoo movers + trending)
    Phase 2: Batch-pull yfinance fundamentals (revenue growth, PE, margins, etc.)
    Phase 3: Score using 10 fundamental factors (max ~107 base pts)
    Phase 4: Enrich top 20 with ALL 13 data sources for intel boost
    Phase 5: Apply intel signals (13+ factors, up to ~80 bonus pts)
    Phase 6: Normalize to 0-100, rank, return top 10

    Key difference from short-term algo:
    - Revenue & earnings growth >>> daily momentum
    - Valuation relative to growth >>> RSI/MACD
    - Analyst consensus + target upside >>> options flow
    - 52W recovery potential >>> volume surge
    - Financial health (margins, FCF) >>> technical signals
    """
    if earnings_list is None:
        earnings_list = []
    earnings_tickers = {e["ticker"]: e for e in earnings_list}

    # ═══ PHASE 1: Build wide candidate pool ═════════════════════════════════
    candidates = list(ANALYST_CANDIDATES)  # Start with curated list

    # Add Yahoo screener movers (catches breakout stocks we might miss)
    try:
        movers = get_yahoo_screener_movers()
        for t in movers:
            if t not in candidates:
                candidates.append(t)
    except Exception:
        pass

    # Add trending tickers (social + volume signals)
    trending_map = {}
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

        trending_sorted = sorted(trending_map.keys(),
                                  key=lambda t: trending_map[t]["source_count"],
                                  reverse=True)
        added = 0
        for t in trending_sorted:
            if t not in candidates:
                candidates.append(t)
                added += 1
            if added >= 15:
                break
        logger.info(f"Trending: {len(trending_map)} tickers found, {added} new added to candidates")
    except Exception as e:
        logger.debug(f"Trending injection failed: {e}")

    # Deduplicate
    candidates = list(dict.fromkeys(candidates))
    logger.info(f"GARP Engine: Scoring {len(candidates)} candidates...")

    # ═══ PHASE 2: Batch-pull fundamentals from yfinance ═════════════════════
    # Also batch download 1yr history for technicals (used as secondary signals)
    try:
        bulk = yf.download(candidates, period="1y", interval="1d",
                           auto_adjust=True, group_by="ticker", progress=False)
    except Exception as e:
        logger.error(f"Bulk download error: {e}")
        bulk = None

    # Junk filter patterns
    JUNK_PATTERNS = re.compile(
        r'^(TQQQ|SQQQ|UVXY|SPXS|SPXL|LABU|LABD|SOXL|SOXS|FNGU|FNGD|TZA|TNA|CRCD|YANG|YINN)',
        re.IGNORECASE
    )

    # ═══ PHASE 3: Score each candidate using Wall Street GARP factors ═══════
    scored = []
    for ticker in candidates:
        try:
            if JUNK_PATTERNS.match(ticker):
                continue

            # Pull yfinance fundamentals
            t = yf.Ticker(ticker)
            info = t.info or {}

            price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            if not price or price < 10:
                continue  # No penny stocks

            mktcap = info.get("marketCap", 0)
            pe_fwd = info.get("forwardPE", 0) or 0
            rev_growth = (info.get("revenueGrowth", 0) or 0) * 100
            earn_growth = (info.get("earningsGrowth", 0) or 0) * 100
            profit_margin = (info.get("profitMargins", 0) or 0) * 100
            free_cf = info.get("freeCashflow", 0) or 0
            target_mean = info.get("targetMeanPrice", 0) or 0
            target_upside = ((target_mean / price) - 1) * 100 if target_mean and price else 0
            analyst_recom = info.get("recommendationMean", 0) or 0  # 1=Strong Buy, 5=Sell
            num_analysts = info.get("numberOfAnalystOpinions", 0) or 0
            inst_hold = (info.get("heldPercentInstitutions", 0) or 0) * 100
            short_pct = (info.get("shortPercentOfFloat", 0) or 0) * 100
            beta = info.get("beta", 1) or 1
            sector = info.get("sector", "N/A")
            company = info.get("shortName", ticker)
            w52_high = info.get("fiftyTwoWeekHigh", price) or price
            w52_low = info.get("fiftyTwoWeekLow", price) or price
            from_52h = ((price / w52_high) - 1) * 100 if w52_high else 0
            debt_equity = info.get("debtToEquity", 0) or 0
            roe = (info.get("returnOnEquity", 0) or 0) * 100

            # Get 1yr history for technicals (secondary signals)
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

            # ════════════════════════════════════════════════════════════════
            # WALL STREET GARP SCORING (10 fundamental factors)
            # Max base score ~107, normalized to 0-100 after intel boost
            # ════════════════════════════════════════════════════════════════
            score = 0
            signals = []
            conviction_count = 0

            # ── F1: Revenue Growth (max 20 pts) — THE growth engine ──────
            if rev_growth >= 50:
                score += 20; signals.append(f"🚀 Revenue +{rev_growth:.0f}%"); conviction_count += 1
            elif rev_growth >= 30:
                score += 17; signals.append(f"📈 Revenue +{rev_growth:.0f}%"); conviction_count += 1
            elif rev_growth >= 20:
                score += 14; signals.append(f"📈 Revenue +{rev_growth:.0f}%")
            elif rev_growth >= 12:
                score += 10
            elif rev_growth >= 5:
                score += 5
            elif rev_growth < 0:
                score -= 5; signals.append(f"⚠ Revenue declining {rev_growth:.0f}%")

            # ── F2: Earnings Growth (max 15 pts) — bottom line power ─────
            if earn_growth >= 80:
                score += 15; signals.append(f"💰 Earnings +{earn_growth:.0f}%"); conviction_count += 1
            elif earn_growth >= 40:
                score += 12; signals.append(f"💰 Earnings +{earn_growth:.0f}%")
            elif earn_growth >= 20:
                score += 9
            elif earn_growth >= 10:
                score += 6
            elif earn_growth > 0:
                score += 3
            elif earn_growth < -20:
                score -= 5; signals.append(f"⚠ Earnings declining {earn_growth:.0f}%")

            # ── F3: Valuation vs Growth (max 15 pts) — the GARP core ────
            # Forward PE relative to revenue growth = crude PEG
            if pe_fwd > 0 and rev_growth > 0:
                peg_approx = pe_fwd / rev_growth
                if peg_approx < 0.5:
                    score += 15; signals.append(f"🎯 Cheap vs growth (PE {pe_fwd:.0f} / {rev_growth:.0f}% growth)")
                    conviction_count += 1
                elif peg_approx < 0.8:
                    score += 12; signals.append(f"🎯 Undervalued (PE {pe_fwd:.0f})")
                elif peg_approx < 1.2:
                    score += 8; signals.append(f"Fair value (PE {pe_fwd:.0f})")
                elif peg_approx < 2.0:
                    score += 4
                else:
                    score -= 3; signals.append(f"⚠ Expensive: PE {pe_fwd:.0f} for {rev_growth:.0f}% growth")
            elif pe_fwd > 100:
                score -= 8; signals.append(f"⚠ Very high PE {pe_fwd:.0f}")
            elif pe_fwd < 0:
                score -= 5

            # ── F4: Analyst Target Upside (max 12 pts) — Street consensus ─
            if target_upside >= 60 and num_analysts >= 15:
                score += 12; signals.append(f"🎯 Street sees +{target_upside:.0f}% ({num_analysts} analysts)")
                conviction_count += 1
            elif target_upside >= 40 and num_analysts >= 10:
                score += 10; signals.append(f"📊 Target +{target_upside:.0f}% ({num_analysts} analysts)")
                conviction_count += 1
            elif target_upside >= 25:
                score += 7; signals.append(f"Analyst target +{target_upside:.0f}%")
            elif target_upside >= 15:
                score += 4
            elif target_upside < 10 and num_analysts >= 10:
                score -= 2; signals.append(f"⚠ Limited upside +{target_upside:.0f}%")

            # ── F5: Analyst Rating (max 10 pts) — conviction of coverage ──
            if analyst_recom > 0:
                if analyst_recom <= 1.4 and num_analysts >= 20:
                    score += 10; signals.append(f"⭐ Strong Buy ({analyst_recom:.2f}, {num_analysts} analysts)")
                    conviction_count += 1
                elif analyst_recom <= 1.7:
                    score += 8; signals.append(f"⭐ Buy consensus ({analyst_recom:.2f})")
                elif analyst_recom <= 2.0:
                    score += 5
                elif analyst_recom >= 3.0:
                    score -= 5; signals.append(f"⚠ Weak rating ({analyst_recom:.2f})")

            # ── F6: Profit Margins (max 8 pts) — business quality ────────
            if profit_margin >= 40:
                score += 8; signals.append(f"🏆 Elite margins {profit_margin:.0f}%")
            elif profit_margin >= 25:
                score += 6; signals.append(f"Strong margins {profit_margin:.0f}%")
            elif profit_margin >= 15:
                score += 4
            elif profit_margin >= 5:
                score += 2
            elif profit_margin < 0:
                score -= 3; signals.append(f"⚠ Unprofitable ({profit_margin:.0f}%)")

            # ── F7: 52W Recovery Potential (max 10 pts) — mean reversion ──
            if from_52h < -40 and rev_growth > 15:
                score += 10; signals.append(f"💎 Deep value: {from_52h:.0f}% off high")
                conviction_count += 1
            elif from_52h < -30 and rev_growth > 10:
                score += 8; signals.append(f"📉 Recovery play: {from_52h:.0f}% off high")
            elif from_52h < -20:
                score += 5
            elif from_52h > -10:
                score -= 2  # Near highs = limited easy upside

            # ── F8: Free Cash Flow Yield (max 7 pts) — cash is king ──────
            if mktcap > 0 and free_cf > 0:
                fcf_yield = (free_cf / mktcap) * 100
                if fcf_yield >= 5:
                    score += 7; signals.append(f"💵 FCF yield {fcf_yield:.1f}%")
                elif fcf_yield >= 3:
                    score += 5
                elif fcf_yield >= 1:
                    score += 3
            elif free_cf < 0:
                score -= 3

            # ── F9: Institutional Ownership (max 5 pts) ──────────────────
            if inst_hold >= 80:
                score += 5; signals.append(f"🏛️ Institutional {inst_hold:.0f}%")
            elif inst_hold >= 60:
                score += 3
            elif inst_hold < 40:
                score -= 2

            # ── F10: Sector Tailwind (max 5 pts) — 2025-2026 themes ─────
            if sector == "Technology" and rev_growth >= 20:
                score += 5; signals.append("🌊 AI/Tech tailwind")
            elif sector == "Healthcare" and rev_growth >= 15:
                score += 4; signals.append("🌊 Healthcare tailwind")
            elif sector in ("Financial Services",) and rev_growth >= 20:
                score += 3; signals.append("🌊 Fintech tailwind")

            # ── RED FLAG PENALTIES ────────────────────────────────────────
            if debt_equity > 200:
                score -= 5; signals.append(f"⚠ High debt D/E {debt_equity:.0f}")
            if short_pct > 15:
                score -= 3
            if beta > 3:
                score -= 3; signals.append(f"⚠ High volatility (beta {beta:.1f})")

            # Get technical signals as SECONDARY confirmation (not primary)
            rsi_sig = tech["rsi_signal"] if tech else "N/A"
            macd_sig = tech["macd_signal"] if tech else "N/A"
            ma_sig = tech["ma_signal"] if tech else "N/A"
            rsi = tech["rsi"] if tech else 50

            # Technical BONUS (secondary, max ~8 pts — not the driver)
            if rsi_sig == "Oversold":
                score += 4; signals.append(f"RSI Oversold ({rsi:.0f}) — bounce setup")
            if "Bullish" in (macd_sig or ""):
                score += 2
            if "Golden" in (ma_sig or ""):
                score += 2; signals.append("Golden Cross")

            scored.append({
                "ticker": ticker,
                "company": company,
                "sector": sector,
                "price": round(price, 2),
                "pct_gain": 0,  # Will be set below
                "vol_surge": 1.0,
                "score": score,  # Will be updated after intel
                "rsi": rsi,
                "rsi_signal": rsi_sig,
                "macd_signal": macd_sig,
                "ma_signal": ma_sig,
                "sentiment": "Neutral",
                "total_calls": 0,
                "total_puts": 0,
                "cp_ratio": None,
                "has_earnings": ticker in earnings_tickers,
                "earnings_date": earnings_tickers[ticker]["date"] if ticker in earnings_tickers else None,
                "eps_estimate": earnings_tickers[ticker]["eps_estimate"] if ticker in earnings_tickers else None,
                # Fundamental data for display
                "rev_growth": round(rev_growth, 1),
                "earn_growth": round(earn_growth, 1),
                "pe_fwd": round(pe_fwd, 1),
                "profit_margin": round(profit_margin, 1),
                "target_upside": round(target_upside, 1),
                "from_52h": round(from_52h, 1),
                "_base_score": score,
                "_signals": list(signals),
                "_conviction_count": conviction_count,
            })
        except Exception as e:
            logger.debug(f"GARP score error {ticker}: {e}")

    logger.info(f"Phase 3 complete: {len(scored)} candidates scored")

    # Sort by base score, take top 20 for enrichment
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:20]

    # ═══ PHASE 4: Enrich top 20 with ALL 13 data sources ═══════════════════
    logger.info("Phase 4: Enriching top 20 with ALL data sources...")

    # Fetch news once for all tickers
    try:
        news_articles = get_news_feed()
    except Exception:
        news_articles = []

    # Fetch market-wide data once (shared across all tickers)
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

    enriched = []
    for s in top:
        ticker = s["ticker"]
        current_price = s.get("price", 0)
        score = s["_base_score"]
        signals = list(s["_signals"])
        conviction_count = s["_conviction_count"]

        # Options (with unusual activity detection)
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

        # Finviz
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

        # ═══ PHASE 5: Intel boost signals (13+ additional factors) ══════

        cp = s.get("cp_ratio")

        # I1. Options flow — institutional money talks
        if cp is not None:
            if cp < 0.4:
                score += 8; signals.append("Extreme call dominance"); conviction_count += 1
            elif cp < 0.7:
                score += 4; signals.append("Bullish options flow")
            elif cp > 2.0:
                score -= 6; signals.append("Heavy put dominance")
            elif cp > 1.5:
                score -= 3

        # I2. Unusual options sweeps — big money positioning
        unusual_call_notional = sum(u["notional_m"] for u in unusual_calls) if unusual_calls else 0
        unusual_put_notional  = sum(u["notional_m"] for u in unusual_puts)  if unusual_puts  else 0
        if unusual_calls and unusual_call_notional > unusual_put_notional:
            score += 8; conviction_count += 1
            signals.append(f"Unusual call sweep ${unusual_call_notional:.1f}M")
        elif unusual_puts and unusual_put_notional > unusual_call_notional * 1.5:
            score -= 6; signals.append(f"Unusual put sweep ${unusual_put_notional:.1f}M")

        # I3. CNN Fear & Greed — contrarian market gauge
        fg = market_intel.get("fg_score")
        if fg is not None:
            if fg <= 20:
                score += 6; signals.append(f"Extreme Fear ({fg}) — contrarian bullish")
                conviction_count += 1
            elif fg <= 35:
                score += 3
            elif fg >= 80:
                score -= 3; signals.append(f"Extreme Greed ({fg}) — caution")

        # I4. VIX level
        vix = market_intel.get("market_vix")
        if vix is not None:
            if 20 <= vix <= 30:
                score += 3; signals.append(f"Elevated VIX ({vix})")
            elif vix > 35:
                score -= 3

        # I5. IV Rank — cheap options = good for buying
        iv_rank = intel.get("iv_rank")
        if iv_rank is not None:
            if iv_rank <= 20:
                score += 4; signals.append(f"Low IV ({iv_rank}%) — cheap options")
            elif iv_rank >= 80:
                score -= 3

        # I6. Insider buying (SEC filings via OpenInsider)
        insider_sent = intel.get("insider_sentiment")
        if insider_sent == "bullish":
            score += 8; signals.append("Insider NET BUYING (SEC filings)")
            conviction_count += 1
        elif insider_sent == "bearish":
            score -= 4; signals.append("Insider NET SELLING")

        # I7. Short Squeeze potential
        sq_score = intel.get("short_squeeze_score")
        if sq_score is not None and sq_score >= 70:
            score += 6; signals.append(f"Short squeeze alert ({sq_score})")
            conviction_count += 1
        elif sq_score is not None and sq_score >= 50:
            score += 3

        # I8. Social sentiment (StockTwits)
        social = intel.get("social_sentiment_score")
        if social is not None:
            if social >= 70:
                score += 4; signals.append(f"Social bullish ({social}%)")
            elif social <= 30:
                score -= 3; signals.append(f"Social bearish ({social}%)")

        # I9. Zacks Rank
        zr = intel.get("zacks_rank")
        if zr is not None:
            if zr == 1:
                score += 8; signals.append("Zacks #1 Strong Buy"); conviction_count += 1
            elif zr == 2:
                score += 4; signals.append("Zacks #2 Buy")
            elif zr == 4:
                score -= 3
            elif zr == 5:
                score -= 6; signals.append("Zacks #5 Strong Sell")

        # I10. Macro — yield curve
        spread = market_intel.get("yield_curve_spread")
        if spread is not None and spread < 0:
            score -= 2; signals.append(f"Inverted yield curve — recession risk")
        t10y = market_intel.get("treasury_10y")
        t10y_prev = market_intel.get("treasury_10y_prev")
        if t10y and t10y_prev:
            rate_change = t10y - t10y_prev
            if rate_change > 0.05 and s.get("sector") in ("Technology", "Consumer Cyclical"):
                score -= 2

        # I11. Earnings beat rate (EarningsWhispers)
        beat_rate = intel.get("beat_rate")
        if beat_rate is not None:
            if beat_rate >= 80:
                score += 4; signals.append(f"Beats earnings {beat_rate}% of time")
                conviction_count += 1
            elif beat_rate <= 35:
                score -= 3

        # I12. Finviz insider + short data (additional to yfinance)
        if fv:
            insider = fv.get("insider_trans", "")
            if insider and insider.startswith("+"):
                score += 5; signals.append(f"Finviz insider buying ({insider})")
            short_float = fv.get("short_float")
            if short_float and short_float > 20:
                score += 4; signals.append(f"Short squeeze: {short_float}% shorted")

        # I13. News sentiment (WSJ, Reuters, CNBC, etc.)
        try:
            news_scan = scan_news_for_ticker(ticker, news_articles)
            if news_scan["count"] > 0:
                net = news_scan["bull"] - news_scan["bear"]
                if net >= 2:
                    score += 6; signals.append(f"Bullish press ({news_scan['count']} articles)")
                    conviction_count += 1
                elif net >= 1:
                    score += 3
                elif net <= -2:
                    score -= 6; signals.append(f"Negative press")
        except Exception:
            pass

        # I14. TRENDING BOOST
        trending_info = trending_map.get(ticker)
        if trending_info:
            src_count = trending_info["source_count"]
            src_names = ", ".join(trending_info["sources"])
            if src_count >= 3:
                score += 10; signals.append(f"🔥 Trending {src_count} sources ({src_names})")
                conviction_count += 1
            elif src_count >= 2:
                score += 6; signals.append(f"📈 Trending ({src_names})")
                conviction_count += 1
            elif src_count == 1:
                score += 3
            s["trending_sources"] = trending_info["sources"]
            s["trending_count"] = src_count

        # ═══ PHASE 6: Normalize & finalize ══════════════════════════════════
        score = min(max(round(score, 1), 0), 100)

        if (conviction_count >= 5 and score >= 80) or score >= 90:
            conviction = "EXTREME"
        elif (conviction_count >= 3 and score >= 60) or score >= 72:
            conviction = "HIGH"
        elif (conviction_count >= 2 and score >= 45) or score >= 55:
            conviction = "MEDIUM"
        else:
            conviction = "LOW"

        # Sentiment from options flow
        if cp is not None:
            s["sentiment"] = "Bullish" if cp < 0.7 else "Bearish" if cp > 1.2 else "Neutral"
        elif "Bullish" in (s.get("macd_signal") or ""):
            s["sentiment"] = "Bullish"
        elif "Bearish" in (s.get("macd_signal") or ""):
            s["sentiment"] = "Bearish"

        s["score"]      = score
        s["signals"]    = signals[:6]  # top 6 for display
        s["conviction"] = conviction

        # Clean up internal keys
        for k in ("_base_score", "_signals", "_conviction_count"):
            s.pop(k, None)

        enriched.append(s)

    enriched.sort(key=lambda x: x["score"], reverse=True)
    return enriched[:10]


# ─── Global Top Calls & Puts ─────────────────────────────────────────────────

def get_global_top_options():
    """
    Aggregate the best calls and puts across top watchlist + GARP candidate
    tickers, scored using ALL 13 data sources.

    Phase 1: Quick scan (use_intel=False) across 25 tickers for speed
    Phase 2: Re-score top candidates with FULL intel (all 13 sources)
    Phase 3: Rank by composite: intel-boosted score + notional flow

    Max 1 call + 1 put per ticker shown in the final top 5.
    """
    # Phase 1: Quick pre-scan to find which tickers have active options
    scan_tickers = list(WATCHLIST[:20])
    # Add top GARP candidates that aren't in WATCHLIST
    for t in ANALYST_CANDIDATES[:15]:
        if t not in scan_tickers:
            scan_tickers.append(t)

    logger.info(f"Global options: scanning {len(scan_tickers)} tickers...")

    # Quick scan without intel to find active chains
    quick_results = {}
    for ticker in scan_tickers:
        try:
            opts = get_options_data(ticker, use_intel=False)
            if opts and (opts["top_calls"] or opts["top_puts"]):
                quick_results[ticker] = opts
        except Exception as e:
            logger.debug(f"Global options quick scan {ticker}: {e}")

    # Phase 2: Re-score top 15 most active with FULL intel (all sources)
    # Sort by total volume to find the most active
    active_tickers = sorted(
        quick_results.keys(),
        key=lambda t: (quick_results[t].get("total_calls", 0) +
                       quick_results[t].get("total_puts", 0)),
        reverse=True
    )[:15]

    logger.info(f"Global options: enriching top {len(active_tickers)} with all data sources...")

    all_calls, all_puts = [], []
    for ticker in active_tickers:
        try:
            # Full intel scan with ALL 13 data sources
            opts = get_options_data(ticker, use_intel=True)
            if not opts:
                # Fall back to quick results
                opts = quick_results.get(ticker)
            if not opts:
                continue
            if opts["top_calls"]:
                all_calls.append(opts["top_calls"][0])
            if opts["top_puts"]:
                all_puts.append(opts["top_puts"][0])
        except Exception as e:
            logger.debug(f"Global options intel {ticker}: {e}")
            # Fall back to quick results
            opts = quick_results.get(ticker)
            if opts:
                if opts["top_calls"]:
                    all_calls.append(opts["top_calls"][0])
                if opts["top_puts"]:
                    all_puts.append(opts["top_puts"][0])

    # Also add remaining tickers from quick scan (use their pre-scored results)
    seen_tickers = set(active_tickers)
    for ticker, opts in quick_results.items():
        if ticker not in seen_tickers:
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
        vol_oi = min(vol / oi, 5)
        return notional * 0.5 + vol * 0.2 + vol_oi * 10000 * 0.3

    all_calls.sort(key=_rank, reverse=True)
    all_puts.sort(key=_rank, reverse=True)

    # Mag 7 tickers — ensure diversity in results
    MAG7 = {"AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA"}

    def _dedupe_diverse(options, limit=8):
        """
        Pick top options with diversity guarantee:
        - First 5 slots: best overall (any ticker)
        - Remaining slots: fill with non-Mag7 tickers to ensure 3+ non-Mag7
        """
        seen = set()
        top5 = []
        overflow = []

        for opt in options:
            if opt["ticker"] not in seen:
                seen.add(opt["ticker"])
                if len(top5) < 5:
                    top5.append(opt)
                else:
                    overflow.append(opt)

        # Count non-Mag7 in top 5
        non_mag7_count = sum(1 for o in top5 if o["ticker"] not in MAG7)

        # If we have fewer than 3 non-Mag7, add more from overflow
        result = list(top5)
        needed = max(0, 3 - non_mag7_count)
        for opt in overflow:
            if needed <= 0 and len(result) >= limit:
                break
            if opt["ticker"] not in MAG7:
                result.append(opt)
                needed -= 1
            elif len(result) < limit:
                result.append(opt)

        return result[:limit]

    return _dedupe_diverse(all_calls, 8), _dedupe_diverse(all_puts, 8)


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
                    "_sort_date": date,
                    "date": date.strftime("%a %b %d"),
                    "eps_estimate": round(info.get("forwardEps", 0) or 0, 2) or "N/A",
                    "sector": info.get("sector", "N/A"),
                    "market_cap": market_cap,
                })
        except Exception:
            continue

    # M-11: Sort by actual date object (not formatted string), then market cap
    earnings.sort(key=lambda x: (x.get("_sort_date", datetime.now()), -(x.get("market_cap") or 0)))
    # Remove internal fields from output
    for e in earnings:
        e.pop("market_cap", None)
        e.pop("_sort_date", None)
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
    import re
    ticker_pattern = re.compile(r'\b' + re.escape(ticker) + r'\b', re.IGNORECASE)
    matches = []
    for a in articles:
        title_lower = (a["title"] + " " + a.get("summary", "")).lower()
        if ticker_pattern.search(title_lower):
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
                    pub_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc) if hasattr(entry, "published_parsed") and entry.published_parsed else datetime.now(timezone.utc)
                except Exception:
                    pub_dt = datetime.now(timezone.utc)
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
                            pub_dt = datetime.fromtimestamp(pub_ts, tz=timezone.utc) if pub_ts else datetime.now(timezone.utc)
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

    # Deduplicate by link URL (more reliable than title prefix)
    seen, unique = set(), []
    for a in articles:
        key = a.get("link", a["title"]).lower().strip()
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
    # ── NASDAQ additions ──────────────────────────────────
    "AACG", "AAME", "AAOI", "AAPG", "AARD", "ABAT", "ABCL", "ABEO", "ABLV", "ABOS",
    "ABSI", "ABTC", "ABTS", "ABUS", "ABVC", "ABVE", "ABVX", "ACB", "ACCL", "ACDC",
    "ACET", "ACFN", "ACGL", "ACHC", "ACHV", "ACIC", "ACIU", "ACLX", "ACNB", "ACNT",
    "ACOG", "ACON", "ACRS", "ACRV", "ACT", "ACTG", "ACTU", "ACXP", "ADAG", "ADAM",
    "ADBE", "ADEA", "ADGM", "ADIL", "ADMA", "ADP", "ADPT", "ADSE", "ADSK", "ADTX",
    "ADUR", "ADV", "ADVB", "ADXN", "AEBI", "AEC", "AEHL", "AEHR", "AEI", "AEIS",
    "AEMD", "AENT", "AERT", "AEVA", "AEYE", "AFBI", "AFCG", "AFJK", "AFRI", "AGAE",
    "AGCC", "AGH", "AGIO", "AGMB", "AGMH", "AGNC", "AGPU", "AGRZ", "AGYS", "AHCO",
    "AHG", "AHMA", "AIDX", "AIFF", "AIFU", "AIHS", "AIIO", "AIMD", "AIOS", "AIOT",
    "AIP", "AIRE", "AIRG", "AIRJ", "AIRO", "AIRS", "AIRT", "AISP", "AIXC", "AIXI",
    "AKAN", "AKBA", "AKTS", "AKTX", "ALAB", "ALAR", "ALBT", "ALCO", "ALDF", "ALDX",
    "ALEC", "ALGS", "ALGT", "ALKT", "ALLO", "ALLR", "ALLT", "ALM", "ALMS", "ALMU",
    "ALNT", "ALOT", "ALOV", "ALOY", "ALPS", "ALRS", "ALT", "ALTI", "ALTO", "ALTS",
    "ALVO", "ALXO", "ALZN", "AMAL", "AMBR", "AMCI", "AMCX", "AMIX", "AMKR", "AMLX",
    "AMOD", "AMPG", "AMPH", "AMPL", "AMRN", "AMRX", "AMSC", "AMST", "AMTX", "AMWD",
    "ANAB", "ANDE", "ANGH", "ANGO", "ANIK", "ANIP", "ANIX", "ANL", "ANNA", "ANNX",
    "ANPA", "ANTA", "ANTX", "ANY", "AOSL", "AOUT", "APC", "APEI", "APGE", "API",
    "APLD", "APLM", "APLS", "APM", "APPF", "APRE", "APVO", "APWC", "APXT", "APYX",
    "AQB", "AQMS", "AQST", "ARAI", "ARAY", "ARBB", "ARBE", "ARBK", "ARCB", "ARCT",
    "AREC", "ARGX", "ARKR", "ARQ", "ARQQ", "ARQT", "ARTL", "ARTV", "ARVN", "ARWR",
    "ASBP", "ASLE", "ASMB", "ASML", "ASND", "ASNS", "ASO", "ASPI", "ASPS", "ASPSZ",
    "ASRT", "ASRV", "ASST", "ASTC", "ASTH", "ASTI", "ASTL", "ASUR", "ASYS", "ATAI",
    "ATAT", "ATCX", "ATEC", "ATER", "ATEX", "ATGL", "ATHE", "ATHR", "ATLC", "ATLN",
    "ATLO", "ATLX", "ATNI", "ATOM", "ATON", "ATOS", "ATPC", "ATRA", "ATRO", "ATXG",
    "ATYR", "AUBN", "AUDC", "AUGO", "AUID", "AUPH", "AUR", "AURA", "AURE", "AUTL",
    "AUUD", "AVAH", "AVAV", "AVBH", "AVBP", "AVIR", "AVO", "AVPT", "AVR", "AVT",
    "AVTX", "AVX", "AVXL", "AWRE", "AXG", "AXGN", "AXTI", "AYTU", "AZ", "AZI",
    "AZTA", "BAFN", "BAND", "BANF", "BANL", "BANX", "BAOS", "BATRA", "BATRK", "BBCP",
    "BBGI", "BBIO", "BBLG", "BBNX", "BBOT", "BBSI", "BCAB", "BCAL", "BCAX", "BCBP",
    "BCDA", "BCG", "BCIC", "BCML", "BCPC", "BCRX", "BCTX", "BCYC", "BDCI", "BDMD",
    "BDRX", "BDSX", "BDTX", "BEEM", "BEEP", "BELFA", "BELFB", "BENF", "BETR", "BFC",
    "BFRG", "BFRI", "BFST", "BGC", "BGIN", "BGL", "BGLC", "BGM", "BGMS", "BHF",
    "BHRB", "BHST", "BIAF", "BIDU", "BILI", "BIOA", "BIOX", "BIRD", "BITF", "BIVI",
    "BIYA", "BJDX", "BKYI", "BL", "BLBD", "BLDP", "BLFS", "BLFY", "BLIN", "BLIV",
    "BLKB", "BLLN", "BLMN", "BLNE", "BLRX", "BLTE", "BLZE", "BMBL", "BMEA", "BMGL",
    "BMHL", "BMM", "BMR", "BMRA", "BMRC", "BNAI", "BNBX", "BNC", "BNGO", "BNKK",
    "BNR", "BNRG", "BNTC", "BNTX", "BNZI", "BODI", "BOF", "BOLD", "BOLT", "BON",
    "BOOM", "BOSC", "BOTJ", "BOXL", "BPOP", "BPRN", "BRAG", "BRAI", "BRBI", "BRCB",
    "BRFH", "BRID", "BRKR", "BRLS", "BRLT", "BRNS", "BRR", "BRTX", "BSBK", "BSET",
    "BSRR", "BSVN", "BSY", "BTAI", "BTBD", "BTBT", "BTCS", "BTCT", "BTDR", "BTM",
    "BTMD", "BTOC", "BTOG", "BTQ", "BTSG", "BTTC", "BULL", "BUSE", "BUUU", "BVC",
    "BVFL", "BVS", "BWAY", "BWB", "BWEN", "BWFG", "BWIN", "BWMN", "BYAH", "BYFC",
    "BYND", "BYRN", "BYSI", "BZ", "BZAI", "BZFD", "BZUN", "CAAS", "CABR", "CAC",
    "CACC", "CADL", "CAEP", "CAI", "CALC", "CALM", "CAMP", "CAMT", "CAN", "CAPR",
    "CAPS", "CAPT", "CAR", "CARE", "CARG", "CARL", "CART", "CASH", "CASS", "CAST",
    "CBAT", "CBC", "CBFV", "CBIO", "CBK", "CBLL", "CBNK", "CBRL", "CBSH", "CBUS",
    "CCB", "CCBG", "CCC", "CCCC", "CCD", "CCEC", "CCEP", "CCG", "CCHH", "CCIX",
    "CCLD", "CCNE", "CCOI", "CCSI", "CCTG", "CCXI", "CD", "CDIO", "CDLX", "CDNA",
    "CDNL", "CDRO", "CDT", "CDTG", "CDW", "CDXS", "CDZI", "CDZIP", "CECO", "CELC",
    "CELH", "CELU", "CELZ", "CENN", "CENT", "CENTA", "CENX", "CEPF", "CEPO", "CEPS",
    "CEPT", "CEPV", "CERS", "CETX", "CETY", "CEVA", "CFBK", "CFFI", "CFFN", "CG",
    "CGC", "CGCT", "CGEM", "CGEN", "CGNT", "CGO", "CGON", "CGTL", "CGTX", "CHA",
    "CHAI", "CHCI", "CHCO", "CHDN", "CHEF", "CHI", "CHKP", "CHMG", "CHNR", "CHR",
    "CHRS", "CHSN", "CHW", "CHY", "CHYM", "CIFR", "CIGI", "CIGL", "CIIT", "CING",
    "CISO", "CISS", "CIVB", "CJMB", "CLAR", "CLBK", "CLBT", "CLDX", "CLFD", "CLGN",
    "CLIK", "CLIR", "CLLS", "CLMB", "CLMT", "CLNE", "CLNN", "CLPS", "CLPT", "CLRB",
    "CLRO", "CLST", "CLWT", "CLYM", "CMBM", "CMCO", "CMCT", "CMII", "CMMB", "CMND",
    "CMPR", "CMPS", "CMPX", "CMRC", "CMTL", "CMTV", "CNCK", "CNDT", "CNET", "CNEY",
    "CNOB", "CNSP", "CNTA", "CNTB", "CNTN", "CNTX", "CNTY", "CNVS", "CNXN", "COCH",
    "COCO", "COCP", "CODA", "CODX", "COEP", "COFS", "COGT", "COKE", "COLB", "COLL",
    "COLM", "COO", "COOT", "CORT", "CORZ", "CORZZ", "COSM", "COYA", "CPBI", "CPHC",
    "CPIX", "CPOP", "CPRX", "CPSH", "CPSS", "CPZ", "CRAI", "CRBP", "CRBU", "CRCT",
    "CRDF", "CRDL", "CRDO", "CRE", "CREG", "CRESY", "CREX", "CRGO", "CRIS", "CRMD",
    "CRML", "CRMT", "CRNC", "CRNT", "CRNX", "CRON", "CRSR", "CRTO", "CRUS", "CRVL",
    "CRVO", "CRVS", "CRWS", "CRWV", "CSAI", "CSBR", "CSGP", "CSGS", "CSIQ", "CSPI",
    "CSQ", "CSTE", "CSTL", "CTKB", "CTLP", "CTMX", "CTNM", "CTNT", "CTOR", "CTRM",
    "CTRN", "CTSO", "CTW", "CTXR", "CUB", "CUE", "CULP", "CUPR", "CURI", "CURR",
    "CURX", "CV", "CVGI", "CVKD", "CVRX", "CVV", "CWBC", "CWD", "CXAI", "CXDO",
    "CYCN", "CYCU", "CYN", "CYPH", "CYRX", "CYTK", "CZFS", "CZNC", "CZR", "CZWI",
    "DAIC", "DAIO", "DAKT", "DARE", "DASH", "DAVE", "DAWN", "DBGI", "DBVT", "DBX",
    "DCBO", "DCGO", "DCOM", "DCOY", "DCTH", "DCX", "DDI", "DEFT", "DERM", "DEVS",
    "DFDV", "DFLI", "DFNS", "DFSC", "DFTX", "DGICA", "DGICB", "DGII", "DGNX", "DGXX",
    "DH", "DHC", "DHIL", "DIBS", "DJCO", "DJT", "DKI", "DLHC", "DLO", "DLPN",
    "DLTH", "DLXY", "DMAC", "DMRA", "DMRC", "DNLI", "DNMX", "DNTH", "DOCU", "DOGZ",
    "DOMH", "DOO", "DORM", "DOX", "DOYU", "DPRO", "DPZ", "DRCT", "DRH", "DRIO",
    "DRMA", "DRS", "DRTS", "DRUG", "DSGN", "DSGR", "DSGX", "DSP", "DSWL", "DSY",
    "DTCK", "DTCX", "DTI", "DTIL", "DTSS", "DTST", "DUO", "DUOL", "DUOT", "DVLT",
    "DWSN", "DWTX", "DXLG", "DXPE", "DXR", "DXST", "DYAI", "DYN", "DYOR", "EBC",
    "EBMT", "EBON", "ECBK", "ECOR", "ECPG", "ECX", "EDAP", "EDBL", "EDHL", "EDRY",
    "EDSA", "EDTK", "EDUC", "EEFT", "EEIQ", "EFOI", "EFSI", "EGAN", "EH", "EHGO",
    "EHLD", "EHTH", "EIKN", "EJH", "EKSO", "ELAB", "ELBM", "ELDN", "ELE", "ELOG",
    "ELSE", "ELTK", "ELTX", "ELUT", "ELVA", "ELVR", "ELWT", "EM", "EMAT", "EMBC",
    "EML", "EMPD", "ENGN", "ENGS", "ENLT", "ENLV", "ENSC", "ENTA", "ENTG", "ENTX",
    "ENVB", "ENVX", "EOLS", "EOSE", "EPRX", "EPSM", "EPSN", "EQ", "EQPT", "ERAS",
    "ERIC", "ERIE", "ERNA", "ESCA", "ESEA", "ESLA", "ESLT", "ESOA", "ESPR", "ESQ",
    "ESTA", "ETHB", "ETHM", "ETON", "ETOR", "ETS", "EU", "EUDA", "EVAX", "EVCM",
    "EVER", "EVGN", "EVLV", "EVO", "EVTV", "EWCZ", "EWTX", "EXE", "EXEL", "EXFY",
    "EXLS", "EXOZ", "EYE", "EYPT", "EZGO", "EZRA", "FA", "FAMI", "FARM", "FATN",
    "FBGL", "FBIO", "FBIZ", "FBLA", "FBLG", "FBNC", "FBRX", "FBYD", "FCAP", "FCBC",
    "FCCO", "FCEL", "FCFS", "FCHL", "FCNCA", "FCNCP", "FCUV", "FDBC", "FDMT", "FDSB",
    "FEAM", "FEBO", "FEED", "FEIM", "FELE", "FEMY", "FENC", "FER", "FFAI", "FFBC",
    "FFIC", "FGBI", "FGI", "FGL", "FGMC", "FGNX", "FHB", "FHTX", "FIEE", "FIGR",
    "FIP", "FISI", "FISV", "FITBI", "FIVN", "FIZZ", "FKWL", "FLD", "FLEX", "FLGT",
    "FLL", "FLNA", "FLNT", "FLUX", "FLWS", "FLX", "FLXS", "FLYE", "FMAO", "FMBH",
    "FMFC", "FMNB", "FMST", "FNGR", "FNKO", "FNLC", "FNUC", "FNWB", "FNWD", "FOFO",
    "FOLD", "FONR", "FORA", "FORR", "FORTY", "FOSL", "FOXF", "FOXX", "FRAF", "FRBA",
    "FRD", "FRGT", "FRHC", "FRME", "FRMEP", "FRMI", "FRMM", "FROG", "FRPH", "FRPT",
    "FRST", "FRSX", "FSBC", "FSEA", "FSLY", "FSTR", "FSUN", "FSV", "FTAI", "FTCI",
    "FTDR", "FTEK", "FTFT", "FTHM", "FTLF", "FTRE", "FTRK", "FUFU", "FULC", "FULT",
    "FUNC", "FUND", "FUSB", "FUSE", "FUTU", "FVCB", "FWDI", "FWONA", "FWONK", "FWRD",
    "FWRG", "FXNC", "GABC", "GAIA", "GALT", "GAMB", "GAME", "GANX", "GASS", "GAUZ",
    "GBFH", "GBLI", "GCBC", "GCL", "GCMG", "GCT", "GCTK", "GDC", "GDEN", "GDEV",
    "GDHG", "GDRX", "GDS", "GDTC", "GDYN", "GECC", "GEG", "GEHC", "GELS", "GENB",
    "GENK", "GEVO", "GFAI", "GFS", "GGAL", "GGR", "GGRP", "GH", "GHRS", "GIBO",
    "GIFT", "GIG", "GIGM", "GIII", "GILT", "GIPR", "GITS", "GIX", "GLBS", "GLE",
    "GLIBA", "GLIBK", "GLMD", "GLNG", "GLOO", "GLPG", "GLPI", "GLRE", "GLSI", "GLUE",
    "GLXG", "GLXY", "GMAB", "GMEX", "GMHS", "GMM", "GNLN", "GNLX", "GNPX", "GNSS",
    "GNTA", "GNTX", "GO", "GOAI", "GOCO", "GOGO", "GOSS", "GOVX", "GP", "GPCR",
    "GPRE", "GPRO", "GRAB", "GRAL", "GRAN", "GRCE", "GRDX", "GREE", "GRFS", "GRI",
    "GRML", "GRNQ", "GRPN", "GRRR", "GRVY", "GRWG", "GSAT", "GSHD", "GSIT", "GSM",
    "GSUN", "GT", "GTBP", "GTEC", "GTEN", "GTIM", "GTM", "GTX", "GURE", "GUTS",
    "GV", "GVH", "GWAV", "GWRS", "GXAI", "GYRE", "GYRO", "HAIN", "HALO", "HAO",
    "HBCP", "HBIO", "HBNB", "HBNC", "HBT", "HCAI", "HCAT", "HCHL", "HCKT", "HCM",
    "HCSG", "HCTI", "HCWB", "HDL", "HDSN", "HELE", "HELP", "HEPS", "HERE", "HERZ",
    "HFBL", "HFFG", "HFWA", "HGBL", "HHS", "HIFS", "HIHO", "HIMX", "HIND", "HIT",
    "HITI", "HIVE", "HKIT", "HKPD", "HLIT", "HLMN", "HLP", "HMR", "HNNA", "HNRG",
    "HNST", "HNVR", "HOFT", "HOLO", "HOTH", "HOUR", "HOVNP", "HOVR", "HOWL", "HPAI",
    "HPK", "HQ", "HQI", "HQY", "HRTX", "HSAI", "HSCS", "HSDT", "HSIC", "HTBK",
    "HTCO", "HTCR", "HTFL", "HTHT", "HTLD", "HTLM", "HTO", "HTOO", "HTZ", "HUBC",
    "HUBG", "HUDI", "HUHU", "HUIZ", "HUMA", "HURA", "HURC", "HUT", "HVII", "HWBK",
    "HWC", "HWH", "HWKN", "HXHX", "HYFM", "HYFT", "HYMC", "HYNE", "HYPD", "HYPR",
    "IART", "IBEX", "IBG", "IBIO", "IBKR", "IBRX", "ICCC", "ICCM", "ICG", "ICLR",
    "ICMB", "ICON", "ICU", "ICUI", "IDAI", "IDCC", "IDN", "IDYA", "IEP", "IFBD",
    "IFRX", "IGIC", "IHRT", "III", "IIIV", "IINN", "IKT", "ILAG", "ILMN", "ILPT",
    "IMA", "IMCC", "IMDX", "IMKTA", "IMMP", "IMMR", "IMMX", "IMNM", "IMNN", "IMOS",
    "IMPP", "IMRN", "IMRX", "IMSR", "IMTE", "IMTX", "IMUX", "IMXI", "INAB", "INBK",
    "INBS", "INBX", "INCR", "INCY", "INDB", "INDI", "INDP", "INDV", "INEO", "INGN",
    "INHD", "INKT", "INLF", "INM", "INMB", "INMD", "INNV", "INO", "INOD", "INSE",
    "INSG", "INSM", "INTA", "INTG", "INTJ", "INTR", "INTS", "INTZ", "INV", "INVE",
    "INVZ", "IOBT", "IONR", "IONS", "IOSP", "IOTR", "IPAR", "IPDN", "IPGP", "IPHA",
    "IPM", "IPSC", "IPST", "IPW", "IPWR", "IPX", "IQ", "IQST", "IRD", "IRDM",
    "IREN", "IRIX", "IRMD", "IRON", "IRWD", "ISBA", "ISPC", "ISPR", "ISSC", "ISTR",
    "ITIC", "ITOC", "ITRI", "ITRM", "ITRN", "IVA", "IVDA", "IVF", "IVVD", "IXHL",
    "IZEA", "IZM", "JAGX", "JAKK", "JANX", "JBDI", "JBHT", "JBIO", "JBSS", "JCAP",
    "JCSE", "JCTC", "JD", "JDZG", "JEM", "JF", "JFB", "JFBR", "JFIN", "JFU",
    "JG", "JJSF", "JL", "JLHL", "JMSB", "JOUT", "JOYY", "JRSH", "JRVR", "JSPR",
    "JTAI", "JUNS", "JVA", "JWEL", "JXG", "JYD", "JYNT", "JZ", "JZXN", "KALA",
    "KALU", "KALV", "KARO", "KBON", "KBSX", "KC", "KDK", "KDP", "KE", "KELYB",
    "KEQU", "KFFB", "KG", "KGEI", "KHC", "KIDS", "KIDZ", "KINS", "KITT", "KLIC",
    "KLRS", "KLTR", "KLXE", "KMDA", "KMRK", "KMTS", "KNDI", "KNSA", "KOD", "KOPN",
    "KOSS", "KPLT", "KPRX", "KPTI", "KRKR", "KRMD", "KRNT", "KRNY", "KRRO", "KRT",
    "KRUS", "KSCP", "KSPI", "KTCC", "KTOS", "KTTA", "KURA", "KUST", "KVHI", "KWM",
    "KXIN", "KYIV", "KYNB", "KYTX", "KZIA", "KZR", "LAB", "LAES", "LAKE", "LAMR",
    "LARK", "LASE", "LASR", "LAUR", "LBGJ", "LBRDA", "LBRDK", "LBRX", "LBTYA", "LBTYB",
    "LBTYK", "LCFY", "LCNB", "LCUT", "LE", "LECO", "LEDS", "LEE", "LEGH", "LEGN",
    "LENZ", "LEXX", "LFCR", "LFMD", "LFS", "LFST", "LFUS", "LFVN", "LFWD", "LGCB",
    "LGCL", "LGHL", "LGIH", "LGN", "LGND", "LGO", "LGVN", "LHAI", "LI", "LICN",
    "LIDR", "LIEN", "LIF", "LIFE", "LILA", "LILAK", "LIMN", "LINC", "LIND", "LINE",
    "LINK", "LIQT", "LITS", "LIVE", "LIVN", "LIXT", "LKFN", "LKQ", "LLYVA", "LLYVK",
    "LMAT", "LMB", "LMFA", "LMNR", "LMRI", "LNAI", "LNKB", "LNKS", "LNSR", "LNTH",
    "LNZA", "LOAN", "LOBO", "LOCO", "LOGI", "LONA", "LOOP", "LOPE", "LOT", "LOVE",
    "LPCN", "LPLA", "LPRO", "LPSN", "LPTH", "LQDA", "LQDT", "LRE", "LRHC", "LRMR",
    "LSAK", "LSBK", "LSCC", "LSE", "LSH", "LSTA", "LSTR", "LTBR", "LTRN", "LTRX",
    "LUCD", "LUCY", "LULU", "LUNG", "LVLU", "LVO", "LWAY", "LWLG", "LX", "LXEH",
    "LXEO", "LXRX", "LYEL", "LYTS", "LZ", "LZMH", "MAAS", "MAMA", "MAMO", "MAPS",
    "MASI", "MASK", "MASS", "MATH", "MAYS", "MAZE", "MB", "MBAI", "MBBC", "MBIN",
    "MBIO", "MBLY", "MBOT", "MBRX", "MBUU", "MBWM", "MBX", "MCBS", "MCFT", "MCHB",
    "MCHX", "MCRB", "MCRI", "MCW", "MDAI", "MDBH", "MDCX", "MDGL", "MDIA", "MDLN",
    "MDRR", "MDWD", "MDXG", "MDXH", "MEDP", "MEGL", "MEHA", "MELI", "MENS", "MEOH",
    "MERC", "MESO", "METC", "METCB", "MFI", "MFIC", "MFIN", "MGIH", "MGN", "MGPI",
    "MGRC", "MGRT", "MGRX", "MGTX", "MGX", "MGYR", "MIDD", "MIGI", "MIMI", "MIND",
    "MIRA", "MIRM", "MIST", "MITK", "MKTX", "MKZR", "MLAB", "MLCI", "MLCO", "MLEC",
    "MLGO", "MLKN", "MLTX", "MLYS", "MMED", "MMLP", "MMYT", "MNDO", "MNDR", "MNOV",
    "MNPR", "MNSB", "MNSBP", "MNST", "MNTK", "MNTS", "MNY", "MOB", "MOBX", "MODD",
    "MOLN", "MOMO", "MORN", "MOVE", "MPAA", "MPB", "MPLT", "MQ", "MRAM", "MRBK",
    "MRCY", "MRDN", "MREO", "MRKR", "MRLN", "MRM", "MRNO", "MRTN", "MRVI", "MRVL",
    "MRX", "MSAI", "MSBI", "MSGM", "MSGY", "MSLE", "MSS", "MSW", "MTC", "MTEK",
    "MTEN", "MTEX", "MTLS", "MTRX", "MTVA", "MVBF", "MVIS", "MVST", "MWH", "MWYN",
    "MXCT", "MXL", "MYGN", "MYPS", "MYSE", "MYSZ", "MZTI", "NA", "NAAS", "NAGE",
    "NAII", "NAKA", "NAMI", "NAMM", "NAMS", "NATH", "NATR", "NAUT", "NAVI", "NAVN",
    "NB", "NBBK", "NBIS", "NBIX", "NBN", "NBP", "NBTX", "NCEL", "NCI", "NCMI",
    "NCNA", "NCPL", "NCRA", "NCSM", "NCT", "NCTY", "NDLS", "NDRA", "NDSN", "NECB",
    "NEGG", "NEO", "NEON", "NEOV", "NEPH", "NERV", "NESR", "NEUP", "NEWT", "NEXM",
    "NEXN", "NEXT", "NFBK", "NFE", "NGEN", "NGNE", "NHIC", "NHTC", "NICE", "NIPG",
    "NIU", "NIVF", "NIXX", "NKLR", "NKSH", "NKTR", "NKTX", "NMFC", "NMRA", "NMRK",
    "NMTC", "NN", "NNBR", "NNDM", "NNE", "NNNN", "NNOX", "NODK", "NOEM", "NOMA",
    "NOTV", "NPCE", "NPT", "NRC", "NRDS", "NRIM", "NRIX", "NRSN", "NRXP", "NSIT",
    "NSPR", "NSSC", "NSTS", "NSYS", "NTCL", "NTCT", "NTES", "NTGR", "NTHI", "NTIC",
    "NTNX", "NTRB", "NTRP", "NTSK", "NTWK", "NUAI", "NUCL", "NUTX", "NUVL", "NUWE",
    "NVA", "NVCT", "NVEC", "NVMI", "NVNI", "NVNO", "NVTS", "NVVE", "NVX", "NWE",
    "NWFL", "NWGL", "NWL", "NWPX", "NWS", "NWSA", "NWTG", "NXGL", "NXL", "NXPI",
    "NXPL", "NXST", "NXT", "NXTC", "NXTS", "NXTT", "NXXT", "NYAX", "NYXH", "OABI",
    "OBAI", "OBIO", "OBT", "OCC", "OCCI", "OCFC", "OCG", "OCGN", "OCS", "OCUL",
    "ODD", "ODYS", "OESX", "OFAL", "OFIX", "OFLX", "OFS", "OGI", "OIO", "OKTA",
    "OKUR", "OKYO", "OLB", "OLED", "OLMA", "OLOX", "OM", "OMAB", "OMDA", "OMER",
    "OMEX", "OMH", "OMSE", "ON", "ONB", "ONC", "ONCO", "ONCY", "ONDS", "ONEG",
    "ONFO", "ONMD", "OPAL", "OPBK", "OPCH", "OPK", "OPRA", "OPRT", "OPRX", "OPTX",
    "OPXS", "ORBS", "ORGN", "ORGO", "ORIC", "ORIO", "ORIQ", "ORIS", "ORKA", "ORKT",
    "ORMP", "ORRF", "OS", "OSBC", "OSPN", "OSRH", "OSS", "OSUR", "OSW", "OTEX",
    "OTLK", "OTLY", "OUST", "OVBC", "OVID", "OVLY", "OWLS", "OXBR", "OXLC", "OXSQ",
    "OZK", "PACB", "PAL", "PALI", "PAMT", "PANL", "PARK", "PASG", "PATK", "PAVM",
    "PAVS", "PAX", "PAYO", "PAYP", "PAYS", "PAYX", "PBFS", "PBHC", "PBM", "PBYI",
    "PCAR", "PCB", "PCLA", "PCSA", "PCSC", "PCT", "PCVX", "PCYO", "PDC", "PDD",
    "PDEX", "PDFS", "PDLB", "PDSB", "PDYN", "PEBK", "PEBO", "PECO", "PEGA", "PENG",
    "PEPG", "PERI", "PESI", "PETZ", "PFAI", "PFG", "PFSA", "PFX", "PGC", "PGEN",
    "PGNY", "PGY", "PHAR", "PHAT", "PHIO", "PHOE", "PHUN", "PHVS", "PI", "PICS",
    "PIII", "PKBK", "PKOH", "PLAB", "PLAY", "PLBC", "PLBL", "PLBY", "PLCE", "PLMR",
    "PLPC", "PLRX", "PLRZ", "PLSE", "PLSM", "PLUG", "PLUR", "PLUS", "PLUT", "PLXS",
    "PLYX", "PMAX", "PMCB", "PMEC", "PMN", "PMTS", "PMVP", "PN", "PNBK", "PNRG",
    "PNTG", "POCI", "PODC", "POET", "POLA", "POM", "PONY", "POOL", "POWL", "PPBT",
    "PPC", "PPCB", "PPHC", "PPIH", "PPSI", "PPTA", "PRAA", "PRCT", "PRDO", "PRE",
    "PRFX", "PRGS", "PRHI", "PRME", "PROF", "PROK", "PROP", "PROV", "PRPL", "PRPO",
    "PRQR", "PRSO", "PRTC", "PRTH", "PRVA", "PRZO", "PSHG", "PSIG", "PSIX", "PSKY",
    "PSMT", "PSNL", "PSNY", "PSTV", "PTCT", "PTEN", "PTGX", "PTLE", "PTON", "PTRN",
    "PULM", "PURR", "PVLA", "PWP", "PXS", "PYPD", "PYXS", "PZZA", "QCLS", "QCRH",
    "QDEL", "QFIN", "QH", "QMCO", "QNCX", "QNRX", "QNST", "QNTM", "QQQX", "QRHC",
    "QS", "QSI", "QTI", "QTRX", "QTTB", "QUCY", "QUIK", "QURE", "QVCGA", "RADX",
    "RAIL", "RAIN", "RAND", "RANI", "RAPP", "RAVE", "RAY", "RAYA", "RBB", "RBBN",
    "RBCAA", "RBKB", "RBNE", "RCAT", "RCEL", "RCKT", "RCKY", "RCMT", "RCON", "RCT",
    "RDCM", "RDGT", "RDHL", "RDI", "RDIB", "RDNT", "RDVT", "RDWR", "RDZN", "REAL",
    "REAX", "REBN", "RECT", "REE", "REFI", "REFR", "REKR", "RELL", "RELY", "RENT",
    "RENX", "REPL", "RETO", "REVB", "REYN", "RFIL", "RGC", "RGCO", "RGEN", "RGLD",
    "RGNX", "RGP", "RGS", "RICK", "RIGL", "RILY", "RIME", "RITR", "RJET", "RKDA",
    "RLAY", "RLMD", "RLYB", "RMBI", "RMCF", "RMCO", "RMNI", "RMR", "RMSG", "RMTI",
    "RNA", "RNAC", "RNAZ", "RNTX", "RNW", "RNXT", "ROAD", "ROC", "ROCK", "ROIV",
    "ROKU", "ROMA", "ROP", "RPAY", "RPD", "RPGL", "RPID", "RPRX", "RR", "RRBI",
    "RRR", "RSSS", "RSVR", "RUBI", "RUM", "RUSHB", "RVPH", "RVSB", "RVSN", "RVYL",
    "RWAY", "RXST", "RYAAY", "RYET", "RYM", "RYOJ", "RYTM", "RZLT", "RZLV", "SABR",
    "SABS", "SAFT", "SAFX", "SAGT", "SAIC", "SAIH", "SAIL", "SAMG", "SANA", "SANG",
    "SANM", "SATL", "SATS", "SBAC", "SBC", "SBCF", "SBET", "SBFG", "SBFM", "SBGI",
    "SBLK", "SBLX", "SBRA", "SCAG", "SCHL", "SCKT", "SCLX", "SCNI", "SCNX", "SCOR",
    "SCPQ", "SCSC", "SCVL", "SCWO", "SCYX", "SCZM", "SDA", "SDGR", "SDOT", "SDST",
    "SEAT", "SEED", "SEER", "SEGG", "SEIC", "SELF", "SELX", "SENEB", "SENS", "SEPN",
    "SERA", "SERV", "SEV", "SEVN", "SEZL", "SFBC", "SFD", "SFHG", "SFST", "SFWL",
    "SGA", "SGC", "SGHT", "SGLY", "SGML", "SGMT", "SGP", "SGRP", "SGRY", "SHAZ",
    "SHBI", "SHC", "SHEN", "SHFS", "SHIM", "SHIP", "SHMD", "SHOO", "SHPH", "SIBN",
    "SIEB", "SIFY", "SIGA", "SIGI", "SILC", "SILO", "SIMO", "SINT", "SION", "SIRI",
    "SJ", "SKBL", "SKIN", "SKK", "SKWD", "SKYE", "SKYQ", "SKYT", "SKYX", "SLDB",
    "SLDE", "SLDP", "SLE", "SLGB", "SLGL", "SLM", "SLMT", "SLN", "SLNG", "SLNH",
    "SLNO", "SLP", "SLS", "SLSN", "SLXN", "SMBC", "SMID", "SMMT", "SMPL", "SMSI",
    "SMTC", "SMTI", "SMTK", "SMX", "SMXT", "SNAL", "SNBR", "SNCY", "SND", "SNDK",
    "SNDL", "SNDX", "SNES", "SNEX", "SNFCA", "SNGX", "SNOA", "SNSE", "SNT", "SNTG",
    "SNTI", "SNWV", "SNY", "SNYR", "SOBR", "SOGP", "SOHU", "SOLS", "SONM", "SOPA",
    "SOPH", "SORA", "SOTK", "SOWG", "SPAI", "SPCB", "SPFI", "SPHL", "SPPL", "SPRB",
    "SPRC", "SPRO", "SPRY", "SPT", "SPWH", "SQFT", "SRAD", "SRBK", "SRCE", "SRTA",
    "SRTS", "SRZN", "SSBI", "SSII", "SSM", "SSNC", "SSP", "SSRM", "SSTI", "SSYS",
    "STAA", "STAK", "STBA", "STEX", "STFS", "STHO", "STI", "STIM", "STKE", "STKH",
    "STKL", "STKS", "STNE", "STOK", "STRA", "STRO", "STRR", "STRS", "STRT", "STRZ",
    "STSS", "STTK", "SUGP", "SUIG", "SUIS", "SUNE", "SUNS", "SUPX", "SURG", "SUUN",
    "SVC", "SVCC", "SVCO", "SVRA", "SVRE", "SVRN", "SWAG", "SWBI", "SWIM", "SWKH",
    "SWMR", "SWVL", "SXTC", "SXTP", "SY", "SYBT", "SYM", "SYNA", "SYPR", "SYRE",
    "TACT", "TANH", "TAOP", "TAOX", "TARA", "TARS", "TATT", "TAYD", "TBBK", "TBCH",
    "TBH", "TBHC", "TBLA", "TBLD", "TBPH", "TBRG", "TC", "TCBI", "TCBS", "TCMD",
    "TCOM", "TCRT", "TCRX", "TCX", "TDIC", "TDOG", "TDTH", "TDUP", "TEAD", "TEAM",
    "TECX", "TELA", "TELO", "TEM", "TENX", "TERN", "TFSL", "TGHL", "TGL", "TGTX",
    "TH", "THCH", "THFF", "THH", "THRM", "THRY", "TIGO", "TIL", "TILE", "TIPT",
    "TITN", "TIVC", "TJGC", "TKLF", "TKNO", "TLF", "TLIH", "TLN", "TLNC", "TLPH",
    "TLRY", "TLS", "TLSA", "TLSI", "TLX", "TMC", "TMCI", "TNDM", "TNGX", "TNMG",
    "TNON", "TNXP", "TNYA", "TOI", "TOMZ", "TONX", "TOP", "TORO", "TOUR", "TOYO",
    "TPCS", "TPG", "TRDA", "TREE", "TRI", "TRIB", "TRIP", "TRMB", "TRMD", "TRMK",
    "TRNR", "TRNS", "TRON", "TROO", "TRS", "TRSG", "TRST", "TRUG", "TRUP", "TRVG",
    "TRVI", "TSAT", "TSBK", "TSEM", "TSHA", "TSSI", "TSUI", "TTAN", "TTEC", "TTEK",
    "TTGT", "TTMI", "TTRX", "TULP", "TURB", "TUSK", "TVGN", "TVRD", "TVTX", "TW",
    "TWAV", "TWFG", "TWG", "TWIN", "TWLV", "TWST", "TXG", "TXMD", "TYGO", "TYRA",
    "TZOO", "UBCP", "UBFO", "UBSI", "UBXG", "UCAR", "UCL", "UCTT", "UDMY", "UEIC",
    "UFCS", "UFG", "UFPT", "UG", "UGRO", "UHG", "UK", "ULBI", "ULCC", "ULH",
    "ULTA", "UNB", "UNCY", "UNIT", "UNTY", "UONE", "UONEK", "UPB", "UPBD", "UPC",
    "UPLD", "UPWK", "UPXI", "URGN", "UROY", "USAR", "USAU", "USCB", "USEA", "USEG",
    "USGO", "USIO", "USLM", "UTHR", "UTMD", "UTSI", "UVSP", "UXIN", "VABK", "VALN",
    "VALU", "VANI", "VAVX", "VBIX", "VBNK", "VC", "VCEL", "VCIC", "VCIG", "VCTR",
    "VCYT", "VEEA", "VEEE", "VELO", "VEON", "VERI", "VERU", "VERX", "VFF", "VFS",
    "VGAS", "VHC", "VHCP", "VHUB", "VICR", "VINP", "VIOT", "VIR", "VIRC", "VISN",
    "VITL", "VIVO", "VIVS", "VKTX", "VLGEA", "VLY", "VMAR", "VMD", "VMET", "VNCE",
    "VNDA", "VNET", "VNOM", "VOD", "VOR", "VOXR", "VRA", "VRAX", "VRCA", "VRDN",
    "VREX", "VRME", "VRNS", "VRRM", "VRSK", "VRSN", "VS", "VSA", "VSAT", "VSEC",
    "VSEE", "VSME", "VSNT", "VSTD", "VSTM", "VTGN", "VTIX", "VTSI", "VTVT", "VUZI",
    "VVOS", "VWAV", "VYGR", "VYNE", "WABC", "WAFDP", "WAFU", "WAI", "WALD", "WASH",
    "WATT", "WAVE", "WAY", "WB", "WBTN", "WBUY", "WCT", "WDAY", "WDFC", "WEN",
    "WERN", "WEST", "WETH", "WETO", "WEYS", "WFCF", "WFF", "WFRD", "WGRX", "WGS",
    "WHLR", "WHWK", "WILC", "WIMI", "WINA", "WING", "WIX", "WKEY", "WKSP", "WLDS",
    "WLFC", "WLTH", "WMG", "WNEB", "WNW", "WOK", "WOOF", "WORX", "WPRT", "WRAP",
    "WRD", "WSBC", "WSBF", "WSBK", "WSC", "WSHP", "WTBA", "WTF", "WTO", "WTW",
    "WULF", "WVE", "WVVI", "WW", "WWD", "WXM", "WYFI", "WYHG", "XAIR", "XBIO",
    "XBIT", "XBP", "XCH", "XCUR", "XELB", "XELLL", "XERS", "XFOR", "XGN", "XHG",
    "XHLD", "XLO", "XMTR", "XNCR", "XNET", "XOMA", "XOS", "XP", "XPEL", "XPON",
    "XRAY", "XRTX", "XRX", "XTKG", "XTLB", "XWEL", "XWIN", "XXII", "YAAS", "YB",
    "YDDL", "YDES", "YDKG", "YHC", "YHGJ", "YI", "YIBO", "YJ", "YMAT", "YMT",
    "YOUL", "YQ", "YSXT", "YTRA", "YXT", "YYAI", "YYGH", "Z", "ZBAI", "ZBAO",
    "ZBIO", "ZCMD", "ZD", "ZDAI", "ZENA", "ZEO", "ZG", "ZGM", "ZJK", "ZJYL",
    "ZKIN", "ZLAB", "ZM", "ZNB", "ZNTL", "ZOOZ", "ZSTK", "ZTEK", "ZURA", "ZVRA",
    "ZYBT", "ZYME",
    # ── NYSE additions ────────────────────────────────────
    "AAMI", "AAP", "AAT", "AAUC", "AB", "ABCB", "ABM", "ABR", "ABX", "ACA",
    "ACCO", "ACEL", "ACH", "ACI", "ACM", "ACP", "ACR", "ACRE", "ACV", "AD",
    "ADC", "ADCT", "ADNT", "ADT", "ADX", "AEE", "AEG", "AER", "AESI", "AFB",
    "AFG", "AG", "AGBK", "AGD", "AGI", "AGL", "AGM", "AGO", "AGRO", "AGX",
    "AHR", "AHRT", "AHT", "AII", "AIN", "AIO", "AIR", "AIT", "AIV", "AJG",
    "AKA", "AKO-A", "AKO-B", "AKR", "AL", "ALC", "ALG", "ALH", "ALIT", "ALSN",
    "ALTG", "ALV", "ALX", "AM", "AMBP", "AMBQ", "AMC", "AMCR", "AMG", "AMN",
    "AMPX", "AMPY", "AMR", "AMRC", "AMRZ", "AMTB", "AMTM", "ANDG", "ANGX", "ANRO",
    "ANVS", "AOD", "AOMR", "AON", "AORT", "AOS", "AP", "APAM", "APG", "APH",
    "APLE", "APO", "AQN", "AR", "ARCO", "ARDC", "ARDT", "ARE", "ARES", "ARI",
    "ARIS", "ARL", "ARMK", "AROC", "ARR", "ARW", "ARX", "AS", "ASA", "ASB",
    "ASC", "ASG", "ASGI", "ASGN", "ASH", "ASIC", "ASIX", "ASPN", "ASR", "ATEN",
    "ATKR", "ATMU", "ATO", "ATR", "ATS", "AU", "AUB", "AUNA", "AVA", "AVBC",
    "AVD", "AVK", "AVNS", "AVNT", "AVTR", "AVY", "AWF", "AWI", "AWP", "AX",
    "AXR", "AXS", "AXTA", "AYI", "AZN", "AZZ", "B", "BAH", "BAK", "BALL",
    "BALY", "BAM", "BANC", "BAP", "BB", "BBAI", "BBAR", "BBBY", "BBDC", "BBN",
    "BBT", "BBU", "BBUC", "BBVA", "BBW", "BBWI", "BC", "BCAT", "BCC", "BCE",
    "BCH", "BCO", "BCS", "BCSS", "BCX", "BDC", "BDJ", "BDN", "BE", "BEBE",
    "BEP", "BEPC", "BEPH", "BEPI", "BEPJ", "BETA", "BF-A", "BF-B", "BFLY", "BFS",
    "BGB", "BGH", "BGR", "BGS", "BGSF", "BGSI", "BGT", "BGX", "BGY", "BH",
    "BHC", "BHE", "BHK", "BHR", "BHV", "BHVN", "BIO", "BIO-B", "BIP", "BIPC",
    "BIPI", "BIRK", "BIT", "BKD", "BKE", "BKH", "BKKT", "BKSY", "BKT", "BKU",
    "BKV", "BLCO", "BLD", "BLND", "BLSH", "BLW", "BLX", "BMA", "BME", "BMEZ",
    "BMI", "BMN", "BMO", "BN", "BNED", "BNJ", "BNL", "BNS", "BNT", "BOBS",
    "BOC", "BOE", "BOH", "BORR", "BOW", "BP", "BPRE", "BR", "BRBR", "BRC",
    "BRCC", "BRK-A", "BRO", "BROS", "BRSL", "BRSP", "BRT", "BRW", "BRX", "BSAC",
    "BSL", "BST", "BSTZ", "BTE", "BTGO", "BTI", "BTO", "BTT", "BTU", "BTX",
    "BTZ", "BUD", "BUI", "BUR", "BURL", "BV", "BVN", "BW", "BWA", "BWG",
    "BWLP", "BWMX", "BWXT", "BX", "BXC", "BXMT", "BXMX", "BY", "BYD", "BZH",
    "CAAP", "CABO", "CACI", "CAE", "CAF", "CAH", "CAL", "CALY", "CANG", "CARS",
    "CAVA", "CBAN", "CBL", "CBNA", "CBRE", "CBT", "CBU", "CBZ", "CC", "CCIF",
    "CCJ", "CCK", "CCO", "CCS", "CCU", "CCZ", "CDE", "CDP", "CDRE", "CEE",
    "CFND", "CFR", "CGAU", "CHCT", "CHE", "CHGG", "CHH", "CHMI", "CHT", "CIA",
    "CIF", "CII", "CIM", "CINT", "CION", "CLB", "CLF", "CLH", "CLPR", "CLS",
    "CLVT", "CLW", "CM", "CMBT", "CMC", "CMDB", "CMI", "CMP", "CMRE", "CMTG",
    "CMU", "CNA", "CNH", "CNK", "CNM", "CNMD", "CNNE", "CNO", "CNQ", "CNR",
    "CNS", "CNX", "COHR", "COMP", "CON", "COOK", "COR", "COSO", "COUR", "CPA",
    "CPAY", "CPF", "CPK", "CPNG", "CPS", "CQP", "CR", "CRBG", "CRC", "CRCL",
    "CRD-A", "CRD-B", "CRGY", "CRH", "CRI", "CRK", "CRS", "CRT", "CSAN", "CSL",
    "CSTM", "CSV", "CSW", "CTEV", "CTO", "CTOS", "CTRA", "CTRE", "CTRI", "CTS",
    "CUBI", "CUK", "CURB", "CURV", "CUZ", "CVE", "CVEO", "CVI", "CVLG", "CVSA",
    "CWAN", "CWEN", "CWH", "CWK", "CWT", "CX", "CXE", "CXH", "CXM", "CXT",
    "CXW", "CYD", "CYH", "DAC", "DAN", "DAR", "DB", "DBD", "DBI", "DBL",
    "DBRG", "DCH", "DCI", "DCO", "DDD", "DDS", "DDT", "DEA", "DEC", "DEI",
    "DEO", "DFH", "DFIN", "DGX", "DHF", "DHT", "DHX", "DIAX", "DIN", "DINO",
    "DK", "DKS", "DLB", "DLNG", "DLX", "DLY", "DMA", "DMB", "DMO", "DNA",
    "DNP", "DOC", "DOLE", "DOUG", "DPG", "DSL", "DSM", "DSU", "DSX", "DT",
    "DTE", "DTF", "DTM", "DV", "DX", "DXC", "DXYZ", "E", "EAF", "EAI",
    "EARN", "EAT", "EBF", "EBS", "ECAT", "ECC", "ECG", "ECO", "ECVT", "ED",
    "EDD", "EDF", "EDU", "EE", "EEA", "EEX", "EFC", "EFR", "EFT", "EFX",
    "EFXT", "EG", "EGP", "EGY", "EHAB", "EHC", "EHI", "EIC", "EIG", "EIX",
    "ELAN", "ELC", "ELME", "ELS", "ELV", "EMA", "EMBJ", "EMD", "EME", "EMF",
    "EMO", "EMP", "ENB", "ENJ", "ENO", "ENOV", "ENR", "ENS", "ENVA", "EOD",
    "EOI", "EOS", "EOT", "EPAC", "EPC", "EPD", "EPR", "EPRT", "EQBK", "EQH",
    "EQNR", "EQS", "EQT", "ERO", "ESE", "ESI", "ESRT", "ETB", "ETD", "ETG",
    "ETJ", "ETO", "ETV", "ETW", "ETX", "ETY", "EVC", "EVEX", "EVF", "EVG",
    "EVH", "EVMN", "EVN", "EVR", "EVT", "EVTC", "EXG", "EXK", "EXP", "FAF",
    "FBK", "FBP", "FBRT", "FC", "FCF", "FCN", "FCT", "FDP", "FDS", "FERG",
    "FET", "FF", "FFA", "FFWM", "FG", "FHI", "FICO", "FIG", "FIHL", "FINS",
    "FIX", "FLC", "FLG", "FLNG", "FLO", "FLOC", "FLR", "FLS", "FLUT", "FMN",
    "FMX", "FMY", "FN", "FNB", "FND", "FNF", "FNV", "FOA", "FOF", "FOR",
    "FOUR", "FPH", "FPI", "FPS", "FR", "FRA", "FRO", "FRT", "FSCO", "FSM",
    "FSS", "FSSL", "FT", "FTHY", "FTI", "FTK", "FTS", "FTV", "FTW", "FUBO",
    "FUL", "FUN", "FVR", "FVRR", "GAB", "GAM", "GAP", "GBAB", "GBTG", "GCO",
    "GCTS", "GCV", "GDL", "GDO", "GDOT", "GDV", "GEF", "GEL", "GEO", "GETY",
    "GEV", "GF", "GFL", "GFR", "GGB", "GGT", "GGZ", "GHC", "GHM", "GHY",
    "GIB", "GIC", "GIL", "GJH", "GJO", "GJP", "GJR", "GJT", "GLOB", "GME",
    "GMED", "GNE", "GNK", "GNL", "GNT", "GOF", "GOLF", "GOOS", "GPC", "GPGI",
    "GPK", "GPMT", "GPOR", "GPRK", "GRC", "GRDN", "GRMN", "GRND", "GRNT", "GROV",
    "GRX", "GSL", "GTES", "GTN", "GUG", "GUT", "GWH", "GWRE", "GXO", "H",
    "HAE", "HAFN", "HBB", "HBM", "HCC", "HCI", "HDB", "HE", "HEI-A", "HEQ",
    "HFRO", "HG", "HGLB", "HGTY", "HGV", "HHH", "HII", "HIO", "HIPO", "HIX",
    "HL", "HLF", "HLI", "HLIO", "HLLY", "HMC", "HMN", "HMY", "HNGE", "HNI",
    "HOG", "HOMB", "HOV", "HP", "HPF", "HPP", "HQH", "HQL", "HRB", "HRI",
    "HRTG", "HSBC", "HSHP", "HSY", "HTB", "HTD", "HTH", "HUBB", "HUN", "HVT",
    "HVT-A", "HXL", "HY", "HYI", "HYT", "HZO", "IAE", "IAG", "IBN", "IBTA",
    "ICL", "IDA", "IDE", "IDT", "IEX", "IFF", "IFN", "IFS", "IGA", "IGD",
    "IGI", "IGR", "IHD", "IHS", "IIF", "IIIN", "IIM", "IMAX", "INFQ", "ING",
    "INGM", "INGR", "INN", "INR", "INSP", "INVX", "IOT", "IP", "IPI", "IQI",
    "IRM", "IRT", "ISD", "ITGR", "ITT", "IVR", "IVT", "J", "JAN", "JBI",
    "JBK", "JBL", "JBS", "JBTM", "JCE", "JCI", "JEF", "JELD", "JFR", "JGH",
    "JHG", "JHI", "JHS", "JHX", "JILL", "JLL", "JLS", "JMM", "JOE", "JOF",
    "JQC", "JRI", "JRS", "JXN", "KAI", "KB", "KBDC", "KBH", "KBR", "KD",
    "KEN", "KEP", "KEX", "KF", "KFS", "KGC", "KGS", "KIO", "KKR", "KLAR",
    "KLC", "KMPR", "KMT", "KN", "KNF", "KNSL", "KNTK", "KNX", "KODK", "KOP",
    "KORE", "KOS", "KRC", "KREF", "KRG", "KRMN", "KRO", "KSS", "KT", "KTB",
    "KTF", "KTN", "KVUE", "KVYO", "KW", "KWR", "KYN", "LAC", "LADR", "LANV",
    "LAR", "LAW", "LAZ", "LB", "LBRT", "LC", "LCII", "LDI", "LEA", "LEG",
    "LEO", "LEU", "LEVI", "LFT", "LGI", "LH", "LHX", "LII", "LION", "LND",
    "LNN", "LOAR", "LOB", "LOCL", "LOMA", "LPG", "LPL", "LRN", "LSPD", "LTC",
    "LTH", "LUCK", "LUMN", "LVWR", "LXU", "LZB", "LZM", "M", "MAC", "MAGN",
    "MANE", "MANU", "MATV", "MATX", "MAX", "MBC", "MBI", "MC", "MCB", "MCI",
    "MCK", "MCN", "MCR", "MCS", "MCY", "MD", "MDU", "MDV", "MEC", "MED",
    "MEG", "MEGI", "MEI", "MFA", "MFC", "MFG", "MFM", "MG", "MGA", "MGF",
    "MGY", "MH", "MHD", "MHF", "MHO", "MIAX", "MICC", "MIN", "MIR", "MITT",
    "MIY", "MKL", "MLI", "MLM", "MLP", "MLR", "MMD", "MMI", "MMM", "MMS",
    "MMT", "MMU", "MNTN", "MOD", "MOV", "MPA", "MPT", "MPV", "MPX", "MQY",
    "MRP", "MRSH", "MSA", "MSB", "MSD", "MSDL", "MSGE", "MSGS", "MSI", "MSIF",
    "MSM", "MT", "MTN", "MTR", "MTUS", "MTW", "MTX", "MTZ", "MUA", "MUC",
    "MUFG", "MUJ", "MUR", "MUX", "MWA", "MX", "MXE", "MXF", "MYE", "MYI",
    "MYN", "NABL", "NAC", "NAD", "NAN", "NAT", "NATL", "NAZ", "NBB", "NBHC",
    "NBR", "NBXG", "NC", "NCA", "NCDL", "NCV", "NCZ", "NDMO", "NE", "NEA",
    "NEU", "NEXA", "NFG", "NFJ", "NGS", "NGVC", "NGVT", "NHI", "NIC", "NIE",
    "NIM", "NIQ", "NJR", "NKX", "NL", "NLOP", "NLY", "NMAI", "NMAX", "NMCO",
    "NMG", "NMI", "NMS", "NMT", "NMZ", "NNI", "NNN", "NNY", "NOA", "NOG",
    "NOM", "NOMD", "NOTE", "NOV", "NP", "NPB", "NPCT", "NPK", "NPKI", "NPO",
    "NPV", "NPWR", "NQP", "NRDY", "NREF", "NRGV", "NRK", "NRP", "NRT", "NSP",
    "NTB", "NTR", "NTZ", "NUS", "NUV", "NUVB", "NUW", "NVG", "NVGS", "NVO",
    "NVRI", "NVS", "NVST", "NVT", "NWN", "NX", "NXDR", "NXDT", "NXE", "NXG",
    "NXJ", "NXP", "NXRT", "NYC", "NYT", "NZF", "OBK", "OC", "ODC", "ODV",
    "OEC", "OFRM", "OGN", "OGS", "OHI", "OI", "OIA", "OII", "OIS", "OKLO",
    "OLN", "OLP", "OMF", "ONIT", "ONL", "ONTF", "OOMA", "OPAD", "OPFI", "OPLN",
    "OPP", "OPTU", "OPY", "OR", "ORA", "ORC", "ORI", "ORN", "OSCR", "OSG",
    "OSK", "OTF", "OUT", "OVV", "OWL", "OWLT", "OXM", "PAAS", "PAC", "PACK",
    "PACS", "PAGS", "PAI", "PAM", "PAR", "PARR", "PAXS", "PAY", "PAYC", "PB",
    "PBA", "PBF", "PBH", "PBI", "PBR", "PBT", "PCF", "PCM", "PCN", "PCOR",
    "PCQ", "PD", "PDCC", "PDI", "PDM", "PDO", "PDS", "PDT", "PDX", "PEB",
    "PEG", "PEN", "PEO", "PERF", "PEW", "PFL", "PFN", "PFS", "PFSI", "PGP",
    "PGZ", "PHG", "PHI", "PHIN", "PHK", "PHR", "PII", "PIM", "PIPR", "PJT",
    "PKG", "PKST", "PL", "PML", "PMM", "PMO", "PMT", "PNI", "PNNT", "POR",
    "POST", "PPLC", "PPT", "PR", "PRA", "PRI", "PRKS", "PRLB", "PRM", "PRMB",
    "PRSU", "PSBD", "PSFE", "PSN", "PSO", "PSQH", "PSTG", "PTY", "PUK", "PWR",
    "PXED", "PYT", "Q", "QBTS", "QGEN", "QUAD", "QXO", "R", "RA", "RACE",
    "RAL", "RAMP", "RBA", "RBC", "RBRK", "RC", "RCI", "RCS", "RDDT", "RDW",
    "RDY", "RES", "REXR", "REZI", "RFI", "RFL", "RFM", "RFMZ", "RGA", "RGR",
    "RGT", "RHLD", "RIG", "RIO", "RITM", "RIV", "RJF", "RKT", "RLI", "RLJ",
    "RLTY", "RM", "RMAX", "RMI", "RMM", "RMMZ", "RMT", "RNG", "ROG", "ROL",
    "RPC", "RPT", "RQI", "RRC", "RSF", "RSI", "RSKD", "RVI", "RVLV", "RVT",
    "RWT", "RXO", "RY", "RYAM", "RYAN", "RYN", "RYZ", "SA", "SABA", "SAM",
    "SAN", "SAP", "SAR", "SARO", "SB", "SBDS", "SBH", "SBI", "SBR", "SBSI",
    "SBXD", "SBXE", "SCCO", "SCD", "SCI", "SCL", "SCM", "SD", "SDHC", "SDHY",
    "SDRL", "SEE", "SEG", "SEI", "SEM", "SEMR", "SES", "SF", "SFBS", "SFL",
    "SG", "SGHC", "SGI", "SGU", "SI", "SID", "SIG", "SII", "SILA", "SITC",
    "SITE", "SJT", "SKE", "SKIL", "SKLZ", "SKM", "SKT", "SKY", "SKYH", "SLF",
    "SLGN", "SLQT", "SLVM", "SMA", "SMBK", "SMC", "SMG", "SMHI", "SMP", "SMR",
    "SMRT", "SMWB", "SN", "SNA", "SNDA", "SNDR", "SNN", "SNX", "SOBO", "SOC",
    "SOLV", "SOMN", "SON", "SOR", "SOS", "SPCE", "SPE", "SPH", "SPHR", "SPIR",
    "SPMC", "SPNT", "SPRU", "SPXC", "SPXX", "SQM", "SR", "SRFM", "SRG", "SRI",
    "SRL", "SRV", "SSB", "SSD", "SST", "SSTK", "ST", "STC", "STE", "STEL",
    "STK", "STLA", "STM", "STN", "STNG", "STUB", "STVN", "STWD", "SU", "SUI",
    "SUNB", "SVV", "SW", "SWX", "SWZ", "SXC", "SXI", "SXT", "TAC", "TALO",
    "TBBB", "TBN", "TCBX", "TCI", "TD", "TDAY", "TDC", "TDF", "TDS", "TDW",
    "TE", "TECK", "TEI", "TEN", "TEO", "TEX", "TFII", "TFIN", "TFPM", "TFX",
    "TG", "TGE", "TGLS", "TGS", "TGT", "THC", "THG", "THO", "THQ", "THR",
    "THW", "TIC", "TISI", "TK", "TKC", "TKO", "TKR", "TLK", "TM", "TNC",
    "TNET", "TNK", "TNL", "TPB", "TPC", "TPH", "TPL", "TR", "TRAK", "TRC",
    "TREX", "TRNO", "TROX", "TRP", "TRTX", "TRU", "TSI", "TSLX", "TSM", "TSQ",
    "TTAM", "TTC", "TTE", "TTI", "TU", "TV", "TVC", "TVE", "TWI", "TWLO",
    "TWN", "TWO", "TXNM", "TY", "TYG", "TYL", "UA", "UAA", "UAMY", "UBS",
    "UCB", "UE", "UFI", "UGI", "UHAL", "UHS", "UHT", "UI", "UIS", "ULS",
    "UMC", "UMH", "UNF", "UNFI", "UP", "USA", "USFD", "USNA", "USPH", "UTF",
    "UTI", "UTL", "UTZ", "UVE", "UVV", "UWMC", "VAC", "VAL", "VATE", "VBF",
    "VCV", "VCX", "VEL", "VET", "VG", "VGI", "VGM", "VHI", "VIA", "VIK",
    "VIRT", "VKQ", "VLN", "VLT", "VLTO", "VMC", "VMI", "VMO", "VNO", "VNT",
    "VOYA", "VOYG", "VPG", "VPV", "VRE", "VRT", "VRTS", "VSCO", "VSH", "VST",
    "VSTS", "VTEX", "VTN", "VTOL", "VTS", "VVR", "VVV", "VVX", "VYX", "WAL",
    "WBI", "WBX", "WCC", "WCN", "WD", "WDI", "WEA", "WEAV", "WEX", "WFG",
    "WGO", "WH", "WHD", "WHG", "WHR", "WIA", "WIT", "WIW", "WK", "WKC",
    "WLK", "WLY", "WLYB", "WMK", "WMS", "WNC", "WOLF", "WOR", "WRBY", "WS",
    "WSM", "WSO", "WSO-B", "WSR", "WST", "WT", "WTI", "WTM", "WTS", "WTTR",
    "WU", "WWW", "WY", "XFLH", "XFLT", "XHR", "XPER", "XPRO", "XRN", "XXI",
    "XYZ", "XZO", "YELP", "YETI", "YEXT", "YOU", "YPF", "YUMC", "ZGN", "ZIM",
    "ZIP", "ZTR", "ZVIA", "ZWS",
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
    # ── NASDAQ/NYSE additions ─────────────────────────────
    "AAP": "Consumer", "ABVE": "Consumer", "ACEL": "Consumer", "ADSK": "Consumer", "AENT": "Consumer", "AGAE": "Consumer", "ALGT": "Consumer", "ALV": "Consumer",
    "AMC": "Consumer", "ARKR": "Consumer", "BGS": "Consumer", "BHR": "Consumer", "BOF": "Consumer", "BRAG": "Consumer", "BRFH": "Consumer", "BRID": "Consumer",
    "BRLS": "Consumer", "BYD": "Consumer", "CAAS": "Consumer", "CALM": "Consumer", "CHH": "Consumer", "CRSR": "Consumer", "CZR": "Consumer", "DLPN": "Consumer",
    "FIZZ": "Consumer", "FLO": "Consumer", "FLUT": "Consumer", "FRSX": "Consumer", "FUN": "Consumer", "FWRG": "Consumer", "GBTG": "Consumer", "GDEN": "Consumer",
    "GENK": "Consumer", "GIII": "Consumer", "GLPI": "Consumer", "GTIM": "Consumer", "H": "Consumer", "HBNB": "Consumer", "HFFG": "Consumer", "IBG": "Consumer",
    "INN": "Consumer", "INSE": "Consumer", "JJSF": "Consumer", "KUST": "Consumer", "LI": "Consumer", "LUCK": "Consumer", "LWAY": "Consumer", "MLCO": "Consumer",
    "MNST": "Consumer", "MSGE": "Consumer", "NOMD": "Consumer", "PAL": "Consumer", "PEB": "Consumer", "PFAI": "Consumer", "PLAY": "Consumer", "PSNY": "Consumer",
    "RAVE": "Consumer", "REE": "Consumer", "SEG": "Consumer", "SEGG": "Consumer", "SENEB": "Consumer", "SFD": "Consumer", "SMPL": "Consumer", "SPHR": "Consumer",
    "STKH": "Consumer", "STRZ": "Consumer", "TBBB": "Consumer", "TNL": "Consumer", "TZOO": "Consumer", "UNFI": "Consumer", "USFD": "Consumer", "VFS": "Consumer",
    "WBTN": "Consumer", "WFCF": "Consumer", "WH": "Consumer", "WILC": "Consumer", "WYHG": "Consumer", "XHR": "Consumer",
    "ACFN": "Energy", "ADSE": "Energy", "AEC": "Energy", "AEIS": "Energy", "AESI": "Energy", "AIOT": "Energy", "AMPY": "Energy", "APC": "Energy",
    "AQN": "Energy", "ARIS": "Energy", "ASTI": "Energy", "ATO": "Energy", "B": "Energy", "BE": "Energy", "BEP": "Energy", "BEPC": "Energy",
    "BGR": "Energy", "BLDP": "Energy", "BNRG": "Energy", "BTE": "Energy", "BTU": "Energy", "BUI": "Energy", "BVN": "Energy", "CBAT": "Energy",
    "CDE": "Energy", "CETY": "Energy", "CLNE": "Energy", "CNEY": "Energy", "COOT": "Energy", "CQP": "Energy", "CREG": "Energy", "CRGY": "Energy",
    "CSIQ": "Energy", "CTRA": "Energy", "CVE": "Energy", "CVI": "Energy", "CWEN": "Energy", "DEC": "Energy", "DFLI": "Energy", "DGXX": "Energy",
    "DLNG": "Energy", "DTE": "Energy", "EE": "Energy", "EFOI": "Energy", "EGY": "Energy", "EMO": "Energy", "ENGS": "Energy", "ENLT": "Energy",
    "EOSE": "Energy", "EPSN": "Energy", "ESOA": "Energy", "EU": "Energy", "EXE": "Energy", "FCEL": "Energy", "FET": "Energy", "FF": "Energy",
    "FLUX": "Energy", "FMST": "Energy", "FNUC": "Energy", "FPS": "Energy", "FSM": "Energy", "FTCI": "Energy", "FTEK": "Energy", "GASS": "Energy",
    "GEL": "Energy", "GNE": "Energy", "GP": "Energy", "GPOR": "Energy", "HL": "Energy", "HMY": "Energy", "HNRG": "Energy", "HPK": "Energy",
    "HTOO": "Energy", "HYMC": "Energy", "ICON": "Energy", "IMPP": "Energy", "IMSR": "Energy", "IPW": "Energy", "IPWR": "Energy", "KEP": "Energy",
    "KGEI": "Energy", "KGS": "Energy", "KLXE": "Energy", "KOS": "Energy", "KYN": "Energy", "LBRT": "Energy", "LEU": "Energy", "LSE": "Energy",
    "MGY": "Energy", "MNTK": "Energy", "MUR": "Energy", "MWH": "Energy", "NESR": "Energy", "NEXM": "Energy", "NFE": "Energy", "NFG": "Energy",
    "NGS": "Energy", "NNE": "Energy", "NOEM": "Energy", "NOG": "Energy", "NPWR": "Energy", "NRGV": "Energy", "NRT": "Energy", "NUAI": "Energy",
    "NUCL": "Energy", "NWE": "Energy", "NXE": "Energy", "NXT": "Energy", "ODC": "Energy", "OESX": "Energy", "OGS": "Energy", "OIS": "Energy",
    "OMSE": "Energy", "OPAL": "Energy", "PBF": "Energy", "PBR": "Energy", "PEGA": "Energy", "PLUG": "Energy", "PN": "Energy", "PNRG": "Energy",
    "POLA": "Energy", "PPSI": "Energy", "PSIX": "Energy", "PTEN": "Energy", "RAYA": "Energy", "RBNE": "Energy", "RNW": "Energy", "RPGL": "Energy",
    "SCZM": "Energy", "SD": "Energy", "SDST": "Energy", "SEI": "Energy", "SLDP": "Energy", "SMR": "Energy", "SMXT": "Energy", "SPRU": "Energy",
    "SRV": "Energy", "SSRM": "Energy", "SU": "Energy", "SUNE": "Energy", "SWX": "Energy", "SXC": "Energy", "TALO": "Energy", "TE": "Energy",
    "TEN": "Energy", "TGS": "Energy", "TLN": "Energy", "TRP": "Energy", "TURB": "Energy", "TUSK": "Energy", "TXNM": "Energy", "TYG": "Energy",
    "TYGO": "Energy", "UCAR": "Energy", "UFG": "Energy", "USEG": "Energy", "USGO": "Energy", "VEEE": "Energy", "VET": "Energy", "VGAS": "Energy",
    "VIVO": "Energy", "VNOM": "Energy", "VTS": "Energy", "WAVE": "Energy", "WPRT": "Energy", "XELLL": "Energy", "ZEO": "Energy",
    "AAMI": "Finance", "ABCB": "Finance", "ACGL": "Finance", "ACIC": "Finance", "ACP": "Finance", "AFCG": "Finance", "AFG": "Finance", "AGM": "Finance",
    "AGNC": "Finance", "AII": "Finance", "AIV": "Finance", "ALDF": "Finance", "ALRS": "Finance", "AMAL": "Finance", "AMTB": "Finance", "AOMR": "Finance",
    "APAM": "Finance", "ARDC": "Finance", "ARL": "Finance", "ASIC": "Finance", "ASRV": "Finance", "ATON": "Finance", "AUB": "Finance", "AUBN": "Finance",
    "AVBC": "Finance", "AVBH": "Finance", "AX": "Finance", "AXS": "Finance", "BAFN": "Finance", "BAM": "Finance", "BANX": "Finance", "BBT": "Finance",
    "BCAL": "Finance", "BCAT": "Finance", "BCBP": "Finance", "BCG": "Finance", "BCIC": "Finance", "BCSS": "Finance", "BFC": "Finance", "BGB": "Finance",
    "BGX": "Finance", "BHF": "Finance", "BHRB": "Finance", "BKU": "Finance", "BLFY": "Finance", "BMO": "Finance", "BMRC": "Finance", "BNS": "Finance",
    "BOH": "Finance", "BOTJ": "Finance", "BPRN": "Finance", "BR": "Finance", "BRR": "Finance", "BRSP": "Finance", "BRW": "Finance", "BSBK": "Finance",
    "BSRR": "Finance", "BSVN": "Finance", "BTO": "Finance", "BTZ": "Finance", "BUR": "Finance", "BVFL": "Finance", "BWFG": "Finance", "BWIN": "Finance",
    "BXMT": "Finance", "BY": "Finance", "BYFC": "Finance", "CACC": "Finance", "CARE": "Finance", "CASH": "Finance", "CBAN": "Finance", "CBFV": "Finance",
    "CBNA": "Finance", "CBNK": "Finance", "CBU": "Finance", "CCB": "Finance", "CCBG": "Finance", "CCEC": "Finance", "CCIF": "Finance", "CCIX": "Finance",
    "CCNE": "Finance", "CCXI": "Finance", "CDTG": "Finance", "CFBK": "Finance", "CFFI": "Finance", "CFFN": "Finance", "CFR": "Finance", "CHMG": "Finance",
    "CHMI": "Finance", "CHYM": "Finance", "CIM": "Finance", "CION": "Finance", "CLBK": "Finance", "CLST": "Finance", "CM": "Finance", "CMII": "Finance",
    "CMTG": "Finance", "CMTV": "Finance", "CNA": "Finance", "CNO": "Finance", "CNOB": "Finance", "COFS": "Finance", "COLB": "Finance", "CPF": "Finance",
    "CRBG": "Finance", "CUBI": "Finance", "CXH": "Finance", "CZFS": "Finance", "CZWI": "Finance", "DB": "Finance", "DBL": "Finance", "DDT": "Finance",
    "DFIN": "Finance", "DHIL": "Finance", "DMO": "Finance", "DX": "Finance", "EAI": "Finance", "EARN": "Finance", "EBC": "Finance", "EBMT": "Finance",
    "ECAT": "Finance", "ECBK": "Finance", "ECC": "Finance", "ECPG": "Finance", "EFC": "Finance", "EFSI": "Finance", "ELC": "Finance", "EMP": "Finance",
    "ENJ": "Finance", "ENO": "Finance", "ESQ": "Finance", "FBIZ": "Finance", "FBK": "Finance", "FBLA": "Finance", "FBNC": "Finance", "FBP": "Finance",
    "FCAP": "Finance", "FCBC": "Finance", "FCF": "Finance", "FDBC": "Finance", "FDSB": "Finance", "FFBC": "Finance", "FFIC": "Finance", "FIHL": "Finance",
    "FINS": "Finance", "FISI": "Finance", "FITBI": "Finance", "FLG": "Finance", "FMAO": "Finance", "FMY": "Finance", "FNF": "Finance", "FNLC": "Finance",
    "FNWB": "Finance", "FNWD": "Finance", "FRAF": "Finance", "FRBA": "Finance", "FRST": "Finance", "FRT": "Finance", "FSBC": "Finance", "FSCO": "Finance",
    "FSEA": "Finance", "FSSL": "Finance", "FSUN": "Finance", "FULT": "Finance", "FVCB": "Finance", "GABC": "Finance", "GAM": "Finance", "GBAB": "Finance",
    "GBFH": "Finance", "GCBC": "Finance", "GECC": "Finance", "GIG": "Finance", "GIX": "Finance", "GLRE": "Finance", "GPMT": "Finance", "GRNQ": "Finance",
    "GSHD": "Finance", "HBCP": "Finance", "HBNC": "Finance", "HBT": "Finance", "HDB": "Finance", "HERZ": "Finance", "HFBL": "Finance", "HFWA": "Finance",
    "HG": "Finance", "HNVR": "Finance", "HQL": "Finance", "HRTG": "Finance", "HVII": "Finance", "HYNE": "Finance", "IBN": "Finance", "ICMB": "Finance",
    "IFS": "Finance", "IGI": "Finance", "IGIC": "Finance", "IIF": "Finance", "INBK": "Finance", "INDB": "Finance", "ISBA": "Finance", "ISTR": "Finance",
    "ITIC": "Finance", "IVR": "Finance", "JCAP": "Finance", "JEF": "Finance", "JHI": "Finance", "JLS": "Finance", "JMSB": "Finance", "JOF": "Finance",
    "JQC": "Finance", "JXN": "Finance", "KB": "Finance", "KBON": "Finance", "KFFB": "Finance", "KFS": "Finance", "KNSL": "Finance", "KRNY": "Finance",
    "LADR": "Finance", "LARK": "Finance", "LC": "Finance", "LKFN": "Finance", "LNKB": "Finance", "LOAN": "Finance", "LPLA": "Finance", "LPRO": "Finance",
    "LSBK": "Finance", "MBBC": "Finance", "MBIN": "Finance", "MBWM": "Finance", "MCB": "Finance", "MCBS": "Finance", "MCHB": "Finance", "MCI": "Finance",
    "MDBH": "Finance", "MFA": "Finance", "MFC": "Finance", "MFG": "Finance", "MFIC": "Finance", "MFIN": "Finance", "MGYR": "Finance", "MITT": "Finance",
    "MKZR": "Finance", "MLCI": "Finance", "MMU": "Finance", "MPB": "Finance", "MPV": "Finance", "MRNO": "Finance", "MSBI": "Finance", "MSDL": "Finance",
    "MUFG": "Finance", "MVBF": "Finance", "NBBK": "Finance", "NBHC": "Finance", "NBN": "Finance", "NCDL": "Finance", "NCPL": "Finance", "NECB": "Finance",
    "NFBK": "Finance", "NHIC": "Finance", "NIC": "Finance", "NKSH": "Finance", "NLY": "Finance", "NMCO": "Finance", "NP": "Finance", "NRIM": "Finance",
    "NSTS": "Finance", "NTB": "Finance", "NVG": "Finance", "NWFL": "Finance", "NYC": "Finance", "NZF": "Finance", "OBK": "Finance", "OBT": "Finance",
    "OCCI": "Finance", "OCFC": "Finance", "OFS": "Finance", "ONB": "Finance", "OPBK": "Finance", "OPRT": "Finance", "ORC": "Finance", "ORIQ": "Finance",
    "ORRF": "Finance", "OSBC": "Finance", "OVLY": "Finance", "OWL": "Finance", "OXLC": "Finance", "OXSQ": "Finance", "OZK": "Finance", "PAI": "Finance",
    "PAX": "Finance", "PBFS": "Finance", "PBHC": "Finance", "PCB": "Finance", "PCSC": "Finance", "PDCC": "Finance", "PDLB": "Finance", "PEBK": "Finance",
    "PEBO": "Finance", "PFG": "Finance", "PFS": "Finance", "PFSI": "Finance", "PGC": "Finance", "PKBK": "Finance", "PLBC": "Finance", "PLUT": "Finance",
    "PMT": "Finance", "PNBK": "Finance", "PNNT": "Finance", "PROV": "Finance", "PSBD": "Finance", "RAND": "Finance", "RBB": "Finance", "RBCAA": "Finance",
    "RBKB": "Finance", "RC": "Finance", "RGA": "Finance", "RITM": "Finance", "RJF": "Finance", "RMBI": "Finance", "RPC": "Finance", "RSF": "Finance",
    "RVSB": "Finance", "RY": "Finance", "SABA": "Finance", "SAFT": "Finance", "SAMG": "Finance", "SAR": "Finance", "SBCF": "Finance", "SBFG": "Finance",
    "SCD": "Finance", "SCM": "Finance", "SEIC": "Finance", "SF": "Finance", "SFBC": "Finance", "SIEB": "Finance", "SIGI": "Finance", "SKWD": "Finance",
    "SLDE": "Finance", "SLF": "Finance", "SMBC": "Finance", "SMBK": "Finance", "SNFCA": "Finance", "SOR": "Finance", "SPFI": "Finance", "SPMC": "Finance",
    "SRBK": "Finance", "SSB": "Finance", "SSBI": "Finance", "STBA": "Finance", "STEL": "Finance", "SUUN": "Finance", "SVCC": "Finance", "SYBT": "Finance",
    "TBBK": "Finance", "TCBI": "Finance", "TCI": "Finance", "TD": "Finance", "TFIN": "Finance", "TFSL": "Finance", "THFF": "Finance", "THG": "Finance",
    "TLNC": "Finance", "TOP": "Finance", "TREE": "Finance", "TRST": "Finance", "TSBK": "Finance", "TSLX": "Finance", "TWLV": "Finance", "TWO": "Finance",
    "UBCP": "Finance", "UBSI": "Finance", "UCB": "Finance", "UNB": "Finance", "UNTY": "Finance", "USCB": "Finance", "UVE": "Finance", "UVSP": "Finance",
    "VABK": "Finance", "VBNK": "Finance", "VCIC": "Finance", "VCTR": "Finance", "VEL": "Finance", "VGM": "Finance", "VHCP": "Finance", "VINP": "Finance",
    "VIRT": "Finance", "VLY": "Finance", "VOYA": "Finance", "VRTS": "Finance", "VTN": "Finance", "WABC": "Finance", "WAL": "Finance", "WASH": "Finance",
    "WHLR": "Finance", "WNEB": "Finance", "WSBF": "Finance", "WSBK": "Finance", "WTBA": "Finance", "WTF": "Finance", "WTM": "Finance", "XFLH": "Finance",
    "XXI": "Finance", "Z": "Finance",
    "AAPG": "Healthcare", "AARD": "Healthcare", "ABCL": "Healthcare", "ABEO": "Healthcare", "ABOS": "Healthcare", "ABUS": "Healthcare", "ABVC": "Healthcare", "ACH": "Healthcare",
    "ACHC": "Healthcare", "ACIU": "Healthcare", "ACRS": "Healthcare", "ACRV": "Healthcare", "ACTU": "Healthcare", "ACXP": "Healthcare", "ADCT": "Healthcare", "ADGM": "Healthcare",
    "ADIL": "Healthcare", "ADMA": "Healthcare", "ADPT": "Healthcare", "ADXN": "Healthcare", "AEMD": "Healthcare", "AFJK": "Healthcare", "AGIO": "Healthcare", "AGL": "Healthcare",
    "AGMB": "Healthcare", "AHCO": "Healthcare", "AHG": "Healthcare", "AHR": "Healthcare", "AHT": "Healthcare", "AIFF": "Healthcare", "AKBA": "Healthcare", "AKTS": "Healthcare",
    "AKTX": "Healthcare", "ALDX": "Healthcare", "ALGS": "Healthcare", "ALLO": "Healthcare", "ALLR": "Healthcare", "ALT": "Healthcare", "ALXO": "Healthcare", "ALZN": "Healthcare",
    "AMIX": "Healthcare", "AMLX": "Healthcare", "AMN": "Healthcare", "AMPH": "Healthcare", "AMRX": "Healthcare", "ANIK": "Healthcare", "ANIP": "Healthcare", "ANRO": "Healthcare",
    "ANTX": "Healthcare", "APGE": "Healthcare", "APLE": "Healthcare", "APLS": "Healthcare", "APRE": "Healthcare", "APVO": "Healthcare", "APYX": "Healthcare", "AQST": "Healthcare",
    "ARCT": "Healthcare", "ARDT": "Healthcare", "ARQT": "Healthcare", "ARTV": "Healthcare", "ARWR": "Healthcare", "ASBP": "Healthcare", "ASND": "Healthcare", "ASTH": "Healthcare",
    "ATHE": "Healthcare", "ATOS": "Healthcare", "ATRA": "Healthcare", "ATYR": "Healthcare", "AUPH": "Healthcare", "AUTL": "Healthcare", "AVAH": "Healthcare", "AVBP": "Healthcare",
    "AVIR": "Healthcare", "AVNS": "Healthcare", "AVTX": "Healthcare", "AYTU": "Healthcare", "BBIO": "Healthcare", "BBLG": "Healthcare", "BBOT": "Healthcare", "BCAX": "Healthcare",
    "BCRX": "Healthcare", "BCTX": "Healthcare", "BCYC": "Healthcare", "BDMD": "Healthcare", "BDRX": "Healthcare", "BDTX": "Healthcare", "BHC": "Healthcare", "BJDX": "Healthcare",
    "BME": "Healthcare", "BMEZ": "Healthcare", "BMGL": "Healthcare", "BNGO": "Healthcare", "BNR": "Healthcare", "BNTC": "Healthcare", "BOLT": "Healthcare", "BRNS": "Healthcare",
    "BTAI": "Healthcare", "BTSG": "Healthcare", "BYAH": "Healthcare", "CADL": "Healthcare", "CAH": "Healthcare", "CAMP": "Healthcare", "CAPR": "Healthcare", "CBIO": "Healthcare",
    "CCCC": "Healthcare", "CDIO": "Healthcare", "CELZ": "Healthcare", "CGEM": "Healthcare", "CGON": "Healthcare", "CGTX": "Healthcare", "CHCT": "Healthcare", "CHRS": "Healthcare",
    "CLDX": "Healthcare", "CLGN": "Healthcare", "CLPT": "Healthcare", "CMMB": "Healthcare", "CMPX": "Healthcare", "CNSP": "Healthcare", "CNTA": "Healthcare", "CNTB": "Healthcare",
    "CNTX": "Healthcare", "COCH": "Healthcare", "COCP": "Healthcare", "CODX": "Healthcare", "COEP": "Healthcare", "COLL": "Healthcare", "CORT": "Healthcare", "COSM": "Healthcare",
    "COYA": "Healthcare", "CPIX": "Healthcare", "CPRX": "Healthcare", "CRBP": "Healthcare", "CRDF": "Healthcare", "CRDL": "Healthcare", "CRNX": "Healthcare", "CRVS": "Healthcare",
    "CSBR": "Healthcare", "CTMX": "Healthcare", "CTNM": "Healthcare", "CTOR": "Healthcare", "CTXR": "Healthcare", "CUE": "Healthcare", "CURX": "Healthcare", "CVKD": "Healthcare",
    "CYCN": "Healthcare", "CYH": "Healthcare", "DAWN": "Healthcare", "DCOY": "Healthcare", "DERM": "Healthcare", "DFTX": "Healthcare", "DGX": "Healthcare", "DH": "Healthcare",
    "DHC": "Healthcare", "DMAC": "Healthcare", "DMRA": "Healthcare", "DNLI": "Healthcare", "DNTH": "Healthcare", "DOC": "Healthcare", "DRH": "Healthcare", "DRIO": "Healthcare",
    "DRMA": "Healthcare", "DRTS": "Healthcare", "DSGN": "Healthcare", "DWTX": "Healthcare", "DYN": "Healthcare", "EDSA": "Healthcare", "EHC": "Healthcare", "EHTH": "Healthcare",
    "EIKN": "Healthcare", "ELAN": "Healthcare", "ELDN": "Healthcare", "ELTX": "Healthcare", "ELV": "Healthcare", "ENTA": "Healthcare", "EPRX": "Healthcare", "ERNA": "Healthcare",
    "ESLA": "Healthcare", "ESPR": "Healthcare", "ETON": "Healthcare", "EUDA": "Healthcare", "EVH": "Healthcare", "EWTX": "Healthcare", "FBIO": "Healthcare", "FBLG": "Healthcare",
    "FDMT": "Healthcare", "FEED": "Healthcare", "FENC": "Healthcare", "FHTX": "Healthcare", "FLNA": "Healthcare", "FOLD": "Healthcare", "FULC": "Healthcare", "GALT": "Healthcare",
    "GANX": "Healthcare", "GDTC": "Healthcare", "GEHC": "Healthcare", "GH": "Healthcare", "GLMD": "Healthcare", "GLUE": "Healthcare", "GMED": "Healthcare", "GOCO": "Healthcare",
    "GPCR": "Healthcare", "GRCE": "Healthcare", "GRDN": "Healthcare", "GRX": "Healthcare", "GTBP": "Healthcare", "GUTS": "Healthcare", "GYRE": "Healthcare", "HALO": "Healthcare",
    "HAO": "Healthcare", "HCAT": "Healthcare", "HCSG": "Healthcare", "HCTI": "Healthcare", "HCWB": "Healthcare", "HIT": "Healthcare", "HKPD": "Healthcare", "HNGE": "Healthcare",
    "HOTH": "Healthcare", "HOWL": "Healthcare", "HQH": "Healthcare", "HQY": "Healthcare", "HRTX": "Healthcare", "IBRX": "Healthcare", "ICCM": "Healthcare", "ICU": "Healthcare",
    "ICUI": "Healthcare", "IKT": "Healthcare", "IMDX": "Healthcare", "IMMX": "Healthcare", "IMNM": "Healthcare", "IMRX": "Healthcare", "IMUX": "Healthcare", "INDP": "Healthcare",
    "INDV": "Healthcare", "INKT": "Healthcare", "INM": "Healthcare", "INO": "Healthcare", "INSP": "Healthcare", "INTS": "Healthcare", "IOBT": "Healthcare", "IONS": "Healthcare",
    "IPHA": "Healthcare", "IPSC": "Healthcare", "IRWD": "Healthcare", "ITRM": "Healthcare", "IXHL": "Healthcare", "JAGX": "Healthcare", "JANX": "Healthcare", "JSPR": "Healthcare",
    "JUNS": "Healthcare", "KALV": "Healthcare", "KLRS": "Healthcare", "KMTS": "Healthcare", "KNSA": "Healthcare", "KPRX": "Healthcare", "KPTI": "Healthcare", "KRMD": "Healthcare",
    "KTTA": "Healthcare", "KURA": "Healthcare", "KYTX": "Healthcare", "KZIA": "Healthcare", "LBRX": "Healthcare", "LEGN": "Healthcare", "LENZ": "Healthcare", "LFCR": "Healthcare",
    "LFST": "Healthcare", "LGND": "Healthcare", "LIMN": "Healthcare", "LIXT": "Healthcare", "LRMR": "Healthcare", "LSTA": "Healthcare", "LTRN": "Healthcare", "LUCD": "Healthcare",
    "LXEO": "Healthcare", "LXRX": "Healthcare", "LYEL": "Healthcare", "MAZE": "Healthcare", "MBOT": "Healthcare", "MBRX": "Healthcare", "MCRB": "Healthcare", "MD": "Healthcare",
    "MDCX": "Healthcare", "MDGL": "Healthcare", "MDXH": "Healthcare", "MENS": "Healthcare", "MGX": "Healthcare", "MIRA": "Healthcare", "MIRM": "Healthcare", "MIST": "Healthcare",
    "MLTX": "Healthcare", "MLYS": "Healthcare", "MNDR": "Healthcare", "MNPR": "Healthcare", "MODD": "Healthcare", "MPLT": "Healthcare", "MPT": "Healthcare", "MREO": "Healthcare",
    "MRKR": "Healthcare", "MRM": "Healthcare", "NAMS": "Healthcare", "NAUT": "Healthcare", "NBIX": "Healthcare", "NEO": "Healthcare", "NERV": "Healthcare", "NEUP": "Healthcare",
    "NGEN": "Healthcare", "NGNE": "Healthcare", "NHI": "Healthcare", "NHTC": "Healthcare", "NKTR": "Healthcare", "NMRA": "Healthcare", "NMTC": "Healthcare", "NNNN": "Healthcare",
    "NPCE": "Healthcare", "NRIX": "Healthcare", "NRSN": "Healthcare", "NRXP": "Healthcare", "NUTX": "Healthcare", "NVCT": "Healthcare", "NVNO": "Healthcare", "OCC": "Healthcare",
    "OCUL": "Healthcare", "OFIX": "Healthcare", "OHI": "Healthcare", "OKUR": "Healthcare", "OKYO": "Healthcare", "OLMA": "Healthcare", "OM": "Healthcare", "OMDA": "Healthcare",
    "ONCY": "Healthcare", "OPCH": "Healthcare", "OPK": "Healthcare", "OPTX": "Healthcare", "ORIC": "Healthcare", "ORKA": "Healthcare", "ORMP": "Healthcare", "OSCR": "Healthcare",
    "OTLK": "Healthcare", "OVID": "Healthcare", "PARK": "Healthcare", "PBH": "Healthcare", "PBM": "Healthcare", "PBYI": "Healthcare", "PCSA": "Healthcare", "PDSB": "Healthcare",
    "PHAT": "Healthcare", "PHIO": "Healthcare", "PIII": "Healthcare", "PLRX": "Healthcare", "PLYX": "Healthcare", "PMCB": "Healthcare", "PMN": "Healthcare", "PMVP": "Healthcare",
    "POCI": "Healthcare", "PPBT": "Healthcare", "PPCB": "Healthcare", "PROF": "Healthcare", "PRQR": "Healthcare", "PRSU": "Healthcare", "PRTC": "Healthcare", "PRVA": "Healthcare",
    "PSTV": "Healthcare", "PTCT": "Healthcare", "PTGX": "Healthcare", "PVLA": "Healthcare", "PYXS": "Healthcare", "QNCX": "Healthcare", "QNRX": "Healthcare", "QNTM": "Healthcare",
    "RANI": "Healthcare", "RAPP": "Healthcare", "RCEL": "Healthcare", "RCKT": "Healthcare", "RDHL": "Healthcare", "RICK": "Healthcare", "RIGL": "Healthcare", "RLAY": "Healthcare",
    "RLMD": "Healthcare", "RMTI": "Healthcare", "RNA": "Healthcare", "RNAC": "Healthcare", "RNAZ": "Healthcare", "RNTX": "Healthcare", "RPRX": "Healthcare", "RVPH": "Healthcare",
    "RYTM": "Healthcare", "SABS": "Healthcare", "SANA": "Healthcare", "SBC": "Healthcare", "SBFM": "Healthcare", "SBRA": "Healthcare", "SCNI": "Healthcare", "SEM": "Healthcare",
    "SGP": "Healthcare", "SHC": "Healthcare", "SHPH": "Healthcare", "SILO": "Healthcare", "SION": "Healthcare", "SKIN": "Healthcare", "SLN": "Healthcare", "SLNO": "Healthcare",
    "SLXN": "Healthcare", "SMMT": "Healthcare", "SNDX": "Healthcare", "SNOA": "Healthcare", "SNSE": "Healthcare", "SNWV": "Healthcare", "SPRO": "Healthcare", "SPRY": "Healthcare",
    "SRTA": "Healthcare", "SRTS": "Healthcare", "STAA": "Healthcare", "STIM": "Healthcare", "STKS": "Healthcare", "STOK": "Healthcare", "STRO": "Healthcare", "SXTC": "Healthcare",
    "SXTP": "Healthcare", "SYRE": "Healthcare", "TARA": "Healthcare", "TARS": "Healthcare", "TBPH": "Healthcare", "TCRT": "Healthcare", "TCRX": "Healthcare", "TECX": "Healthcare",
    "TELO": "Healthcare", "TENX": "Healthcare", "TERN": "Healthcare", "TGTX": "Healthcare", "TH": "Healthcare", "THC": "Healthcare", "THQ": "Healthcare", "THW": "Healthcare",
    "TIVC": "Healthcare", "TLX": "Healthcare", "TMCI": "Healthcare", "TNGX": "Healthcare", "TNON": "Healthcare", "TNXP": "Healthcare", "TNYA": "Healthcare", "TOI": "Healthcare",
    "TRDA": "Healthcare", "TRIB": "Healthcare", "TRVI": "Healthcare", "TTRX": "Healthcare", "TVRD": "Healthcare", "TVTX": "Healthcare", "TXG": "Healthcare", "TXMD": "Healthcare",
    "UHS": "Healthcare", "UHT": "Healthcare", "UNCY": "Healthcare", "UPC": "Healthcare", "URGN": "Healthcare", "USNA": "Healthcare", "UTHR": "Healthcare", "UTMD": "Healthcare",
    "VANI": "Healthcare", "VIR": "Healthcare", "VKTX": "Healthcare", "VMD": "Healthcare", "VNDA": "Healthcare", "VOR": "Healthcare", "VRCA": "Healthcare", "VRDN": "Healthcare",
    "VSEE": "Healthcare", "VTGN": "Healthcare", "VTVT": "Healthcare", "VVOS": "Healthcare", "VYGR": "Healthcare", "VYNE": "Healthcare", "WGRX": "Healthcare", "WHWK": "Healthcare",
    "WOK": "Healthcare", "WOOF": "Healthcare", "WST": "Healthcare", "XBIT": "Healthcare", "XERS": "Healthcare", "XFOR": "Healthcare", "XLO": "Healthcare", "XRTX": "Healthcare",
    "XTLB": "Healthcare", "ZBIO": "Healthcare", "ZJYL": "Healthcare", "ZNTL": "Healthcare", "ZVRA": "Healthcare", "ZYBT": "Healthcare",
    "CDP": "Industrial", "CNH": "Industrial", "CSW": "Industrial", "CVLG": "Industrial", "DFNS": "Industrial", "ECG": "Industrial", "GIC": "Industrial", "GTES": "Industrial",
    "GXO": "Industrial", "IDE": "Industrial", "ILPT": "Industrial", "JBHT": "Industrial", "JFB": "Industrial", "JYD": "Industrial", "KNX": "Industrial", "KTOS": "Industrial",
    "LOMA": "Industrial", "MDV": "Industrial", "MEC": "Industrial", "MOD": "Industrial", "MRTN": "Industrial", "MSM": "Industrial", "NOA": "Industrial", "ONEG": "Industrial",
    "PANL": "Industrial", "ROAD": "Industrial", "SLGB": "Industrial", "SNCY": "Industrial", "SSD": "Industrial", "ULH": "Industrial", "VIA": "Industrial", "VIRC": "Industrial",
    "ZJK": "Industrial",
    "AAUC": "Materials", "AG": "Materials", "AGI": "Materials", "AMBP": "Materials", "AMR": "Materials", "AQMS": "Materials", "ASA": "Materials", "ASTL": "Materials",
    "ATCX": "Materials", "AU": "Materials", "AUGO": "Materials", "BGL": "Materials", "BMM": "Materials", "CENX": "Materials", "CGAU": "Materials", "CLW": "Materials",
    "CMC": "Materials", "CMP": "Materials", "CRML": "Materials", "ERO": "Materials", "EXK": "Materials", "FLXS": "Materials", "GDHG": "Materials", "GNT": "Materials",
    "GPK": "Materials", "HBM": "Materials", "IAG": "Materials", "IIIN": "Materials", "IP": "Materials", "KALU": "Materials", "KGC": "Materials", "KMT": "Materials",
    "KRT": "Materials", "LZM": "Materials", "MTUS": "Materials", "NAMM": "Materials", "NVA": "Materials", "PAAS": "Materials", "PKG": "Materials", "RGLD": "Materials",
    "SA": "Materials", "SBXD": "Materials", "SBXE": "Materials", "SCCO": "Materials", "TFPM": "Materials", "TMC": "Materials", "USAU": "Materials", "USLM": "Materials",
    "WS": "Materials",
    "AACG": "Other", "AAME": "Other", "AAOI": "Other", "AAT": "Other", "AB": "Other", "ABLV": "Other", "ABM": "Other", "ABSI": "Other",
    "ABTC": "Other", "ABTS": "Other", "ABVX": "Other", "ABX": "Other", "ACA": "Other", "ACB": "Other", "ACCL": "Other", "ACCO": "Other",
    "ACDC": "Other", "ACET": "Other", "ACHV": "Other", "ACI": "Other", "ACLX": "Other", "ACM": "Other", "ACNB": "Other", "ACNT": "Other",
    "ACOG": "Other", "ACON": "Other", "ACT": "Other", "ACTG": "Other", "ACV": "Other", "ADAG": "Other", "ADAM": "Other", "ADBE": "Other",
    "ADEA": "Other", "ADNT": "Other", "ADT": "Other", "ADTX": "Other", "ADV": "Other", "ADVB": "Other", "ADX": "Other", "AEBI": "Other",
    "AEE": "Other", "AEG": "Other", "AEHL": "Other", "AEHR": "Other", "AEI": "Other", "AER": "Other", "AEYE": "Other", "AFB": "Other",
    "AFBI": "Other", "AFRI": "Other", "AGBK": "Other", "AGCC": "Other", "AGD": "Other", "AGH": "Other", "AGMH": "Other", "AGO": "Other",
    "AGRO": "Other", "AGRZ": "Other", "AGX": "Other", "AGYS": "Other", "AHMA": "Other", "AIDX": "Other", "AIFU": "Other", "AIMD": "Other",
    "AIN": "Other", "AIOS": "Other", "AIP": "Other", "AIR": "Other", "AIRE": "Other", "AIRG": "Other", "AIRO": "Other", "AIRT": "Other",
    "AIXI": "Other", "AJG": "Other", "AKA": "Other", "AKAN": "Other", "AKO-A": "Other", "AKO-B": "Other", "AL": "Other", "ALAB": "Other",
    "ALBT": "Other", "ALC": "Other", "ALCO": "Other", "ALEC": "Other", "ALG": "Other", "ALH": "Other", "ALIT": "Other", "ALLT": "Other",
    "ALM": "Other", "ALMS": "Other", "ALMU": "Other", "ALNT": "Other", "ALOT": "Other", "ALOV": "Other", "ALOY": "Other", "ALPS": "Other",
    "ALSN": "Other", "ALTG": "Other", "ALTI": "Other", "ALTO": "Other", "ALTS": "Other", "ALVO": "Other", "ALX": "Other", "AM": "Other",
    "AMBQ": "Other", "AMBR": "Other", "AMCR": "Other", "AMCX": "Other", "AMG": "Other", "AMOD": "Other", "AMPG": "Other", "AMPL": "Other",
    "AMRC": "Other", "AMRN": "Other", "AMRZ": "Other", "AMSC": "Other", "AMST": "Other", "AMTM": "Other", "AMTX": "Other", "AMWD": "Other",
    "ANAB": "Other", "ANDE": "Other", "ANDG": "Other", "ANGH": "Other", "ANGO": "Other", "ANGX": "Other", "ANIX": "Other", "ANNA": "Other",
    "ANNX": "Other", "ANPA": "Other", "ANTA": "Other", "ANVS": "Other", "ANY": "Other", "AOD": "Other", "AON": "Other", "AORT": "Other",
    "AOS": "Other", "AOUT": "Other", "AP": "Other", "APEI": "Other", "APG": "Other", "APH": "Other", "API": "Other", "APLM": "Other",
    "APM": "Other", "APO": "Other", "APPF": "Other", "APWC": "Other", "APXT": "Other", "AR": "Other", "ARAY": "Other", "ARBB": "Other",
    "ARCB": "Other", "ARCO": "Other", "AREC": "Other", "ARES": "Other", "ARGX": "Other", "ARMK": "Other", "AROC": "Other", "ARQ": "Other",
    "ARTL": "Other", "ARVN": "Other", "ARW": "Other", "ARX": "Other", "AS": "Other", "ASB": "Other", "ASC": "Other", "ASG": "Other",
    "ASGI": "Other", "ASGN": "Other", "ASH": "Other", "ASIX": "Other", "ASLE": "Other", "ASMB": "Other", "ASML": "Other", "ASNS": "Other",
    "ASO": "Other", "ASPI": "Other", "ASPN": "Other", "ASPS": "Other", "ASPSZ": "Other", "ASR": "Other", "ASRT": "Other", "ASST": "Other",
    "ASTC": "Other", "ASYS": "Other", "ATAI": "Other", "ATAT": "Other", "ATEC": "Other", "ATEN": "Other", "ATER": "Other", "ATEX": "Other",
    "ATHR": "Other", "ATKR": "Other", "ATLC": "Other", "ATLN": "Other", "ATLO": "Other", "ATLX": "Other", "ATNI": "Other", "ATOM": "Other",
    "ATPC": "Other", "ATR": "Other", "ATRO": "Other", "ATS": "Other", "ATXG": "Other", "AUDC": "Other", "AUID": "Other", "AUNA": "Other",
    "AURA": "Other", "AURE": "Other", "AUUD": "Other", "AVA": "Other", "AVAV": "Other", "AVD": "Other", "AVK": "Other", "AVNT": "Other",
    "AVO": "Other", "AVPT": "Other", "AVT": "Other", "AVTR": "Other", "AVXL": "Other", "AVY": "Other", "AWF": "Other", "AWI": "Other",
    "AWP": "Other", "AWRE": "Other", "AXG": "Other", "AXGN": "Other", "AXR": "Other", "AXTA": "Other", "AXTI": "Other", "AYI": "Other",
    "AZ": "Other", "AZN": "Other", "AZTA": "Other", "AZZ": "Other", "BAH": "Other", "BAK": "Other", "BALL": "Other", "BALY": "Other",
    "BANC": "Other", "BAND": "Other", "BANF": "Other", "BANL": "Other", "BAOS": "Other", "BAP": "Other", "BATRA": "Other", "BATRK": "Other",
    "BB": "Other", "BBAI": "Other", "BBAR": "Other", "BBBY": "Other", "BBCP": "Other", "BBDC": "Other", "BBGI": "Other", "BBN": "Other",
    "BBNX": "Other", "BBSI": "Other", "BBU": "Other", "BBUC": "Other", "BBVA": "Other", "BBW": "Other", "BBWI": "Other", "BC": "Other",
    "BCAB": "Other", "BCC": "Other", "BCDA": "Other", "BCE": "Other", "BCH": "Other", "BCML": "Other", "BCO": "Other", "BCPC": "Other",
    "BCS": "Other", "BCX": "Other", "BDC": "Other", "BDCI": "Other", "BDJ": "Other", "BDSX": "Other", "BEBE": "Other", "BEEM": "Other",
    "BEEP": "Other", "BELFA": "Other", "BELFB": "Other", "BENF": "Other", "BEPH": "Other", "BEPI": "Other", "BEPJ": "Other", "BETR": "Other",
    "BF-A": "Other", "BF-B": "Other", "BFLY": "Other", "BFRI": "Other", "BFS": "Other", "BFST": "Other", "BGC": "Other", "BGH": "Other",
    "BGLC": "Other", "BGM": "Other", "BGMS": "Other", "BGSF": "Other", "BGSI": "Other", "BGT": "Other", "BGY": "Other", "BH": "Other",
    "BHE": "Other", "BHK": "Other", "BHST": "Other", "BHV": "Other", "BHVN": "Other", "BIDU": "Other", "BILI": "Other", "BIO": "Other",
    "BIO-B": "Other", "BIOA": "Other", "BIOX": "Other", "BIP": "Other", "BIPC": "Other", "BIPI": "Other", "BIRD": "Other", "BIT": "Other",
    "BITF": "Other", "BIVI": "Other", "BIYA": "Other", "BKD": "Other", "BKE": "Other", "BKH": "Other", "BKKT": "Other", "BKT": "Other",
    "BKV": "Other", "BKYI": "Other", "BL": "Other", "BLBD": "Other", "BLCO": "Other", "BLD": "Other", "BLFS": "Other", "BLIV": "Other",
    "BLKB": "Other", "BLLN": "Other", "BLMN": "Other", "BLND": "Other", "BLNE": "Other", "BLRX": "Other", "BLSH": "Other", "BLTE": "Other",
    "BLW": "Other", "BLX": "Other", "BLZE": "Other", "BMA": "Other", "BMBL": "Other", "BMEA": "Other", "BMHL": "Other", "BMI": "Other",
    "BMN": "Other", "BMR": "Other", "BMRA": "Other", "BN": "Other", "BNAI": "Other", "BNBX": "Other", "BNC": "Other", "BNED": "Other",
    "BNJ": "Other", "BNKK": "Other", "BNL": "Other", "BNT": "Other", "BNTX": "Other", "BOBS": "Other", "BOC": "Other", "BODI": "Other",
    "BOE": "Other", "BOLD": "Other", "BON": "Other", "BOOM": "Other", "BORR": "Other", "BOSC": "Other", "BOW": "Other", "BOXL": "Other",
    "BP": "Other", "BPOP": "Other", "BRAI": "Other", "BRBI": "Other", "BRBR": "Other", "BRC": "Other", "BRCB": "Other", "BRCC": "Other",
    "BRK-A": "Other", "BRKR": "Other", "BRLT": "Other", "BRO": "Other", "BROS": "Other", "BRSL": "Other", "BRT": "Other", "BRTX": "Other",
    "BSAC": "Other", "BSET": "Other", "BSL": "Other", "BSY": "Other", "BTBD": "Other", "BTCS": "Other", "BTGO": "Other", "BTI": "Other",
    "BTM": "Other", "BTMD": "Other", "BTOC": "Other", "BTOG": "Other", "BTT": "Other", "BTTC": "Other", "BUD": "Other", "BULL": "Other",
    "BURL": "Other", "BUSE": "Other", "BUUU": "Other", "BV": "Other", "BVC": "Other", "BVS": "Other", "BW": "Other", "BWA": "Other",
    "BWAY": "Other", "BWEN": "Other", "BWG": "Other", "BWLP": "Other", "BWMN": "Other", "BWMX": "Other", "BX": "Other", "BXC": "Other",
    "BXMX": "Other", "BYND": "Other", "BYSI": "Other", "BZ": "Other", "BZAI": "Other", "BZFD": "Other", "BZH": "Other", "BZUN": "Other",
    "CAAP": "Other", "CABO": "Other", "CABR": "Other", "CAC": "Other", "CACI": "Other", "CAE": "Other", "CAEP": "Other", "CAF": "Other",
    "CAI": "Other", "CAL": "Other", "CALC": "Other", "CALY": "Other", "CAMT": "Other", "CAN": "Other", "CANG": "Other", "CAPS": "Other",
    "CAPT": "Other", "CAR": "Other", "CARG": "Other", "CARL": "Other", "CARS": "Other", "CART": "Other", "CASS": "Other", "CAST": "Other",
    "CAVA": "Other", "CBC": "Other", "CBK": "Other", "CBL": "Other", "CBLL": "Other", "CBRE": "Other", "CBRL": "Other", "CBSH": "Other",
    "CBT": "Other", "CBUS": "Other", "CBZ": "Other", "CC": "Other", "CCC": "Other", "CCD": "Other", "CCEP": "Other", "CCG": "Other",
    "CCHH": "Other", "CCJ": "Other", "CCK": "Other", "CCO": "Other", "CCOI": "Other", "CCS": "Other", "CCU": "Other", "CCZ": "Other",
    "CDLX": "Other", "CDNA": "Other", "CDNL": "Other", "CDRE": "Other", "CDRO": "Other", "CDT": "Other", "CDW": "Other", "CDXS": "Other",
    "CDZI": "Other", "CDZIP": "Other", "CECO": "Other", "CEE": "Other", "CELC": "Other", "CELH": "Other", "CELU": "Other", "CENN": "Other",
    "CENT": "Other", "CENTA": "Other", "CEPF": "Other", "CEPO": "Other", "CEPS": "Other", "CEPT": "Other", "CEPV": "Other", "CERS": "Other",
    "CETX": "Other", "CEVA": "Other", "CFND": "Other", "CG": "Other", "CGC": "Other", "CGCT": "Other", "CGEN": "Other", "CGO": "Other",
    "CHA": "Other", "CHCI": "Other", "CHCO": "Other", "CHDN": "Other", "CHE": "Other", "CHEF": "Other", "CHGG": "Other", "CHI": "Other",
    "CHNR": "Other", "CHR": "Other", "CHSN": "Other", "CHT": "Other", "CHW": "Other", "CHY": "Other", "CIA": "Other", "CIF": "Other",
    "CIGI": "Other", "CIGL": "Other", "CII": "Other", "CIIT": "Other", "CING": "Other", "CINT": "Other", "CISO": "Other", "CISS": "Other",
    "CIVB": "Other", "CJMB": "Other", "CLAR": "Other", "CLB": "Other", "CLBT": "Other", "CLF": "Other", "CLFD": "Other", "CLH": "Other",
    "CLIK": "Other", "CLLS": "Other", "CLMB": "Other", "CLMT": "Other", "CLNN": "Other", "CLPS": "Other", "CLRB": "Other", "CLRO": "Other",
    "CLS": "Other", "CLVT": "Other", "CLWT": "Other", "CLYM": "Other", "CMBM": "Other", "CMBT": "Other", "CMCO": "Other", "CMCT": "Other",
    "CMDB": "Other", "CMI": "Other", "CMND": "Other", "CMPR": "Other", "CMPS": "Other", "CMRC": "Other", "CMRE": "Other", "CMTL": "Other",
    "CMU": "Other", "CNCK": "Other", "CNDT": "Other", "CNK": "Other", "CNM": "Other", "CNMD": "Other", "CNNE": "Other", "CNQ": "Other",
    "CNR": "Other", "CNS": "Other", "CNTN": "Other", "CNTY": "Other", "CNVS": "Other", "CNX": "Other", "CNXN": "Other", "COCO": "Other",
    "CODA": "Other", "COGT": "Other", "COHR": "Other", "COKE": "Other", "COLM": "Other", "COMP": "Other", "CON": "Other", "COO": "Other",
    "COOK": "Other", "COR": "Other", "CORZ": "Other", "CORZZ": "Other", "COSO": "Other", "COUR": "Other", "CPA": "Other", "CPAY": "Other",
    "CPBI": "Other", "CPHC": "Other", "CPNG": "Other", "CPOP": "Other", "CPS": "Other", "CPSS": "Other", "CPZ": "Other", "CR": "Other",
    "CRAI": "Other", "CRBU": "Other", "CRC": "Other", "CRCT": "Other", "CRD-A": "Other", "CRD-B": "Other", "CRE": "Other", "CRESY": "Other",
    "CREX": "Other", "CRGO": "Other", "CRH": "Other", "CRI": "Other", "CRIS": "Other", "CRK": "Other", "CRMD": "Other", "CRMT": "Other",
    "CRNC": "Other", "CRNT": "Other", "CRON": "Other", "CRT": "Other", "CRTO": "Other", "CRUS": "Other", "CRVL": "Other", "CRVO": "Other",
    "CRWS": "Other", "CRWV": "Other", "CSAN": "Other", "CSGP": "Other", "CSGS": "Other", "CSL": "Other", "CSPI": "Other", "CSQ": "Other",
    "CSTE": "Other", "CSTL": "Other", "CSTM": "Other", "CSV": "Other", "CTEV": "Other", "CTKB": "Other", "CTLP": "Other", "CTNT": "Other",
    "CTOS": "Other", "CTRI": "Other", "CTRM": "Other", "CTRN": "Other", "CTS": "Other", "CTSO": "Other", "CTW": "Other", "CUB": "Other",
    "CUK": "Other", "CULP": "Other", "CUPR": "Other", "CURB": "Other", "CURI": "Other", "CURR": "Other", "CURV": "Other", "CUZ": "Other",
    "CV": "Other", "CVEO": "Other", "CVGI": "Other", "CVRX": "Other", "CVSA": "Other", "CVV": "Other", "CWBC": "Other", "CWD": "Other",
    "CWH": "Other", "CWK": "Other", "CX": "Other", "CXAI": "Other", "CXDO": "Other", "CXE": "Other", "CXM": "Other", "CXT": "Other",
    "CXW": "Other", "CYCU": "Other", "CYN": "Other", "CYRX": "Other", "CYTK": "Other", "CZNC": "Other", "DAC": "Other", "DAIC": "Other",
    "DAKT": "Other", "DAN": "Other", "DAR": "Other", "DARE": "Other", "DASH": "Other", "DAVE": "Other", "DBD": "Other", "DBI": "Other",
    "DBX": "Other", "DCBO": "Other", "DCGO": "Other", "DCH": "Other", "DCI": "Other", "DCO": "Other", "DCOM": "Other", "DCTH": "Other",
    "DDD": "Other", "DDI": "Other", "DDS": "Other", "DEA": "Other", "DEI": "Other", "DEO": "Other", "DEVS": "Other", "DFDV": "Other",
    "DFH": "Other", "DGICA": "Other", "DGICB": "Other", "DGII": "Other", "DGNX": "Other", "DHF": "Other", "DHT": "Other", "DHX": "Other",
    "DIAX": "Other", "DIBS": "Other", "DIN": "Other", "DINO": "Other", "DJCO": "Other", "DK": "Other", "DKI": "Other", "DKS": "Other",
    "DLB": "Other", "DLHC": "Other", "DLO": "Other", "DLTH": "Other", "DLX": "Other", "DLXY": "Other", "DLY": "Other", "DMA": "Other",
    "DMB": "Other", "DMRC": "Other", "DNA": "Other", "DNMX": "Other", "DNP": "Other", "DOCU": "Other", "DOGZ": "Other", "DOLE": "Other",
    "DOMH": "Other", "DOO": "Other", "DORM": "Other", "DOUG": "Other", "DOX": "Other", "DOYU": "Other", "DPRO": "Other", "DPZ": "Other",
    "DRS": "Other", "DRUG": "Other", "DSGR": "Other", "DSGX": "Other", "DSL": "Other", "DSM": "Other", "DSU": "Other", "DSWL": "Other",
    "DSX": "Other", "DT": "Other", "DTCK": "Other", "DTF": "Other", "DTI": "Other", "DTIL": "Other", "DTM": "Other", "DUO": "Other",
    "DUOL": "Other", "DV": "Other", "DWSN": "Other", "DXLG": "Other", "DXPE": "Other", "DXR": "Other", "DXST": "Other", "DXYZ": "Other",
    "DYAI": "Other", "E": "Other", "EAF": "Other", "EAT": "Other", "EBF": "Other", "EBON": "Other", "EBS": "Other", "ECO": "Other",
    "ECOR": "Other", "ECVT": "Other", "ECX": "Other", "ED": "Other", "EDAP": "Other", "EDBL": "Other", "EDD": "Other", "EDF": "Other",
    "EDRY": "Other", "EDUC": "Other", "EEA": "Other", "EEFT": "Other", "EEIQ": "Other", "EEX": "Other", "EFR": "Other", "EFT": "Other",
    "EFX": "Other", "EFXT": "Other", "EG": "Other", "EGAN": "Other", "EGP": "Other", "EH": "Other", "EHAB": "Other", "EHGO": "Other",
    "EHI": "Other", "EHLD": "Other", "EIC": "Other", "EIG": "Other", "EIX": "Other", "EJH": "Other", "EKSO": "Other", "ELAB": "Other",
    "ELBM": "Other", "ELE": "Other", "ELME": "Other", "ELOG": "Other", "ELS": "Other", "ELSE": "Other", "ELTK": "Other", "ELUT": "Other",
    "ELVA": "Other", "ELVR": "Other", "ELWT": "Other", "EM": "Other", "EMA": "Other", "EMBC": "Other", "EMBJ": "Other", "EMD": "Other",
    "EME": "Other", "EMF": "Other", "EML": "Other", "ENB": "Other", "ENGN": "Other", "ENLV": "Other", "ENOV": "Other", "ENR": "Other",
    "ENS": "Other", "ENSC": "Other", "ENTG": "Other", "ENTX": "Other", "ENVA": "Other", "ENVB": "Other", "ENVX": "Other", "EOD": "Other",
    "EOI": "Other", "EOLS": "Other", "EOS": "Other", "EOT": "Other", "EPAC": "Other", "EPC": "Other", "EPD": "Other", "EPR": "Other",
    "EPSM": "Other", "EQ": "Other", "EQBK": "Other", "EQH": "Other", "EQNR": "Other", "EQPT": "Other", "EQS": "Other", "EQT": "Other",
    "ERAS": "Other", "ERIC": "Other", "ERIE": "Other", "ESCA": "Other", "ESEA": "Other", "ESI": "Other", "ESLT": "Other", "ESTA": "Other",
    "ETB": "Other", "ETD": "Other", "ETG": "Other", "ETHB": "Other", "ETHM": "Other", "ETJ": "Other", "ETO": "Other", "ETOR": "Other",
    "ETS": "Other", "ETV": "Other", "ETW": "Other", "ETX": "Other", "ETY": "Other", "EVAX": "Other", "EVC": "Other", "EVCM": "Other",
    "EVER": "Other", "EVEX": "Other", "EVF": "Other", "EVG": "Other", "EVGN": "Other", "EVMN": "Other", "EVN": "Other", "EVO": "Other",
    "EVR": "Other", "EVT": "Other", "EVTC": "Other", "EVTV": "Other", "EWCZ": "Other", "EXEL": "Other", "EXFY": "Other", "EXG": "Other",
    "EXLS": "Other", "EXOZ": "Other", "EXP": "Other", "EYE": "Other", "EYPT": "Other", "EZRA": "Other", "FA": "Other", "FAF": "Other",
    "FAMI": "Other", "FARM": "Other", "FATN": "Other", "FBGL": "Other", "FBRX": "Other", "FBYD": "Other", "FC": "Other", "FCCO": "Other",
    "FCFS": "Other", "FCHL": "Other", "FCN": "Other", "FCNCA": "Other", "FCNCP": "Other", "FCT": "Other", "FCUV": "Other", "FDP": "Other",
    "FDS": "Other", "FEAM": "Other", "FEBO": "Other", "FEIM": "Other", "FEMY": "Other", "FER": "Other", "FERG": "Other", "FFA": "Other",
    "FFWM": "Other", "FG": "Other", "FGBI": "Other", "FGI": "Other", "FGL": "Other", "FGMC": "Other", "FGNX": "Other", "FHB": "Other",
    "FHI": "Other", "FICO": "Other", "FIEE": "Other", "FIG": "Other", "FISV": "Other", "FIVN": "Other", "FIX": "Other", "FKWL": "Other",
    "FLC": "Other", "FLD": "Other", "FLEX": "Other", "FLGT": "Other", "FLL": "Other", "FLNG": "Other", "FLNT": "Other", "FLOC": "Other",
    "FLR": "Other", "FLS": "Other", "FLWS": "Other", "FLX": "Other", "FLYE": "Other", "FMBH": "Other", "FMFC": "Other", "FMN": "Other",
    "FMNB": "Other", "FMX": "Other", "FN": "Other", "FNB": "Other", "FND": "Other", "FNGR": "Other", "FNKO": "Other", "FNV": "Other",
    "FOA": "Other", "FOF": "Other", "FONR": "Other", "FOR": "Other", "FORA": "Other", "FORR": "Other", "FORTY": "Other", "FOSL": "Other",
    "FOUR": "Other", "FOXF": "Other", "FOXX": "Other", "FPH": "Other", "FPI": "Other", "FRA": "Other", "FRD": "Other", "FRHC": "Other",
    "FRME": "Other", "FRMEP": "Other", "FRMI": "Other", "FRMM": "Other", "FRO": "Other", "FROG": "Other", "FRPH": "Other", "FRPT": "Other",
    "FSLY": "Other", "FSS": "Other", "FSTR": "Other", "FSV": "Other", "FT": "Other", "FTDR": "Other", "FTFT": "Other", "FTHM": "Other",
    "FTHY": "Other", "FTI": "Other", "FTK": "Other", "FTLF": "Other", "FTRE": "Other", "FTRK": "Other", "FTS": "Other", "FTV": "Other",
    "FTW": "Other", "FUBO": "Other", "FUFU": "Other", "FUL": "Other", "FUNC": "Other", "FUND": "Other", "FUSB": "Other", "FUSE": "Other",
    "FUTU": "Other", "FVRR": "Other", "FWDI": "Other", "FWONA": "Other", "FWONK": "Other", "FWRD": "Other", "FXNC": "Other", "GAB": "Other",
    "GAIA": "Other", "GAMB": "Other", "GAME": "Other", "GAP": "Other", "GAUZ": "Other", "GBLI": "Other", "GCL": "Other", "GCMG": "Other",
    "GCO": "Other", "GCTK": "Other", "GCV": "Other", "GDC": "Other", "GDEV": "Other", "GDL": "Other", "GDO": "Other", "GDOT": "Other",
    "GDRX": "Other", "GDS": "Other", "GDV": "Other", "GDYN": "Other", "GEF": "Other", "GEG": "Other", "GELS": "Other", "GENB": "Other",
    "GETY": "Other", "GEV": "Other", "GEVO": "Other", "GF": "Other", "GFL": "Other", "GFR": "Other", "GFS": "Other", "GGAL": "Other",
    "GGB": "Other", "GGR": "Other", "GGRP": "Other", "GGT": "Other", "GGZ": "Other", "GHC": "Other", "GHM": "Other", "GHRS": "Other",
    "GHY": "Other", "GIB": "Other", "GIBO": "Other", "GIFT": "Other", "GIGM": "Other", "GIL": "Other", "GILT": "Other", "GIPR": "Other",
    "GJH": "Other", "GJO": "Other", "GJP": "Other", "GJR": "Other", "GJT": "Other", "GLBS": "Other", "GLE": "Other", "GLIBA": "Other",
    "GLIBK": "Other", "GLNG": "Other", "GLOB": "Other", "GLOO": "Other", "GLPG": "Other", "GLSI": "Other", "GLXG": "Other", "GMAB": "Other",
    "GME": "Other", "GMHS": "Other", "GNK": "Other", "GNL": "Other", "GNLN": "Other", "GNLX": "Other", "GNPX": "Other", "GNSS": "Other",
    "GNTA": "Other", "GNTX": "Other", "GO": "Other", "GOAI": "Other", "GOF": "Other", "GOGO": "Other", "GOLF": "Other", "GOOS": "Other",
    "GOSS": "Other", "GOVX": "Other", "GPC": "Other", "GPGI": "Other", "GPRE": "Other", "GPRK": "Other", "GPRO": "Other", "GRAB": "Other",
    "GRAL": "Other", "GRAN": "Other", "GRC": "Other", "GREE": "Other", "GRFS": "Other", "GRI": "Other", "GRML": "Other", "GRMN": "Other",
    "GRND": "Other", "GRNT": "Other", "GROV": "Other", "GRPN": "Other", "GRVY": "Other", "GRWG": "Other", "GSAT": "Other", "GSL": "Other",
    "GSM": "Other", "GT": "Other", "GTEN": "Other", "GTN": "Other", "GTX": "Other", "GUG": "Other", "GURE": "Other", "GV": "Other",
    "GVH": "Other", "GWH": "Other", "GYRO": "Other", "HAE": "Other", "HAFN": "Other", "HAIN": "Other", "HBB": "Other", "HBIO": "Other",
    "HCC": "Other", "HCHL": "Other", "HCI": "Other", "HCKT": "Other", "HCM": "Other", "HDL": "Other", "HEI-A": "Other", "HELE": "Other",
    "HELP": "Other", "HEPS": "Other", "HEQ": "Other", "HERE": "Other", "HFRO": "Other", "HGBL": "Other", "HGLB": "Other", "HGTY": "Other",
    "HGV": "Other", "HHH": "Other", "HHS": "Other", "HIFS": "Other", "HIHO": "Other", "HII": "Other", "HIND": "Other", "HIO": "Other",
    "HIPO": "Other", "HITI": "Other", "HIX": "Other", "HKIT": "Other", "HLF": "Other", "HLI": "Other", "HLIT": "Other", "HLLY": "Other",
    "HLMN": "Other", "HLP": "Other", "HMC": "Other", "HMN": "Other", "HMR": "Other", "HNI": "Other", "HNNA": "Other", "HNST": "Other",
    "HOFT": "Other", "HOG": "Other", "HOMB": "Other", "HOUR": "Other", "HOV": "Other", "HOVNP": "Other", "HOVR": "Other", "HP": "Other",
    "HPF": "Other", "HPP": "Other", "HQI": "Other", "HRB": "Other", "HRI": "Other", "HSBC": "Other", "HSCS": "Other", "HSDT": "Other",
    "HSHP": "Other", "HSIC": "Other", "HSY": "Other", "HTB": "Other", "HTBK": "Other", "HTCO": "Other", "HTCR": "Other", "HTD": "Other",
    "HTFL": "Other", "HTH": "Other", "HTHT": "Other", "HTLD": "Other", "HTLM": "Other", "HTO": "Other", "HTZ": "Other", "HUBB": "Other",
    "HUBG": "Other", "HUDI": "Other", "HUHU": "Other", "HUIZ": "Other", "HUMA": "Other", "HUN": "Other", "HURA": "Other", "HURC": "Other",
    "HUT": "Other", "HVT": "Other", "HVT-A": "Other", "HWBK": "Other", "HWC": "Other", "HWH": "Other", "HWKN": "Other", "HXHX": "Other",
    "HXL": "Other", "HY": "Other", "HYFM": "Other", "HYFT": "Other", "HYI": "Other", "HYPD": "Other", "HYPR": "Other", "HYT": "Other",
    "HZO": "Other", "IAE": "Other", "IART": "Other", "IBEX": "Other", "IBIO": "Other", "IBKR": "Other", "IBTA": "Other", "ICCC": "Other",
    "ICG": "Other", "ICL": "Other", "ICLR": "Other", "IDA": "Other", "IDAI": "Other", "IDN": "Other", "IDT": "Other", "IDYA": "Other",
    "IEP": "Other", "IEX": "Other", "IFBD": "Other", "IFF": "Other", "IFN": "Other", "IFRX": "Other", "IGA": "Other", "IGD": "Other",
    "IHD": "Other", "IHRT": "Other", "IHS": "Other", "III": "Other", "IIIV": "Other", "IIM": "Other", "ILAG": "Other", "ILMN": "Other",
    "IMA": "Other", "IMAX": "Other", "IMCC": "Other", "IMKTA": "Other", "IMMP": "Other", "IMMR": "Other", "IMNN": "Other", "IMRN": "Other",
    "IMTX": "Other", "IMXI": "Other", "INAB": "Other", "INBS": "Other", "INBX": "Other", "INCR": "Other", "INCY": "Other", "INEO": "Other",
    "INFQ": "Other", "ING": "Other", "INGM": "Other", "INGN": "Other", "INGR": "Other", "INHD": "Other", "INLF": "Other", "INMB": "Other",
    "INMD": "Other", "INNV": "Other", "INR": "Other", "INSG": "Other", "INSM": "Other", "INTA": "Other", "INTG": "Other", "INTJ": "Other",
    "INTR": "Other", "INTZ": "Other", "INV": "Other", "INVE": "Other", "INVX": "Other", "IONR": "Other", "IOSP": "Other", "IOT": "Other",
    "IOTR": "Other", "IPAR": "Other", "IPDN": "Other", "IPGP": "Other", "IPI": "Other", "IPM": "Other", "IPST": "Other", "IPX": "Other",
    "IQ": "Other", "IQI": "Other", "IQST": "Other", "IRD": "Other", "IRDM": "Other", "IREN": "Other", "IRIX": "Other", "IRMD": "Other",
    "IRON": "Other", "ISD": "Other", "ISPC": "Other", "ISSC": "Other", "ITGR": "Other", "ITOC": "Other", "ITRI": "Other", "ITRN": "Other",
    "ITT": "Other", "IVA": "Other", "IVDA": "Other", "IVF": "Other", "IVT": "Other", "IVVD": "Other", "IZEA": "Other", "IZM": "Other",
    "J": "Other", "JAKK": "Other", "JAN": "Other", "JBDI": "Other", "JBI": "Other", "JBIO": "Other", "JBK": "Other", "JBL": "Other",
    "JBS": "Other", "JBSS": "Other", "JBTM": "Other", "JCE": "Other", "JCI": "Other", "JCSE": "Other", "JCTC": "Other", "JD": "Other",
    "JDZG": "Other", "JELD": "Other", "JEM": "Other", "JF": "Other", "JFBR": "Other", "JFIN": "Other", "JFR": "Other", "JFU": "Other",
    "JG": "Other", "JGH": "Other", "JHG": "Other", "JHS": "Other", "JHX": "Other", "JILL": "Other", "JL": "Other", "JLHL": "Other",
    "JLL": "Other", "JMM": "Other", "JOE": "Other", "JOUT": "Other", "JOYY": "Other", "JRI": "Other", "JRSH": "Other", "JRVR": "Other",
    "JVA": "Other", "JWEL": "Other", "JXG": "Other", "JYNT": "Other", "JZXN": "Other", "KAI": "Other", "KALA": "Other", "KARO": "Other",
    "KBDC": "Other", "KBH": "Other", "KBR": "Other", "KBSX": "Other", "KD": "Other", "KDP": "Other", "KE": "Other", "KELYB": "Other",
    "KEN": "Other", "KEQU": "Other", "KEX": "Other", "KF": "Other", "KG": "Other", "KHC": "Other", "KIDS": "Other", "KIDZ": "Other",
    "KINS": "Other", "KIO": "Other", "KKR": "Other", "KLAR": "Other", "KLC": "Other", "KLIC": "Other", "KLTR": "Other", "KMDA": "Other",
    "KMPR": "Other", "KMRK": "Other", "KN": "Other", "KNF": "Other", "KNTK": "Other", "KOD": "Other", "KODK": "Other", "KOP": "Other",
    "KOPN": "Other", "KORE": "Other", "KOSS": "Other", "KPLT": "Other", "KRKR": "Other", "KRMN": "Other", "KRO": "Other", "KRRO": "Other",
    "KRUS": "Other", "KSCP": "Other", "KSPI": "Other", "KSS": "Other", "KT": "Other", "KTB": "Other", "KTCC": "Other", "KTF": "Other",
    "KTN": "Other", "KVHI": "Other", "KVUE": "Other", "KVYO": "Other", "KW": "Other", "KWM": "Other", "KWR": "Other", "KXIN": "Other",
    "KYIV": "Other", "KYNB": "Other", "KZR": "Other", "LAB": "Other", "LAC": "Other", "LAES": "Other", "LAKE": "Other", "LAMR": "Other",
    "LANV": "Other", "LAR": "Other", "LASE": "Other", "LASR": "Other", "LAUR": "Other", "LAW": "Other", "LAZ": "Other", "LB": "Other",
    "LBGJ": "Other", "LBRDA": "Other", "LBRDK": "Other", "LBTYA": "Other", "LBTYB": "Other", "LBTYK": "Other", "LCFY": "Other", "LCII": "Other",
    "LCNB": "Other", "LCUT": "Other", "LDI": "Other", "LE": "Other", "LEA": "Other", "LEDS": "Other", "LEE": "Other", "LEG": "Other",
    "LEO": "Other", "LEVI": "Other", "LEXX": "Other", "LFMD": "Other", "LFS": "Other", "LFT": "Other", "LFUS": "Other", "LFVN": "Other",
    "LFWD": "Other", "LGCB": "Other", "LGCL": "Other", "LGHL": "Other", "LGI": "Other", "LGIH": "Other", "LGN": "Other", "LGO": "Other",
    "LGVN": "Other", "LH": "Other", "LHAI": "Other", "LICN": "Other", "LIDR": "Other", "LIEN": "Other", "LIF": "Other", "LII": "Other",
    "LILA": "Other", "LILAK": "Other", "LINC": "Other", "LIND": "Other", "LINE": "Other", "LINK": "Other", "LION": "Other", "LIQT": "Other",
    "LITS": "Other", "LIVE": "Other", "LIVN": "Other", "LKQ": "Other", "LLYVA": "Other", "LLYVK": "Other", "LMAT": "Other", "LMB": "Other",
    "LMFA": "Other", "LMNR": "Other", "LMRI": "Other", "LNKS": "Other", "LNN": "Other", "LNSR": "Other", "LNTH": "Other", "LNZA": "Other",
    "LOAR": "Other", "LOB": "Other", "LOCL": "Other", "LOCO": "Other", "LOGI": "Other", "LONA": "Other", "LOOP": "Other", "LOPE": "Other",
    "LOVE": "Other", "LPCN": "Other", "LPG": "Other", "LPL": "Other", "LPSN": "Other", "LQDA": "Other", "LQDT": "Other", "LRHC": "Other",
    "LRN": "Other", "LSH": "Other", "LSPD": "Other", "LSTR": "Other", "LTBR": "Other", "LTC": "Other", "LTH": "Other", "LTRX": "Other",
    "LUCY": "Other", "LULU": "Other", "LUNG": "Other", "LVLU": "Other", "LVO": "Other", "LVWR": "Other", "LWLG": "Other", "LX": "Other",
    "LXEH": "Other", "LXU": "Other", "LYTS": "Other", "LZ": "Other", "LZB": "Other", "M": "Other", "MAAS": "Other", "MAC": "Other",
    "MAGN": "Other", "MAMA": "Other", "MAMO": "Other", "MANE": "Other", "MANU": "Other", "MASI": "Other", "MASS": "Other", "MATV": "Other",
    "MATX": "Other", "MAX": "Other", "MAYS": "Other", "MB": "Other", "MBAI": "Other", "MBC": "Other", "MBI": "Other", "MBIO": "Other",
    "MBLY": "Other", "MBUU": "Other", "MBX": "Other", "MC": "Other", "MCFT": "Other", "MCHX": "Other", "MCK": "Other", "MCR": "Other",
    "MCRI": "Other", "MCS": "Other", "MCW": "Other", "MCY": "Other", "MDIA": "Other", "MDLN": "Other", "MDRR": "Other", "MDU": "Other",
    "MDWD": "Other", "MDXG": "Other", "MED": "Other", "MEDP": "Other", "MEG": "Other", "MEGI": "Other", "MEGL": "Other", "MEHA": "Other",
    "MEI": "Other", "MELI": "Other", "MEOH": "Other", "MERC": "Other", "MESO": "Other", "METC": "Other", "METCB": "Other", "MFI": "Other",
    "MFM": "Other", "MG": "Other", "MGA": "Other", "MGF": "Other", "MGIH": "Other", "MGN": "Other", "MGPI": "Other", "MGRC": "Other",
    "MGRT": "Other", "MGRX": "Other", "MGTX": "Other", "MH": "Other", "MHD": "Other", "MHF": "Other", "MHO": "Other", "MIAX": "Other",
    "MICC": "Other", "MIDD": "Other", "MIGI": "Other", "MIMI": "Other", "MIN": "Other", "MITK": "Other", "MIY": "Other", "MKL": "Other",
    "MKTX": "Other", "MLAB": "Other", "MLEC": "Other", "MLGO": "Other", "MLI": "Other", "MLKN": "Other", "MLM": "Other", "MLP": "Other",
    "MLR": "Other", "MMD": "Other", "MMED": "Other", "MMI": "Other", "MMLP": "Other", "MMM": "Other", "MMS": "Other", "MMT": "Other",
    "MMYT": "Other", "MNDO": "Other", "MNOV": "Other", "MNSB": "Other", "MNSBP": "Other", "MNTN": "Other", "MNTS": "Other", "MNY": "Other",
    "MOB": "Other", "MOBX": "Other", "MOLN": "Other", "MOMO": "Other", "MORN": "Other", "MOV": "Other", "MOVE": "Other", "MPA": "Other",
    "MPAA": "Other", "MPX": "Other", "MQ": "Other", "MQY": "Other", "MRBK": "Other", "MRCY": "Other", "MRDN": "Other", "MRLN": "Other",
    "MRP": "Other", "MRSH": "Other", "MRX": "Other", "MSA": "Other", "MSB": "Other", "MSD": "Other", "MSGM": "Other", "MSGS": "Other",
    "MSGY": "Other", "MSI": "Other", "MSIF": "Other", "MSLE": "Other", "MSS": "Other", "MSW": "Other", "MT": "Other", "MTC": "Other",
    "MTEK": "Other", "MTEN": "Other", "MTEX": "Other", "MTLS": "Other", "MTN": "Other", "MTR": "Other", "MTRX": "Other", "MTVA": "Other",
    "MTW": "Other", "MTZ": "Other", "MUA": "Other", "MUC": "Other", "MUJ": "Other", "MUX": "Other", "MVIS": "Other", "MVST": "Other",
    "MWYN": "Other", "MXCT": "Other", "MXE": "Other", "MXF": "Other", "MXL": "Other", "MYE": "Other", "MYGN": "Other", "MYI": "Other",
    "MYN": "Other", "MYPS": "Other", "MYSE": "Other", "MYSZ": "Other", "MZTI": "Other", "NA": "Other", "NABL": "Other", "NAC": "Other",
    "NAD": "Other", "NAGE": "Other", "NAII": "Other", "NAKA": "Other", "NAN": "Other", "NAT": "Other", "NATH": "Other", "NATL": "Other",
    "NATR": "Other", "NAVI": "Other", "NAVN": "Other", "NAZ": "Other", "NB": "Other", "NBB": "Other", "NBIS": "Other", "NBP": "Other",
    "NBR": "Other", "NBTX": "Other", "NBXG": "Other", "NC": "Other", "NCA": "Other", "NCEL": "Other", "NCI": "Other", "NCMI": "Other",
    "NCNA": "Other", "NCRA": "Other", "NCSM": "Other", "NCT": "Other", "NCTY": "Other", "NCV": "Other", "NCZ": "Other", "NDLS": "Other",
    "NDMO": "Other", "NDRA": "Other", "NDSN": "Other", "NE": "Other", "NEA": "Other", "NEGG": "Other", "NEON": "Other", "NEOV": "Other",
    "NEPH": "Other", "NEU": "Other", "NEWT": "Other", "NEXA": "Other", "NEXN": "Other", "NEXT": "Other", "NFJ": "Other", "NGVC": "Other",
    "NGVT": "Other", "NICE": "Other", "NIE": "Other", "NIM": "Other", "NIPG": "Other", "NIQ": "Other", "NIVF": "Other", "NIXX": "Other",
    "NJR": "Other", "NKLR": "Other", "NKTX": "Other", "NKX": "Other", "NL": "Other", "NLOP": "Other", "NMAI": "Other", "NMAX": "Other",
    "NMFC": "Other", "NMG": "Other", "NMI": "Other", "NMRK": "Other", "NMS": "Other", "NMT": "Other", "NMZ": "Other", "NN": "Other",
    "NNBR": "Other", "NNDM": "Other", "NNI": "Other", "NNOX": "Other", "NNY": "Other", "NODK": "Other", "NOM": "Other", "NOMA": "Other",
    "NOTE": "Other", "NOTV": "Other", "NOV": "Other", "NPB": "Other", "NPCT": "Other", "NPK": "Other", "NPKI": "Other", "NPO": "Other",
    "NPT": "Other", "NPV": "Other", "NQP": "Other", "NRC": "Other", "NRDS": "Other", "NRDY": "Other", "NRK": "Other", "NRP": "Other",
    "NSIT": "Other", "NSP": "Other", "NSPR": "Other", "NSYS": "Other", "NTCT": "Other", "NTES": "Other", "NTGR": "Other", "NTNX": "Other",
    "NTR": "Other", "NTRB": "Other", "NTRP": "Other", "NTSK": "Other", "NTZ": "Other", "NUS": "Other", "NUV": "Other", "NUVB": "Other",
    "NUVL": "Other", "NUW": "Other", "NUWE": "Other", "NVEC": "Other", "NVGS": "Other", "NVMI": "Other", "NVNI": "Other", "NVO": "Other",
    "NVRI": "Other", "NVS": "Other", "NVST": "Other", "NVVE": "Other", "NVX": "Other", "NWGL": "Other", "NWL": "Other", "NWN": "Other",
    "NWPX": "Other", "NWS": "Other", "NWSA": "Other", "NWTG": "Other", "NX": "Other", "NXDR": "Other", "NXG": "Other", "NXGL": "Other",
    "NXJ": "Other", "NXP": "Other", "NXPL": "Other", "NXRT": "Other", "NXST": "Other", "NXTC": "Other", "NXXT": "Other", "NYAX": "Other",
    "NYT": "Other", "NYXH": "Other", "OABI": "Other", "OBAI": "Other", "OBIO": "Other", "OC": "Other", "OCG": "Other", "OCGN": "Other",
    "OCS": "Other", "ODD": "Other", "ODV": "Other", "OEC": "Other", "OFAL": "Other", "OFLX": "Other", "OFRM": "Other", "OGI": "Other",
    "OGN": "Other", "OI": "Other", "OIA": "Other", "OII": "Other", "OIO": "Other", "OKLO": "Other", "OKTA": "Other", "OLB": "Other",
    "OLED": "Other", "OLN": "Other", "OLOX": "Other", "OLP": "Other", "OMAB": "Other", "OMER": "Other", "OMEX": "Other", "OMF": "Other",
    "OMH": "Other", "ONC": "Other", "ONCO": "Other", "ONDS": "Other", "ONFO": "Other", "ONIT": "Other", "ONL": "Other", "ONMD": "Other",
    "ONTF": "Other", "OOMA": "Other", "OPAD": "Other", "OPFI": "Other", "OPLN": "Other", "OPP": "Other", "OPRA": "Other", "OPRX": "Other",
    "OPTU": "Other", "OPXS": "Other", "OPY": "Other", "OR": "Other", "ORBS": "Other", "ORGN": "Other", "ORGO": "Other", "ORI": "Other",
    "ORIS": "Other", "ORN": "Other", "OS": "Other", "OSG": "Other", "OSK": "Other", "OSPN": "Other", "OSRH": "Other", "OSS": "Other",
    "OSW": "Other", "OTEX": "Other", "OTLY": "Other", "OUT": "Other", "OVBC": "Other", "OVV": "Other", "OWLS": "Other", "OWLT": "Other",
    "OXBR": "Other", "OXM": "Other", "PAC": "Other", "PACB": "Other", "PACK": "Other", "PACS": "Other", "PALI": "Other", "PAM": "Other",
    "PAMT": "Other", "PARR": "Other", "PASG": "Other", "PATK": "Other", "PAVM": "Other", "PAXS": "Other", "PAY": "Other", "PAYO": "Other",
    "PAYP": "Other", "PAYS": "Other", "PAYX": "Other", "PB": "Other", "PBA": "Other", "PBI": "Other", "PBT": "Other", "PCAR": "Other",
    "PCF": "Other", "PCLA": "Other", "PCM": "Other", "PCN": "Other", "PCQ": "Other", "PCVX": "Other", "PCYO": "Other", "PD": "Other",
    "PDC": "Other", "PDD": "Other", "PDEX": "Other", "PDFS": "Other", "PDI": "Other", "PDO": "Other", "PDS": "Other", "PDT": "Other",
    "PDX": "Other", "PECO": "Other", "PEG": "Other", "PEN": "Other", "PENG": "Other", "PEO": "Other", "PEPG": "Other", "PERF": "Other",
    "PERI": "Other", "PESI": "Other", "PETZ": "Other", "PFL": "Other", "PFN": "Other", "PFSA": "Other", "PFX": "Other", "PGEN": "Other",
    "PGNY": "Other", "PGP": "Other", "PHAR": "Other", "PHG": "Other", "PHI": "Other", "PHIN": "Other", "PHK": "Other", "PHOE": "Other",
    "PHR": "Other", "PHUN": "Other", "PHVS": "Other", "PI": "Other", "PICS": "Other", "PII": "Other", "PIM": "Other", "PIPR": "Other",
    "PJT": "Other", "PKOH": "Other", "PL": "Other", "PLAB": "Other", "PLBL": "Other", "PLBY": "Other", "PLCE": "Other", "PLMR": "Other",
    "PLPC": "Other", "PLRZ": "Other", "PLSE": "Other", "PLSM": "Other", "PLUR": "Other", "PLUS": "Other", "PLXS": "Other", "PMAX": "Other",
    "PMEC": "Other", "PML": "Other", "PMM": "Other", "PMO": "Other", "PMTS": "Other", "PNI": "Other", "PNTG": "Other", "PODC": "Other",
    "POM": "Other", "POOL": "Other", "POST": "Other", "POWL": "Other", "PPC": "Other", "PPHC": "Other", "PPIH": "Other", "PPLC": "Other",
    "PPT": "Other", "PPTA": "Other", "PR": "Other", "PRA": "Other", "PRAA": "Other", "PRDO": "Other", "PRE": "Other", "PRHI": "Other",
    "PRI": "Other", "PRKS": "Other", "PRLB": "Other", "PRM": "Other", "PRMB": "Other", "PRME": "Other", "PROK": "Other", "PROP": "Other",
    "PRPL": "Other", "PRPO": "Other", "PRSO": "Other", "PSFE": "Other", "PSHG": "Other", "PSIG": "Other", "PSKY": "Other", "PSMT": "Other",
    "PSN": "Other", "PSNL": "Other", "PSO": "Other", "PSQH": "Other", "PSTG": "Other", "PTLE": "Other", "PTON": "Other", "PTRN": "Other",
    "PTY": "Other", "PUK": "Other", "PULM": "Other", "PURR": "Other", "PWP": "Other", "PWR": "Other", "PXED": "Other", "PXS": "Other",
    "PYPD": "Other", "PYT": "Other", "PZZA": "Other", "Q": "Other", "QCRH": "Other", "QDEL": "Other", "QFIN": "Other", "QGEN": "Other",
    "QH": "Other", "QNST": "Other", "QQQX": "Other", "QRHC": "Other", "QTI": "Other", "QTRX": "Other", "QTTB": "Other", "QUAD": "Other",
    "QUCY": "Other", "QUIK": "Other", "QURE": "Other", "QVCGA": "Other", "QXO": "Other", "R": "Other", "RA": "Other", "RACE": "Other",
    "RADX": "Other", "RAIL": "Other", "RAL": "Other", "RAMP": "Other", "RAY": "Other", "RBA": "Other", "RBBN": "Other", "RBC": "Other",
    "RBRK": "Other", "RCAT": "Other", "RCI": "Other", "RCKY": "Other", "RCS": "Other", "RDCM": "Other", "RDDT": "Other", "RDGT": "Other",
    "RDI": "Other", "RDIB": "Other", "RDNT": "Other", "RDVT": "Other", "RDW": "Other", "RDWR": "Other", "RDY": "Other", "RDZN": "Other",
    "REAL": "Other", "REAX": "Other", "REBN": "Other", "RECT": "Other", "REFR": "Other", "REKR": "Other", "RELL": "Other", "RELY": "Other",
    "RENT": "Other", "RENX": "Other", "REPL": "Other", "RES": "Other", "RETO": "Other", "REVB": "Other", "REYN": "Other", "RFIL": "Other",
    "RFL": "Other", "RFM": "Other", "RFMZ": "Other", "RGC": "Other", "RGCO": "Other", "RGEN": "Other", "RGNX": "Other", "RGP": "Other",
    "RGR": "Other", "RGS": "Other", "RGT": "Other", "RHLD": "Other", "RIG": "Other", "RILY": "Other", "RIME": "Other", "RIO": "Other",
    "RIV": "Other", "RJET": "Other", "RKDA": "Other", "RKT": "Other", "RLI": "Other", "RLJ": "Other", "RLYB": "Other", "RM": "Other",
    "RMAX": "Other", "RMCF": "Other", "RMCO": "Other", "RMI": "Other", "RMM": "Other", "RMMZ": "Other", "RMNI": "Other", "RMR": "Other",
    "RMSG": "Other", "RMT": "Other", "RNG": "Other", "RNXT": "Other", "ROC": "Other", "ROCK": "Other", "ROG": "Other", "ROIV": "Other",
    "ROKU": "Other", "ROL": "Other", "ROMA": "Other", "RPAY": "Other", "RPD": "Other", "RPID": "Other", "RRBI": "Other", "RRC": "Other",
    "RRR": "Other", "RSI": "Other", "RSKD": "Other", "RSSS": "Other", "RSVR": "Other", "RUBI": "Other", "RUM": "Other", "RUSHB": "Other",
    "RVI": "Other", "RVLV": "Other", "RVSN": "Other", "RVT": "Other", "RVYL": "Other", "RWAY": "Other", "RWT": "Other", "RXO": "Other",
    "RXST": "Other", "RYAAY": "Other", "RYAM": "Other", "RYAN": "Other", "RYM": "Other", "RYOJ": "Other", "RYZ": "Other", "RZLT": "Other",
    "SABR": "Other", "SAFX": "Other", "SAGT": "Other", "SAIC": "Other", "SAIH": "Other", "SAIL": "Other", "SAM": "Other", "SAN": "Other",
    "SANM": "Other", "SAP": "Other", "SARO": "Other", "SATL": "Other", "SATS": "Other", "SB": "Other", "SBAC": "Other", "SBDS": "Other",
    "SBET": "Other", "SBGI": "Other", "SBH": "Other", "SBI": "Other", "SBLK": "Other", "SBR": "Other", "SBSI": "Other", "SCAG": "Other",
    "SCHL": "Other", "SCI": "Other", "SCKT": "Other", "SCL": "Other", "SCLX": "Other", "SCNX": "Other", "SCOR": "Other", "SCPQ": "Other",
    "SCSC": "Other", "SCVL": "Other", "SCYX": "Other", "SDGR": "Other", "SDHC": "Other", "SDHY": "Other", "SDOT": "Other", "SDRL": "Other",
    "SEAT": "Other", "SEE": "Other", "SEED": "Other", "SEER": "Other", "SELF": "Other", "SELX": "Other", "SEMR": "Other", "SENS": "Other",
    "SEPN": "Other", "SERA": "Other", "SEV": "Other", "SEZL": "Other", "SFBS": "Other", "SFHG": "Other", "SFL": "Other", "SFST": "Other",
    "SFWL": "Other", "SG": "Other", "SGA": "Other", "SGC": "Other", "SGHC": "Other", "SGHT": "Other", "SGI": "Other", "SGML": "Other",
    "SGMT": "Other", "SGRP": "Other", "SGRY": "Other", "SGU": "Other", "SHBI": "Other", "SHEN": "Other", "SHFS": "Other", "SHIM": "Other",
    "SHIP": "Other", "SHMD": "Other", "SHOO": "Other", "SI": "Other", "SIBN": "Other", "SID": "Other", "SIG": "Other", "SII": "Other",
    "SILC": "Other", "SIRI": "Other", "SITC": "Other", "SITE": "Other", "SJ": "Other", "SJT": "Other", "SKBL": "Other", "SKE": "Other",
    "SKIL": "Other", "SKK": "Other", "SKLZ": "Other", "SKM": "Other", "SKT": "Other", "SKY": "Other", "SKYE": "Other", "SKYH": "Other",
    "SKYQ": "Other", "SKYX": "Other", "SLDB": "Other", "SLE": "Other", "SLGN": "Other", "SLM": "Other", "SLMT": "Other", "SLNG": "Other",
    "SLNH": "Other", "SLP": "Other", "SLQT": "Other", "SLS": "Other", "SLSN": "Other", "SLVM": "Other", "SMC": "Other", "SMG": "Other",
    "SMHI": "Other", "SMID": "Other", "SMP": "Other", "SMRT": "Other", "SMTC": "Other", "SMTI": "Other", "SMTK": "Other", "SMWB": "Other",
    "SMX": "Other", "SN": "Other", "SNA": "Other", "SNAL": "Other", "SNBR": "Other", "SND": "Other", "SNDA": "Other", "SNDK": "Other",
    "SNDL": "Other", "SNDR": "Other", "SNES": "Other", "SNEX": "Other", "SNGX": "Other", "SNN": "Other", "SNTG": "Other", "SNTI": "Other",
    "SNX": "Other", "SNY": "Other", "SNYR": "Other", "SOBO": "Other", "SOBR": "Other", "SOC": "Other", "SOGP": "Other", "SOHU": "Other",
    "SOLS": "Other", "SOLV": "Other", "SOMN": "Other", "SON": "Other", "SONM": "Other", "SOPA": "Other", "SOPH": "Other", "SORA": "Other",
    "SOS": "Other", "SOTK": "Other", "SOWG": "Other", "SPAI": "Other", "SPCB": "Other", "SPCE": "Other", "SPE": "Other", "SPH": "Other",
    "SPHL": "Other", "SPIR": "Other", "SPNT": "Other", "SPPL": "Other", "SPRB": "Other", "SPRC": "Other", "SPT": "Other", "SPWH": "Other",
    "SPXX": "Other", "SQM": "Other", "SR": "Other", "SRAD": "Other", "SRCE": "Other", "SRFM": "Other", "SRG": "Other", "SRI": "Other",
    "SRL": "Other", "SRZN": "Other", "SSII": "Other", "SSM": "Other", "SSP": "Other", "SST": "Other", "SSTI": "Other", "SSTK": "Other",
    "SSYS": "Other", "STAK": "Other", "STC": "Other", "STE": "Other", "STEX": "Other", "STFS": "Other", "STHO": "Other", "STKE": "Other",
    "STKL": "Other", "STLA": "Other", "STM": "Other", "STN": "Other", "STNE": "Other", "STNG": "Other", "STRA": "Other", "STRR": "Other",
    "STRS": "Other", "STRT": "Other", "STTK": "Other", "STUB": "Other", "STVN": "Other", "SUGP": "Other", "SUI": "Other", "SUIG": "Other",
    "SUIS": "Other", "SUNB": "Other", "SURG": "Other", "SVC": "Other", "SVCO": "Other", "SVRA": "Other", "SVRE": "Other", "SVRN": "Other",
    "SVV": "Other", "SW": "Other", "SWAG": "Other", "SWBI": "Other", "SWIM": "Other", "SWKH": "Other", "SWMR": "Other", "SWVL": "Other",
    "SWZ": "Other", "SXI": "Other", "SY": "Other", "SYM": "Other", "SYNA": "Other", "SYPR": "Other", "TAC": "Other", "TANH": "Other",
    "TAOP": "Other", "TAOX": "Other", "TAYD": "Other", "TBCH": "Other", "TBH": "Other", "TBHC": "Other", "TBLA": "Other", "TBLD": "Other",
    "TBN": "Other", "TBRG": "Other", "TC": "Other", "TCBS": "Other", "TCBX": "Other", "TCOM": "Other", "TCX": "Other", "TDAY": "Other",
    "TDF": "Other", "TDIC": "Other", "TDOG": "Other", "TDUP": "Other", "TEAD": "Other", "TEAM": "Other", "TECK": "Other", "TEI": "Other",
    "TELA": "Other", "TEO": "Other", "TEX": "Other", "TFII": "Other", "TFX": "Other", "TG": "Other", "TGE": "Other", "TGHL": "Other",
    "TGL": "Other", "TGLS": "Other", "TGT": "Other", "THCH": "Other", "THH": "Other", "THO": "Other", "THR": "Other", "THRM": "Other",
    "THRY": "Other", "TIC": "Other", "TIGO": "Other", "TIL": "Other", "TILE": "Other", "TIPT": "Other", "TISI": "Other", "TITN": "Other",
    "TJGC": "Other", "TK": "Other", "TKC": "Other", "TKLF": "Other", "TKNO": "Other", "TKO": "Other", "TKR": "Other", "TLF": "Other",
    "TLIH": "Other", "TLK": "Other", "TLPH": "Other", "TLRY": "Other", "TLS": "Other", "TLSA": "Other", "TLSI": "Other", "TM": "Other",
    "TNC": "Other", "TNDM": "Other", "TNET": "Other", "TNK": "Other", "TNMG": "Other", "TOMZ": "Other", "TONX": "Other", "TORO": "Other",
    "TOUR": "Other", "TOYO": "Other", "TPB": "Other", "TPC": "Other", "TPCS": "Other", "TPG": "Other", "TPH": "Other", "TPL": "Other",
    "TR": "Other", "TRAK": "Other", "TRC": "Other", "TREX": "Other", "TRI": "Other", "TRIP": "Other", "TRMB": "Other", "TRMD": "Other",
    "TRMK": "Other", "TRNR": "Other", "TRNS": "Other", "TRON": "Other", "TROO": "Other", "TROX": "Other", "TRS": "Other", "TRTX": "Other",
    "TRU": "Other", "TRUG": "Other", "TRUP": "Other", "TRVG": "Other", "TSAT": "Other", "TSHA": "Other", "TSI": "Other", "TSQ": "Other",
    "TSSI": "Other", "TSUI": "Other", "TTAM": "Other", "TTAN": "Other", "TTC": "Other", "TTE": "Other", "TTEC": "Other", "TTEK": "Other",
    "TTGT": "Other", "TU": "Other", "TULP": "Other", "TV": "Other", "TVC": "Other", "TVE": "Other", "TVGN": "Other", "TW": "Other",
    "TWAV": "Other", "TWFG": "Other", "TWG": "Other", "TWI": "Other", "TWIN": "Other", "TWLO": "Other", "TWN": "Other", "TWST": "Other",
    "TY": "Other", "TYRA": "Other", "UA": "Other", "UAA": "Other", "UAMY": "Other", "UBFO": "Other", "UBS": "Other", "UCTT": "Other",
    "UDMY": "Other", "UE": "Other", "UEIC": "Other", "UFCS": "Other", "UFI": "Other", "UG": "Other", "UGI": "Other", "UGRO": "Other",
    "UHAL": "Other", "UHG": "Other", "UI": "Other", "UIS": "Other", "UK": "Other", "ULBI": "Other", "ULCC": "Other", "ULS": "Other",
    "ULTA": "Other", "UMC": "Other", "UMH": "Other", "UNF": "Other", "UNIT": "Other", "UONE": "Other", "UONEK": "Other", "UP": "Other",
    "UPB": "Other", "UPBD": "Other", "UPWK": "Other", "UPXI": "Other", "UROY": "Other", "USA": "Other", "USAR": "Other", "USEA": "Other",
    "USIO": "Other", "USPH": "Other", "UTF": "Other", "UTI": "Other", "UTL": "Other", "UTSI": "Other", "UTZ": "Other", "UVV": "Other",
    "UWMC": "Other", "UXIN": "Other", "VAC": "Other", "VAL": "Other", "VALN": "Other", "VALU": "Other", "VATE": "Other", "VAVX": "Other",
    "VBF": "Other", "VBIX": "Other", "VC": "Other", "VCEL": "Other", "VCIG": "Other", "VCV": "Other", "VCX": "Other", "VCYT": "Other",
    "VEEA": "Other", "VELO": "Other", "VEON": "Other", "VERI": "Other", "VERU": "Other", "VERX": "Other", "VFF": "Other", "VG": "Other",
    "VGI": "Other", "VHC": "Other", "VHI": "Other", "VHUB": "Other", "VICR": "Other", "VIK": "Other", "VISN": "Other", "VITL": "Other",
    "VIVS": "Other", "VKQ": "Other", "VLGEA": "Other", "VLT": "Other", "VLTO": "Other", "VMC": "Other", "VMET": "Other", "VMI": "Other",
    "VMO": "Other", "VNCE": "Other", "VNET": "Other", "VNT": "Other", "VOD": "Other", "VOXR": "Other", "VPG": "Other", "VPV": "Other",
    "VRA": "Other", "VRAX": "Other", "VRE": "Other", "VREX": "Other", "VRME": "Other", "VRNS": "Other", "VRRM": "Other", "VRSK": "Other",
    "VRSN": "Other", "VRT": "Other", "VS": "Other", "VSAT": "Other", "VSCO": "Other", "VSEC": "Other", "VSME": "Other", "VSNT": "Other",
    "VST": "Other", "VSTD": "Other", "VSTM": "Other", "VSTS": "Other", "VTEX": "Other", "VTIX": "Other", "VTOL": "Other", "VTSI": "Other",
    "VUZI": "Other", "VVR": "Other", "VVV": "Other", "VVX": "Other", "VWAV": "Other", "VYX": "Other", "WAFDP": "Other", "WAFU": "Other",
    "WAI": "Other", "WALD": "Other", "WATT": "Other", "WAY": "Other", "WB": "Other", "WBUY": "Other", "WBX": "Other", "WCC": "Other",
    "WCN": "Other", "WCT": "Other", "WD": "Other", "WDAY": "Other", "WDFC": "Other", "WDI": "Other", "WEA": "Other", "WEAV": "Other",
    "WEN": "Other", "WERN": "Other", "WEST": "Other", "WEX": "Other", "WEYS": "Other", "WFF": "Other", "WFG": "Other", "WFRD": "Other",
    "WGO": "Other", "WGS": "Other", "WHD": "Other", "WHG": "Other", "WHR": "Other", "WIA": "Other", "WINA": "Other", "WING": "Other",
    "WIT": "Other", "WIW": "Other", "WIX": "Other", "WK": "Other", "WKC": "Other", "WKEY": "Other", "WKSP": "Other", "WLDS": "Other",
    "WLFC": "Other", "WLK": "Other", "WLTH": "Other", "WLY": "Other", "WLYB": "Other", "WMG": "Other", "WMK": "Other", "WMS": "Other",
    "WNC": "Other", "WOLF": "Other", "WOR": "Other", "WORX": "Other", "WRBY": "Other", "WRD": "Other", "WSBC": "Other", "WSC": "Other",
    "WSHP": "Other", "WSM": "Other", "WSO": "Other", "WSO-B": "Other", "WT": "Other", "WTI": "Other", "WTO": "Other", "WTW": "Other",
    "WU": "Other", "WULF": "Other", "WVE": "Other", "WVVI": "Other", "WW": "Other", "WWD": "Other", "WWW": "Other", "WXM": "Other",
    "WY": "Other", "WYFI": "Other", "XAIR": "Other", "XBIO": "Other", "XBP": "Other", "XCH": "Other", "XCUR": "Other", "XELB": "Other",
    "XGN": "Other", "XHG": "Other", "XHLD": "Other", "XMTR": "Other", "XNCR": "Other", "XNET": "Other", "XOMA": "Other", "XOS": "Other",
    "XP": "Other", "XPEL": "Other", "XPER": "Other", "XPON": "Other", "XPRO": "Other", "XRAY": "Other", "XRX": "Other", "XTKG": "Other",
    "XWEL": "Other", "XWIN": "Other", "XXII": "Other", "XYZ": "Other", "XZO": "Other", "YB": "Other", "YDES": "Other", "YELP": "Other",
    "YETI": "Other", "YEXT": "Other", "YHC": "Other", "YHGJ": "Other", "YI": "Other", "YIBO": "Other", "YJ": "Other", "YMAT": "Other",
    "YMT": "Other", "YOU": "Other", "YOUL": "Other", "YPF": "Other", "YSXT": "Other", "YTRA": "Other", "YUMC": "Other", "YXT": "Other",
    "YYAI": "Other", "YYGH": "Other", "ZBAI": "Other", "ZCMD": "Other", "ZD": "Other", "ZENA": "Other", "ZG": "Other", "ZGM": "Other",
    "ZGN": "Other", "ZIM": "Other", "ZIP": "Other", "ZKIN": "Other", "ZM": "Other", "ZNB": "Other", "ZOOZ": "Other", "ZSTK": "Other",
    "ZTEK": "Other", "ZTR": "Other", "ZURA": "Other", "ZVIA": "Other", "ZYME": "Other",
    "ABR": "Real Estate", "ACR": "Real Estate", "ACRE": "Real Estate", "ADC": "Real Estate", "AHRT": "Real Estate", "AKR": "Real Estate", "ARE": "Real Estate", "ARI": "Real Estate",
    "ARR": "Real Estate", "BDN": "Real Estate", "BPRE": "Real Estate", "BRX": "Real Estate", "CLPR": "Real Estate", "CTO": "Real Estate", "CTRE": "Real Estate", "EPRT": "Real Estate",
    "ESRT": "Real Estate", "FBRT": "Real Estate", "FR": "Real Estate", "FVR": "Real Estate", "GEO": "Real Estate", "IGR": "Real Estate", "IRM": "Real Estate", "IRT": "Real Estate",
    "JRS": "Real Estate", "KRC": "Real Estate", "KREF": "Real Estate", "KRG": "Real Estate", "LEGH": "Real Estate", "LND": "Real Estate", "LRE": "Real Estate", "NNN": "Real Estate",
    "NREF": "Real Estate", "NXDT": "Real Estate", "PDM": "Real Estate", "PGZ": "Real Estate", "PKST": "Real Estate", "REFI": "Real Estate", "REXR": "Real Estate", "RFI": "Real Estate",
    "RITR": "Real Estate", "RLTY": "Real Estate", "RPT": "Real Estate", "RQI": "Real Estate", "RYN": "Real Estate", "SEVN": "Real Estate", "SILA": "Real Estate", "SMA": "Real Estate",
    "SQFT": "Real Estate", "STWD": "Real Estate", "SUNS": "Real Estate", "TRNO": "Real Estate", "VNO": "Real Estate", "WSR": "Real Estate", "XRN": "Real Estate",
    "ABAT": "Tech", "AD": "Tech", "ADP": "Tech", "ADUR": "Tech", "AERT": "Tech", "AEVA": "Tech", "AGPU": "Tech", "AIHS": "Tech",
    "AIIO": "Tech", "AIO": "Tech", "AIRJ": "Tech", "AIRS": "Tech", "AISP": "Tech", "AIT": "Tech", "AIXC": "Tech", "ALAR": "Tech",
    "ALKT": "Tech", "AMCI": "Tech", "AMKR": "Tech", "AMPX": "Tech", "ANL": "Tech", "AOSL": "Tech", "APLD": "Tech", "AQB": "Tech",
    "ARAI": "Tech", "ARBE": "Tech", "ARBK": "Tech", "ARQQ": "Tech", "ASUR": "Tech", "ATGL": "Tech", "ATMU": "Tech", "AVR": "Tech",
    "AVX": "Tech", "AZI": "Tech", "BETA": "Tech", "BFRG": "Tech", "BGIN": "Tech", "BIAF": "Tech", "BKSY": "Tech", "BLIN": "Tech",
    "BNZI": "Tech", "BST": "Tech", "BSTZ": "Tech", "BTBT": "Tech", "BTCT": "Tech", "BTDR": "Tech", "BTQ": "Tech", "BTX": "Tech",
    "BWXT": "Tech", "BYRN": "Tech", "CCLD": "Tech", "CCSI": "Tech", "CCTG": "Tech", "CD": "Tech", "CGNT": "Tech", "CGTL": "Tech",
    "CHAI": "Tech", "CHKP": "Tech", "CIFR": "Tech", "CLIR": "Tech", "CNET": "Tech", "CPSH": "Tech", "CRCL": "Tech", "CRDO": "Tech",
    "CRS": "Tech", "CSAI": "Tech", "CYD": "Tech", "CYPH": "Tech", "DAIO": "Tech", "DBGI": "Tech", "DBRG": "Tech", "DBVT": "Tech",
    "DCX": "Tech", "DEFT": "Tech", "DFSC": "Tech", "DJT": "Tech", "DRCT": "Tech", "DSP": "Tech", "DSY": "Tech", "DTCX": "Tech",
    "DTSS": "Tech", "DTST": "Tech", "DUOT": "Tech", "DVLT": "Tech", "DXC": "Tech", "DYOR": "Tech", "EDHL": "Tech", "EDTK": "Tech",
    "EDU": "Tech", "EMAT": "Tech", "EMPD": "Tech", "ESE": "Tech", "EVLV": "Tech", "EZGO": "Tech", "FIGR": "Tech", "FIP": "Tech",
    "FOFO": "Tech", "FRGT": "Tech", "FTAI": "Tech", "GCT": "Tech", "GCTS": "Tech", "GFAI": "Tech", "GITS": "Tech", "GLXY": "Tech",
    "GMEX": "Tech", "GMM": "Tech", "GRDX": "Tech", "GRRR": "Tech", "GSIT": "Tech", "GSUN": "Tech", "GTEC": "Tech", "GTM": "Tech",
    "GWAV": "Tech", "GWRE": "Tech", "GXAI": "Tech", "HCAI": "Tech", "HDSN": "Tech", "HIMX": "Tech", "HIVE": "Tech", "HLIO": "Tech",
    "HOLO": "Tech", "HPAI": "Tech", "HQ": "Tech", "HSAI": "Tech", "HUBC": "Tech", "IDCC": "Tech", "IINN": "Tech", "IMOS": "Tech",
    "IMTE": "Tech", "INDI": "Tech", "INOD": "Tech", "ISPR": "Tech", "JTAI": "Tech", "JZ": "Tech", "KC": "Tech", "KDK": "Tech",
    "KITT": "Tech", "KNDI": "Tech", "KRNT": "Tech", "LHX": "Tech", "LIFE": "Tech", "LNAI": "Tech", "LOBO": "Tech", "LOT": "Tech",
    "LPTH": "Tech", "LSAK": "Tech", "LSCC": "Tech", "LUMN": "Tech", "LZMH": "Tech", "MAPS": "Tech", "MASK": "Tech", "MATH": "Tech",
    "MCN": "Tech", "MDAI": "Tech", "MIND": "Tech", "MIR": "Tech", "MRAM": "Tech", "MRVI": "Tech", "MRVL": "Tech", "MSAI": "Tech",
    "MTX": "Tech", "MX": "Tech", "NAAS": "Tech", "NAMI": "Tech", "NIU": "Tech", "NSSC": "Tech", "NTCL": "Tech", "NTHI": "Tech",
    "NTIC": "Tech", "NTWK": "Tech", "NVTS": "Tech", "NXL": "Tech", "NXPI": "Tech", "NXTS": "Tech", "NXTT": "Tech", "ODYS": "Tech",
    "ON": "Tech", "ORA": "Tech", "ORIO": "Tech", "ORKT": "Tech", "OSUR": "Tech", "OTF": "Tech", "PAGS": "Tech", "PAR": "Tech",
    "PAVS": "Tech", "PAYC": "Tech", "PCOR": "Tech", "PCT": "Tech", "PDYN": "Tech", "PEW": "Tech", "PGY": "Tech", "POET": "Tech",
    "PONY": "Tech", "PRCT": "Tech", "PRFX": "Tech", "PRGS": "Tech", "PRTH": "Tech", "PRZO": "Tech", "QBTS": "Tech", "QCLS": "Tech",
    "QMCO": "Tech", "QS": "Tech", "QSI": "Tech", "RAIN": "Tech", "RCMT": "Tech", "RCON": "Tech", "RCT": "Tech", "REZI": "Tech",
    "ROP": "Tech", "RR": "Tech", "RYET": "Tech", "RZLV": "Tech", "SANG": "Tech", "SBLX": "Tech", "SDA": "Tech", "SERV": "Tech",
    "SES": "Tech", "SGLY": "Tech", "SHAZ": "Tech", "SIFY": "Tech", "SIGA": "Tech", "SIMO": "Tech", "SINT": "Tech", "SKYT": "Tech",
    "SLGL": "Tech", "SMSI": "Tech", "SNT": "Tech", "SPXC": "Tech", "SSNC": "Tech", "ST": "Tech", "STI": "Tech", "STK": "Tech",
    "STSS": "Tech", "SUPX": "Tech", "SXT": "Tech", "TACT": "Tech", "TATT": "Tech", "TCMD": "Tech", "TDC": "Tech", "TDS": "Tech",
    "TDTH": "Tech", "TEM": "Tech", "TRSG": "Tech", "TSEM": "Tech", "TSM": "Tech", "TTI": "Tech", "TTMI": "Tech", "TYL": "Tech",
    "UBXG": "Tech", "UCL": "Tech", "UFPT": "Tech", "UPLD": "Tech", "VIOT": "Tech", "VLN": "Tech", "VMAR": "Tech", "VOYG": "Tech",
    "VSA": "Tech", "VSH": "Tech", "WETH": "Tech", "WETO": "Tech", "WIMI": "Tech", "WNW": "Tech", "WRAP": "Tech", "WTS": "Tech",
    "XFLT": "Tech", "YAAS": "Tech", "YDDL": "Tech", "YDKG": "Tech", "YQ": "Tech", "ZBAO": "Tech", "ZDAI": "Tech", "ZLAB": "Tech",
    "BWB": "Utilities", "CPK": "Utilities", "CWAN": "Utilities", "CWT": "Utilities", "DPG": "Utilities", "FELE": "Utilities", "FFAI": "Utilities", "GUT": "Utilities",
    "GWRS": "Utilities", "HE": "Utilities", "LECO": "Utilities", "MWA": "Utilities", "NVT": "Utilities", "POR": "Utilities", "SCWO": "Utilities", "TDW": "Utilities",
    "WBI": "Utilities", "WTTR": "Utilities", "ZWS": "Utilities",
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
    # ── NASDAQ/NYSE additions ─────────────────────────────
    "AACG": "ATA Creativity Global",
    "AAME": "Atlantic American",
    "AAMI": "Acadian Asset Management",
    "AAOI": "Applied Optoelectronics",
    "AAP": "Advance Auto Parts",
    "AAPG": "Ascentage Pharma",
    "AARD": "Aardvark",
    "AAT": "American Assets Trust",
    "AAUC": "Allied Gold",
    "AB": "AllianceBernstein Holding L.",
    "ABAT": "American Battery Technology",
    "ABCB": "Ameris Bancorp",
    "ABCL": "AbCellera Biologics",
    "ABEO": "Abeona",
    "ABLV": "Able View Global",
    "ABM": "ABM Industries",
    "ABOS": "Acumen",
    "ABR": "Arbor Realty Trust",
    "ABSI": "Absci",
    "ABTC": "American Bitcoin Class A",
    "ABTS": "Abits Inc",
    "ABUS": "Arbutus Biopharma",
    "ABVC": "ABVC BioPharma",
    "ABVE": "Above Food Ingredients",
    "ABVX": "Abivax SA",
    "ABX": "Abacus Global Management, Cl",
    "ACA": "Arcosa",
    "ACB": "Aurora Cannabis",
    "ACCL": "Acco",
    "ACCO": "Acco Brands",
    "ACDC": "ProFrac Holding Class A",
    "ACEL": "Accel Entertainment",
    "ACET": "Adicet Bio",
    "ACFN": "Acorn Energy",
    "ACGL": "Arch Capital",
    "ACH": "Accendra Health",
    "ACHC": "Acadia Healthcare",
    "ACHV": "Achieve Life Sciences",
    "ACI": "Albertsons Companies, Class",
    "ACIC": "American Coastal Insurance",
    "ACIU": "AC Immune SA",
    "ACLX": "Arcellx",
    "ACM": "AECOM",
    "ACNB": "ACNB",
    "ACNT": "Ascent Industries",
    "ACOG": "Alpha Cognition",
    "ACON": "Aclarion",
    "ACP": "abrdn Income Credit Strategi",
    "ACR": "ACRES Commercial Realty",
    "ACRE": "Ares Commercial Real Estate",
    "ACRS": "Aclaris",
    "ACRV": "Acrivon",
    "ACT": "Enact",
    "ACTG": "Acacia Research (Acacia Tech",
    "ACTU": "Actuate Common stock",
    "ACV": "Virtus Diversified Income &",
    "ACXP": "Acurx",
    "AD": "Array Digital Infrastructure",
    "ADAG": "Adagene",
    "ADAM": "Adamas Trust",
    "ADBE": "Adobe",
    "ADC": "Agree Realty",
    "ADCT": "ADC SA",
    "ADEA": "Adeia",
    "ADGM": "Adagio Medical Inc",
    "ADIL": "Adial Inc",
    "ADMA": "ADMA Biologics Inc",
    "ADNT": "Adient",
    "ADP": "Automatic Data Processing",
    "ADPT": "Adaptive Biotechnologies",
    "ADSE": "ADS-TEC ENERGY",
    "ADSK": "Autodesk",
    "ADT": "ADT",
    "ADTX": "Aditxt",
    "ADUR": "Aduro Clean",
    "ADV": "Advantage Solutions Class A",
    "ADVB": "Advanced Biomed",
    "ADX": "Adams Diversified Equity Fun",
    "ADXN": "Addex Ltd",
    "AEBI": "Aebi Schmidt Holding",
    "AEC": "Anfield Energy",
    "AEE": "Ameren",
    "AEG": "Aegon New York Registry Shar",
    "AEHL": "Antelope Enterprise",
    "AEHR": "Aehr Test Systems",
    "AEI": "Alset (TX)",
    "AEIS": "Advanced Energy Industries",
    "AEMD": "Aethlon Medical",
    "AENT": "Alliance Entertainment Holdi",
    "AER": "AerCap",
    "AERT": "Aeries Technology Class A Or",
    "AESI": "Atlas Energy Solutions",
    "AEVA": "Aeva",
    "AEYE": "AudioEye",
    "AFB": "AllianceBernstein National M",
    "AFBI": "Affinity Bancshares (MD)",
    "AFCG": "Advanced Flower Capital",
    "AFG": "American Financial",
    "AFJK": "Aimei Health Technology Ltd",
    "AFRI": "Forafric Global",
    "AG": "First Majestic Silver (Canad",
    "AGAE": "Allied Gaming & Entertainmen",
    "AGBK": "AGI Inc Class A",
    "AGCC": "Agencia Comercial Spirits Lt",
    "AGD": "abrdn Global Dynamic Dividen",
    "AGH": "Aureus Greenway",
    "AGI": "Alamos Gold Class A",
    "AGIO": "Agios",
    "AGL": "agilon health, inc.",
    "AGM": "Federal Agricultural Mortgag",
    "AGMB": "AgomAb NV",
    "AGMH": "AGM",
    "AGNC": "AGNC Investment",
    "AGO": "Assured Guaranty",
    "AGPU": "Axe Compute",
    "AGRO": "Adecoagro",
    "AGRZ": "Agroz",
    "AGX": "Argan",
    "AGYS": "Agilysys (DE)",
    "AHCO": "AdaptHealth",
    "AHG": "Akso Health ADS",
    "AHMA": "Ambitions Enterprise Managem",
    "AHR": "American Healthcare REIT",
    "AHRT": "AH Realty Trust",
    "AHT": "Ashford Hospitality Trust In",
    "AIDX": "20/20 Biolabs",
    "AIFF": "Firefly Neuroscience",
    "AIFU": "AIFU Class A Ordinary Share",
    "AIHS": "Senmiao Technology",
    "AII": "American Integrity Insurance",
    "AIIO": "Robo.ai",
    "AIMD": "Ainos",
    "AIN": "Albany",
    "AIO": "Virtus Artificial Intelligen",
    "AIOS": "AIOS Tech Class A",
    "AIOT": "PowerFleet",
    "AIP": "Arteris",
    "AIR": "AAR",
    "AIRE": "reAlpha Tech",
    "AIRG": "Airgain",
    "AIRJ": "AirJoule Class A",
    "AIRO": "AIRO",
    "AIRS": "AirSculpt",
    "AIRT": "Air T",
    "AISP": "Airship AI Class A",
    "AIT": "Applied Industrial",
    "AIV": "Apartment Investment and Man",
    "AIXC": "AIxCrypto",
    "AIXI": "XIAO-I",
    "AJG": "Arthur J. Gallagher &",
    "AKA": "a.k.a. Brands Holding",
    "AKAN": "Akanda",
    "AKBA": "Akebia",
    "AKO-A": "Embotelladora Andina",
    "AKO-B": "Embotelladora Andina",
    "AKR": "Acadia Realty Trust",
    "AKTS": "Aktis Oncology Common stock",
    "AKTX": "Akari ADS",
    "AL": "Air Lease Class A",
    "ALAB": "Astera Labs",
    "ALAR": "Alarum American Depositary S",
    "ALBT": "Avalon GloboCare",
    "ALC": "Alcon",
    "ALCO": "Alico",
    "ALDF": "Aldel Financial II",
    "ALDX": "Aldeyra",
    "ALEC": "Alector",
    "ALG": "Alamo",
    "ALGS": "Aligos",
    "ALGT": "Allegiant Travel",
    "ALH": "Alliance Laundry",
    "ALIT": "Alight, Class A",
    "ALKT": "Alkami Technology",
    "ALLO": "Allogene",
    "ALLR": "Allarity",
    "ALLT": "Allot",
    "ALM": "Almonty Industries",
    "ALMS": "Alumis",
    "ALMU": "Aeluma",
    "ALNT": "Allient",
    "ALOT": "AstroNova",
    "ALOV": "Aldabra 4 Liquidity Opportun",
    "ALOY": "REalloys",
    "ALPS": "ALPS Inc Ordinary Share",
    "ALRS": "Alerus Financial",
    "ALSN": "Allison Transmission",
    "ALT": "Altimmune",
    "ALTG": "Alta Equipment Class A",
    "ALTI": "AlTi Global Class A",
    "ALTO": "Alto Ingredients",
    "ALTS": "ALT5 Sigma",
    "ALV": "Autoliv",
    "ALVO": "Alvotech",
    "ALX": "Alexander\'s",
    "ALXO": "ALX Oncology",
    "ALZN": "Alzamend Neuro",
    "AM": "Antero Midstream",
    "AMAL": "Amalgamated Financial (DE)",
    "AMBP": "Ardagh Metal Packaging",
    "AMBQ": "Ambiq Micro",
    "AMBR": "Amber Holding",
    "AMC": "AMC Entertainment, Class A",
    "AMCI": "AMC Robotics",
    "AMCR": "Amcor",
    "AMCX": "AMC Networks Class A",
    "AMG": "Affiliated Managers",
    "AMIX": "Autonomix Medical",
    "AMKR": "Amkor Technology",
    "AMLX": "Amylyx",
    "AMN": "AMN Healthcare Services Inc",
    "AMOD": "Alpha Modus Class A",
    "AMPG": "Amplitech",
    "AMPH": "Amphastar",
    "AMPL": "Amplitude Class A",
    "AMPX": "Amprius",
    "AMPY": "Amplify Energy",
    "AMR": "Alpha Metallurgical Resource",
    "AMRC": "Ameresco, Class A",
    "AMRN": "Amarin",
    "AMRX": "Amneal Class A",
    "AMRZ": "Amrize Ltd",
    "AMSC": "American Superconductor",
    "AMST": "Amesite",
    "AMTB": "Amerant Bancorp Class A",
    "AMTM": "Amentum",
    "AMTX": "Aemetis (DE)",
    "AMWD": "American Woodmark",
    "ANAB": "AnaptysBio",
    "ANDE": "Andersons (The)",
    "ANDG": "Andersen Class A",
    "ANGH": "Anghami",
    "ANGO": "AngioDynamics",
    "ANGX": "Angel Studios, Class A",
    "ANIK": "Anika",
    "ANIP": "ANI",
    "ANIX": "Anixa Biosciences",
    "ANL": "Adlai Nortye",
    "ANNA": "AleAnna Class A",
    "ANNX": "Annexon",
    "ANPA": "Rich Sparkle",
    "ANRO": "Alto Neuroscience",
    "ANTA": "Antalpha Platform Holding",
    "ANTX": "AN2",
    "ANVS": "Annovis Bio",
    "ANY": "Sphere 3D",
    "AOD": "abrdn Total Dynamic Dividend",
    "AOMR": "Angel Oak Mortgage REIT",
    "AON": "Aon (Ireland)",
    "AORT": "Artivion",
    "AOS": "A.O. Smith",
    "AOSL": "Alpha and Omega Semiconducto",
    "AOUT": "American Outdoor Brands",
    "AP": "Ampco-Pittsburgh",
    "APAM": "Artisan Partners Asset Manag",
    "APC": "ARKO Petroleum Class A",
    "APEI": "American Public Education",
    "APG": "APi",
    "APGE": "Apogee",
    "APH": "Amphenol",
    "API": "Agora",
    "APLD": "Applied Digital",
    "APLE": "Apple Hospitality REIT",
    "APLM": "Apollomics",
    "APLS": "Apellis",
    "APM": "Aptorum",
    "APO": "Apollo Global Management, (N",
    "APPF": "AppFolio Class A",
    "APRE": "Aprea Common stock",
    "APVO": "Aptevo",
    "APWC": "Asia Pacific Wire & Cable (B",
    "APXT": "Apex Treasury Class A Ordina",
    "APYX": "Apyx Medical",
    "AQB": "AquaBounty",
    "AQMS": "Aqua Metals",
    "AQN": "Algonquin Power & Utilities",
    "AQST": "Aquestive",
    "AR": "Antero Resources",
    "ARAI": "Arrive AI",
    "ARAY": "Accuray",
    "ARBB": "ARB IOT",
    "ARBE": "Arbe Robotics",
    "ARBK": "Argo Blockchain",
    "ARCB": "ArcBest",
    "ARCO": "Arcos Dorados Class A Shares",
    "ARCT": "Arcturus",
    "ARDC": "Ares Dynamic Credit Allocati",
    "ARDT": "Ardent Health",
    "ARE": "Alexandria Real Estate Equit",
    "AREC": "American Resources Class A",
    "ARES": "Ares Management Class A",
    "ARGX": "argenx",
    "ARI": "Apollo Commercial Real Estat",
    "ARIS": "Aris Mining",
    "ARKR": "Ark Restaurants",
    "ARL": "American Realty Investors",
    "ARMK": "Aramark",
    "AROC": "Archrock",
    "ARQ": "Arq",
    "ARQQ": "Arqit Quantum",
    "ARQT": "Arcutis Biotherapeutics",
    "ARR": "ARMOUR Residential REIT",
    "ARTL": "Artelo Biosciences",
    "ARTV": "Artiva Biotherapeutics",
    "ARVN": "Arvinas",
    "ARW": "Arrow Electronics",
    "ARWR": "Arrowhead",
    "ARX": "Accelerant Class A",
    "AS": "Amer Sports",
    "ASA": "ASA  Gold and Precious Metal",
    "ASB": "Associated Banc-Corp",
    "ASBP": "Aspire Biopharma",
    "ASC": "Ardmore Shipping",
    "ASG": "Liberty All-Star Growth Fund",
    "ASGI": "abrdn Global Infrastructure",
    "ASGN": "ASGN",
    "ASH": "Ashland",
    "ASIC": "Ategrity Specialty Insurance",
    "ASIX": "AdvanSix",
    "ASLE": "AerSale",
    "ASMB": "Assembly Biosciences",
    "ASML": "ASML Holding New York Regist",
    "ASND": "Ascendis Pharma A/S",
    "ASNS": "Actelis Networks",
    "ASO": "Academy Sports and Outdoors",
    "ASPI": "ASP Isotopes",
    "ASPN": "Aspen Aerogels",
    "ASPS": "Altisource Portfolio Solutio",
    "ASPSZ": "Altisource Portfolio Solutio",
    "ASR": "Grupo Aeroportuario del Sure",
    "ASRT": "Assertio",
    "ASRV": "AmeriServ Financial",
    "ASST": "Strive Class A",
    "ASTC": "Astrotech (DE)",
    "ASTH": "Astrana Health",
    "ASTI": "Ascent Solar",
    "ASTL": "Algoma Steel",
    "ASUR": "Asure Software Inc",
    "ASYS": "Amtech Systems",
    "ATAI": "AtaiBeckley",
    "ATAT": "Atour Lifestyle",
    "ATCX": "Atlas Critical Minerals",
    "ATEC": "Alphatec",
    "ATEN": "A10 Networks",
    "ATER": "Aterian",
    "ATEX": "Anterix",
    "ATGL": "Alpha Technology",
    "ATHE": "Alterity",
    "ATHR": "Aether",
    "ATKR": "Atkore",
    "ATLC": "Atlanticus",
    "ATLN": "Atlantic",
    "ATLO": "Ames National",
    "ATLX": "Atlas Lithium",
    "ATMU": "Atmus Filtration",
    "ATNI": "ATN",
    "ATO": "Atmos Energy",
    "ATOM": "Atomera",
    "ATON": "AlphaTON Capital",
    "ATOS": "Atossa",
    "ATPC": "Agape ATP",
    "ATR": "AptarGroup",
    "ATRA": "Atara Biotherapeutics",
    "ATRO": "Astronics",
    "ATS": "ATS",
    "ATXG": "Addentax",
    "ATYR": "aTyr Pharma",
    "AU": "AngloGold Ashanti",
    "AUB": "Atlantic Union Bankshares",
    "AUBN": "Auburn National Bancorporati",
    "AUDC": "AudioCodes",
    "AUGO": "Aura Minerals",
    "AUID": "authID",
    "AUNA": "Auna SA",
    "AUPH": "Aurinia Inc",
    "AUR": "Aurora Innovation Class A",
    "AURA": "Aura Biosciences",
    "AURE": "Aurelion",
    "AUTL": "Autolus American Depositary",
    "AUUD": "Auddia",
    "AVA": "Avista",
    "AVAH": "Aveanna Healthcare",
    "AVAV": "AeroVironment",
    "AVBC": "Avidia Bancorp",
    "AVBH": "Avidbank Common stock",
    "AVBP": "ArriVent BioPharma",
    "AVD": "American Vanguard ($0.10 Par",
    "AVIR": "Atea",
    "AVK": "Advent Convertible and Incom",
    "AVNS": "Avanos Medical",
    "AVNT": "Avient",
    "AVO": "Mission Produce",
    "AVPT": "AvePoint Class A",
    "AVR": "Anteris Global",
    "AVT": "Avnet",
    "AVTR": "Avantor",
    "AVTX": "Avalo",
    "AVX": "Avax One Technology",
    "AVXL": "Anavex Life Sciences",
    "AVY": "Avery Dennison",
    "AWF": "Alliancebernstein Global Hig",
    "AWI": "Armstrong World Industries I",
    "AWP": "abrdn Global Premier Propert",
    "AWRE": "Aware",
    "AX": "Axos Financial",
    "AXG": "Solowin Class A Ordinary Sha",
    "AXGN": "Axogen",
    "AXR": "AMREP",
    "AXS": "Axis Capital",
    "AXTA": "Axalta Coating Systems",
    "AXTI": "AXT Inc",
    "AYI": "Acuity",
    "AYTU": "Aytu BioPharma",
    "AZ": "A2Z Cust2Mate Solutions",
    "AZI": "Autozi Internet Technology (",
    "AZN": "AstraZeneca",
    "AZTA": "Azenta",
    "AZZ": "AZZ",
    "B": "Barrick Mining",
    "BAFN": "BayFirst Financial",
    "BAH": "Booz Allen Hamilton Holding",
    "BAK": "Braskem SA ADR",
    "BALL": "Ball",
    "BALY": "Bally\'s",
    "BAM": "Brookfield Asset Management",
    "BANC": "Banc of California",
    "BAND": "Bandwidth Class A",
    "BANF": "BancFirst",
    "BANL": "CBL",
    "BANX": "ArrowMark Financial",
    "BAOS": "Baosheng Media Ordinary shar",
    "BAP": "Credicorp",
    "BATRA": "Atlanta Braves Series A",
    "BATRK": "Atlanta Braves Series C",
    "BB": "BlackBerry",
    "BBAI": "BigBear.ai",
    "BBAR": "Banco BBVA Argentina ADS",
    "BBBY": "Bed Bath & Beyond",
    "BBCP": "Concrete Pumping",
    "BBDC": "Barings BDC",
    "BBGI": "Beasley Broadcast Class A",
    "BBIO": "BridgeBio Pharma",
    "BBLG": "Bone Biologics Corp",
    "BBN": "BlackRock Taxable Municipal",
    "BBNX": "Beta Bionics",
    "BBOT": "BridgeBio Oncology",
    "BBSI": "Barrett Business Services",
    "BBT": "Beacon Financial Common stoc",
    "BBU": "Brookfield Business Partners",
    "BBUC": "Brookfield Business Class A",
    "BBVA": "Banco Bilbao Vizcaya Argenta",
    "BBW": "Build-A-Bear Workshop",
    "BBWI": "Bath & Body Works",
    "BC": "Brunswick",
    "BCAB": "BioAtla",
    "BCAL": "California BanCorp",
    "BCAT": "BlackRock Capital Allocation",
    "BCAX": "Bicara",
    "BCBP": "BCB Bancorp (NJ)",
    "BCC": "Boise Cascade, L.L.C.",
    "BCDA": "BioCardia",
    "BCE": "BCE",
    "BCG": "Binah Capital",
    "BCH": "Banco De Chile ADS",
    "BCIC": "BCP Investment",
    "BCML": "BayCom Corp",
    "BCO": "Brinks (The)",
    "BCPC": "Balchem",
    "BCRX": "BioCryst",
    "BCS": "Barclays",
    "BCSS": "Bain Capital GSS Investment",
    "BCTX": "BriaCell",
    "BCX": "BlackRock Resources",
    "BCYC": "Bicycle",
    "BDC": "Belden Inc",
    "BDCI": "BTC Development",
    "BDJ": "Blackrock Enhanced Equity Di",
    "BDMD": "Baird Medical Investment Ltd",
    "BDN": "Brandywine Realty Trust",
    "BDRX": "Biodexa American Depositary",
    "BDSX": "Biodesix",
    "BDTX": "Black Diamond",
    "BE": "Bloom Energy Class A",
    "BEBE": "TGE Value Creative Solutions",
    "BEEM": "Beam Global",
    "BEEP": "Mobile Infrastructure",
    "BELFA": "Bel Fuse Class A",
    "BELFB": "Bel Fuse Class B",
    "BENF": "Beneficient Class A",
    "BEP": "Brookfield Renewable Partner",
    "BEPC": "Brookfield Renewable Brookfi",
    "BEPH": "Brookfield BRP (Canada) 4.62",
    "BEPI": "Brookfield BRP (Canada) 4.87",
    "BEPJ": "Brookfield BRP (Canada) 7.25",
    "BETA": "Beta, Class A",
    "BETR": "Better Home & Finance Holdin",
    "BF-A": "Brown Forman",
    "BF-B": "Brown Forman",
    "BFC": "Bank First",
    "BFLY": "Butterfly Network, Class A",
    "BFRG": "Bullfrog AI",
    "BFRI": "Biofrontera",
    "BFS": "Saul Centers",
    "BFST": "Business First Bancshares",
    "BGB": "Blackstone Strategic Credit",
    "BGC": "BGC Class A",
    "BGH": "Barings Global Short Duratio",
    "BGIN": "Bgin Blockchain",
    "BGL": "Blue Gold",
    "BGLC": "BioNexus Gene Lab Corp Commo",
    "BGM": "BGM",
    "BGMS": "Bio Green Med Solution",
    "BGR": "BlackRock Energy and Resourc",
    "BGS": "B&G Foods",
    "BGSF": "BGSF",
    "BGSI": "Boyd Services",
    "BGT": "BlackRock Floating Rate Inco",
    "BGX": "Blackstone Long Short Credit",
    "BGY": "Blackrock Enhanced Dividend",
    "BH": "Biglari Class B",
    "BHC": "Bausch Health Companies",
    "BHE": "Benchmark Electronics",
    "BHF": "Brighthouse Financial",
    "BHK": "Blackrock Core Bond Trust",
    "BHR": "Braemar Hotels & Resorts",
    "BHRB": "Burke & Herbert Financial Se",
    "BHST": "BioHarvest Sciences",
    "BHV": "BlackRock Virginia Municipal",
    "BHVN": "Biohaven",
    "BIAF": "bioAffinity",
    "BIDU": "Baidu ADS",
    "BILI": "Bilibili",
    "BIO": "Bio-Rad Laboratories, Class",
    "BIO-B": "Bio-Rad Laboratories",
    "BIOA": "BioAge Labs",
    "BIOX": "Bioceres Crop Solutions",
    "BIP": "Brookfield Infrastructure Pa",
    "BIPC": "Brookfield Infrastructure Br",
    "BIPI": "BIP Bermuda I 5.125% Perpetu",
    "BIRD": "Allbirds Class A",
    "BIRK": "Birkenstock Holding",
    "BIT": "BlackRock Multi-Sector Incom",
    "BITF": "Bitfarms",
    "BIVI": "BioVie Class A",
    "BIYA": "Baiya",
    "BJDX": "Bluejay Diagnostics",
    "BKD": "Brookdale Senior Living",
    "BKE": "Buckle, (The)",
    "BKH": "Black Hills",
    "BKKT": "Bakkt, Class A",
    "BKSY": "BlackSky Technology Class A",
    "BKT": "BlackRock Income Trust (The)",
    "BKU": "BankUnited",
    "BKV": "BKV",
    "BKYI": "BIO-key",
    "BL": "BlackLine",
    "BLBD": "Blue Bird",
    "BLCO": "Bausch + Lomb",
    "BLD": "TopBuild",
    "BLDP": "Ballard Power Systems",
    "BLFS": "BioLife Solutions",
    "BLFY": "Blue Foundry Bancorp",
    "BLIN": "Bridgeline Digital",
    "BLIV": "BeLive Ordinary Share",
    "BLKB": "Blackbaud",
    "BLLN": "BillionToOne Class A",
    "BLMN": "Bloomin\' Brands",
    "BLND": "Blend Labs, Class A",
    "BLNE": "Beeline",
    "BLRX": "BioLineRx",
    "BLSH": "Bullish",
    "BLTE": "Belite Bio Inc",
    "BLW": "Blackrock Duration Income Tr",
    "BLX": "Banco Latinoamericano de Com",
    "BLZE": "Backblaze Class A",
    "BMA": "Banco Macro  ADR (representi",
    "BMBL": "Bumble Class A",
    "BME": "Blackrock Health Sciences Tr",
    "BMEA": "Biomea Fusion",
    "BMEZ": "BlackRock Health Sciences Te",
    "BMGL": "Basel Medical Ltd",
    "BMHL": "Bluemount",
    "BMI": "Badger Meter",
    "BMM": "Blue Moon Metals",
    "BMN": "BlackRock 2037 Municipal Tar",
    "BMO": "Bank Of Montreal",
    "BMR": "Beamr Imaging Ordinary Share",
    "BMRA": "Biomerica",
    "BMRC": "Bank of Marin Bancorp",
    "BN": "Brookfield Class A Voting Sh",
    "BNAI": "Brand Engagement Network",
    "BNBX": "BNB Plus",
    "BNC": "CEA Industries",
    "BNED": "Barnes & Noble Education",
    "BNGO": "Bionano Genomics",
    "BNJ": "Brookfield Finance 4.50% Per",
    "BNKK": "Bonk",
    "BNL": "Broadstone Net Lease",
    "BNR": "Burning Rock Biotech",
    "BNRG": "Brenmiller Energy Ltd",
    "BNS": "Bank Nova Scotia Halifax Pfd",
    "BNT": "Brookfield Wealth Solutions",
    "BNTC": "Benitec Biopharma",
    "BNTX": "BioNTech American Depositary",
    "BNZI": "Banzai Class A",
    "BOBS": "Bob\'s Discount Furniture",
    "BOC": "Boston Omaha Class A",
    "BODI": "The Beachbody Class A",
    "BOE": "Blackrock Enhanced Global Di",
    "BOF": "BranchOut Food",
    "BOH": "Bank of Hawaii",
    "BOLD": "Boundless Bio",
    "BOLT": "Bolt Biotherapeutics",
    "BON": "Bon Natural Life",
    "BOOM": "DMC Global",
    "BORR": "Borr Drilling",
    "BOSC": "B.O.S. Better Online Solutio",
    "BOTJ": "Bank of the James Financial",
    "BOW": "Bowhead Specialty",
    "BOXL": "Boxlight Class A",
    "BP": "BP p.l.c.",
    "BPOP": "Popular",
    "BPRE": "Bluerock Private Real Estate",
    "BPRN": "Princeton Bancorp (PA)",
    "BR": "Broadridge Financial Solutio",
    "BRAG": "Bragg Gaming",
    "BRAI": "Braiin",
    "BRBI": "BRBI BR Partners ADSs",
    "BRBR": "BellRing Brands",
    "BRC": "Brady",
    "BRCB": "Black Rock Coffee Bar Class",
    "BRCC": "BRC Class A",
    "BRFH": "Barfresh Food",
    "BRID": "Bridgford Foods",
    "BRK-A": "Berkshire Hathaway",
    "BRKR": "Bruker",
    "BRLS": "Borealis Foods Class A",
    "BRLT": "Brilliant Earth Class A",
    "BRNS": "Barinthus Biotherapeutics",
    "BRO": "Brown & Brown",
    "BROS": "Dutch Bros Class A",
    "BRR": "ProCap Financial",
    "BRSL": "Brightstar Lottery Trading u",
    "BRSP": "BrightSpire Capital, Class A",
    "BRT": "BRT Apartments (MD)",
    "BRTX": "BioRestorative Therapies (NV",
    "BRW": "Saba Capital Income & Opport",
    "BRX": "Brixmor Property",
    "BSAC": "Banco Santander - Chile ADS",
    "BSBK": "Bogota Financial",
    "BSET": "Bassett Furniture Industries",
    "BSL": "Blackstone Senior Floating R",
    "BSRR": "Sierra Bancorp",
    "BST": "BlackRock Science and Techno",
    "BSTZ": "BlackRock Science and Techno",
    "BSVN": "Bank7 Common stock",
    "BSY": "Bentley Systems Class B",
    "BTAI": "BioXcel",
    "BTBD": "BT Brands",
    "BTBT": "Bit Digital",
    "BTCS": "BTCS",
    "BTCT": "BTC Digital",
    "BTDR": "Bitdeer",
    "BTE": "Baytex Energy Corp",
    "BTGO": "BitGo, Class A",
    "BTI": "British American Tobacco  In",
    "BTM": "Bitcoin Depot Class A",
    "BTMD": "Biote Class A",
    "BTO": "John Hancock Financial Oppor",
    "BTOC": "Armlogi Holding",
    "BTOG": "Bit Origin",
    "BTQ": "BTQ",
    "BTSG": "BrightSpring Health Services",
    "BTT": "BlackRock Municipal 2030 Tar",
    "BTTC": "Black Titan",
    "BTU": "Peabody Energy",
    "BTX": "BlackRock Technology and Pri",
    "BTZ": "BlackRock Credit Allocation",
    "BUD": "Anheuser-Busch Inbev SA Spon",
    "BUI": "BlackRock Utility, Infrastru",
    "BULL": "Webull",
    "BUR": "Burford Capital",
    "BURL": "Burlington Stores",
    "BUSE": "First Busey Class A",
    "BUUU": "BUUU Class A Ordinary Share",
    "BV": "BrightView",
    "BVC": "BitVentures",
    "BVFL": "BV Financial",
    "BVN": "Buenaventura Mining",
    "BVS": "Bioventus Class A",
    "BW": "Babcock & Wilcox Enterprises",
    "BWA": "BorgWarner",
    "BWAY": "BrainsWay",
    "BWB": "Bridgewater Bancshares",
    "BWEN": "Broadwind",
    "BWFG": "Bankwell Financial",
    "BWG": "BrandywineGLOBAL Global Inco",
    "BWIN": "The Baldwin Insurance Class",
    "BWLP": "BW LPG",
    "BWMN": "Bowman Consulting",
    "BWMX": "Betterware de Mexico,P.I. de",
    "BWXT": "BWX",
    "BX": "Blackstone",
    "BXC": "Bluelinx",
    "BXMT": "Blackstone Mortgage Trust",
    "BXMX": "Nuveen S&P 500 Buy-Write Inc",
    "BY": "Byline Bancorp",
    "BYAH": "Park Ha Biological Technolog",
    "BYD": "Boyd Gaming",
    "BYFC": "Broadway Financial Class A",
    "BYND": "Beyond Meat",
    "BYRN": "Byrna",
    "BYSI": "BeyondSpring",
    "BZ": "KANZHUN LIMITED",
    "BZAI": "Blaize",
    "BZFD": "BuzzFeed Class A",
    "BZH": "Beazer Homes USA",
    "BZUN": "Baozun",
    "CAAP": "Corporacion America Airports",
    "CAAS": "China Automotive Systems Ord",
    "CABO": "Cable One",
    "CABR": "Caring Brands",
    "CAC": "Camden National",
    "CACC": "Credit Acceptance",
    "CACI": "CACI, Class A",
    "CADL": "Candel",
    "CAE": "CAE",
    "CAEP": "Cantor Equity Partners III",
    "CAF": "Morgan Stanley China A Share",
    "CAH": "Cardinal Health",
    "CAI": "Caris Life Sciences",
    "CAL": "Caleres",
    "CALC": "CalciMedica",
    "CALM": "Cal-Maine Foods",
    "CALY": "Callaway Golf",
    "CAMP": "CAMP4",
    "CAMT": "Camtek",
    "CAN": "Canaan",
    "CANG": "Cango",
    "CAPR": "Capricor",
    "CAPS": "Capstone Holding",
    "CAPT": "Captivision",
    "CAR": "Avis Budget",
    "CARE": "Carter Bankshares",
    "CARG": "CarGurus Class A",
    "CARL": "Carlsmed",
    "CARS": "Cars.com",
    "CART": "Maplebear",
    "CASH": "Pathward Financial",
    "CASS": "Cass Information Systems Inc",
    "CAST": "FreeCast Class A",
    "CAVA": "CAVA",
    "CBAN": "Colony Bankcorp",
    "CBAT": "CBAK Energy Technology",
    "CBC": "Central Bancompany Class A",
    "CBFV": "CB Financial Services",
    "CBIO": "Crescent Biopharma",
    "CBK": "Commercial Bancgroup",
    "CBL": "CBL & Associates Properties",
    "CBLL": "CeriBell",
    "CBNA": "Chain Bridge Bancorp, Class",
    "CBNK": "Capital Bancorp",
    "CBRE": "CBRE Inc Class A",
    "CBRL": "Cracker Barrel Old Country S",
    "CBSH": "Commerce Bancshares",
    "CBT": "Cabot",
    "CBU": "Community Financial System",
    "CBUS": "Cibus Class A",
    "CBZ": "CBIZ",
    "CC": "Chemours (The)",
    "CCB": "Coastal Financial",
    "CCBG": "Capital City Bank",
    "CCC": "CCC Intelligent Solutions",
    "CCCC": "C4",
    "CCD": "Calamos Dynamic Convertible",
    "CCEC": "Capital Clean Energy Carrier",
    "CCEP": "Coca-Cola Europacific Partne",
    "CCG": "Cheche",
    "CCHH": "CCH Ltd",
    "CCIF": "Carlyle Credit Income Fund S",
    "CCIX": "Churchill Capital Corp IX",
    "CCJ": "Cameco",
    "CCK": "Crown",
    "CCLD": "CareCloud",
    "CCNE": "CNB Financial",
    "CCO": "Clear Channel Outdoor",
    "CCOI": "Cogent Communications",
    "CCS": "Century Communities",
    "CCSI": "Consensus Cloud Solutions",
    "CCTG": "CCSC Technology",
    "CCU": "Compania Cervecerias Unidas",
    "CCXI": "Churchill Capital Corp XI",
    "CCZ": "Comcast ZONES",
    "CD": "Chaince Digital",
    "CDE": "Coeur Mining",
    "CDIO": "Cardio Diagnostics Common st",
    "CDLX": "Cardlytics",
    "CDNA": "CareDx",
    "CDNL": "Cardinal Infrastructure Clas",
    "CDP": "COPT Defense Properties",
    "CDRE": "Cadre",
    "CDRO": "Codere Online Luxembourg",
    "CDT": "CDT Equity",
    "CDTG": "CDT Environmental Technology",
    "CDW": "CDW",
    "CDXS": "Codexis",
    "CDZI": "CADIZ",
    "CDZIP": "Cadiz Depositary Shares",
    "CECO": "CECO Environmental",
    "CEE": "The Central and Eastern Euro",
    "CELC": "Celcuity",
    "CELH": "Celsius",
    "CELU": "Celularity Class A",
    "CELZ": "Creative Medical Technology",
    "CENN": "Cenntro",
    "CENT": "Central Garden & Pet",
    "CENTA": "Central Garden & Pet Class A",
    "CENX": "Century Aluminum",
    "CEPF": "Cantor Equity Partners IV",
    "CEPO": "Cantor Equity Partners I",
    "CEPS": "Cantor Equity Partners VI",
    "CEPT": "Cantor Equity Partners II Cl",
    "CEPV": "Cantor Equity Partners V",
    "CERS": "Cerus",
    "CETX": "Cemtrex",
    "CETY": "Clean Energy",
    "CEVA": "CEVA",
    "CFBK": "CF Bankshares",
    "CFFI": "C&F Financial",
    "CFFN": "Capitol Federal Financial",
    "CFND": "C1 Fund",
    "CFR": "Cullen/Frost Bankers",
    "CG": "The Carlyle",
    "CGAU": "Centerra Gold",
    "CGC": "Canopy Growth",
    "CGCT": "Cartesian Growth III",
    "CGEM": "Cullinan",
    "CGEN": "Compugen",
    "CGNT": "Cognyte Software",
    "CGO": "Calamos Global Total Return",
    "CGON": "CG Oncology Common stock",
    "CGTL": "Creative Global Technology",
    "CGTX": "Cognition",
    "CHA": "Chagee",
    "CHAI": "Core AI",
    "CHCI": "Comstock Holding Companies C",
    "CHCO": "City Holding",
    "CHCT": "Community Healthcare Trust",
    "CHDN": "Churchill Downs",
    "CHE": "Chemed Corp",
    "CHEF": "The Chefs\' Warehouse",
    "CHGG": "Chegg",
    "CHH": "Choice Hotels",
    "CHI": "Calamos Convertible Opportun",
    "CHKP": "Check Point Software",
    "CHMG": "Chemung Financial Corp",
    "CHMI": "Cherry Hill Mortgage Investm",
    "CHNR": "China Natural Resources",
    "CHR": "Cheer Holding Class A Ordina",
    "CHRS": "Coherus Oncology",
    "CHSN": "Chanson Holding",
    "CHT": "Chunghwa Telecom",
    "CHW": "Calamos Global Dynamic Incom",
    "CHY": "Calamos Convertible and High",
    "CHYM": "Chime Financial Class A",
    "CIA": "Citizens, Class A ($1.00 Par",
    "CIF": "MFS Intermediate High Income",
    "CIFR": "Cipher Digital",
    "CIGI": "Colliers Subordinate Voting",
    "CIGL": "Concorde Ltd",
    "CII": "BlackRock Enhanced Large Cap",
    "CIIT": "Tianci",
    "CIM": "Chimera Investment",
    "CING": "Cingulate",
    "CINT": "CI&T Inc Class A",
    "CION": "CION Investment",
    "CISO": "CISO Global",
    "CISS": "C3is",
    "CIVB": "Civista Bancshares",
    "CJMB": "Callan JMB",
    "CLAR": "Clarus",
    "CLB": "Core Laboratories",
    "CLBK": "Columbia Financial",
    "CLBT": "Cellebrite DI",
    "CLDX": "Celldex",
    "CLF": "Cleveland-Cliffs",
    "CLFD": "Clearfield",
    "CLGN": "CollPlant Biotechnologies Lt",
    "CLH": "Clean Harbors",
    "CLIK": "Click Class A Ordinary Share",
    "CLIR": "ClearSign (DE)",
    "CLLS": "Cellectis",
    "CLMB": "Climb Global Solutions",
    "CLMT": "Calumet",
    "CLNE": "Clean Energy Fuels",
    "CLNN": "Clene",
    "CLPR": "Clipper Realty",
    "CLPS": "CLPS Incorporation",
    "CLPT": "ClearPoint Neuro",
    "CLRB": "Cellectar Biosciences",
    "CLRO": "ClearOne (DE)",
    "CLS": "Celestica",
    "CLST": "Catalyst Bancorp",
    "CLVT": "Clarivate Plc",
    "CLW": "Clearwater Paper",
    "CLWT": "Euro Tech",
    "CLYM": "Climb Bio",
    "CM": "Canadian Imperial Bank of Co",
    "CMBM": "Cambium Networks",
    "CMBT": "CMB.TECH NV",
    "CMC": "Commercial Metals",
    "CMCO": "Columbus McKinnon",
    "CMCT": "Creative Media & Community T",
    "CMDB": "Costamare Bulkers",
    "CMI": "Cummins",
    "CMII": "Columbus Circle Capital Corp",
    "CMMB": "Chemomab American Depositary",
    "CMND": "Clearmind Medicine",
    "CMP": "Compass Minerals Intl Inc",
    "CMPR": "Cimpress (Ireland)",
    "CMPS": "COMPASS Pathways Plc",
    "CMPX": "Compass",
    "CMRC": "Commerce.com Series 1",
    "CMRE": "Costamare $0.0001 par value",
    "CMTG": "Claros Mortgage Trust",
    "CMTL": "Comtech Telecommunications",
    "CMTV": "Community Bancorp.",
    "CMU": "MFS Municipal Income Trust",
    "CNA": "CNA Financial",
    "CNCK": "Coincheck",
    "CNDT": "Conduent",
    "CNET": "ZW Data Action",
    "CNEY": "CN Energy",
    "CNH": "CNH Industrial",
    "CNK": "Cinemark Inc Cinemark",
    "CNM": "Core & Main, Class A",
    "CNMD": "CONMED",
    "CNNE": "Cannae",
    "CNO": "CNO Financial",
    "CNOB": "ConnectOne Bancorp",
    "CNQ": "Canadian Natural Resources",
    "CNR": "Core Natural Resources",
    "CNS": "Cohen & Steers Inc",
    "CNSP": "CNS",
    "CNTA": "Centessa",
    "CNTB": "Connect Biopharma",
    "CNTN": "Canton Strategic",
    "CNTX": "Context",
    "CNTY": "Century Casinos",
    "CNVS": "Cineverse Class A",
    "CNX": "CNX Resources",
    "CNXN": "PC Connection",
    "COCH": "Envoy Medical Class A",
    "COCO": "The Vita Coco",
    "COCP": "Cocrystal Pharma",
    "CODA": "Coda Octopus Common stock",
    "CODX": "Co-Diagnostics",
    "COEP": "Coeptis",
    "COFS": "ChoiceOne Financial Services",
    "COGT": "Cogent Biosciences",
    "COHR": "Coherent",
    "COKE": "Coca-Cola Consolidated",
    "COLB": "Columbia Banking System",
    "COLL": "Collegium Pharmaceutical",
    "COLM": "Columbia Sportswear",
    "COMP": "Compass, Class A",
    "CON": "Concentra Parent",
    "COO": "The Cooper Companies",
    "COOK": "Traeger",
    "COOT": "Australian Oilseeds",
    "COR": "Cencora",
    "CORT": "Corcept",
    "CORZ": "Core Scientific",
    "CORZZ": "Core Scientific Tranche 2 Wa",
    "COSM": "Cosmos Health",
    "COSO": "CoastalSouth Bancshares",
    "COUR": "Coursera",
    "COYA": "Coya",
    "CPA": "Copa, Class A",
    "CPAY": "Corpay",
    "CPBI": "Central Plains Bancshares",
    "CPF": "Central Pacific Financial Co",
    "CPHC": "Canterbury Park Holding \'New",
    "CPIX": "Cumberland",
    "CPK": "Chesapeake Utilities",
    "CPNG": "Coupang, Class A",
    "CPOP": "Pop Culture Ltd",
    "CPRX": "Catalyst",
    "CPS": "Cooper-Standard",
    "CPSH": "CPS",
    "CPSS": "Consumer Portfolio Services",
    "CPZ": "Calamos Long/Short Equity &",
    "CQP": "Cheniere Energy Partners, LP",
    "CR": "Crane",
    "CRAI": "CRA",
    "CRBG": "Corebridge Financial",
    "CRBP": "Corbus",
    "CRBU": "Caribou Biosciences",
    "CRC": "California Resources",
    "CRCL": "Circle Internet, Class A",
    "CRCT": "Cricut Class A",
    "CRD-A": "Crawford &",
    "CRD-B": "Crawford &",
    "CRDF": "Cardiff Oncology",
    "CRDL": "Cardiol Class A",
    "CRDO": "Credo Technology Holding Ltd",
    "CRE": "Cre8 Enterprise",
    "CREG": "Smart Powerr",
    "CRESY": "CresudC.I.F. y A.",
    "CREX": "Creative Realities",
    "CRGO": "Freightos Ordinary shares",
    "CRGY": "Crescent Energy Class A",
    "CRH": "CRH",
    "CRI": "Carter\'s",
    "CRIS": "Curis",
    "CRK": "Comstock Resources",
    "CRMD": "CorMedix",
    "CRML": "Critical Metals",
    "CRMT": "America\'s Car-Mart Inc",
    "CRNC": "Cerence",
    "CRNT": "Ceragon Networks",
    "CRNX": "Crinetics",
    "CRON": "Cronos Common Share",
    "CRS": "Carpenter Technology",
    "CRSR": "Corsair Gaming",
    "CRT": "Cross Timbers Royalty Trust",
    "CRTO": "Criteo",
    "CRUS": "Cirrus Logic",
    "CRVL": "CorVel",
    "CRVO": "CervoMed",
    "CRVS": "Corvus",
    "CRWS": "Crown Crafts Inc",
    "CRWV": "CoreWeave Class A",
    "CSAI": "Cloudastructure Class A",
    "CSAN": "Cosan ADS",
    "CSBR": "Champions Oncology",
    "CSGP": "CoStar",
    "CSGS": "CSG Systems",
    "CSIQ": "Canadian Solar (ON)",
    "CSL": "Carlisle Companies",
    "CSPI": "CSP",
    "CSQ": "Calamos Strategic Total Retu",
    "CSTE": "Caesarstone",
    "CSTL": "Castle Biosciences",
    "CSTM": "Constellium (France)",
    "CSV": "Carriage Services",
    "CSW": "CSW Industrials",
    "CTEV": "Claritev Class A",
    "CTKB": "Cytek Biosciences",
    "CTLP": "Cantaloupe",
    "CTMX": "CytomX",
    "CTNM": "Contineum Class A",
    "CTNT": "Cheetah Net Supply Chain Ser",
    "CTO": "CTO Realty Growth",
    "CTOR": "Citius Oncology",
    "CTOS": "Custom Truck One Source",
    "CTRA": "Coterra Energy",
    "CTRE": "CareTrust REIT",
    "CTRI": "Centuri",
    "CTRM": "Castor Maritime",
    "CTRN": "Citi Trends",
    "CTS": "CTS",
    "CTSO": "Cytosorbents",
    "CTW": "CTW Cayman",
    "CTXR": "Citius",
    "CUB": "Lionheart",
    "CUBI": "Customers Bancorp",
    "CUE": "Cue Biopharma",
    "CUK": "Carnival Plc ADS ADS",
    "CULP": "Culp",
    "CUPR": "Cuprina (Cayman)",
    "CURB": "Curbline Properties",
    "CURI": "CuriosityStream Class A",
    "CURR": "Currenc",
    "CURV": "Torrid",
    "CURX": "Curanex Inc",
    "CUZ": "Cousins Properties",
    "CV": "CapsoVision",
    "CVE": "Cenovus Energy Inc",
    "CVEO": "Civeo (Canada)",
    "CVGI": "Commercial Vehicle",
    "CVI": "CVR Energy",
    "CVKD": "Cadrenal",
    "CVLG": "Covenant Logistics, Class A",
    "CVRX": "CVRx",
    "CVSA": "Covista",
    "CVV": "CVD Equipment",
    "CWAN": "Clearwater Analytics, Class",
    "CWBC": "Community West Bancshares",
    "CWD": "CaliberCos Class A",
    "CWEN": "Clearway Energy, Class C",
    "CWH": "Camping World, Class A",
    "CWK": "Cushman & Wakefield",
    "CWT": "California Water Service",
    "CX": "Cemex,B. de C.V. Sponsored A",
    "CXAI": "CXApp Class A",
    "CXDO": "Crexendo",
    "CXE": "MFS High Income Municipal Tr",
    "CXH": "MFS Investment Grade Municip",
    "CXM": "Sprinklr, Class A",
    "CXT": "Crane NXT",
    "CXW": "CoreCivic",
    "CYCN": "Cyclerion",
    "CYCU": "Cycurion",
    "CYD": "China Yuchai",
    "CYH": "Community Health Systems",
    "CYN": "Cyngn",
    "CYPH": "Cypherpunk",
    "CYRX": "CryoPort",
    "CYTK": "Cytokinetics",
    "CZFS": "Citizens Financial Services",
    "CZNC": "Citizens & Northern Corp",
    "CZR": "Caesars Entertainment",
    "CZWI": "Citizens Community Bancorp",
    "DAC": "Danaos",
    "DAIC": "CID HoldCo",
    "DAIO": "Data I/O",
    "DAKT": "Daktronics",
    "DAN": "Dana",
    "DAR": "Darling Ingredients",
    "DARE": "Dare Bioscience",
    "DASH": "DoorDash Class A",
    "DAVE": "Dave Class A",
    "DAWN": "Day One Biopharmaceuticals",
    "DB": "Deutsche Bank",
    "DBD": "Diebold Nixdorf Common stock",
    "DBGI": "Digital Brands",
    "DBI": "Designer Brands Class A",
    "DBL": "DoubleLine Opportunistic Cre",
    "DBRG": "DigitalBridge",
    "DBVT": "DBV",
    "DBX": "Dropbox Class A",
    "DCBO": "Docebo",
    "DCGO": "DocGo",
    "DCH": "Dauch",
    "DCI": "Donaldson",
    "DCO": "Ducommun",
    "DCOM": "Dime Community Bancshares",
    "DCOY": "Decoy",
    "DCTH": "Delcath Systems",
    "DCX": "Digital Currency X Technolog",
    "DDD": "3D Systems",
    "DDI": "DoubleDown Interactive",
    "DDS": "Dillard\'s",
    "DDT": "Dillard\'s Capital Trust I",
    "DEA": "Easterly Government Properti",
    "DEC": "Diversified Energy",
    "DEFT": "Defi",
    "DEI": "Douglas Emmett",
    "DEO": "Diageo",
    "DERM": "Journey Medical",
    "DEVS": "DevvStream",
    "DFDV": "DeFi Development",
    "DFH": "Dream Finders Homes, Class A",
    "DFIN": "Donnelley Financial Solution",
    "DFLI": "Dragonfly Energy (NV)",
    "DFNS": "T3 Defense",
    "DFSC": "DEFSEC",
    "DFTX": "Definium",
    "DGICA": "Donegal Class A",
    "DGICB": "Donegal Class B",
    "DGII": "Digi",
    "DGNX": "Diginex",
    "DGX": "Quest Diagnostics",
    "DGXX": "Digi Power X Subordinate Vot",
    "DH": "Definitive Healthcare Class",
    "DHC": "Diversified Healthcare Trust",
    "DHF": "BNY Mellon High Yield Strate",
    "DHIL": "Diamond Hill Investment Clas",
    "DHT": "DHT",
    "DHX": "DHI",
    "DIAX": "Nuveen Dow 30SM Dynamic Over",
    "DIBS": "1stdibs.com",
    "DIN": "Dine Brands Global",
    "DINO": "HF Sinclair",
    "DJCO": "Daily Journal (S.C.)",
    "DJT": "Trump Media & Technology",
    "DK": "Delek US",
    "DKI": "DarkIris",
    "DKS": "Dick\'s Sporting Goods Inc",
    "DLB": "Dolby Laboratories",
    "DLHC": "DLH",
    "DLNG": "Dynagas LNG Partners LP Comm",
    "DLO": "DLocal Class A",
    "DLPN": "Dolphin Entertainment",
    "DLTH": "Duluth Class B",
    "DLX": "Deluxe",
    "DLXY": "Delixy",
    "DLY": "DoubleLine Yield Opportuniti",
    "DMA": "Destra Multi-Alternative Fun",
    "DMAC": "DiaMedica",
    "DMB": "BNY Mellon Municipal Bond In",
    "DMO": "Western Asset Mortgage Oppor",
    "DMRA": "Damora",
    "DMRC": "Digimarc",
    "DNA": "Ginkgo Bioworks, Class A",
    "DNLI": "Denali",
    "DNMX": "Dynamix III",
    "DNP": "DNP Select Income Fund",
    "DNTH": "Dianthus",
    "DOC": "Healthpeak Properties",
    "DOCU": "DocuSign",
    "DOGZ": "Dogness (International) Clas",
    "DOLE": "Dole",
    "DOMH": "Dominari",
    "DOO": "BRP Common Subordinate Votin",
    "DORM": "Dorman Products",
    "DOUG": "Douglas Elliman",
    "DOX": "Amdocs",
    "DOYU": "DouYu ADS",
    "DPG": "Duff & Phelps Utility and In",
    "DPRO": "Draganfly",
    "DPZ": "Domino\'s Pizza Inc",
    "DRCT": "Direct Digital Class A",
    "DRH": "Diamondrock Hospitality",
    "DRIO": "DarioHealth",
    "DRMA": "Dermata",
    "DRS": "Leonardo DRS",
    "DRTS": "Alpha Tau Medical",
    "DRUG": "Bright Minds Biosciences",
    "DSGN": "Design",
    "DSGR": "Distribution Solutions",
    "DSGX": "Descartes Systems (The)",
    "DSL": "DoubleLine Income Solutions",
    "DSM": "BNY Mellon Strategic Municip",
    "DSP": "Viant Technology Class A",
    "DSU": "Blackrock Debt Strategies Fu",
    "DSWL": "Deswell Industries",
    "DSX": "Diana Shipping inc. common s",
    "DSY": "Big Tree Cloud",
    "DT": "Dynatrace",
    "DTCK": "Davis Commodities",
    "DTCX": "Datacentrex",
    "DTE": "DTE Energy",
    "DTF": "DTF Tax-Free Income 2028 Ter",
    "DTI": "Drilling Tools",
    "DTIL": "Precision BioSciences",
    "DTM": "DT Midstream",
    "DTSS": "Datasea",
    "DTST": "Data Storage",
    "DUO": "Fangdd Network",
    "DUOL": "Duolingo Class A",
    "DUOT": "Duos",
    "DV": "DoubleVerify",
    "DVLT": "Datavault AI",
    "DWSN": "Dawson Geophysical",
    "DWTX": "Dogwood",
    "DX": "Dynex Capital",
    "DXC": "DXC Technology",
    "DXLG": "Destination XL",
    "DXPE": "DXP Enterprises",
    "DXR": "Daxor",
    "DXST": "Decent Holding Inc",
    "DXYZ": "Destiny Tech100",
    "DYAI": "Dyadic",
    "DYN": "Dyne",
    "DYOR": "Insight Digital Partners II",
    "E": "ENI S.p.A.",
    "EAF": "GrafTech",
    "EAI": "Entergy Arkansas, LLC First",
    "EARN": "Ellington Credit",
    "EAT": "Brinker",
    "EBC": "Eastern Bankshares",
    "EBF": "Ennis",
    "EBMT": "Eagle Bancorp Montana",
    "EBON": "Ebang",
    "EBS": "Emergent BioSolutions",
    "ECAT": "BlackRock ESG Capital Alloca",
    "ECBK": "ECB Bancorp",
    "ECC": "Eagle Point Credit",
    "ECG": "Everus Construction",
    "ECO": "Okeanis Eco Tankers",
    "ECOR": "electroCore",
    "ECPG": "Encore Capital Inc",
    "ECVT": "Ecovyst",
    "ECX": "ECARX Class A Ordinary share",
    "ED": "Consolidated Edison",
    "EDAP": "EDAP TMS",
    "EDBL": "Edible Garden",
    "EDD": "Morgan Stanley Emerging Mark",
    "EDF": "Virtus Stone Harbor Emerging",
    "EDHL": "Everbright Digital Holding",
    "EDRY": "EuroDry",
    "EDSA": "Edesa Biotech",
    "EDTK": "Skillful Craftsman Education",
    "EDU": "New Oriental Education & Tec",
    "EDUC": "Educational Development",
    "EE": "Excelerate Energy, Class A",
    "EEA": "The European Equity Fund",
    "EEFT": "Euronet Worldwide",
    "EEIQ": "EpicQuest Education",
    "EEX": "Emerald Holding",
    "EFC": "Ellington Financial",
    "EFOI": "Energy Focus",
    "EFR": "Eaton Vance Senior Floating-",
    "EFSI": "Eagle Financial Services Inc",
    "EFT": "Eaton Vance Floating Rate In",
    "EFX": "Equifax",
    "EFXT": "Enerflex Ltd",
    "EG": "Everest",
    "EGAN": "eGain",
    "EGP": "EastGroup Properties",
    "EGY": "VAALCO Energy",
    "EH": "EHang ADS",
    "EHAB": "Enhabit",
    "EHC": "Encompass Health",
    "EHGO": "Eshallgo",
    "EHI": "Western Asset Global High In",
    "EHLD": "Euroholdings",
    "EHTH": "eHealth",
    "EIC": "Eagle Point Income",
    "EIG": "Employers Inc",
    "EIKN": "Eikon",
    "EIX": "Edison",
    "EJH": "E-Home Household Service",
    "EKSO": "Ekso Bionics",
    "ELAB": "PMGC",
    "ELAN": "Elanco Animal Health",
    "ELBM": "Electra Battery Materials",
    "ELC": "Entergy Louisiana, Collatera",
    "ELDN": "Eledon",
    "ELE": "Elemental Royalty",
    "ELME": "Elme Communities",
    "ELOG": "Eastern",
    "ELS": "Equity Lifestyle Properties",
    "ELSE": "Electro-Sensors",
    "ELTK": "Eltek",
    "ELTX": "Elicio",
    "ELUT": "Elutia Class A",
    "ELV": "Elevance Health",
    "ELVA": "Electrovaya",
    "ELVR": "Elevra Lithium",
    "ELWT": "Elauwit Connection",
    "EM": "Smart Share Global",
    "EMA": "Emera",
    "EMAT": "Evolution Metals &",
    "EMBC": "Embecta",
    "EMBJ": "Embraer",
    "EMD": "Western Asset Emerging Marke",
    "EME": "EMCOR",
    "EMF": "Templeton Emerging Markets F",
    "EML": "Eastern (The)",
    "EMO": "ClearBridge Energy Midstream",
    "EMP": "Entergy Mississippi, LLC Fir",
    "EMPD": "Empery Digital Common stock",
    "ENB": "Enbridge Inc",
    "ENGN": "enGene",
    "ENGS": "Energys",
    "ENJ": "Entergy New Orleans, LLC Fir",
    "ENLT": "Enlight Renewable Energy",
    "ENLV": "Enlivex",
    "ENO": "Entergy New Orleans, LLC Fir",
    "ENOV": "Enovis",
    "ENR": "Energizer",
    "ENS": "EnerSys",
    "ENSC": "Ensysce Biosciences",
    "ENTA": "Enanta",
    "ENTG": "Entegris",
    "ENTX": "Entera Bio",
    "ENVA": "Enova",
    "ENVB": "Enveric Biosciences",
    "ENVX": "Enovix",
    "EOD": "Allspring Global Dividend Op",
    "EOI": "Eaton Vance Enhance Equity I",
    "EOLS": "Evolus",
    "EOS": "Eaton Vance Enhance Equity I",
    "EOSE": "Eos Energy Enterprises Class",
    "EOT": "Eaton Vance Municipal Income",
    "EPAC": "Enerpac Tool",
    "EPC": "Edgewell Personal Care",
    "EPD": "Enterprise Products Partners",
    "EPR": "EPR Properties",
    "EPRT": "Essential Properties Realty",
    "EPRX": "Eupraxia",
    "EPSM": "Epsium Enterprise",
    "EPSN": "Epsilon Energy Common Share",
    "EQ": "Equillium",
    "EQBK": "Equity Bancshares, Class A",
    "EQH": "Equitable",
    "EQNR": "Equinor ASA",
    "EQPT": "EquipmentShare.com Inc Class",
    "EQS": "Equus Total Return",
    "EQT": "EQT",
    "ERAS": "Erasca",
    "ERIC": "Ericsson",
    "ERIE": "Erie Indemnity Class A",
    "ERNA": "Ernexa",
    "ERO": "Ero Copper",
    "ESCA": "Escalade",
    "ESE": "ESCO",
    "ESEA": "Euroseas (Marshall Islands)",
    "ESI": "Element Solutions",
    "ESLA": "Estrella Immunopharma",
    "ESLT": "Elbit Systems",
    "ESOA": "Energy Services of America",
    "ESPR": "Esperion",
    "ESQ": "Esquire Financial",
    "ESRT": "Empire State Realty Trust, C",
    "ESTA": "Establishment Labs",
    "ETB": "Eaton Vance Tax-Managed Buy-",
    "ETD": "Ethan Allen Interiors",
    "ETG": "Eaton Vance Tax-Advantaged G",
    "ETHB": "iShares Staked Ethereum Trus",
    "ETHM": "Dynamix",
    "ETJ": "Eaton Vance Risk-Managed Div",
    "ETO": "Eaton Vance Tax-Advantage Gl",
    "ETON": "Eton",
    "ETOR": "eToro Class A",
    "ETS": "Elite Express Holding Class",
    "ETV": "Eaton Vance Eaton Vance Tax-",
    "ETW": "Eaton Vance Eaton Vance Tax-",
    "ETX": "Eaton Vance Municipal Income",
    "ETY": "Eaton Vance Tax-Managed Dive",
    "EU": "enCore Energy",
    "EUDA": "EUDA Health",
    "EVAX": "Evaxion A/S American Deposit",
    "EVC": "Entravision Communications",
    "EVCM": "EverCommerce",
    "EVER": "EverQuote Class A",
    "EVEX": "Eve Holding",
    "EVF": "Eaton Vance Senior Income Tr",
    "EVG": "Eaton Vance Short Diversifie",
    "EVGN": "Evogene Ltd",
    "EVH": "Evolent Health Class A",
    "EVLV": "Evolv Class A",
    "EVMN": "Evommune",
    "EVN": "Eaton Vance Municipal Income",
    "EVO": "Evotec",
    "EVR": "Evercore Class A",
    "EVT": "Eaton Vance Tax Advantaged D",
    "EVTC": "Evertec",
    "EVTV": "Envirotech Vehicles",
    "EWCZ": "European Wax Center Class A",
    "EWTX": "Edgewise",
    "EXE": "Expand Energy",
    "EXEL": "Exelixis",
    "EXFY": "Expensify Class A",
    "EXG": "Eaton Vance Tax-Managed Glob",
    "EXK": "Endeavour Silver (Canada)",
    "EXLS": "ExlService",
    "EXOZ": "eXoZymes",
    "EXP": "Eagle Materials Inc",
    "EYE": "National Vision",
    "EYPT": "EyePoint",
    "EZGO": "EZGO",
    "EZRA": "Reliance Global",
    "FA": "First Advantage",
    "FAF": "First American (New)",
    "FAMI": "Farmmi",
    "FARM": "Farmer Brothers",
    "FATN": "FatPipe",
    "FBGL": "FBS Global",
    "FBIO": "Fortress Biotech",
    "FBIZ": "First Business Financial Ser",
    "FBK": "FB Financial",
    "FBLA": "FB Bancorp",
    "FBLG": "FibroBiologics",
    "FBNC": "First Bancorp",
    "FBP": "First BanCorp. New",
    "FBRT": "Franklin BSP Realty Trust",
    "FBRX": "Forte Biosciences",
    "FBYD": "Falcon\'s Beyond Global Class",
    "FC": "Franklin Covey",
    "FCAP": "First Capital",
    "FCBC": "First Community Bankshares (",
    "FCCO": "First Community",
    "FCEL": "FuelCell Energy",
    "FCF": "First Commonwealth Financial",
    "FCFS": "FirstCash",
    "FCHL": "Fitness Champs",
    "FCN": "FTI Consulting",
    "FCNCA": "First Citizens BancShares Cl",
    "FCNCP": "First Citizens BancShares De",
    "FCT": "First Trust Senior Floating",
    "FCUV": "Focus Universal",
    "FDBC": "Fidelity D & D Bancorp",
    "FDMT": "4D Molecular",
    "FDP": "Fresh Del Monte Produce",
    "FDS": "FactSet Research Systems",
    "FDSB": "Fifth District Bancorp",
    "FEAM": "5E Advanced Materials",
    "FEBO": "Fenbo",
    "FEED": "ENvue Medical",
    "FEIM": "Frequency Electronics",
    "FELE": "Franklin Electric",
    "FEMY": "Femasys",
    "FENC": "Fennec",
    "FER": "Ferrovial",
    "FERG": "Ferguson Enterprises",
    "FET": "Forum Energy",
    "FF": "FutureFuel  Common shares",
    "FFA": "First Trust Enhanced Equity",
    "FFAI": "Faraday Future Intelligent E",
    "FFBC": "First Financial Bancorp.",
    "FFIC": "Flushing Financial",
    "FFWM": "First Foundation",
    "FG": "F&G Annuities & Life",
    "FGBI": "First Guaranty Bancshares",
    "FGI": "FGI Industries",
    "FGL": "Founder",
    "FGMC": "FG Merger II Common stock",
    "FGNX": "FG Nexus",
    "FHB": "First Hawaiian",
    "FHI": "Federated Hermes",
    "FHTX": "Foghorn",
    "FICO": "Fair Isaac",
    "FIEE": "FiEE Inc",
    "FIG": "Figma, Class A",
    "FIGR": "Figure Technology Solutions",
    "FIHL": "Fidelis Insurance",
    "FINS": "Angel Oak Financial Strategi",
    "FIP": "FTAI Infrastructure",
    "FISI": "Financial Institutions",
    "FISV": "Fiserv",
    "FITBI": "Fifth Third Bancorp Deposita",
    "FIVN": "Five9",
    "FIX": "Comfort Systems USA",
    "FIZZ": "National Beverage",
    "FKWL": "Franklin Wireless",
    "FLC": "Flaherty & Crumrine Total Re",
    "FLD": "Fold Class A",
    "FLEX": "Flex",
    "FLG": "Flagstar Bank, N.A.",
    "FLGT": "Fulgent Genetics",
    "FLL": "Full House Resorts",
    "FLNA": "Filana",
    "FLNG": "FLEX LNG",
    "FLNT": "Fluent",
    "FLO": "Flowers Foods",
    "FLOC": "Flowco Class A",
    "FLR": "Fluor",
    "FLS": "Flowserve",
    "FLUT": "Flutter Entertainment",
    "FLUX": "Flux Power",
    "FLWS": "1-800-FLOWERS.COM",
    "FLX": "BingEx",
    "FLXS": "Flexsteel Industries",
    "FLYE": "Fly-E",
    "FMAO": "Farmers & Merchants Bancorp",
    "FMBH": "First Mid Bancshares",
    "FMFC": "Kandal M Venture",
    "FMN": "Federated Hermes Premier Mun",
    "FMNB": "Farmers National Banc",
    "FMST": "Foremost Clean Energy",
    "FMX": "Fomento Economico MexicanoB.",
    "FMY": "First Trust Mortgage Income",
    "FN": "Fabrinet",
    "FNB": "F.N.B.",
    "FND": "Floor & Decor",
    "FNF": "Fidelity National Financial",
    "FNGR": "FingerMotion",
    "FNKO": "Funko Class A",
    "FNLC": "First Bancorp Inc  (ME)",
    "FNUC": "Frontier Nuclear and Mineral",
    "FNV": "Franco-Nevada",
    "FNWB": "First Northwest Bancorp",
    "FNWD": "Finward Bancorp",
    "FOA": "Finance of America Companies",
    "FOF": "Cohen & Steers Closed-End Op",
    "FOFO": "Hang Feng Technology Innovat",
    "FOLD": "Amicus",
    "FONR": "Fonar",
    "FOR": "Forestar Inc",
    "FORA": "Forian",
    "FORR": "Forrester Research",
    "FORTY": "Formula Systems (1985)",
    "FOSL": "Fossil",
    "FOUR": "Shift4 Payments, Class A",
    "FOXF": "Fox Factory Holding",
    "FOXX": "Foxx Development",
    "FPH": "Five Point, LLC Class A",
    "FPI": "Farmland Partners",
    "FPS": "Forgent Power Solutions, Cla",
    "FR": "First Industrial Realty Trus",
    "FRA": "Blackrock Floating Rate Inco",
    "FRAF": "Franklin Financial Services",
    "FRBA": "First Bank",
    "FRD": "Friedman Industries",
    "FRGT": "Freight",
    "FRHC": "Freedom Holding",
    "FRME": "First Merchants",
    "FRMEP": "First Merchants Depository S",
    "FRMI": "Fermi",
    "FRMM": "Forum Markets",
    "FRO": "Frontline Plc",
    "FROG": "JFrog",
    "FRPH": "FRP",
    "FRPT": "Freshpet",
    "FRST": "Primis Financial",
    "FRSX": "Foresight Autonomous",
    "FRT": "Federal Realty Investment Tr",
    "FSBC": "Five Star Bancorp",
    "FSCO": "FS Credit Opportunities",
    "FSEA": "First Seacoast Bancorp",
    "FSLY": "Fastly Class A",
    "FSM": "Fortuna Mining",
    "FSS": "Federal Signal",
    "FSSL": "FS Specialty Lending Fund",
    "FSTR": "L.B. Foster",
    "FSUN": "FirstSun Capital Bancorp",
    "FSV": "FirstService",
    "FT": "Franklin Universal Trust",
    "FTAI": "FTAI Aviation",
    "FTCI": "FTC Solar",
    "FTDR": "Frontdoor",
    "FTEK": "Fuel Tech",
    "FTFT": "Future FinTech",
    "FTHM": "Fathom",
    "FTHY": "First Trust High Yield Oppor",
    "FTI": "TechnipFMC Ordinary Share",
    "FTK": "Flotek Industries",
    "FTLF": "FitLife Brands",
    "FTRE": "Fortrea",
    "FTRK": "FAST TRACK GROUP Ordinary sh",
    "FTS": "Fortis",
    "FTV": "Fortive",
    "FTW": "Presidio Production Class A",
    "FUBO": "FuboTV Class A",
    "FUFU": "BitFuFu",
    "FUL": "H. B. Fuller",
    "FULC": "Fulcrum",
    "FULT": "Fulton Financial",
    "FUN": "Six Flags Entertainment New",
    "FUNC": "First United",
    "FUND": "Sprott Focus Trust",
    "FUSB": "First US Bancshares",
    "FUSE": "Fusemachines Common stock",
    "FUTU": "Futu",
    "FVCB": "FVCBankcorp",
    "FVR": "FrontView REIT",
    "FVRR": "Fiverr, no par value",
    "FWDI": "Forward Industries",
    "FWONA": "Liberty Media Series A Liber",
    "FWONK": "Liberty Media Series C Liber",
    "FWRD": "Forward Air",
    "FWRG": "First Watch Restaurant",
    "FXNC": "First National",
    "GAB": "Gabelli Equity Trust, (The)",
    "GABC": "German American Bancorp",
    "GAIA": "Gaia Class A",
    "GALT": "Galectin",
    "GAM": "General American Investors",
    "GAMB": "Gambling.com",
    "GAME": "GameSquare",
    "GANX": "Gain",
    "GAP": "Gap, (The)",
    "GASS": "StealthGas",
    "GAUZ": "Gauzy",
    "GBAB": "Guggenheim Taxable Municipal",
    "GBFH": "GBank Financial",
    "GBLI": "Global Indemnity LLC Class A",
    "GBTG": "Global Business Travel, Clas",
    "GCBC": "Greene County Bancorp",
    "GCL": "GCL Global Ltd",
    "GCMG": "GCM Grosvenor Class A",
    "GCO": "Genesco",
    "GCT": "GigaCloud Technology Inc",
    "GCTK": "GlucoTrack",
    "GCTS": "GCT Semiconductor Holding",
    "GCV": "Gabelli Convertible and Inco",
    "GDC": "GD Culture",
    "GDEN": "Golden Entertainment",
    "GDEV": "GDEV",
    "GDHG": "Golden Heaven",
    "GDL": "GDL Fund, The",
    "GDO": "Western Asset Global Corpora",
    "GDOT": "Green Dot Class A, $0.001 pa",
    "GDRX": "GoodRx Class A",
    "GDS": "GDS ADS",
    "GDTC": "CytoMed",
    "GDV": "Gabelli Dividend & Income Tr",
    "GDYN": "Grid Dynamics Class A",
    "GECC": "Great Elm Capital",
    "GEF": "Greif Class A",
    "GEG": "Great Elm",
    "GEHC": "GE HealthCare",
    "GEL": "Genesis Energy, L.P. Common",
    "GELS": "Gelteq",
    "GENB": "Generate Biomedicines",
    "GENK": "GEN Restaurant Class A",
    "GEO": "Geo Inc (The) REIT",
    "GETY": "Getty Images, Class A",
    "GEV": "GE Vernova",
    "GEVO": "Gevo",
    "GF": "New Germany Fund, (The)",
    "GFAI": "Guardforce AI",
    "GFL": "GFL Environmental Subordinat",
    "GFR": "Greenfire Resources",
    "GFS": "GlobalFoundries",
    "GGAL": "Grupo Financiero Galicia",
    "GGB": "Gerdau",
    "GGR": "Gogoro",
    "GGRP": "The Glimpse",
    "GGT": "Gabelli Multi-Media Trust, (",
    "GGZ": "Gabelli Global Small and Mid",
    "GH": "Guardant Health",
    "GHC": "Graham",
    "GHM": "Graham",
    "GHRS": "GH Research",
    "GHY": "PGIM Global High Yield Fund",
    "GIB": "CGI",
    "GIBO": "GIBO",
    "GIC": "Global Industrial",
    "GIFT": "Giftify",
    "GIG": "GigCapital7 Class A Ordinary",
    "GIGM": "GigaMedia",
    "GIII": "G-III Apparel LTD.",
    "GIL": "Gildan Activewear, Class A S",
    "GILT": "Gilat Satellite Networks",
    "GIPR": "Generation Income Properties",
    "GITS": "Global Interactive",
    "GIX": "GigCapital9 Class A Ordinary",
    "GJH": "Synthetic Fixed-Income Secur",
    "GJO": "Synthetic Fixed-Income Secur",
    "GJP": "Synthetic Fixed-Income Secur",
    "GJR": "Synthetic Fixed-Income Secur",
    "GJT": "Synthetic Fixed-Income Secur",
    "GLBS": "Globus Maritime",
    "GLE": "Global Engine Holding",
    "GLIBA": "GCI Liberty Series A GCI",
    "GLIBK": "GCI Liberty Series C GCI",
    "GLMD": "Galmed",
    "GLNG": "Golar Lng Ltd",
    "GLOB": "Globant",
    "GLOO": "Gloo Class A",
    "GLPG": "Galapagos NV",
    "GLPI": "Gaming and Leisure Propertie",
    "GLRE": "Greenlight Capital Re",
    "GLSI": "Greenwich LifeSciences",
    "GLUE": "Monte Rosa",
    "GLXG": "Galaxy Payroll",
    "GLXY": "Galaxy Digital Class A",
    "GMAB": "Genmab A/S ADS",
    "GME": "GameStop",
    "GMED": "Globus Medical, Class A",
    "GMEX": "GMEX ROBOTICS CORPORATION",
    "GMHS": "Gamehaus",
    "GMM": "Global Mofy AI",
    "GNE": "Genie Energy Class B Stock",
    "GNK": "Genco Shipping & Trading New",
    "GNL": "Global Net Lease",
    "GNLN": "Greenlane Class A",
    "GNLX": "Genelux",
    "GNPX": "Genprex",
    "GNSS": "Genasys",
    "GNT": "GAMCO Natural Resources, Gol",
    "GNTA": "Genenta Science S.p.A.",
    "GNTX": "Gentex",
    "GO": "Grocery Outlet Holding",
    "GOAI": "Eva Live",
    "GOCO": "GoHealth Class A",
    "GOF": "Guggenheim Strategic Opportu",
    "GOGO": "Gogo",
    "GOLF": "Acushnet",
    "GOOS": "Canada Goose Subordinate Vot",
    "GOSS": "Gossamer Bio",
    "GOVX": "GeoVax Labs",
    "GP": "GreenPower Motor",
    "GPC": "Genuine Parts",
    "GPCR": "Structure",
    "GPGI": "GPGI, Class A",
    "GPK": "Graphic Packaging Holding",
    "GPMT": "Granite Point Mortgage Trust",
    "GPOR": "Gulfport Energy",
    "GPRE": "Green Plains",
    "GPRK": "Geopark Ltd",
    "GPRO": "GoPro Class A",
    "GRAB": "Grab",
    "GRAL": "GRAIL",
    "GRAN": "Grande",
    "GRC": "Gorman-Rupp (The)",
    "GRCE": "Grace",
    "GRDN": "Guardian Pharmacy Services,",
    "GRDX": "GridAI",
    "GREE": "Greenidge Generation Class A",
    "GRFS": "Grifols",
    "GRI": "GRI Bio",
    "GRML": "Greenland Mines",
    "GRMN": "Garmin (Switzerland)",
    "GRND": "Grindr",
    "GRNQ": "Greenpro Capital",
    "GRNT": "Granite Ridge Resources",
    "GROV": "Grove Collaborative, Class A",
    "GRPN": "Groupon",
    "GRRR": "Gorilla Technology Ordinary",
    "GRVY": "GRAVITY",
    "GRWG": "GrowGeneration",
    "GRX": "The Gabelli Healthcare & Wel",
    "GSAT": "Globalstar",
    "GSHD": "Goosehead Insurance Class A",
    "GSIT": "GSI Technology",
    "GSL": "Global Ship Lease Inc New Cl",
    "GSM": "Ferroglobe",
    "GSUN": "Golden Sun Technology",
    "GT": "The Goodyear Tire & Rubber",
    "GTBP": "GT Biopharma",
    "GTEC": "Greenland Holding",
    "GTEN": "Gores X Class A ordinary sha",
    "GTES": "Gates Industrial",
    "GTIM": "Good Times Restaurants",
    "GTM": "ZoomInfo Inc",
    "GTN": "Gray Media",
    "GTX": "Garrett Motion",
    "GUG": "Guggenheim Active Allocation",
    "GURE": "Gulf Resources (NV)",
    "GUT": "Gabelli Utility Trust (The)",
    "GUTS": "Fractyl Health",
    "GV": "Visionary",
    "GVH": "Globavend",
    "GWAV": "Greenwave Technology Solutio",
    "GWH": "ESS Tech",
    "GWRE": "Guidewire Software",
    "GWRS": "Global Water Resources",
    "GXAI": "Gaxos.ai",
    "GXO": "GXO Logistics",
    "GYRE": "Gyre",
    "GYRO": "Gyrodyne LLC",
    "H": "Hyatt Hotels Class A",
    "HAE": "Haemonetics",
    "HAFN": "Hafnia",
    "HAIN": "Hain Celestial (The)",
    "HALO": "Halozyme",
    "HAO": "Haoxi Health Technology",
    "HBB": "Hamilton Beach Brands Holdin",
    "HBCP": "Home Bancorp",
    "HBIO": "Harvard Bioscience",
    "HBM": "Hudbay Minerals (Canada)",
    "HBNB": "Hotel101 Global",
    "HBNC": "Horizon Bancorp",
    "HBT": "HBT Financial",
    "HCAI": "Huachen AI Parking Managemen",
    "HCAT": "Health Catalyst Inc",
    "HCC": "Warrior Met Coal",
    "HCHL": "Happy City Class A Ordinary",
    "HCI": "HCI",
    "HCKT": "Hackett Inc (The).",
    "HCM": "HUTCHMED (China)",
    "HCSG": "Healthcare Services",
    "HCTI": "Healthcare Triangle",
    "HCWB": "HCW Biologics",
    "HDB": "HDFC Bank",
    "HDL": "SUPER HI INTERNATIONAL HOLDI",
    "HDSN": "Hudson",
    "HE": "Hawaiian Electric Industries",
    "HEI-A": "Heico",
    "HELE": "Helen of Troy",
    "HELP": "Cybin",
    "HEPS": "D-Market Electronic Services",
    "HEQ": "John Hancock Diversified Inc",
    "HERE": "Here",
    "HERZ": "Herzfeld Credit Income Fund",
    "HFBL": "Home Federal Bancorp of Loui",
    "HFFG": "HF Foods",
    "HFRO": "Highland Opportunities and I",
    "HFWA": "Heritage Financial",
    "HG": "Hamilton Insurance, Class B",
    "HGBL": "Heritage Global",
    "HGLB": "Highland Global Allocation F",
    "HGTY": "Hagerty, Class A",
    "HGV": "Hilton Grand Vacations",
    "HHH": "Howard Hughes",
    "HHS": "Harte Hanks",
    "HIFS": "Hingham Institution for Savi",
    "HIHO": "Highway",
    "HII": "Huntington Ingalls Industrie",
    "HIMX": "Himax",
    "HIND": "Vyome",
    "HIO": "Western Asset High Income Op",
    "HIPO": "Hippo",
    "HIT": "Health In Tech Class A",
    "HITI": "High Tide",
    "HIVE": "HIVE Digital",
    "HIX": "Western Asset High Income Fu",
    "HKIT": "Hitek Global Class A Ordinar",
    "HKPD": "Cellyan Biotechnology Ltd",
    "HL": "Hecla Mining",
    "HLF": "Herbalife",
    "HLI": "Houlihan Lokey, Class A",
    "HLIO": "Helios",
    "HLIT": "Harmonic",
    "HLLY": "Holley",
    "HLMN": "Hillman Solutions",
    "HLP": "Hongli",
    "HMC": "Honda Motor",
    "HMN": "Horace Mann Educators",
    "HMR": "Heidmar Maritime",
    "HMY": "Harmony Gold Mining",
    "HNGE": "Hinge Health, Class A",
    "HNI": "HNI",
    "HNNA": "Hennessy Advisors",
    "HNRG": "Hallador Energy",
    "HNST": "The Honest",
    "HNVR": "Hanover Bancorp",
    "HOFT": "Hooker Furnishings",
    "HOG": "Harley-Davidson",
    "HOLO": "MicroCloud Hologram",
    "HOMB": "Home BancShares",
    "HOTH": "Hoth",
    "HOUR": "Hour Loop",
    "HOV": "Hovnanian Enterprises, Class",
    "HOVNP": "Hovnanian Enterprises Inc De",
    "HOVR": "New Horizon Aircraft Class A",
    "HOWL": "Werewolf",
    "HP": "Helmerich & Payne",
    "HPAI": "Helport AI",
    "HPF": "John Hancock Pfd Income Fund",
    "HPK": "HighPeak Energy",
    "HPP": "Hudson Pacific Properties",
    "HQ": "Horizon Quantum",
    "HQH": "abrdn Healthcare Investors S",
    "HQI": "HireQuest (DE)",
    "HQL": "abrdn Life Sciences Investor",
    "HQY": "HealthEquity",
    "HRB": "H&R Block",
    "HRI": "Herc",
    "HRTG": "Heritage Insurance",
    "HRTX": "Heron",
    "HSAI": "Hesai American Depositary Sh",
    "HSBC": "HSBC,.",
    "HSCS": "HeartSciences",
    "HSDT": "Solana Class A (DE)",
    "HSHP": "Himalaya Shipping",
    "HSIC": "Henry Schein",
    "HSY": "The Hershey",
    "HTB": "HomeTrust Bancshares",
    "HTBK": "Heritage Commerce Corp",
    "HTCO": "High-Trend",
    "HTCR": "Heartcore Enterprises",
    "HTD": "John Hancock Tax Advantaged",
    "HTFL": "Heartflow",
    "HTH": "Hilltop",
    "HTHT": "H World",
    "HTLD": "Heartland Express",
    "HTLM": "HomesToLife Ltd",
    "HTO": "H2O America",
    "HTOO": "Fusion Fuel Green",
    "HTZ": "Hertz Global Inc",
    "HUBB": "Hubbell Inc",
    "HUBC": "Hub Cyber Security",
    "HUBG": "Hub Class A",
    "HUDI": "Huadi",
    "HUHU": "HUHUTECH",
    "HUIZ": "Huize Holding",
    "HUMA": "Humacyte",
    "HUN": "Huntsman",
    "HURA": "TuHURA Biosciences",
    "HURC": "Hurco Companies",
    "HUT": "Hut 8",
    "HVII": "Hennessy Capital Investment",
    "HVT": "Haverty Furniture Companies",
    "HVT-A": "Haverty Furniture Companies",
    "HWBK": "Hawthorn Bancshares",
    "HWC": "Hancock Whitney",
    "HWH": "HWH",
    "HWKN": "Hawkins",
    "HXHX": "Haoxin",
    "HXL": "Hexcel",
    "HY": "Hyster-Yale, Class A common",
    "HYFM": "Hydrofarm",
    "HYFT": "MindWalk",
    "HYI": "Western Asset High Yield Opp",
    "HYMC": "Hycroft Mining Holding Class",
    "HYNE": "Hoyne Bancorp",
    "HYPD": "Hyperion DeFi",
    "HYPR": "Hyperfine Class A",
    "HYT": "Blackrock Corporate High Yie",
    "HZO": "MarineMax,  (FL)",
    "IAE": "Voya Asia Pacific High Divid",
    "IAG": "Iamgold",
    "IART": "Integra LifeSciences",
    "IBEX": "IBEX",
    "IBG": "Innovation Beverage",
    "IBIO": "iBio",
    "IBKR": "Interactive Brokers Class A",
    "IBN": "ICICI Bank",
    "IBRX": "ImmunityBio",
    "IBTA": "Ibotta, Class A",
    "ICCC": "ImmuCell",
    "ICCM": "IceCure Medical",
    "ICG": "Intchains",
    "ICL": "ICL",
    "ICLR": "ICON",
    "ICMB": "Investcorp Credit Management",
    "ICON": "Icon Energy Common stock",
    "ICU": "SeaStar Medical Holding",
    "ICUI": "ICU Medical",
    "IDA": "IDACORP",
    "IDAI": "T Stamp Class A",
    "IDCC": "InterDigital",
    "IDE": "Voya Infrastructure, Industr",
    "IDN": "Intellicheck",
    "IDT": "IDT Class B",
    "IDYA": "IDEAYA Biosciences",
    "IEP": "Icahn Enterprises L.P.",
    "IEX": "IDEX",
    "IFBD": "Infobird Ltd",
    "IFF": "International Flavors & Frag",
    "IFN": "Aberdeen India Fund",
    "IFRX": "InflaRx",
    "IFS": "Intercorp Financial Services",
    "IGA": "Voya Global Advantage and Pr",
    "IGD": "Voya Global Equity Dividend",
    "IGI": "Western Asset Investment Gra",
    "IGIC": "International General Insura",
    "IGR": "CBRE Global Real Estate Inco",
    "IHD": "Voya Emerging Markets High I",
    "IHRT": "iHeartMedia Class A",
    "IHS": "IHS Holding",
    "IIF": "Morgan Stanley India Investm",
    "III": "Information Services",
    "IIIN": "Insteel Industries",
    "IIIV": "i3 Verticals Class A",
    "IIM": "Invesco Value Municipal Inco",
    "IINN": "Inspira Oxy B.H.N.",
    "IKT": "Inhibikase",
    "ILAG": "Intelligent Living Applicati",
    "ILMN": "Illumina",
    "ILPT": "Industrial Logistics Propert",
    "IMA": "ImageneBio",
    "IMAX": "Imax",
    "IMCC": "IM Cannabis",
    "IMDX": "Insight Molecular Diagnostic",
    "IMKTA": "Ingles Markets Class A",
    "IMMP": "Immutep",
    "IMMR": "Immersion",
    "IMMX": "Immix Biopharma",
    "IMNM": "Immunome",
    "IMNN": "Imunon",
    "IMOS": "ChipMOS TECHNOLOGIES INC.",
    "IMPP": "Imperial Petroleum",
    "IMRN": "Immuron",
    "IMRX": "Immuneering Class A",
    "IMSR": "Terrestrial Energy",
    "IMTE": "Integrated Media Technology",
    "IMTX": "Immatics",
    "IMUX": "Immunic",
    "IMXI": "International Money Express",
    "INAB": "IN8bio",
    "INBK": "First Internet Bancorp",
    "INBS": "Intelligent Bio Solutions",
    "INBX": "Inhibrx Biosciences",
    "INCR": "Intercure",
    "INCY": "Incyte",
    "INDB": "Independent Bank",
    "INDI": "indie Semiconductor Class A",
    "INDP": "Indaptus",
    "INDV": "Indivior",
    "INEO": "INNEOVA",
    "INFQ": "Infleqtion",
    "ING": "ING",
    "INGM": "Ingram Micro Holding",
    "INGN": "Inogen Inc",
    "INGR": "Ingredion",
    "INHD": "Inno",
    "INKT": "MiNK",
    "INLF": "INLIF LIMITED",
    "INM": "InMed",
    "INMB": "INmune Bio Common stock",
    "INMD": "InMode",
    "INN": "Summit Hotel Properties",
    "INNV": "InnovAge Holding",
    "INO": "Inovio",
    "INOD": "Innodata",
    "INR": "Infinity Natural Resources,",
    "INSE": "Inspired Entertainment",
    "INSG": "Inseego",
    "INSM": "Insmed",
    "INSP": "Inspire Medical Systems",
    "INTA": "Intapp",
    "INTG": "Intergroup (The)",
    "INTJ": "Intelligent",
    "INTR": "Inter & Class A",
    "INTS": "Intensity Common stock",
    "INTZ": "Intrusion",
    "INV": "Innventure",
    "INVE": "Identiv",
    "INVX": "Innovex",
    "INVZ": "Innoviz Ordinary shares",
    "IOBT": "IO Biotech",
    "IONR": "ioneer Ltd",
    "IONS": "Ionis",
    "IOSP": "Innospec",
    "IOT": "Samsara Class A",
    "IOTR": "iOThree",
    "IP": "International Paper",
    "IPAR": "Interparfums",
    "IPDN": "Professional Diversity Netwo",
    "IPGP": "IPG Photonics",
    "IPHA": "Innate Pharma ADS",
    "IPI": "Intrepid Potash",
    "IPM": "Intelligent Protection Manag",
    "IPSC": "Century",
    "IPST": "IP Strategy",
    "IPW": "iPower",
    "IPWR": "Ideal Power",
    "IPX": "IperionX American Depositary",
    "IQ": "iQIYI",
    "IQI": "Invesco Quality Municipal In",
    "IQST": "iQSTEL",
    "IRD": "Opus Genetics",
    "IRDM": "Iridium Communications Inc",
    "IREN": "IREN",
    "IRIX": "IRIDEX",
    "IRM": "Iron Mountain (Delaware)Comm",
    "IRMD": "iRadimed",
    "IRON": "Disc Medicine",
    "IRT": "Independence Realty Trust",
    "IRWD": "Ironwood Class A",
    "ISBA": "Isabella Bank Common stock",
    "ISD": "PGIM High Yield Bond Fund",
    "ISPC": "iSpecimen",
    "ISPR": "Ispire Technology",
    "ISSC": "Innovative Solutions and Sup",
    "ISTR": "Investar Holding",
    "ITGR": "Integer",
    "ITIC": "Investors Title",
    "ITOC": "iTonic Ltd",
    "ITRI": "Itron",
    "ITRM": "Iterum Ordinary Share",
    "ITRN": "Ituran Location and Control",
    "ITT": "ITT",
    "IVA": "Inventiva",
    "IVDA": "Iveda Solutions",
    "IVF": "INVO Fertility",
    "IVR": "INVESCO MORTGAGE CAPITAL INC",
    "IVT": "InvenTrust Properties",
    "IVVD": "Invivyd",
    "IXHL": "Incannex Healthcare",
    "IZEA": "IZEA Worldwide",
    "IZM": "ICZOOM",
    "J": "Jacobs Solutions",
    "JAGX": "Jaguar Health",
    "JAKK": "JAKKS Pacific",
    "JAN": "Janus Living, Class A-1",
    "JANX": "Janux",
    "JBDI": "JBDI",
    "JBHT": "J.B. Hunt Transport Services",
    "JBI": "Janus",
    "JBIO": "Jade Biosciences",
    "JBK": "Lehman ABS 3.50 3.50% Adjust",
    "JBL": "Jabil",
    "JBS": "JBS Class A",
    "JBSS": "John B. Sanfilippo & Son",
    "JBTM": "JBT Marel",
    "JCAP": "Jefferson Capital",
    "JCE": "Nuveen Core Equity Alpha Fun",
    "JCI": "Johnson Controls Ordinary Sh",
    "JCSE": "JE Cleantech",
    "JCTC": "Jewett-Cameron Trading",
    "JD": "JD.com",
    "JDZG": "JIADE LIMITED",
    "JEF": "Jefferies Financial",
    "JELD": "JELD-WEN Holding",
    "JEM": "707 Cayman",
    "JF": "J and Friends",
    "JFB": "JFB Construction Class A",
    "JFBR": "Jeffs\' Brands Ltd",
    "JFIN": "Jiayin",
    "JFR": "Nuveen Floating Rate Income",
    "JFU": "9F",
    "JG": "Aurora Mobile",
    "JGH": "Nuveen Global High Income Fu",
    "JHG": "Janus Henderson",
    "JHI": "John Hancock Investors Trust",
    "JHS": "John Hancock Income Securiti",
    "JHX": "James Hardie Industries.",
    "JILL": "J. Jill",
    "JJSF": "J & J Snack Foods",
    "JL": "J-Long",
    "JLHL": "Julong Holding",
    "JLL": "Jones Lang LaSalle",
    "JLS": "Nuveen Mortgage and Income F",
    "JMM": "Nuveen Multi-Market Income F",
    "JMSB": "John Marshall Bancorp",
    "JOE": "St. Joe (The)",
    "JOF": "Japan Smaller Capitalization",
    "JOUT": "Johnson Outdoors Class A",
    "JOYY": "JOYY",
    "JQC": "Nuveen Credit Strategies Inc",
    "JRI": "Nuveen Real Asset Income and",
    "JRS": "Nuveen Real Estate Income Fu",
    "JRSH": "Jerash (US)",
    "JRVR": "James River",
    "JSPR": "Jasper",
    "JTAI": "Jet.AI",
    "JUNS": "Jupiter Neurosciences",
    "JVA": "Coffee Holding",
    "JWEL": "Jowell Global",
    "JXG": "JX Luxventure",
    "JXN": "Jackson Financial Class A",
    "JYD": "Jayud Global Logistics",
    "JYNT": "The Joint",
    "JZ": "Jianzhi Education Technology",
    "JZXN": "Jiuzi",
    "KAI": "Kadant Inc",
    "KALA": "KALA BIO",
    "KALU": "Kaiser Aluminum",
    "KALV": "KalVista",
    "KARO": "Karooooo",
    "KB": "KB Financial Inc",
    "KBDC": "Kayne Anderson BDC",
    "KBH": "KB Home",
    "KBON": "Karbon Capital Partners",
    "KBR": "KBR",
    "KBSX": "FST",
    "KC": "Kingsoft Cloud",
    "KD": "Kyndryl",
    "KDK": "Kodiak AI",
    "KDP": "Keurig Dr Pepper",
    "KE": "Kimball Electronics",
    "KELYB": "Kelly Services Class B",
    "KEN": "Kenon",
    "KEP": "Korea Electric Power",
    "KEQU": "Kewaunee Scientific",
    "KEX": "Kirby",
    "KF": "Korea Fund, (The) New",
    "KFFB": "Kentucky First Federal Banco",
    "KFS": "Kingsway Financial Services,",
    "KG": "Kestrel",
    "KGC": "Kinross Gold",
    "KGEI": "Kolibri Global Energy Common",
    "KGS": "Kodiak Gas Services",
    "KHC": "The Kraft Heinz",
    "KIDS": "OrthoPediatrics",
    "KIDZ": "Classover Class B",
    "KINS": "Kingstone Companies",
    "KIO": "KKR Income Opportunities Fun",
    "KITT": "Nauticus Robotics Common sto",
    "KKR": "KKR &",
    "KLAR": "Klarna",
    "KLC": "KinderCare Learning Companie",
    "KLIC": "Kulicke and Soffa Industries",
    "KLRS": "Kalaris",
    "KLTR": "Kaltura",
    "KLXE": "KLX Energy Services",
    "KMDA": "Kamada",
    "KMPR": "Kemper",
    "KMRK": "K-Tech Solutions",
    "KMT": "Kennametal",
    "KMTS": "Kestra Medical",
    "KN": "Knowles",
    "KNDI": "Kandi",
    "KNF": "Knife Riv Holding",
    "KNSA": "Kiniksa",
    "KNSL": "Kinsale Capital",
    "KNTK": "Kinetik Class A",
    "KNX": "Knight-Swift Transportation",
    "KOD": "Kodiak Sciences Inc",
    "KODK": "Eastman Kodak Common New",
    "KOP": "Koppers",
    "KOPN": "Kopin",
    "KORE": "KORE",
    "KOS": "Kosmos Energy (DE)",
    "KOSS": "Koss",
    "KPLT": "Katapult",
    "KPRX": "Kiora",
    "KPTI": "Karyopharm",
    "KRC": "Kilroy Realty",
    "KREF": "KKR Real Estate Finance Trus",
    "KRG": "Kite Realty Trust",
    "KRKR": "36Kr",
    "KRMD": "KORU Medical Systems (DE)",
    "KRMN": "Karman",
    "KRNT": "Kornit Digital",
    "KRNY": "Kearny Financial Corp",
    "KRO": "Kronos Worldwide Inc",
    "KRRO": "Korro Bio",
    "KRT": "Karat Packaging",
    "KRUS": "Kura Sushi USA Class A",
    "KSCP": "Knightscope Class A",
    "KSPI": "Joint Stock Kaspi.kz",
    "KSS": "Kohl\'s",
    "KT": "KT",
    "KTB": "Kontoor Brands",
    "KTCC": "Key Tronic",
    "KTF": "DWS Municipal Income Trust",
    "KTN": "Structured Products Corp 8.2",
    "KTOS": "Kratos Defense & Security So",
    "KTTA": "Pasithea",
    "KURA": "Kura Oncology",
    "KUST": "Kustom Entertainment",
    "KVHI": "KVH Industries",
    "KVUE": "Kenvue",
    "KVYO": "Klaviyo, Series A",
    "KW": "Kennedy-Wilson",
    "KWM": "K Wave Media",
    "KWR": "Quaker Houghton",
    "KXIN": "Kaixin",
    "KYIV": "Kyivstar",
    "KYN": "Kayne Anderson Energy Infras",
    "KYNB": "Kyntra Bio",
    "KYTX": "Kyverna",
    "KZIA": "Kazia",
    "KZR": "Kezar Life Sciences",
    "LAB": "Standard BioTools",
    "LAC": "Lithium Americas",
    "LADR": "Ladder Capital Corp Class A",
    "LAES": "SEALSQ Corp",
    "LAKE": "Lakeland Industries",
    "LAMR": "Lamar Advertising Class A",
    "LANV": "Lanvin",
    "LAR": "Lithium Argentina",
    "LARK": "Landmark Bancorp",
    "LASE": "Laser Photonics",
    "LASR": "nLIGHT",
    "LAUR": "Laureate Education",
    "LAW": "CS Disco",
    "LAZ": "Lazard",
    "LB": "LandBridge LLC Class A Share",
    "LBGJ": "Li Bang",
    "LBRDA": "Liberty Broadband Class A",
    "LBRDK": "Liberty Broadband Class C",
    "LBRT": "Liberty Energy Class A commo",
    "LBRX": "LB Inc",
    "LBTYA": "Liberty Global Class A",
    "LBTYB": "Liberty Global Class B",
    "LBTYK": "Liberty Global Class C",
    "LC": "LendingClub",
    "LCFY": "Locafy Ordinary Share",
    "LCII": "LCI Industries",
    "LCNB": "LCNB",
    "LCUT": "Lifetime Brands",
    "LDI": "loanDepot, Class A",
    "LE": "Lands\' End",
    "LEA": "Lear",
    "LECO": "Lincoln Electric",
    "LEDS": "SemiLEDS",
    "LEE": "Lee Enterprises",
    "LEG": "Leggett & Platt",
    "LEGH": "Legacy Housing (TX)",
    "LEGN": "Legend Biotech",
    "LENZ": "LENZ",
    "LEO": "BNY Mellon Strategic Municip",
    "LEU": "Centrus Energy Class A",
    "LEVI": "Levi Strauss & Co Class A",
    "LEXX": "Lexaria Bioscience",
    "LFCR": "Lifecore Biomedical",
    "LFMD": "LifeMD",
    "LFS": "LEIFRAS",
    "LFST": "LifeStance Health",
    "LFT": "Lument Finance Trust",
    "LFUS": "Littelfuse",
    "LFVN": "Lifevantage (Delaware)",
    "LFWD": "Lifeward",
    "LGCB": "Linkage Global",
    "LGCL": "Lucas GC",
    "LGHL": "Lion Holding American Deposi",
    "LGI": "Lazard Global Total Return a",
    "LGIH": "LGI Homes",
    "LGN": "Legence Class A Common stock",
    "LGND": "Ligand",
    "LGO": "Largo",
    "LGVN": "Longeveron Class A",
    "LH": "Labcorp",
    "LHAI": "Linkhome Common stock",
    "LHX": "L3Harris",
    "LI": "Li Auto",
    "LICN": "Lichen",
    "LIDR": "AEye Class A",
    "LIEN": "Chicago Atlantic BDC",
    "LIF": "Life360",
    "LIFE": "Ethos Class A",
    "LII": "Lennox",
    "LILA": "Liberty Latin America Class",
    "LILAK": "Liberty Latin America Class",
    "LIMN": "Liminatus Pharma Class A",
    "LINC": "Lincoln Educational Services",
    "LIND": "Lindblad Expeditions",
    "LINE": "Lineage",
    "LINK": "Interlink Electronics",
    "LION": "Lionsgate Studios Corp",
    "LIQT": "LiqTech",
    "LITS": "Lite Strategy",
    "LIVE": "Live Ventures",
    "LIVN": "LivaNova",
    "LIXT": "Lixte Biotechnology",
    "LKFN": "Lakeland Financial",
    "LKQ": "LKQ",
    "LLYVA": "Liberty Live Series A Libert",
    "LLYVK": "Liberty Live Series C Libert",
    "LMAT": "LeMaitre Vascular",
    "LMB": "Limbach",
    "LMFA": "LM Funding America",
    "LMNR": "Limoneira Co",
    "LMRI": "Lumexa Imaging",
    "LNAI": "Lunai Bioworks",
    "LND": "Brasilagro Brazilian Agric R",
    "LNKB": "LINKBANCORP",
    "LNKS": "Linkers Industries",
    "LNN": "Lindsay",
    "LNSR": "LENSAR",
    "LNTH": "Lantheus",
    "LNZA": "LanzaTech Global",
    "LOAN": "Manhattan Bridge Capital Inc",
    "LOAR": "Loar",
    "LOB": "Live Oak Bancshares",
    "LOBO": "LOBO TECHNOLOGIES LTD.",
    "LOCL": "Local Bounti",
    "LOCO": "El Pollo Loco",
    "LOGI": "Logitech",
    "LOMA": "Loma Negra Compania Industri",
    "LONA": "LeonaBio",
    "LOOP": "Loop Industries",
    "LOPE": "Grand Canyon Education",
    "LOT": "Lotus Technology",
    "LOVE": "The Lovesac",
    "LPCN": "Lipocine",
    "LPG": "Dorian LPG",
    "LPL": "LG Display Co, Ltd AMERICAN",
    "LPLA": "LPL Financial",
    "LPRO": "Open Lending",
    "LPSN": "LivePerson",
    "LPTH": "LightPath Class A",
    "LQDA": "Liquidia",
    "LQDT": "Liquidity Services",
    "LRE": "Lead Real Estate Ltd",
    "LRHC": "La Rosa",
    "LRMR": "Larimar",
    "LRN": "Stride",
    "LSAK": "Lesaka",
    "LSBK": "Lake Shore Bancorp",
    "LSCC": "Lattice Semiconductor",
    "LSE": "Leishen Energy Holding",
    "LSH": "Lakeside Holding",
    "LSPD": "Lightspeed Commerce Subordin",
    "LSTA": "Lisata",
    "LSTR": "Landstar System",
    "LTBR": "Lightbridge",
    "LTC": "LTC Properties",
    "LTH": "Life Time",
    "LTRN": "Lantern Pharma",
    "LTRX": "Lantronix",
    "LUCD": "Lucid Diagnostics",
    "LUCK": "Lucky Strike Entertainment C",
    "LUCY": "Innovative Eyewear",
    "LULU": "lululemon athletica inc.",
    "LUMN": "Lumen",
    "LUNG": "Pulmonx",
    "LVLU": "Lulu\'s Fashion Lounge",
    "LVO": "LiveOne",
    "LVWR": "LiveWire",
    "LWAY": "Lifeway Foods",
    "LWLG": "Lightwave Logic",
    "LX": "LexinFintech",
    "LXEH": "Lixiang Education Holding",
    "LXEO": "Lexeo",
    "LXRX": "Lexicon",
    "LXU": "LSB Industries",
    "LYEL": "Lyell Immunopharma",
    "LYTS": "LSI Industries",
    "LZ": "LegalZoom.com",
    "LZB": "La-Z-Boy",
    "LZM": "Lifezone Metals",
    "LZMH": "LZ Technology",
    "M": "Macy\'s Inc",
    "MAAS": "Maase",
    "MAC": "Macerich (The)",
    "MAGN": "Magnera",
    "MAMA": "Mama\'s Creations",
    "MAMO": "Massimo",
    "MANE": "Veradermics",
    "MANU": "Manchester United",
    "MAPS": "WM Technology Class A",
    "MASI": "Masimo",
    "MASK": "3 E Network Technology Ltd",
    "MASS": "908 Devices",
    "MATH": "Metalpha Technology Holding",
    "MATV": "Mativ",
    "MATX": "Matson",
    "MAX": "MediaAlpha, Class A",
    "MAYS": "J. W. Mays",
    "MAZE": "Maze",
    "MB": "MasterBeef",
    "MBAI": "Check-Cap Ordinary Share",
    "MBBC": "Marathon Bancorp",
    "MBC": "MasterBrand",
    "MBI": "MBIA",
    "MBIN": "Merchants Bancorp",
    "MBIO": "Mustang Bio",
    "MBLY": "Mobileye Global Class A",
    "MBOT": "Microbot Medical",
    "MBRX": "Moleculin Biotech",
    "MBUU": "Malibu Boats Class A",
    "MBWM": "Mercantile Bank",
    "MBX": "MBX Biosciences",
    "MC": "Moelis & Class A",
    "MCB": "Metropolitan Bank Holding",
    "MCBS": "MetroCity Bankshares",
    "MCFT": "MasterCraft Boat",
    "MCHB": "Mechanics Bancorp Class A",
    "MCHX": "Marchex Class B",
    "MCI": "Barings Corporate Investors",
    "MCK": "McKesson",
    "MCN": "XAI Madison Equity Premium I",
    "MCR": "MFS Charter Income Trust",
    "MCRB": "Seres",
    "MCRI": "Monarch Casino & Resort",
    "MCS": "Marcus (The)",
    "MCW": "Mister Car Wash",
    "MCY": "Mercury General",
    "MD": "Pediatrix Medical",
    "MDAI": "Spectral AI Class A",
    "MDBH": "MDB Capital LLC Class A comm",
    "MDCX": "Medicus Pharma",
    "MDGL": "Madrigal",
    "MDIA": "Mediaco Holding Class A",
    "MDLN": "Medline Class A",
    "MDRR": "Medalist Diversified",
    "MDU": "MDU Resources, (Holding)",
    "MDV": "Modiv Industrial, Class C",
    "MDWD": "MediWound",
    "MDXG": "MiMedx Inc",
    "MDXH": "MDxHealth SA",
    "MEC": "Mayville Engineering",
    "MED": "MEDIFAST INC",
    "MEDP": "Medpace",
    "MEG": "Montrose Environmental",
    "MEGI": "NYLI CBRE Global Infrastruct",
    "MEGL": "Magic Empire Global",
    "MEHA": "Functional Brands",
    "MEI": "Methode Electronics",
    "MELI": "MercadoLibre",
    "MENS": "Jyong Biotech",
    "MEOH": "Methanex",
    "MERC": "Mercer",
    "MESO": "Mesoblast",
    "METC": "Ramaco Resources Class A",
    "METCB": "Ramaco Resources Class B",
    "MFA": "MFA Financial",
    "MFC": "Manulife Financial",
    "MFG": "Mizuho Financial, Sponosred",
    "MFI": "mF",
    "MFIC": "MidCap Financial Investment",
    "MFIN": "Medallion Financial",
    "MFM": "MFS Municipal Income Trust",
    "MG": "Mistras Inc",
    "MGA": "Magna",
    "MGF": "MFS Government Markets Incom",
    "MGIH": "Millennium",
    "MGN": "Megan",
    "MGPI": "MGP Ingredients",
    "MGRC": "McGrath RentCorp",
    "MGRT": "Mega Fortune",
    "MGRX": "Mangoceuticals",
    "MGTX": "MeiraGTx",
    "MGX": "Metagenomi",
    "MGY": "Magnolia Oil & Gas Class A",
    "MGYR": "Magyar Bancorp",
    "MH": "McGraw Hill",
    "MHD": "Blackrock MuniHoldings Fund",
    "MHF": "Western Asset Municipal High",
    "MHO": "M/I Homes",
    "MIAX": "Miami",
    "MICC": "The Magnum Ice Cream",
    "MIDD": "Middleby (The)",
    "MIGI": "Mawson Infrastructure",
    "MIMI": "Mint Incorporation",
    "MIN": "MFS Intermediate Income Trus",
    "MIND": "MIND Technology (DE)",
    "MIR": "Mirion, Class A",
    "MIRA": "MIRA",
    "MIRM": "Mirum",
    "MIST": "Milestone",
    "MITK": "Mitek Systems",
    "MITT": "TPG Mortgage Investment Trus",
    "MIY": "Blackrock MuniYield Michigan",
    "MKL": "Markel",
    "MKTX": "MarketAxess",
    "MKZR": "MacKenzie Realty Capital",
    "MLAB": "Mesa Laboratories",
    "MLCI": "Mount Logan Capital",
    "MLCO": "Melco Resorts & Entertainmen",
    "MLEC": "Moolec Science SA",
    "MLGO": "MicroAlgo",
    "MLI": "Mueller Industries",
    "MLKN": "MillerKnoll",
    "MLM": "Martin Marietta Materials",
    "MLP": "Maui Land & Pineapple",
    "MLR": "Miller Industries",
    "MLTX": "MoonLake Immunotherapeutics",
    "MLYS": "Mineralys",
    "MMD": "NYLI MacKay DefinedTerm Muni",
    "MMED": "MiniMed",
    "MMI": "Marcus & Millichap",
    "MMLP": "Martin Midstream Partners L.",
    "MMM": "3M",
    "MMS": "Maximus",
    "MMT": "MFS Multimarket Income Trust",
    "MMU": "Western Asset Managed Munici",
    "MMYT": "MakeMyTrip",
    "MNDO": "MIND C.T.I.",
    "MNDR": "Mobile-health Network Soluti",
    "MNOV": "Medicinova Inc",
    "MNPR": "Monopar",
    "MNSB": "MainStreet Bancshares",
    "MNSBP": "MainStreet Bancshares Deposi",
    "MNST": "Monster Beverage",
    "MNTK": "Montauk Renewables",
    "MNTN": "MNTN, Class A",
    "MNTS": "Momentus Class A",
    "MNY": "MoneyHero",
    "MOB": "Mobilicom",
    "MOBX": "Mobix Labs Class A",
    "MOD": "Modine Manufacturing",
    "MODD": "Modular Medical",
    "MOLN": "Molecular Partners",
    "MOMO": "Hello",
    "MORN": "Morningstar",
    "MOV": "Movado",
    "MOVE": "Corvex",
    "MPA": "Blackrock MuniYield Pennsylv",
    "MPAA": "Motorcar Parts  of America",
    "MPB": "Mid Penn Bancorp",
    "MPLT": "MapLight",
    "MPT": "Medical Properties Trust, co",
    "MPV": "Barings Participation Invest",
    "MPX": "Marine Products",
    "MQ": "Marqeta Class A",
    "MQY": "Blackrock MuniYield Quality",
    "MRAM": "Everspin",
    "MRBK": "Meridian",
    "MRCY": "Mercury Systems Inc",
    "MRDN": "Meridian",
    "MREO": "Mereo BioPharma",
    "MRKR": "Marker",
    "MRLN": "Merlin",
    "MRM": "MEDIROM Healthcare American",
    "MRNO": "Murano Global Investments",
    "MRP": "Millrose Properties, Class A",
    "MRSH": "Marsh",
    "MRTN": "Marten Transport",
    "MRVI": "Maravai LifeSciences Class A",
    "MRVL": "Marvell Technology",
    "MRX": "Marex",
    "MSA": "MSA Safety",
    "MSAI": "MultiSensor AI",
    "MSB": "Mesabi Trust",
    "MSBI": "Midland States Bancorp",
    "MSD": "Morgan Stanley Emerging Mark",
    "MSDL": "Morgan Stanley Direct Lendin",
    "MSGE": "Madison Square Garden Entert",
    "MSGM": "Motorsport Games Class A",
    "MSGS": "Madison Square Garden Sports",
    "MSGY": "Masonglory",
    "MSI": "Motorola Solutions",
    "MSIF": "MSC Income Fund",
    "MSLE": "Satellos Bioscience",
    "MSM": "MSC Industrial Direct",
    "MSS": "Maison Solutions Class A",
    "MSW": "Ming Shing",
    "MT": "Arcelor Mittal NY Registry S",
    "MTC": "MMTec",
    "MTEK": "Maris-Tech",
    "MTEN": "Mingteng",
    "MTEX": "Mannatech",
    "MTLS": "Materialise NV",
    "MTN": "Vail Resorts",
    "MTR": "Mesa Royalty Trust",
    "MTRX": "Matrix Service",
    "MTUS": "Metallus",
    "MTVA": "MetaVia",
    "MTW": "Manitowoc, (The)",
    "MTX": "Minerals",
    "MTZ": "MasTec",
    "MUA": "Blackrock MuniAssets Fund",
    "MUC": "Blackrock MuniHoldings Calif",
    "MUFG": "Mitsubishi UFJ Financial",
    "MUJ": "Blackrock MuniHoldings New J",
    "MUR": "Murphy Oil",
    "MUX": "McEwen",
    "MVBF": "MVB Financial",
    "MVIS": "MicroVision",
    "MVST": "Microvast",
    "MWA": "MUELLER WATER PRODUCTS",
    "MWH": "SOLV Energy Class A",
    "MWYN": "Marwynn Common stock",
    "MX": "Magnachip Semiconductor",
    "MXCT": "MaxCyte",
    "MXE": "Mexico Equity and Income Fun",
    "MXF": "Mexico Fund, (The)",
    "MXL": "MaxLinear",
    "MYE": "Myers Industries",
    "MYGN": "Myriad Genetics",
    "MYI": "Blackrock MuniYield Quality",
    "MYN": "Blackrock MuniYield New York",
    "MYPS": "PLAYSTUDIOS  Class A",
    "MYSE": "Myseum",
    "MYSZ": "My Size",
    "MZTI": "The Marzetti",
    "NA": "Nano Labs Ltd",
    "NAAS": "NaaS Technology",
    "NABL": "N-able",
    "NAC": "Nuveen California Quality Mu",
    "NAD": "Nuveen Quality Municipal Inc",
    "NAGE": "Niagen Bioscience",
    "NAII": "Natural Alternatives",
    "NAKA": "Nakamoto",
    "NAMI": "Jinxin Technology Holding",
    "NAMM": "Namib Minerals",
    "NAMS": "NewAmsterdam Pharma",
    "NAN": "Nuveen New York Quality Muni",
    "NAT": "Nordic American Tankers",
    "NATH": "Nathan\'s Famous",
    "NATL": "NCR Atleos",
    "NATR": "Nature\'s Sunshine Products",
    "NAUT": "Nautilus Biotechnology",
    "NAVI": "Navient",
    "NAVN": "Navan Class A",
    "NAZ": "Nuveen Arizona Quality Munic",
    "NB": "NioCorp Developments",
    "NBB": "Nuveen Taxable Municipal Inc",
    "NBBK": "NB Bancorp",
    "NBHC": "National Bank",
    "NBIS": "Nebius",
    "NBIX": "Neurocrine Biosciences",
    "NBN": "Northeast Bank",
    "NBP": "NovaBridge Biosciences",
    "NBR": "Nabors Industries",
    "NBTX": "Nanobiotix",
    "NBXG": "Neuberger Next Generation Co",
    "NC": "NACCO Industries",
    "NCA": "Nuveen California Municipal",
    "NCDL": "Nuveen Churchill Direct Lend",
    "NCEL": "NewcelX",
    "NCI": "Neo-Concept",
    "NCMI": "National CineMedia",
    "NCNA": "NuCana American Depositary S",
    "NCPL": "Netcapital",
    "NCRA": "Nocera",
    "NCSM": "NCS Multistage",
    "NCT": "Intercont (Cayman) Class A O",
    "NCTY": "The9",
    "NCV": "Virtus Convertible & Income",
    "NCZ": "Virtus Convertible & Income",
    "NDLS": "Noodles & Class A",
    "NDMO": "Nuveen Dynamic Municipal Opp",
    "NDRA": "ENDRA Life Sciences",
    "NDSN": "Nordson",
    "NE": "Noble A",
    "NEA": "Nuveen AMT-Free Quality Muni",
    "NECB": "NorthEast Community Bancorp",
    "NEGG": "Newegg Commerce",
    "NEO": "NeoGenomics",
    "NEON": "Neonode",
    "NEOV": "NeoVolta",
    "NEPH": "Nephros",
    "NERV": "Minerva Neurosciences Inc",
    "NESR": "National Energy Services Reu",
    "NEU": "NewMarket Corp",
    "NEUP": "Neuphoria",
    "NEWT": "NewtekOne",
    "NEXA": "Nexa Resources",
    "NEXM": "NexMetals Mining",
    "NEXN": "Nexxen",
    "NEXT": "NextDecade",
    "NFBK": "Northfield Bancorp (Delaware",
    "NFE": "New Fortress Energy Class A",
    "NFG": "National Fuel Gas",
    "NFJ": "Virtus Dividend, Interest &",
    "NGEN": "NervGen Pharma Common stock",
    "NGNE": "Neurogene",
    "NGS": "Natural Gas Services",
    "NGVC": "Natural Grocers by Vitamin C",
    "NGVT": "Ingevity",
    "NHI": "National Health Investors",
    "NHIC": "NewHold Investment Corp III",
    "NHTC": "Natural Health Trends",
    "NIC": "Nicolet Bankshares",
    "NICE": "NICE Ltd",
    "NIE": "Virtus Equity & Convertible",
    "NIM": "Nuveen Select Maturities Mun",
    "NIPG": "NIP",
    "NIQ": "NIQ Global Intelligence",
    "NIU": "Niu",
    "NIVF": "NewGenIvf",
    "NIXX": "Nixxy",
    "NJR": "NewJersey Resources",
    "NKLR": "Terra Innovatum Global Ordin",
    "NKSH": "National Bankshares",
    "NKTR": "Nektar",
    "NKTX": "Nkarta",
    "NKX": "Nuveen California AMT-Free Q",
    "NL": "NL Industries",
    "NLOP": "Net Lease Office Properties",
    "NLY": "Annaly Capital Management",
    "NMAI": "Nuveen Multi-Asset Income Fu",
    "NMAX": "Newsmax, Class B",
    "NMCO": "Nuveen Municipal Credit Oppo",
    "NMFC": "New Mountain Finance",
    "NMG": "Nouveau Monde Graphite",
    "NMI": "Nuveen Municipal Income Fund",
    "NMRA": "Neumora",
    "NMRK": "Newmark Class A",
    "NMS": "Nuveen Minnesota Quality Mun",
    "NMT": "Nuveen Massachusetts Quality",
    "NMTC": "NeuroOne Medical",
    "NMZ": "Nuveen Municipal High Income",
    "NN": "NextNav",
    "NNBR": "NN",
    "NNDM": "Nano Dimension",
    "NNE": "Nano Nuclear Energy",
    "NNI": "Nelnet",
    "NNN": "NNN REIT",
    "NNNN": "Anbio Biotechnology",
    "NNOX": "NANO-X IMAGING LTD",
    "NNY": "Nuveen New York Municipal Va",
    "NOA": "North American Construction",
    "NODK": "NI",
    "NOEM": "CO2 Energy Transition",
    "NOG": "Northern Oil and Gas",
    "NOM": "Nuveen Missouri Quality Muni",
    "NOMA": "NOMADAR Class A",
    "NOMD": "Nomad Foods",
    "NOTE": "FiscalNote, Class A common s",
    "NOTV": "Inotiv",
    "NOV": "NOV",
    "NP": "Neptune Insurance Class A",
    "NPB": "Northpointe Bancshares",
    "NPCE": "Neuropace",
    "NPCT": "Nuveen Core Plus Impact Fund",
    "NPK": "National Presto Industries",
    "NPKI": "NPK",
    "NPO": "Enpro",
    "NPT": "Texxon Holding Ordinary shar",
    "NPV": "Nuveen Virginia Quality Muni",
    "NPWR": "NET Power Class A",
    "NQP": "Nuveen Pennsylvania Quality",
    "NRC": "National Research (Delaware)",
    "NRDS": "NerdWallet Class A",
    "NRDY": "Nerdy Class A",
    "NREF": "NexPoint Real Estate Finance",
    "NRGV": "Energy Vault",
    "NRIM": "Northrim BanCorp Inc",
    "NRIX": "Nurix Common stock",
    "NRK": "Nuveen New York AMT-Free Qua",
    "NRP": "Natural Resource Partners LP",
    "NRSN": "NeuroSense",
    "NRT": "North European Oil Royality",
    "NRXP": "NRX",
    "NSIT": "Insight Enterprises",
    "NSP": "Insperity",
    "NSPR": "InspireMD",
    "NSSC": "NAPCO Security",
    "NSTS": "NSTS Bancorp",
    "NSYS": "Nortech Systems",
    "NTB": "Bank of N.T. Butterfield & S",
    "NTCL": "NETCLASS TECHNOLOGY INC",
    "NTCT": "NetScout Systems",
    "NTES": "NetEase",
    "NTGR": "NETGEAR",
    "NTHI": "NeOnc",
    "NTIC": "Northern",
    "NTNX": "Nutanix Class A",
    "NTR": "Nutrien",
    "NTRB": "Nutriband",
    "NTRP": "NextTrip",
    "NTSK": "Netskope Class A",
    "NTWK": "NetSol Common  Stock",
    "NTZ": "Natuzzi, S.p.A.",
    "NUAI": "New Era Energy & Digital",
    "NUCL": "Eagle Nuclear Energy Common",
    "NUS": "Nu Skin Enterprises",
    "NUTX": "Nutex Health",
    "NUV": "Nuveen Municipal Value Fund",
    "NUVB": "Nuvation Bio Class A",
    "NUVL": "Nuvalent Class A",
    "NUW": "Nuveen AMT-Free Municipal Va",
    "NUWE": "Nuwellis",
    "NVA": "Nova Minerals",
    "NVCT": "Nuvectis Pharma",
    "NVEC": "NVE",
    "NVG": "Nuveen AMT-Free Municipal Cr",
    "NVGS": "Navigator (Marshall Islands)",
    "NVMI": "Nova",
    "NVNI": "Nvni",
    "NVNO": "enVVeno Medical",
    "NVO": "Novo Nordisk A/S",
    "NVRI": "Enviri",
    "NVS": "Novartis",
    "NVST": "Envista",
    "NVT": "nVent Electric",
    "NVTS": "Navitas Semiconductor",
    "NVVE": "Nuvve Holding",
    "NVX": "NOVONIX",
    "NWE": "NorthWestern Energy",
    "NWFL": "Norwood Financial",
    "NWGL": "CL Workshop",
    "NWL": "Newell Brands",
    "NWN": "Northwest Natural Holding",
    "NWPX": "NWPX Infrastructure",
    "NWS": "News Class B",
    "NWSA": "News Class A",
    "NWTG": "Newton Golf",
    "NX": "Quanex Building Products",
    "NXDR": "Nextdoor, Class A",
    "NXDT": "NexPoint Diversified Real Es",
    "NXE": "Nexgen Energy",
    "NXG": "NXG NextGen Infrastructure I",
    "NXGL": "NexGel Inc",
    "NXJ": "Nuveen New Jersey Qualified",
    "NXL": "Nexalin Technology",
    "NXP": "Nuveen Select Tax Free Incom",
    "NXPI": "NXP Semiconductors",
    "NXPL": "NextPlat Corp",
    "NXRT": "NexPoint Residential Trust",
    "NXST": "Nexstar Media",
    "NXT": "Nextpower Class A",
    "NXTC": "NextCure",
    "NXTS": "Nexentis",
    "NXTT": "Next Technology Holding",
    "NXXT": "NextNRG",
    "NYAX": "Nayax",
    "NYC": "American Strategic Investmen",
    "NYT": "New York Times (The)",
    "NYXH": "Nyxoah SA",
    "NZF": "Nuveen Municipal Credit Inco",
    "OABI": "OmniAb",
    "OBAI": "Our Bond",
    "OBIO": "Orchestra BioMed",
    "OBK": "Origin Bancorp",
    "OBT": "Orange County Bancorp",
    "OC": "Owens Corning Inc New",
    "OCC": "Optical Cable",
    "OCCI": "OFS Credit",
    "OCFC": "OceanFirst Financial",
    "OCG": "Oriental Culture Holding LTD",
    "OCGN": "Ocugen",
    "OCS": "Oculis Holding Ordinary shar",
    "OCUL": "Ocular Therapeutix",
    "ODC": "Oil-Dri Of America",
    "ODD": "ODDITY Tech",
    "ODV": "Osisko Development",
    "ODYS": "Odysight.ai",
    "OEC": "Orion",
    "OESX": "Orion Energy Systems",
    "OFAL": "OFA",
    "OFIX": "Orthofix Medical (DE)",
    "OFLX": "Omega Flex",
    "OFRM": "Once Upon a Farm, PBC",
    "OFS": "OFS Capital",
    "OGI": "Organigram Global",
    "OGN": "Organon &",
    "OGS": "ONE Gas",
    "OHI": "Omega Healthcare Investors",
    "OI": "O-I Glass",
    "OIA": "Invesco Municipal Income Opp",
    "OII": "Oceaneering",
    "OIO": "OIO",
    "OIS": "Oil States",
    "OKLO": "Oklo Class A common stock",
    "OKTA": "Okta Class A",
    "OKUR": "OnKure Class A",
    "OKYO": "OKYO Pharma",
    "OLB": "The OLB",
    "OLED": "Universal Display",
    "OLMA": "Olema",
    "OLN": "Olin",
    "OLOX": "Olenox Industries",
    "OLP": "One Liberty Properties",
    "OM": "Outset Medical",
    "OMAB": "Grupo Aeroportuario del Cent",
    "OMDA": "Omada Health",
    "OMER": "Omeros",
    "OMEX": "Odyssey Marine Exploration",
    "OMF": "OneMain",
    "OMH": "Ohmyhome",
    "OMSE": "OMS Energy",
    "ON": "ON Semiconductor",
    "ONB": "Old National Bancorp",
    "ONC": "BeOne Medicines",
    "ONCO": "Onconetix",
    "ONCY": "Oncolytics Biotech",
    "ONDS": "Ondas",
    "ONEG": "OneConstruction",
    "ONFO": "Onfolio",
    "ONIT": "Onity",
    "ONL": "Orion Properties",
    "ONMD": "OneMedNet Corp Class A",
    "ONTF": "ON24",
    "OOMA": "Ooma",
    "OPAD": "Offerpad Solutions Class A",
    "OPAL": "OPAL Fuels Class A",
    "OPBK": "OP Bancorp",
    "OPCH": "Option Care Health",
    "OPFI": "OppFi Class A",
    "OPK": "OPKO Health",
    "OPLN": "OPENLANE",
    "OPP": "RiverNorth/DoubleLine Strate",
    "OPRA": "Opera",
    "OPRT": "Oportun Financial",
    "OPRX": "OptimizeRx",
    "OPTU": "Optimum Communications, Clas",
    "OPTX": "Syntec Optics Class A",
    "OPXS": "Optex Systems",
    "OPY": "Oppenheimer, Class A (DE)",
    "OR": "OR Royalties",
    "ORA": "Ormat",
    "ORBS": "Eightco",
    "ORC": "Orchid Island Capital",
    "ORGN": "Origin Materials",
    "ORGO": "Organogenesis Class A",
    "ORI": "Old Republic",
    "ORIC": "Oric",
    "ORIO": "Orion Digital",
    "ORIQ": "Origin Investment Corp I",
    "ORIS": "Oriental Rise",
    "ORKA": "Oruka",
    "ORKT": "Orangekloud Technology",
    "ORMP": "Oramed",
    "ORN": "Orion, Common",
    "ORRF": "Orrstown Financial Services",
    "OS": "OneStream Class A",
    "OSBC": "Old Second Bancorp",
    "OSCR": "Oscar Health, Class A",
    "OSG": "Octave Specialty",
    "OSK": "Oshkosh (Holding)Common Stoc",
    "OSPN": "OneSpan",
    "OSRH": "OSR",
    "OSS": "One Stop Systems",
    "OSUR": "OraSure",
    "OSW": "OneSpaWorld",
    "OTEX": "Open Text",
    "OTF": "Blue Owl Technology Finance",
    "OTLK": "Outlook",
    "OTLY": "Oatly AB",
    "OUST": "Ouster",
    "OUT": "OUTFRONT Media",
    "OVBC": "Ohio Valley Banc",
    "OVID": "Ovid",
    "OVLY": "Oak Valley Bancorp (CA)",
    "OVV": "Ovintiv (DE)",
    "OWL": "Blue Owl Capital Class A",
    "OWLS": "OBOOK Class A",
    "OWLT": "Owlet, Class A",
    "OXBR": "Oxbridge Re",
    "OXLC": "Oxford Lane Capital",
    "OXM": "Oxford Industries",
    "OXSQ": "Oxford Square Capital",
    "OZK": "Bank OZK",
    "PAAS": "Pan American Silver",
    "PAC": "Grupo Aeroportuario Del Paci",
    "PACB": "Pacific Biosciences of Calif",
    "PACK": "Ranpak Corp Class A",
    "PACS": "PACS",
    "PAGS": "PagSeguro Digital Class A",
    "PAI": "Western Asset Investment Gra",
    "PAL": "Proficient Auto Logistics",
    "PALI": "Palisade Bio",
    "PAM": "Pampa Energia",
    "PAMT": "PAMT CORP",
    "PANL": "Pangaea Logistics Solutions",
    "PAR": "PAR Technology",
    "PARK": "Park Dental Partners",
    "PARR": "Par Pacific",
    "PASG": "Passage Bio",
    "PATK": "Patrick Industries",
    "PAVM": "PAVmed",
    "PAVS": "Paranovus Entertainment Tech",
    "PAX": "Patria Investments Class A",
    "PAXS": "PIMCO Access Income Fund",
    "PAY": "Paymentus, Class A",
    "PAYC": "Paycom Software",
    "PAYO": "Payoneer Global",
    "PAYP": "PayPay",
    "PAYS": "Paysign",
    "PAYX": "Paychex",
    "PB": "Prosperity Bancshares",
    "PBA": "Pembina Pipeline (Canada)",
    "PBF": "PBF Energy Class A",
    "PBFS": "Pioneer Bancorp",
    "PBH": "Prestige Consumer Healthcare",
    "PBHC": "Pathfinder Bancorp (MD)",
    "PBI": "Pitney Bowes",
    "PBM": "Psyence Biomedical",
    "PBR": "Petroleo Brasileiro Petrobra",
    "PBT": "Permian Basin Royalty Trust",
    "PBYI": "Puma Biotechnology Inc",
    "PCAR": "PACCAR",
    "PCB": "PCB Bancorp",
    "PCF": "High Income Securities Fund",
    "PCLA": "PicoCELA",
    "PCM": "PCM Fund",
    "PCN": "Pimco Corporate & Income Str",
    "PCOR": "Procore",
    "PCQ": "PIMCO California Municipal I",
    "PCSA": "Processa",
    "PCSC": "Perceptive Capital Solutions",
    "PCT": "PureCycle Common stock",
    "PCVX": "Vaxcyte",
    "PCYO": "Pure Cycle",
    "PD": "PagerDuty",
    "PDC": "Perpetuals.com Ltd",
    "PDCC": "Pearl Diver Credit",
    "PDD": "PDD",
    "PDEX": "Pro-Dex",
    "PDFS": "PDF Solutions",
    "PDI": "PIMCO Dynamic Income Fund",
    "PDLB": "Ponce Financial",
    "PDM": "Piedmont Realty Trust, Class",
    "PDO": "PIMCO Dynamic Income Opportu",
    "PDS": "Precision Drilling",
    "PDSB": "PDS Biotechnology",
    "PDT": "John Hancock Premium Dividen",
    "PDX": "PIMCO Dynamic Income Strateg",
    "PDYN": "Palladyne AI",
    "PEB": "Pebblebrook Hotel Trust",
    "PEBK": "Peoples Bancorp of North Car",
    "PEBO": "Peoples Bancorp",
    "PECO": "Phillips Edison &",
    "PEG": "Public Service Enterprise",
    "PEGA": "Pegasystems",
    "PEN": "Penumbra",
    "PENG": "Penguin Solutions",
    "PEO": "Adams Natural Resources Fund",
    "PEPG": "PepGen",
    "PERF": "Perfect Class A Ordinary Sha",
    "PERI": "Perion Network",
    "PESI": "Perma-Fix Environmental Serv",
    "PETZ": "TDH",
    "PEW": "GrabAGun Digital",
    "PFAI": "Pinnacle Food Class A",
    "PFG": "Principal Financial Inc",
    "PFL": "PIMCO Income Strategy Fund S",
    "PFN": "PIMCO Income Strategy Fund I",
    "PFS": "Provident Financial Services",
    "PFSA": "Profusa",
    "PFSI": "PennyMac Financial Services",
    "PFX": "PhenixFIN",
    "PGC": "Peapack-Gladstone Financial",
    "PGEN": "Precigen",
    "PGNY": "Progyny",
    "PGP": "Pimco Global StocksPlus & In",
    "PGY": "Pagaya",
    "PGZ": "Principal Real Estate Income",
    "PHAR": "Pharming ADS each representi",
    "PHAT": "Phathom",
    "PHG": "Koninklijke Philips NY Regis",
    "PHI": "PLDT Sponsored ADR",
    "PHIN": "PHINIA",
    "PHIO": "Phio",
    "PHK": "Pimco High Income Fund",
    "PHOE": "Phoenix Asia",
    "PHR": "Phreesia",
    "PHUN": "Phunware",
    "PHVS": "Pharvaris",
    "PI": "Impinj",
    "PICS": "PicS Class A",
    "PII": "Polaris",
    "PIII": "P3 Health Partners Class A",
    "PIM": "Putnam Master Intermediate I",
    "PIPR": "Piper Sandler Companies",
    "PJT": "PJT Partners Class A",
    "PKBK": "Parke Bancorp",
    "PKG": "Packaging of America",
    "PKOH": "Park-Ohio",
    "PKST": "Peakstone Realty Trust",
    "PL": "Planet Labs PBC Class A",
    "PLAB": "Photronics",
    "PLAY": "Dave & Buster\'s Entertainmen",
    "PLBC": "Plumas Bancorp",
    "PLBL": "Polibeli Ltd",
    "PLBY": "Playboy",
    "PLCE": "Children\'s Place (The)",
    "PLMR": "Palomar Common stock",
    "PLPC": "Preformed Line Products",
    "PLRX": "Pliant",
    "PLRZ": "Polyrizon",
    "PLSE": "Pulse Biosciences Inc (DE)",
    "PLSM": "Pulsenmore",
    "PLUG": "Plug Power",
    "PLUR": "Pluri",
    "PLUS": "ePlus inc.",
    "PLUT": "Plutus Financial",
    "PLXS": "Plexus",
    "PLYX": "Polaryx",
    "PMAX": "Powell Max",
    "PMCB": "PharmaCyte  Biotech",
    "PMEC": "Primech",
    "PML": "Pimco Municipal Income Fund",
    "PMM": "Putnam Managed Municipal Inc",
    "PMN": "ProMIS Neurosciences (ON)",
    "PMO": "Putnam Municipal Opportuniti",
    "PMT": "PennyMac Mortgage Investment",
    "PMTS": "CPI Card",
    "PMVP": "PMV",
    "PN": "Skycorp Solar",
    "PNBK": "Patriot National Bancorp",
    "PNI": "Pimco New York Municipal Inc",
    "PNNT": "PennantPark Investment",
    "PNRG": "PrimeEnergy Resources",
    "PNTG": "The Pennant",
    "POCI": "Precision Optics Common stoc",
    "PODC": "PodcastOne",
    "POET": "POET",
    "POLA": "Polar Power",
    "POM": "POMDOCTOR LIMITED",
    "PONY": "Pony AI",
    "POOL": "Pool",
    "POR": "Portland General Electric Co",
    "POST": "Post",
    "POWL": "Powell Industries",
    "PPBT": "Purple Biotech",
    "PPC": "Pilgrim\'s Pride",
    "PPCB": "Propanc Biopharma",
    "PPHC": "Public Policy Holding",
    "PPIH": "Perma-Pipe",
    "PPLC": "PPL Corporate Units",
    "PPSI": "Pioneer Power Solutions",
    "PPT": "Putnam Premier Income Trust",
    "PPTA": "Perpetua Resources",
    "PR": "Permian Resources Class A",
    "PRA": "ProAssurance",
    "PRAA": "PRA",
    "PRCT": "PROCEPT BioRobotics",
    "PRDO": "Perdoceo Education",
    "PRE": "Prenetics Global Class A Ord",
    "PRFX": "PRF",
    "PRGS": "Progress Software (DE)",
    "PRHI": "Presurance",
    "PRI": "Primerica",
    "PRKS": "United Parks & Resorts",
    "PRLB": "Proto Labs, Common stock",
    "PRM": "Perimeter Solutions, SA",
    "PRMB": "Primo Brands Class A",
    "PRME": "Prime Medicine",
    "PROF": "Profound Medical",
    "PROK": "ProKidney",
    "PROP": "Prairie Operating",
    "PROV": "Provident Financial",
    "PRPL": "Purple Innovation",
    "PRPO": "Precipio",
    "PRQR": "ProQR",
    "PRSO": "Peraso",
    "PRSU": "Pursuit Attractions and Hosp",
    "PRTC": "PureTech Health",
    "PRTH": "Priority Technology",
    "PRVA": "Privia Health",
    "PRZO": "ParaZero",
    "PSBD": "Palmer Square Capital BDC",
    "PSFE": "Paysafe",
    "PSHG": "Performance Shipping",
    "PSIG": "PS",
    "PSIX": "Power Solutions",
    "PSKY": "Paramount Skydance Class B",
    "PSMT": "PriceSmart",
    "PSN": "Parsons",
    "PSNL": "Personalis",
    "PSNY": "Polestar Automotive Holding",
    "PSO": "Pearson, Plc",
    "PSQH": "PSQ, Class A",
    "PSTG": "Everpure, Class A common sto",
    "PSTV": "PLUS THERAPEUTICS",
    "PTCT": "PTC",
    "PTEN": "Patterson-UTI Energy",
    "PTGX": "Protagonist",
    "PTLE": "PTL LTD",
    "PTON": "Peloton Interactive Class A",
    "PTRN": "Pattern Series A",
    "PTY": "Pimco Corporate & Income Opp",
    "PUK": "Prudential Public",
    "PULM": "Pulmatrix",
    "PURR": "Hyperliquid Strategies Inc",
    "PVLA": "Palvella",
    "PWP": "Perella Weinberg Partners Cl",
    "PWR": "Quanta Services",
    "PXED": "Phoenix Education Partners",
    "PXS": "Pyxis Tankers",
    "PYPD": "PolyPid",
    "PYT": "PPlus Tr GSC-2 Tr Ctf Fltg R",
    "PYXS": "Pyxis Oncology",
    "PZZA": "Papa John\'s",
    "Q": "Qnity Electronics",
    "QBTS": "D-Wave Quantum",
    "QCLS": "Q/C",
    "QCRH": "QCR",
    "QDEL": "QuidelOrtho",
    "QFIN": "Qfin",
    "QGEN": "Qiagen",
    "QH": "Quhuo",
    "QMCO": "Quantum",
    "QNCX": "Quince",
    "QNRX": "Quoin",
    "QNST": "QuinStreet",
    "QNTM": "Quantum Biopharma Class B Su",
    "QQQX": "Nuveen NASDAQ 100 Dynamic Ov",
    "QRHC": "Quest Resource Holding",
    "QS": "QuantumScape Class A",
    "QSI": "Quantum-Si Class A",
    "QTI": "QT Imaging",
    "QTRX": "Quanterix",
    "QTTB": "Q32 Bio",
    "QUAD": "Quad Graphics Class A",
    "QUCY": "Mainz Biomed",
    "QUIK": "QuickLogic",
    "QURE": "uniQure",
    "QVCGA": "QVC Series A",
    "QXO": "QXO",
    "R": "Ryder System",
    "RA": "Brookfield Real Assets Incom",
    "RACE": "Ferrari",
    "RADX": "Radiopharm Theranostics",
    "RAIL": "FreightCar America",
    "RAIN": "Rain Enhancement Holdco Clas",
    "RAL": "Ralliant",
    "RAMP": "LiveRamp",
    "RAND": "Rand Capital",
    "RANI": "Rani Class A",
    "RAPP": "Rapport",
    "RAVE": "Rave Restaurant",
    "RAY": "Raytech Holding",
    "RAYA": "Erayak Power Solution",
    "RBA": "RB Global",
    "RBB": "RBB Bancorp",
    "RBBN": "Ribbon Communications",
    "RBC": "RBC Bearings",
    "RBCAA": "Republic Bancorp Class A",
    "RBKB": "Rhinebeck Bancorp",
    "RBNE": "Robin Energy",
    "RBRK": "Rubrik, Class A",
    "RC": "Ready Capital",
    "RCAT": "Red Cat",
    "RCEL": "Avita Medical",
    "RCI": "Rogers Communication",
    "RCKT": "Rocket",
    "RCKY": "Rocky Brands",
    "RCMT": "RCM",
    "RCON": "Recon Technology",
    "RCS": "PIMCO Strategic Income Fund",
    "RCT": "RedCloud",
    "RDCM": "Radcom",
    "RDDT": "Reddit, Class A",
    "RDGT": "Ridgetech",
    "RDHL": "Redhill Biopharma",
    "RDI": "Reading Inc Class A",
    "RDIB": "Reading Inc Class B",
    "RDNT": "RadNet",
    "RDVT": "Red Violet",
    "RDW": "Redwire",
    "RDWR": "Radware",
    "RDY": "Dr. Reddy\'s Laboratories Ltd",
    "RDZN": "Roadzen",
    "REAL": "The RealReal",
    "REAX": "The Real Brokerage",
    "REBN": "Reborn Coffee",
    "RECT": "Rectitude Ltd",
    "REE": "REE Automotive",
    "REFI": "Chicago Atlantic Real Estate",
    "REFR": "Research Frontiers",
    "REKR": "Rekor Systems",
    "RELL": "Richardson Electronics",
    "RELY": "Remitly Global",
    "RENT": "Rent the Runway Class A",
    "RENX": "RenX Enterprises",
    "REPL": "Replimune",
    "RES": "RPC",
    "RETO": "ReTo Eco-Solutions Class A S",
    "REVB": "Revelation Biosciences",
    "REXR": "Rexford Industrial Realty",
    "REYN": "Reynolds Consumer Products",
    "REZI": "Resideo",
    "RFI": "Cohen & Steers Total Return",
    "RFIL": "RF Industries",
    "RFL": "Rafael, Class B",
    "RFM": "RiverNorth Flexible Municipa",
    "RFMZ": "RiverNorth Flexible Municipa",
    "RGA": "Reinsurance of America",
    "RGC": "Regencell Bioscience",
    "RGCO": "RGC Resources",
    "RGEN": "Repligen",
    "RGLD": "Royal Gold",
    "RGNX": "REGENXBIO",
    "RGP": "Resources Connection",
    "RGR": "Sturm, Ruger &",
    "RGS": "Regis",
    "RGT": "Royce Global Trust",
    "RHLD": "Resolute Management",
    "RICK": "RCI Hospitality",
    "RIG": "Transocean Ltd (Switzerland)",
    "RIGL": "Rigel",
    "RILY": "BRC",
    "RIME": "Algorhythm",
    "RIO": "Rio Tinto Plc",
    "RITM": "Rithm Capital",
    "RITR": "Reitar Logtech Ordinary shar",
    "RIV": "RiverNorth Opportunities Fun",
    "RJET": "Republic Airways",
    "RJF": "Raymond James Financial",
    "RKDA": "Arcadia Biosciences",
    "RKT": "Rocket Companies, Class A",
    "RLAY": "Relay",
    "RLI": "RLI (DE)",
    "RLJ": "RLJ Lodging Trust $0.01 par",
    "RLMD": "Relmada",
    "RLTY": "Cohen & Steers Real Estate O",
    "RLYB": "Rallybio",
    "RM": "Regional Management",
    "RMAX": "RE/MAX, Class A",
    "RMBI": "Richmond Mutual Bancorporati",
    "RMCF": "Rocky Mountain Chocolate Fac",
    "RMCO": "Royalty Management Holding C",
    "RMI": "RiverNorth Opportunistic Mun",
    "RMM": "RiverNorth Managed Duration",
    "RMMZ": "RiverNorth Managed Duration",
    "RMNI": "Rimini Street (DE)",
    "RMR": "The RMR Class A",
    "RMSG": "Real Messenger",
    "RMT": "Royce Micro-Cap Trust",
    "RMTI": "Rockwell Medical",
    "RNA": "Atrium",
    "RNAC": "Cartesian",
    "RNAZ": "TransCode",
    "RNG": "RingCentral, Class A",
    "RNTX": "Rein",
    "RNW": "ReNew Energy Global",
    "RNXT": "RenovoRx",
    "ROAD": "Construction Partners Class",
    "ROC": "Rank One Computing Common st",
    "ROCK": "Gibraltar Industries",
    "ROG": "Rogers",
    "ROIV": "Roivant Sciences",
    "ROKU": "Roku Class A",
    "ROL": "Rollins",
    "ROMA": "Roma Green Finance",
    "ROP": "Roper",
    "RPAY": "Repay Class A",
    "RPC": "Ridgepost Capital, Class A",
    "RPD": "Rapid7",
    "RPGL": "Republic Power",
    "RPID": "Rapid Micro Biosystems Class",
    "RPRX": "Royalty Pharma",
    "RPT": "Rithm Property Trust Common",
    "RQI": "Cohen & Steers Quality Incom",
    "RR": "Richtech Robotics Class B",
    "RRBI": "Red River Bancshares",
    "RRC": "Range Resources",
    "RRR": "Red Rock Resorts Class A",
    "RSF": "RiverNorth Capital and Incom",
    "RSI": "Rush Street Interactive, Cla",
    "RSKD": "Riskified",
    "RSSS": "Research Solutions Inc",
    "RSVR": "Reservoir Media",
    "RUBI": "Rubico",
    "RUM": "Rumble Class A",
    "RUSHB": "Rush Enterprises Class B",
    "RVI": "Robinhood Ventures Fund I",
    "RVLV": "Revolve, Class A",
    "RVPH": "Reviva",
    "RVSB": "Riverview Bancorp Inc",
    "RVSN": "Rail Vision Ordinary Share",
    "RVT": "Royce Small-Cap Trust",
    "RVYL": "Ryvyl",
    "RWAY": "Runway Growth Finance",
    "RWT": "Redwood Trust",
    "RXO": "RXO",
    "RXST": "RxSight",
    "RY": "Royal Bank Of Canada",
    "RYAAY": "Ryanair",
    "RYAM": "Rayonier Advanced Materials",
    "RYAN": "Ryan Specialty, Class A",
    "RYET": "Ruanyun Edai Technology Ordi",
    "RYM": "RYTHM",
    "RYN": "Rayonier REIT",
    "RYOJ": "rYojbaba",
    "RYTM": "Rhythm",
    "RYZ": "Ryerson Holding",
    "RZLT": "Rezolute (NV)",
    "RZLV": "Rezolve AI",
    "SA": "Seabridge Gold, (Canada)",
    "SABA": "Saba Capital Income & Opport",
    "SABR": "Sabre",
    "SABS": "SAB Biotherapeutics",
    "SAFT": "Safety Insurance",
    "SAFX": "XCF Global Class A",
    "SAGT": "SAGTEC GLOBAL LIMITED Class",
    "SAIC": "Science Applications",
    "SAIH": "SAIHEAT",
    "SAIL": "SailPoint",
    "SAM": "Boston Beer, (The)",
    "SAMG": "Silvercrest Asset Management",
    "SAN": "Banco Santander, Sponsored A",
    "SANA": "Sana Biotechnology",
    "SANG": "Sangoma",
    "SANM": "Sanmina",
    "SAP": "SAP  ADS",
    "SAR": "Saratoga Investment Corp New",
    "SARO": "StandardAero",
    "SATL": "Satellogic Class A",
    "SATS": "EchoStar",
    "SB": "Safe Bulkers ($0.001 par val",
    "SBAC": "SBA Communications Class A",
    "SBC": "SBC Medical",
    "SBCF": "Seacoast Banking of Florida",
    "SBDS": "Solo Brands, Class A",
    "SBET": "Sharplink",
    "SBFG": "SB Financial",
    "SBFM": "Sunshine Biopharma",
    "SBGI": "Sinclair Class A",
    "SBH": "Sally Beauty, (Name to be ch",
    "SBI": "Western Asset Intermediate M",
    "SBLK": "Star Bulk Carriers",
    "SBLX": "StableX",
    "SBR": "Sabine Royalty Trust",
    "SBRA": "Sabra Health Care REIT",
    "SBSI": "Southside Bancshares",
    "SBXD": "SilverBox Corp IV",
    "SBXE": "SilverBox Corp V",
    "SCAG": "Scage Future",
    "SCCO": "Southern Copper",
    "SCD": "LMP Capital and Income Fund",
    "SCHL": "Scholastic",
    "SCI": "Service",
    "SCKT": "Socket Mobile",
    "SCL": "Stepan",
    "SCLX": "Scilex Holding",
    "SCM": "Stellus Capital Investment",
    "SCNI": "Scinai Immunotherapeutics",
    "SCNX": "Scienture",
    "SCOR": "comScore",
    "SCPQ": "Social Commerce Partners",
    "SCSC": "ScanSource",
    "SCVL": "Shoe Carnival",
    "SCWO": "374Water",
    "SCYX": "SCYNEXIS",
    "SCZM": "Santacruz Silver Mining",
    "SD": "SandRidge Energy",
    "SDA": "SunCar Technology",
    "SDGR": "Schrodinger",
    "SDHC": "Smith Douglas Homes Class A",
    "SDHY": "PGIM Short Duration High Yie",
    "SDOT": "Sadot",
    "SDRL": "Seadrill",
    "SDST": "Stardust Power",
    "SEAT": "Vivid Seats Class A",
    "SEE": "Sealed Air",
    "SEED": "Origin Agritech",
    "SEER": "Seer Class A",
    "SEG": "Seaport Entertainment",
    "SEGG": "Sports Entertainment Gaming",
    "SEI": "Solaris Energy Infrastructur",
    "SEIC": "SEI Investments",
    "SELF": "Global Self Storage",
    "SELX": "Semilux",
    "SEM": "Select Medical",
    "SEMR": "SEMrush, Class A",
    "SENEB": "Seneca Foods Class B",
    "SENS": "Senseonics",
    "SEPN": "Septerna",
    "SERA": "Sera Prognostics Class A",
    "SERV": "Serve Robotics",
    "SES": "SES AI Class A",
    "SEV": "Aptera Motors Class B",
    "SEVN": "Seven Hills Realty Trust",
    "SEZL": "Sezzle",
    "SF": "Stifel Financial",
    "SFBC": "Sound Financial Bancorp",
    "SFBS": "ServisFirst Bancshares",
    "SFD": "Smithfield Foods",
    "SFHG": "Samfine Creation Class A Ord",
    "SFL": "SFL Ltd",
    "SFST": "Southern First Bancshares",
    "SFWL": "Shengfeng Development",
    "SG": "Sweetgreen, Class A",
    "SGA": "Saga Communications Class A",
    "SGC": "Superior of Companies",
    "SGHC": "Super (SGHC)",
    "SGHT": "Sight Sciences",
    "SGI": "Somnigroup",
    "SGLY": "Singularity Future Technolog",
    "SGML": "Sigma Lithium",
    "SGMT": "Sagimet Biosciences Series A",
    "SGP": "SpyGlass Pharma",
    "SGRP": "SPAR",
    "SGRY": "Surgery Partners",
    "SGU": "Star L.P.",
    "SHAZ": "SharonAI Class A",
    "SHBI": "Shore Bancshares Inc",
    "SHC": "Sotera Health",
    "SHEN": "Shenandoah Telecommunication",
    "SHFS": "SHF Class A",
    "SHIM": "Shimmick",
    "SHIP": "Seanergy Maritime",
    "SHMD": "SCHMID",
    "SHOO": "Steven Madden",
    "SHPH": "Shuttle",
    "SI": "Shoulder Innovations",
    "SIBN": "SI-BONE",
    "SID": "Companhia Siderurgica Nacion",
    "SIEB": "Siebert Financial",
    "SIFY": "Sify",
    "SIG": "Signet Jewelers",
    "SIGA": "SIGA",
    "SIGI": "Selective Insurance",
    "SII": "Sprott",
    "SILA": "Sila Realty Trust",
    "SILC": "Silicom Ltd",
    "SILO": "Silo Pharma",
    "SIMO": "Silicon Motion Technology",
    "SINT": "SiNtx",
    "SION": "Sionna",
    "SIRI": "SiriusXM",
    "SITC": "SITE Centers",
    "SITE": "SiteOne Landscape Supply",
    "SJ": "Scienjoy Holding",
    "SJT": "San Juan Basin Royalty Trust",
    "SKBL": "Skyline Builders Holding",
    "SKE": "Skeena Resources",
    "SKIL": "Skillsoft Class A",
    "SKIN": "The Beauty Health Class A",
    "SKK": "SKK",
    "SKLZ": "Skillz Class A",
    "SKM": "SK Telecom",
    "SKT": "Tanger",
    "SKWD": "Skyward Specialty Insurance",
    "SKY": "Champion Homes",
    "SKYE": "Skye Bioscience",
    "SKYH": "Sky Harbour Class A",
    "SKYQ": "Sky Quarry",
    "SKYT": "SkyWater Technology",
    "SKYX": "SKYX Platforms",
    "SLDB": "Solid Biosciences",
    "SLDE": "Slide Insurance",
    "SLDP": "Solid Power Class A",
    "SLE": "Super League Enterprise",
    "SLF": "Sun Life Financial",
    "SLGB": "Smart Logistics Global",
    "SLGL": "Sol-Gel",
    "SLGN": "Silgan",
    "SLM": "SLM",
    "SLMT": "Brera",
    "SLN": "Silence Plc American Deposit",
    "SLNG": "Stabilis Solutions",
    "SLNH": "Soluna",
    "SLNO": "Soleno",
    "SLP": "Simulations Plus",
    "SLQT": "SelectQuote",
    "SLS": "SELLAS Life Sciences",
    "SLSN": "Solesence",
    "SLVM": "Sylvamo",
    "SLXN": "Silexion Corp",
    "SMA": "SmartStop Self Storage REIT",
    "SMBC": "Southern Missouri Bancorp",
    "SMBK": "SmartFinancial",
    "SMC": "Summit Midstream",
    "SMG": "Scotts Miracle-Gro (The)",
    "SMHI": "SEACOR Marine",
    "SMID": "Smith-Midland",
    "SMMT": "Summit",
    "SMP": "Standard Motor Products",
    "SMPL": "The Simply Good Foods",
    "SMR": "NuScale Power Class A",
    "SMRT": "SmartRent, Class A",
    "SMSI": "Smith Micro Software",
    "SMTC": "Semtech",
    "SMTI": "Sanara MedTech",
    "SMTK": "SmartKem",
    "SMWB": "Similarweb",
    "SMX": "SMX (Security Matters) Publi",
    "SMXT": "Solarmax Technology",
    "SN": "SharkNinja",
    "SNA": "Snap-On",
    "SNAL": "Snail Class A",
    "SNBR": "Sleep Number",
    "SNCY": "Sun Country Airlines",
    "SND": "Smart Sand",
    "SNDA": "Sonida Senior Living",
    "SNDK": "Sandisk",
    "SNDL": "SNDL",
    "SNDR": "Schneider National",
    "SNDX": "Syndax",
    "SNES": "SenesTech",
    "SNEX": "StoneX",
    "SNFCA": "Security National Financial",
    "SNGX": "Soligenix",
    "SNN": "Smith & Nephew SNATS",
    "SNOA": "Sonoma",
    "SNSE": "Sensei Biotherapeutics",
    "SNT": "Senstar",
    "SNTG": "Sentage",
    "SNTI": "Senti Biosciences",
    "SNWV": "SANUWAVE Health",
    "SNX": "TD SYNNEX",
    "SNY": "Sanofi ADS",
    "SNYR": "Synergy CHC",
    "SOBO": "South Bow",
    "SOBR": "SOBR Safe",
    "SOC": "Sable Offshore",
    "SOGP": "Sound",
    "SOHU": "Sohu.com",
    "SOLS": "Solstice Advanced Materials",
    "SOLV": "Solventum",
    "SOMN": "Southern (The) 2025 Series A",
    "SON": "Sonoco Products",
    "SONM": "DNA X",
    "SOPA": "Society Pass",
    "SOPH": "SOPHiA GENETICS SA",
    "SOR": "Source Capital, Cmn Shs of B",
    "SORA": "AsiaStrategy",
    "SOS": "SOS",
    "SOTK": "Sono-Tek",
    "SOWG": "Sow Good",
    "SPAI": "Safe Pro",
    "SPCB": "SuperCom (Israel)",
    "SPCE": "Virgin Galactic",
    "SPE": "Special Opportunities Fund",
    "SPFI": "South Plains Financial",
    "SPH": "Suburban Propane Partners, L",
    "SPHL": "Springview Ltd",
    "SPHR": "Sphere Entertainment Class A",
    "SPIR": "Spire Global, Class A",
    "SPMC": "Sound Point Meridian Capital",
    "SPNT": "SiriusPoint",
    "SPPL": "SIMPPLE LTD.",
    "SPRB": "Spruce Biosciences",
    "SPRC": "SciSparc",
    "SPRO": "Spero",
    "SPRU": "Spruce Power Holding Class A",
    "SPRY": "ARS",
    "SPT": "Sprout Social Inc Class A",
    "SPWH": "Sportsman\'s Warehouse",
    "SPXC": "SPX",
    "SPXX": "Nuveen S&P 500 Dynamic Overw",
    "SQFT": "Presidio Property Trust Clas",
    "SQM": "Sociedad Quimica y Minera",
    "SR": "Spire",
    "SRAD": "Sportradar",
    "SRBK": "SR Bancorp Common stock",
    "SRCE": "1st Source",
    "SRFM": "Surf Air Mobility",
    "SRG": "Seritage Growth Properties C",
    "SRI": "Stoneridge",
    "SRL": "Scully Royalty",
    "SRTA": "Strata Critical Medical Clas",
    "SRTS": "Sensus Healthcare",
    "SRV": "NXG Cushing Midstream Energy",
    "SRZN": "Surrozen",
    "SSB": "SouthState Bank",
    "SSBI": "Summit State Bank",
    "SSD": "Simpson Manufacturing",
    "SSII": "SS Innovations",
    "SSM": "Sono",
    "SSNC": "SS&C",
    "SSP": "E.W. Scripps (The) Class A",
    "SSRM": "SSR Mining",
    "SST": "System1, Class A",
    "SSTI": "SoundThinking",
    "SSTK": "Shutterstock",
    "SSYS": "Stratasys (Israel)",
    "ST": "Sensata Holding",
    "STAA": "STAAR Surgical",
    "STAK": "STAK",
    "STBA": "S&T Bancorp",
    "STC": "Stewart Information Services",
    "STE": "STERIS (Ireland)",
    "STEL": "Stellar Bancorp",
    "STEX": "Streamex",
    "STFS": "Star Fashion Culture",
    "STHO": "Star Shares",
    "STI": "Solidion Technology",
    "STIM": "Neuronetics",
    "STK": "Columbia Seligman Premium Te",
    "STKE": "Sol Strategies",
    "STKH": "Steakholder Foods",
    "STKL": "SunOpta",
    "STKS": "The ONE Hospitality",
    "STLA": "Stellantis",
    "STM": "STMicroelectronics",
    "STN": "Stantec Inc",
    "STNE": "StoneCo Class A",
    "STNG": "Scorpio Tankers",
    "STOK": "Stoke",
    "STRA": "Strategic Education",
    "STRO": "Sutro Biopharma",
    "STRR": "Star Equity",
    "STRS": "Stratus Properties",
    "STRT": "STRATTECCURITY CORPORATION",
    "STRZ": "Starz Entertainment",
    "STSS": "Sharps Technology",
    "STTK": "Shattuck Labs",
    "STUB": "StubHub, Class A",
    "STVN": "Stevanato S.p.A.",
    "STWD": "STARWOOD PROPERTY TRUST, INC",
    "SU": "Suncor Energy",
    "SUGP": "SU",
    "SUI": "Sun Communities",
    "SUIG": "Sui",
    "SUIS": "Canary Staked SUIS ETF Share",
    "SUNB": "Sunbelt Rentals",
    "SUNE": "SUNation Energy",
    "SUNS": "Sunrise Realty Trust",
    "SUPX": "SuperX AI Technology",
    "SURG": "SurgePays",
    "SUUN": "PowerBank",
    "SVC": "Service Properties Trust",
    "SVCC": "Stellar V Capital",
    "SVCO": "Silvaco",
    "SVRA": "Savara",
    "SVRE": "SaverOne 2014",
    "SVRN": "OceanPal",
    "SVV": "Savers Value Village",
    "SW": "Smurfit WestRock",
    "SWAG": "Stran &",
    "SWBI": "Smith & Wesson Brands",
    "SWIM": "Latham",
    "SWKH": "SWK",
    "SWMR": "Swarmer Inc",
    "SWVL": "Swvl Corp Class A",
    "SWX": "Southwest Gas, (DE)",
    "SWZ": "Total Return Securities Fund",
    "SXC": "SunCoke Energy",
    "SXI": "Standex",
    "SXT": "Sensient",
    "SXTC": "China SXT",
    "SXTP": "60 Degrees",
    "SY": "So-Young",
    "SYBT": "Stock Yards Bancorp",
    "SYM": "Symbotic Class A",
    "SYNA": "Synaptics $0.001 Par Value",
    "SYPR": "Sypris Solutions",
    "SYRE": "Spyre",
    "TAC": "TransAlta",
    "TACT": "TransAct",
    "TALO": "Talos Energy",
    "TANH": "Tantech Class A",
    "TAOP": "Taoping",
    "TAOX": "Tao Synergies",
    "TARA": "Protara",
    "TARS": "Tarsus",
    "TATT": "TAT",
    "TAYD": "Taylor Devices",
    "TBBB": "BBB Foods Class A",
    "TBBK": "The Bancorp Inc",
    "TBCH": "Turtle Beach",
    "TBH": "Brag House",
    "TBHC": "The Brand House Collective",
    "TBLA": "Taboola.com",
    "TBLD": "Thornburg Income Builder Opp",
    "TBN": "Tamboran Resources Common st",
    "TBPH": "Theravance Biopharma",
    "TBRG": "TruBridge",
    "TC": "Token Cat",
    "TCBI": "Texas Capital Bancshares",
    "TCBS": "Texas Community Bancshares",
    "TCBX": "Third Coast Bancshares",
    "TCI": "Transcontinental Realty Inve",
    "TCMD": "Tactile Systems Technology",
    "TCOM": "Trip.com",
    "TCRT": "Alaunos",
    "TCRX": "TScan",
    "TCX": "Tucows Class A",
    "TD": "Toronto Dominion Bank (The)",
    "TDAY": "USA TODAY",
    "TDC": "Teradata",
    "TDF": "Templeton Dragon Fund",
    "TDIC": "Dreamland",
    "TDOG": "21Shares Dogecoin ETF",
    "TDS": "Telephone and Data Systems",
    "TDTH": "Trident Digital Tech Ltd",
    "TDUP": "ThredUp Class A",
    "TDW": "Tidewater",
    "TE": "T1 Energy",
    "TEAD": "Teads Holding",
    "TEAM": "Atlassian Class A",
    "TECK": "Teck Resources Ltd",
    "TECX": "Tectonic Therapeutic",
    "TEI": "Templeton Emerging Markets I",
    "TELA": "TELA Bio",
    "TELO": "Telomir",
    "TEM": "Tempus AI Class A",
    "TEN": "Tsakos Energy Navigation Ltd",
    "TENX": "Tenax",
    "TEO": "Telecom Argentina SA",
    "TERN": "Terns",
    "TEX": "Terex",
    "TFII": "TFI",
    "TFIN": "Triumph Financial",
    "TFPM": "Triple Flag Precious Metals",
    "TFSL": "TFS Financial",
    "TFX": "Teleflex",
    "TG": "Tredegar",
    "TGE": "The Generation Essentials",
    "TGHL": "The GrowHub",
    "TGL": "Treasure Global",
    "TGLS": "Tecnoglass",
    "TGS": "Transportadora de Gas del Su",
    "TGT": "Target",
    "TGTX": "TG",
    "TH": "Target Hospitality",
    "THC": "Tenet Healthcare",
    "THCH": "TH Ordinary shares",
    "THFF": "First Financial",
    "THG": "Hanover Insurance Inc",
    "THH": "TryHard",
    "THO": "Thor Industries",
    "THQ": "abrdn Healthcare Opportuniti",
    "THR": "Thermon",
    "THRM": "Gentherm Inc",
    "THRY": "Thryv",
    "THW": "abrdn World Healthcare Fund",
    "TIC": "TIC Solutions",
    "TIGO": "Millicom Cellular",
    "TIL": "Instil Bio",
    "TILE": "Interface",
    "TIPT": "Tiptree",
    "TISI": "Team",
    "TITN": "Titan Machinery",
    "TIVC": "Tivic Health Systems",
    "TJGC": "TJGC",
    "TK": "Teekay",
    "TKC": "Turkcell Iletisim Hizmetleri",
    "TKLF": "Tokyo Lifestyle",
    "TKNO": "Alpha Teknova",
    "TKO": "TKO, Class A",
    "TKR": "Timken (The)",
    "TLF": "Tandy Leather Factory",
    "TLIH": "Ten-League",
    "TLK": "PT Telekomunikasi Indonesia,",
    "TLN": "Talen Energy",
    "TLNC": "Talon Capital",
    "TLPH": "Talphera",
    "TLRY": "Tilray Brands",
    "TLS": "Telos",
    "TLSA": "Tiziana Life Sciences",
    "TLSI": "TriSalus Life Sciences",
    "TLX": "Telix",
    "TM": "Toyota Motor",
    "TMC": "TMC the metals company",
    "TMCI": "Treace Medical Concepts",
    "TNC": "Tennant",
    "TNDM": "Tandem Diabetes Care",
    "TNET": "TriNet",
    "TNGX": "Tango",
    "TNK": "Teekay Tankers",
    "TNL": "Travel   Leisure Common  Sto",
    "TNMG": "TNL Mediagene",
    "TNON": "Tenon Medical",
    "TNXP": "Tonix Holding",
    "TNYA": "Tenaya",
    "TOI": "The Oncology Institute",
    "TOMZ": "TOMI Environmental Solutions",
    "TONX": "TON Strategy",
    "TOP": "TOP Financial",
    "TORO": "Toro",
    "TOUR": "Tuniu",
    "TOYO": "TOYO Ltd",
    "TPB": "Turning Point Brands",
    "TPC": "Tutor Perini",
    "TPCS": "TechPrecision Common stock",
    "TPG": "TPG Class A",
    "TPH": "Tri Pointe Homes",
    "TPL": "Texas Pacific Land",
    "TR": "Tootsie Roll Industries",
    "TRAK": "ReposiTrak",
    "TRC": "Tejon Ranch Co",
    "TRDA": "Entrada",
    "TREE": "LendingTree",
    "TREX": "Trex",
    "TRI": "Thomson Reuters",
    "TRIB": "Trinity Biotech",
    "TRIP": "TripAdvisor",
    "TRMB": "Trimble",
    "TRMD": "TORM Class A",
    "TRMK": "Trustmark",
    "TRNO": "Terreno Realty",
    "TRNR": "Interactive Strength",
    "TRNS": "Transcat",
    "TRON": "Tron",
    "TROO": "TROOPS",
    "TROX": "Tronox (UK)",
    "TRP": "TC Energy",
    "TRS": "TriMas",
    "TRSG": "Tungray Inc",
    "TRST": "TrustCo Bank Corp NY",
    "TRTX": "TPG RE Finance Trust",
    "TRU": "TransUnion",
    "TRUG": "TruGolf Class A",
    "TRUP": "Trupanion",
    "TRVG": "trivago",
    "TRVI": "Trevi",
    "TSAT": "Telesat Class A and Class B",
    "TSBK": "Timberland Bancorp",
    "TSEM": "Tower Semiconductor",
    "TSHA": "Taysha Gene Therapies",
    "TSI": "TCW Strategic Income Fund",
    "TSLX": "Sixth Street Specialty Lendi",
    "TSM": "Taiwan Semiconductor Manufac",
    "TSQ": "Townsquare Media, Class A",
    "TSSI": "TSS",
    "TSUI": "21shares Sui ETF Shares",
    "TTAM": "Titan America SA",
    "TTAN": "ServiceTitan Class A",
    "TTC": "Toro (The)",
    "TTE": "TotalEnergies",
    "TTEC": "TTEC",
    "TTEK": "Tetra Tech",
    "TTGT": "TechTarget",
    "TTI": "Tetra",
    "TTMI": "TTM",
    "TTRX": "Turn",
    "TU": "Telus",
    "TULP": "Bloomia",
    "TURB": "Turbo Energy",
    "TUSK": "Mammoth Energy Services",
    "TV": "Grupo TelevisaB.",
    "TVC": "Tennessee Valley Authority",
    "TVE": "Tennessee Valley Authority",
    "TVGN": "Tevogen Bio",
    "TVRD": "Tvardi",
    "TVTX": "Travere",
    "TW": "Tradeweb Markets Class A",
    "TWAV": "TaoWeave",
    "TWFG": "TWFG Class A",
    "TWG": "Top Wealth Holding",
    "TWI": "Titan, (DE)",
    "TWIN": "Twin Disc",
    "TWLO": "Twilio Class A",
    "TWLV": "Twelve Seas Investment III",
    "TWN": "Taiwan Fund, (The)",
    "TWO": "Two Harbors Investment Corp",
    "TWST": "Twist Bioscience",
    "TXG": "10x Genomics Class A",
    "TXMD": "TherapeuticsMD",
    "TXNM": "TXNM Energy",
    "TY": "Tri Continental",
    "TYG": "Tortoise Energy Infrastructu",
    "TYGO": "Tigo Energy",
    "TYL": "Tyler",
    "TYRA": "Tyra Biosciences",
    "TZOO": "Travelzoo",
    "UA": "Under Armour, Class C",
    "UAA": "Under Armour, Class A",
    "UAMY": "United States Antimony",
    "UBCP": "United Bancorp",
    "UBFO": "United Security Bancshares",
    "UBS": "UBS Registered",
    "UBSI": "United Bankshares",
    "UBXG": "U-BX Technology",
    "UCAR": "U Power",
    "UCB": "United Community Banks",
    "UCL": "uCloudlink",
    "UCTT": "Ultra Clean",
    "UDMY": "Udemy",
    "UE": "Urban Edge Properties",
    "UEIC": "Universal Electronics",
    "UFCS": "United Fire",
    "UFG": "Uni-Fuels",
    "UFI": "Unifi, New",
    "UFPT": "UFP",
    "UG": "United-Guardian",
    "UGI": "UGI",
    "UGRO": "urban-gro",
    "UHAL": "U-Haul Holding",
    "UHG": "United Homes Inc Class A",
    "UHS": "Universal Health Services",
    "UHT": "Universal Health Realty Inco",
    "UI": "Ubiquiti",
    "UIS": "Unisys New",
    "UK": "Ucommune Ltd",
    "ULBI": "Ultralife",
    "ULCC": "Frontier",
    "ULH": "Universal Logistics",
    "ULS": "UL Solutions Class A",
    "ULTA": "Ulta Beauty",
    "UMC": "United Microelectronics (NEW",
    "UMH": "UMH Properties",
    "UNB": "Union Bankshares",
    "UNCY": "Unicycive",
    "UNF": "Unifirst",
    "UNFI": "United Natural Foods",
    "UNIT": "Uniti",
    "UNTY": "Unity Bancorp",
    "UONE": "Urban One Class A",
    "UONEK": "Urban One Class D",
    "UP": "Wheels Up Experience Class A",
    "UPB": "Upstream Bio",
    "UPBD": "Upbound",
    "UPC": "Universe",
    "UPLD": "Upland Software",
    "UPWK": "Upwork",
    "UPXI": "Upexi",
    "URGN": "UroGen Pharma",
    "UROY": "Uranium Royalty",
    "USA": "Liberty All-Star Equity Fund",
    "USAR": "USA Rare Earth Class A",
    "USAU": "U.S. Gold",
    "USCB": "USCB Financial Class A",
    "USEA": "United Maritime",
    "USEG": "U.S. Energy (DE)",
    "USFD": "US Foods Holding",
    "USGO": "U.S. GoldMining Common stock",
    "USIO": "Usio",
    "USLM": "United States Lime & Mineral",
    "USNA": "USANA Health Sciences",
    "USPH": "U.S. Physical Therapy",
    "UTF": "Cohen & Steers Infrastructur",
    "UTHR": "United",
    "UTI": "Universal Technical Institut",
    "UTL": "UNITIL",
    "UTMD": "Utah Medical Products",
    "UTSI": "UTStarcom",
    "UTZ": "Utz Brands Inc Class A",
    "UVE": "UNIVERSAL INSURANCE HOLDINGS",
    "UVSP": "Univest Financial",
    "UVV": "Universal",
    "UWMC": "UWM Class A",
    "UXIN": "Uxin ADS",
    "VABK": "Virginia National Bankshares",
    "VAC": "Marriott Vacations Worldwide",
    "VAL": "Valaris",
    "VALN": "Valneva",
    "VALU": "Value Line",
    "VANI": "Vivani Medical (DE)",
    "VATE": "INNOVATE",
    "VAVX": "VanEck Avalanche ETF",
    "VBF": "Invesco Bond Fund",
    "VBIX": "Viewbix",
    "VBNK": "VersaBank",
    "VC": "Visteon",
    "VCEL": "Vericel",
    "VCIC": "Vine Hill Capital Investment",
    "VCIG": "VCI Global Ordinary Share",
    "VCTR": "Victory Capital Class A",
    "VCV": "Invesco California Value Mun",
    "VCX": "Fundrise Innovation Fund, LL",
    "VCYT": "Veracyte",
    "VEEA": "Veea",
    "VEEE": "Twin Vee PowerCats",
    "VEL": "Velocity Financial",
    "VELO": "Velo3D Common stock",
    "VEON": "VEON ADS",
    "VERI": "Veritone",
    "VERU": "Veru",
    "VERX": "Vertex Class A",
    "VET": "Vermilion Energy Common (Can",
    "VFF": "Village Farms",
    "VFS": "VinFast Auto",
    "VG": "Venture Global, Class A comm",
    "VGAS": "Verde Clean Fuels Class A",
    "VGI": "Virtus Global Multi-Sector I",
    "VGM": "Invesco Trust for Investment",
    "VHC": "VirnetX Holding Corp",
    "VHCP": "Vine Hill Capital Investment",
    "VHI": "Valhi",
    "VHUB": "VenHub Global",
    "VIA": "Via Transportation, Class A",
    "VICR": "Vicor",
    "VIK": "Viking Ltd",
    "VINP": "Vinci Compass Investments Cl",
    "VIOT": "Viomi Technology Ltd",
    "VIR": "Vir Biotechnology",
    "VIRC": "Virco Manufacturing",
    "VIRT": "Virtu Financial, Class A",
    "VISN": "Vistance Networks",
    "VITL": "Vital Farms",
    "VIVO": "VivoPower",
    "VIVS": "VivoSim Labs",
    "VKQ": "Invesco Municipal Trust",
    "VKTX": "Viking",
    "VLGEA": "Village Super Market Class A",
    "VLN": "Valens Semiconductor",
    "VLT": "Invesco High Income Trust II",
    "VLTO": "Veralto Corp",
    "VLY": "Valley National Bancorp",
    "VMAR": "Vision Marine",
    "VMC": "Vulcan Materials (Holding)",
    "VMD": "Viemed Healthcare",
    "VMET": "Versamet Royalties",
    "VMI": "Valmont Industries",
    "VMO": "Invesco Municipal Opportunit",
    "VNCE": "Vince Holding",
    "VNDA": "Vanda",
    "VNET": "VNET",
    "VNO": "Vornado Realty Trust",
    "VNOM": "Viper Energy Class A",
    "VNT": "Vontier",
    "VOD": "Vodafone Plc",
    "VOR": "Vor Biopharma",
    "VOXR": "Vox Royalty",
    "VOYA": "Voya Financial",
    "VOYG": "Voyager, Class A",
    "VPG": "Vishay Precision",
    "VPV": "Invesco Pennsylvania Value M",
    "VRA": "Vera Bradley",
    "VRAX": "Virax Biolabs",
    "VRCA": "Verrica",
    "VRDN": "Viridian",
    "VRE": "Veris Residential",
    "VREX": "Varex Imaging",
    "VRME": "VerifyMe",
    "VRNS": "Varonis Systems",
    "VRRM": "Verra Mobility Class A",
    "VRSK": "Verisk Analytics",
    "VRSN": "VeriSign",
    "VRT": "Vertiv, LLC Class A",
    "VRTS": "Virtus Investment Partners",
    "VS": "Versus Systems",
    "VSA": "VisionSys AI",
    "VSAT": "ViaSat",
    "VSCO": "Victorias Secret &",
    "VSEC": "VSE",
    "VSEE": "VSee Health",
    "VSH": "Vishay Intertechnology",
    "VSME": "VS Media",
    "VSNT": "Versant Media Class A",
    "VST": "Vistra",
    "VSTD": "Vestand Class A",
    "VSTM": "Verastem",
    "VSTS": "Vestis",
    "VTEX": "VTEX Class A",
    "VTGN": "Vistagen",
    "VTIX": "Virtuix Class A",
    "VTN": "Invesco Trust for Investment",
    "VTOL": "Bristow",
    "VTS": "Vitesse Energy",
    "VTSI": "VirTra",
    "VTVT": "vTv Class A",
    "VUZI": "Vuzix",
    "VVOS": "Vivos",
    "VVR": "Invesco Senior Income Trust",
    "VVV": "Valvoline",
    "VVX": "V2X",
    "VWAV": "VisionWave",
    "VYGR": "Voyager",
    "VYNE": "VYNE",
    "VYX": "NCR Voyix",
    "WABC": "Westamerica Bancorporation",
    "WAFDP": "WaFd Depositary Shares",
    "WAFU": "Wah Fu Education",
    "WAI": "Top KingWin Ltd",
    "WAL": "Western Alliance Bancorporat",
    "WALD": "Waldencast Class A Ordinary",
    "WASH": "Washington Trust Bancorp",
    "WATT": "Energous",
    "WAVE": "Eco Wave Power Global AB (pu",
    "WAY": "Waystar Holding",
    "WB": "Weibo American Depositary Sh",
    "WBI": "WaterBridge Infrastructure L",
    "WBTN": "WEBTOON Entertainment Common",
    "WBUY": "WEBUY GLOBAL LTD.",
    "WBX": "Wallbox",
    "WCC": "WESCO",
    "WCN": "Waste Connections",
    "WCT": "Wellchange Class A Ordinary",
    "WD": "Walker & Dunlop",
    "WDAY": "Workday Class A",
    "WDFC": "WD-40",
    "WDI": "Western Asset Diversified In",
    "WEA": "Western Asset Bond Fund Shar",
    "WEAV": "Weave Communications",
    "WEN": "Wendy\'s (The)",
    "WERN": "Werner Enterprises",
    "WEST": "Westrock Coffee",
    "WETH": "Wetouch Technology",
    "WETO": "Wetour Robotics",
    "WEX": "WEX common stock",
    "WEYS": "Weyco",
    "WFCF": "Where Food Comes From",
    "WFF": "WF Holding",
    "WFG": "West Fraser Timber Ltd Commo",
    "WFRD": "Weatherford",
    "WGO": "Winnebago Industries",
    "WGRX": "Wellgistics Health",
    "WGS": "GeneDx Class A",
    "WH": "Wyndham Hotels & Resorts",
    "WHD": "Cactus, Class A",
    "WHG": "Westwood Inc",
    "WHLR": "Wheeler Real Estate Investme",
    "WHR": "Whirlpool",
    "WHWK": "Whitehawk",
    "WIA": "Western Asset Inflation-Link",
    "WILC": "G. Willi-Food",
    "WIMI": "WiMi Hologram Cloud",
    "WINA": "Winmark",
    "WING": "Wingstop",
    "WIT": "Wipro",
    "WIW": "Western Asset Inflation-Link",
    "WIX": "Wix.com",
    "WK": "Workiva Class A",
    "WKC": "World Kinect",
    "WKEY": "WISeKey Holding Ltd",
    "WKSP": "Worksport",
    "WLDS": "Wearable Devices Ordinary Sh",
    "WLFC": "Willis Lease Finance",
    "WLK": "Westlake",
    "WLTH": "Wealthfront",
    "WLY": "John Wiley & Sons",
    "WLYB": "John Wiley & Sons",
    "WMG": "Warner Music Class A",
    "WMK": "Weis Markets",
    "WMS": "Advanced Drainage Systems",
    "WNC": "Wabash National",
    "WNEB": "Western New England Bancorp",
    "WNW": "Meiwu Technology",
    "WOK": "WORK Medical Technology LTD",
    "WOLF": "Wolfspeed, New",
    "WOOF": "Petco Health and Wellness Cl",
    "WOR": "Worthington Enterprises",
    "WORX": "SCWorx",
    "WPRT": "Westport Fuel Systems Inc",
    "WRAP": "Wrap",
    "WRBY": "Warby Parker Class A",
    "WRD": "WeRide",
    "WS": "Worthington Steel",
    "WSBC": "WesBanco",
    "WSBF": "Waterstone Financial (MD)",
    "WSBK": "Winchester Bancorp",
    "WSC": "WillScot Class A",
    "WSHP": "WeShop",
    "WSM": "Williams-Sonoma, (DE)",
    "WSO": "Watsco",
    "WSO-B": "Watsco",
    "WSR": "Whitestone REIT",
    "WST": "West Pharmaceutical Services",
    "WT": "WisdomTree",
    "WTBA": "West Bancorporation",
    "WTF": "Waton Financial",
    "WTI": "W&T Offshore",
    "WTM": "White Mountains Insurance",
    "WTO": "UTime",
    "WTS": "Watts Water, Class A",
    "WTTR": "Select Water Solutions, Clas",
    "WTW": "Willis Towers Watson Public",
    "WU": "Western Union (The)",
    "WULF": "TeraWulf",
    "WVE": "Wave Life Sciences",
    "WVVI": "Willamette Valley Vineyards",
    "WW": "WW",
    "WWD": "Woodward",
    "WWW": "Wolverine World Wide",
    "WXM": "WF",
    "WY": "Weyerhaeuser",
    "WYFI": "WhiteFiber",
    "WYHG": "Wing Yip Food",
    "XAIR": "Beyond Air",
    "XBIO": "Xenetic Biosciences",
    "XBIT": "XBiotech",
    "XBP": "XBP Global",
    "XCH": "XCHG American Depositary Sha",
    "XCUR": "Exicure",
    "XELB": "Xcel Brands",
    "XELLL": "Xcel Energy 6.25% Junior Sub",
    "XERS": "Xeris Biopharma",
    "XFLH": "XFLH Capital",
    "XFLT": "XAI Octagon Floating Rate &",
    "XFOR": "X4",
    "XGN": "Exagen",
    "XHG": "XChange TEC.INC",
    "XHLD": "TEN",
    "XHR": "Xenia Hotels & Resorts",
    "XLO": "Xilio",
    "XMTR": "Xometry Class A",
    "XNCR": "Xencor",
    "XNET": "Xunlei",
    "XOMA": "XOMA Royalty",
    "XOS": "Xos",
    "XP": "XP Class A",
    "XPEL": "XPEL",
    "XPER": "Xperi",
    "XPON": "Expion360",
    "XPRO": "Expro",
    "XRAY": "DENTSPLY SIRONA",
    "XRN": "Chiron Real Estate",
    "XRTX": "XORTX",
    "XRX": "Xerox",
    "XTKG": "X3",
    "XTLB": "XTL Biopharmaceuticals",
    "XWEL": "XWELL",
    "XWIN": "XMAX",
    "XXI": "Twenty One Capital, Class A",
    "XXII": "22nd Century",
    "XYZ": "Block, Class A",
    "XZO": "Exzeo",
    "YAAS": "Youxin Technology Ltd Class",
    "YB": "Yuanbao",
    "YDDL": "One and One Green. Inc",
    "YDES": "YD Bio",
    "YDKG": "Yueda Digital Holding Class",
    "YELP": "Yelp",
    "YETI": "YETI",
    "YEXT": "Yext",
    "YHC": "LQR House",
    "YHGJ": "Yunhong Green CTI",
    "YI": "111",
    "YIBO": "Planet Image",
    "YJ": "Yunji",
    "YMAT": "J-Star Holding",
    "YMT": "Yimutian",
    "YOU": "Clear Secure, Class A",
    "YOUL": "Youlife",
    "YPF": "YPF Sociedad Anonima",
    "YQ": "17 Education & Technology",
    "YSXT": "YSX Tech. Ltd",
    "YTRA": "Yatra Online",
    "YUMC": "Yum China",
    "YXT": "YXT.COM GROUP HOLDING LIMITE",
    "YYAI": "AiRWA",
    "YYGH": "YY Holding",
    "Z": "Zillow",
    "ZBAI": "ATIF",
    "ZBAO": "Zhibao Technology",
    "ZBIO": "Zenas BioPharma",
    "ZCMD": "Zhongchao",
    "ZD": "Ziff Davis",
    "ZDAI": "DirectBooking Technology",
    "ZENA": "ZenaTech",
    "ZEO": "Zeo Energy Class A",
    "ZG": "Zillow Class A",
    "ZGM": "Zenta",
    "ZGN": "Ermenegildo Zegna",
    "ZIM": "ZIM Integrated Shipping Serv",
    "ZIP": "ZipRecruiter, Class A",
    "ZJK": "ZJK Industrial",
    "ZJYL": "JIN MEDICAL INTERNATIONAL LT",
    "ZKIN": "ZK Ltd Ordinary Share",
    "ZLAB": "Zai Lab",
    "ZM": "Zoom Communications Class A",
    "ZNB": "Zeta Network",
    "ZNTL": "Zentalis",
    "ZOOZ": "ZOOZ Strategy",
    "ZSTK": "ZeroStack",
    "ZTEK": "Zentek",
    "ZTR": "Virtus Total Return Fund",
    "ZURA": "Zura Bio",
    "ZVIA": "Zevia PBC Class A",
    "ZVRA": "Zevra",
    "ZWS": "Zurn Elkay Water Solutions",
    "ZYBT": "Zhengye Biotechnology Holdin",
    "ZYME": "Zymeworks",
    "VEEV": "Veeva Systems", "AMWL": "American Well", "TDOC": "Teladoc",
}


def get_stock_list():
    """Download stock data in chunks; return sorted by pct_change desc."""
    tickers = STOCK_UNIVERSE
    result = []
    chunk_size = 50

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
        time.sleep(0.2)  # avoid rate limiting across chunks

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
