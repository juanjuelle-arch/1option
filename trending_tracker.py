"""
trending_tracker.py — Track trending watchlist performance over time.
Snapshots trending picks when they appear, updates prices, calculates returns.
"""

import logging
from datetime import datetime, date, timezone
import yfinance as yf

logger = logging.getLogger(__name__)


def snapshot_trending(trending_list, db, TrendingSnapshot):
    """
    Called after each trending refresh. Saves new trending tickers and closes old ones.
    - New trending ticker not in DB for today -> create snapshot with entry price
    - Trending ticker already in DB for today -> skip (don't duplicate)
    - Trending ticker was active but no longer in list -> mark as closed
    """
    if not trending_list:
        return

    today = date.today()
    current_tickers = set()

    for item in trending_list:
        ticker = item.get("ticker", "")
        price = item.get("price", 0)
        if not ticker:
            continue

        current_tickers.add(ticker)

        # Check if already snapshotted today
        existing = TrendingSnapshot.query.filter_by(
            ticker=ticker, picked_date=today
        ).first()

        if existing:
            # Update current price
            existing.current_price = price
            existing.conviction_score = item.get("conviction_score", existing.conviction_score)
            existing.conviction_label = item.get("conviction_label", existing.conviction_label)
            if existing.price and existing.price > 0 and price:
                if price > (existing.peak_price or 0):
                    existing.peak_price = price
                    existing.peak_return = round((price - existing.price) / existing.price * 100, 2)
                existing.pct_return = round((price - existing.price) / existing.price * 100, 2)
            continue

        # Check if this ticker was recently trending and still active
        recent = TrendingSnapshot.query.filter_by(
            ticker=ticker, is_active=True
        ).order_by(TrendingSnapshot.picked_date.desc()).first()

        if recent:
            # Update the existing active trending pick instead of creating new one
            recent.current_price = price
            recent.conviction_score = item.get("conviction_score", recent.conviction_score)
            recent.conviction_label = item.get("conviction_label", recent.conviction_label)
            recent.source_count = item.get("source_count", recent.source_count)
            sources = item.get("sources", [])
            if sources:
                recent.sources = ", ".join(sources) if isinstance(sources, list) else sources
            if price and recent.price and recent.price > 0:
                if price > (recent.peak_price or 0):
                    recent.peak_price = price
                    recent.peak_return = round((price - recent.price) / recent.price * 100, 2)
                recent.pct_return = round((price - recent.price) / recent.price * 100, 2)
            continue

        # New trending pick — create snapshot
        sources = item.get("sources", [])
        sources_str = ", ".join(sources) if isinstance(sources, list) else (sources or "")

        snapshot = TrendingSnapshot(
            ticker=ticker,
            price=price if price else None,
            current_price=price if price else None,
            change_pct=item.get("change_pct"),
            conviction_score=item.get("conviction_score", 0),
            conviction_label=item.get("conviction_label", "LOW"),
            source_count=item.get("source_count"),
            sources=sources_str if sources_str else None,
            pct_return=0.0,
            peak_price=price if price else None,
            peak_return=0.0,
            picked_date=today,
            is_active=True,
        )
        db.session.add(snapshot)
        logger.info(f"New trending snapshot: {ticker} @ ${price}")

    # Close trending picks that are no longer in the list
    active_trending = TrendingSnapshot.query.filter_by(is_active=True).all()
    for at in active_trending:
        if at.ticker not in current_tickers:
            at.is_active = False
            at.closed_at = datetime.now(timezone.utc)
            at.closed_price = at.current_price
            at.closed_return = at.pct_return
            logger.info(f"Trending closed: {at.ticker} — return: {at.pct_return}%")

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Trending snapshot commit error: {e}")


def update_trending_prices(db, TrendingSnapshot):
    """
    Update current prices for all active trending picks + recent closed ones (last 30 days).
    Called periodically by scheduler.
    """
    from datetime import timedelta

    cutoff = date.today() - timedelta(days=30)
    picks = TrendingSnapshot.query.filter(
        (TrendingSnapshot.is_active == True) |
        (TrendingSnapshot.picked_date >= cutoff)
    ).all()

    if not picks:
        return

    tickers = list(set(p.ticker for p in picks))
    logger.info(f"Updating prices for {len(tickers)} tracked trending picks...")

    try:
        data = yf.download(tickers, period="1d", interval="1d",
                           auto_adjust=True, progress=False)
        if data.empty:
            return

        for pick in picks:
            try:
                close_col = data["Close"]
                if hasattr(close_col, 'columns'):
                    if pick.ticker in close_col.columns:
                        price = float(close_col[pick.ticker].dropna().iloc[-1])
                    else:
                        continue
                else:
                    price = float(close_col.dropna().iloc[-1])

                pick.current_price = round(price, 2)
                entry_price = pick.price or 0
                if entry_price > 0:
                    pick.pct_return = round((price - entry_price) / entry_price * 100, 2)

                    if price > (pick.peak_price or 0):
                        pick.peak_price = round(price, 2)
                        pick.peak_return = round((price - entry_price) / entry_price * 100, 2)

            except Exception as e:
                logger.debug(f"Trending price update {pick.ticker}: {e}")

        db.session.commit()
        logger.info(f"Trending prices updated for {len(tickers)} tickers")

    except Exception as e:
        db.session.rollback()
        logger.error(f"Trending price update error: {e}")


def get_trending_performance_stats(TrendingSnapshot):
    """
    Calculate overall track record stats for trending picks.
    """
    all_picks = TrendingSnapshot.query.order_by(TrendingSnapshot.picked_date.desc()).all()
    if not all_picks:
        return {
            "total_tracked": 0,
            "winners": 0,
            "losers": 0,
            "win_rate": 0,
            "avg_return": 0,
            "best_trending": None,
            "worst_trending": None,
            "active_trending": [],
            "closed_trending": [],
        }

    # Unique picks (by ticker + date, deduped)
    unique_picks = {}
    for p in all_picks:
        key = f"{p.ticker}_{p.picked_date}"
        if key not in unique_picks:
            unique_picks[key] = p

    unique_list = list(unique_picks.values())
    total = len(unique_list)
    win_count = sum(1 for p in unique_list if p.pct_return > 0)

    avg_return = sum(p.pct_return for p in unique_list) / total if total else 0

    best = max(unique_list, key=lambda p: p.pct_return) if unique_list else None
    worst = min(unique_list, key=lambda p: p.pct_return) if unique_list else None

    active = [p for p in all_picks if p.is_active]
    closed = [p for p in all_picks if not p.is_active]

    # Sort closed by date desc
    closed.sort(key=lambda p: p.picked_date, reverse=True)

    return {
        "total_tracked": total,
        "winners": win_count,
        "losers": total - win_count,
        "win_rate": round(win_count / total * 100, 1) if total else 0,
        "avg_return": round(avg_return, 2),
        "best_trending": best,
        "worst_trending": worst,
        "active_trending": active,
        "closed_trending": closed[:50],  # Last 50 closed trending picks
    }
