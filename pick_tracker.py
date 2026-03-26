"""
pick_tracker.py — Track pick performance over time.
Snapshots picks when they enter Top 10, updates prices, calculates returns.
"""

import logging
from datetime import datetime, date, timezone
import yfinance as yf

logger = logging.getLogger(__name__)


def snapshot_picks(picks, db, PickSnapshot):
    """
    Called after each GARP refresh. Saves new picks and closes old ones.
    - New pick not in DB for today → create snapshot with entry price
    - Pick already in DB for today → skip (don't duplicate)
    - Pick was active but no longer in top 10 → mark as closed
    """
    if not picks:
        return

    today = date.today()
    current_tickers = set()

    for i, pick in enumerate(picks[:10]):
        ticker = pick.get("ticker", "")
        price = pick.get("price", 0)
        if not ticker or not price:
            continue

        current_tickers.add(ticker)

        # Check if already snapshotted today
        existing = PickSnapshot.query.filter_by(
            ticker=ticker, picked_date=today
        ).first()

        if existing:
            # Update current price
            existing.current_price = price
            if price > (existing.peak_price or 0):
                existing.peak_price = price
                existing.peak_return = round((price - existing.entry_price) / existing.entry_price * 100, 2)
            existing.pct_return = round((price - existing.entry_price) / existing.entry_price * 100, 2)
            continue

        # Check if this ticker was picked recently (within last 7 days) and still active
        recent = PickSnapshot.query.filter_by(
            ticker=ticker, is_active=True
        ).order_by(PickSnapshot.picked_date.desc()).first()

        if recent:
            # Update the existing active pick instead of creating new one
            recent.current_price = price
            recent.rank = i + 1
            recent.score = pick.get("score", recent.score)
            if price > (recent.peak_price or 0):
                recent.peak_price = price
                recent.peak_return = round((price - recent.entry_price) / recent.entry_price * 100, 2)
            recent.pct_return = round((price - recent.entry_price) / recent.entry_price * 100, 2)
            continue

        # New pick — create snapshot
        snapshot = PickSnapshot(
            ticker=ticker,
            company=pick.get("company", ticker),
            sector=pick.get("sector", ""),
            entry_price=price,
            current_price=price,
            score=pick.get("score", 0),
            conviction=pick.get("conviction", ""),
            rank=i + 1,
            rev_growth=pick.get("rev_growth"),
            earn_growth=pick.get("earn_growth"),
            pe_fwd=pick.get("pe_fwd"),
            target_upside=pick.get("target_upside"),
            pct_return=0.0,
            peak_price=price,
            peak_return=0.0,
            picked_date=today,
            is_active=True,
        )
        db.session.add(snapshot)
        logger.info(f"New pick snapshot: {ticker} @ ${price}")

    # Close picks that are no longer in top 10
    active_picks = PickSnapshot.query.filter_by(is_active=True).all()
    for ap in active_picks:
        if ap.ticker not in current_tickers:
            ap.is_active = False
            ap.closed_at = datetime.now(timezone.utc)
            ap.closed_price = ap.current_price
            ap.closed_return = ap.pct_return
            logger.info(f"Pick closed: {ap.ticker} — return: {ap.pct_return}%")

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Pick snapshot commit error: {e}")


def update_pick_prices(db, PickSnapshot):
    """
    Update current prices for all active picks + recent closed picks (last 30 days).
    Called periodically by scheduler.
    """
    from datetime import timedelta

    cutoff = date.today() - timedelta(days=30)
    picks = PickSnapshot.query.filter(
        (PickSnapshot.is_active == True) |
        (PickSnapshot.picked_date >= cutoff)
    ).all()

    if not picks:
        return

    tickers = list(set(p.ticker for p in picks))
    logger.info(f"Updating prices for {len(tickers)} tracked picks...")

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
                pick.pct_return = round((price - pick.entry_price) / pick.entry_price * 100, 2)

                if price > (pick.peak_price or 0):
                    pick.peak_price = round(price, 2)
                    pick.peak_return = round((price - pick.entry_price) / pick.entry_price * 100, 2)

            except Exception as e:
                logger.debug(f"Price update {pick.ticker}: {e}")

        db.session.commit()
        logger.info(f"Pick prices updated for {len(tickers)} tickers")

    except Exception as e:
        db.session.rollback()
        logger.error(f"Price update error: {e}")


def get_performance_stats(PickSnapshot):
    """
    Calculate overall track record stats for the performance page.
    """
    from datetime import timedelta
    from sqlalchemy import func

    all_picks = PickSnapshot.query.order_by(PickSnapshot.picked_date.desc()).all()
    if not all_picks:
        return {
            "total_picks": 0,
            "winners": 0,
            "losers": 0,
            "win_rate": 0,
            "avg_return": 0,
            "best_pick": None,
            "worst_pick": None,
            "active_picks": [],
            "closed_picks": [],
            "total_return": 0,
        }

    winners = [p for p in all_picks if p.pct_return > 0]
    losers = [p for p in all_picks if p.pct_return <= 0]

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
        "total_picks": total,
        "winners": win_count,
        "losers": total - win_count,
        "win_rate": round(win_count / total * 100, 1) if total else 0,
        "avg_return": round(avg_return, 2),
        "best_pick": best,
        "worst_pick": worst,
        "active_picks": active,
        "closed_picks": closed[:50],  # Last 50 closed picks
        "total_return": round(sum(p.pct_return for p in unique_list), 2),
    }
