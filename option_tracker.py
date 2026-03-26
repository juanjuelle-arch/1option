"""
option_tracker.py — Track options (calls/puts) performance over time.
Snapshots options when they enter Top 8, updates prices, calculates returns.
"""

import logging
from datetime import datetime, date, timezone, timedelta
import yfinance as yf

logger = logging.getLogger(__name__)


def snapshot_options(calls, puts, db, OptionSnapshot):
    """
    Called after each options refresh. Saves new options and closes old ones.
    - New option not in DB → create snapshot with entry price
    - Option already tracked → update current price
    - Option was active but no longer in top lists → mark as closed
    """
    if not calls and not puts:
        return

    today = date.today()
    current_keys = set()

    # Process both calls and puts
    all_options = []
    for opt in (calls or []):
        all_options.append({**opt, "option_type": "CALL"})
    for opt in (puts or []):
        all_options.append({**opt, "option_type": "PUT"})

    for opt in all_options:
        ticker = opt.get("ticker", "")
        strike = opt.get("strike", 0)
        expiry = opt.get("expiry", "")
        opt_type = opt.get("option_type", "")
        price = opt.get("last_price", 0)

        if not ticker or not price or not strike:
            continue

        # Unique key for this contract
        key = f"{ticker}_{strike}_{expiry}_{opt_type}"
        current_keys.add(key)

        # Check if already tracked (active)
        existing = OptionSnapshot.query.filter_by(
            ticker=ticker, strike=strike, expiry=expiry,
            option_type=opt_type, is_active=True
        ).first()

        if existing:
            # Update current price
            existing.current_price = price
            if existing.entry_price and existing.entry_price > 0:
                existing.pct_return = round((price - existing.entry_price) / existing.entry_price * 100, 2)
                if price > (existing.peak_price or 0):
                    existing.peak_price = price
                    existing.peak_return = round((price - existing.entry_price) / existing.entry_price * 100, 2)
            continue

        # New option — create snapshot
        snapshot = OptionSnapshot(
            ticker=ticker,
            option_type=opt_type,
            strike=strike,
            expiry=expiry,
            entry_price=price,
            current_price=price,
            volume=opt.get("volume"),
            open_interest=opt.get("open_interest"),
            iv=opt.get("iv"),
            pct_return=0.0,
            peak_price=price,
            peak_return=0.0,
            picked_date=today,
            is_active=True,
        )
        db.session.add(snapshot)
        logger.info(f"New option snapshot: {ticker} ${strike} {opt_type} @ ${price}")

    # Close options that are no longer in top lists
    active_options = OptionSnapshot.query.filter_by(is_active=True).all()
    for ao in active_options:
        key = f"{ao.ticker}_{ao.strike}_{ao.expiry}_{ao.option_type}"
        if key not in current_keys:
            ao.is_active = False
            ao.closed_at = datetime.now(timezone.utc)
            ao.closed_price = ao.current_price
            ao.closed_return = ao.pct_return
            logger.info(f"Option closed: {ao.ticker} ${ao.strike} {ao.option_type} — return: {ao.pct_return}%")

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Option snapshot commit error: {e}")


def update_option_prices(db, OptionSnapshot):
    """
    Update current prices for all active options using yfinance options chain.
    Called periodically by scheduler.
    """
    cutoff = date.today() - timedelta(days=14)
    options = OptionSnapshot.query.filter(
        (OptionSnapshot.is_active == True) |
        (OptionSnapshot.picked_date >= cutoff)
    ).all()

    if not options:
        return

    # Group by ticker to batch options chain lookups
    by_ticker = {}
    for opt in options:
        by_ticker.setdefault(opt.ticker, []).append(opt)

    logger.info(f"Updating prices for {len(options)} tracked options across {len(by_ticker)} tickers...")

    for ticker, opts in by_ticker.items():
        try:
            stock = yf.Ticker(ticker)
            # Get all available expiry dates
            expiry_dates = stock.options if hasattr(stock, 'options') else []

            for opt in opts:
                try:
                    if opt.expiry not in expiry_dates:
                        # Expiry passed or not available — close it
                        if opt.is_active:
                            opt.is_active = False
                            opt.closed_at = datetime.now(timezone.utc)
                            opt.closed_price = opt.current_price
                            opt.closed_return = opt.pct_return
                        continue

                    chain = stock.option_chain(opt.expiry)
                    if opt.option_type == "CALL":
                        df = chain.calls
                    else:
                        df = chain.puts

                    # Find the matching strike
                    row = df[df["strike"] == opt.strike]
                    if row.empty:
                        continue

                    price = float(row.iloc[0]["lastPrice"])
                    if price <= 0:
                        continue

                    opt.current_price = round(price, 2)
                    if opt.entry_price and opt.entry_price > 0:
                        opt.pct_return = round((price - opt.entry_price) / opt.entry_price * 100, 2)
                        if price > (opt.peak_price or 0):
                            opt.peak_price = round(price, 2)
                            opt.peak_return = round((price - opt.entry_price) / opt.entry_price * 100, 2)

                    # Also update stock price
                    try:
                        hist = stock.history(period="1d")
                        if not hist.empty:
                            opt.stock_current_price = round(float(hist["Close"].iloc[-1]), 2)
                    except Exception:
                        pass

                except Exception as e:
                    logger.debug(f"Option price update {opt.ticker} ${opt.strike}: {e}")

        except Exception as e:
            logger.debug(f"Options chain lookup {ticker}: {e}")

    try:
        db.session.commit()
        logger.info(f"Option prices updated for {len(options)} contracts")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Option price update error: {e}")


def get_options_performance_stats(OptionSnapshot):
    """
    Calculate overall options track record stats for the performance page.
    """
    all_options = OptionSnapshot.query.order_by(OptionSnapshot.picked_date.desc()).all()
    if not all_options:
        return {
            "total_options": 0,
            "total_calls": 0,
            "total_puts": 0,
            "winners": 0,
            "losers": 0,
            "win_rate": 0,
            "avg_return": 0,
            "best_option": None,
            "worst_option": None,
            "active_options": [],
            "closed_options": [],
        }

    # Deduplicate by contract key
    unique = {}
    for o in all_options:
        key = f"{o.ticker}_{o.strike}_{o.expiry}_{o.option_type}_{o.picked_date}"
        if key not in unique:
            unique[key] = o

    unique_list = list(unique.values())
    total = len(unique_list)
    calls_count = sum(1 for o in unique_list if o.option_type == "CALL")
    puts_count = sum(1 for o in unique_list if o.option_type == "PUT")
    win_count = sum(1 for o in unique_list if o.pct_return > 0)
    avg_return = sum(o.pct_return for o in unique_list) / total if total else 0

    best = max(unique_list, key=lambda o: o.pct_return) if unique_list else None
    worst = min(unique_list, key=lambda o: o.pct_return) if unique_list else None

    active = [o for o in all_options if o.is_active]
    closed = [o for o in all_options if not o.is_active]
    closed.sort(key=lambda o: o.picked_date, reverse=True)

    return {
        "total_options": total,
        "total_calls": calls_count,
        "total_puts": puts_count,
        "winners": win_count,
        "losers": total - win_count,
        "win_rate": round(win_count / total * 100, 1) if total else 0,
        "avg_return": round(avg_return, 2),
        "best_option": best,
        "worst_option": worst,
        "active_options": active,
        "closed_options": closed[:50],
    }
