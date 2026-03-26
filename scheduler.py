"""
scheduler.py — APScheduler background jobs with staggered startup.
"""

from apscheduler.schedulers.background import BackgroundScheduler
import cache
import data_fetcher
import logging

try:
    from trending import get_trending_watchlist
except ImportError:
    get_trending_watchlist = None

logger = logging.getLogger(__name__)


def refresh_main():
    """Refresh core dashboard data — market, sentiment, news, earnings. Runs every 2 min."""
    logger.info("Refreshing main cache...")
    try:
        market = data_fetcher.get_market_overview()
        cache.set("market_overview", market)

        sentiment = data_fetcher.get_market_sentiment()
        cache.set("sentiment", sentiment)

        news = data_fetcher.get_news_feed()
        cache.set("news", news)

        earnings = data_fetcher.get_earnings_calendar()
        cache.set("earnings", earnings)

        logger.info("Main cache refresh complete.")
    except Exception as e:
        logger.error(f"Main refresh error: {e}")


def refresh_picks():
    """Refresh Top 10 Picks (Wall Street GARP engine + all sources). Runs every 5 min."""
    logger.info("Refreshing picks with GARP engine...")
    try:
        earnings = cache.get("earnings") or []
        picks = data_fetcher.get_top_picks(earnings)
        cache.set("picks", picks)
        logger.info(f"Picks cached: {len(picks)} picks")

        # Snapshot picks for performance tracking
        _snapshot_picks(picks)

    except Exception as e:
        logger.error(f"Picks refresh error: {e}")


def _snapshot_picks(picks):
    """Save pick snapshots to database for performance tracking."""
    try:
        from flask import current_app
        from models import db, PickSnapshot
        from pick_tracker import snapshot_picks

        # Only run if we have a Flask app context
        if current_app:
            snapshot_picks(picks, db, PickSnapshot)
    except RuntimeError:
        # No app context (running outside Flask) — try creating one
        try:
            from app import app
            from models import db, PickSnapshot
            from pick_tracker import snapshot_picks
            with app.app_context():
                snapshot_picks(picks, db, PickSnapshot)
        except Exception as e:
            logger.debug(f"Pick snapshot skipped (no app context): {e}")
    except Exception as e:
        logger.warning(f"Pick snapshot error: {e}")


def refresh_pick_prices():
    """Update current prices for all tracked picks. Runs every 15 min."""
    logger.info("Updating tracked pick prices...")
    try:
        from flask import current_app
        from models import db, PickSnapshot
        from pick_tracker import update_pick_prices

        if current_app:
            update_pick_prices(db, PickSnapshot)
    except RuntimeError:
        try:
            from app import app
            from models import db, PickSnapshot
            from pick_tracker import update_pick_prices
            with app.app_context():
                update_pick_prices(db, PickSnapshot)
        except Exception as e:
            logger.debug(f"Pick price update skipped: {e}")
    except Exception as e:
        logger.warning(f"Pick price update error: {e}")


def refresh_options():
    """Refresh global top calls/puts (full intel from all 13 sources). Runs every 5 min."""
    logger.info("Refreshing options with full intel...")
    try:
        top_calls, top_puts = data_fetcher.get_global_top_options()
        cache.set("top_calls", top_calls)
        cache.set("top_puts", top_puts)
        logger.info(f"Options cached: {len(top_calls)} calls, {len(top_puts)} puts")
    except Exception as e:
        logger.error(f"Options refresh failed: {e}")


def refresh_trending():
    """Refresh trending watchlist (all sources enriched). Runs every 5 min."""
    logger.info("Refreshing trending watchlist...")
    try:
        if get_trending_watchlist:
            trending = get_trending_watchlist()
            cache.set("trending", trending)
            logger.info(f"Trending cached: {len(trending)} tickers")
    except Exception as e:
        logger.error(f"Trending refresh failed: {e}")


def refresh_stocks():
    """Refresh full stock browser list. Runs every 30 min (5300+ tickers)."""
    logger.info("Refreshing stock list cache...")
    try:
        stock_list = data_fetcher.get_stock_list()
        cache.set("stock_list", stock_list)
        logger.info(f"Stock list cache refreshed: {len(stock_list)} tickers.")
    except Exception as e:
        logger.error(f"Stock list refresh error: {e}")


def start_scheduler():
    """Start background scheduler with staggered jobs to avoid Railway timeouts."""
    from datetime import datetime, timedelta

    scheduler = BackgroundScheduler()

    # Core data — lightweight, every 2 min
    scheduler.add_job(refresh_main, "interval", minutes=2, id="refresh_main")

    # Heavy jobs — full 13-source intel, every 5 min (staggered to spread load)
    scheduler.add_job(refresh_picks, "interval", minutes=5, id="refresh_picks")
    scheduler.add_job(refresh_options, "interval", minutes=5, id="refresh_options",
                      next_run_time=None)
    scheduler.add_job(refresh_trending, "interval", minutes=5, id="refresh_trending",
                      next_run_time=None)

    # Pick price updates — every 15 min
    scheduler.add_job(refresh_pick_prices, "interval", minutes=15, id="refresh_pick_prices")

    # Stock browser — every 30 min (5300+ tickers takes time to download)
    scheduler.add_job(refresh_stocks, "interval", minutes=30, id="refresh_stocks")

    # Staggered startup
    now = datetime.now()
    scheduler.add_job(refresh_picks, "date",
                      run_date=now + timedelta(seconds=5),
                      id="refresh_picks_startup")
    scheduler.add_job(refresh_stocks, "date",
                      run_date=now + timedelta(seconds=30),
                      id="refresh_stocks_startup")
    scheduler.add_job(refresh_options, "date",
                      run_date=now + timedelta(minutes=2),
                      id="refresh_options_startup")
    scheduler.add_job(refresh_trending, "date",
                      run_date=now + timedelta(minutes=2, seconds=30),
                      id="refresh_trending_startup")

    scheduler.start()
    logger.info("Scheduler started — main/2min, picks+options+trending/5min, pick-prices/15min, stocks/30min")
    return scheduler
