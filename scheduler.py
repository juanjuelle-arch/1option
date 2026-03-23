"""
scheduler.py — APScheduler background jobs to refresh data cache every 5 minutes.
"""

from apscheduler.schedulers.background import BackgroundScheduler
import cache
import data_fetcher
import logging

logger = logging.getLogger(__name__)

def refresh_main():
    """Refresh dashboard data — picks, sentiment, news, options. Runs every 2 min."""
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

        picks = data_fetcher.get_top_picks(earnings)
        cache.set("picks", picks)

        top_calls, top_puts = data_fetcher.get_global_top_options()
        cache.set("top_calls", top_calls)
        cache.set("top_puts", top_puts)

        logger.info("Main cache refresh complete.")
    except Exception as e:
        logger.error(f"Main refresh error: {e}")


def refresh_stocks():
    """Refresh full stock browser list. Runs every 15 min to avoid rate limits."""
    logger.info("Refreshing stock list cache...")
    try:
        stock_list = data_fetcher.get_stock_list()
        cache.set("stock_list", stock_list)
        logger.info(f"Stock list cache refreshed: {len(stock_list)} tickers.")
    except Exception as e:
        logger.error(f"Stock list refresh error: {e}")


def start_scheduler():
    """Start background scheduler."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(refresh_main, "interval", minutes=2, id="refresh_main")
    scheduler.add_job(refresh_stocks, "interval", minutes=15, id="refresh_stocks")
    # Run stock list once at startup
    scheduler.add_job(refresh_stocks, "date", id="refresh_stocks_startup")
    scheduler.start()
    logger.info("Scheduler started — main every 2 min, stocks every 15 min.")
    return scheduler
