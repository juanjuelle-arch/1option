"""
worker.py — Standalone scheduler process for data fetching.
Runs separately from web workers so data fetching doesn't block HTTP requests.
Shares Redis cache with web workers.
"""

import os
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a minimal Flask app for database context
from flask import Flask
from dotenv import load_dotenv
from models import db

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "worker-key")

_db_url = os.environ.get("DATABASE_URL", "sqlite:///1option.db")
if _db_url.startswith("postgres://"):
    _db_url = _db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = _db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

if "postgresql" in _db_url:
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_size": 3,
        "pool_recycle": 300,
        "pool_pre_ping": True,
        "max_overflow": 5,
    }

db.init_app(app)

with app.app_context():
    db.create_all()

# Import and start scheduler
from scheduler import start_scheduler, refresh_main

logger.info("Worker process starting — running initial data load...")

with app.app_context():
    refresh_main()

logger.info("Initial load complete. Starting scheduler...")
scheduler = start_scheduler()

# Keep the process alive
try:
    while True:
        time.sleep(60)
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown()
    logger.info("Worker process stopped.")
