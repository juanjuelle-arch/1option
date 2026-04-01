"""
app.py — 1.OPTION Flask application
Routes: landing, dashboard, auth, Stripe, API endpoints
Production-ready: PostgreSQL, Redis cache, gzip, health checks
"""

import os
import re
import logging
from datetime import timedelta
from urllib.parse import urlparse
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, abort, send_from_directory, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_compress import Compress
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import stripe
import cache
import data_fetcher
from models import db, User, PickSnapshot, OptionSnapshot, TrendingSnapshot
from pick_tracker import get_performance_stats
from option_tracker import get_options_performance_stats
from trending_tracker import get_trending_performance_stats
try:
    from market_scraper import get_enriched_ticker_profile, get_cboe_pc_ratio
except ImportError:
    get_enriched_ticker_profile = None
    get_cboe_pc_ratio = None
try:
    from trending import get_trending_watchlist
except ImportError:
    get_trending_watchlist = None

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IS_PRODUCTION = bool(os.environ.get("RAILWAY_ENVIRONMENT"))
IS_WORKER = os.environ.get("DYNO_TYPE") == "worker" or os.environ.get("IS_WORKER") == "true"

# ─── App Setup ───────────────────────────────────────────────────────────────

app = Flask(__name__)

# Gzip compression — 60-80% bandwidth savings
Compress(app)

# Secret key
_secret = os.environ.get("SECRET_KEY", "")
if IS_PRODUCTION and not _secret:
    raise RuntimeError("SECRET_KEY environment variable must be set in production!")
app.secret_key = _secret or os.urandom(32).hex()

# ─── Database Config (PostgreSQL in production, SQLite locally) ──────────────

_db_url = os.environ.get("DATABASE_URL", "sqlite:///1option.db")
# Railway PostgreSQL uses postgres:// but SQLAlchemy requires postgresql://
if _db_url.startswith("postgres://"):
    _db_url = _db_url.replace("postgres://", "postgresql://", 1)

# Verify PostgreSQL driver AND connection, fallback to SQLite
if "postgresql" in _db_url:
    try:
        import psycopg2
        # Test actual connection before committing to PostgreSQL
        _test_conn = psycopg2.connect(_db_url.replace("postgresql://", "postgres://", 1),
                                       connect_timeout=5)
        _test_conn.close()
        logger.info("Database: PostgreSQL connected")
    except Exception as e:
        logger.warning(f"Database: PostgreSQL unavailable ({e}), falling back to SQLite")
        _db_url = "sqlite:///1option.db"

app.config["SQLALCHEMY_DATABASE_URI"] = _db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Connection pooling for PostgreSQL
if "postgresql" in _db_url:
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_size": 5,
        "pool_recycle": 300,
        "pool_pre_ping": True,
        "max_overflow": 10,
    }

# Only disable static caching in dev
if not IS_PRODUCTION:
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# ─── Security Config ─────────────────────────────────────────────────────────

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)
if IS_PRODUCTION:
    app.config["SESSION_COOKIE_SECURE"] = True

# CSRF protection
csrf = CSRFProtect(app)

# Rate limiting — always use memory (safest for startup)
_redis_url = os.environ.get("REDIS_URL", "")

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "60 per hour"],
    storage_uri=_redis_url if _redis_url else "memory://",
)

# Ticker validation
TICKER_RE = re.compile(r"^[A-Z0-9]{1,5}([.\-][A-Z]{1,2})?$")
VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "5y"}

# Email validation
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Dummy hash for constant-time login
_DUMMY_HASH = generate_password_hash("dummy_password_for_timing", method='pbkdf2:sha256')

@app.after_request
def add_headers(response):
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' unpkg.com; "
        "style-src 'self' 'unsafe-inline' fonts.googleapis.com; "
        "font-src fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
    if IS_PRODUCTION:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    if request.path.startswith("/static/"):
        if IS_PRODUCTION:
            response.headers["Cache-Control"] = "public, max-age=3600"
        else:
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response

# Force HTTPS in production
@app.before_request
def force_https():
    if IS_PRODUCTION and not request.is_secure and request.headers.get("X-Forwarded-Proto", "http") != "https":
        url = request.url.replace("http://", "https://", 1)
        return redirect(url, code=301)

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.environ.get("STRIPE_PRICE_ID", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access the dashboard."

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ─── Init DB ────────────────────────────────────────────────────────────────

with app.app_context():
    try:
        db.create_all()
        logger.info(f"Database initialized: {app.config['SQLALCHEMY_DATABASE_URI'][:20]}...")
    except Exception as e:
        logger.error(f"Database init failed: {e}")
        logger.warning("App will start but database features may not work")

# ─── Scheduler: only run in web process if no separate worker ────────────────

_RUN_SCHEDULER = os.environ.get("DISABLE_SCHEDULER", "").lower() != "true"

if _RUN_SCHEDULER and not IS_WORKER:
    try:
        import threading
        from scheduler import start_scheduler, refresh_main as refresh_all
        def _initial_load():
            with app.app_context():
                try:
                    refresh_all()
                except Exception as e:
                    logger.error(f"Initial data load failed: {e}")
        threading.Thread(target=_initial_load, daemon=True).start()
        _scheduler = start_scheduler()
        logger.info("Scheduler started in web process")
    except Exception as e:
        logger.error(f"Scheduler failed to start: {e}")

# ─── Health Check ────────────────────────────────────────────────────────────

@app.route("/health")
@csrf.exempt
def health():
    """Health check for Railway monitoring."""
    status = {
        "status": "ok",
        "redis": cache.is_redis_connected(),
        "database": "postgresql" if "postgresql" in str(app.config["SQLALCHEMY_DATABASE_URI"]) else "sqlite",
    }
    try:
        db.session.execute(db.text("SELECT 1"))
        status["db_connected"] = True
    except Exception:
        status["db_connected"] = False
        status["status"] = "degraded"
    return jsonify(status), 200 if status["status"] == "ok" else 503

# ─── Context Processor ───────────────────────────────────────────────────────

@app.context_processor
def inject_market():
    """Inject live market bar into every template."""
    return {"market_overview": cache.get("market_overview") or []}

# ─── Public Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    try:
        picks = cache.get("picks") or []
        news = cache.get("news") or []
        sentiment = cache.get("sentiment") or None
        return render_template("index.html",
                               picks=picks[:3],
                               news=news[:3],
                               sentiment=sentiment,
                               updated_at=cache.get_updated_at("picks"))
    except Exception as e:
        logger.error(f"Index route error: {e}", exc_info=True)
        return f"<h1>Debug Error</h1><pre>{e}</pre>", 500

@app.route("/pricing")
def pricing():
    return render_template("pricing.html")

# ─── Auth Routes ─────────────────────────────────────────────────────────────

@app.route("/signup", methods=["GET", "POST"])
@limiter.limit("5 per minute")
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not email or not password:
            flash("All fields are required.", "error")
            return render_template("signup.html")
        if len(name) > 100:
            flash("Name is too long (max 100 characters).", "error")
            return render_template("signup.html")
        if not EMAIL_RE.match(email):
            flash("Please enter a valid email address.", "error")
            return render_template("signup.html")
        if len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
            return render_template("signup.html")
        if not re.search(r"[A-Za-z]", password) or not re.search(r"[0-9]", password):
            flash("Password must contain both letters and numbers.", "error")
            return render_template("signup.html")
        if User.query.filter_by(email=email).first():
            flash("An account with that email already exists.", "error")
            return render_template("signup.html")
        user = User(email=email, name=name)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash("Account created! Subscribe to unlock full access.", "success")
        return redirect(url_for("pricing"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            session.permanent = True
            next_page = request.args.get("next")
            if next_page:
                parsed = urlparse(next_page)
                if parsed.netloc or parsed.scheme or next_page.startswith("//"):
                    next_page = None
            return redirect(next_page or url_for("dashboard"))
        if not user:
            check_password_hash(_DUMMY_HASH, password)
        logger.warning(f"Failed login attempt for {email} from {request.remote_addr}")
        flash("Invalid email or password.", "error")
    return render_template("login.html")

@app.route("/logout", methods=["POST", "GET"])
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))

# ─── Paid Dashboard ──────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    if not current_user.is_subscribed:
        flash("Subscribe to access the full dashboard.", "info")
        return redirect(url_for("pricing"))
    picks = cache.get("picks") or []
    news = cache.get("news") or []
    sentiment = cache.get("sentiment") or None
    earnings = cache.get("earnings") or []
    top_calls = cache.get("top_calls") or []
    top_puts = cache.get("top_puts") or []
    trending = cache.get("trending") or []
    return render_template("dashboard.html",
                           picks=picks,
                           news=news,
                           sentiment=sentiment,
                           earnings=earnings,
                           top_calls=top_calls,
                           top_puts=top_puts,
                           trending=trending,
                           updated_at=cache.get_updated_at("picks"))

# ─── Stripe ──────────────────────────────────────────────────────────────────

@app.route("/create-checkout-session", methods=["POST"])
@login_required
def create_checkout_session():
    if not stripe.api_key or not STRIPE_PRICE_ID:
        if os.environ.get("DEMO_MODE", "").lower() == "true":
            current_user.is_subscribed = True
            db.session.commit()
            flash("Demo mode: subscription activated!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Payment system is not configured. Please contact support.", "error")
            return redirect(url_for("pricing"))
    try:
        checkout_session = stripe.checkout.Session.create(
            customer_email=current_user.email,
            payment_method_types=["card"],
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            mode="subscription",
            success_url=request.host_url + "payment-success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=request.host_url + "pricing",
            metadata={"user_id": current_user.id},
        )
        return redirect(checkout_session.url, code=303)
    except Exception as e:
        logger.error(f"Stripe error: {e}")
        flash("Payment setup failed. Please try again.", "error")
        return redirect(url_for("pricing"))

@app.route("/payment-success")
@login_required
def payment_success():
    session_id = request.args.get("session_id")
    if not session_id or not stripe.api_key:
        flash("Payment could not be verified. Please try again.", "error")
        return redirect(url_for("pricing"))
    try:
        checkout = stripe.checkout.Session.retrieve(session_id)
        # Verify session belongs to this user (prevent session_id replay attacks)
        if checkout.metadata and checkout.metadata.get("user_id"):
            if str(checkout.metadata["user_id"]) != str(current_user.id):
                flash("Payment session mismatch. Please try again.", "error")
                return redirect(url_for("pricing"))
        if checkout.payment_status == "paid":
            current_user.is_subscribed = True
            current_user.stripe_customer_id = checkout.customer
            current_user.stripe_subscription_id = checkout.subscription
            db.session.commit()
            flash("Welcome to 1.OPTION Pro! Your dashboard is ready.", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Payment was not completed. Please try again.", "error")
            return redirect(url_for("pricing"))
    except Exception as e:
        logger.error(f"Payment success error: {e}")
        flash("Payment verification failed. Please contact support.", "error")
        return redirect(url_for("pricing"))

@app.route("/webhook", methods=["POST"])
@csrf.exempt
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        event_type = event["type"]

        if event_type == "customer.subscription.deleted":
            customer_id = event["data"]["object"]["customer"]
            user = User.query.filter_by(stripe_customer_id=customer_id).first()
            if user:
                user.is_subscribed = False
                db.session.commit()
                logger.info(f"Subscription cancelled for user {user.email}")

        elif event_type == "invoice.payment_failed":
            customer_id = event["data"]["object"]["customer"]
            user = User.query.filter_by(stripe_customer_id=customer_id).first()
            if user:
                user.is_subscribed = False
                db.session.commit()
                logger.warning(f"Payment failed — subscription revoked for {user.email}")

        elif event_type == "customer.subscription.updated":
            sub = event["data"]["object"]
            customer_id = sub["customer"]
            user = User.query.filter_by(stripe_customer_id=customer_id).first()
            if user:
                is_active = sub["status"] in ("active", "trialing")
                user.is_subscribed = is_active
                db.session.commit()
                logger.info(f"Subscription updated for {user.email}: active={is_active}")

    except Exception as e:
        logger.warning(f"Webhook error from {request.remote_addr}: {e}")
        return jsonify({"error": "Webhook error"}), 400
    return jsonify({"status": "ok"})

# ─── API Endpoints ───────────────────────────────────────────────────────────

@app.route("/api/market")
@limiter.limit("60 per minute")
def api_market():
    return jsonify(cache.get("market_overview") or [])

@app.route("/api/picks")
@login_required
def api_picks():
    if not current_user.is_subscribed:
        return jsonify({"error": "Subscription required"}), 403
    return jsonify({
        "picks": cache.get("picks") or [],
        "updated_at": cache.get_updated_at("picks"),
    })

@app.route("/api/sentiment")
@login_required
def api_sentiment():
    if not current_user.is_subscribed:
        return jsonify({"error": "Subscription required"}), 403
    return jsonify(cache.get("sentiment") or {})

@app.route("/api/options")
@login_required
def api_options_global():
    if not current_user.is_subscribed:
        return jsonify({"error": "Subscription required"}), 403
    try:
        top_calls = cache.get("top_calls") or []
        top_puts = cache.get("top_puts") or []
        market_data = {}
        if get_cboe_pc_ratio:
            market_data = get_cboe_pc_ratio()
        return jsonify({
            "top_calls": top_calls,
            "top_puts": top_puts,
            "market": {
                "vix": market_data.get("vix"),
                "put_call_ratio": market_data.get("total_pc"),
            },
            "updated_at": cache.get_updated_at("picks"),
        })
    except Exception as e:
        logger.error(f"API options: {e}")
        return jsonify({"error": "Failed to fetch options"}), 500

@app.route("/api/options/<ticker>")
@login_required
def api_options_ticker(ticker):
    if not current_user.is_subscribed:
        return jsonify({"error": "Subscription required"}), 403
    ticker = ticker.upper()
    if not TICKER_RE.match(ticker):
        return jsonify({"error": "Invalid ticker"}), 400
    try:
        opts = data_fetcher.get_options_data(ticker)
        if not opts:
            return jsonify({"error": f"No options data for {ticker}"}), 404
        return jsonify({
            "ticker": ticker,
            "total_calls": opts["total_calls"],
            "total_puts": opts["total_puts"],
            "put_call_ratio": opts["cp_ratio"],
            "top_calls": opts["top_calls"],
            "top_puts": opts["top_puts"],
            "unusual_calls": opts.get("unusual_calls", []),
            "unusual_puts": opts.get("unusual_puts", []),
            "nearest_expiry": opts["expiry"],
        })
    except Exception as e:
        logger.error(f"API options {ticker}: {e}")
        return jsonify({"error": "Failed to fetch options"}), 500

@app.route("/api/intel/<ticker>")
@login_required
def api_intel_ticker(ticker):
    if not current_user.is_subscribed:
        return jsonify({"error": "Subscription required"}), 403
    ticker = ticker.upper()
    if not TICKER_RE.match(ticker):
        return jsonify({"error": "Invalid ticker"}), 400
    if not get_enriched_ticker_profile:
        return jsonify({"error": "Intelligence module not available"}), 503
    try:
        profile = get_enriched_ticker_profile(ticker)
        return jsonify(profile)
    except Exception as e:
        logger.error(f"API intel {ticker}: {e}")
        return jsonify({"error": "Failed to fetch intelligence"}), 500

@app.route("/api/market/vix")
def api_vix():
    try:
        if get_cboe_pc_ratio:
            data = get_cboe_pc_ratio()
            return jsonify(data)
        return jsonify({"error": "CBOE module not available"}), 503
    except Exception as e:
        logger.error(f"API VIX: {e}")
        return jsonify({"error": "Failed to fetch VIX"}), 500

@app.route("/api/trending")
@login_required
def api_trending():
    if not current_user.is_subscribed:
        return jsonify({"error": "Subscription required"}), 403
    trending = cache.get("trending") or []
    if not trending and get_trending_watchlist:
        try:
            trending = get_trending_watchlist()
            cache.set("trending", trending)
        except Exception as e:
            logger.error(f"API trending: {e}")
    return jsonify({"trending": trending, "updated_at": cache.get_updated_at("trending")})

# ─── Demo Access ─────────────────────────────────────────────────────────────

@app.route("/demo-access")
@limiter.limit("3 per minute")
def demo_access():
    demo_token = os.environ.get("DEMO_TOKEN", "")
    if not demo_token or request.args.get("token") != demo_token:
        abort(404)
    user = User.query.filter_by(email='demo@1option.com').first()
    if not user:
        user = User(email='demo@1option.com', name='Demo User', is_subscribed=True)
        user.set_password(os.urandom(16).hex())
        db.session.add(user)
        db.session.commit()
    login_user(user)
    return redirect(url_for("dashboard"))

# ─── Performance Tracker ─────────────────────────────────────────────────

@app.route("/performance")
@login_required
def performance():
    if not current_user.is_subscribed:
        return redirect(url_for("pricing"))
    pick_stats = get_performance_stats(PickSnapshot)
    option_stats = get_options_performance_stats(OptionSnapshot)
    trending_stats = get_trending_performance_stats(TrendingSnapshot)
    return render_template("performance.html", stats=pick_stats, opt_stats=option_stats, trend_stats=trending_stats)

@app.route("/api/performance")
@login_required
def api_performance():
    if not current_user.is_subscribed:
        return jsonify({"error": "Subscription required"}), 403
    pick_stats = get_performance_stats(PickSnapshot)
    option_stats = get_options_performance_stats(OptionSnapshot)
    trending_stats = get_trending_performance_stats(TrendingSnapshot)
    return jsonify({
        "picks": {
            "total_picks": pick_stats["total_picks"],
            "win_rate": pick_stats["win_rate"],
            "avg_return": pick_stats["avg_return"],
            "winners": pick_stats["winners"],
            "losers": pick_stats["losers"],
        },
        "options": {
            "total_options": option_stats["total_options"],
            "total_calls": option_stats["total_calls"],
            "total_puts": option_stats["total_puts"],
            "win_rate": option_stats["win_rate"],
            "avg_return": option_stats["avg_return"],
        },
        "trending": {
            "total_tracked": trending_stats["total_tracked"],
            "win_rate": trending_stats["win_rate"],
            "avg_return": trending_stats["avg_return"],
            "winners": trending_stats["winners"],
            "losers": trending_stats["losers"],
        },
    })

# ─── Stocks Browser ──────────────────────────────────────────────────────────

def _get_fallback_stock_list():
    """Return a static stock list from built-in data when cache is still warming up."""
    from data_fetcher import STOCK_UNIVERSE, TICKER_SECTOR_MAP, COMPANY_NAMES
    fallback = []
    for ticker in STOCK_UNIVERSE:
        fallback.append({
            "ticker": ticker,
            "name": COMPANY_NAMES.get(ticker, ""),
            "price": 0.0,
            "pct_change": 0.0,
            "positive": True,
            "sector": TICKER_SECTOR_MAP.get(ticker, "Other"),
            "sparkline": [],
        })
    fallback.sort(key=lambda x: x["ticker"])
    return fallback

@app.route("/stocks")
@login_required
def stocks():
    if not current_user.is_subscribed:
        return redirect(url_for("pricing"))
    stock_list = cache.get("stock_list") or []
    # Fallback: if cache is still warming up, show static list from built-in data
    if not stock_list:
        stock_list = _get_fallback_stock_list()
    # Show all if requested, otherwise limit to 200 for fast page load
    show_all = request.args.get("all") == "1"
    display_list = stock_list if show_all else stock_list[:200]
    return render_template("stocks.html", stocks=display_list, total_count=len(stock_list))

@app.route("/api/stocks/search")
@login_required
@limiter.limit("60 per minute")
def api_stock_search():
    """Search stocks by ticker or name — returns up to 50 matches."""
    if not current_user.is_subscribed:
        return jsonify({"error": "Subscription required"}), 403
    q = request.args.get("q", "").strip().lower()
    if not q or len(q) < 1:
        return jsonify([])
    stock_list = cache.get("stock_list") or _get_fallback_stock_list()
    matches = [s for s in stock_list
               if q in s["ticker"].lower() or q in s.get("name", "").lower()
               or q in s.get("sector", "").lower()]
    return jsonify(matches[:50])

@app.route("/stocks/<ticker>")
@login_required
def stock_detail(ticker):
    if not current_user.is_subscribed:
        return redirect(url_for("pricing"))
    ticker = ticker.upper()
    if not TICKER_RE.match(ticker):
        flash("Invalid ticker format.", "error")
        return redirect(url_for("stocks"))
    detail = data_fetcher.get_stock_detail(ticker)
    if not detail.get("price"):
        flash(f"Ticker {ticker} not found or data unavailable.", "error")
        return redirect(url_for("stocks"))
    news = cache.get("news") or []
    ticker_news = [a for a in news if ticker.lower() in a.get("title", "").lower()][:8]
    all_news = ticker_news + [n for n in detail.get("news", [])
                              if n["title"] not in {a.get("title") for a in ticker_news}]
    return render_template("stock_detail.html", ticker=ticker, detail=detail, news=all_news[:10])

@app.route("/api/stocks/<ticker>/chart")
@login_required
def api_stock_chart(ticker):
    if not current_user.is_subscribed:
        return jsonify({"error": "Subscription required"}), 403
    ticker = ticker.upper()
    if not TICKER_RE.match(ticker):
        return jsonify({"error": "Invalid ticker"}), 400
    period = request.args.get("period", "1mo")
    if period not in VALID_PERIODS:
        return jsonify({"error": f"Invalid period. Valid: {', '.join(sorted(VALID_PERIODS))}"}), 400
    data = data_fetcher.get_stock_chart(ticker, period)
    return jsonify(data)

# ─── Error Handlers ──────────────────────────────────────────────────────────

@app.errorhandler(404)
def page_not_found(e):
    return render_template("errors/404.html"), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"500 error: {e}")
    return render_template("errors/500.html"), 500

@app.errorhandler(429)
def rate_limited(e):
    return render_template("errors/429.html"), 429

# ─── Static Files ────────────────────────────────────────────────────────────

@app.route("/robots.txt")
def robots():
    return send_from_directory(app.static_folder, "robots.txt")


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=debug, port=port)
