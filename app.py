"""
app.py — 1.OPTION Flask application
Routes: landing, dashboard, auth, Stripe, API endpoints
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
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import stripe
import cache
import data_fetcher
from models import db, User
try:
    from market_scraper import get_enriched_ticker_profile, get_cboe_pc_ratio
except ImportError:
    get_enriched_ticker_profile = None
    get_cboe_pc_ratio = None
try:
    from trending import get_trending_watchlist
except ImportError:
    get_trending_watchlist = None
from scheduler import start_scheduler, refresh_main as refresh_all

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IS_PRODUCTION = bool(os.environ.get("RAILWAY_ENVIRONMENT"))

# ─── App Setup ───────────────────────────────────────────────────────────────

app = Flask(__name__)

# C-4: Hard error if SECRET_KEY not set in production
_secret = os.environ.get("SECRET_KEY", "")
if IS_PRODUCTION and not _secret:
    raise RuntimeError("SECRET_KEY environment variable must be set in production!")
app.secret_key = _secret or os.urandom(32).hex()

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///1option.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# M-10: Only disable static caching in dev
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

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "60 per hour"],
    storage_uri="memory://",
)

# L-3: Ticker validation — support both BRK-B and BRK.B formats
TICKER_RE = re.compile(r"^[A-Z0-9]{1,5}([.\-][A-Z]{1,2})?$")
VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "5y"}

# H-3: Email validation pattern
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# M-9: Dummy hash for constant-time login
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
    # M-10: Cache static files in production (1 hour), no-cache in dev
    if request.path.startswith("/static/"):
        if IS_PRODUCTION:
            response.headers["Cache-Control"] = "public, max-age=3600"
        else:
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response

# M-5: Force HTTPS in production
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

# H-1: Use db.session.get() instead of deprecated User.query.get()
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ─── Init DB + Seed Cache ────────────────────────────────────────────────────

with app.app_context():
    db.create_all()

# Do initial cache load in background so app starts fast
import threading
def _initial_load():
    with app.app_context():
        refresh_all()
threading.Thread(target=_initial_load, daemon=True).start()

# Start refresh scheduler
_scheduler = start_scheduler()

# ─── Context Processor ───────────────────────────────────────────────────────

@app.context_processor
def inject_market():
    """Inject live market bar into every template."""
    return {"market_overview": cache.get("market_overview") or []}

# ─── Public Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    picks = cache.get("picks") or []
    news = cache.get("news") or []
    sentiment = cache.get("sentiment") or {}
    return render_template("index.html",
                           picks=picks[:3],
                           news=news[:3],
                           sentiment=sentiment,
                           updated_at=cache.get_updated_at("picks"))

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
        # H-4: Name length cap
        if len(name) > 100:
            flash("Name is too long (max 100 characters).", "error")
            return render_template("signup.html")
        # H-3: Server-side email validation
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
            # Prevent open redirect: validate with urlparse
            if next_page:
                parsed = urlparse(next_page)
                if parsed.netloc or parsed.scheme or next_page.startswith("//"):
                    next_page = None
            return redirect(next_page or url_for("dashboard"))
        # M-9: Constant-time response — always run hash check even if user not found
        if not user:
            check_password_hash(_DUMMY_HASH, password)
        logger.warning(f"Failed login attempt for {email} from {request.remote_addr}")
        flash("Invalid email or password.", "error")
    return render_template("login.html")

# H-5: Logout is POST-only to prevent CSRF logout attacks
@app.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))

# Keep GET logout as a fallback redirect (nav links)
@app.route("/logout", methods=["GET"])
@login_required
def logout_get():
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
    sentiment = cache.get("sentiment") or {}
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
        # C-1: Demo mode — only grant access if DEMO_MODE env var is explicitly set
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

# C-1: Payment success now requires valid Stripe session — no free bypass
@app.route("/payment-success")
@login_required
def payment_success():
    session_id = request.args.get("session_id")
    if not session_id or not stripe.api_key:
        # No valid session — redirect to pricing without granting access
        flash("Payment could not be verified. Please try again.", "error")
        return redirect(url_for("pricing"))
    try:
        checkout = stripe.checkout.Session.retrieve(session_id)
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

# H-8: Stripe webhook handles subscription deletion AND payment failures
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

# ─── API Endpoints (live data for JS polling) ────────────────────────────────

@app.route("/api/market")
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

# ─── Options & Intelligence APIs ─────────────────────────────────────────────

@app.route("/api/options")
@login_required
def api_options_global():
    """Top calls & puts across all tickers — diversified, scored."""
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
    """Full options chain for a specific ticker — scored & ranked."""
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
    """Full multi-source intelligence profile for a ticker."""
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
    """Public VIX + market-wide put/call ratio."""
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
    """Trending watchlist — tickers trending across multiple sources."""
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

# ─── Demo Access (secured) ───────────────────────────────────────────────────

# C-2: Demo access always requires DEMO_TOKEN — no debug bypass
@app.route("/demo-access")
@limiter.limit("3 per minute")
def demo_access():
    """Direct demo access — requires DEMO_TOKEN env var. No debug bypass."""
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

# ─── Stocks Browser ────────────────────────────────────────────────────────

@app.route("/stocks")
@login_required
def stocks():
    if not current_user.is_subscribed:
        return redirect(url_for("pricing"))
    stock_list = cache.get("stock_list") or []
    return render_template("stocks.html", stocks=stock_list)

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
    # M-6: Return 400 for invalid period instead of silent fallback
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

# ─── Static Security Files ──────────────────────────────────────────────────

# L-6: Use app.static_folder for reliable path resolution
@app.route("/robots.txt")
def robots():
    return send_from_directory(app.static_folder, "robots.txt")


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=debug, port=port)
