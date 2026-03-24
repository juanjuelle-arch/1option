"""
app.py — StockPulse Flask application
Routes: landing, dashboard, auth, Stripe, API endpoints
"""

import os
import re
import logging
from urllib.parse import urlparse
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, abort
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
import stripe
import cache
import data_fetcher
from models import db, User
from scheduler import start_scheduler, refresh_main as refresh_all

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App Setup ───────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(32).hex())
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///stockpulse.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # no CSS/JS caching during dev

# ─── Security Config ─────────────────────────────────────────────────────────
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
if os.environ.get("RAILWAY_ENVIRONMENT"):
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

# Ticker validation pattern: 1-5 alphanumeric chars, optional -X suffix (e.g. BRK-B)
TICKER_RE = re.compile(r"^[A-Z0-9]{1,5}(-[A-Z])?$")
VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "5y"}

@app.after_request
def add_headers(response):
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if os.environ.get("RAILWAY_ENVIRONMENT"):
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    # Prevent stale CSS/JS on the external proxy
    if request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response

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
    return User.query.get(int(user_id))

# ─── Init DB + Seed Cache ────────────────────────────────────────────────────

with app.app_context():
    db.create_all()

# Do initial cache load in background so app starts fast
import threading
def _initial_load():
    with app.app_context():
        refresh_all()
threading.Thread(target=_initial_load, daemon=True).start()

# Start 5-min refresh scheduler
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
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
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
            next_page = request.args.get("next")
            # Prevent open redirect: validate with urlparse
            if next_page:
                parsed = urlparse(next_page)
                if parsed.netloc or parsed.scheme or next_page.startswith("//"):
                    next_page = None
            return redirect(next_page or url_for("dashboard"))
        logger.warning(f"Failed login attempt for {email} from {request.remote_addr}")
        flash("Invalid email or password.", "error")
    return render_template("login.html")

@app.route("/logout")
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
    sentiment = cache.get("sentiment") or {}
    earnings = cache.get("earnings") or []
    top_calls = cache.get("top_calls") or []
    top_puts = cache.get("top_puts") or []
    return render_template("dashboard.html",
                           picks=picks,
                           news=news,
                           sentiment=sentiment,
                           earnings=earnings,
                           top_calls=top_calls,
                           top_puts=top_puts,
                           updated_at=cache.get_updated_at("picks"))

# ─── Stripe ──────────────────────────────────────────────────────────────────

@app.route("/create-checkout-session", methods=["POST"])
@login_required
def create_checkout_session():
    if not stripe.api_key or not STRIPE_PRICE_ID:
        # Demo mode — skip Stripe, grant access directly
        current_user.is_subscribed = True
        db.session.commit()
        flash("Demo mode: subscription activated!", "success")
        return redirect(url_for("dashboard"))
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
    if session_id and stripe.api_key:
        try:
            checkout = stripe.checkout.Session.retrieve(session_id)
            if checkout.payment_status == "paid":
                current_user.is_subscribed = True
                current_user.stripe_customer_id = checkout.customer
                current_user.stripe_subscription_id = checkout.subscription
                db.session.commit()
        except Exception as e:
            logger.error(f"Payment success error: {e}")
    else:
        current_user.is_subscribed = True
        db.session.commit()
    flash("Welcome to StockPulse Pro! Your dashboard is ready.", "success")
    return redirect(url_for("dashboard"))

@app.route("/webhook", methods=["POST"])
@csrf.exempt
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        if event["type"] == "customer.subscription.deleted":
            customer_id = event["data"]["object"]["customer"]
            user = User.query.filter_by(stripe_customer_id=customer_id).first()
            if user:
                user.is_subscribed = False
                db.session.commit()
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

# ─── Dev: grant free access for testing ──────────────────────────────────────

@app.route("/dev-activate")
@login_required
def dev_activate():
    if app.debug:
        current_user.is_subscribed = True
        db.session.commit()
        flash("Dev mode: subscription activated!", "success")
        return redirect(url_for("dashboard"))
    return redirect(url_for("index"))

@app.route("/demo-access")
@limiter.limit("3 per minute")
def demo_access():
    """Direct demo access — requires DEMO_TOKEN env var or debug mode."""
    demo_token = os.environ.get("DEMO_TOKEN", "")
    if not app.debug and (not demo_token or request.args.get("token") != demo_token):
        abort(404)
    user = User.query.filter_by(email='demo@stockpulse.com').first()
    if not user:
        user = User(email='demo@stockpulse.com', name='Demo User', is_subscribed=True)
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
    # Merge with ticker-specific news
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
        period = "1mo"
    data = data_fetcher.get_stock_chart(ticker, period)
    return jsonify(data)


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=debug, port=port)
