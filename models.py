"""
models.py — Database models for 1.OPTION
User accounts + Pick history for performance tracking.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __table_args__ = (db.Index('ix_user_email', 'email'),)

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    is_subscribed = db.Column(db.Boolean, default=False)
    stripe_customer_id = db.Column(db.String(100), nullable=True, index=True)
    stripe_subscription_id = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.email}>"


class PickSnapshot(db.Model):
    """
    Records each time a stock appears in the Top 10 Picks.
    Tracks entry price so we can calculate performance over time.
    """
    __tablename__ = "pick_snapshots"
    __table_args__ = (
        db.Index('ix_pick_ticker_date', 'ticker', 'picked_date'),
    )

    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), nullable=False)
    company = db.Column(db.String(100), nullable=False)
    sector = db.Column(db.String(50), nullable=True)

    # Price at time of pick
    entry_price = db.Column(db.Float, nullable=False)
    # Current/latest price (updated periodically)
    current_price = db.Column(db.Float, nullable=True)

    # GARP score at time of pick
    score = db.Column(db.Integer, nullable=False)
    conviction = db.Column(db.String(20), nullable=True)
    rank = db.Column(db.Integer, nullable=True)  # 1-10

    # Fundamental data at pick time
    rev_growth = db.Column(db.Float, nullable=True)
    earn_growth = db.Column(db.Float, nullable=True)
    pe_fwd = db.Column(db.Float, nullable=True)
    target_upside = db.Column(db.Float, nullable=True)

    # Performance
    pct_return = db.Column(db.Float, default=0.0)  # (current - entry) / entry * 100
    peak_price = db.Column(db.Float, nullable=True)  # Highest price since pick
    peak_return = db.Column(db.Float, default=0.0)  # Best % return achieved

    # Timestamps
    picked_date = db.Column(db.Date, nullable=False)
    picked_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    # Status: active (still in top 10), closed (dropped out)
    is_active = db.Column(db.Boolean, default=True)
    closed_at = db.Column(db.DateTime, nullable=True)
    closed_price = db.Column(db.Float, nullable=True)
    closed_return = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f"<Pick {self.ticker} {self.picked_date} entry=${self.entry_price} return={self.pct_return}%>"


class OptionSnapshot(db.Model):
    """
    Records each time an option appears in the Top 8 Calls or Top 8 Puts.
    Tracks entry price so we can calculate performance over time.
    """
    __tablename__ = "option_snapshots"
    __table_args__ = (
        db.Index('ix_opt_ticker_strike_date', 'ticker', 'strike', 'expiry', 'option_type', 'picked_date'),
    )

    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), nullable=False)
    option_type = db.Column(db.String(4), nullable=False)  # CALL or PUT
    strike = db.Column(db.Float, nullable=False)
    expiry = db.Column(db.String(20), nullable=False)

    # Price at time of pick
    entry_price = db.Column(db.Float, nullable=False)   # option last_price when first picked
    current_price = db.Column(db.Float, nullable=True)   # latest option price

    # Contract details at pick time
    volume = db.Column(db.Integer, nullable=True)
    open_interest = db.Column(db.Integer, nullable=True)
    iv = db.Column(db.Float, nullable=True)               # implied volatility %

    # Underlying stock price at entry
    stock_entry_price = db.Column(db.Float, nullable=True)
    stock_current_price = db.Column(db.Float, nullable=True)

    # Performance
    pct_return = db.Column(db.Float, default=0.0)          # (current - entry) / entry * 100
    peak_price = db.Column(db.Float, nullable=True)        # highest option price since pick
    peak_return = db.Column(db.Float, default=0.0)         # best % return achieved

    # Timestamps
    picked_date = db.Column(db.Date, nullable=False)
    picked_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    # Status
    is_active = db.Column(db.Boolean, default=True)
    closed_at = db.Column(db.DateTime, nullable=True)
    closed_price = db.Column(db.Float, nullable=True)
    closed_return = db.Column(db.Float, nullable=True)

    @property
    def contract_label(self):
        return f"{self.ticker} ${self.strike:.0f} {self.option_type} {self.expiry}"

    def __repr__(self):
        return f"<Option {self.ticker} ${self.strike} {self.option_type} entry=${self.entry_price} return={self.pct_return}%>"


class TrendingSnapshot(db.Model):
    """
    Records each time a stock appears in the Trending watchlist.
    Tracks entry price so we can calculate performance over time.
    """
    __tablename__ = "trending_snapshots"
    __table_args__ = (
        db.Index('ix_trending_ticker_date', 'ticker', 'picked_date'),
    )

    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), nullable=False)

    # Price when first spotted
    price = db.Column(db.Float, nullable=True)
    # Current/latest price (updated periodically)
    current_price = db.Column(db.Float, nullable=True)
    # Daily change % when spotted
    change_pct = db.Column(db.Float, nullable=True)

    # Conviction
    conviction_score = db.Column(db.Integer, nullable=False)  # 0-100
    conviction_label = db.Column(db.String(20), nullable=False)  # EXTREME/HIGH/MEDIUM/LOW

    # Sources
    source_count = db.Column(db.Integer, nullable=True)
    sources = db.Column(db.String(200), nullable=True)  # comma-separated source names

    # Performance
    pct_return = db.Column(db.Float, default=0.0)  # (current - price) / price * 100
    peak_price = db.Column(db.Float, nullable=True)
    peak_return = db.Column(db.Float, default=0.0)

    # Timestamps
    picked_date = db.Column(db.Date, nullable=False)
    picked_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    # Status
    is_active = db.Column(db.Boolean, default=True)
    closed_at = db.Column(db.DateTime, nullable=True)
    closed_price = db.Column(db.Float, nullable=True)
    closed_return = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f"<Trending {self.ticker} {self.picked_date} entry=${self.price} return={self.pct_return}%>"
