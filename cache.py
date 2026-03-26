"""
cache.py — Redis-backed data cache with in-memory fallback.
Production: uses Redis (shared across workers/replicas).
Local dev: falls back to thread-safe in-memory dict.
"""

import os
import json
import logging
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ─── Try Redis ──────────────────────────────────────────────────────────────

_redis = None
_REDIS_URL = os.environ.get("REDIS_URL", "")

if _REDIS_URL:
    try:
        import redis
        _redis = redis.from_url(_REDIS_URL, decode_responses=True, socket_timeout=5,
                                socket_connect_timeout=5, retry_on_timeout=True)
        _redis.ping()
        logger.info("Cache: Redis connected")
    except Exception as e:
        logger.warning(f"Cache: Redis unavailable ({e}), falling back to memory")
        _redis = None

# ─── In-memory fallback ─────────────────────────────────────────────────────

_mem_cache = {}
_lock = threading.Lock()

PREFIX = "1opt:"
DEFAULT_TTL = 900  # 15 min — data refreshes every 2-5 min, so this is generous


def _serialize(data):
    """Safely serialize data to JSON, handling non-serializable types."""
    try:
        return json.dumps(data, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(data))


def get(key):
    """Get cached data by key."""
    if _redis:
        try:
            raw = _redis.get(PREFIX + key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.debug(f"Redis get error for {key}: {e}")
            # Fall through to memory
    with _lock:
        entry = _mem_cache.get(key)
        if entry is None:
            return None
        return entry["data"]


def set(key, data, ttl=DEFAULT_TTL):
    """Set cached data with optional TTL (seconds)."""
    now = datetime.now(timezone.utc).isoformat()
    if _redis:
        try:
            _redis.setex(PREFIX + key, ttl, _serialize(data))
            _redis.setex(PREFIX + key + ":ts", ttl, now)
            return
        except Exception as e:
            logger.debug(f"Redis set error for {key}: {e}")
            # Fall through to memory
    with _lock:
        _mem_cache[key] = {
            "data": data,
            "updated_at": datetime.now(timezone.utc),
        }


def get_updated_at(key):
    """Get human-readable timestamp for last update."""
    if _redis:
        try:
            ts = _redis.get(PREFIX + key + ":ts")
            if ts:
                dt = datetime.fromisoformat(ts)
                return dt.strftime("%I:%M:%S %p UTC")
        except Exception:
            pass
    with _lock:
        entry = _mem_cache.get(key)
        if entry:
            return entry["updated_at"].strftime("%I:%M:%S %p UTC")
    return "Never"


def is_stale(key, max_age_seconds=300):
    """Check if cached data is older than max_age_seconds."""
    if _redis:
        try:
            ttl = _redis.ttl(PREFIX + key)
            if ttl < 0:
                return True
            # Key exists — check age from timestamp
            ts = _redis.get(PREFIX + key + ":ts")
            if ts:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds()
                return age > max_age_seconds
            return True
        except Exception:
            pass
    with _lock:
        entry = _mem_cache.get(key)
        if entry is None:
            return True
        age = (datetime.now(timezone.utc) - entry["updated_at"]).total_seconds()
        return age > max_age_seconds


def is_redis_connected():
    """Health check — is Redis reachable?"""
    if not _redis:
        return False
    try:
        _redis.ping()
        return True
    except Exception:
        return False
