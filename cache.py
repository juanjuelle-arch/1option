"""
cache.py — Thread-safe in-memory data cache with timestamps.
Data is served instantly from cache; background scheduler refreshes periodically.
"""

import threading
from datetime import datetime, timezone

_cache = {}
_lock = threading.Lock()

def get(key):
    with _lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        return entry["data"]

def set(key, data):
    with _lock:
        _cache[key] = {
            "data": data,
            "updated_at": datetime.now(timezone.utc),
        }

def get_updated_at(key):
    with _lock:
        entry = _cache.get(key)
        if entry:
            return entry["updated_at"].strftime("%I:%M:%S %p UTC")
        return "Never"

def is_stale(key, max_age_seconds=300):
    with _lock:
        entry = _cache.get(key)
        if entry is None:
            return True
        age = (datetime.now(timezone.utc) - entry["updated_at"]).total_seconds()
        return age > max_age_seconds
