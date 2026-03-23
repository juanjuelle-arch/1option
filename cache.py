"""
cache.py — In-memory data cache with timestamps.
Data is served instantly from cache; background scheduler refreshes every 5 min.
"""

from datetime import datetime

_cache = {}

def get(key):
    entry = _cache.get(key)
    if entry is None:
        return None
    return entry["data"]

def set(key, data):
    _cache[key] = {
        "data": data,
        "updated_at": datetime.now(),
    }

def get_updated_at(key):
    entry = _cache.get(key)
    if entry:
        return entry["updated_at"].strftime("%I:%M:%S %p")
    return "Never"

def is_stale(key, max_age_seconds=300):
    entry = _cache.get(key)
    if entry is None:
        return True
    age = (datetime.now() - entry["updated_at"]).total_seconds()
    return age > max_age_seconds
