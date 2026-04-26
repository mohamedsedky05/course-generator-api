import hashlib
import time
from typing import Any, Optional

_store: dict = {}
TTL = 3600  # 1 hour


def make_key(video_url: str) -> str:
    return hashlib.sha256(video_url.strip().lower().encode()).hexdigest()


def get(key: str) -> Optional[Any]:
    entry = _store.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > TTL:
        del _store[key]
        return None
    return entry["data"]


def set(key: str, data: Any) -> None:
    _store[key] = {"data": data, "ts": time.time()}


def invalidate(key: str) -> None:
    _store.pop(key, None)


def size() -> int:
    # Evict expired entries on size check
    now = time.time()
    expired = [k for k, v in _store.items() if now - v["ts"] > TTL]
    for k in expired:
        del _store[k]
    return len(_store)
