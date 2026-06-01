

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from typing import Any, Hashable, Optional


def stable_hash(value: Any) -> str:


    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        payload = repr(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_fingerprint(path: Optional[str]) -> str:
    if not path:
        return "missing"
    normalized = path.removeprefix("file://")
    try:
        stat = os.stat(normalized)
        return f"{normalized}|mtime={stat.st_mtime_ns}|size={stat.st_size}"
    except OSError:
        return f"{normalized}|missing"


class LRUCache:


    def __init__(self, max_entries: int = 2048):
        self.max_entries = max_entries
        self._data: OrderedDict[Hashable, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Any:
        with self._lock:
            if key not in self._data:
                return None
            value = self._data.pop(key)
            self._data[key] = value
            return value

    def set(self, key: Hashable, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.pop(key)
            self._data[key] = value
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


VIDEO_PREPROCESS_CACHE = LRUCache(max_entries=1024)
JUDGE_RESULT_CACHE = LRUCache(max_entries=8192)
