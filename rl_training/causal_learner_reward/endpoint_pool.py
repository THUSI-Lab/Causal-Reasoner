

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit


logger = logging.getLogger(__name__)


class EndpointPoolError(RuntimeError):
    pass


class NoEligibleEndpointError(EndpointPoolError):
    pass


@dataclass(frozen=True)
class JudgeEndpoint:


    base_url: str
    model_name: Optional[str] = None
    weight: float = 1.0

    def effective_model(self, default_model: str) -> str:
        return self.model_name or default_model

    def state_key(self, default_model: str) -> str:
        return endpoint_state_key(self.base_url, self.effective_model(default_model))


@dataclass
class EndpointRuntime:
    endpoint: JudgeEndpoint
    semaphore: asyncio.Semaphore
    fail_count: int = 0
    cooldown_until: float = 0.0

    def is_available(self, now: float) -> bool:
        if self.cooldown_until and now >= self.cooldown_until:
            self.cooldown_until = 0.0
            self.fail_count = 0
        return not self.cooldown_until or now >= self.cooldown_until


def endpoint_state_key(base_url: str, model_name: str) -> str:
    return f"{base_url}|{model_name}"


def normalize_base_url(url: str) -> str:


    raw = (url or "").strip()
    if not raw:
        raise EndpointPoolError("empty_endpoint_url")
    if "://" not in raw:
        raw = f"http://{raw}"
    parts = urlsplit(raw)
    if not parts.netloc:
        raise EndpointPoolError(f"invalid_endpoint_url: {url}")
    path = parts.path.rstrip("/")
    if not path:
        path = "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")


class JudgeEndpointPool:


    def __init__(
        self,
        *,
        endpoints_file: str,
        default_model_name: str,
        reload_interval: float,
        max_concurrent_per_endpoint: int,
        failure_threshold: int,
        cooldown_seconds: float,
        name: str = "causal_judge",
    ):
        self._file_path = endpoints_file
        self._default_model_name = default_model_name
        self._reload_interval = float(reload_interval)
        self._max_concurrent_per_endpoint = int(max_concurrent_per_endpoint)
        self._failure_threshold = int(failure_threshold)
        self._cooldown_seconds = float(cooldown_seconds)
        self._name = name

        self._endpoints: List[JudgeEndpoint] = []
        self._runtime_by_key: Dict[str, EndpointRuntime] = {}
        self._file_mtime: Optional[float] = None
        self._last_check: float = 0.0

        if self._reload_interval < 0:
            raise ValueError("endpoint_reload_interval must be non-negative")
        if self._max_concurrent_per_endpoint <= 0:
            raise ValueError("max_concurrent_judge_requests_per_endpoint must be positive")
        if self._failure_threshold <= 0:
            raise ValueError("endpoint_failure_threshold must be positive")
        if self._cooldown_seconds < 0:
            raise ValueError("endpoint_cooldown_seconds must be non-negative")

        self._reload(force=True, strict=True)

    @property
    def endpoints(self) -> List[JudgeEndpoint]:
        self.reload_if_needed()
        return list(self._endpoints)

    @property
    def active_base_urls(self) -> set[str]:
        return {endpoint.base_url for endpoint in self.endpoints}

    def reload_if_needed(self) -> None:
        now = time.time()
        if self._reload_interval > 0 and (now - self._last_check) < self._reload_interval:
            return
        self._last_check = now
        self._reload(force=False, strict=False)

    def select_endpoint(self, avoid_keys: Optional[set[str]] = None) -> EndpointRuntime:
        self.reload_if_needed()
        now = time.monotonic()
        weighted = self._eligible_runtimes(now=now, avoid_keys=avoid_keys or set())
        if not weighted and avoid_keys:
            weighted = self._eligible_runtimes(now=now, avoid_keys=set())
        if not weighted:
            raise NoEligibleEndpointError(f"[{self._name}] no eligible endpoints")
        runtimes, weights = zip(*weighted)
        return random.choices(list(runtimes), weights=list(weights), k=1)[0]

    def record_success(self, state_key: str) -> None:
        runtime = self._runtime_by_key.get(state_key)
        if runtime is None:
            return
        runtime.fail_count = 0
        runtime.cooldown_until = 0.0

    def record_failure(self, state_key: str, *, cooldown: bool) -> None:
        runtime = self._runtime_by_key.get(state_key)
        if runtime is None or not cooldown:
            return
        runtime.fail_count += 1
        if runtime.fail_count >= self._failure_threshold:
            runtime.cooldown_until = time.monotonic() + self._cooldown_seconds
            logger.warning(
                "[%s] endpoint cooldown: %s model=%s failures=%s cooldown=%ss",
                self._name,
                runtime.endpoint.base_url,
                runtime.endpoint.model_name or self._default_model_name,
                runtime.fail_count,
                self._cooldown_seconds,
            )

    def _eligible_runtimes(self, *, now: float, avoid_keys: set[str]) -> List[tuple[EndpointRuntime, float]]:
        weighted: List[tuple[EndpointRuntime, float]] = []
        for endpoint in self._endpoints:
            key = endpoint.state_key(self._default_model_name)
            if key in avoid_keys:
                continue
            runtime = self._runtime_by_key.get(key)
            if runtime is None or not runtime.is_available(now):
                continue
            weighted.append((runtime, endpoint.weight))
        return weighted

    def _reload(self, *, force: bool, strict: bool) -> None:
        if not self._file_path:
            raise EndpointPoolError("missing_endpoints_file")
        if not os.path.exists(self._file_path):
            message = f"[{self._name}] endpoints file not found: {self._file_path}"
            if strict:
                raise FileNotFoundError(message)
            logger.warning(message)
            return

        try:
            mtime = os.path.getmtime(self._file_path)
        except OSError as exc:
            if strict:
                raise
            logger.warning("[%s] cannot stat endpoint file %s: %s", self._name, self._file_path, exc)
            return

        if not force and self._file_mtime == mtime:
            return

        try:
            with open(self._file_path, "r", encoding="utf-8") as handle:
                content = handle.read()
            endpoints = self._parse_content(content)
            if not endpoints:
                raise EndpointPoolError("empty_endpoints_file")
        except Exception as exc:
            if strict:
                raise
            logger.error("[%s] endpoint reload failed, keeping previous state: %s", self._name, exc)
            return

        self._install_endpoints(endpoints)
        self._file_mtime = mtime
        logger.info("[%s] loaded %s judge endpoints from %s", self._name, len(endpoints), self._file_path)

    def _install_endpoints(self, endpoints: Sequence[JudgeEndpoint]) -> None:
        previous = self._runtime_by_key
        runtime_by_key: Dict[str, EndpointRuntime] = {}
        for endpoint in endpoints:
            key = endpoint.state_key(self._default_model_name)
            old = previous.get(key)
            if old is None:
                old = EndpointRuntime(
                    endpoint=endpoint,
                    semaphore=asyncio.Semaphore(self._max_concurrent_per_endpoint),
                )
            else:
                old.endpoint = endpoint
            runtime_by_key[key] = old
        self._endpoints = list(endpoints)
        self._runtime_by_key = runtime_by_key

    def _parse_content(self, content: str) -> List[JudgeEndpoint]:
        if not content.strip():
            return []
        if self._file_path.endswith(".json"):
            return self._parse_json_config(json.loads(content))
        try:
            return self._parse_json_config(json.loads(content))
        except json.JSONDecodeError:
            return self._parse_txt_config(content)

    def _parse_json_config(self, data: Any) -> List[JudgeEndpoint]:
        if isinstance(data, list):
            groups: Sequence[Any] = [{"endpoints": data, "weight": 1}]
        elif isinstance(data, Mapping):
            if "groups" in data:
                groups = data.get("groups") or []
            elif "endpoints" in data:
                groups = [data]
            else:
                raise EndpointPoolError("json endpoint config must contain groups or endpoints")
        else:
            raise EndpointPoolError("json endpoint config must be an object or list")

        endpoints: List[JudgeEndpoint] = []
        for group in groups:
            if not isinstance(group, Mapping):
                raise EndpointPoolError("endpoint group must be an object")
            group_weight = _positive_weight(group.get("weight", 1))
            group_model = _optional_text(group.get("model_name"))
            group_endpoints = group.get("endpoints") or []
            if not isinstance(group_endpoints, list):
                raise EndpointPoolError("group endpoints must be a list")
            for item in group_endpoints:
                endpoints.append(self._parse_endpoint_item(item, group_model, group_weight))
        return endpoints

    def _parse_txt_config(self, content: str) -> List[JudgeEndpoint]:
        endpoints: List[JudgeEndpoint] = []
        for raw in content.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            endpoints.append(JudgeEndpoint(base_url=normalize_base_url(line), weight=1.0))
        return endpoints

    def _parse_endpoint_item(
        self,
        item: Any,
        group_model_name: Optional[str],
        group_weight: float,
    ) -> JudgeEndpoint:
        if isinstance(item, str):
            return JudgeEndpoint(
                base_url=normalize_base_url(item),
                model_name=group_model_name,
                weight=group_weight,
            )
        if not isinstance(item, Mapping):
            raise EndpointPoolError("endpoint item must be a URL string or object")

        raw_url = item.get("url", item.get("base_url", ""))
        model_name = _optional_text(item.get("model_name")) or group_model_name
        weight = _positive_weight(item.get("weight", group_weight))
        return JudgeEndpoint(base_url=normalize_base_url(str(raw_url)), model_name=model_name, weight=weight)


def _positive_weight(value: Any) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise EndpointPoolError(f"endpoint weight must be positive: {value}") from exc
    if weight <= 0:
        raise EndpointPoolError(f"endpoint weight must be positive: {value}")
    return weight


def _optional_text(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
