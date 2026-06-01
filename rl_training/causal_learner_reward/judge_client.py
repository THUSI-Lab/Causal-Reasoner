

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

try:
    from .cache import JUDGE_RESULT_CACHE, stable_hash
    from .config import JudgeAPIConfig
    from .endpoint_pool import (
        EndpointRuntime,
        JudgeEndpoint,
        JudgeEndpointPool,
        NoEligibleEndpointError,
        normalize_base_url,
    )
    from .judge_prompts import build_judge_messages
    from .video_payload import MissingVideoError, UnsupportedVideoInputError, VideoPayload, prepare_video_payload
except ImportError:
    from cache import JUDGE_RESULT_CACHE, stable_hash                
    from config import JudgeAPIConfig                
    from endpoint_pool import (                
        EndpointRuntime,
        JudgeEndpoint,
        JudgeEndpointPool,
        NoEligibleEndpointError,
        normalize_base_url,
    )
    from judge_prompts import build_judge_messages                
    from video_payload import (                
        MissingVideoError,
        UnsupportedVideoInputError,
        VideoPayload,
        prepare_video_payload,
    )


logger = logging.getLogger(__name__)


@dataclass
class JudgeRequest:
    task_id: str
    question: str
    ground_truth: Mapping[str, Any]
    model_output: str
    extra_info: Mapping[str, Any]
    rubric_id: str


@dataclass
class JudgeResult:
    payload: Optional[Dict[str, Any]]
    status: str
    error: Optional[str] = None
    video_key: Optional[str] = None
    cached: bool = False
    request_count: int = 0
    retry_count: int = 0
    pool_timeout_count: int = 0
    endpoint_timeout_count: int = 0
    cache_hit: int = 0
    video_preprocess_count: int = 0


@dataclass
class _AttemptStats:
    request_count: int = 0
    retry_count: int = 0
    pool_timeout_count: int = 0
    endpoint_timeout_count: int = 0
    cache_hit: int = 0
    video_preprocess_count: int = 0

    def to_result_kwargs(self) -> Dict[str, int]:
        return {
            "request_count": self.request_count,
            "retry_count": self.retry_count,
            "pool_timeout_count": self.pool_timeout_count,
            "endpoint_timeout_count": self.endpoint_timeout_count,
            "cache_hit": self.cache_hit,
            "video_preprocess_count": self.video_preprocess_count,
        }


@dataclass(frozen=True)
class _RetryDecision:
    retry: bool
    cooldown_endpoint: bool
    status: str = "judge_request_error"
    pool_timeout: bool = False
    endpoint_timeout: bool = False


class MultimodalJudgeClient:


    def __init__(self, config: JudgeAPIConfig):
        self.config = config
        self.config.validate()
        self._clients: Dict[tuple[str, str], Any] = {}
        self._http_client: Any = None
        self._http_timeout: Any = None
        self._endpoint_pool: Optional[JudgeEndpointPool] = None
        self._static_endpoint = JudgeEndpoint(base_url=normalize_base_url(self.config.base_url))
        self._static_runtime = EndpointRuntime(
            endpoint=self._static_endpoint,
            semaphore=asyncio.Semaphore(self.config.max_concurrent_judge_requests_per_endpoint),
        )
        if self.config.endpoints_file:
            self._endpoint_pool = JudgeEndpointPool(
                endpoints_file=self.config.endpoints_file,
                default_model_name=self.config.model_name,
                reload_interval=self.config.endpoint_reload_interval,
                max_concurrent_per_endpoint=self.config.max_concurrent_judge_requests_per_endpoint,
                failure_threshold=self.config.endpoint_failure_threshold,
                cooldown_seconds=self.config.endpoint_cooldown_seconds,
            )

    def judge_many_sync(self, requests: List[JudgeRequest]) -> List[JudgeResult]:
        if not requests:
            return []
        return _run_sync(self._judge_many_async_with_cleanup(requests))

    def judge_one_sync(self, request: JudgeRequest) -> JudgeResult:
        return self.judge_many_sync([request])[0]

    async def _judge_many_async_with_cleanup(self, requests: List[JudgeRequest]) -> List[JudgeResult]:
        try:
            return await self._judge_many_async(requests)
        finally:
            await self._close_async_resources()

    async def _judge_many_async(self, requests: List[JudgeRequest]) -> List[JudgeResult]:
        results: List[JudgeResult] = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_judge_requests)
        for start in range(0, len(requests), self.config.judge_chunk_size):
            chunk = requests[start : start + self.config.judge_chunk_size]
            chunk_results = await asyncio.gather(*(self._call_one(req, semaphore) for req in chunk))
            results.extend(chunk_results)
        return results

    async def _call_one(self, request: JudgeRequest, global_semaphore: asyncio.Semaphore) -> JudgeResult:
        stats = _AttemptStats()
        try:
            video_payload = prepare_video_payload(request.extra_info, self.config)
        except (MissingVideoError, UnsupportedVideoInputError, OSError, ValueError) as exc:
            return JudgeResult(payload=None, status="video_input_error", error=str(exc), **stats.to_result_kwargs())

        stats.video_preprocess_count = video_payload.preprocess_count
        system_prompt, user_text = build_judge_messages(
            task_id=request.task_id,
            question=request.question,
            ground_truth=request.ground_truth,
            model_output=request.model_output,
        )

        last_error = "judge_failed"
        avoid_endpoint_keys: set[str] = set()

        for attempt in range(self.config.max_retries):
            try:
                runtime = self._select_endpoint_runtime(avoid_endpoint_keys)
            except NoEligibleEndpointError as exc:
                return JudgeResult(
                    payload=None,
                    status="judge_request_error",
                    error=str(exc),
                    video_key=video_payload.video_key,
                    **stats.to_result_kwargs(),
                )

            endpoint = runtime.endpoint
            effective_model = endpoint.effective_model(self.config.model_name)
            endpoint_key = endpoint.state_key(self.config.model_name)
            cache_key = self._judge_cache_key(
                request=request,
                video_payload=video_payload,
                effective_model=effective_model,
                system_prompt=system_prompt,
                user_text=user_text,
            )
            if self.config.enable_judge_result_cache:
                cached = JUDGE_RESULT_CACHE.get(cache_key)
                if cached is not None:
                    stats.cache_hit = 1
                    return JudgeResult(
                        payload=cached,
                        status="ok",
                        video_key=video_payload.video_key,
                        cached=True,
                        **stats.to_result_kwargs(),
                    )

            try:
                async with global_semaphore:
                    async with runtime.semaphore:
                        stats.request_count += 1
                        response_text = await self._request_judge(
                            request=request,
                            video_payload=video_payload,
                            endpoint=endpoint,
                            effective_model=effective_model,
                            system_prompt=system_prompt,
                            user_text=user_text,
                        )
                payload = parse_json_response(response_text)
                self._record_endpoint_success(endpoint_key)
                if self.config.enable_judge_result_cache:
                    JUDGE_RESULT_CACHE.set(cache_key, payload)
                return JudgeResult(
                    payload=payload,
                    status="ok",
                    video_key=video_payload.video_key,
                    **stats.to_result_kwargs(),
                )
            except Exception as exc:
                decision = _classify_judge_exception(exc)
                last_error = f"{decision.status}: {exc}"
                self._record_endpoint_failure(endpoint_key, cooldown=decision.cooldown_endpoint)
                if decision.cooldown_endpoint:
                    avoid_endpoint_keys.add(endpoint_key)
                if decision.pool_timeout:
                    stats.pool_timeout_count += 1
                if decision.endpoint_timeout:
                    stats.endpoint_timeout_count += 1
                logger.warning(
                    "judge request failed on %s model=%s attempt=%s/%s retry=%s cooldown=%s: %s",
                    endpoint.base_url,
                    effective_model,
                    attempt + 1,
                    self.config.max_retries,
                    decision.retry,
                    decision.cooldown_endpoint,
                    exc,
                    exc_info=True,
                )
                if not decision.retry or attempt >= self.config.max_retries - 1:
                    return JudgeResult(
                        payload=None,
                        status=decision.status,
                        error=last_error,
                        video_key=video_payload.video_key,
                        **stats.to_result_kwargs(),
                    )
                stats.retry_count += 1
                await self._sleep_before_retry(attempt)

        return JudgeResult(
            payload=None,
            status="judge_request_error",
            error=last_error,
            video_key=video_payload.video_key,
            **stats.to_result_kwargs(),
        )

    def _judge_cache_key(
        self,
        *,
        request: JudgeRequest,
        video_payload: VideoPayload,
        effective_model: str,
        system_prompt: str,
        user_text: str,
    ) -> str:
        return stable_hash(
            {
                "rubric_id": request.rubric_id,
                "task_id": request.task_id,
                "video_key": video_payload.video_key,
                "gt_hash": stable_hash(request.ground_truth),
                "model_output_hash": stable_hash(request.model_output),
                "api_family": self.config.mode,
                "api_transport": _transport_name(self.config),
                "video_transport": self.config.video_transport,
                "native_video_content_type": self.config.native_video_content_type,
                "judge_model": effective_model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "prompt_hash": stable_hash({"system": system_prompt, "user": user_text}),
            }
        )

    async def _request_judge(
        self,
        *,
        request: JudgeRequest,
        video_payload: VideoPayload,
        endpoint: JudgeEndpoint,
        effective_model: str,
        system_prompt: str,
        user_text: str,
    ) -> str:
        if self.config.mode == "vllm":
            return await self._request_chat_completions(
                request=request,
                video_payload=video_payload,
                endpoint=endpoint,
                effective_model=effective_model,
                system_prompt=system_prompt,
                user_text=user_text,
                provider="vllm",
            )
        if self.config.mode == "openai" and self.config.openai_use_responses_api:
            return await self._request_responses(
                request=request,
                video_payload=video_payload,
                endpoint=endpoint,
                effective_model=effective_model,
                system_prompt=system_prompt,
                user_text=user_text,
            )
        return await self._request_chat_completions(
            request=request,
            video_payload=video_payload,
            endpoint=endpoint,
            effective_model=effective_model,
            system_prompt=system_prompt,
            user_text=user_text,
            provider="openai",
        )

    async def _request_chat_completions(
        self,
        *,
        request: JudgeRequest,
        video_payload: VideoPayload,
        endpoint: JudgeEndpoint,
        effective_model: str,
        system_prompt: str,
        user_text: str,
        provider: str,
    ) -> str:
        del request
        client = self._get_openai_client(endpoint.base_url, self.config.resolved_api_key)
        content: List[Dict[str, Any]] = []
        if video_payload.video_url:
            content.append({"type": "video_url", "video_url": {"url": video_payload.video_url}})
        for frame in video_payload.frames_base64 or []:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
        content.append({"type": "text", "text": user_text})

        kwargs: Dict[str, Any] = {
            "model": effective_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self._get_http_timeout(),
        }
        extra_body: Dict[str, Any] = {}
        if provider == "vllm" and video_payload.mm_processor_kwargs:
            extra_body["mm_processor_kwargs"] = video_payload.mm_processor_kwargs
        if extra_body:
            kwargs["extra_body"] = extra_body
        completion = await asyncio.wait_for(
            client.chat.completions.create(**kwargs),
            timeout=self.config.timeout + self.config.request_timeout_buffer,
        )
        return completion.choices[0].message.content or ""

    async def _request_responses(
        self,
        *,
        request: JudgeRequest,
        video_payload: VideoPayload,
        endpoint: JudgeEndpoint,
        effective_model: str,
        system_prompt: str,
        user_text: str,
    ) -> str:
        del request
        client = self._get_openai_client(endpoint.base_url, self.config.resolved_api_key)
        user_content: List[Dict[str, Any]] = []
        if video_payload.video_url:
            user_content.append({"type": self.config.native_video_content_type, "video_url": video_payload.video_url})
        for frame in video_payload.frames_base64 or []:
            user_content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{frame}"})
        user_content.append({"type": "input_text", "text": user_text})

        response = await asyncio.wait_for(
            client.responses.create(
                model=effective_model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": user_content},
                ],
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                timeout=self._get_http_timeout(),
            ),
            timeout=self.config.timeout + self.config.request_timeout_buffer,
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text
        return _extract_responses_text(response)

    def _get_openai_client(self, base_url: str, api_key: str):
        key = (base_url, api_key)
        if key in self._clients:
            return self._clients[key]
        try:
            import httpx
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("openai and httpx packages are required for external judge calls") from exc

        if self._http_client is None:
            self._http_timeout = httpx.Timeout(
                connect=self.config.http_connect_timeout,
                read=self.config.effective_http_read_timeout,
                write=self.config.http_write_timeout,
                pool=self.config.http_pool_timeout,
            )
            self._http_client = httpx.AsyncClient(
                timeout=self._http_timeout,
                trust_env=self.config.effective_http_trust_env,
                limits=httpx.Limits(
                    max_connections=self.config.http_max_connections,
                    max_keepalive_connections=self.config.http_max_keepalive_connections,
                ),
            )
        self._clients[key] = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=self._http_timeout,
            http_client=self._http_client,
            max_retries=self.config.openai_sdk_max_retries,
        )
        return self._clients[key]

    def _select_endpoint_runtime(self, avoid_endpoint_keys: set[str]) -> EndpointRuntime:
        if self._endpoint_pool is None:
            return self._static_runtime
        runtime = self._endpoint_pool.select_endpoint(avoid_keys=avoid_endpoint_keys)
        self._cleanup_deleted_endpoint_clients(self._endpoint_pool.active_base_urls)
        return runtime

    def _record_endpoint_success(self, endpoint_key: str) -> None:
        if self._endpoint_pool is None:
            self._static_runtime.fail_count = 0
            self._static_runtime.cooldown_until = 0.0
            return
        self._endpoint_pool.record_success(endpoint_key)

    def _record_endpoint_failure(self, endpoint_key: str, *, cooldown: bool) -> None:
        if self._endpoint_pool is None:
            return
        self._endpoint_pool.record_failure(endpoint_key, cooldown=cooldown)

    def _cleanup_deleted_endpoint_clients(self, active_base_urls: set[str]) -> None:
        for key in list(self._clients):
            base_url, _api_key = key
            if base_url not in active_base_urls:
                self._clients.pop(key, None)

    def _get_http_timeout(self):
        if self._http_timeout is not None:
            return self._http_timeout
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("httpx package is required for external judge calls") from exc
        self._http_timeout = httpx.Timeout(
            connect=self.config.http_connect_timeout,
            read=self.config.effective_http_read_timeout,
            write=self.config.http_write_timeout,
            pool=self.config.http_pool_timeout,
        )
        return self._http_timeout

    async def _close_async_resources(self) -> None:
        http_client = self._http_client
        self._http_client = None
        self._clients.clear()
        close = getattr(http_client, "aclose", None)
        if close is not None:
            await close()

    async def _sleep_before_retry(self, attempt: int) -> None:
        delay = min(self.config.retry_backoff_max, self.config.retry_backoff_base * (2**attempt))
        if self.config.retry_jitter:
            delay += random.uniform(0.0, self.config.retry_jitter * max(delay, 1.0))
        if delay > 0:
            await asyncio.sleep(delay)


def _transport_name(config: JudgeAPIConfig) -> str:
    if config.mode == "openai" and config.openai_use_responses_api:
        return "responses"
    return "chat_completions"


def _classify_judge_exception(exc: Exception) -> _RetryDecision:
    class_name = exc.__class__.__name__
    status_code = _status_code(exc)

    if isinstance(exc, NoEligibleEndpointError):
        return _RetryDecision(retry=False, cooldown_endpoint=False, status="judge_request_error")
    if isinstance(exc, asyncio.TimeoutError):
        return _RetryDecision(
            retry=True,
            cooldown_endpoint=True,
            status="judge_request_error",
            endpoint_timeout=True,
        )
    if class_name == "PoolTimeout":
        return _RetryDecision(
            retry=True,
            cooldown_endpoint=False,
            status="judge_request_error",
            pool_timeout=True,
        )
    if class_name in {"ConnectTimeout", "ReadTimeout", "WriteTimeout", "TimeoutException", "APITimeoutError"}:
        return _RetryDecision(
            retry=True,
            cooldown_endpoint=True,
            status="judge_request_error",
            endpoint_timeout=True,
        )
    if status_code in {400, 401, 403, 404, 422}:
        return _RetryDecision(retry=False, cooldown_endpoint=False, status="judge_request_error")
    if status_code == 429 or (status_code is not None and status_code >= 500):
        return _RetryDecision(retry=True, cooldown_endpoint=True, status="judge_request_error")
    if class_name in {
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "ConnectError",
        "ReadError",
        "WriteError",
        "RemoteProtocolError",
    }:
        return _RetryDecision(retry=True, cooldown_endpoint=True, status="judge_request_error")
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return _RetryDecision(retry=True, cooldown_endpoint=False, status="judge_request_error")
    return _RetryDecision(retry=False, cooldown_endpoint=False, status="judge_request_error")


def _status_code(exc: Exception) -> Optional[int]:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def parse_json_response(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("empty_judge_response")
    raw = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("judge_response_must_be_object")
    return payload


def get_judge_client(config: JudgeAPIConfig) -> MultimodalJudgeClient:




    return MultimodalJudgeClient(config)


def reset_judge_clients() -> None:
    _CLIENTS.clear()


def _extract_responses_text(response: Any) -> str:
    chunks: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        raise RuntimeError("causal_learner judge sync API cannot run inside an active event loop")
    return asyncio.run(coro)


_CLIENTS: Dict[str, MultimodalJudgeClient] = {}
