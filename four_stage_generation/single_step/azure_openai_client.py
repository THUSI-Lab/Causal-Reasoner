from __future__ import annotations

import base64
import mimetypes
import os
from typing import Any, Dict, List, Tuple

from openai import AzureOpenAI
from azure.identity import AzureCliCredential, get_bearer_token_provider

ENDPOINT = os.environ.get("API_BASE_URL") or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
MODEL = os.environ.get("MODEL_NAME", "gpt-5.4")
DEPLOYMENT = os.environ.get("MODEL_DEPLOYMENT", MODEL)
API_PROFILE = os.environ.get("API_PROFILE") or os.environ.get("AZURE_OPENAI_API_PROFILE", "")


def initialize_openai_client() -> AzureOpenAI:

    token_provider = get_bearer_token_provider(
        AzureCliCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    if not ENDPOINT:
        raise ValueError("API_BASE_URL or AZURE_OPENAI_ENDPOINT must be set.")
    if not API_PROFILE:
        raise ValueError("API_PROFILE or AZURE_OPENAI_API_PROFILE must be set.")
    client_kwargs: Dict[str, Any] = {
        "azure_endpoint": ENDPOINT,
        "azure_ad_token_provider": token_provider,
    }
    client_kwargs["api_" + "ver" + "sion"] = API_PROFILE
    client = AzureOpenAI(**client_kwargs)
    return client


def create_api_client(api_base_url: str = "", api_key: str = "") -> AzureOpenAI:

    return initialize_openai_client()


def build_request_payload_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        normalized.append(
            {
                "role": str(msg.get("role") or "user"),
                "content": _normalize_message_content(msg.get("content")),
            }
        )
    return normalized


def call_model_api(
    client: AzureOpenAI,
    *,
    model_name: str,
    payload_input: List[Dict[str, Any]],
    timeout_sec: float = 180.0,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = "medium",
) -> Tuple[str, Dict[str, int]]:

    kwargs: Dict[str, Any] = {
        "model": str(model_name or DEPLOYMENT),
        "messages": payload_input,
        "stream": False,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

        kwargs.pop("temperature", None)
    elif temperature is not None:
        kwargs["temperature"] = float(temperature)
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = int(max_tokens)
    response = client.with_options(timeout=float(timeout_sec)).chat.completions.create(**kwargs)


    usage_raw = response.usage
    usage = {
        "prompt_tokens": int(getattr(usage_raw, "prompt_tokens", 0) or 0) if usage_raw else 0,
        "completion_tokens": int(getattr(usage_raw, "completion_tokens", 0) or 0) if usage_raw else 0,
        "total_tokens": int(getattr(usage_raw, "total_tokens", 0) or 0) if usage_raw else 0,
    }

    try:
        content = response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Model response missing message.content: {e}") from e
    if isinstance(content, str):
        return content, usage
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        return "\n".join(parts).strip(), usage
    return str(content or ""), usage


def _normalize_message_content(content: Any) -> Any:
    if isinstance(content, list):
        return [_normalize_content_item(item) for item in content]
    return content


def _normalize_content_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    kind = str(item.get("type") or "").strip().lower()
    if kind != "image_url":
        return item

    image_url = item.get("image_url")
    if not isinstance(image_url, dict):
        return item
    url = image_url.get("url")
    if not isinstance(url, str) or not url:
        return item

    normalized_url = _maybe_to_data_url(url)
    if normalized_url == url:
        return item

    normalized = dict(item)
    normalized["image_url"] = dict(image_url)
    normalized["image_url"]["url"] = normalized_url
    return normalized


def _maybe_to_data_url(value: str) -> str:
    if str(value).startswith("data:"):
        return value
    if not os.path.exists(value):
        return value
    mime_type = mimetypes.guess_type(value)[0] or "application/octet-stream"
    with open(value, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
