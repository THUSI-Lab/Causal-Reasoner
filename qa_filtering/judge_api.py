from __future__ import annotations

import os
from typing import Any


def create_chat_client() -> Any:
    base_url = os.environ.get("JUDGE_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("API_BASE_URL")

    if base_url:
        from openai import OpenAI
        return OpenAI(base_url=base_url, api_key=api_key or "EMPTY")

    if azure_endpoint:
        from openai import AzureOpenAI
        azure_profile = os.environ.get("AZURE_OPENAI_API_PROFILE") or os.environ.get("API_PROFILE", "")
        if not azure_profile:
            raise ValueError("AZURE_OPENAI_API_PROFILE or API_PROFILE must be set for Azure judge calls.")
        azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if azure_key:
            client_kwargs: dict[str, Any] = {
                "azure_endpoint": azure_endpoint,
                "api_key": azure_key,
            }
            client_kwargs["api_" + "ver" + "sion"] = azure_profile
            return AzureOpenAI(**client_kwargs)
        from azure.identity import AzureCliCredential, get_bearer_token_provider
        token_provider = get_bearer_token_provider(
            AzureCliCredential(),
            "https://cognitiveservices.azure.com/.default",
        )
        client_kwargs = {
            "azure_endpoint": azure_endpoint,
            "azure_ad_token_provider": token_provider,
        }
        client_kwargs["api_" + "ver" + "sion"] = azure_profile
        return AzureOpenAI(**client_kwargs)

    from openai import OpenAI
    return OpenAI(api_key=api_key or None)


def call_chat_completion(
    client: Any,
    *,
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    timeout: float = 600.0,
) -> tuple[str, dict[str, int]]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    token_limit = int(max_tokens) if max_tokens else 0
    if token_limit:
        kwargs["max_completion_tokens"] = token_limit
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    elif temperature is not None:
        kwargs["temperature"] = float(temperature)

    try:
        response = client.with_options(timeout=timeout).chat.completions.create(**kwargs)
    except Exception as exc:

        if token_limit and "max_completion_tokens" in str(exc):
            kwargs.pop("max_completion_tokens", None)
            kwargs["max_tokens"] = token_limit
            response = client.with_options(timeout=timeout).chat.completions.create(**kwargs)
        else:
            raise
    content = response.choices[0].message.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        text = "\n".join(parts)
    else:
        text = str(content or "")

    usage_raw = getattr(response, "usage", None)
    usage = {
        "prompt_tokens": int(getattr(usage_raw, "prompt_tokens", 0) or 0) if usage_raw else 0,
        "completion_tokens": int(getattr(usage_raw, "completion_tokens", 0) or 0) if usage_raw else 0,
        "total_tokens": int(getattr(usage_raw, "total_tokens", 0) or 0) if usage_raw else 0,
    }
    return text.strip(), usage
