"""Nemotron provider implementation."""

import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

from .base import BaseLLMProvider, LLMMessage, LLMResponse

load_dotenv()

DEFAULT_BUZZ_NEMOTRON_API = "https://example.invalid"
DEFAULT_NEMOTRON_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"

_NEMOTRON_ENDPOINTS = (
    (
        "BUZZ_NEMOTRON_3_NANO_30B_FP8_API",
        "BUZZ_NEMOTRON_3_NANO_30B_FP8_AUTH_TOKEN",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    ),
    (
        "BUZZ_NEMOTRON_3_SUPER_120B_FP8_API",
        "BUZZ_NEMOTRON_3_SUPER_120B_FP8_AUTH_TOKEN",
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
    ),
    (
        "BUZZ_NEMOTRON_3_NANO_30B_BF16_API",
        "BUZZ_NEMOTRON_3_NANO_30B_BF16_AUTH_TOKEN",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    ),
)


def _normalize_model_name(model: Optional[str]) -> str:
    return (model or "").strip().lower()


def _resolve_connection_from_env(model_hint: Optional[str] = None) -> Tuple[str, str, str]:
    """
    Resolve Nemotron API URL, auth token, and model from existing .env keys.

    If a model is provided, select API/token for that exact model.
    Otherwise use the first configured endpoint where both API and token are present.
    """
    normalized_hint = _normalize_model_name(model_hint)
    if normalized_hint:
        for api_env, token_env, default_model in _NEMOTRON_ENDPOINTS:
            if _normalize_model_name(default_model) == normalized_hint:
                api_value = os.getenv(api_env).strip().rstrip("/")
                token_value = (os.getenv(token_env) or "").strip()
                return api_value, token_value, default_model

    for api_env, token_env, default_model in _NEMOTRON_ENDPOINTS:
        api_value = (os.getenv(api_env) or "").strip()
        token_value = (os.getenv(token_env) or "").strip()
        if api_value and token_value:
            return api_value.rstrip("/"), token_value, default_model
        if api_value:
            return api_value.rstrip("/"), token_value, default_model

    raise ValueError(f"No Nemotron API found for model: {model_hint}")


class NemotronProvider(BaseLLMProvider):
    """Provider for Nemotron via OpenAI-compatible /v1/chat/completions API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = 0.0,
        max_tokens: int = 1000,
        verify_ssl: bool = False,
        timeout_s: float = 60.0,
        **kwargs
    ):
        env_base_url, env_api_key, env_model = _resolve_connection_from_env(model)

        base_url = (base_url or env_base_url).strip().rstrip("/")
        model = model or env_model
        api_key = api_key or env_api_key
        if not api_key:
            raise ValueError(
                "Nemotron api_key must be set via api_key or one of: "
                "BUZZ_NEMOTRON_3_NANO_30B_FP8_AUTH_TOKEN, "
                "BUZZ_NEMOTRON_3_SUPER_120B_FP8_AUTH_TOKEN, "
                "BUZZ_NEMOTRON_3_NANO_30B_BF16_AUTH_TOKEN."
            )

        super().__init__(api_key, model, temperature, max_tokens, **kwargs)

        self.base_url = base_url
        self.endpoint_url = f"{self.base_url}/v1/chat/completions"
        self.system_prompt = system_prompt or ""

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = httpx.Timeout(timeout_s, connect=10.0)
        self.sync_client = httpx.Client(headers=headers, verify=verify_ssl, timeout=timeout)
        self.async_client = httpx.AsyncClient(headers=headers, verify=verify_ssl, timeout=timeout)

    def close(self) -> None:
        self.sync_client.close()

    async def aclose(self) -> None:
        await self.async_client.aclose()

    def _messages_to_api(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        api = [{"role": msg.role, "content": msg.content or ""} for msg in messages]
        if self.system_prompt and (not api or api[0].get("role") != "system"):
            api = [{"role": "system", "content": self.system_prompt}] + api
        return api

    def _prepare_request_payload(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._messages_to_api(messages),
            "max_tokens": self._get_max_tokens(max_tokens),
        }
        temp_value = self._get_temperature(temperature)
        if temp_value is not None:
            payload["temperature"] = temp_value
        payload.update(self.extra_params)
        payload.update(kwargs)
        return payload

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        if "error" in data:
            raise RuntimeError(f"LLM error: {data['error']}")

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Unexpected response format (no choices): {data}")

        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = (message.get("content") or "").strip()

        usage = data.get("usage")
        if usage:
            usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        return LLMResponse(
            content=content,
            model=self.model,
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            metadata={"response_id": data.get("id")},
        )

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        payload = self._prepare_request_payload(messages, temperature, max_tokens, **kwargs)

        resp = await self.async_client.post(self.endpoint_url, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}") from e

        return self._parse_response(resp.json())

    def generate_sync(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        payload = self._prepare_request_payload(messages, temperature, max_tokens, **kwargs)

        resp = self.sync_client.post(self.endpoint_url, json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}") from e

        return self._parse_response(resp.json())
