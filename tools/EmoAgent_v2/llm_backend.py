"""
Nemotron backend for EmoAgent.

Drop-in replacement for the OpenAI client. Returns an openai.OpenAI instance
pointed at Nemotron's OpenAI-compatible endpoint.

Usage in any EmoAgent file:
    from llm_backend import get_client, DEFAULT_MODEL
    client = get_client()
    response = client.chat.completions.create(model=DEFAULT_MODEL, messages=...)
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Preference order: largest model first, fallback to smaller
_ENDPOINTS = [
    (
        "BUZZ_NEMOTRON_3_SUPER_120B_FP8_API",
        "BUZZ_NEMOTRON_3_SUPER_120B_FP8_AUTH_TOKEN",
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
    ),
    (
        "BUZZ_NEMOTRON_3_NANO_30B_FP8_API",
        "BUZZ_NEMOTRON_3_NANO_30B_FP8_AUTH_TOKEN",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    ),
    (
        "BUZZ_NEMOTRON_3_NANO_30B_BF16_API",
        "BUZZ_NEMOTRON_3_NANO_30B_BF16_AUTH_TOKEN",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    ),
]

_client: OpenAI | None = None
DEFAULT_MODEL: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"  # updated on first call


def _resolve_endpoint() -> tuple[str, str, str]:
    for api_env, token_env, model in _ENDPOINTS:
        api = (os.getenv(api_env) or "").strip().rstrip("/")
        token = (os.getenv(token_env) or "").strip()
        if api and token:
            return api, token, model
    raise ValueError(
        "No Nemotron credentials found in environment. "
        "Set BUZZ_NEMOTRON_3_NANO_30B_FP8_API + BUZZ_NEMOTRON_3_NANO_30B_FP8_AUTH_TOKEN "
        "(or SUPER_120B variants) in your .env file."
    )


def get_client() -> OpenAI:
    """Return a cached OpenAI client configured for Nemotron."""
    global _client, DEFAULT_MODEL
    if _client is None:
        base_url, api_key, DEFAULT_MODEL = _resolve_endpoint()
        _client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key=api_key,
        )
    return _client


def count_tokens(messages: list[dict], response_text: str) -> tuple[int, int]:
    """
    Approximate token counts. Uses tiktoken if available, falls back to word count.
    Returns (input_tokens, output_tokens).
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")  # generic BPE, works for most models
        input_tokens = sum(len(enc.encode(m.get("content", ""))) for m in messages)
        output_tokens = len(enc.encode(response_text))
        return input_tokens, output_tokens
    except Exception:
        # word-count fallback
        input_tokens = sum(len((m.get("content") or "").split()) for m in messages)
        output_tokens = len(response_text.split())
        return input_tokens, output_tokens
