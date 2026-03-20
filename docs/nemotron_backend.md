# Nemotron LLM Backend (EmoAgent)
@ `tools/EmoAgent/llm_backend.py`

## What It Is

EmoAgent is now routed to the Nemotron models available on the BUZZ hackathon cluster
It exposes three symbols:


`get_client()` Returns a cached client configured for Nemotron
`DEFAULT_MODEL` Set at first call; reflects the best available model
`count_tokens(messages, response_text)` Approximate (input, output) token counts

- Nemotron exposes an **OpenAI-compatible REST API** at `$BUZZ_*_API/v1/chat/completions`.
- The `openai.OpenAI(base_url=..., api_key=...)` constructor redirects any `.chat.completions.create(...)` call to an arbitrary endpoint.
- Zero changes to call-site argument structure (`messages`, `temperature`, `max_tokens`, etc.) are required.

---

## How It Works

### Endpoint Resolution

Three Nemotron variants are tried in preference order:

```
120B FP8  ->  30B FP8  ->  30B BF16
```

Each requires two environment variables:
- `BUZZ_NEMOTRON_3_*_API` base URL (e.g. `https://nemotron.buzz...`)
- `BUZZ_NEMOTRON_3_*_AUTH_TOKEN` bearer token

`_resolve_endpoint()` iterates the list and returns the first pair where both variables are non-empty. This means you get the best available model automatically.

### Client Caching

```python
_client: OpenAI | None = None

def get_client() -> OpenAI:
    global _client, DEFAULT_MODEL
    if _client is None:
        base_url, api_key, DEFAULT_MODEL = _resolve_endpoint()
        _client = OpenAI(base_url=f"{base_url}/v1", api_key=api_key)
    return _client
```

The client is module-level singleton. Initialised once on first call; subsequent calls return the cached instance. `DEFAULT_MODEL` is set as a side effect so callers don't need to know which variant is available.

### Token Counting

`count_tokens()` tries `tiktoken` first (BPE encoding `cl100k_base` — a generic BPE that works acceptably for non-GPT models), and falls back to word count. This is used by EmoAgent's token-budget tracking. It is approximate: neither `cl100k_base` nor word count matches Nemotron's tokeniser exactly, but the error is small enough for budget heuristics.

---

## How to Use It

### In Any EmoAgent File

```python
from llm_backend import get_client, DEFAULT_MODEL

response = get_client().chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[{"role": "user", "content": "..."}],
    temperature=0.7,
    max_tokens=512,
)
```

### Environment Setup (`.env` or shell)

```bash
# Prefer 120B if available:
BUZZ_NEMOTRON_3_SUPER_120B_FP8_API=https://...
BUZZ_NEMOTRON_3_SUPER_120B_FP8_AUTH_TOKEN=REDACTED

# Fallback to 30B:
BUZZ_NEMOTRON_3_NANO_30B_FP8_API=https://...
BUZZ_NEMOTRON_3_NANO_30B_FP8_AUTH_TOKEN=REDACTED
```

Only one pair needs to be set. The others are optional.

---

## Sources

- **OpenAI Python SDK** `openai.OpenAI(base_url=..., api_key=...)` pattern for alternative endpoints:
  https://github.com/openai/openai-python#configuring-the-http-client
- **Nemotron-3 model family** NVIDIA Nemotron-3 Super 120B / Nano 30B:
  https://developer.nvidia.com/blog/nemotron-3
- **tiktoken** OpenAI BPE tokeniser, `cl100k_base` encoding:
  https://github.com/openai/tiktoken
