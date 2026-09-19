import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


# Production gateway config (used if no PORTKEY_CONFIG_ID 'pc-...' slug is provided):
#   - Fallback: primary @rag/llama-3.3-70b-versatile → @brag/llama-3.1-8b-instant on failure
#   - Cache: semantic mode (requires Portkey Enterprise — silently falls back to simple on free/starter)
#   - Retry: 2 attempts on rate limit / server error before triggering the fallback target
GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {
        "attempts": 2,
        "on_status_codes": [429, 500, 502, 503, 504]
    },
    "targets": [
        {"override_params": {"model": f"@{settings.GROQ_SLUG}/openai/gpt-oss-120b"}},
        {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/openai/gpt-oss-120b"}},
        {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/openai/gpt-oss-20b"}},
        {"override_params": {"model": f"@{settings.GROQ_SLUG}/openai/gpt-oss-20b"}},
    ]
}

# Use saved config slug (pc-...) if configured in .env, otherwise use inline dictionary config
ACTIVE_PORTKEY_CONFIG = settings.PORTKEY_CONFIG_ID if settings.PORTKEY_CONFIG_ID else GATEWAY_CONFIG

portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=ACTIVE_PORTKEY_CONFIG
)


def get_langchain_llm(feature: str = "rag", model_name: str | None = None) -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.
    """
    selected_model = model_name or "openai/gpt-oss-120b"
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG}/{selected_model}",
        temperature=0,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=ACTIVE_PORTKEY_CONFIG,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production"
            }
        )
    )

def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"