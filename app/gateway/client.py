import os
import logfire
from langchain_groq import ChatGroq
from app.config import settings

# Initialize Portkey / Groq Client
_portkey_available = bool(getattr(settings, "PORTKEY_API_KEY", None))

if _portkey_available:
    try:
        from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
        portkey_client = Portkey(
            api_key=settings.PORTKEY_API_KEY,
            virtual_key=getattr(settings, "GROQ_SLUG", "rag"),
        )
    except Exception as e:
        logfire.warning(f"Failed to initialize Portkey client: {e}. Falling back to Groq/OpenAI.")
        _portkey_available = False

if not _portkey_available:
    from openai import OpenAI
    
    class _GroqPortkeyWrapper:
        """Wrapper around OpenAI/Groq client to support default model parameter."""
        def __init__(self):
            self._client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=settings.GROQ_API_KEY or "missing_key",
            )
            self.chat = self._ChatWrapper(self._client)

        class _ChatWrapper:
            def __init__(self, client):
                self._client = client
                self.completions = self._CompletionsWrapper(client)

            class _CompletionsWrapper:
                def __init__(self, client):
                    self._client = client

                def create(self, **kwargs):
                    if "model" not in kwargs:
                        kwargs["model"] = getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile")
                    return self._client.chat.completions.create(**kwargs)

    portkey_client = _GroqPortkeyWrapper()


def get_langchain_llm(feature: str = "general"):
    """
    Returns a LangChain LLM instance routed through Portkey AI Gateway if available,
    or directly via ChatGroq as fallback.
    """
    if _portkey_available:
        try:
            from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
            from langchain_openai import ChatOpenAI
            
            headers = createHeaders(
                api_key=settings.PORTKEY_API_KEY,
                virtual_key=getattr(settings, "GROQ_SLUG", "rag"),
                metadata={
                    "feature": feature,
                    "_user": "rag-system",
                    "environment": "production"
                }
            )
            return ChatOpenAI(
                model=getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile"),
                base_url=PORTKEY_GATEWAY_URL,
                default_headers=headers,
                temperature=0.1
            )
        except Exception as e:
            logfire.warning(f"Portkey ChatOpenAI initialization failed: {e}. Using direct ChatGroq.")

    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.1
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
