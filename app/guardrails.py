import os
import logfire

_rails = None

def initialize_rails():
    """
    Initialize NeMo Guardrails or safety layer.
    """
    global _rails
    try:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "guardrails")
        if os.path.exists(config_path):
            from nemoguardrails import RailsConfig, LLMRails
            config = RailsConfig.from_path(config_path)
            _rails = LLMRails(config)
            logfire.info("🛡️ NeMo Guardrails initialized from config.")
        else:
            logfire.info("🛡️ Guardrails initialized with standard safety filters.")
    except Exception as e:
        logfire.warning(f"Guardrails initialization warning: {e}")
        _rails = None


def guard(query: str) -> tuple[bool, str]:
    """
    Evaluates input query against guardrails.
    Returns: (is_blocked: bool, response_message: str)
    """
    if not query or not query.strip():
        return True, "Please provide a valid, non-empty query."

    # If NeMo LLMRails is active and configured
    if _rails is not None:
        try:
            response = _rails.generate(messages=[{"role": "user", "content": query}])
            content = response.get("content", "") if isinstance(response, dict) else str(response)
            if "I cannot" in content or "not allowed" in content:
                return True, content
        except Exception as e:
            logfire.warning(f"NeMo rails check error: {e}")

    # Standard fast heuristics for harmful / prompt injection patterns
    lowered = query.lower()
    blocked_keywords = [
        "ignore previous instructions",
        "system prompt override",
        "drop table",
        "<script>",
    ]
    for pattern in blocked_keywords:
        if pattern in lowered:
            return True, "I cannot fulfill this request as it violates safety guidelines."

    return False, ""
