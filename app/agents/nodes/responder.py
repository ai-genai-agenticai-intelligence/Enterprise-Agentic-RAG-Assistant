import logfire
from app.config import settings
from app.agents.state import AgentState
from app.gateway import portkey_client, extract_cache_status


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("✍️ LLM Synthesis"):
        # Multi-tiered fallback strategy:
        # Tier 1: Portkey Primary Key with 120b (@rag/openai/gpt-oss-120b)
        # Tier 2: Portkey Fallback Key with 120b (@brag/openai/gpt-oss-120b)
        # Tier 3: Portkey Fallback Key with 20b (@brag/openai/gpt-oss-20b)
        # Tier 4: Direct Groq Fallback Client
        candidate_models = [
            f"@{settings.GROQ_SLUG}/openai/gpt-oss-120b",
            f"@{settings.GROQ_SLUG_2}/openai/gpt-oss-120b",
            f"@{settings.GROQ_SLUG_2}/openai/gpt-oss-20b",
            f"@{settings.GROQ_SLUG}/openai/gpt-oss-20b",
        ]

        response = None
        content = None
        is_cache_hit = False
        last_error = None

        # 1. Try Portkey with progressive fallbacks
        for target_model in candidate_models:
            try:
                logfire.info(f"Attempting generation via Portkey with model: {target_model}")
                response = portkey_client.chat.completions.create(
                    model=target_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                content = response.choices[0].message.content
                cache_status = extract_cache_status(response)
                is_cache_hit = cache_status == "HIT"
                break
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "429" in err_str or "rate_limit" in err_str:
                    logfire.warning(f"⚠️ Portkey rate limit hit on {target_model}: {e}. Trying fallback target...")
                else:
                    logfire.warning(f"⚠️ Portkey error on {target_model}: {e}. Trying fallback target...")

        # 2. If Portkey failed on all targets, fallback directly to Groq client
        if content is None:
            fallback_keys = [k for k in [settings.GROQ_FALLBACK_API_KEY, settings.GROQ_API_KEY] if k]
            for groq_key in fallback_keys:
                for direct_model in ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
                    try:
                        logfire.info(f"Attempting direct Groq fallback with model: {direct_model}")
                        from groq import Groq
                        direct_client = Groq(api_key=groq_key)
                        res = direct_client.chat.completions.create(
                            model=direct_model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.1
                        )
                        content = res.choices[0].message.content
                        logfire.info(f"✅ Recovered successfully via direct Groq fallback ({direct_model}).")
                        break
                    except Exception as ge:
                        last_error = ge
                        logfire.warning(f"⚠️ Direct Groq failed on {direct_model}: {ge}")
                if content is not None:
                    break

        if content is None:
            err_msg = str(last_error) if last_error else "All LLM generation attempts failed."
            logfire.error(f"❌ All LLM generation attempts failed: {err_msg}")
            fallback_answer = "I apologize, but I am currently unable to generate a response due to high traffic on our upstream AI providers. Please try again in a few moments."
            return {
                "final_answer": fallback_answer,
                "status": "Service temporarily degraded.",
                "plan": state.get("plan", []) + ["Generation Failed (Rate Limit)"],
                "messages": [{"role": "assistant", "content": fallback_answer}]
            }

        if is_cache_hit:
            logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
            plan_update = state["plan"] + ["Cache: Hit ⚡"]
            status = "Cache hit — instant response."
        else:
            logfire.info("✅ Response synthesised via LLM.")
            plan_update = state["plan"]
            status = "Response generated."

        return {
            "final_answer": content,
            "status": status,
            "plan": plan_update,
            "messages": [{"role": "assistant", "content": content}]
        }
