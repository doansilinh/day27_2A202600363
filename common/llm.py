"""LLM factory. Returns an OpenAI-compatible chat model."""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env, or set OPENROUTER_API_KEY "
            "if you want to use OpenRouter."
        )

    kwargs = {
        "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        "api_key": api_key,
        "temperature": temperature,
    }
    base_url = os.environ.get("LLM_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)
