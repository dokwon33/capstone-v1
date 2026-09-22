"""Optional B-owned LangChain factory; config.py remains the model-name authority."""

from __future__ import annotations


def make_chat_model(*, model: str, temperature: float):
    """Use the team's installed provider integration and externally supplied credentials.

    SDK retries are disabled: LangGraph RetryPolicy owns the retry budget. Provider
    plugins differ, so a team-specific factory may replace this function through the
    B binding file, without changing this module or any common contract.
    """
    if not model.strip():
        raise ValueError("config.py model name must not be blank")
    from langchain.chat_models import init_chat_model

    return init_chat_model(model, temperature=temperature, max_retries=0)
