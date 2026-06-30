"""Swappable LLM client.

Groq, Gemini, and OpenRouter all expose an OpenAI-compatible chat-completions
endpoint, so a single thin wrapper around the `openai` SDK — parameterised by
base_url/api_key/model from env — covers all three. Switching providers is a
.env edit, not a code change (see .env.example for the alternatives).
"""
from __future__ import annotations

from openai import OpenAI

from src.config import Settings


class LLMClient:
    def __init__(self, settings: Settings):
        self._client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
        self._model = settings.llm_model

    def chat(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto"):
        kwargs = {}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        return self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            # Deterministic output for factual banking Q&A — same question
            # should give the same answer every time, not vary run to run.
            # Doesn't fix hallucination/underclaiming on its own (those are
            # separate, harder problems — see guardrails discussion), but
            # it's a free, standard baseline for this kind of task.
            temperature=0,
            **kwargs,
        )
