from __future__ import annotations

from typing import Any

import httpx
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings


class OpenAIClient:
    """LangChain ChatOpenAI client for OpenAI-compatible chat providers."""

    def __init__(self) -> None:
        """Initialize the reusable ChatOpenAI wrapper from application settings."""
        kwargs: dict[str, Any] = {
            "model": settings.ninerouter_model,
            "base_url": settings.ninerouter_url,
            "timeout": 120,
        }
        if settings.ninerouter_key:
            kwargs["api_key"] = settings.ninerouter_key
        self._client = ChatOpenAI(**kwargs)

    @staticmethod
    def _content_from_message(message: AIMessage) -> str:
        """Return string content from a LangChain AI message.

        Raises:
            ValueError: If the model response does not contain text content.
        """
        content = message.content
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            ]
            text = "".join(text_parts)
            if text:
                return text
        raise ValueError("LLM response did not include assistant content.")

    async def _invoke(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        """Invoke ChatOpenAI and return assistant text content."""
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        message = await self._client.ainvoke(prompt, **kwargs)
        return self._content_from_message(message)

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a JSON-mode response for the QA pipeline."""
        return await self._invoke(
            prompt,
            temperature,
            max_tokens,
            json_mode=True,
        )

    async def generate_text(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a plain-text response for report enrichment."""
        return await self._invoke(
            prompt,
            temperature,
            max_tokens,
            json_mode=False,
        )

    async def is_alive(self) -> bool:
        """Return whether the configured OpenAI-compatible endpoint responds."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{settings.ninerouter_url.rstrip('/')}/models",
                    headers=self._headers,
                )
                return response.status_code == 200
        except Exception:
            return False

    @property
    def _headers(self) -> dict[str, str]:
        """Build headers for endpoint health checks without exposing secrets."""
        headers = {"Content-Type": "application/json"}
        if settings.ninerouter_key:
            headers["Authorization"] = f"Bearer {settings.ninerouter_key}"
        return headers


llm = OpenAIClient()
