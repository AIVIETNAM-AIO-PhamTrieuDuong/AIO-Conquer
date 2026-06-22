import json
import httpx
from app.core.config import settings


def _collect_sse(raw_text: str) -> str:
    """Extract assistant content from OpenAI-compatible responses.

    The provider may return either Server-Sent Events with delta chunks or a
    normal JSON response with message.content. Empty content is treated as a
    model response error so downstream state never receives a blank response.
    """
    try:
        response = json.loads(raw_text)
        message = response["choices"][0].get("message", {})
        content = message.get("content")
        if content:
            return content
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        pass

    result: list[str] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            message = choice.get("message", {})
            if content := delta.get("content") or message.get("content"):
                result.append(content)
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
    content = "".join(result)
    if content:
        return content
    raise ValueError("LLM response did not include assistant content.")


class NineRouterClient:
    """OpenAI-compatible client that talks to a 9Router instance via SSE."""

    def __init__(self):
        self._base_url = settings.ninerouter_url.rstrip("/")
        self._model = settings.ninerouter_model
        self._headers = {"Content-Type": "application/json"}
        if settings.ninerouter_key:
            self._headers["Authorization"] = f"Bearer {settings.ninerouter_key}"

    def _payload(self, prompt: str, temperature: float, max_tokens: int, json_mode: bool = False) -> dict:
        payload: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def _post(self, payload: dict) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()
            print(resp)
            return _collect_sse(resp.text)

    async def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
        """JSON-mode generation (replaces Gemini response_mime_type=application/json)."""
        return await self._post(self._payload(prompt, temperature, max_tokens, json_mode=True))

    async def generate_text(self, prompt: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        """Plain-text generation."""
        return await self._post(self._payload(prompt, temperature, max_tokens, json_mode=False))

    async def is_alive(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self._base_url}/models", headers=self._headers)
                return resp.status_code == 200
        except Exception:
            return False


llm = NineRouterClient()
