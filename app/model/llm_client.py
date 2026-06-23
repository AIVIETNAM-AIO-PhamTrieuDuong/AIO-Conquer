import json
import httpx
from app.core.config import settings


def _collect_sse(raw_text: str) -> str:
    """Collect content chunks from a true SSE stream into a single string.

    Each line looks like:  data: {...}  or  data: [DONE]
    We concatenate every delta.content token in order.
    """
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
            delta = chunk["choices"][0].get("delta", {})
            if content := delta.get("content"):
                result.append(content)
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
    return "".join(result)


def _parse_response(raw_text: str) -> str:
    """Extract assistant text from a 9Router response.

    The router replies with a single JSON completion object (``message.content``)
    followed by a trailing ``data: [DONE]`` marker, but may also fall back to a
    real SSE stream of ``delta.content`` chunks. Handle both shapes.
    """
    text = raw_text.strip()

    # Strip the trailing SSE done marker if the body is otherwise plain JSON.
    done_marker = "data: [DONE]"
    if text.endswith(done_marker):
        text = text[: -len(done_marker)].strip()

    # Case 1 — single JSON completion object.
    try:
        obj = json.loads(text)
        choices = obj.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            if content := message.get("content"):
                return content
            delta = choices[0].get("delta") or {}
            if content := delta.get("content"):
                return content
    except json.JSONDecodeError:
        pass

    # Case 2 — true SSE stream of delta chunks.
    return _collect_sse(raw_text)


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
            return _parse_response(resp.text)

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
