from google import genai
from google.genai import types
from app.core.config import settings


class GeminiClient:
    def __init__(self):
        self._client = genai.Client(api_key=settings.google_api_key)
        self._model = settings.gemini_model

    async def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )
        return response.text

    async def generate_text(self, prompt: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text

    async def is_alive(self) -> bool:
        try:
            await self._client.aio.models.list()
            return True
        except Exception:
            return False


llm = GeminiClient()
