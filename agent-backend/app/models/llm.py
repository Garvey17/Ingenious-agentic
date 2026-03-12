"""
LLM provider abstraction layer.
GeminiLLM uses the new google-genai 1.x SDK.
OpenAILLM uses the openai 2.x SDK.
"""

from typing import Any, AsyncIterator, Optional
from abc import ABC, abstractmethod
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings, get_logger

logger = get_logger(__name__)


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """Generate a completion from the LLM."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming completion from the LLM."""
        pass


# ---------------------------------------------------------------------------
# Gemini — stable google-generativeai sync SDK, wrapped in asyncio.to_thread
# (avoids google-genai 1.x aiohttp transport bugs on Windows)
# ---------------------------------------------------------------------------
class GeminiLLM(BaseLLM):
    """Google Gemini LLM provider using the stable google-generativeai SDK."""

    def __init__(self):
        import google.generativeai as genai

        api_key = settings.google_api_key or ""
        genai.configure(api_key=api_key)

        self.model_name = settings.llm_model or settings.gemini_model
        self.default_temperature = settings.llm_temperature or settings.gemini_temperature
        self.default_max_tokens = settings.llm_max_tokens or settings.gemini_max_tokens
        self._genai = genai

        logger.info(f"Initialized Gemini LLM with model: {self.model_name}")

    def _sync_generate(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        """Synchronous Gemini call — run via asyncio.to_thread."""
        import google.generativeai as genai

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Build the full prompt (older SDK doesn't have a dedicated system role)
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
        )
        response = model.generate_content(full_prompt)
        return response.text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """Generate a completion from Gemini (runs sync SDK in thread pool)."""
        import asyncio

        try:
            temp = temperature if temperature is not None else self.default_temperature
            tokens = max_tokens if max_tokens is not None else self.default_max_tokens
            json_mode = bool(response_format and response_format.get("type") == "json_object")

            result = await asyncio.to_thread(
                self._sync_generate, prompt, system_prompt, temp, tokens, json_mode
            )
            return result
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Streaming — falls back to single generate for simplicity."""
        result = await self.generate(prompt, system_prompt, temperature, max_tokens)
        yield result


# ---------------------------------------------------------------------------
# OpenAI — openai 2.x SDK
# ---------------------------------------------------------------------------

class OpenAILLM(BaseLLM):
    """OpenAI LLM provider."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key or "")
        self.model_name = settings.llm_model or settings.openai_model
        self.default_temperature = settings.llm_temperature or settings.openai_temperature
        self.default_max_tokens = settings.llm_max_tokens or settings.openai_max_tokens

        logger.info(f"Initialized OpenAI LLM with model: {self.model_name}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """Generate a completion from OpenAI."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs: dict = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature or self.default_temperature,
                "max_tokens": max_tokens or self.default_max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = await self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming completion from OpenAI."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_llm() -> BaseLLM:
    """Get the configured LLM provider."""
    if settings.llm_provider == "gemini":
        return GeminiLLM()
    elif settings.llm_provider == "openai":
        return OpenAILLM()
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


# Lazy global LLM instance
_llm: BaseLLM | None = None


def llm() -> BaseLLM:
    """Return (or lazily create) the global LLM instance."""
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm
