"""
LLM and Embeddings wrappers for OpenAI and Gemini.
"""

import asyncio
from typing import List, Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings, get_logger

logger = get_logger(__name__)


# ===========================================================================
# LLM Providers
# ===========================================================================

class GeminiLLM:
    """Wraps Google Gemini so it can be called the same way as OpenAI."""

    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=settings.google_api_key or "")
        self.model_name = settings.gemini_model
        self.temperature = settings.gemini_temperature
        self.max_tokens = settings.gemini_max_tokens
        self._genai = genai
        logger.info(f"Initialized Gemini LLM: {self.model_name}")

    def _run_sync(self, prompt: str, system_prompt: Optional[str], temperature: float, max_tokens: int) -> str:
        """Calls the Gemini SDK synchronously (run in a thread to avoid blocking)."""
        config = self._genai.types.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        model = self._genai.GenerativeModel(model_name=self.model_name, generation_config=config)
        return model.generate_content(full_prompt).text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def ask(self, prompt: str, system_prompt: Optional[str] = None,
                  temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                  response_format: Optional[dict] = None) -> str:
        try:
            return await asyncio.to_thread(
                self._run_sync, prompt, system_prompt,
                temperature or self.temperature, max_tokens or self.max_tokens
            )
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            raise


class OpenAILLM:
    """Wraps OpenAI chat completions."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key or "")
        self.model_name = settings.openai_model
        self.temperature = settings.openai_temperature
        self.max_tokens = settings.openai_max_tokens
        logger.info(f"Initialized OpenAI LLM: {self.model_name}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def ask(self, prompt: str, system_prompt: Optional[str] = None,
                  temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                  response_format: Optional[dict] = None) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = await self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            raise


# Shared LLM instance — created once when first needed
_llm = None

def get_llm():
    """Returns the shared LLM. Picks provider from config and creates it on first call."""
    global _llm
    if _llm is None:
        if settings.llm_provider == "gemini":
            _llm = GeminiLLM()
        elif settings.llm_provider == "openai":
            _llm = OpenAILLM()
        else:
            raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
    return _llm


# ===========================================================================
# Embedding Providers
# ===========================================================================

class GeminiEmbedding:
    """Google Gemini embedding provider."""

    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=settings.google_api_key or "")
        self.model_name = settings.gemini_embedding_model
        self.dimension = settings.qdrant_vector_size
        logger.info(f"Initialized Gemini Embedding: {self.model_name}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def embed(self, text: str) -> List[float]:
        import google.generativeai as genai
        try:
            result = genai.embed_content(model=self.model_name, content=text, task_type="retrieval_document")
            return result["embedding"]
        except Exception as e:
            logger.error(f"Gemini embed error: {e}")
            raise

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        import google.generativeai as genai
        try:
            result = genai.embed_content(model=self.model_name, content=texts, task_type="retrieval_document")
            return result["embedding"]
        except Exception as e:
            logger.error(f"Gemini embed_many error: {e}")
            raise


class OpenAIEmbedding:
    """OpenAI embedding provider."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key or "")
        self.model_name = settings.openai_embedding_model
        self.dimension = settings.qdrant_vector_size
        logger.info(f"Initialized OpenAI Embedding: {self.model_name}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def embed(self, text: str) -> List[float]:
        try:
            response = await self.client.embeddings.create(model=self.model_name, input=text)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embed error: {e}")
            raise

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        try:
            response = await self.client.embeddings.create(model=self.model_name, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI embed_many error: {e}")
            raise


# Shared embedding instance — created once when first needed
_embedding_model = None

def get_embedding_model():
    """Returns the shared embedding model. Picks provider from config on first call."""
    global _embedding_model
    if _embedding_model is None:
        if settings.embedding_provider == "gemini":
            _embedding_model = GeminiEmbedding()
        elif settings.embedding_provider == "openai":
            _embedding_model = OpenAIEmbedding()
        else:
            raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")
    return _embedding_model
