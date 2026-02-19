"""
Embedding model abstraction layer.
Provides a unified interface for generating text embeddings.
"""

from typing import List
from abc import ABC, abstractmethod
import google.generativeai as genai
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings, get_logger

logger = get_logger(__name__)


class BaseEmbedding(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass


class GeminiEmbedding(BaseEmbedding):
    """Google Gemini embedding provider."""
    
    def __init__(self):
        """Initialize Gemini embedding client."""
        genai.configure(api_key=settings.google_api_key or "")
        self.model_name = settings.gemini_embedding_model
        self._dimension = settings.qdrant_vector_size
        
        logger.info(f"Initialized Gemini Embedding with model: {self.model_name}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document",
            )
            return result["embedding"]
            
        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        try:
            # Gemini supports batch embedding
            result = genai.embed_content(
                model=self.model_name,
                content=texts,
                task_type="retrieval_document",
            )
            return result["embedding"]
            
        except Exception as e:
            logger.error(f"Gemini batch embedding error: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding provider."""
    
    def __init__(self):
        """Initialize OpenAI embedding client."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key or "")
        self.model_name = settings.openai_embedding_model
        self._dimension = settings.qdrant_vector_size
        
        logger.info(f"Initialized OpenAI Embedding with model: {self.model_name}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text,
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
            return [item.embedding for item in response.data]
            
        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


def get_embedding_model() -> BaseEmbedding:
    """
    Get the configured embedding model.
    
    Returns:
        Embedding model instance based on settings
    """
    provider = settings.embedding_provider
    if provider == "gemini":
        return GeminiEmbedding()
    elif provider == "openai":
        return OpenAIEmbedding()
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


# Global embedding model instance — lazy to avoid import-time API errors
_embedding_model: BaseEmbedding | None = None


def embedding_model() -> BaseEmbedding:
    """Return (or lazily create) the global embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = get_embedding_model()
    return _embedding_model
