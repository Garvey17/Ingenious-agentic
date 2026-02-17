"""
LLM provider abstraction layer.
Provides a unified interface for different LLM providers (Gemini, OpenAI).
"""

from typing import Any, AsyncIterator, Optional
from abc import ABC, abstractmethod
import google.generativeai as genai
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


class GeminiLLM(BaseLLM):
    """Google Gemini LLM provider."""
    
    def __init__(self):
        """Initialize Gemini client."""
        genai.configure(api_key=settings.google_api_key)
        self.model_name = settings.llm_model
        self.default_temperature = settings.llm_temperature
        self.default_max_tokens = settings.llm_max_tokens
        
        logger.info(f"Initialized Gemini LLM with model: {self.model_name}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """
        Generate a completion from Gemini.
        
        Args:
            prompt: User prompt
            system_prompt: System instruction
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Response format specification (e.g., JSON schema)
            
        Returns:
            Generated text
        """
        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt,
            )
            
            generation_config = {
                "temperature": temperature or self.default_temperature,
                "max_output_tokens": max_tokens or self.default_max_tokens,
            }
            
            # Add JSON mode if response format is specified
            if response_format:
                generation_config["response_mime_type"] = "application/json"
            
            response = await model.generate_content_async(
                prompt,
                generation_config=generation_config,
            )
            
            return response.text
            
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
        """
        Generate a streaming completion from Gemini.
        
        Args:
            prompt: User prompt
            system_prompt: System instruction
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            Text chunks
        """
        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt,
            )
            
            generation_config = {
                "temperature": temperature or self.default_temperature,
                "max_output_tokens": max_tokens or self.default_max_tokens,
            }
            
            response = await model.generate_content_async(
                prompt,
                generation_config=generation_config,
                stream=True,
            )
            
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider."""
    
    def __init__(self):
        """Initialize OpenAI client."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model_name = settings.llm_model
        self.default_temperature = settings.llm_temperature
        self.default_max_tokens = settings.llm_max_tokens
        
        logger.info(f"Initialized OpenAI LLM with model: {self.model_name}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """
        Generate a completion from OpenAI.
        
        Args:
            prompt: User prompt
            system_prompt: System instruction
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Response format specification
            
        Returns:
            Generated text
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            kwargs = {
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
        """
        Generate a streaming completion from OpenAI.
        
        Args:
            prompt: User prompt
            system_prompt: System instruction
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            Text chunks
        """
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


def get_llm() -> BaseLLM:
    """
    Get the configured LLM provider.
    
    Returns:
        LLM instance based on settings
    """
    if settings.llm_provider == "gemini":
        return GeminiLLM()
    elif settings.llm_provider == "openai":
        return OpenAILLM()
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


# Global LLM instance
llm = get_llm()
