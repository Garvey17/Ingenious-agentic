"""
Configuration & Logging module for the Deep Research Desk.
Consolidates all app settings and basic logging setups.
"""

import logging
import os
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application Settings
    app_name: str = Field(default="Deep Research Desk", alias="APP_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", alias="ENVIRONMENT"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    
    # API Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    
    # LLM Provider Configuration (Default to OpenAI as requested)
    llm_provider: Literal["openai", "gemini"] = Field(
        default="openai", alias="LLM_PROVIDER"
    )
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", alias="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.7, alias="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=4096, alias="OPENAI_MAX_TOKENS")
    
    # Google Gemini Configuration
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-pro", alias="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.7, alias="GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(default=4096, alias="GEMINI_MAX_TOKENS")
    

    # Embedding Models
    embedding_provider: Literal["openai", "gemini"] = Field(
        default="openai", alias="EMBEDDING_PROVIDER"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )
    gemini_embedding_model: str = Field(
        default="models/embedding-001", alias="GEMINI_EMBEDDING_MODEL"
    )
    
    # Web Search Tool
    search_provider: Literal["tavily", "serper", "placeholder"] = Field(
        default="tavily", alias="SEARCH_PROVIDER"
    )
    tavily_api_key: Optional[str] = Field(default=None, alias="TAVILY_API_KEY")
    
    # Vector Database (Qdrant)
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(
        default="research_memory", alias="QDRANT_COLLECTION_NAME"
    )
    qdrant_vector_size: int = Field(default=1536, alias="QDRANT_VECTOR_SIZE")
    
    # Memory Configuration
    enable_memory: bool = Field(default=False, alias="ENABLE_MEMORY")
    memory_top_k: int = Field(default=5, alias="MEMORY_TOP_K")
    
    # Agent Configuration
    max_iterations: int = Field(default=3, alias="MAX_ITERATIONS")
    critic_threshold: float = Field(default=7.0, alias="CRITIC_THRESHOLD")
    
    # CORS Settings
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:3001",
        alias="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(
        default=True, alias="CORS_ALLOW_CREDENTIALS"
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        if not self.cors_origins:
            return []
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"


settings = Settings()


# --- Simple Logging Setup ---
def setup_logging() -> None:
    """Basic standard logging configuration."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Reduce noise from chatty libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Retrieve logger."""
    return logging.getLogger(name)


