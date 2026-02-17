"""
Centralized configuration management for Deep Research Desk.
Uses Pydantic Settings for environment-based configuration with validation.
"""

from typing import Literal, Optional
from pydantic import Field, validator
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
    reload: bool = Field(default=True, alias="RELOAD")
    
    # LLM Provider Configuration
    llm_provider: Literal["openai", "gemini"] = Field(
        default="openai", alias="LLM_PROVIDER"
    )
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", alias="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.7, ge=0.0, le=2.0, alias="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=4096, gt=0, alias="OPENAI_MAX_TOKENS")
    
    # Google Gemini Configuration
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-pro", alias="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.7, ge=0.0, le=2.0, alias="GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(default=4096, gt=0, alias="GEMINI_MAX_TOKENS")
    
    # Also support generic LLM_* aliases for compatibility/simplicity
    llm_model: Optional[str] = Field(default=None, alias="LLM_MODEL")
    llm_temperature: Optional[float] = Field(default=None, alias="LLM_TEMPERATURE")
    llm_max_tokens: Optional[int] = Field(default=None, alias="LLM_MAX_TOKENS")

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
    serper_api_key: Optional[str] = Field(default=None, alias="SERPER_API_KEY")
    
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
    memory_top_k: int = Field(default=5, gt=0, alias="MEMORY_TOP_K")
    
    # Agent Configuration
    max_iterations: int = Field(default=3, gt=0, le=10, alias="MAX_ITERATIONS")
    critic_threshold: float = Field(default=7.0, ge=0.0, le=10.0, alias="CRITIC_THRESHOLD")
    
    # CORS Settings
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:3001",
        alias="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(
        default=True, alias="CORS_ALLOW_CREDENTIALS"
    )
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(default=10, gt=0, alias="RATE_LIMIT_PER_MINUTE")
    
    # Logging
    log_format: Literal["json", "console"] = Field(default="json", alias="LOG_FORMAT")
    log_file_enabled: bool = Field(default=False, alias="LOG_FILE_ENABLED")
    log_file_path: str = Field(default="logs/app.log", alias="LOG_FILE_PATH")
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse comma-separated CORS origins into a string (will be split later)."""
        # Just return the string as-is, we'll handle it in a property
        return v
    
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        if not self.cors_origins:
            return []
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    def get_llm_api_key(self) -> str:
        """Get the API key for the selected LLM provider."""
        if self.llm_provider == "openai":
            if not self.openai_api_key:
                # Allow empty for dev if not strictly enforced
                return ""
            return self.openai_api_key
        elif self.llm_provider == "gemini":
            if not self.google_api_key:
                # Allow empty for dev if not strictly enforced
                return ""
            return self.google_api_key
        return ""
    
    def get_search_api_key(self) -> Optional[str]:
        """Get the API key for the selected search provider."""
        if self.search_provider == "tavily":
            return self.tavily_api_key
        elif self.search_provider == "serper":
            return self.serper_api_key
        return None


# Global settings instance
settings = Settings()
