"""
Centralized logging configuration for the application.
Provides structured logging with JSON output for production and colored console for development.
"""

import logging
import sys
from typing import Any
from pythonjsonlogger import jsonlogger
import colorlog
from app.config.settings import settings


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional context."""
    
    def add_fields(self, log_record: dict, record: logging.LogRecord, message_dict: dict) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["environment"] = settings.environment
        
        # Add request context if available
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
        if hasattr(record, "user_id"):
            log_record["user_id"] = record.user_id
        if hasattr(record, "agent_name"):
            log_record["agent_name"] = record.agent_name


def setup_logging() -> None:
    """Configure application logging based on environment."""
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    if settings.is_production:
        # JSON logging for production
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s",
            rename_fields={"timestamp": "asctime"}
        )
    else:
        # Colored console logging for development
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            }
        )
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter for adding contextual information to logs.
    Useful for tracking requests, agents, and user sessions.
    """
    
    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Add extra context to log records."""
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger_with_context(name: str, **context) -> LoggerAdapter:
    """
    Get a logger with additional context.
    
    Args:
        name: Logger name
        **context: Additional context to include in all log messages
        
    Returns:
        Logger adapter with context
        
    Example:
        logger = get_logger_with_context(__name__, request_id="123", agent_name="researcher")
        logger.info("Starting research")  # Will include request_id and agent_name
    """
    base_logger = get_logger(name)
    return LoggerAdapter(base_logger, context)
