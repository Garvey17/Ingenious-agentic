"""
Abstract base class for all tools in the Deep Research Desk system.
Tools are external capabilities that agents can use (web search, memory, etc).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ToolInput(BaseModel):
    """Base model for tool inputs."""
    pass


class ToolOutput(BaseModel):
    """Base model for tool outputs."""
    success: bool = Field(description="Whether the tool execution was successful")
    data: Optional[Any] = Field(default=None, description="Tool output data")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    Tools provide external capabilities to agents:
    - Web search
    - Memory storage/retrieval
    - File operations
    - API calls
    
    In Phase 4, tools will be accessed through MCP servers.
    For now, this provides the interface that MCP will implement.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize the tool.
        
        Args:
            name: Tool name (e.g., "web_search")
            description: Human-readable description of what the tool does
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, input_data: ToolInput) -> ToolOutput:
        """
        Execute the tool with the given input.
        
        Args:
            input_data: Tool-specific input parameters
        
        Returns:
            Tool execution result
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert tool to dictionary representation for agent context.
        
        Returns:
            Dictionary with tool metadata
        """
        return {
            "name": self.name,
            "description": self.description,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
