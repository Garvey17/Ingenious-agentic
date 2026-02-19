"""
Base agent class for all specialized agents.
Provides common functionality for LLM interaction, tool calling, and output parsing.
"""

from typing import Any, Optional, Type, TypeVar
from abc import ABC, abstractmethod
import json
from pydantic import BaseModel, ValidationError

from app.models.llm import get_llm
from app.config import get_logger_with_context

T = TypeVar("T", bound=BaseModel)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Agents are specialized roles that:
    - Have a specific system prompt
    - Can call tools (via MCP in future phases)
    - Return structured outputs (Pydantic models)
    """
    
    def __init__(self, agent_name: str, system_prompt: str):
        """
        Initialize the base agent.
        
        Args:
            agent_name: Name of the agent (e.g., "researcher", "analyst")
            system_prompt: System prompt defining agent behavior
        """
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.logger = get_logger_with_context(__name__, agent_name=agent_name)
        self.llm = get_llm()
        
        self.logger.info(f"Initialized {agent_name} agent")
    
    async def run(
        self,
        input_data: dict[str, Any],
        output_schema: Type[T],
        **kwargs
    ) -> T:
        """
        Run the agent with the given input.
        
        Args:
            input_data: Input data for the agent
            output_schema: Pydantic model for output validation
            **kwargs: Additional arguments for LLM generation
            
        Returns:
            Validated output matching the schema
            
        Raises:
            ValidationError: If output doesn't match schema
            Exception: If LLM generation fails
        """
        self.logger.info(f"Running {self.agent_name} agent")
        
        try:
            # Build the prompt
            prompt = self._build_prompt(input_data)
            
            # Generate response with JSON mode
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                response_format={"type": "json_object"},
                **kwargs
            )
            
            # Parse and validate response
            output = self._parse_response(response, output_schema)
            
            self.logger.info(f"{self.agent_name} agent completed successfully")
            return output
            
        except ValidationError as e:
            self.logger.error(f"Output validation failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}")
            raise
    
    @abstractmethod
    def _build_prompt(self, input_data: dict[str, Any]) -> str:
        """
        Build the prompt for the LLM based on input data.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Formatted prompt string
        """
        pass
    
    def _parse_response(self, response: str, output_schema: Type[T]) -> T:
        """
        Parse and validate LLM response against the output schema.
        
        Args:
            response: Raw LLM response (JSON string)
            output_schema: Pydantic model for validation
            
        Returns:
            Validated output
            
        Raises:
            ValidationError: If response doesn't match schema
            json.JSONDecodeError: If response is not valid JSON
        """
        try:
            # Parse JSON
            data = json.loads(response)
            
            # Validate against schema
            validated_output = output_schema(**data)
            
            return validated_output
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Raw response: {response}")
            raise
        except ValidationError as e:
            self.logger.error(f"Response validation failed: {e}")
            self.logger.debug(f"Parsed data: {data}")
            raise
    
    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Call a tool (placeholder for MCP integration in Phase 4).
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Tool arguments
            
        Returns:
            Tool result
        """
        self.logger.warning(
            f"Tool calling not yet implemented. Attempted to call: {tool_name}"
        )
        # This will be implemented in Phase 4 with MCP
        raise NotImplementedError("Tool calling will be implemented in Phase 4")
