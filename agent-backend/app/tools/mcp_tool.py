"""
Tool wrapper that bridges MCP-discovered tools into generic LangChain tools.

When the Orchestrator starts, it will fetch tools from the MCP server,
dynamically generate these McpLangChainTool classes, and feed them into the agents.
"""

from typing import Any

# We use the official pydantic dynamic model builder to recreate the
# inputSchema provided by the MCP server so the LLM knows what args to pass.
from pydantic import Field, create_model

from app.tools.base import BaseTool, ToolInput, ToolOutput
from app.mcp.client import mcp_client
from app.config import get_logger

logger = get_logger(__name__)


# Generic output wrapper since MCP tool outputs are just strings/text content
class McpToolOutput(ToolOutput):
    result: str


class McpLangChainTool(BaseTool):
    """
    A dynamic adapter that allows any MCP-exposed tool to be safely executed
    by a LangChain/LangGraph agent.
    """

    name: str = "mcp_tool"
    description: str = "A generic MCP tool wrapper."
    
    # We store the MCP name to know what to ask the server to execute
    mcp_tool_name: str = Field(default="")
    
    async def execute(self, input_data: ToolInput) -> ToolOutput:
        """Forward the dynamically populated input args to the MCP client."""
        # Convert the dynamic Pydantic args model back into a raw dict
        args = input_data.model_dump()
        
        logger.debug(f"Executing MCP adapter for '{self.mcp_tool_name}'")
        try:
            # Result is always guaranteed to be a string
            result_str = await mcp_client.call_tool(self.mcp_tool_name, arguments=args)
            return McpToolOutput(result=result_str)
        except Exception as e:
            logger.error(f"MCP execute failed: {e}")
            return McpToolOutput(result=f"Tool error: {str(e)}")


def _build_pydantic_model_from_schema(schema: dict[str, Any], class_name: str) -> type[ToolInput]:
    """
    Generates a Pydantic Model (ToolInput subclass) given a JSON Schema dictionary.
    This creates the exact data structure the LLM needs to populate when calling the tool.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    
    fields = {}
    for prop_name, prop_details in properties.items():
        # Mapping JSON Schema types to Python types
        json_type = prop_details.get("type", "string")
        if json_type == "string":
            py_type = str
        elif json_type == "integer":
            py_type = int
        elif json_type == "number":
            py_type = float
        elif json_type == "boolean":
            py_type = bool
        elif json_type == "array":
            # Very basic array support
            py_type = list
        elif json_type == "object":
            py_type = dict
        else:
            py_type = Any
            
        # Wrap in Optional if not required
        if prop_name not in required:
            # Provide a fallback default depending on type just to be safe
            fields[prop_name] = (py_type | None, Field(default=None, description=prop_details.get("description", "")))
        else:
            fields[prop_name] = (py_type, Field(..., description=prop_details.get("description", "")))

    # Create the dynamic Pydantic model inheriting from ToolInput
    dynamic_model = create_model(class_name, __base__=ToolInput, **fields)
    return dynamic_model


async def get_all_mcp_tools() -> list[McpLangChainTool]:
    """
    Fetch all tools from the MCP server and build LangChain adapter instances for each.
    """
    if not mcp_client.is_connected:
        return []
        
    try:
        raw_tools = await mcp_client.get_tools()
        adapters = []
        
        for rt in raw_tools:
            name = rt["name"]
            desc = rt["description"]
            schema = rt["inputSchema"]
            
            # 1. Build the dynamic args model using the JSON Schema
            args_model = _build_pydantic_model_from_schema(schema, f"{name.title()}Args")
            
            # 2. Instantiate the wrapper tool
            adapter = McpLangChainTool(
                name=name,
                description=desc,
                args_schema=args_model,
                mcp_tool_name=name,
            )
            adapters.append(adapter)
            
        return adapters
        
    except Exception as e:
        logger.error(f"Failed to generate MCP tool adapters: {e}")
        return []
