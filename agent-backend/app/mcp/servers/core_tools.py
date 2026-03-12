"""
Model Context Protocol (MCP) Server for Deep Research Desk.

This standalone script runs an MCP server over standard input/output (stdio).
It exposes our existing Python tools (WebSearch, Summarize, Memory) to any
agent connecting via the MCP protocol.

When `ENABLE_MCP=true`, the FastAPI backend spawns this as a subprocess
and routes agent tool calls through it, proving dynamic tool discovery.
"""

import sys
import logging
import asyncio
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from app.tools.web_search import WebSearchTool, WebSearchInput
from app.tools.summarize import SummarizeTool, SummarizeInput
from app.tools.memory_tool import MemoryTool, MemoryReadInput, MemoryWriteInput

# Configure basic logging to stderr so it doesn't pollute stdout (which MCP uses)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[mcp-server] %(levelname)s: %(message)s",
)
logger = logging.getLogger("mcp_server")

# Instantiate our local tools
search_tool = WebSearchTool()
summarize_tool = SummarizeTool()
memory_tool = MemoryTool()

# Create the MCP Server instance
app_server = Server("deep-research-tools")


@app_server.list_tools()
async def list_tools() -> list[Tool]:
    """Expose our local tools to the MCP client."""
    logger.info("Client requested tool list")
    
    return [
        Tool(
            name="web_search",
            description=search_tool.description,
            inputSchema=WebSearchInput.model_json_schema(),
        ),
        Tool(
            name="summarize_text",
            description=summarize_tool.description,
            inputSchema=SummarizeInput.model_json_schema(),
        ),
        Tool(
            name="memory_read",
            description="Retrieve semantically relevant past research before planning.",
            inputSchema=MemoryReadInput.model_json_schema(),
        ),
        Tool(
            name="memory_write",
            description="Store a completed research result.",
            inputSchema=MemoryWriteInput.model_json_schema(),
        ),
    ]


@app_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Route incomimg MCP tool invocations to the correct local Python tool."""
    logger.info(f"Executing tool: {name}")
    
    try:
        if name == "web_search":
            input_data = WebSearchInput(**arguments)
            result = await search_tool.execute(input_data)
            return [TextContent(type="text", text=result.model_dump_json())]

        elif name == "summarize_text":
            input_data = SummarizeInput(**arguments)
            result = await summarize_tool.execute(input_data)
            return [TextContent(type="text", text=result.model_dump_json())]

        elif name == "memory_read":
            input_data = MemoryReadInput(**arguments)
            result = await memory_tool.execute(input_data)
            return [TextContent(type="text", text=result.model_dump_json())]

        elif name == "memory_write":
            input_data = MemoryWriteInput(**arguments)
            result = await memory_tool.execute(input_data)
            return [TextContent(type="text", text=result.model_dump_json())]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Error executing {name}: {e}")
        # Return errors normally so the agent can see them
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server over stdio."""
    logger.info("Starting local MCP stdio server")
    async with stdio_server() as (read_stream, write_stream):
        await app_server.run(
            read_stream,
            write_stream,
            initialization_options=app_server.create_initialization_options(),
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server terminated by user")
    except Exception as e:
        logger.error(f"Server crashed: {e}")
