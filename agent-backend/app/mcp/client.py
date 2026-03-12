"""
MCP Client Manager.

Responsible for:
1. Spawning the local MCP server as a subprocess.
2. Establishing the stdio RPC connection.
3. Fetching available tools and translating them into generic LangChain tools.
"""

import sys
import asyncio
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

from app.config import get_logger, settings

logger = get_logger(__name__)


class McpClientManager:
    def __init__(self):
        self._session: Optional[ClientSession] = None
        self._exit_stack = None
        
        # Stored references to keep the stdio connection alive
        self._read_stream = None
        self._write_stream = None
        self._process = None

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    async def connect(self):
        """Start the local MCP server subprocess and establish a session."""
        if self._session is not None:
            return  # Already connected

        # The MCP server is a python script in our app bundle
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.mcp.servers.core_tools"],
            env=None,  # inherits current environment
        )

        logger.info("Initializing MCP stdio client...")

        from contextlib import AsyncExitStack
        self._exit_stack = AsyncExitStack()

        try:
            self._read_stream, self._write_stream = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(self._read_stream, self._write_stream)
            )
            
            await self._session.initialize()
            logger.info("Successfully connected to local MCP server")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            if self._exit_stack:
                await self._exit_stack.aclose()
            self._session = None
            raise

    async def get_tools(self) -> list:
        """Fetch all exposed tools from the MCP server."""
        if not self._session:
            raise RuntimeError("MCP client not connected")
        
        response = await self._session.list_tools()
        # response.tools is a list of mcp.types.Tool objects
        # We return the dict representation for easier downstream parsing
        tools = [{"name": t.name, "description": t.description, "inputSchema": t.inputSchema} for t in response.tools]
        logger.debug(f"Discovered {len(tools)} tools via MCP")
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool on the remote MCP server."""
        if not self._session:
            raise RuntimeError("MCP client not connected")
        
        logger.debug(f"Calling MCP tool '{name}'")
        try:
            result: CallToolResult = await self._session.call_tool(name, arguments)
            
            if result.isError:
                logger.warning(f"MCP tool '{name}' returned an error context")
                
            # Content is usually a list of TextContent objects
            text_outputs = [c.text for c in result.content if getattr(c, "type", "text") == "text"]
            return "\n".join(text_outputs)
            
        except Exception as e:
            logger.error(f"Error calling MCP tool '{name}': {e}")
            raise

    async def disconnect(self):
        """Close session and terminate subprocess."""
        if self._exit_stack:
            logger.info("Disconnecting from MCP server")
            await self._exit_stack.aclose()
            self._session = None
            self._exit_stack = None


# Global singleton
mcp_client = McpClientManager()
