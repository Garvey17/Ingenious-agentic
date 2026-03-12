"""
MCP API — routes for inspecting and interacting with MCP tools.

Endpoints:
  GET /api/mcp/tools  — lists all tools discovered via the MCP connection
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any

from app.config import settings, get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


class McpToolsResponse(BaseModel):
    mcp_enabled: bool
    tools: list[dict[str, Any]]
    count: int


@router.get("/tools", response_model=McpToolsResponse)
async def list_mcp_tools():
    """
    List all tools currently discovered via the Model Context Protocol.
    Returns an empty list if MCP is disabled or disconnected.
    """
    if not settings.enable_mcp:
        return McpToolsResponse(mcp_enabled=False, tools=[], count=0)

    try:
        from app.mcp.client import mcp_client
        if not mcp_client.is_connected:
            return McpToolsResponse(mcp_enabled=True, tools=[], count=0)
            
        tools = await mcp_client.get_tools()
        return McpToolsResponse(
            mcp_enabled=True,
            tools=tools,
            count=len(tools),
        )
    except Exception as e:
        logger.error(f"[mcp_api] Error fetching tools: {e}")
        return McpToolsResponse(mcp_enabled=True, tools=[], count=0)
