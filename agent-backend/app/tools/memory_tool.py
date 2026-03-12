"""
MemoryTool — BaseTool wrapper for the MemoryRetriever.

Exposes two operations:
- read:  retrieve semantically relevant past research
- write: store a completed research result

All operations are safe to call without Qdrant running — they return
empty / False and log a warning rather than raising.
"""

from __future__ import annotations

from typing import Any, Optional

from app.tools.base import BaseTool, ToolInput, ToolOutput
from app.config import settings, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Input / Output schemas
# ---------------------------------------------------------------------------

class MemoryReadInput(ToolInput):
    """Input for reading from memory."""
    query: str
    top_k: int = 5
    score_threshold: float = 0.5


class MemoryReadOutput(ToolOutput):
    """Output from reading memory."""
    memories: list[dict[str, Any]]
    count: int

    @classmethod
    def empty(cls) -> "MemoryReadOutput":
        return cls(memories=[], count=0)


class MemoryWriteInput(ToolInput):
    """Input for writing to memory."""
    request_id: str
    topic: str
    summary: str
    sources: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}


class MemoryWriteOutput(ToolOutput):
    """Output from writing to memory."""
    stored: bool
    request_id: str


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class MemoryTool(BaseTool):
    """
    Tool for reading and writing research memory via Qdrant.

    Usage:
        tool = MemoryTool()

        # Read
        result = await tool.read(MemoryReadInput(query="AI trends 2025"))

        # Write
        result = await tool.write(MemoryWriteInput(
            request_id="abc-123",
            topic="AI trends 2025",
            summary="AI is transforming...",
            sources=[...]
        ))
    """

    name: str = "memory"
    description: str = (
        "Reads and writes research memory. "
        "Use 'read' to retrieve relevant past research before planning. "
        "Use 'write' to persist a completed research result."
    )

    def __init__(self, retriever=None):
        """
        Args:
            retriever: Optional MemoryRetriever instance.
                       If None, builds a default one from settings.
                       Inject a mock here in tests.
        """
        self._retriever = retriever  # lazy if None

    def _get_retriever(self):
        if self._retriever is not None:
            return self._retriever
        from app.memory.retriever import MemoryRetriever
        self._retriever = MemoryRetriever()
        return self._retriever

    # ------------------------------------------------------------------
    # Required BaseTool methods
    # ------------------------------------------------------------------

    async def execute(self, input_data: ToolInput) -> ToolOutput:
        """
        Route to read or write based on input type.
        This satisfies the BaseTool interface.
        """
        if isinstance(input_data, MemoryReadInput):
            return await self.read(input_data)
        if isinstance(input_data, MemoryWriteInput):
            return await self.write(input_data)
        raise TypeError(f"Unsupported input type: {type(input_data)}")

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    async def read(self, input_data: MemoryReadInput) -> MemoryReadOutput:
        """
        Retrieve semantically relevant past research.

        Returns an empty MemoryReadOutput if memory is disabled,
        Qdrant is unreachable, or embedding fails.
        """
        if not settings.enable_memory:
            logger.debug("[memory_tool] read skipped — ENABLE_MEMORY=false")
            return MemoryReadOutput.empty()

        try:
            retriever = self._get_retriever()
            memories = await retriever.retrieve_relevant(
                topic=input_data.query,
                top_k=input_data.top_k,
                score_threshold=input_data.score_threshold,
            )
            return MemoryReadOutput(memories=memories, count=len(memories))

        except Exception as e:
            logger.error(f"[memory_tool] read error: {e}")
            return MemoryReadOutput.empty()

    async def write(self, input_data: MemoryWriteInput) -> MemoryWriteOutput:
        """
        Store a completed research result in Qdrant.

        Returns stored=False (not an exception) when memory is disabled
        or any error occurs.
        """
        if not settings.enable_memory:
            logger.debug("[memory_tool] write skipped — ENABLE_MEMORY=false")
            return MemoryWriteOutput(stored=False, request_id=input_data.request_id)

        try:
            retriever = self._get_retriever()
            stored = await retriever.store_research(
                request_id=input_data.request_id,
                topic=input_data.topic,
                summary=input_data.summary,
                sources=input_data.sources,
                metadata=input_data.metadata,
            )
            return MemoryWriteOutput(stored=stored, request_id=input_data.request_id)

        except Exception as e:
            logger.error(f"[memory_tool] write error: {e}")
            return MemoryWriteOutput(stored=False, request_id=input_data.request_id)
