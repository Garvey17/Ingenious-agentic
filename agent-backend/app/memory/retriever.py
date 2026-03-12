"""
Memory Retriever — high-level interface combining embeddings + Qdrant.

Responsibilities:
- Embed topics/summaries using the configured embedding model
- Store completed research into Qdrant
- Retrieve semantically relevant past research before planning

Graceful degradation: all methods return empty/safe values when
embeddings fail or Qdrant is unreachable. Never raises to callers.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

from app.config import settings, get_logger
from app.memory.qdrant_client import QdrantMemoryClient

logger = get_logger(__name__)


class MemoryRetriever:
    """
    Combines the embedding model with the Qdrant client.

    Args:
        qdrant:    QdrantMemoryClient instance (inject mock in tests).
        embedder:  Any object with an async embed_text(str) -> list[float] method.
                   If None, lazily loads the configured embedding model.
    """

    def __init__(
        self,
        qdrant: QdrantMemoryClient | None = None,
        embedder=None,
    ):
        self._qdrant = qdrant or QdrantMemoryClient()
        self._embedder = embedder  # lazy if None
        self._initialized = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embedder(self):
        """Lazily return the configured embedding model."""
        if self._embedder is not None:
            return self._embedder
        try:
            from app.models.embeddings import get_embedding_model
            self._embedder = get_embedding_model()
            return self._embedder
        except Exception as e:
            logger.error(f"[memory] Failed to load embedding model: {e}")
            return None

    async def _embed(self, text: str) -> list[float] | None:
        """Embed text, returning None on failure."""
        embedder = self._get_embedder()
        if embedder is None:
            return None
        try:
            return await embedder.embed_text(text)
        except Exception as e:
            logger.error(f"[memory] Embedding failed: {e}")
            return None

    async def _ensure_ready(self) -> bool:
        """Ensure the Qdrant collection exists. Idempotent."""
        if self._initialized:
            return True
        embedder = self._get_embedder()
        if embedder is None:
            return False
        vector_size = embedder.dimension
        ok = await self._qdrant.ensure_collection(vector_size)
        self._initialized = ok
        return ok

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def store_research(
        self,
        request_id: str,
        topic: str,
        summary: str,
        sources: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> bool:
        """
        Embed and store a completed research result in Qdrant.

        Args:
            request_id: UUID from the research session (used as Qdrant point ID).
            topic:      Research topic (used as the query text for retrieval).
            summary:    Executive summary or condensed report text.
            sources:    List of source dicts to store in the payload.
            metadata:   Extra metadata (e.g. depth, timestamp).

        Returns:
            True if stored successfully, False on any error.
        """
        if not await self._ensure_ready():
            logger.warning("[memory] store_research skipped — memory not ready")
            return False

        # Build text to embed: topic + summary for richer semantics
        embed_text = f"{topic}\n\n{summary}"
        vector = await self._embed(embed_text)
        if vector is None:
            logger.warning("[memory] store_research skipped — embedding failed")
            return False

        payload = {
            "request_id": request_id,
            "topic": topic,
            "summary": summary,
            "sources": sources or [],
            "stored_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }

        ok = await self._qdrant.upsert(
            point_id=request_id,
            vector=vector,
            payload=payload,
        )
        if ok:
            logger.info(f"[memory] Stored research for topic='{topic}' id={request_id}")
        return ok

    async def retrieve_relevant(
        self,
        topic: str,
        top_k: int | None = None,
        score_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Find past research most semantically similar to the given topic.

        Args:
            topic:           The research topic to search for.
            top_k:           Max results. Defaults to settings.memory_top_k.
            score_threshold: Minimum cosine similarity (0-1).

        Returns:
            List of memory payload dicts, ordered by relevance.
            Empty list on any error or no relevant results.
        """
        if not await self._ensure_ready():
            logger.debug("[memory] retrieve_relevant skipped — memory not ready")
            return []

        k = top_k if top_k is not None else settings.memory_top_k
        vector = await self._embed(topic)
        if vector is None:
            logger.warning("[memory] retrieve_relevant skipped — embedding failed")
            return []

        results = await self._qdrant.search(
            vector=vector,
            top_k=k,
            score_threshold=score_threshold,
        )
        logger.info(
            f"[memory] Retrieved {len(results)} past memories for topic='{topic}'"
        )
        return results

    async def delete(self, request_id: str) -> bool:
        """Delete a stored memory by request_id."""
        return await self._qdrant.delete(request_id)

    async def count(self) -> int:
        """Return total number of stored memories."""
        return await self._qdrant.count()
