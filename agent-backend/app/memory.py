"""
Memory module — connects to Qdrant and stores/searches research results.
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

from app.config import settings, get_logger
from app.models import get_embedding_model

logger = get_logger(__name__)


class Memory:
    """
    Handles everything to do with Qdrant:
    - Connecting to the database
    - Creating the collection if it doesn't exist
    - Saving, searching, and deleting research entries
    """

    def __init__(self):
        self.collection = settings.qdrant_collection_name
        self._client = None
        self._ready = False

    def _connect(self):
        """Opens (or reuses) the connection to Qdrant."""
        if self._client:
            return self._client
        try:
            from qdrant_client import QdrantClient
            if settings.qdrant_api_key:
                self._client = QdrantClient(
                    host=settings.qdrant_host, port=settings.qdrant_port,
                    api_key=settings.qdrant_api_key, https=True, timeout=10,
                )
            else:
                self._client = QdrantClient(
                    host=settings.qdrant_host, port=settings.qdrant_port, timeout=10,
                )
            logger.info(f"Connected to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
        except Exception as e:
            logger.error(f"Could not connect to Qdrant: {e}")
        return self._client

    async def _setup(self) -> bool:
        """Creates the Qdrant collection on first use if it doesn't already exist."""
        if self._ready:
            return True

        def _run():
            client = self._connect()
            if not client:
                return False
            try:
                from qdrant_client.models import Distance, VectorParams
                existing = [c.name for c in client.get_collections().collections]
                if self.collection not in existing:
                    size = get_embedding_model().dimension
                    client.create_collection(
                        collection_name=self.collection,
                        vectors_config=VectorParams(size=size, distance=Distance.COSINE),
                    )
                    logger.info(f"Created Qdrant collection '{self.collection}'")
                return True
            except Exception as e:
                logger.error(f"Could not set up Qdrant collection: {e}")
                return False

        self._ready = await asyncio.to_thread(_run)
        return self._ready

    async def save_research(self, request_id: str, topic: str, summary: str,
                             sources: List[dict] = None, metadata: dict = None) -> bool:
        """Embeds the topic+summary and saves it to Qdrant."""
        if not settings.enable_memory or not await self._setup():
            return False
        try:
            vector = await get_embedding_model().embed(f"{topic}\n\n{summary}")
            payload = {
                "request_id": request_id, "topic": topic, "summary": summary,
                "sources": sources or [],
                "stored_at": datetime.now(timezone.utc).isoformat(),
                **(metadata or {}),
            }

            def _run():
                from qdrant_client.models import PointStruct
                self._connect().upsert(
                    collection_name=self.collection,
                    points=[PointStruct(id=request_id, vector=vector, payload=payload)],
                )
                return True

            return await asyncio.to_thread(_run)
        except Exception as e:
            logger.error(f"Memory save failed: {e}")
            return False

    async def search_past_research(self, topic: str, top_k: Optional[int] = None,
                                    score_threshold: float = 0.5) -> List[dict]:
        """Searches Qdrant for past research related to the given topic."""
        if not settings.enable_memory or not await self._setup():
            return []
        try:
            k = top_k or settings.memory_top_k
            vector = await get_embedding_model().embed(topic)

            def _run():
                results = self._connect().search(
                    collection_name=self.collection, query_vector=vector,
                    limit=k, score_threshold=score_threshold, with_payload=True,
                )
                return [{"score": hit.score, **hit.payload} for hit in results]

            return await asyncio.to_thread(_run)
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []

    async def delete_by_id(self, request_id: str) -> bool:
        """Deletes a specific memory entry by its request ID."""
        if not settings.enable_memory:
            return False
        try:
            def _run():
                from qdrant_client.models import PointIdsList
                self._connect().delete(
                    collection_name=self.collection,
                    points_selector=PointIdsList(points=[request_id]),
                )
                return True
            return await asyncio.to_thread(_run)
        except Exception as e:
            logger.error(f"Memory delete failed: {e}")
            return False

    async def get_total_count(self) -> int:
        """Returns the total number of memories stored in Qdrant."""
        if not settings.enable_memory or not await self._setup():
            return 0
        try:
            def _run():
                return self._connect().count(collection_name=self.collection).count
            return await asyncio.to_thread(_run)
        except Exception as e:
            logger.error(f"Memory count failed: {e}")
            return 0
