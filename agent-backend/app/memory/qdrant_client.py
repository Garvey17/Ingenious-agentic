"""
Qdrant memory client — async wrapper around qdrant-client.

Design principles:
- All public methods are async and safe to call when Qdrant is unavailable
  (they catch connection errors and return empty / False rather than crashing)
- The client is completely bypassed when ENABLE_MEMORY=false
- Testable: inject a custom client via the `client` constructor arg (for mocks)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from uuid import uuid4

from app.config import settings, get_logger

logger = get_logger(__name__)


class QdrantMemoryClient:
    """
    Async wrapper around the qdrant-client library.

    Raises no exceptions to callers — errors are logged and safe defaults returned.
    All heavy I/O is offloaded to asyncio.to_thread so this is safe inside FastAPI.
    """

    COLLECTION: str = ""   # set in __init__

    def __init__(self, client: Any = None):
        """
        Args:
            client: Optional pre-built qdrant_client.QdrantClient instance.
                    If None, a real client is built from settings.
                    Pass a mock here in tests.
        """
        self.COLLECTION = settings.qdrant_collection_name
        self._client = client  # may be None until _get_client() is called
        self._ready: bool = False   # becomes True after ensure_collection succeeds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        """Lazily build the real Qdrant client."""
        if self._client is not None:
            return self._client

        try:
            from qdrant_client import QdrantClient

            if settings.qdrant_api_key:
                # Cloud / authenticated instance
                self._client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    api_key=settings.qdrant_api_key,
                    https=True,
                    timeout=10,
                )
            else:
                # Local Docker
                self._client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    timeout=10,
                )

            logger.info(
                f"Qdrant client connected to {settings.qdrant_host}:{settings.qdrant_port}"
            )
            return self._client

        except Exception as e:
            logger.error(f"[qdrant] Failed to create client: {e}")
            return None

    def _sync_ensure_collection(self, vector_size: int) -> bool:
        """Synchronous collection creation — called via to_thread."""
        client = self._get_client()
        if client is None:
            return False

        try:
            from qdrant_client.models import Distance, VectorParams

            existing = [c.name for c in client.get_collections().collections]
            if self.COLLECTION not in existing:
                client.create_collection(
                    collection_name=self.COLLECTION,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    f"[qdrant] Created collection '{self.COLLECTION}' "
                    f"(dim={vector_size}, distance=COSINE)"
                )
            else:
                logger.debug(f"[qdrant] Collection '{self.COLLECTION}' already exists")

            self._ready = True
            return True

        except Exception as e:
            logger.error(f"[qdrant] ensure_collection failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def ensure_collection(self, vector_size: int | None = None) -> bool:
        """
        Create the Qdrant collection if it does not exist.

        Args:
            vector_size: Embedding dimension. Defaults to settings.qdrant_vector_size.

        Returns:
            True on success, False on any error (Qdrant unreachable, etc.)
        """
        size = vector_size or settings.qdrant_vector_size
        return await asyncio.to_thread(self._sync_ensure_collection, size)

    async def upsert(
        self,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> bool:
        """
        Store or overwrite a memory point.

        Args:
            point_id:  Unique ID (use request_id).
            vector:    Embedding vector.
            payload:   Arbitrary metadata dict (topic, summary, sources, etc.)

        Returns:
            True on success, False on error.
        """
        def _sync():
            client = self._get_client()
            if client is None:
                return False
            try:
                from qdrant_client.models import PointStruct

                client.upsert(
                    collection_name=self.COLLECTION,
                    points=[PointStruct(id=point_id, vector=vector, payload=payload)],
                )
                logger.debug(f"[qdrant] Upserted point {point_id}")
                return True
            except Exception as e:
                logger.error(f"[qdrant] upsert failed for {point_id}: {e}")
                return False

        return await asyncio.to_thread(_sync)

    async def search(
        self,
        vector: list[float],
        top_k: int = 5,
        score_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Semantic similarity search.

        Args:
            vector:           Query embedding.
            top_k:            Maximum number of results.
            score_threshold:  Minimum cosine similarity to include.

        Returns:
            List of payload dicts ordered by similarity (best first).
            Empty list on any error or no results.
        """
        def _sync():
            client = self._get_client()
            if client is None:
                return []
            try:
                results = client.search(
                    collection_name=self.COLLECTION,
                    query_vector=vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                )
                return [
                    {"score": hit.score, **hit.payload}
                    for hit in results
                ]
            except Exception as e:
                logger.error(f"[qdrant] search failed: {e}")
                return []

        return await asyncio.to_thread(_sync)

    async def delete(self, point_id: str) -> bool:
        """
        Delete a memory point by ID.

        Returns:
            True on success, False on error.
        """
        def _sync():
            client = self._get_client()
            if client is None:
                return False
            try:
                from qdrant_client.models import PointIdsList

                client.delete(
                    collection_name=self.COLLECTION,
                    points_selector=PointIdsList(points=[point_id]),
                )
                logger.debug(f"[qdrant] Deleted point {point_id}")
                return True
            except Exception as e:
                logger.error(f"[qdrant] delete failed for {point_id}: {e}")
                return False

        return await asyncio.to_thread(_sync)

    async def count(self) -> int:
        """Return the number of stored memories (0 on error)."""
        def _sync():
            client = self._get_client()
            if client is None:
                return 0
            try:
                return client.count(collection_name=self.COLLECTION).count
            except Exception as e:
                logger.error(f"[qdrant] count failed: {e}")
                return 0

        return await asyncio.to_thread(_sync)
