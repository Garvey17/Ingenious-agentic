"""
In-memory session store for research requests.

Stores ResearchState dicts keyed by request_id.
Will be replaced by a persistent database in Phase 6.
"""

import asyncio
from typing import Optional
from app.graph.state import ResearchState
from app.config.logging import get_logger

logger = get_logger(__name__)

# Thread-safe in-memory store
_store: dict[str, ResearchState] = {}
_lock = asyncio.Lock()


async def save_session(request_id: str, state: ResearchState) -> None:
    """Persist (or create) a session."""
    async with _lock:
        _store[request_id] = state
    logger.debug(f"[session_store] Saved session {request_id}")


async def get_session(request_id: str) -> Optional[ResearchState]:
    """Retrieve a session by ID, or None if not found."""
    async with _lock:
        return _store.get(request_id)


async def update_session(request_id: str, updates: dict) -> Optional[ResearchState]:
    """Merge updates into an existing session."""
    async with _lock:
        if request_id not in _store:
            logger.warning(f"[session_store] Session not found: {request_id}")
            return None
        _store[request_id] = {**_store[request_id], **updates}
        return _store[request_id]


async def list_sessions() -> list[str]:
    """Return all active session IDs."""
    async with _lock:
        return list(_store.keys())


async def delete_session(request_id: str) -> bool:
    """Remove a session. Returns True if it existed."""
    async with _lock:
        if request_id in _store:
            del _store[request_id]
            return True
        return False
