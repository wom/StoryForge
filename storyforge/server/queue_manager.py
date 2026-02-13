"""Queue management for StoryForge MCP server.

Implements single-threaded execution with FIFO queue for story generation requests.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..shared.types import ErrorCode, MCPError


@dataclass
class QueuedRequest:
    """Represents a queued story generation request."""

    session_id: str
    handler: Callable
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    future: asyncio.Future
    queued_at: float


class QueueManager:
    """Manages story generation queue with single-threaded execution."""

    def __init__(self, max_queue_size: int = 10):
        """
        Initialize queue manager.

        Args:
            max_queue_size: Maximum number of requests that can be queued (default: 10)
        """
        self.max_queue_size = max_queue_size
        self.active_session: str | None = None
        self.queue: list[QueuedRequest] = []
        self._lock = asyncio.Lock()
        self._processing = False
        self._queue_status_cache: dict[str, Any] | None = None
        self._queue_status_cache_time: float = 0.0
        self._cache_ttl = 0.5  # 500ms cache TTL
        self._active_futures: dict[str, asyncio.Future] = {}  # Track active generation tasks for polling

    async def enqueue(
        self,
        session_id: str,
        handler: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> asyncio.Future:
        """
        Enqueue a story generation request.

        Args:
            session_id: Unique session identifier
            handler: Async function to execute
            *args: Positional arguments for handler
            **kwargs: Keyword arguments for handler

        Returns:
            Future that will resolve when request completes

        Raises:
            MCPError: If queue is full or concurrent limit exceeded
        """
        async with self._lock:
            # Check if there's already an active session
            if self.active_session is not None:
                # Check queue capacity
                if len(self.queue) >= self.max_queue_size:
                    raise MCPError(
                        code=ErrorCode.QUEUE_FULL,
                        message=f"Queue is full ({self.max_queue_size} requests pending)",
                        details={
                            "queue_length": len(self.queue),
                            "max_queue_size": self.max_queue_size,
                        },
                        recoverable=True,
                        recovery_hint="Wait for current requests to complete and try again",
                    )

                # Queue the request
                future: asyncio.Future = asyncio.Future()
                request = QueuedRequest(
                    session_id=session_id,
                    handler=handler,
                    args=args,
                    kwargs=kwargs,
                    future=future,
                    queued_at=time.time(),
                )
                self.queue.append(request)
                self._invalidate_cache()
                return future

            # No active session, mark as active and queue it
            self.active_session = session_id
            self._invalidate_cache()

            # Create future and request
            future = asyncio.Future()
            request = QueuedRequest(
                session_id=session_id,
                handler=handler,
                args=args,
                kwargs=kwargs,
                future=future,
                queued_at=time.time(),
            )

            # Start processing task if not already running
            if not self._processing:
                asyncio.create_task(self._process_request(request))

        return future

    async def _process_request(self, request: QueuedRequest) -> None:
        """
        Process a single request and then continue with the queue.

        Args:
            request: Request to process
        """
        self._processing = True

        # Execute the request
        try:
            result = await request.handler(*request.args, **request.kwargs)
            request.future.set_result(result)
        except Exception as e:
            request.future.set_exception(e)

        # Continue processing queue
        await self._process_queue()

    async def _process_queue(self) -> None:
        """Process queued requests in FIFO order."""
        self._processing = True

        while True:
            async with self._lock:
                # Check if there are queued requests
                if not self.queue:
                    self.active_session = None
                    self._processing = False
                    self._invalidate_cache()
                    break

                # Get next request (FIFO)
                request = self.queue.pop(0)
                self.active_session = request.session_id
                self._invalidate_cache()

            # Execute request outside of lock to allow concurrent status queries
            try:
                result = await request.handler(*request.args, **request.kwargs)
                request.future.set_result(result)
            except Exception as e:
                request.future.set_exception(e)

    def get_status(self) -> dict[str, Any]:
        """
        Get current queue status.

        Returns cached status if available (500ms TTL).

        Returns:
            Dict with active_session, queue, and queue_length
        """
        current_time = time.time()

        # Return cached status if still valid
        if self._queue_status_cache is not None and current_time - self._queue_status_cache_time < self._cache_ttl:
            return self._queue_status_cache

        # Build fresh status
        status = {
            "active_session": self.active_session,
            "queue": [{"session_id": req.session_id, "position": idx + 1} for idx, req in enumerate(self.queue)],
            "queue_length": len(self.queue),
        }

        # Cache the status
        self._queue_status_cache = status
        self._queue_status_cache_time = current_time

        return status

    def _invalidate_cache(self) -> None:
        """Invalidate the queue status cache."""
        self._queue_status_cache = None
        self._queue_status_cache_time = 0.0

    def _create_resolved_future(self, result: Any) -> asyncio.Future:
        """Create a future that is already resolved with a result."""
        future: asyncio.Future = asyncio.Future()
        future.set_result(result)
        return future

    def _create_rejected_future(self, error: Exception) -> asyncio.Future:
        """Create a future that is already rejected with an error."""
        future: asyncio.Future = asyncio.Future()
        future.set_exception(error)
        return future

    def is_busy(self) -> bool:
        """Check if the server is currently processing a request."""
        return self.active_session is not None

    def get_position(self, session_id: str) -> int | None:
        """
        Get the queue position for a session.

        Args:
            session_id: Session to check

        Returns:
            Position in queue (1-based) or None if not in queue
        """
        for idx, req in enumerate(self.queue):
            if req.session_id == session_id:
                return idx + 1
        return None
