"""Test queue management functionality."""

import asyncio

import pytest

from storyforge.server.queue_manager import QueueManager
from storyforge.shared.types import ErrorCode, MCPError

# Configure anyio to use asyncio only
pytest_plugins = ("anyio",)


@pytest.mark.anyio
async def test_queue_manager_basic():
    """Test basic queue manager functionality."""
    queue_manager = QueueManager(max_queue_size=2)

    # Simple handler that returns a value after a delay
    async def simple_handler(value: int, delay: float = 0.1) -> list[dict]:
        await asyncio.sleep(delay)
        return [{"result": value}]

    # First request should execute immediately
    future1 = await queue_manager.enqueue("session1", simple_handler, 1, delay=0.2)
    assert queue_manager.is_busy()
    assert queue_manager.active_session == "session1"

    # Second request should be queued
    future2 = await queue_manager.enqueue("session2", simple_handler, 2, delay=0.1)
    assert queue_manager.is_busy()
    assert queue_manager.active_session == "session1"
    assert len(queue_manager.queue) == 1

    # Check queue status
    status = queue_manager.get_status()
    assert status["active_session"] == "session1"
    assert status["queue_length"] == 1
    assert status["queue"][0]["session_id"] == "session2"
    assert status["queue"][0]["position"] == 1

    # Wait for both to complete
    result1 = await future1
    result2 = await future2

    assert result1 == [{"result": 1}]
    assert result2 == [{"result": 2}]

    # Queue should be empty
    assert not queue_manager.is_busy()
    assert queue_manager.active_session is None
    assert len(queue_manager.queue) == 0


@pytest.mark.anyio
async def test_queue_manager_full():
    """Test queue full error."""
    queue_manager = QueueManager(max_queue_size=1)

    async def slow_handler() -> list[dict]:
        await asyncio.sleep(1.0)
        return [{"result": "done"}]

    # First request executes
    _future1 = await queue_manager.enqueue("session1", slow_handler)

    # Second request queues
    _future2 = await queue_manager.enqueue("session2", slow_handler)

    # Third request should fail with QUEUE_FULL
    with pytest.raises(MCPError) as exc_info:
        await queue_manager.enqueue("session3", slow_handler)

    assert exc_info.value.code == ErrorCode.QUEUE_FULL
    assert "full" in exc_info.value.message.lower()


@pytest.mark.anyio
async def test_queue_manager_fifo():
    """Test FIFO queue ordering."""
    queue_manager = QueueManager(max_queue_size=5)
    results = []

    async def ordered_handler(session_id: str) -> list[dict]:
        await asyncio.sleep(0.05)  # Small delay to ensure ordering
        results.append(session_id)
        return [{"session": session_id}]

    # Start first request
    future1 = await queue_manager.enqueue("session1", ordered_handler, "session1")

    # Queue several more
    future2 = await queue_manager.enqueue("session2", ordered_handler, "session2")
    future3 = await queue_manager.enqueue("session3", ordered_handler, "session3")
    future4 = await queue_manager.enqueue("session4", ordered_handler, "session4")

    # Wait for all to complete
    await asyncio.gather(future1, future2, future3, future4)

    # Results should be in FIFO order
    assert results == ["session1", "session2", "session3", "session4"]


@pytest.mark.anyio
async def test_queue_manager_error_handling():
    """Test error handling in queued tasks."""
    queue_manager = QueueManager(max_queue_size=2)

    async def failing_handler() -> list[dict]:
        await asyncio.sleep(0.05)
        raise ValueError("Intentional test error")

    async def success_handler() -> list[dict]:
        await asyncio.sleep(0.05)
        return [{"result": "success"}]

    # First request fails
    future1 = await queue_manager.enqueue("session1", failing_handler)

    # Second request should still execute
    future2 = await queue_manager.enqueue("session2", success_handler)

    # First should raise error
    with pytest.raises(ValueError, match="Intentional test error"):
        await future1

    # Second should succeed
    result2 = await future2
    assert result2 == [{"result": "success"}]


@pytest.mark.anyio
async def test_queue_status_caching():
    """Test queue status caching."""
    queue_manager = QueueManager()

    async def slow_handler() -> list[dict]:
        await asyncio.sleep(0.5)
        return [{"result": "done"}]

    # Start a request
    await queue_manager.enqueue("session1", slow_handler)

    # Get status twice quickly
    status1 = queue_manager.get_status()
    status2 = queue_manager.get_status()

    # Should be the same object (cached)
    assert status1 is status2

    # Wait for cache to expire (500ms)
    await asyncio.sleep(0.6)

    # Should get fresh status
    status3 = queue_manager.get_status()
    assert status3 is not status1


if __name__ == "__main__":
    asyncio.run(test_queue_manager_basic())
    asyncio.run(test_queue_manager_full())
    asyncio.run(test_queue_manager_fifo())
    asyncio.run(test_queue_manager_error_handling())
    asyncio.run(test_queue_status_caching())
    print("✓ All queue manager tests passed")
