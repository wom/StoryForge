"""Pytest configuration for StoryForge tests."""

import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    """Configure anyio to use asyncio only."""
    return "asyncio"
