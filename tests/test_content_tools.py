"""Tests for content management tools."""

import pytest

from storyforge.server.tools.content import register_content_tools


@pytest.mark.anyio
async def test_content_tools_registration():
    """Test that content management tools are registered correctly."""
    from mcp.server import Server

    server = Server("test-server")
    register_content_tools(server)

    # Test should pass if no exceptions raised during registration
    assert True, "Content tools registered successfully"


# Note: Comprehensive tests for list_context_files, get_context_content,
# and save_as_context will be added after basic integration testing.
# These require:
# - Sample context files in test fixtures
# - Mock session checkpoints with story content
# - ContextManager with test data
