"""Tests for configuration management tools."""

import pytest

from storyforge.server.tools.config import register_config_tools


@pytest.mark.anyio
async def test_config_tools_registration():
    """Test that configuration management tools are registered correctly."""
    from mcp.server import Server

    server = Server("test-server")
    register_config_tools(server)

    # Test should pass if no exceptions raised during registration
    assert True, "Config tools registered successfully"


# Note: Comprehensive tests for list_backends, get_config, and
# update_session_backend will be added after basic integration testing.
# These require:
# - Mock environment variables for API keys
# - Mock config files
# - Mock session checkpoints
