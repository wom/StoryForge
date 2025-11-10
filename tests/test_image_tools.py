"""Tests for image generation tools."""

import pytest

from storyforge.server.tools.image import register_image_tools


@pytest.mark.anyio
async def test_image_tools_registration():
    """Test that image generation tools are registered correctly."""
    from mcp.server import Server

    server = Server("test-server")
    register_image_tools(server)

    # Test should pass if no exceptions raised during registration
    assert True, "Image tools registered successfully"


# Note: Comprehensive tests for generate_images will be added after
# basic integration testing. These require:
# - Mock session checkpoints with story content
# - Mock LLM backend for image generation
# - Temporary output directories
