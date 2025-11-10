"""Tests for story refinement tools."""

import pytest

from storyforge.server.tools.refinement import register_refinement_tools


@pytest.mark.anyio
async def test_refinement_tools_registration():
    """Test that refinement tools are registered correctly."""
    from mcp.server import Server

    server = Server("test-server")
    register_refinement_tools(server)

    # Test should pass if no exceptions raised during registration
    assert True, "Refinement tools registered successfully"


# Note: Comprehensive tests for refine_story will be added after
# basic integration testing. These require:
# - Mock session checkpoints with story content
# - Mock LLM backend for story refinement
# - Validation of refinement prompt construction
# - Testing refinement history tracking in checkpoint
