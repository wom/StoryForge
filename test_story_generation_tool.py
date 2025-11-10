"""Test for story generation MCP tool registration."""

from mcp.server import Server

from storyforge.server.tools import register_all_tools
from storyforge.server.tools.story import register_story_tools


def test_story_tool_registration():
    """Test that story generation tool registers correctly."""
    # Create a test server
    server = Server("test-server")

    # Register story tools
    register_story_tools(server)

    print("✓ Story generation tool registered successfully")


def test_all_tools_registration():
    """Test that all tools (including story) register correctly."""
    # Create a test server
    server = Server("test-server")

    # Register all tools
    register_all_tools(server)

    print("✓ All tools registered successfully")


if __name__ == "__main__":
    test_story_tool_registration()
    test_all_tools_registration()


