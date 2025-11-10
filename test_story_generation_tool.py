"""Test for story generation MCP tool registration."""

from mcp.server import Server

from storyforge.server.queue_manager import QueueManager
from storyforge.server.tools import register_all_tools
from storyforge.server.tools.story import register_story_tools


def test_story_tool_registration():
    """Test that story generation tool registers correctly."""
    # Create a test server and queue manager
    server = Server("test-server")
    queue_manager = QueueManager()

    # Register story tools
    register_story_tools(server, queue_manager)

    print("✓ Story generation tool registered successfully")


def test_all_tools_registration():
    """Test that all tools (including story) register correctly."""
    # Create a test server and queue manager
    server = Server("test-server")
    queue_manager = QueueManager()

    # Register all tools
    register_all_tools(server, queue_manager)

    print("✓ All tools registered successfully")

if __name__ == "__main__":
    test_story_tool_registration()
    test_all_tools_registration()


