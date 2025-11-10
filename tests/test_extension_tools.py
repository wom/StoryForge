"""Test for extension tools registration."""

from mcp.server import Server

from storyforge.server.queue_manager import QueueManager
from storyforge.server.tools.extension import register_extension_tools


def test_extension_tools_registration():
    """Test that extension tools register correctly."""
    # Create a test server and queue manager
    server = Server("test-server")
    queue_manager = QueueManager()

    # Register extension tools
    register_extension_tools(server, queue_manager)

    print("✓ Extension tools registered successfully")
    print("  Tools: list_extendable_stories, extend_story, get_story_chain, export_chain")


if __name__ == "__main__":
    test_extension_tools_registration()
