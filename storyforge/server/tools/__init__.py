"""Tool registration for StoryForge MCP server."""

from mcp.server import Server

from ..queue_manager import QueueManager
from .session import register_session_tools
from .story import register_story_tools


def register_all_tools(server: Server, queue_manager: QueueManager) -> None:
    """Register all MCP tools with the server."""
    register_session_tools(server, queue_manager)
    register_story_tools(server, queue_manager)
    # More tool registrations will be added here
