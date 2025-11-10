"""Tool registration for StoryForge MCP server."""

from mcp.server import Server

from .session import register_session_tools
from .story import register_story_tools


def register_all_tools(server: Server) -> None:
    """Register all MCP tools with the server."""
    register_session_tools(server)
    register_story_tools(server)
    # More tool registrations will be added here
