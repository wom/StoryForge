"""StoryForge thin CLI client."""

from .mcp_client import MCPClient, get_client, run_sync

__all__ = ["MCPClient", "get_client", "run_sync"]
