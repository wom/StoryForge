"""MCP client wrapper for StoryForge thin CLI."""

import asyncio
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """
    Thin MCP client for communicating with StoryForge server.

    Manages connection lifecycle and provides wrapper methods for all MCP tools.
    """

    def __init__(self):
        """Initialize MCP client (connection established on first use)."""
        self.session: ClientSession | None = None
        self._read_stream = None
        self._write_stream = None

    async def connect(self) -> None:
        """Establish connection to MCP server."""
        if self.session is not None:
            return  # Already connected

        # Server parameters for stdio transport
        # Use sys.executable to ensure we use the same Python interpreter
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "storyforge.server.mcp_server"],
            env=None,
        )

        # Create stdio client (it's a context manager)
        stdio_transport = stdio_client(server_params)
        self._read_stream, self._write_stream = await stdio_transport.__aenter__()
        self.session = ClientSession(self._read_stream, self._write_stream)

        # Initialize session
        await self.session.initialize()

    async def disconnect(self) -> None:
        """Close connection to MCP server."""
        if self.session:
            await self.session.__aexit__(None, None, None)
            self.session = None

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Call an MCP tool and return the result.

        Args:
            tool_name: Name of the tool to call
            arguments: Dictionary of tool arguments

        Returns:
            Tool result (typically a list with response data)

        Raises:
            Exception: If tool call fails or connection issues occur
        """
        if self.session is None:
            await self.connect()

        assert self.session is not None, "Session must be initialized"

        try:
            result = await self.session.call_tool(tool_name, arguments)
            return result.content if hasattr(result, "content") else result
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"Failed to call tool '{tool_name}': {e}") from e

    # Session Management Tools

    async def list_sessions(
        self,
        status_filter: str | None = None,
        min_date: str | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        """List all sessions with optional filtering."""
        filters = {}
        if status_filter:
            filters["status"] = status_filter
        if min_date:
            filters["min_date"] = min_date
        if search:
            filters["search"] = search

        args = {"filter": filters} if filters else {}
        result = await self.call_tool("storyforge_list_sessions", args)
        return result[0]["sessions"] if result else []

    async def get_session_status(self, session_id: str) -> dict[str, Any]:
        """Get status of a specific session."""
        result = await self.call_tool("storyforge_get_session_status", {"session_id": session_id})
        return result[0] if result else {}

    async def continue_session(self, session_id: str, from_phase: str | None = None) -> dict[str, Any]:
        """Continue a session from a specific phase."""
        args = {"session_id": session_id}
        if from_phase:
            args["from_phase"] = from_phase
        result = await self.call_tool("storyforge_continue_session", args)
        return result[0] if result else {}

    async def delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete a session and its checkpoint."""
        result = await self.call_tool("storyforge_delete_session", {"session_id": session_id})
        return result[0] if result else {}

    async def get_queue_status(self) -> dict[str, Any]:
        """Get current queue status."""
        result = await self.call_tool("storyforge_get_queue_status", {})
        return result[0] if result else {}

    # Story Generation

    async def generate_story(
        self,
        prompt: str,
        length: str | None = None,
        age_range: str | None = None,
        style: str | None = None,
        tone: str | None = None,
        theme: str | None = None,
        learning_focus: str | None = None,
        setting: str | None = None,
        characters: list[str] | None = None,
        image_style: str | None = None,
        output_directory: str | None = None,
        use_context: bool | None = None,
        backend: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate a new story."""
        args: dict[str, Any] = {"prompt": prompt}

        # Add optional parameters
        if length is not None:
            args["length"] = length
        if age_range is not None:
            args["age_range"] = age_range
        if style is not None:
            args["style"] = style
        if tone is not None:
            args["tone"] = tone
        if theme is not None:
            args["theme"] = theme
        if learning_focus is not None:
            args["learning_focus"] = learning_focus
        if setting is not None:
            args["setting"] = setting
        if characters is not None:
            args["characters"] = characters
        if image_style is not None:
            args["image_style"] = image_style
        if output_directory is not None:
            args["output_directory"] = output_directory
        if use_context is not None:
            args["use_context"] = use_context
        if backend is not None:
            args["backend"] = backend
        if session_id is not None:
            args["session_id"] = session_id

        result = await self.call_tool("storyforge_generate_story", args)
        return result[0] if result else {}

    # Extension Tools

    async def list_extendable_stories(self) -> list[dict[str, Any]]:
        """List stories that can be extended."""
        result = await self.call_tool("storyforge_list_extendable_stories", {})
        return result[0]["stories"] if result else []

    async def extend_story(
        self,
        context_file: str,
        ending_type: str = "cliffhanger",
        output_directory: str | None = None,
    ) -> dict[str, Any]:
        """Extend an existing story."""
        args: dict[str, Any] = {
            "context_file": context_file,
            "ending_type": ending_type,
        }
        if output_directory:
            args["output_directory"] = output_directory

        result = await self.call_tool("storyforge_extend_story", args)
        return result[0] if result else {}

    async def get_story_chain(self, context_file: str) -> dict[str, Any]:
        """Get the complete chain for a story."""
        result = await self.call_tool("storyforge_get_story_chain", {"context_file": context_file})
        return result[0] if result else {}

    async def export_chain(self, context_file: str, output_file: str | None = None) -> dict[str, Any]:
        """Export a story chain to a single file."""
        args: dict[str, Any] = {"context_file": context_file}
        if output_file:
            args["output_file"] = output_file

        result = await self.call_tool("storyforge_export_chain", args)
        return result[0] if result else {}

    # Content Management

    async def list_context_files(
        self,
        has_chain: bool | None = None,
        min_date: str | None = None,
        max_chain_length: int | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        """List context files with optional filtering."""
        filters: dict[str, Any] = {}
        if has_chain is not None:
            filters["has_chain"] = has_chain
        if min_date:
            filters["min_date"] = min_date
        if max_chain_length is not None:
            filters["max_chain_length"] = max_chain_length
        if search:
            filters["search"] = search

        args = {"filter": filters} if filters else {}
        result = await self.call_tool("storyforge_list_context_files", args)
        return result[0]["files"] if result else []

    async def get_context_content(self, context_file: str) -> dict[str, Any]:
        """Get content of a context file."""
        result = await self.call_tool("storyforge_get_context_content", {"context_file": context_file})
        return result[0] if result else {}

    async def save_as_context(self, session_id: str, filename: str | None = None) -> dict[str, Any]:
        """Save a session as a context file."""
        args: dict[str, Any] = {"session_id": session_id}
        if filename:
            args["filename"] = filename

        result = await self.call_tool("storyforge_save_as_context", args)
        return result[0] if result else {}

    # Image Generation

    async def generate_images(
        self,
        session_id: str,
        num_images: int,
        image_style: str | None = None,
    ) -> dict[str, Any]:
        """Generate additional images for a session."""
        args: dict[str, Any] = {
            "session_id": session_id,
            "num_images": num_images,
        }
        if image_style:
            args["image_style"] = image_style

        result = await self.call_tool("storyforge_generate_images", args)
        return result[0] if result else {}

    # Configuration

    async def list_backends(self) -> list[dict[str, Any]]:
        """List available LLM backends."""
        result = await self.call_tool("storyforge_list_backends", {})
        return result[0]["backends"] if result else []

    async def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        result = await self.call_tool("storyforge_get_config", {})
        return result[0] if result else {}

    async def update_session_backend(self, session_id: str, new_backend: str) -> dict[str, Any]:
        """Update the backend for a session."""
        result = await self.call_tool(
            "storyforge_update_session_backend",
            {"session_id": session_id, "new_backend": new_backend},
        )
        return result[0] if result else {}

    # Refinement

    async def refine_story(
        self,
        session_id: str,
        refinement_instructions: str,
        backend: str | None = None,
    ) -> dict[str, Any]:
        """Refine a story with specific instructions."""
        args: dict[str, Any] = {
            "session_id": session_id,
            "refinement_instructions": refinement_instructions,
        }
        if backend:
            args["backend"] = backend

        result = await self.call_tool("storyforge_refine_story", args)
        return result[0] if result else {}


# Global client instance (lazy initialization)
_client: MCPClient | None = None


def get_client() -> MCPClient:
    """Get the global MCP client instance."""
    global _client
    if _client is None:
        _client = MCPClient()
    return _client


async def run_async(coro):
    """Helper to run async coroutines from sync code."""
    return await coro


def run_sync(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)
