"""StoryForge MCP Server implementation."""

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server

from ..shared.errors import StoryForgeError
from ..shared.types import ErrorResponse
from .queue_manager import QueueManager
from .tools import register_all_tools


class StoryForgeMCPServer:
    """MCP server for StoryForge."""

    def __init__(self) -> None:
        """Initialize the MCP server."""
        self.server = Server("storyforge-server")
        self.queue_manager = QueueManager(max_queue_size=10)

        # Register all tools with queue manager
        register_all_tools(self.server, self.queue_manager)

    def format_error(self, error: Exception) -> dict:
        """Format an exception as a structured error response."""
        if isinstance(error, StoryForgeError):
            return ErrorResponse(
                code=error.code,
                message=error.message,
                details=error.details,
                recoverable=error.recoverable,
                recovery_hint=error.recovery_hint,
            ).to_dict()
        else:
            return ErrorResponse(
                code="INTERNAL_ERROR",
                message=str(error),
                details={"type": type(error).__name__},
                recoverable=False,
                recovery_hint=None,
            ).to_dict()

    async def run(self) -> None:
        """Run the MCP server with stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )


async def main() -> None:
    """Main entry point for the MCP server."""
    server = StoryForgeMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
