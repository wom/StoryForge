"""StoryForge MCP Server implementation."""

import asyncio
import logging
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server

from ..shared.errors import StoryForgeError
from ..shared.types import ErrorResponse
from .queue_manager import QueueManager
from .tools import register_all_tools

# Configure debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[SERVER] %(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,  # Use stderr so it doesn't interfere with stdio protocol
)


class StoryForgeMCPServer:
    """MCP server for StoryForge."""

    def __init__(self) -> None:
        """Initialize the MCP server."""
        logging.debug("Initializing StoryForgeMCPServer")
        self.server = Server("storyforge-server")
        self.queue_manager = QueueManager(max_queue_size=10)

        # Register all tools with queue manager
        logging.debug("Registering all tools")
        register_all_tools(self.server, self.queue_manager)
        logging.debug("Server initialization complete")

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
        logging.debug("Starting MCP server with stdio transport")
        try:
            async with stdio_server() as (read_stream, write_stream):
                logging.debug("stdio_server context entered, streams ready")
                logging.debug(f"Read stream: {read_stream}")
                logging.debug(f"Write stream: {write_stream}")
                logging.debug("Starting server.run()")
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options(),
                )
                logging.debug("server.run() completed")
        except Exception as e:
            logging.error(f"Error in server.run(): {e}", exc_info=True)
            raise


async def main() -> None:
    """Main entry point for the MCP server."""
    logging.debug("=== MCP Server Starting ===")
    logging.debug(f"Python: {sys.version}")
    logging.debug(f"CWD: {sys.path[0]}")
    server = StoryForgeMCPServer()
    logging.debug("Calling server.run()")
    await server.run()
    logging.debug("=== MCP Server Exiting ===")


if __name__ == "__main__":
    asyncio.run(main())
