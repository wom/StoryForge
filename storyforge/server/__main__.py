"""MCP server entry point for StoryForge."""

from .mcp_server import main

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
