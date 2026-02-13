#!/usr/bin/env python3
"""
Quick MCP server inspector script.

Tests the server in isolation without needing the client.
Based on MCP SDK testing patterns.
"""

import asyncio
import json

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def inspect_server():
    """Inspect the MCP server to see what tools are registered."""
    print("=" * 80)
    print("MCP Server Inspector")
    print("=" * 80)
    print()

    # Connect to server via stdio
    server_params = StdioServerParameters(
        command="python3",
        args=["-m", "storyforge.server"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize connection
            print("📡 Initializing connection...")
            await session.initialize()
            print("✅ Connected successfully!")
            print()

            # List all tools
            print("🔧 Available Tools:")
            print("-" * 80)
            tools_response = await session.list_tools()
            
            if not tools_response.tools:
                print("⚠️  NO TOOLS REGISTERED!")
            else:
                for i, tool in enumerate(tools_response.tools, 1):
                    print(f"\n{i}. {tool.name}")
                    if tool.description:
                        print(f"   Description: {tool.description}")
                    print(f"   Input Schema: {json.dumps(tool.inputSchema, indent=2)}")
            
            print()
            print("=" * 80)
            print(f"Total tools registered: {len(tools_response.tools)}")
            print("=" * 80)


if __name__ == "__main__":
    asyncio.run(inspect_server())
