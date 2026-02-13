#!/usr/bin/env python3
"""Test FastMCP server directly without client."""

import sys
sys.path.insert(0, "/home/wom/src/StoryForge")

from storyforge.server.fastmcp_server import mcp

if __name__ == "__main__":
    print("Starting FastMCP server test")
    # Try to list tools
    print("Testing server startup...")
    mcp.run()
