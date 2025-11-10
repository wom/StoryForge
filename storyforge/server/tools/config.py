"""Configuration management tools for StoryForge MCP server."""

from typing import Any

from mcp.server import Server
from mcp.types import Tool

from ...checkpoint import CheckpointManager
from ...config import Config
from ...llm_backend import list_available_backends
from ...shared.types import ErrorCode, MCPError
from ..path_resolver import PathResolver


def register_config_tools(server: Server) -> None:
    """Register configuration management tools with the MCP server."""
    path_resolver = PathResolver()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available configuration management tools."""
        return [
            Tool(
                name="storyforge_list_backends",
                description="List all available LLM backends with their status and capabilities",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="storyforge_get_config",
                description="Get the current StoryForge configuration",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="storyforge_update_session_backend",
                description="Update the LLM backend for a running session",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID to update",
                        },
                        "new_backend": {
                            "type": "string",
                            "description": "New backend name (gemini, openai, anthropic)",
                        },
                    },
                    "required": ["session_id", "new_backend"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
        """Handle configuration management tool calls."""
        try:
            if name == "storyforge_list_backends":
                return handle_list_backends()
            elif name == "storyforge_get_config":
                return handle_get_config()
            elif name == "storyforge_update_session_backend":
                return handle_update_session_backend(arguments, path_resolver)
            else:
                raise ValueError(f"Unknown tool: {name}")
        except MCPError:
            raise
        except Exception as e:
            raise MCPError(
                code=ErrorCode.INTERNAL_ERROR,
                message=str(e),
                recoverable=False,
            ) from e


def handle_list_backends() -> list[Any]:
    """
    List all available LLM backends with their status and capabilities.

    Returns:
        List with backends array containing backend info
    """
    available_backends = list_available_backends()

    backends_list = []
    for backend_name, backend_info in available_backends.items():
        # Determine capabilities based on backend type
        capabilities = {
            "story_generation": True,  # All backends support story generation
            "image_generation": backend_name in ["gemini", "openai"],  # Anthropic doesn't support images
        }

        backend_data = {
            "name": backend_name,
            "available": backend_info["available"],
            "api_key_set": backend_info["available"],  # If available, API key must be set
            "capabilities": capabilities,
            "reason": backend_info.get("reason", ""),
        }
        backends_list.append(backend_data)

    return [{"backends": backends_list}]


def handle_get_config() -> list[Any]:
    """
    Get the current StoryForge configuration.

    Returns:
        List with config dict and config_path
    """
    try:
        config = Config()
        config.load_config()

        # Build config dictionary from all sections
        config_dict: dict[str, Any] = {}
        for section in config.config.sections():
            config_dict[section] = dict(config.config.items(section))

        # Get config path if loaded from file
        config_path = str(config.config_path) if config.config_path else "defaults (no config file loaded)"

        return [
            {
                "config": config_dict,
                "config_path": config_path,
            }
        ]

    except Exception as e:
        raise MCPError(
            code=ErrorCode.CONFIG_ERROR,
            message=f"Failed to load configuration: {e}",
            recoverable=True,
            recovery_hint="Check configuration file syntax and permissions",
        ) from e


def handle_update_session_backend(arguments: dict[str, Any], path_resolver: PathResolver) -> list[Any]:
    """
    Update the LLM backend for a running session.

    This allows switching backends mid-session, useful for:
    - Using different backends for story vs refinement
    - Switching to a backend with image generation capabilities
    - Falling back to a different backend if one fails

    Returns:
        List with updated status and backend name
    """
    session_id = arguments["session_id"]
    new_backend = arguments["new_backend"].lower()

    # Validate backend name
    valid_backends = ["gemini", "openai", "anthropic"]
    if new_backend not in valid_backends:
        raise MCPError(
            code=ErrorCode.INVALID_PARAMETER,
            message=f"Invalid backend: {new_backend}. Must be one of: {', '.join(valid_backends)}",
            recoverable=True,
        )

    # Check if backend is available
    available_backends = list_available_backends()
    if new_backend not in available_backends or not available_backends[new_backend]["available"]:
        reason = available_backends.get(new_backend, {}).get("reason", "Unknown")
        raise MCPError(
            code=ErrorCode.BACKEND_UNAVAILABLE,
            message=f"Backend '{new_backend}' is not available: {reason}",
            recoverable=True,
            recovery_hint="Use storyforge_list_backends to see available backends",
        )

    # Load checkpoint
    checkpoint_manager = CheckpointManager()
    checkpoint_path = path_resolver.get_checkpoint_path(session_id)

    if not checkpoint_path.exists():
        raise MCPError(
            code=ErrorCode.SESSION_NOT_FOUND,
            message=f"Session not found: {session_id}",
            recoverable=True,
            recovery_hint="Use storyforge_list_sessions to see available sessions",
        )

    try:
        checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)
        if checkpoint is None:
            raise MCPError(
                code=ErrorCode.CHECKPOINT_CORRUPT,
                message=f"Failed to load checkpoint for session: {session_id}",
                recoverable=False,
            )

        # Update backend in resolved config
        checkpoint.resolved_config["backend"] = new_backend

        # Save updated checkpoint
        checkpoint_manager.save_checkpoint(checkpoint)

        return [
            {
                "updated": True,
                "backend": new_backend,
                "session_id": session_id,
            }
        ]

    except MCPError:
        raise
    except Exception as e:
        raise MCPError(
            code=ErrorCode.CONFIG_ERROR,
            message=f"Failed to update session backend: {e}",
            recoverable=False,
        ) from e
