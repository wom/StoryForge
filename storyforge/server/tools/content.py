"""Content management tools for StoryForge MCP server."""

from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.types import Tool

from ...context import ContextManager
from ...shared.types import ErrorCode, MCPError
from ..path_resolver import PathResolver


def register_content_tools(server: Server) -> None:
    """Register content management tools with the MCP server."""
    path_resolver = PathResolver()
    context_manager = ContextManager()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available content management tools."""
        return [
            Tool(
                name="storyforge_list_context_files",
                description="List all context files with optional filtering",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filter": {
                            "type": "object",
                            "properties": {
                                "has_chain": {
                                    "type": "boolean",
                                    "description": "Filter for stories that are part of a chain",
                                },
                                "min_date": {
                                    "type": "string",
                                    "description": "Minimum date (ISO format)",
                                },
                                "max_chain_length": {
                                    "type": "integer",
                                    "description": "Maximum chain length",
                                },
                                "search": {
                                    "type": "string",
                                    "description": "Search text in prompts/titles",
                                },
                            },
                        },
                    },
                },
            ),
            Tool(
                name="storyforge_get_context_content",
                description="Get the full content of a context file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_file": {
                            "type": "string",
                            "description": "Path to context file",
                        },
                    },
                    "required": ["context_file"],
                },
            ),
            Tool(
                name="storyforge_save_as_context",
                description="Save a session's story as a context file for future extension",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID to save",
                        },
                        "filename": {
                            "type": "string",
                            "description": "Optional custom filename",
                        },
                    },
                    "required": ["session_id"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
        """Handle content management tool calls."""
        try:
            if name == "storyforge_list_context_files":
                return handle_list_context_files(arguments, context_manager)
            elif name == "storyforge_get_context_content":
                return handle_get_context_content(arguments, context_manager)
            elif name == "storyforge_save_as_context":
                return handle_save_as_context(arguments, path_resolver, context_manager)
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


def handle_list_context_files(arguments: dict[str, Any], context_manager: ContextManager) -> list[Any]:
    """
    List all context files with optional filtering.

    Returns:
        List with files array containing context file info
    """
    filters = arguments.get("filter", {})

    # Get all available contexts
    available_contexts = context_manager.list_available_contexts()

    # Apply filters
    filtered_contexts = []
    for ctx in available_contexts:
        # Filter by has_chain
        if "has_chain" in filters:
            chain = context_manager.get_story_chain(ctx["filepath"])
            has_chain = len(chain) > 1
            if filters["has_chain"] != has_chain:
                continue

        # Filter by max_chain_length
        if "max_chain_length" in filters:
            chain = context_manager.get_story_chain(ctx["filepath"])
            if len(chain) > filters["max_chain_length"]:
                continue

        # Filter by search term
        if "search" in filters:
            search_term = filters["search"].lower()
            prompt = ctx.get("prompt", "").lower()
            filename = ctx.get("filename", "").lower()
            if search_term not in prompt and search_term not in filename:
                continue

        # Filter by min_date (if provided)
        if "min_date" in filters:
            # Simple string comparison (ISO format sorts correctly)
            ctx_date = ctx.get("timestamp", "")
            if ctx_date < filters["min_date"]:
                continue

        # Build file info
        filepath = Path(ctx["filepath"])
        file_info = {
            "filename": ctx.get("filename", ""),
            "path": str(filepath),
            "size": filepath.stat().st_size if filepath.exists() else 0,
            "modified": ctx.get("timestamp", ""),
            "metadata": {
                "prompt": ctx.get("prompt", ""),
                "generated_at": ctx.get("timestamp", ""),
                "extended_from": ctx.get("extended_from"),
            },
        }
        filtered_contexts.append(file_info)

    return [{"files": filtered_contexts}]


def handle_get_context_content(arguments: dict[str, Any], context_manager: ContextManager) -> list[Any]:
    """
    Get the full content of a context file.

    Returns:
        List with metadata and story content
    """
    context_file = Path(arguments["context_file"])

    if not context_file.exists():
        raise MCPError(
            code=ErrorCode.CONTEXT_FILE_NOT_FOUND,
            message=f"Context file not found: {context_file}",
            recoverable=True,
            recovery_hint="Use storyforge_list_context_files to see available files",
        )

    try:
        # Load the context content
        story_content, metadata = context_manager.load_context_for_extension(context_file)

        # Extract just the story part (skip metadata header)
        story_text = story_content
        if "## Story" in story_content:
            story_text = story_content.split("## Story", 1)[1].strip()

        return [
            {
                "metadata": metadata,
                "story": story_text,
            }
        ]

    except Exception as e:
        raise MCPError(
            code=ErrorCode.FILE_IO_ERROR,
            message=f"Failed to read context file: {e}",
            recoverable=False,
        ) from e


def handle_save_as_context(
    arguments: dict[str, Any],
    path_resolver: PathResolver,
    context_manager: ContextManager,
) -> list[Any]:
    """
    Save a session's story as a context file for future extension.

    Returns:
        List with context file path info
    """
    session_id = arguments["session_id"]
    custom_filename = arguments.get("filename")

    # Load checkpoint to get story content
    from ...checkpoint import CheckpointManager

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

        # Get story content
        story_content = checkpoint.generated_content.get("story")
        if not story_content:
            raise MCPError(
                code=ErrorCode.GENERATION_FAILED,
                message="No story content found in session",
                recoverable=False,
            )

        # Build metadata for context file
        original_inputs = checkpoint.original_inputs
        resolved_config = checkpoint.resolved_config

        metadata = {
            "prompt": original_inputs.get("prompt", ""),
            "characters": resolved_config.get("characters", []),
            "theme": resolved_config.get("theme"),
            "setting": resolved_config.get("setting"),
            "age_group": resolved_config.get("age_range"),
            "tone": resolved_config.get("tone"),
            "art_style": resolved_config.get("image_style"),
            "learning_focus": resolved_config.get("learning_focus"),
        }

        # Generate filename if not provided
        if not custom_filename:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_slug = original_inputs.get("prompt", "story")[:30].replace(" ", "_")
            custom_filename = f"context_{prompt_slug}_{timestamp}.md"

        # Ensure .md extension
        if not custom_filename.endswith(".md"):
            custom_filename += ".md"

        # Save as context file
        context_dir = path_resolver.context_dir
        context_file = context_dir / custom_filename

        # Build context file content
        context_content = _build_context_content(metadata, story_content)

        # Write context file
        context_file.write_text(context_content, encoding="utf-8")

        return [
            {
                "context_file": custom_filename,
                "path": str(context_file),
            }
        ]

    except MCPError:
        raise
    except Exception as e:
        raise MCPError(
            code=ErrorCode.FILE_IO_ERROR,
            message=f"Failed to save context file: {e}",
            recoverable=False,
        ) from e


def _build_context_content(metadata: dict[str, Any], story: str) -> str:
    """Build formatted context file content."""
    from datetime import datetime

    lines = []

    # Add metadata header
    lines.append("# Story Context")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    if metadata.get("prompt"):
        lines.append(f"**Prompt:** {metadata['prompt']}")
        lines.append("")

    if metadata.get("characters"):
        chars = metadata["characters"]
        if isinstance(chars, list):
            chars = ", ".join(chars)
        lines.append(f"**Characters:** {chars}")
        lines.append("")

    if metadata.get("theme"):
        lines.append(f"**Theme:** {metadata['theme']}")
        lines.append("")

    if metadata.get("setting"):
        lines.append(f"**Setting:** {metadata['setting']}")
        lines.append("")

    if metadata.get("age_group"):
        lines.append(f"**Age Group:** {metadata['age_group']}")
        lines.append("")

    if metadata.get("tone"):
        lines.append(f"**Tone:** {metadata['tone']}")
        lines.append("")

    if metadata.get("art_style"):
        lines.append(f"**Art Style:** {metadata['art_style']}")
        lines.append("")

    if metadata.get("learning_focus"):
        lines.append(f"**Learning Focus:** {metadata['learning_focus']}")
        lines.append("")

    # Add story content
    lines.append("---")
    lines.append("")
    lines.append("## Story")
    lines.append("")
    lines.append(story)
    lines.append("")

    return "\n".join(lines)
