"""Story extension tools for StoryForge MCP server."""

from pathlib import Path
from typing import Any, Literal, cast

from mcp.server import Server
from mcp.types import Tool

from ...checkpoint import CheckpointData, CheckpointManager, ExecutionPhase
from ...config import load_config
from ...context import ContextManager
from ...phase_executor import PhaseExecutor
from ...prompt import Prompt
from ...shared.types import ErrorCode, MCPError
from ..path_resolver import PathResolver
from ..queue_manager import QueueManager


def register_extension_tools(server: Server, queue_manager: QueueManager) -> None:
    """Register story extension tools with the MCP server."""
    path_resolver = PathResolver()
    checkpoint_manager = CheckpointManager()
    context_manager = ContextManager()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available extension tools."""
        return [
            Tool(
                name="storyforge_list_extendable_stories",
                description="List all stories that can be extended",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="storyforge_extend_story",
                description="Extend an existing story with a continuation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_file": {
                            "type": "string",
                            "description": "Path to the context file to extend",
                        },
                        "new_prompt": {
                            "type": "string",
                            "description": "Optional new prompt for the continuation",
                        },
                        "ending_type": {
                            "type": "string",
                            "enum": ["wrap_up", "cliffhanger"],
                            "description": "Type of ending (wrap_up for resolution, cliffhanger for sequel)",
                        },
                        "same_config": {
                            "type": "boolean",
                            "description": "Use same config as original story (default: true)",
                        },
                        "output_directory": {
                            "type": "string",
                            "description": "Optional output directory for extended story",
                        },
                        "backend": {
                            "type": "string",
                            "description": "Optional LLM backend override",
                        },
                    },
                    "required": ["context_file", "ending_type"],
                },
            ),
            Tool(
                name="storyforge_get_story_chain",
                description="Get the complete chain of story extensions",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_file": {
                            "type": "string",
                            "description": "Path to any story in the chain",
                        },
                    },
                    "required": ["context_file"],
                },
            ),
            Tool(
                name="storyforge_export_chain",
                description="Export a complete story chain as a single file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_file": {
                            "type": "string",
                            "description": "Path to any story in the chain",
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Optional output path for exported file",
                        },
                    },
                    "required": ["context_file"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
        """Handle extension tool calls."""
        try:
            if name == "storyforge_list_extendable_stories":
                return await handle_list_extendable_stories()
            elif name == "storyforge_extend_story":
                # Generate session ID for queue management
                import uuid

                session_id = str(uuid.uuid4())

                # Enqueue the extension request
                future = await queue_manager.enqueue(
                    session_id=session_id,
                    handler=handle_extend_story,
                    arguments=arguments,
                    checkpoint_manager=checkpoint_manager,
                    path_resolver=path_resolver,
                    context_manager=context_manager,
                    session_id_override=session_id,
                )

                # Wait for completion
                result: list[Any] = await future
                return result
            elif name == "storyforge_get_story_chain":
                return handle_get_story_chain(arguments, context_manager)
            elif name == "storyforge_export_chain":
                return handle_export_chain(arguments, context_manager, path_resolver)
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


async def handle_list_extendable_stories() -> list[Any]:
    """
    List all stories that can be extended.

    Returns:
        List with stories array containing available context files
    """
    context_manager = ContextManager()

    available_contexts = context_manager.list_available_contexts()

    stories = []
    for ctx in available_contexts:
        # Get chain info
        chain = context_manager.get_story_chain(ctx["filepath"])

        story_info = {
            "context_file": str(ctx["filepath"]),
            "title": ctx.get("filename", ""),
            "prompt": ctx.get("prompt", "")[:100],  # First 100 chars
            "generated_at": ctx.get("timestamp", ""),
            "has_chain": len(chain) > 1,
            "chain_length": len(chain),
        }
        stories.append(story_info)

    return [{"stories": stories}]


async def handle_extend_story(
    arguments: dict[str, Any],
    checkpoint_manager: CheckpointManager,
    path_resolver: PathResolver,
    context_manager: ContextManager,
    session_id_override: str | None = None,
) -> list[Any]:
    """
    Handle extend_story tool call.

    Extends an existing story with a continuation.
    """
    # Extract arguments
    context_file = Path(arguments["context_file"])
    new_prompt = arguments.get("new_prompt")
    ending_type_str = arguments["ending_type"]
    # same_config = arguments.get("same_config", True)  # Reserved for future use
    output_directory = arguments.get("output_directory")
    backend = arguments.get("backend")

    # Validate ending type
    if ending_type_str not in ["wrap_up", "cliffhanger"]:
        raise MCPError(
            code=ErrorCode.INVALID_PARAMETER,
            message=f"Invalid ending_type: {ending_type_str}. Must be 'wrap_up' or 'cliffhanger'",
            recoverable=True,
            recovery_hint="Use ending_type='wrap_up' or ending_type='cliffhanger'",
        )

    ending_type = cast(Literal["wrap_up", "cliffhanger"], ending_type_str)

    # Check if context file exists
    if not context_file.exists():
        raise MCPError(
            code=ErrorCode.CONTEXT_FILE_NOT_FOUND,
            message=f"Context file not found: {context_file}",
            recoverable=True,
            recovery_hint="Use storyforge_list_extendable_stories to see available stories",
        )

    # Load story content and metadata
    try:
        story_content, metadata = context_manager.load_context_for_extension(context_file)
    except Exception as e:
        raise MCPError(
            code=ErrorCode.FILE_IO_ERROR,
            message=f"Failed to load context file: {e}",
            recoverable=False,
        ) from e

    # Parse characters from metadata
    characters_value = metadata.get("characters", [])
    if isinstance(characters_value, str):
        characters = [c.strip() for c in characters_value.split(",") if c.strip()]
    else:
        characters = characters_value if characters_value else []

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        raise MCPError(
            code=ErrorCode.CONFIG_ERROR,
            message=f"Failed to load configuration: {e}",
            recoverable=True,
            recovery_hint="Check configuration file exists and is valid",
        ) from e

    # Build prompt for continuation
    prompt = Prompt(
        prompt=new_prompt or "",  # Optional new direction
        characters=characters if characters else None,
        theme=metadata.get("theme"),
        age_range=str(metadata.get("age_group") or config.get_field_value("story", "age_range") or "preschool"),
        tone=str(metadata.get("tone") or config.get_field_value("story", "tone") or "heartwarming"),
        length=str(config.get_field_value("story", "length") or "short"),
        style=str(config.get_field_value("story", "style") or "adventure"),
        image_style=str(metadata.get("art_style") or config.get_field_value("story", "image_style") or "chibi"),
        context=story_content,
        continuation_mode=True,
        ending_type=ending_type,
    )

    # Generate output directory with _extended suffix
    if not output_directory:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = f"storyforge_output_{timestamp}_extended"

    # Prepare CLI arguments for checkpoint
    cli_arguments = {
        "continuation_mode": True,
        "ending_type": ending_type,
        "output_dir": output_directory,
        "age_range": prompt.age_range,
        "length": prompt.length,
        "style": prompt.style,
        "tone": prompt.tone,
        "image_style": prompt.image_style,
        "theme": prompt.theme,
        "characters": prompt.characters,
        "context_file": str(context_file),
        "backend": backend,
    }

    # Resolve configuration
    default_backend = "gemini"
    if config.config.has_option("backend", "name"):
        default_backend = config.config.get("backend", "name")

    resolved_config = {
        "backend": backend or default_backend,
        "output_directory": output_directory,
        "use_context": True,
        "verbose": False,
        "debug": False,
        "length": prompt.length,
        "age_range": prompt.age_range,
        "style": prompt.style,
        "tone": prompt.tone,
        "theme": prompt.theme,
        "image_style": prompt.image_style,
        "learning_focus": metadata.get("learning_focus"),
        "setting": metadata.get("setting"),
        "characters": prompt.characters,
        "continuation_mode": True,
        "ending_type": ending_type,
    }

    # Create initial checkpoint
    checkpoint_data = CheckpointData.create_new(
        original_prompt=new_prompt or f"Continuation of {context_file.name}",
        cli_arguments=cli_arguments,
        resolved_config=resolved_config,
    )

    # Override session_id if provided (for queue management)
    if session_id_override:
        checkpoint_data.session_id = session_id_override

    # Save initial checkpoint
    checkpoint_manager.save_checkpoint(checkpoint_data)

    # Execute story extension
    try:
        executor = PhaseExecutor(checkpoint_manager)
        executor.execute_from_checkpoint(checkpoint_data, ExecutionPhase.INIT)

        # Reload checkpoint to get final state
        final_checkpoint = checkpoint_manager.load_checkpoint(
            path_resolver.get_checkpoint_path(checkpoint_data.session_id)
        )

        if final_checkpoint is None:
            raise MCPError(
                code=ErrorCode.CHECKPOINT_CORRUPT,
                message="Failed to load final checkpoint",
                recoverable=False,
            )

        # Build result
        result = {
            "session_id": checkpoint_data.session_id,
            "parent_story": str(context_file),
            "status": "completed" if final_checkpoint.current_phase == ExecutionPhase.COMPLETED.value else "failed",
            "story": final_checkpoint.generated_content.get("story"),
            "images": final_checkpoint.generated_content.get("images", []),
            "checkpoint_path": str(path_resolver.get_checkpoint_path(checkpoint_data.session_id)),
        }

        return [result]

    except Exception as e:
        raise MCPError(
            code=ErrorCode.GENERATION_FAILED,
            message=f"Story extension failed: {e}",
            recoverable=False,
        ) from e


def handle_get_story_chain(arguments: dict[str, Any], context_manager: ContextManager) -> list[Any]:
    """
    Get the complete story chain for a context file.

    Returns:
        List with chain array containing all stories in the chain
    """
    context_file = Path(arguments["context_file"])

    if not context_file.exists():
        raise MCPError(
            code=ErrorCode.CONTEXT_FILE_NOT_FOUND,
            message=f"Context file not found: {context_file}",
            recoverable=True,
            recovery_hint="Use storyforge_list_extendable_stories to see available stories",
        )

    try:
        chain = context_manager.get_story_chain(context_file)

        chain_data = []
        for idx, story in enumerate(chain, 1):
            # Read story preview
            story_preview = ""
            story_path = story.get("filepath")
            if story_path and Path(story_path).exists():
                with open(story_path, encoding="utf-8") as f:
                    content = f.read()
                    # Extract story content (skip metadata)
                    if "## Story" in content:
                        story_text = content.split("## Story", 1)[1]
                        # Get first 200 words
                        story_preview = " ".join(story_text.split()[:200])

            chain_data.append(
                {
                    "part_number": idx,
                    "context_file": str(story.get("filepath", "")),
                    "prompt": story.get("prompt", "")[:100],
                    "generated_at": story.get("timestamp", ""),
                    "story_preview": story_preview,
                }
            )

        return [{"chain": chain_data}]

    except Exception as e:
        raise MCPError(
            code=ErrorCode.FILE_IO_ERROR,
            message=f"Failed to get story chain: {e}",
            recoverable=False,
        ) from e


def handle_export_chain(
    arguments: dict[str, Any], context_manager: ContextManager, path_resolver: PathResolver
) -> list[Any]:
    """
    Export a complete story chain as a single file.

    Returns:
        List with export info (path, total parts, word count)
    """
    context_file = Path(arguments["context_file"])
    output_path = arguments.get("output_path")

    if not context_file.exists():
        raise MCPError(
            code=ErrorCode.CONTEXT_FILE_NOT_FOUND,
            message=f"Context file not found: {context_file}",
            recoverable=True,
            recovery_hint="Use storyforge_list_extendable_stories to see available stories",
        )

    try:
        # Generate default output path if not provided
        if not output_path:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"complete_story_{context_file.stem}_{timestamp}.txt"

        output_file = Path(output_path)

        # Write chain to file
        result_path = context_manager.write_chain_to_file(context_file, output_file)

        # Get chain info for response
        chain = context_manager.get_story_chain(context_file)

        # Calculate total word count
        total_words = 0
        for story in chain:
            story_path = story.get("filepath")
            if story_path and Path(story_path).exists():
                with open(story_path, encoding="utf-8") as f:
                    content = f.read()
                    if "## Story" in content:
                        story_text = content.split("## Story", 1)[1]
                        total_words += len(story_text.split())

        return [
            {
                "export_path": str(result_path),
                "total_parts": len(chain),
                "combined_word_count": total_words,
            }
        ]

    except Exception as e:
        raise MCPError(
            code=ErrorCode.FILE_IO_ERROR,
            message=f"Failed to export chain: {e}",
            recoverable=False,
        ) from e
