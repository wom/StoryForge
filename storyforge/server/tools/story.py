"""Story generation tools for StoryForge MCP server."""

from typing import Any

from mcp.server import Server
from mcp.types import Tool

from ...checkpoint import CheckpointData, CheckpointManager, ExecutionPhase
from ...config import load_config
from ...phase_executor import PhaseExecutor
from ...shared.types import ErrorCode, MCPError
from ..path_resolver import PathResolver


def register_story_tools(server: Server) -> None:
    """Register story generation tools with the MCP server."""
    path_resolver = PathResolver()
    checkpoint_manager = CheckpointManager()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available story generation tools."""
        return [
            Tool(
                name="storyforge_generate_story",
                description="Generate a new story with full 11-phase execution",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Story prompt/description",
                        },
                        "age_range": {
                            "type": "string",
                            "description": "Target age range (e.g., '8-10')",
                        },
                        "style": {
                            "type": "string",
                            "description": "Story style (e.g., 'adventure', 'mystery')",
                        },
                        "tone": {
                            "type": "string",
                            "description": "Story tone (e.g., 'light', 'serious')",
                        },
                        "length": {
                            "type": "string",
                            "description": "Story length (e.g., 'short', 'medium', 'long')",
                        },
                        "theme": {
                            "type": "string",
                            "description": "Story theme (optional)",
                        },
                        "characters": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of character names (optional)",
                        },
                        "setting": {
                            "type": "string",
                            "description": "Story setting (optional)",
                        },
                        "learning_focus": {
                            "type": "string",
                            "description": "Educational focus (optional)",
                        },
                        "image_style": {
                            "type": "string",
                            "description": "Image generation style (optional)",
                        },
                        "backend": {
                            "type": "string",
                            "description": "LLM backend to use (optional)",
                        },
                        "output_directory": {
                            "type": "string",
                            "description": "Output directory path (optional)",
                        },
                    },
                    "required": ["prompt", "age_range", "style", "tone", "length"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
        """Handle story generation tool calls."""
        try:
            if name == "storyforge_generate_story":
                return await handle_generate_story(arguments, checkpoint_manager, path_resolver)
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


async def handle_generate_story(
    arguments: dict[str, Any],
    checkpoint_manager: CheckpointManager,
    path_resolver: PathResolver,
) -> list[Any]:
    """
    Handle generate_story tool call.

    This executes the full 11-phase story generation workflow:
    1. INIT - Initialize session
    2. CONFIG_LOAD - Load configuration
    3. BACKEND_INIT - Initialize LLM backend
    4. PROMPT_CONFIRM - Confirm prompt (auto-confirmed in MCP mode)
    5. CONTEXT_LOAD - Load context files
    6. PROMPT_BUILD - Build final prompt
    7. STORY_GENERATE - Generate story
    8. STORY_SAVE - Save story to file
    9. IMAGE_DECISION - Decide on image generation (based on config)
    10. IMAGE_GENERATE - Generate images if enabled
    11. CONTEXT_SAVE - Save context for chaining
    """
    # Extract arguments
    prompt = arguments["prompt"]
    age_range = arguments["age_range"]
    style = arguments["style"]
    tone = arguments["tone"]
    length = arguments["length"]

    # Optional arguments
    theme = arguments.get("theme")
    characters = arguments.get("characters", [])
    setting = arguments.get("setting")
    learning_focus = arguments.get("learning_focus")
    image_style = arguments.get("image_style")
    backend = arguments.get("backend")
    output_directory = arguments.get("output_directory")

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

    # Override config with CLI arguments
    cli_arguments = {
        "prompt": prompt,
        "age_range": age_range,
        "style": style,
        "tone": tone,
        "length": length,
    }

    if theme:
        cli_arguments["theme"] = theme
    if characters:
        cli_arguments["characters"] = characters
    if setting:
        cli_arguments["setting"] = setting
    if learning_focus:
        cli_arguments["learning_focus"] = learning_focus
    if image_style:
        cli_arguments["image_style"] = image_style
    if backend:
        cli_arguments["backend"] = backend
    if output_directory:
        cli_arguments["output_directory"] = output_directory

    # Resolve configuration
    default_backend = "gemini"
    if config.config.has_option("backend", "name"):
        default_backend = config.config.get("backend", "name")

    resolved_config = {
        "backend": backend or default_backend,
        "output_directory": output_directory or str(path_resolver.output_directory),
        "use_context": True,  # Always use context in MCP mode
        "verbose": False,  # No verbose output in MCP mode
        "debug": False,
        "length": length,
        "age_range": age_range,
        "style": style,
        "tone": tone,
        "theme": theme,
        "image_style": image_style,
        "learning_focus": learning_focus,
        "setting": setting,
        "characters": characters,
    }

    # Create initial checkpoint
    checkpoint_data = CheckpointData.create_new(
        original_prompt=prompt,
        cli_arguments=cli_arguments,
        resolved_config=resolved_config,
    )

    # Save initial checkpoint
    checkpoint_manager.save_checkpoint(checkpoint_data)

    # Execute story generation
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

        # Extract results
        story = final_checkpoint.generated_content.get("story")
        images = final_checkpoint.generated_content.get("images", [])
        image_paths = []

        if isinstance(images, list):
            for img in images:
                if isinstance(img, dict):
                    image_paths.append(img.get("path", ""))
                else:
                    image_paths.append(str(img))

        return [
            {
                "session_id": final_checkpoint.session_id,
                "status": final_checkpoint.status,
                "current_phase": final_checkpoint.current_phase,
                "story": story,
                "images": image_paths,
                "checkpoint_path": str(path_resolver.get_checkpoint_path(final_checkpoint.session_id)),
            }
        ]

    except Exception as e:
        # Mark checkpoint as failed
        checkpoint_data.mark_failed(str(e))
        checkpoint_manager.save_checkpoint(checkpoint_data)

        raise MCPError(
            code=ErrorCode.GENERATION_FAILED,
            message=f"Story generation failed: {e}",
            recoverable=True,
            recovery_hint="Check backend configuration and API keys, then retry",
        ) from e
