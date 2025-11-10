"""Image generation tools for StoryForge MCP server."""

from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.types import Tool

from ...checkpoint import CheckpointManager
from ...llm_backend import LLMBackend, get_backend
from ...prompt import Prompt
from ...shared.types import ErrorCode, MCPError
from ..path_resolver import PathResolver


def register_image_tools(server: Server) -> None:
    """Register image generation tools with the MCP server."""
    path_resolver = PathResolver()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available image generation tools."""
        return [
            Tool(
                name="storyforge_generate_images",
                description="Generate additional images for an existing session",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID to generate images for",
                        },
                        "num_images": {
                            "type": "integer",
                            "description": "Number of images to generate",
                            "minimum": 1,
                            "maximum": 10,
                        },
                        "image_style": {
                            "type": "string",
                            "description": "Optional image style override",
                        },
                    },
                    "required": ["session_id", "num_images"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
        """Handle image generation tool calls."""
        try:
            if name == "storyforge_generate_images":
                return await handle_generate_images(arguments, path_resolver)
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


async def handle_generate_images(arguments: dict[str, Any], path_resolver: PathResolver) -> list[Any]:
    """
    Generate additional images for an existing session.

    This tool generates new images based on the story and configuration
    from an existing session, useful for adding more illustrations or
    regenerating images with different styles.

    Returns:
        List with generated image info
    """
    session_id = arguments["session_id"]
    num_images = arguments["num_images"]
    image_style_override = arguments.get("image_style")

    # Validate num_images
    if num_images < 1 or num_images > 10:
        raise MCPError(
            code=ErrorCode.INVALID_PARAMETER,
            message="num_images must be between 1 and 10",
            recoverable=True,
        )

    # Load checkpoint to get story and config
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

        # Get resolved config
        resolved_config = checkpoint.resolved_config
        output_dir = resolved_config.get("output_directory")
        if not output_dir:
            raise MCPError(
                code=ErrorCode.CONFIG_ERROR,
                message="No output directory configured for session",
                recoverable=False,
            )

        # Override image style if provided
        if image_style_override:
            resolved_config["image_style"] = image_style_override

        # Initialize backend
        backend_name = resolved_config.get("backend")
        backend: LLMBackend | None = None
        if backend_name:
            backend = get_backend(backend_name)
        else:
            # Auto-detect available backend
            backend = get_backend()

        if backend is None:
            raise MCPError(
                code=ErrorCode.BACKEND_UNAVAILABLE,
                message="No LLM backend available for image generation",
                recoverable=True,
                recovery_hint="Ensure at least one API key is set (GEMINI_API_KEY, OPENAI_API_KEY)",
            )

        # Reconstruct prompt from original inputs
        original_inputs = checkpoint.original_inputs
        prompt = Prompt(
            prompt=original_inputs.get("prompt", ""),
            characters=resolved_config.get("characters", []),
            theme=resolved_config.get("theme"),
            setting=resolved_config.get("setting"),
            age_range=resolved_config.get("age_range") or "",
            tone=resolved_config.get("tone") or "",
            image_style=resolved_config.get("image_style") or "",
            learning_focus=resolved_config.get("learning_focus"),
        )

        # Generate image prompts from story
        context = resolved_config.get("context", "")
        image_prompts = backend.generate_image_prompt(
            story=story_content,
            context=context,
            num_prompts=num_images,
        )

        if not image_prompts:
            raise MCPError(
                code=ErrorCode.GENERATION_FAILED,
                message="Failed to generate image prompts",
                recoverable=True,
            )

        # Generate images
        generated_images = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get base image name
        image_name = backend.generate_image_name(prompt, story_content)

        # Get existing image count to continue numbering
        existing_images = checkpoint.generated_content.get("generated_images", [])
        start_index = len(existing_images) + 1

        for i, image_prompt in enumerate(image_prompts[:num_images], start_index):
            # Generate image
            image_object, image_bytes = backend.generate_image(prompt, reference_image_bytes=None)

            if image_bytes:
                # Determine image format
                image_format = "png"
                if hasattr(image_object, "format"):
                    fmt = getattr(image_object, "format", None)
                    if fmt:
                        image_format = fmt.lower()

                # Build filename
                image_filename = f"{image_name}_{i:02d}.{image_format}"
                image_path = output_path / image_filename

                # Save image
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                # Record image info
                image_info = {
                    "filename": image_filename,
                    "path": str(image_path),
                    "prompt": image_prompt,
                }
                generated_images.append(image_info)

                # Update checkpoint with new image
                checkpoint.generated_content.setdefault("generated_images", []).append(
                    {
                        "filename": image_filename,
                        "path": str(image_path),
                    }
                )

        # Save updated checkpoint
        checkpoint_manager.save_checkpoint(checkpoint)

        return [{"images": generated_images}]

    except MCPError:
        raise
    except Exception as e:
        raise MCPError(
            code=ErrorCode.GENERATION_FAILED,
            message=f"Failed to generate images: {e}",
            recoverable=False,
        ) from e
