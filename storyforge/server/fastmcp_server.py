"""StoryForge FastMCP Server implementation.

This server uses the FastMCP framework to expose all StoryForge functionality
via the Model Context Protocol. It follows best practices from the MCP Python SDK:

- Uses @mcp.tool() decorator for automatic tool registration
- Implements lifespan management for shared resources
- Uses Context injection for logging and progress reporting
- Provides typed error handling and proper async patterns
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from ..checkpoint import CheckpointData, CheckpointManager, ExecutionPhase
from ..config import load_config
from ..phase_executor import PhaseExecutor
from ..shared.types import ErrorCode, MCPError
from .path_resolver import PathResolver
from .queue_manager import QueueManager


@dataclass
class AppContext:
    """Application context with shared resources.

    This context is initialized during server startup and made available
    to all tools via dependency injection.
    """

    checkpoint_manager: CheckpointManager
    path_resolver: PathResolver
    queue_manager: QueueManager


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with resource initialization and cleanup.

    This lifespan manager ensures that shared resources are properly initialized
    on server startup and cleaned up on shutdown.
    """
    # Startup: Initialize shared resources
    checkpoint_manager = CheckpointManager(auto_cleanup=False)  # Disable auto_cleanup for MCP (silent mode)
    path_resolver = PathResolver()
    queue_manager = QueueManager(max_queue_size=10)

    try:
        yield AppContext(
            checkpoint_manager=checkpoint_manager,
            path_resolver=path_resolver,
            queue_manager=queue_manager,
        )
    finally:
        # Shutdown: Cleanup resources
        # QueueManager and other resources can perform cleanup here if needed
        pass


# Create FastMCP server with lifespan management
mcp = FastMCP(
    "StoryForge",
    lifespan=app_lifespan,
)


# ===== STORY GENERATION TOOLS =====


@mcp.tool()
async def storyforge_generate_story(
    prompt: str,
    age_range: str,
    style: str,
    tone: str,
    length: str,
    theme: str | None = None,
    characters: list[str] | None = None,
    setting: str | None = None,
    learning_focus: str | None = None,
    image_style: str | None = None,
    backend: str | None = None,
    output_directory: str | None = None,
    debug: bool = False,  # Debug mode: use test_story.txt instead of API calls
    ctx: Context[ServerSession, AppContext] | None = None,
) -> dict[str, Any]:
    """Generate a new story with full 11-phase execution.

    This executes the complete story generation workflow:
    1. INIT - Initialize session
    2. CONFIG_LOAD - Load configuration
    3. BACKEND_INIT - Initialize LLM backend
    4. PROMPT_CONFIRM - Confirm prompt (auto-confirmed in MCP mode)
    5. CONTEXT_LOAD - Load context files
    6. PROMPT_BUILD - Build final prompt
    7. STORY_GENERATE - Generate story
    8. STORY_SAVE - Save story to file
    9. IMAGE_DECISION - Decide on image generation
    10. IMAGE_GENERATE - Generate images if enabled
    11. CONTEXT_SAVE - Save context for chaining

    Args:
        prompt: Story prompt/description
        age_range: Target age range (e.g., '8-10')
        style: Story style (e.g., 'adventure', 'mystery')
        tone: Story tone (e.g., 'light', 'serious')
        length: Story length ('short', 'medium', 'long')
        theme: Story theme (optional)
        characters: List of character names (optional)
        setting: Story setting (optional)
        learning_focus: Educational focus (optional)
        image_style: Image generation style (optional)
        backend: LLM backend to use (optional)
        output_directory: Output directory path (optional)
        ctx: MCP context for logging and progress (auto-injected)

    Returns:
        Dictionary containing:
        - session_id: Unique session identifier
        - status: Execution status
        - current_phase: Current execution phase
        - story: Generated story text
        - images: List of generated image paths
        - checkpoint_path: Path to checkpoint file
    """
    if ctx is None:
        raise MCPError(
            code=ErrorCode.INTERNAL_ERROR,
            message="Context not available",
            recoverable=False,
        )

    # Access shared resources from lifespan context
    app_ctx = ctx.request_context.lifespan_context
    checkpoint_manager = app_ctx.checkpoint_manager
    path_resolver = app_ctx.path_resolver
    queue_manager = app_ctx.queue_manager

    # Log start
    await ctx.info(f"Starting story generation: {prompt[:50]}...")

    # Generate session ID
    import uuid

    session_id = str(uuid.uuid4())

    # Load configuration
    try:
        config = load_config()
        await ctx.debug("Configuration loaded successfully")
    except Exception as e:
        await ctx.error(f"Failed to load configuration: {e}")
        raise MCPError(
            code=ErrorCode.CONFIG_ERROR,
            message=f"Failed to load configuration: {e}",
            recoverable=True,
            recovery_hint="Check configuration file exists and is valid",
        ) from e

    # Build CLI arguments
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
        cli_arguments["characters"] = ",".join(characters)  # Convert list to comma-separated string
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

    # Resolve configuration with auto_confirm for MCP mode
    default_backend = "gemini"
    if config.config.has_option("backend", "name"):
        default_backend = config.config.get("backend", "name")

    resolved_config = {
        "backend": backend or default_backend,
        "output_directory": output_directory or str(path_resolver.output_directory),
        "use_context": True,
        "verbose": False,
        "debug": debug,  # Use debug parameter from client
        "auto_confirm": True,  # Skip interactive prompts in MCP mode
        "silent_mode": True,  # Suppress console output in MCP mode (critical for stdio)
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
    checkpoint_data.session_id = session_id

    # Save initial checkpoint
    checkpoint_manager.save_checkpoint(checkpoint_data)
    await ctx.debug(f"Checkpoint created: {session_id}")

    # Enqueue the request
    try:

        async def execute_generation(**kwargs: Any) -> dict[str, Any]:  # Accept kwargs from queue manager
            """Execute story generation with progress reporting."""
            try:
                # Report progress through phases
                await ctx.report_progress(0.0, 1.0, "Phase 1/11: Initializing...")

                # Run synchronous PhaseExecutor in a thread pool executor
                loop = asyncio.get_running_loop()

                try:
                    await loop.run_in_executor(
                        None,  # Use default executor
                        lambda: PhaseExecutor(checkpoint_manager).execute_from_checkpoint(
                            checkpoint_data, ExecutionPhase.INIT
                        ),
                    )
                except Exception as executor_error:
                    await ctx.error(f"PhaseExecutor failed: {type(executor_error).__name__}: {executor_error}")
                    raise

                await ctx.report_progress(1.0, 1.0, "Generation complete!")

                # Reload final checkpoint
                final_checkpoint = checkpoint_manager.load_checkpoint(path_resolver.get_checkpoint_path(session_id))

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

                return {
                    "session_id": final_checkpoint.session_id,
                    "status": final_checkpoint.status,
                    "current_phase": final_checkpoint.current_phase,
                    "story": story,
                    "images": image_paths,
                    "checkpoint_path": str(path_resolver.get_checkpoint_path(final_checkpoint.session_id)),
                }
            except Exception as e:
                import traceback

                error_msg = f"{type(e).__name__}: {e}"
                traceback_str = traceback.format_exc()
                await ctx.error(f"Background task exception: {error_msg}\n{traceback_str}")

                # Update checkpoint with error
                try:
                    checkpoint_path = path_resolver.get_checkpoint_path(session_id)
                    error_checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)
                    if error_checkpoint:
                        error_checkpoint.last_error = error_msg
                        error_checkpoint.status = "failed"
                        checkpoint_manager.save_checkpoint(error_checkpoint)
                except Exception as checkpoint_error:
                    await ctx.error(f"Failed to save error to checkpoint: {checkpoint_error}")

                raise  # Re-raise so future completes with exception

        # Enqueue the request - DON'T await, return immediately
        future = await queue_manager.enqueue(
            session_id=session_id,
            handler=execute_generation,
        )

        # Store future for polling
        queue_manager._active_futures[session_id] = future

        await ctx.info(f"Story generation queued: {session_id}")

        # Return immediately with pending status
        return {
            "session_id": session_id,
            "status": "pending",
            "message": "Story generation has been queued. Use storyforge_get_session to poll for completion.",
            "checkpoint_path": str(path_resolver.get_checkpoint_path(session_id)),
        }

    except Exception as e:
        # Mark checkpoint as failed
        checkpoint_data.mark_failed(str(e))
        checkpoint_manager.save_checkpoint(checkpoint_data)
        await ctx.error(f"Story generation failed: {e}")

        raise MCPError(
            code=ErrorCode.GENERATION_FAILED,
            message=f"Story generation failed: {e}",
            recoverable=True,
            recovery_hint="Check backend configuration and API keys, then retry",
        ) from e


# ===== SESSION MANAGEMENT TOOLS =====


@mcp.tool()
async def storyforge_list_sessions(
    ctx: Context[ServerSession, AppContext] | None = None,
) -> dict[str, Any]:
    """List all available story generation sessions.

    Returns:
        Dictionary containing list of sessions with their metadata
    """
    if ctx is None:
        raise MCPError(code=ErrorCode.INTERNAL_ERROR, message="Context not available", recoverable=False)

    app_ctx = ctx.request_context.lifespan_context
    checkpoint_manager = app_ctx.checkpoint_manager
    path_resolver = app_ctx.path_resolver

    await ctx.debug("Listing sessions")

    try:
        # Get all checkpoint files
        checkpoint_dir = path_resolver.checkpoints_dir  # Use checkpoints_dir instead of checkpoint_directory
        sessions = []

        for checkpoint_path in checkpoint_dir.glob("checkpoint_*.yaml"):
            try:
                checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)
                if checkpoint:
                    # Extract prompt from original_inputs
                    original_prompt = checkpoint.original_inputs.get("prompt", "")
                    sessions.append(
                        {
                            "session_id": checkpoint.session_id,
                            "status": checkpoint.status,
                            "current_phase": checkpoint.current_phase,
                            "original_prompt": original_prompt,
                            "created_at": checkpoint.created_at,
                            "checkpoint_path": str(checkpoint_path),
                        }
                    )
            except Exception as e:
                await ctx.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
                continue

        await ctx.info(f"Found {len(sessions)} sessions")
        return {"sessions": sessions, "count": len(sessions)}

    except Exception as e:
        await ctx.error(f"Failed to list sessions: {e}")
        raise MCPError(
            code=ErrorCode.INTERNAL_ERROR,
            message=f"Failed to list sessions: {e}",
            recoverable=True,
        ) from e


@mcp.tool()
async def storyforge_get_session(
    session_id: str,
    ctx: Context[ServerSession, AppContext] | None = None,
) -> dict[str, Any]:
    """Get detailed information about a specific session.

    This tool checks both the checkpoint file and any active background tasks.
    If the session is still processing, it returns "pending" status.
    If complete, it returns the full results.

    Args:
        session_id: Session ID to retrieve
        ctx: MCP context (auto-injected)

    Returns:
        Complete session checkpoint data with current status
    """
    if ctx is None:
        raise MCPError(code=ErrorCode.INTERNAL_ERROR, message="Context not available", recoverable=False)

    app_ctx = ctx.request_context.lifespan_context
    checkpoint_manager = app_ctx.checkpoint_manager
    path_resolver = app_ctx.path_resolver
    queue_manager = app_ctx.queue_manager

    await ctx.debug(f"Getting session: {session_id}")

    try:
        # Check if there's an active future for this session
        active_futures = queue_manager._active_futures
        await ctx.debug(f"Active futures: {list(active_futures.keys())}")

        if session_id in active_futures:
            future = active_futures[session_id]

            # Check if future is done
            if future.done():
                # Get result (don't delete yet - cache it for multiple polls)
                try:
                    result: dict[str, Any] = await future
                    await ctx.info(f"Session {session_id} completed, returning cached result")
                    return result
                except Exception as e:
                    # On error, we can delete since we won't retry
                    del active_futures[session_id]
                    await ctx.error(f"Future failed with exception: {type(e).__name__}: {e}")

                    # Load checkpoint to get failure details
                    try:
                        checkpoint = checkpoint_manager.load_checkpoint(path_resolver.get_checkpoint_path(session_id))
                        if checkpoint:
                            error_message = checkpoint.last_error or str(e) or "Unknown error"
                            return {
                                "session_id": session_id,
                                "status": "failed",
                                "error_message": error_message,
                                "current_phase": checkpoint.current_phase,
                                "checkpoint_path": str(path_resolver.get_checkpoint_path(session_id)),
                            }
                    except FileNotFoundError:
                        pass
                    # If checkpoint doesn't exist, just return error
                    return {
                        "session_id": session_id,
                        "status": "failed",
                        "error_message": f"{type(e).__name__}: {e}",
                        "checkpoint_path": str(path_resolver.get_checkpoint_path(session_id)),
                    }
            else:
                # Still processing - try to load checkpoint for progress
                try:
                    checkpoint = checkpoint_manager.load_checkpoint(path_resolver.get_checkpoint_path(session_id))
                    if checkpoint:
                        return {
                            "session_id": session_id,
                            "status": "pending",
                            "current_phase": checkpoint.current_phase,
                            "completed_phases": checkpoint.completed_phases,
                            "message": "Story generation in progress",
                            "checkpoint_path": str(path_resolver.get_checkpoint_path(session_id)),
                        }
                except FileNotFoundError:
                    # Checkpoint not created yet - still initializing
                    pass

                # Return minimal pending status if checkpoint not available yet
                return {
                    "session_id": session_id,
                    "status": "pending",
                    "current_phase": "initializing",
                    "message": "Story generation starting...",
                    "checkpoint_path": str(path_resolver.get_checkpoint_path(session_id)),
                }

        # No active future - try to load from checkpoint
        try:
            checkpoint_path = path_resolver.get_checkpoint_path(session_id)
            checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)

            if checkpoint is None:
                raise MCPError(
                    code=ErrorCode.SESSION_NOT_FOUND,
                    message=f"Session not found: {session_id}",
                    recoverable=False,
                )

            return {
                "session_id": checkpoint.session_id,
                "status": checkpoint.status,
                "current_phase": checkpoint.current_phase,
                "original_prompt": checkpoint.original_inputs.get("prompt", ""),
                "cli_arguments": checkpoint.original_inputs.get("cli_arguments", {}),
                "resolved_config": checkpoint.resolved_config,
                "generated_content": checkpoint.generated_content,
                "execution_history": checkpoint.completed_phases,  # Use completed_phases instead
                "created_at": checkpoint.created_at,
                "updated_at": checkpoint.updated_at,
            }
        except FileNotFoundError:
            # Checkpoint doesn't exist yet
            raise MCPError(
                code=ErrorCode.SESSION_NOT_FOUND,
                message=f"Session not found: {session_id}",
                recoverable=False,
            ) from None

    except MCPError:
        raise
    except Exception as e:
        await ctx.error(f"Failed to get session: {e}")
        raise MCPError(
            code=ErrorCode.INTERNAL_ERROR,
            message=f"Failed to get session: {e}",
            recoverable=True,
        ) from e


@mcp.tool()
async def storyforge_delete_session(
    session_id: str,
    ctx: Context[ServerSession, AppContext] | None = None,
) -> dict[str, Any]:
    """Delete a session and its associated files.

    Args:
        session_id: Session ID to delete
        ctx: MCP context (auto-injected)

    Returns:
        Confirmation message
    """
    if ctx is None:
        raise MCPError(code=ErrorCode.INTERNAL_ERROR, message="Context not available", recoverable=False)

    app_ctx = ctx.request_context.lifespan_context
    path_resolver = app_ctx.path_resolver

    await ctx.info(f"Deleting session: {session_id}")

    try:
        checkpoint_path = path_resolver.get_checkpoint_path(session_id)

        if not checkpoint_path.exists():
            raise MCPError(
                code=ErrorCode.SESSION_NOT_FOUND,
                message=f"Session not found: {session_id}",
                recoverable=False,
            )

        # Delete checkpoint file
        checkpoint_path.unlink()

        # TODO: Also delete associated output files (story, images)

        await ctx.info(f"Session deleted: {session_id}")
        return {"success": True, "session_id": session_id, "message": "Session deleted successfully"}

    except MCPError:
        raise
    except Exception as e:
        await ctx.error(f"Failed to delete session: {e}")
        raise MCPError(
            code=ErrorCode.INTERNAL_ERROR,
            message=f"Failed to delete session: {e}",
            recoverable=True,
        ) from e


# ===== STORY EXTENSION TOOLS =====


@mcp.tool()
async def storyforge_extend_story(
    session_id: str,
    extension_prompt: str,
    length: str = "medium",
    ctx: Context[ServerSession, AppContext] | None = None,
) -> dict[str, Any]:
    """Extend an existing story with a new prompt.

    Args:
        session_id: Session ID of the story to extend
        extension_prompt: Prompt for the extension
        length: Length of the extension ('short', 'medium', 'long')
        ctx: MCP context (auto-injected)

    Returns:
        Extended story result
    """
    if ctx is None:
        raise MCPError(code=ErrorCode.INTERNAL_ERROR, message="Context not available", recoverable=False)

    app_ctx = ctx.request_context.lifespan_context
    checkpoint_manager = app_ctx.checkpoint_manager
    path_resolver = app_ctx.path_resolver

    await ctx.info(f"Extending story: {session_id}")

    try:
        # Load original checkpoint
        checkpoint_path = path_resolver.get_checkpoint_path(session_id)
        checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)

        if checkpoint is None:
            raise MCPError(
                code=ErrorCode.SESSION_NOT_FOUND,
                message=f"Session not found: {session_id}",
                recoverable=False,
            )

        # Create extension checkpoint by creating a new checkpoint based on the original
        import uuid

        extension_session_id = str(uuid.uuid4())
        extension_cli_args = checkpoint.original_inputs.get("cli_arguments", {}).copy()
        extension_cli_args["prompt"] = extension_prompt
        extension_cli_args["length"] = length

        extension_checkpoint = CheckpointData.create_new(
            original_prompt=extension_prompt,
            cli_arguments=extension_cli_args,
            resolved_config=checkpoint.resolved_config.copy(),
        )
        extension_checkpoint.session_id = extension_session_id

        # Mark as extension
        extension_checkpoint.original_inputs["parent_session_id"] = session_id
        extension_checkpoint.original_inputs["is_extension"] = True

        checkpoint_manager.save_checkpoint(extension_checkpoint)

        await ctx.debug(f"Extension checkpoint created: {extension_checkpoint.session_id}")

        # Execute extension
        executor = PhaseExecutor(checkpoint_manager)
        executor.execute_from_checkpoint(extension_checkpoint, ExecutionPhase.INIT)

        # Reload final checkpoint
        final_checkpoint = checkpoint_manager.load_checkpoint(
            path_resolver.get_checkpoint_path(extension_checkpoint.session_id)
        )

        if final_checkpoint is None:
            raise MCPError(
                code=ErrorCode.CHECKPOINT_CORRUPT,
                message="Failed to load final checkpoint",
                recoverable=False,
            )

        await ctx.info(f"Story extension complete: {extension_checkpoint.session_id}")

        return {
            "session_id": final_checkpoint.session_id,
            "parent_session_id": session_id,
            "status": final_checkpoint.status,
            "story": final_checkpoint.generated_content.get("story"),
        }

    except MCPError:
        raise
    except Exception as e:
        await ctx.error(f"Failed to extend story: {e}")
        raise MCPError(
            code=ErrorCode.GENERATION_FAILED,
            message=f"Failed to extend story: {e}",
            recoverable=True,
        ) from e


# ===== STORY REFINEMENT TOOLS =====


@mcp.tool()
async def storyforge_refine_story(
    session_id: str,
    refinement_instructions: str,
    backend: str | None = None,
    ctx: Context[ServerSession, AppContext] | None = None,
) -> dict[str, Any]:
    """Refine an existing story with specific instructions.

    Args:
        session_id: Session ID containing the story to refine
        refinement_instructions: Instructions for refinement
        backend: Optional backend to use for refinement
        ctx: MCP context (auto-injected)

    Returns:
        Refined story result
    """
    if ctx is None:
        raise MCPError(code=ErrorCode.INTERNAL_ERROR, message="Context not available", recoverable=False)

    app_ctx = ctx.request_context.lifespan_context
    checkpoint_manager = app_ctx.checkpoint_manager
    path_resolver = app_ctx.path_resolver

    await ctx.info(f"Refining story: {session_id}")

    try:
        # Load checkpoint
        checkpoint_path = path_resolver.get_checkpoint_path(session_id)
        checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)

        if checkpoint is None:
            raise MCPError(
                code=ErrorCode.SESSION_NOT_FOUND,
                message=f"Session not found: {session_id}",
                recoverable=False,
            )

        # Get original story
        original_story = checkpoint.generated_content.get("story")
        if not original_story:
            raise MCPError(
                code=ErrorCode.CHECKPOINT_CORRUPT,  # Use CHECKPOINT_CORRUPT instead of INVALID_REQUEST
                message="No story found in session",
                recoverable=False,
            )

        # Create refinement using the selected backend
        from ..llm_backend import get_backend

        backend_name = backend or checkpoint.resolved_config.get("backend", "gemini")
        llm_backend = get_backend(backend_name)

        # Build refinement prompt
        refinement_prompt = f"""Please refine this story according to these instructions:

Instructions: {refinement_instructions}

Original Story:
{original_story}

Please provide the refined version:"""

        await ctx.debug("Generating refinement...")

        # Generate refined story using generate_text or similar method
        # Note: LLMBackend doesn't have generate_text, so we need to use a Prompt object
        from ..prompt import Prompt

        # Build a temporary prompt for refinement
        temp_prompt = Prompt(
            prompt=refinement_prompt,
            age_range=checkpoint.resolved_config.get("age_range", "8-10"),
            style=checkpoint.resolved_config.get("style", "adventure"),
            tone=checkpoint.resolved_config.get("tone", "light"),
            length=checkpoint.resolved_config.get("length", "medium"),
        )

        refined_story = llm_backend.generate_story(temp_prompt)

        # Update checkpoint with refined story
        checkpoint.generated_content["story"] = refined_story
        checkpoint.generated_content["refinement_history"] = checkpoint.generated_content.get("refinement_history", [])
        checkpoint.generated_content["refinement_history"].append(
            {"instructions": refinement_instructions, "backend": backend_name}
        )

        checkpoint_manager.save_checkpoint(checkpoint)

        await ctx.info("Story refinement complete")

        return {
            "session_id": session_id,
            "status": "refined",
            "story": refined_story,
            "refinement_instructions": refinement_instructions,
        }

    except MCPError:
        raise
    except Exception as e:
        await ctx.error(f"Failed to refine story: {e}")
        raise MCPError(
            code=ErrorCode.GENERATION_FAILED,
            message=f"Failed to refine story: {e}",
            recoverable=True,
        ) from e


# ===== CONTENT RETRIEVAL TOOLS =====


@mcp.tool()
async def storyforge_get_story(
    session_id: str,
    ctx: Context[ServerSession, AppContext] | None = None,
) -> str:
    """Get the generated story text from a session.

    Args:
        session_id: Session ID to retrieve story from
        ctx: MCP context (auto-injected)

    Returns:
        Story text
    """
    if ctx is None:
        raise MCPError(code=ErrorCode.INTERNAL_ERROR, message="Context not available", recoverable=False)

    app_ctx = ctx.request_context.lifespan_context
    checkpoint_manager = app_ctx.checkpoint_manager
    path_resolver = app_ctx.path_resolver

    try:
        checkpoint_path = path_resolver.get_checkpoint_path(session_id)
        checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)

        if checkpoint is None:
            raise MCPError(
                code=ErrorCode.SESSION_NOT_FOUND,
                message=f"Session not found: {session_id}",
                recoverable=False,
            )

        story: str | None = checkpoint.generated_content.get("story")
        if not story:
            raise MCPError(
                code=ErrorCode.CHECKPOINT_CORRUPT,  # Use CHECKPOINT_CORRUPT instead
                message="No story found in session",
                recoverable=False,
            )

        return story

    except MCPError:
        raise
    except Exception as e:
        raise MCPError(
            code=ErrorCode.INTERNAL_ERROR,
            message=f"Failed to get story: {e}",
            recoverable=True,
        ) from e


@mcp.tool()
async def storyforge_get_images(
    session_id: str,
    ctx: Context[ServerSession, AppContext] | None = None,
) -> list[str]:
    """Get the generated image paths from a session.

    Args:
        session_id: Session ID to retrieve images from
        ctx: MCP context (auto-injected)

    Returns:
        List of image file paths
    """
    if ctx is None:
        raise MCPError(code=ErrorCode.INTERNAL_ERROR, message="Context not available", recoverable=False)

    app_ctx = ctx.request_context.lifespan_context
    checkpoint_manager = app_ctx.checkpoint_manager
    path_resolver = app_ctx.path_resolver

    try:
        checkpoint_path = path_resolver.get_checkpoint_path(session_id)
        checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)

        if checkpoint is None:
            raise MCPError(
                code=ErrorCode.SESSION_NOT_FOUND,
                message=f"Session not found: {session_id}",
                recoverable=False,
            )

        images = checkpoint.generated_content.get("images", [])
        image_paths = []

        if isinstance(images, list):
            for img in images:
                if isinstance(img, dict):
                    image_paths.append(img.get("path", ""))
                else:
                    image_paths.append(str(img))

        return image_paths

    except MCPError:
        raise
    except Exception as e:
        raise MCPError(
            code=ErrorCode.INTERNAL_ERROR,
            message=f"Failed to get images: {e}",
            recoverable=True,
        ) from e


# ===== IMAGE GENERATION TOOLS =====


@mcp.tool()
async def storyforge_generate_images(
    session_id: str,
    count: int = 3,
    image_style: str | None = None,
    ctx: Context[ServerSession, AppContext] | None = None,
) -> dict[str, Any]:
    """Generate images for an existing story.

    Args:
        session_id: Session ID containing the story
        count: Number of images to generate (default: 3)
        image_style: Optional image style override
        ctx: MCP context (auto-injected)

    Returns:
        Dictionary with generated image paths
    """
    if ctx is None:
        raise MCPError(code=ErrorCode.INTERNAL_ERROR, message="Context not available", recoverable=False)

    app_ctx = ctx.request_context.lifespan_context
    checkpoint_manager = app_ctx.checkpoint_manager
    path_resolver = app_ctx.path_resolver

    await ctx.info(f"Generating {count} images for session: {session_id}")

    try:
        # Load checkpoint
        checkpoint_path = path_resolver.get_checkpoint_path(session_id)
        checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)

        if checkpoint is None:
            raise MCPError(
                code=ErrorCode.SESSION_NOT_FOUND,
                message=f"Session not found: {session_id}",
                recoverable=False,
            )

        # Update config for image generation
        checkpoint.resolved_config["image_count"] = count
        if image_style:
            checkpoint.resolved_config["image_style"] = image_style

        # Execute image generation phase using PhaseExecutor
        # Note: PhaseExecutor doesn't have execute_phase, so we need to execute from IMAGE_GENERATE phase
        executor = PhaseExecutor(checkpoint_manager)
        executor.execute_from_checkpoint(checkpoint, ExecutionPhase.IMAGE_GENERATE)

        # Reload checkpoint
        final_checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)

        if final_checkpoint is None:
            raise MCPError(
                code=ErrorCode.CHECKPOINT_CORRUPT,
                message="Failed to load checkpoint after image generation",
                recoverable=False,
            )

        images = final_checkpoint.generated_content.get("images", [])
        image_paths = []

        if isinstance(images, list):
            for img in images:
                if isinstance(img, dict):
                    image_paths.append(img.get("path", ""))
                else:
                    image_paths.append(str(img))

        await ctx.info(f"Generated {len(image_paths)} images")

        return {"session_id": session_id, "images": image_paths, "count": len(image_paths)}

    except MCPError:
        raise
    except Exception as e:
        await ctx.error(f"Failed to generate images: {e}")
        raise MCPError(
            code=ErrorCode.GENERATION_FAILED,
            message=f"Failed to generate images: {e}",
            recoverable=True,
        ) from e


# ===== CONFIGURATION TOOLS =====


@mcp.tool()
async def storyforge_get_config(
    ctx: Context[ServerSession, AppContext] | None = None,
) -> dict[str, Any]:
    """Get current StoryForge configuration.

    Args:
        ctx: MCP context (auto-injected)

    Returns:
        Configuration dictionary
    """
    if ctx is None:
        raise MCPError(code=ErrorCode.INTERNAL_ERROR, message="Context not available", recoverable=False)

    try:
        config = load_config()
        config_dict: dict[str, dict[str, str]] = {}

        # Extract all sections and options
        for section in config.config.sections():
            config_dict[section] = {}
            for option in config.config.options(section):
                config_dict[section][option] = config.config.get(section, option)

        return config_dict

    except Exception as e:
        raise MCPError(
            code=ErrorCode.CONFIG_ERROR,
            message=f"Failed to load configuration: {e}",
            recoverable=True,
        ) from e


@mcp.tool()
async def storyforge_list_backends(
    ctx: Context[ServerSession, AppContext] | None = None,
) -> dict[str, Any]:
    """List available LLM backends.

    Args:
        ctx: MCP context (auto-injected)

    Returns:
        Dictionary with available backends and their status
    """
    if ctx is None:
        raise MCPError(code=ErrorCode.INTERNAL_ERROR, message="Context not available", recoverable=False)

    try:
        from ..llm_backend import list_available_backends

        backend_info = list_available_backends()
        backends = []

        for name, info in backend_info.items():
            backends.append(
                {
                    "name": name,
                    "available": info["available"],
                    "reason": info["reason"],
                }
            )

        return {"backends": backends, "count": len(backends)}

    except Exception as e:
        raise MCPError(
            code=ErrorCode.INTERNAL_ERROR,
            message=f"Failed to list backends: {e}",
            recoverable=True,
        ) from e


# ===== RESUME/RETRY TOOLS =====


@mcp.tool()
async def storyforge_resume_session(
    session_id: str,
    ctx: Context[ServerSession, AppContext] | None = None,
) -> dict[str, Any]:
    """Resume a failed or interrupted session.

    Args:
        session_id: Session ID to resume
        ctx: MCP context (auto-injected)

    Returns:
        Resumed session result
    """
    if ctx is None:
        raise MCPError(code=ErrorCode.INTERNAL_ERROR, message="Context not available", recoverable=False)

    app_ctx = ctx.request_context.lifespan_context
    checkpoint_manager = app_ctx.checkpoint_manager
    path_resolver = app_ctx.path_resolver

    await ctx.info(f"Resuming session: {session_id}")

    try:
        checkpoint_path = path_resolver.get_checkpoint_path(session_id)
        checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)

        if checkpoint is None:
            raise MCPError(
                code=ErrorCode.SESSION_NOT_FOUND,
                message=f"Session not found: {session_id}",
                recoverable=False,
            )

        # Resume from current phase
        executor = PhaseExecutor(checkpoint_manager)
        current_phase = ExecutionPhase[checkpoint.current_phase]
        executor.execute_from_checkpoint(checkpoint, current_phase)

        # Reload final checkpoint
        final_checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)

        if final_checkpoint is None:
            raise MCPError(
                code=ErrorCode.CHECKPOINT_CORRUPT,
                message="Failed to load checkpoint after resume",
                recoverable=False,
            )

        await ctx.info(f"Session resumed successfully: {session_id}")

        return {
            "session_id": final_checkpoint.session_id,
            "status": final_checkpoint.status,
            "current_phase": final_checkpoint.current_phase,
            "story": final_checkpoint.generated_content.get("story"),
        }

    except MCPError:
        raise
    except Exception as e:
        await ctx.error(f"Failed to resume session: {e}")
        raise MCPError(
            code=ErrorCode.GENERATION_FAILED,
            message=f"Failed to resume session: {e}",
            recoverable=True,
        ) from e


@mcp.tool()
async def storyforge_retry_phase(
    session_id: str,
    phase: str,
    ctx: Context[ServerSession, AppContext] | None = None,
) -> dict[str, Any]:
    """Retry a specific phase of story generation.

    Args:
        session_id: Session ID to retry
        phase: Phase name to retry (e.g., 'STORY_GENERATE')
        ctx: MCP context (auto-injected)

    Returns:
        Result after retrying the phase
    """
    if ctx is None:
        raise MCPError(code=ErrorCode.INTERNAL_ERROR, message="Context not available", recoverable=False)

    app_ctx = ctx.request_context.lifespan_context
    checkpoint_manager = app_ctx.checkpoint_manager
    path_resolver = app_ctx.path_resolver

    await ctx.info(f"Retrying phase {phase} for session: {session_id}")

    try:
        checkpoint_path = path_resolver.get_checkpoint_path(session_id)
        checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)

        if checkpoint is None:
            raise MCPError(
                code=ErrorCode.SESSION_NOT_FOUND,
                message=f"Session not found: {session_id}",
                recoverable=False,
            )

        # Validate phase
        try:
            phase_enum = ExecutionPhase[phase]
        except KeyError:
            raise MCPError(
                code=ErrorCode.CHECKPOINT_CORRUPT,  # Use existing error code
                message=f"Invalid phase: {phase}",
                recoverable=False,
            ) from None

        # Execute from specified phase
        executor = PhaseExecutor(checkpoint_manager)
        executor.execute_from_checkpoint(checkpoint, phase_enum)

        # Reload checkpoint
        final_checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)

        if final_checkpoint is None:
            raise MCPError(
                code=ErrorCode.CHECKPOINT_CORRUPT,
                message="Failed to load checkpoint after retry",
                recoverable=False,
            )

        await ctx.info(f"Phase {phase} completed successfully")

        return {
            "session_id": final_checkpoint.session_id,
            "status": final_checkpoint.status,
            "current_phase": final_checkpoint.current_phase,
            "retried_phase": phase,
        }

    except MCPError:
        raise
    except Exception as e:
        await ctx.error(f"Failed to retry phase: {e}")
        raise MCPError(
            code=ErrorCode.GENERATION_FAILED,
            message=f"Failed to retry phase: {e}",
            recoverable=True,
        ) from e


# ===== MAIN ENTRY POINT =====


def main() -> None:
    """Main entry point for FastMCP server."""
    # Run with stdio transport (default)
    mcp.run()


if __name__ == "__main__":
    main()
