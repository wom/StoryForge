"""Story refinement tools for StoryForge MCP server."""

from typing import Any

from mcp.server import Server
from mcp.types import Tool

from ...checkpoint import CheckpointManager
from ...llm_backend import LLMBackend, get_backend
from ...prompt import Prompt
from ...shared.types import ErrorCode, MCPError
from ..path_resolver import PathResolver


def register_refinement_tools(server: Server) -> None:
    """Register story refinement tools with the MCP server."""
    path_resolver = PathResolver()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available refinement tools."""
        return [
            Tool(
                name="storyforge_refine_story",
                description="Refine an existing story with specific instructions",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID containing the story to refine",
                        },
                        "refinement_instructions": {
                            "type": "string",
                            "description": (
                                "Instructions for how to refine the story "
                                "(e.g., 'Make it more suspenseful', 'Add more dialogue', 'Simplify the vocabulary')"
                            ),
                        },
                        "backend": {
                            "type": "string",
                            "description": (
                                "Optional backend to use for refinement (gemini, openai, anthropic). "
                                "Uses session's backend if not specified."
                            ),
                        },
                    },
                    "required": ["session_id", "refinement_instructions"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
        """Handle refinement tool calls."""
        try:
            if name == "storyforge_refine_story":
                return await handle_refine_story(arguments, path_resolver)
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


async def handle_refine_story(arguments: dict[str, Any], path_resolver: PathResolver) -> list[Any]:
    """
    Refine an existing story with specific instructions.

    This tool takes an existing story and refines it based on user instructions,
    useful for:
    - Adjusting tone or style
    - Adding more detail or dialogue
    - Simplifying vocabulary for younger readers
    - Making the story more suspenseful, funny, etc.
    - Fixing issues or improving specific aspects

    Returns:
        List with refined story and session_id
    """
    session_id = arguments["session_id"]
    refinement_instructions = arguments["refinement_instructions"]
    backend_name = arguments.get("backend")

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

        # Determine backend to use
        if backend_name:
            backend_name = backend_name.lower()
            backend: LLMBackend | None = get_backend(backend_name)
        else:
            # Use session's backend
            session_backend = resolved_config.get("backend")
            if session_backend:
                backend = get_backend(session_backend)
            else:
                # Auto-detect available backend
                backend = get_backend()

        if backend is None:
            raise MCPError(
                code=ErrorCode.BACKEND_UNAVAILABLE,
                message="No LLM backend available for refinement",
                recoverable=True,
                recovery_hint="Ensure at least one API key is set (GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY)",
            )

        # Build refinement prompt
        refinement_prompt = _build_refinement_prompt(
            original_story=story_content,
            refinement_instructions=refinement_instructions,
            resolved_config=resolved_config,
        )

        # Create a Prompt object for the refinement
        # We'll use the original prompt parameters but override with refinement instructions
        prompt = Prompt(
            prompt=refinement_prompt,
            characters=resolved_config.get("characters", []),
            theme=resolved_config.get("theme"),
            setting=resolved_config.get("setting"),
            age_range=resolved_config.get("age_range") or "",
            tone=resolved_config.get("tone") or "",
            learning_focus=resolved_config.get("learning_focus"),
        )

        # Generate refined story using backend
        refined_story = backend.generate_story(prompt)

        # Check if refinement was successful
        if not refined_story or refined_story.startswith("[Error"):
            raise MCPError(
                code=ErrorCode.GENERATION_FAILED,
                message=f"Failed to refine story: {refined_story}",
                recoverable=True,
            )

        # Update checkpoint with refined story
        # Store both original and refined versions
        if "refinements" not in checkpoint.generated_content:
            checkpoint.generated_content["refinements"] = []

        checkpoint.generated_content["refinements"].append(
            {
                "instructions": refinement_instructions,
                "backend": backend.name,
                "refined_story": refined_story,
            }
        )

        # Update the current story to the refined version
        checkpoint.generated_content["story"] = refined_story

        # Save updated checkpoint
        checkpoint_manager.save_checkpoint(checkpoint)

        return [
            {
                "refined_story": refined_story,
                "session_id": session_id,
                "backend_used": backend.name,
            }
        ]

    except MCPError:
        raise
    except Exception as e:
        raise MCPError(
            code=ErrorCode.GENERATION_FAILED,
            message=f"Failed to refine story: {e}",
            recoverable=False,
        ) from e


def _build_refinement_prompt(
    original_story: str,
    refinement_instructions: str,
    resolved_config: dict[str, Any],
) -> str:
    """
    Build a refinement prompt for the LLM.

    Args:
        original_story: The original story to refine
        refinement_instructions: User's instructions for refinement
        resolved_config: Configuration dict with story parameters

    Returns:
        Formatted refinement prompt
    """
    lines = []

    # Start with the refinement task
    lines.append("Please refine the following story based on these instructions:")
    lines.append(f"\n**Refinement Instructions:** {refinement_instructions}\n")

    # Add context about the story parameters
    age_range = resolved_config.get("age_range")
    if age_range:
        lines.append(f"**Target Age Group:** {age_range}")

    tone = resolved_config.get("tone")
    if tone:
        lines.append(f"**Tone:** {tone}")

    theme = resolved_config.get("theme")
    if theme:
        lines.append(f"**Theme:** {theme}")

    learning_focus = resolved_config.get("learning_focus")
    if learning_focus:
        lines.append(f"**Learning Focus:** {learning_focus}")

    # Add the original story
    lines.append("\n**Original Story:**\n")
    lines.append(original_story)

    lines.append("\n**Your Refined Story:**")
    lines.append("(Please provide the complete refined story, maintaining the story structure and parameters above)")

    return "\n".join(lines)
