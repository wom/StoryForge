"""Session management tools for StoryForge MCP server."""

from typing import Any

import yaml
from mcp.server import Server
from mcp.types import Tool

from ...checkpoint import CheckpointManager, ExecutionPhase
from ...checkpoint import SessionStatus as CheckpointStatus
from ...shared.types import ErrorCode, MCPError, SessionStatus
from ..path_resolver import PathResolver


def register_session_tools(server: Server) -> None:
    """Register session management tools with the MCP server."""
    path_resolver = PathResolver()
    session_manager = SessionManager(path_resolver)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available session management tools."""
        return [
            Tool(
                name="storyforge_list_sessions",
                description="List all available story generation sessions",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "status_filter": {
                            "type": "string",
                            "enum": ["active", "completed", "failed", "all"],
                            "description": "Filter sessions by status",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of sessions to return",
                        },
                    },
                },
            ),
            Tool(
                name="storyforge_get_session_status",
                description="Get detailed status of a specific session",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID to query",
                        },
                    },
                    "required": ["session_id"],
                },
            ),
            Tool(
                name="storyforge_continue_session",
                description="Resume a paused session from a checkpoint",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID to resume (if None, shows list)",
                        },
                        "resume_phase": {
                            "type": "string",
                            "description": "Phase to resume from (optional)",
                        },
                    },
                },
            ),
            Tool(
                name="storyforge_delete_session",
                description="Delete a session and optionally its outputs",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID to delete",
                        },
                        "keep_outputs": {
                            "type": "boolean",
                            "description": "Keep story and image files",
                        },
                    },
                    "required": ["session_id"],
                },
            ),
            Tool(
                name="storyforge_get_queue_status",
                description="Get current queue status",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
        """Handle tool calls."""
        try:
            if name == "storyforge_list_sessions":
                result = session_manager.list_sessions(
                    status_filter=arguments.get("status_filter", "all"),
                    limit=arguments.get("limit", 100),
                )
                return [result]
            elif name == "storyforge_get_session_status":
                result = session_manager.get_session_status(arguments["session_id"])
                return [result]
            elif name == "storyforge_continue_session":
                result = session_manager.continue_session(
                    arguments.get("session_id"),
                    arguments.get("resume_phase"),
                )
                return [result]
            elif name == "storyforge_delete_session":
                result = session_manager.delete_session(
                    arguments["session_id"],
                    arguments.get("keep_outputs", True),
                )
                return [result]
            elif name == "storyforge_get_queue_status":
                # TODO: Implement proper queue management
                return [{"active_session": None, "queue": [], "queue_length": 0}]
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


class SessionManager:
    """Manage StoryForge sessions for MCP server."""

    def __init__(self, path_resolver: PathResolver):
        """Initialize session manager."""
        self.path_resolver = path_resolver
        self.checkpoint_manager = CheckpointManager()

    def list_sessions(self, status_filter: str = "all", limit: int = 100) -> dict[str, Any]:
        """
        List available sessions with optional filtering.

        Args:
            status_filter: Filter by status (active|completed|failed|all)
            limit: Maximum number of sessions to return

        Returns:
            Dictionary with sessions list
        """
        checkpoints = self.path_resolver.list_checkpoints()
        sessions = []

        for checkpoint_path in checkpoints:
            try:
                with open(checkpoint_path) as f:
                    data = yaml.safe_load(f)

                # Map CheckpointStatus to SessionStatus
                checkpoint_status = data.get("status", "active")
                if checkpoint_status == "completed":
                    session_status = SessionStatus.COMPLETED.value
                elif checkpoint_status == "failed":
                    session_status = SessionStatus.FAILED.value
                elif checkpoint_status == "active":
                    session_status = SessionStatus.ACTIVE.value
                else:
                    session_status = SessionStatus.ACTIVE.value

                # Apply filter
                if status_filter != "all" and session_status != status_filter:
                    continue

                sessions.append(
                    {
                        "session_id": checkpoint_path.stem,
                        "created_at": data.get("timestamp", ""),
                        "status": session_status,
                        "prompt": data.get("prompt_text", ""),
                        "current_phase": data.get("current_phase", "init"),
                    }
                )

                if len(sessions) >= limit:
                    break

            except Exception:
                # Skip corrupted checkpoints
                continue

        return {"sessions": sessions}

    def get_session_status(self, session_id: str) -> dict[str, Any]:
        """
        Get detailed status for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session status details

        Raises:
            MCPError: If session not found
        """
        checkpoint_path = self.path_resolver.get_checkpoint_path(session_id)

        if not checkpoint_path.exists():
            raise MCPError(
                code=ErrorCode.SESSION_NOT_FOUND,
                message=f"Session {session_id} not found",
                recoverable=False,
                recovery_hint="Use list_sessions to find available sessions",
            )

        try:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)

            # Calculate progress percent (11 phases excluding COMPLETED)
            phase_order = [
                ExecutionPhase.INIT,
                ExecutionPhase.CONFIG_LOAD,
                ExecutionPhase.BACKEND_INIT,
                ExecutionPhase.PROMPT_CONFIRM,
                ExecutionPhase.CONTEXT_LOAD,
                ExecutionPhase.PROMPT_BUILD,
                ExecutionPhase.STORY_GENERATE,
                ExecutionPhase.STORY_SAVE,
                ExecutionPhase.IMAGE_DECISION,
                ExecutionPhase.IMAGE_GENERATE,
                ExecutionPhase.CONTEXT_SAVE,
            ]

            try:
                current_phase_enum = ExecutionPhase(checkpoint_data.current_phase)
                if current_phase_enum == ExecutionPhase.COMPLETED:
                    progress = 100
                elif current_phase_enum in phase_order:
                    progress = int((phase_order.index(current_phase_enum) / len(phase_order)) * 100)
                else:
                    progress = 0
            except ValueError:
                progress = 0

            # Map checkpoint status to session status
            checkpoint_status = checkpoint_data.status
            if checkpoint_status == CheckpointStatus.COMPLETED.value:
                session_status = SessionStatus.COMPLETED.value
            elif checkpoint_status == CheckpointStatus.FAILED.value:
                session_status = SessionStatus.FAILED.value
            else:
                session_status = SessionStatus.ACTIVE.value

            # Extract story and images from generated_content
            story = checkpoint_data.generated_content.get("story")
            images = checkpoint_data.generated_content.get("images", [])
            image_filenames = []
            if isinstance(images, list):
                for img in images:
                    if isinstance(img, dict):
                        image_filenames.append(img.get("filename", ""))
                    else:
                        image_filenames.append(str(img))

            return {
                "session_id": session_id,
                "status": session_status,
                "current_phase": checkpoint_data.current_phase,
                "completed_phases": checkpoint_data.completed_phases,
                "story": story,
                "images": image_filenames,
                "errors": [],
                "progress_percent": progress,
            }

        except Exception as e:
            raise MCPError(
                code=ErrorCode.CHECKPOINT_CORRUPT,
                message=f"Failed to load checkpoint: {e}",
                recoverable=False,
                recovery_hint="The checkpoint file may be corrupted",
            ) from e

    def delete_session(self, session_id: str, keep_outputs: bool = True) -> dict[str, Any]:
        """
        Delete a session checkpoint.

        Args:
            session_id: Session identifier
            keep_outputs: If True, keep story/image files, only delete checkpoint

        Returns:
            Dictionary with deletion result

        Raises:
            MCPError: If session not found or deletion fails
        """
        checkpoint_path = self.path_resolver.get_checkpoint_path(session_id)

        if not checkpoint_path.exists():
            raise MCPError(
                code=ErrorCode.SESSION_NOT_FOUND,
                message=f"Session {session_id} not found",
                recoverable=False,
            )

        try:
            # Delete checkpoint
            checkpoint_path.unlink()
            cleanup_path = str(checkpoint_path)

            # Optionally delete outputs
            if not keep_outputs:
                # Try to load checkpoint to get output paths
                # (This is a best-effort cleanup)
                pass

            return {"deleted": True, "cleanup_path": cleanup_path}

        except Exception as e:
            raise MCPError(
                code=ErrorCode.FILE_IO_ERROR,
                message=f"Failed to delete session: {e}",
                recoverable=True,
                recovery_hint="Check file permissions",
            ) from e

    def continue_session(self, session_id: str | None, resume_phase: str | None = None) -> dict[str, Any]:
        """
        Continue a paused session.

        Args:
            session_id: Session to continue (if None, returns list)
            resume_phase: Phase to resume from (optional)

        Returns:
            Dictionary with session continuation info

        Raises:
            MCPError: If session not found or invalid phase
        """
        if session_id is None:
            # Return list of resumable sessions
            return self.list_sessions(status_filter="active", limit=50)

        checkpoint_path = self.path_resolver.get_checkpoint_path(session_id)

        if not checkpoint_path.exists():
            raise MCPError(
                code=ErrorCode.SESSION_NOT_FOUND,
                message=f"Session {session_id} not found",
                recoverable=False,
                recovery_hint="Use list_sessions to find available sessions",
            )

        try:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)

            # Validate resume phase if provided
            if resume_phase:
                try:
                    ExecutionPhase(resume_phase)
                except ValueError as e:
                    raise MCPError(
                        code=ErrorCode.INVALID_PHASE,
                        message=f"Invalid phase: {resume_phase}",
                        recoverable=False,
                        recovery_hint="Use a valid ExecutionPhase value",
                    ) from e

            available_phases = [phase.value for phase in ExecutionPhase if phase != ExecutionPhase.COMPLETED]

            return {
                "session_id": session_id,
                "status": SessionStatus.RESUMING.value,
                "available_phases": available_phases,
                "current_phase": checkpoint_data.current_phase,
            }

        except MCPError:
            raise
        except Exception as e:
            raise MCPError(
                code=ErrorCode.CHECKPOINT_CORRUPT,
                message=f"Failed to load checkpoint: {e}",
                recoverable=False,
            ) from e
