"""StoryForge error handling."""

from typing import Any

from .types import ErrorCode


class StoryForgeError(Exception):
    """Base exception for StoryForge errors."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
        recoverable: bool = True,
        recovery_hint: str | None = None,
    ) -> None:
        """Initialize error."""
        super().__init__(message)
        self.code = code.value
        self.message = message
        self.details = details or {}
        self.recoverable = recoverable
        self.recovery_hint = recovery_hint


class BackendUnavailableError(StoryForgeError):
    """LLM backend not available."""

    def __init__(self, backend: str, details: dict[str, Any] | None = None) -> None:
        """Initialize error."""
        super().__init__(
            code=ErrorCode.BACKEND_UNAVAILABLE,
            message=f"Backend '{backend}' is not available or not configured",
            details=details,
            recoverable=True,
            recovery_hint="Check API keys and backend configuration",
        )


class CheckpointCorruptError(StoryForgeError):
    """Checkpoint file is corrupt."""

    def __init__(self, checkpoint_path: str, details: dict[str, Any] | None = None) -> None:
        """Initialize error."""
        super().__init__(
            code=ErrorCode.CHECKPOINT_CORRUPT,
            message=f"Checkpoint file is corrupt: {checkpoint_path}",
            details=details,
            recoverable=False,
            recovery_hint="Delete checkpoint and start a new session",
        )


class SessionNotFoundError(StoryForgeError):
    """Session not found."""

    def __init__(self, session_id: str) -> None:
        """Initialize error."""
        super().__init__(
            code=ErrorCode.SESSION_NOT_FOUND,
            message=f"Session '{session_id}' not found",
            details={"session_id": session_id},
            recoverable=False,
            recovery_hint="Use list_sessions to find available sessions",
        )


class ContextFileNotFoundError(StoryForgeError):
    """Context file not found."""

    def __init__(self, filename: str) -> None:
        """Initialize error."""
        super().__init__(
            code=ErrorCode.CONTEXT_FILE_NOT_FOUND,
            message=f"Context file '{filename}' not found",
            details={"filename": filename},
            recoverable=False,
            recovery_hint="Use list_context_files to find available files",
        )


class InvalidPhaseError(StoryForgeError):
    """Invalid execution phase."""

    def __init__(self, phase: str, valid_phases: list[str]) -> None:
        """Initialize error."""
        super().__init__(
            code=ErrorCode.INVALID_PHASE,
            message=f"Invalid phase '{phase}'",
            details={"phase": phase, "valid_phases": valid_phases},
            recoverable=False,
            recovery_hint=f"Valid phases: {', '.join(valid_phases)}",
        )


class GenerationFailedError(StoryForgeError):
    """LLM generation failed."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize error."""
        super().__init__(
            code=ErrorCode.GENERATION_FAILED,
            message=f"Generation failed: {message}",
            details=details,
            recoverable=True,
            recovery_hint="Retry the operation or check backend status",
        )


class FileIOError(StoryForgeError):
    """File I/O operation failed."""

    def __init__(self, operation: str, path: str, details: dict[str, Any] | None = None) -> None:
        """Initialize error."""
        super().__init__(
            code=ErrorCode.FILE_IO_ERROR,
            message=f"File {operation} failed: {path}",
            details=details,
            recoverable=True,
            recovery_hint="Check file permissions and disk space",
        )


class ConfigError(StoryForgeError):
    """Configuration error."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize error."""
        super().__init__(
            code=ErrorCode.CONFIG_ERROR,
            message=f"Configuration error: {message}",
            details=details,
            recoverable=True,
            recovery_hint="Check configuration file syntax and values",
        )


class QueueFullError(StoryForgeError):
    """Server queue is full."""

    def __init__(self, queue_limit: int) -> None:
        """Initialize error."""
        super().__init__(
            code=ErrorCode.QUEUE_FULL,
            message=f"Server queue is full (limit: {queue_limit})",
            details={"queue_limit": queue_limit},
            recoverable=True,
            recovery_hint="Wait for active sessions to complete",
        )


class ConcurrentLimitError(StoryForgeError):
    """Concurrent execution limit reached."""

    def __init__(self) -> None:
        """Initialize error."""
        super().__init__(
            code=ErrorCode.CONCURRENT_LIMIT,
            message="Server is processing another request",
            details={"limit": 1},
            recoverable=True,
            recovery_hint="Wait for active session to complete or check queue status",
        )
