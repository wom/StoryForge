"""Shared type definitions for StoryForge MCP architecture."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorCode(Enum):
    """MCP error codes for structured error responses."""

    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    CHECKPOINT_CORRUPT = "CHECKPOINT_CORRUPT"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    CONTEXT_FILE_NOT_FOUND = "CONTEXT_FILE_NOT_FOUND"
    INVALID_PHASE = "INVALID_PHASE"
    GENERATION_FAILED = "GENERATION_FAILED"
    FILE_IO_ERROR = "FILE_IO_ERROR"
    CONFIG_ERROR = "CONFIG_ERROR"
    QUEUE_FULL = "QUEUE_FULL"
    CONCURRENT_LIMIT = "CONCURRENT_LIMIT"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class SessionStatus(Enum):
    """Session execution status."""

    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    RESUMING = "resuming"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ErrorResponse:
    """Structured error response for MCP tools."""

    code: str  # ErrorCode enum value
    message: str
    details: dict[str, Any] | None = None
    recoverable: bool = True
    recovery_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "recoverable": self.recoverable,
                "recovery_hint": self.recovery_hint,
            }
        }


class MCPError(Exception):
    """Exception class for MCP tool errors."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
        recoverable: bool = True,
        recovery_hint: str | None = None,
    ):
        """Initialize MCP error with structured information."""
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details
        self.recoverable = recoverable
        self.recovery_hint = recovery_hint

    def to_response(self) -> ErrorResponse:
        """Convert to ErrorResponse for serialization."""
        return ErrorResponse(
            code=self.code.value,
            message=self.message,
            details=self.details,
            recoverable=self.recoverable,
            recovery_hint=self.recovery_hint,
        )
