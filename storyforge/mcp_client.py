"""Shared async client for the bundled StoryForge MCP server."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any, TextIO, TypeVar

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

from .mcp_models import (
    DraftResult,
    ExportRequest,
    ExtensionRequest,
    FinalizeRequest,
    GeneratedStory,
    GeneratedStorySummary,
    GenerationRequest,
    RefinementRequest,
    SessionSummary,
    StorySummary,
    WorkflowResult,
)

ModelT = TypeVar("ModelT", bound=BaseModel)
ProgressCallback = Callable[[float, float | None, str | None], None]


class StoryForgeMCPError(RuntimeError):
    """Raised when the StoryForge MCP server returns a tool error."""


class StoryForgeMCPClient:
    """Typed facade over the bundled stdio MCP connection."""

    def __init__(
        self,
        progress_callback: ProgressCallback | None = None,
        server: Any | None = None,
        suppress_server_output: bool = False,
    ) -> None:
        self.progress_callback = progress_callback
        self.server = server
        self.suppress_server_output = suppress_server_output
        self._client_context: Client | None = None
        self._client: Client | None = None
        self._server_errlog: TextIO | None = None

    async def __aenter__(self) -> StoryForgeMCPClient:
        if self.server is None:
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "storyforge.mcp_server"],
                env=dict(os.environ),
            )
            if self.suppress_server_output:
                self._server_errlog = open(os.devnull, "w", encoding="utf-8")
            client_context = Client(stdio_client(params, errlog=self._server_errlog or sys.stderr))
        else:
            client_context = Client(self.server, raise_exceptions=True)
        self._client_context = client_context
        try:
            self._client = await client_context.__aenter__()
        except BaseException:
            self._close_server_errlog()
            raise
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self._client_context is not None:
                await self._client_context.__aexit__(exc_type, exc, traceback)
        finally:
            self._client = None
            self._client_context = None
            self._close_server_errlog()

    def _close_server_errlog(self) -> None:
        if self._server_errlog is not None:
            self._server_errlog.close()
            self._server_errlog = None

    async def _call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        if self._client is None:
            raise RuntimeError("StoryForge MCP client is not connected")

        async def on_progress(progress: float, total: float | None, message: str | None) -> None:
            if self.progress_callback is not None:
                self.progress_callback(progress, total, message)

        result = await self._client.call_tool(
            name,
            arguments or {},
            progress_callback=on_progress if self.progress_callback is not None else None,
        )
        if result.is_error:
            message = next((getattr(block, "text", "") for block in result.content if hasattr(block, "text")), "")
            raise StoryForgeMCPError(message or f"MCP tool failed: {name}")
        if result.structured_content is None:
            return None
        return result.structured_content

    async def _model(self, name: str, model: type[ModelT], arguments: dict[str, Any] | None = None) -> ModelT:
        return model.model_validate(await self._call(name, arguments))

    async def list_stories(self, chain_only: bool = False) -> list[StorySummary]:
        data = await self._call("storyforge_list_stories", {"chain_only": chain_only})
        values = data.get("result", data) if isinstance(data, dict) else data
        return [StorySummary.model_validate(item) for item in values or []]

    async def get_story(self, story_id: str) -> dict[str, Any]:
        return dict(await self._call("storyforge_get_story", {"story_id": story_id}) or {})

    async def list_generated_stories(self) -> list[GeneratedStorySummary]:
        data = await self._call("storyforge_list_generated_stories")
        values = data.get("result", data) if isinstance(data, dict) else data
        return [GeneratedStorySummary.model_validate(item) for item in values or []]

    async def get_generated_story(self, story_id: str) -> GeneratedStory:
        return await self._model(
            "storyforge_get_generated_story",
            GeneratedStory,
            {"story_id": story_id},
        )

    async def list_sessions(self, limit: int = 15) -> list[SessionSummary]:
        data = await self._call("storyforge_list_sessions", {"limit": limit})
        values = data.get("result", data) if isinstance(data, dict) else data
        return [SessionSummary.model_validate(item) for item in values or []]

    async def create_draft(self, request: GenerationRequest) -> DraftResult:
        return await self._model(
            "storyforge_create_draft",
            DraftResult,
            {"request": request.model_dump(mode="json")},
        )

    async def resume_session(self, session_id: str) -> DraftResult:
        return await self._model(
            "storyforge_resume_session",
            DraftResult,
            {"session_id": session_id},
        )

    async def create_extension_draft(self, request: ExtensionRequest) -> DraftResult:
        return await self._model(
            "storyforge_create_extension_draft",
            DraftResult,
            {"request": request.model_dump(mode="json")},
        )

    async def refine_draft(self, request: RefinementRequest) -> DraftResult:
        return await self._model(
            "storyforge_refine_draft",
            DraftResult,
            {"request": request.model_dump(mode="json")},
        )

    async def finalize_story(self, request: FinalizeRequest) -> WorkflowResult:
        return await self._model(
            "storyforge_finalize_story",
            WorkflowResult,
            {"request": request.model_dump(mode="json")},
        )

    async def export_chain(self, request: ExportRequest) -> WorkflowResult:
        return await self._model(
            "storyforge_export_chain",
            WorkflowResult,
            {"request": request.model_dump(mode="json")},
        )

    async def list_models(self) -> dict[str, list[dict[str, Any]]]:
        return dict(await self._call("storyforge_list_models") or {})

    async def get_config(self) -> dict[str, Any]:
        return dict(await self._call("storyforge_get_config") or {})

    async def init_config(self, path: str | None = None, overwrite: bool = False) -> WorkflowResult:
        return await self._model(
            "storyforge_init_config",
            WorkflowResult,
            {"path": path, "overwrite": overwrite},
        )

    async def read_world(self) -> dict[str, Any]:
        return dict(await self._call("storyforge_read_world") or {})

    async def write_world(self, content: str, overwrite: bool = False) -> dict[str, Any]:
        return dict(
            await self._call(
                "storyforge_write_world",
                {"content": content, "overwrite": overwrite},
            )
            or {}
        )

    async def invalidate_models(self) -> WorkflowResult:
        return await self._model("storyforge_invalidate_models", WorkflowResult)

    async def clear_models(self, confirmed: bool = False) -> WorkflowResult:
        return await self._model("storyforge_clear_models", WorkflowResult, {"confirmed": confirmed})
