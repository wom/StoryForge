"""Bundled stdio MCP server for StoryForge workflows."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from mcp.server import MCPServer
from mcp.server.mcpserver import Context

from .mcp_models import (
    ConfigResult,
    DraftResult,
    ExportRequest,
    ExtensionRequest,
    FinalizeRequest,
    GeneratedStory,
    GeneratedStorySummary,
    GenerationRequest,
    ModelRefreshResult,
    RefinementRequest,
    SessionSummary,
    StorySummary,
    WorkflowResult,
    WorldResult,
)
from .workflow import StoryForgeWorkflow

logging.basicConfig(level=logging.WARNING, format="%(levelname)s storyforge-mcp: %(message)s")

mcp = MCPServer(
    "StoryForge",
    instructions=(
        "Generate, extend, refine, resume, and export StoryForge stories. "
        "Draft tools pause for review; call finalize only after the draft is accepted."
    ),
)


@dataclass
class _WorkflowJob:
    """One serialized mutation executed by the dedicated server worker."""

    operation: Callable[[StoryForgeWorkflow], Any]
    done: threading.Event = field(default_factory=threading.Event)
    progress: queue.SimpleQueue[tuple[str, str, float | None]] = field(default_factory=queue.SimpleQueue)
    result: Any = None
    error: BaseException | None = None


_WORKFLOW_QUEUE: queue.Queue[_WorkflowJob] = queue.Queue(maxsize=1)
_WORKER_STARTED = threading.Event()


def _job_reporter(job: _WorkflowJob) -> Callable[[str, str, float | None], None]:
    """Bind progress reporting to one immutable job reference."""

    def report(kind: str, message: str, progress: float | None) -> None:
        job.progress.put((kind, message, progress))

    return report


def _worker_loop() -> None:
    """Execute workflow mutations one at a time outside the MCP event loop."""
    while True:
        job = _WORKFLOW_QUEUE.get()
        try:
            job.result = job.operation(StoryForgeWorkflow(reporter=_job_reporter(job)))
        except BaseException as error:  # preserve the original exception for MCP error conversion
            job.error = error
        finally:
            job.done.set()
            _WORKFLOW_QUEUE.task_done()


def _ensure_worker() -> None:
    if not _WORKER_STARTED.is_set():
        threading.Thread(target=_worker_loop, name="storyforge-mcp-worker", daemon=True).start()
        _WORKER_STARTED.set()


async def _run_workflow[ResultT](
    ctx: Context,
    operation: Callable[[StoryForgeWorkflow], ResultT],
) -> ResultT:
    """Run a blocking workflow in the server worker and bridge progress to MCP."""
    _ensure_worker()
    job = _WorkflowJob(operation=operation)
    try:
        _WORKFLOW_QUEUE.put_nowait(job)
    except queue.Full as error:
        raise RuntimeError("Another StoryForge generation workflow is already running") from error

    await ctx.report_progress(0.0, 1.0, "queued")
    while not job.done.is_set():
        while not job.progress.empty():
            _kind, message, progress = job.progress.get()
            await ctx.report_progress(progress or 0.0, 1.0, message)
        await asyncio.sleep(0.05)

    while not job.progress.empty():
        _kind, message, progress = job.progress.get()
        await ctx.report_progress(progress or 0.0, 1.0, message)
    if job.error is not None:
        raise job.error
    await ctx.report_progress(1.0, 1.0, "complete")
    return cast(ResultT, job.result)


@mcp.tool()
async def storyforge_list_stories(chain_only: bool = False) -> list[StorySummary]:
    """List saved stories, optionally limiting results to multi-part chains."""
    return StoryForgeWorkflow().list_stories(chain_only)


@mcp.tool()
async def storyforge_get_story(story_id: str) -> dict[str, Any]:
    """Read a saved story, its complete chain, and parsed metadata."""
    return StoryForgeWorkflow().get_story(story_id)


@mcp.tool()
async def storyforge_list_generated_stories() -> list[GeneratedStorySummary]:
    """List generated stories found in local StoryForge output directories."""
    return StoryForgeWorkflow().list_generated_stories()


@mcp.tool()
async def storyforge_get_generated_story(story_id: str) -> GeneratedStory:
    """Read a generated story and discover its associated output images."""
    return StoryForgeWorkflow().get_generated_story(story_id)


@mcp.tool()
async def storyforge_list_sessions(limit: int = 15) -> list[SessionSummary]:
    """List recent StoryForge checkpoints."""
    return StoryForgeWorkflow().list_sessions(limit)


@mcp.tool()
async def storyforge_get_session(session_id: str) -> dict[str, Any]:
    """Read a checkpoint and its recovery state."""
    return StoryForgeWorkflow().get_session(session_id)


@mcp.tool()
async def storyforge_resume_session(session_id: str, ctx: Context) -> DraftResult:
    """Resume a checkpoint and pause when its story is ready for review."""
    return await _run_workflow(ctx, lambda workflow: workflow.resume_session(session_id))


@mcp.tool()
async def storyforge_create_draft(request: GenerationRequest, ctx: Context) -> DraftResult:
    """Generate and save a new story draft, then pause for review."""
    return await _run_workflow(ctx, lambda workflow: workflow.create_draft(request))


@mcp.tool()
async def storyforge_create_extension_draft(request: ExtensionRequest, ctx: Context) -> DraftResult:
    """Generate a continuation draft from a saved story chain."""
    return await _run_workflow(ctx, lambda workflow: workflow.create_extension_draft(request))


@mcp.tool()
async def storyforge_refine_draft(request: RefinementRequest, ctx: Context) -> DraftResult:
    """Revise a staged draft using explicit refinement instructions."""
    return await _run_workflow(
        ctx,
        lambda workflow: workflow.refine_draft(request.session_id, request.instructions),
    )


@mcp.tool()
async def storyforge_finalize_story(request: FinalizeRequest, ctx: Context) -> WorkflowResult:
    """Accept a draft, create selected media, optionally save context, and complete it."""
    return await _run_workflow(ctx, lambda workflow: workflow.finalize_story(request))


@mcp.tool()
async def storyforge_export_chain(request: ExportRequest, ctx: Context) -> WorkflowResult:
    """Export a complete saved story chain to one text file."""
    return await _run_workflow(ctx, lambda workflow: workflow.export_chain(request))


@mcp.tool()
async def storyforge_get_config() -> ConfigResult:
    """Return resolved StoryForge configuration without credentials."""
    return StoryForgeWorkflow().get_config()


@mcp.tool()
async def storyforge_init_config(path: str | None = None, overwrite: bool = False) -> WorkflowResult:
    """Create the default configuration file; overwrite must be explicit."""
    return StoryForgeWorkflow().init_config(path, overwrite)


@mcp.tool()
async def storyforge_write_config(content: str) -> ConfigResult:
    """Validate and replace the active StoryForge configuration file."""
    return StoryForgeWorkflow().write_config(content)


@mcp.tool()
async def storyforge_configure_models(backend: str, story_model: str, image_model: str = "") -> ConfigResult:
    """Persist the selected provider models in StoryForge configuration."""
    return StoryForgeWorkflow().configure_models(backend, story_model, image_model)


@mcp.tool()
async def storyforge_read_world() -> WorldResult:
    """Read the active world definition and resolved path."""
    return StoryForgeWorkflow().read_world()


@mcp.tool()
async def storyforge_write_world(content: str, overwrite: bool = False) -> WorldResult:
    """Write a world definition; replacing an existing file must be explicit."""
    return StoryForgeWorkflow().write_world(content, overwrite)


@mcp.tool()
async def storyforge_list_models() -> dict[str, list[dict[str, Any]]]:
    """List cached model metadata by backend."""
    return StoryForgeWorkflow().list_models()


@mcp.tool()
async def storyforge_refresh_models() -> ModelRefreshResult:
    """Query configured providers now and atomically refresh their model caches."""
    return StoryForgeWorkflow().refresh_models()


@mcp.tool()
async def storyforge_clear_models(confirmed: bool = False) -> WorkflowResult:
    """Delete all model cache files; confirmed must be true."""
    return StoryForgeWorkflow().clear_models(confirmed)


@mcp.resource("storyforge://config")
async def config_resource() -> str:
    """Resolved StoryForge configuration."""
    return StoryForgeWorkflow().get_config().model_dump_json(indent=2)


@mcp.resource("storyforge://world")
async def world_resource() -> str:
    """Current StoryForge world definition."""
    return StoryForgeWorkflow().read_world().model_dump_json(indent=2)


@mcp.resource("storyforge://stories/{story_id}")
async def story_resource(story_id: str) -> dict[str, Any]:
    """A saved story and its lineage."""
    return StoryForgeWorkflow().get_story(story_id)


@mcp.resource("storyforge://sessions/{session_id}")
async def session_resource(session_id: str) -> dict[str, Any]:
    """A StoryForge checkpoint."""
    return StoryForgeWorkflow().get_session(session_id)


def main() -> None:
    """Run the bundled server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
