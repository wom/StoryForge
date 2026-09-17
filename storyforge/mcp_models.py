"""Typed request and result models shared by StoryForge MCP clients and server."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model that rejects misspelled or unsupported protocol fields."""

    model_config = ConfigDict(extra="forbid")


class GenerationRequest(StrictModel):
    """Inputs for a new story draft."""

    prompt: str = Field(min_length=1)
    length: str | None = None
    age_range: str | None = None
    style: str | None = None
    tone: str | None = None
    voice: str | None = None
    theme: str | None = None
    learning_focus: str | None = None
    setting: str | None = None
    characters: list[str] | None = None
    image_style: str | None = None
    image_count: int | None = Field(default=None, ge=1, le=5)
    output_dir: str | None = None
    use_context: bool | None = None
    world_file: str | None = None
    backend: str | None = None
    verbose: bool = False
    debug: bool = False


class ExtensionRequest(StrictModel):
    """Inputs for creating a continuation draft."""

    story_id: str
    ending_type: Literal["wrap_up", "cliffhanger"] = "cliffhanger"
    direction: str | None = None
    backend: str | None = None
    verbose: bool = False
    debug: bool = False


class RefinementRequest(StrictModel):
    """Instructions for revising a staged story."""

    session_id: str
    instructions: str = Field(min_length=1)


class FinalizeRequest(StrictModel):
    """Post-review media and persistence decisions."""

    session_id: str
    video_scene_count: int = Field(default=0, ge=0, le=20)
    image_count: int = Field(default=0, ge=0, le=5)
    save_context: bool = False


class ExportRequest(StrictModel):
    """Story-chain export request."""

    story_id: str
    output: str | None = None


class StorySummary(StrictModel):
    """Serializable story metadata for pickers and external MCP clients."""

    id: str
    filename: str
    filepath: str
    timestamp: str = ""
    prompt: str = ""
    preview: str = ""
    characters: str = ""
    theme: str = ""
    chain_length: int = 1


class GeneratedStorySummary(StrictModel):
    """One generated story available in the local output library."""

    id: str
    title: str
    story_path: str
    generated_at: str = ""
    preview: str = ""
    image_count: int = 0
    context_id: str | None = None


class GeneratedStory(GeneratedStorySummary):
    """Full generated story content and its associated local images."""

    content: str
    output_directory: str
    video_prompt_content: str | None = None
    image_paths: list[str] = Field(default_factory=list)


class SessionSummary(StrictModel):
    """Serializable checkpoint metadata."""

    session_id: str
    created_at: str = ""
    status: str = "unknown"
    current_phase: str = "unknown"
    prompt_preview: str = ""
    completion_percentage: int = 0


class DraftResult(StrictModel):
    """A generated story paused for user review."""

    session_id: str
    status: str
    story: str
    output_directory: str
    checkpoint_phase: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowResult(StrictModel):
    """Completed operation with generated artifacts."""

    session_id: str | None = None
    status: str = "completed"
    story: str | None = None
    output_directory: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    message: str = ""


class WorldResult(StrictModel):
    """World-file contents and resolved location."""

    path: str
    exists: bool
    content: str = ""


class ConfigResult(StrictModel):
    """Resolved configuration data safe for client display."""

    values: dict[str, Any]
    path: str | None = None
    content: str = ""


class ModelRefreshResult(StrictModel):
    """Provider model metadata and status returned by an explicit refresh."""

    models: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    statuses: dict[str, str] = Field(default_factory=dict)
    message: str = ""
