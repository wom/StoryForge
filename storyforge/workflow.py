"""UI-independent StoryForge workflows exposed through MCP."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from yaml import YAMLError, safe_load

from .checkpoint import CheckpointData, CheckpointManager, ExecutionPhase
from .config import Config, load_config
from .context import ContextManager
from .mcp_models import (
    ConfigResult,
    DraftResult,
    ExportRequest,
    ExtensionRequest,
    FinalizeRequest,
    GeneratedStory,
    GeneratedStorySummary,
    GenerationRequest,
    SessionSummary,
    StorySummary,
    WorkflowResult,
    WorldResult,
)
from .paths import create_output_directory_name, resolve_world_file_path
from .phase_executor import PhaseExecutor
from .prompt import Prompt

Reporter = Callable[[str, str, float | None], None]

POST_STORY_PHASES = frozenset(
    {
        ExecutionPhase.VIDEO_DECISION.value,
        ExecutionPhase.IMAGE_DECISION.value,
        ExecutionPhase.VIDEO_PROMPT_GENERATE.value,
        ExecutionPhase.IMAGE_GENERATE.value,
        ExecutionPhase.CONTEXT_SAVE.value,
        ExecutionPhase.COMPLETED.value,
    }
)

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})


class StoryForgeWorkflow:
    """Application service shared by the bundled MCP tools."""

    def __init__(self, reporter: Reporter | None = None) -> None:
        self.reporter = reporter

    @staticmethod
    def _config_value(config: Config, section: str, field: str, supplied: Any) -> Any:
        return supplied if supplied is not None else config.get_field_value(section, field)

    @staticmethod
    def _output_directory(requested: str | None, *, extended: bool = False) -> str:
        if requested:
            return requested
        return create_output_directory_name(extended=extended)

    def _executor(self, manager: CheckpointManager) -> PhaseExecutor:
        return PhaseExecutor(manager, reporter=self.reporter)

    @staticmethod
    def _draft_result(checkpoint: CheckpointData) -> DraftResult:
        return DraftResult(
            session_id=checkpoint.session_id,
            status=checkpoint.status,
            story=str(checkpoint.generated_content.get("story") or ""),
            output_directory=str(checkpoint.resolved_config.get("output_directory") or ""),
            checkpoint_phase=checkpoint.current_phase,
            metadata={
                "prompt": checkpoint.original_inputs.get("prompt", ""),
                "refinements": checkpoint.generated_content.get("refinements"),
                "image_count": checkpoint.resolved_config.get("image_count", 3),
                "image_style": checkpoint.resolved_config.get("image_style", "chibi"),
            },
        )

    @staticmethod
    def _find_checkpoint(manager: CheckpointManager, session_id: str) -> CheckpointData:
        matches = list(manager.checkpoint_dir.glob(f"checkpoint_{session_id}.yaml"))
        if not matches:
            raise FileNotFoundError(f"StoryForge session not found: {session_id}")
        return manager.load_checkpoint(matches[0])

    def list_stories(self, chain_only: bool = False) -> list[StorySummary]:
        manager = ContextManager()
        stories: list[StorySummary] = []
        for item in manager.list_available_contexts():
            chain_length = len(manager.get_story_chain(item["filepath"]))
            if chain_only and chain_length < 2:
                continue
            characters = item.get("characters", "")
            if isinstance(characters, list):
                characters = ", ".join(str(value) for value in characters)
            stories.append(
                StorySummary(
                    id=Path(item["filepath"]).stem,
                    filename=str(item.get("filename", Path(item["filepath"]).stem)),
                    filepath=str(item["filepath"]),
                    timestamp=str(item.get("timestamp", "")),
                    prompt=str(item.get("prompt", "")),
                    preview=str(item.get("preview", "")),
                    characters=str(characters),
                    theme=str(item.get("theme", "")),
                    chain_length=chain_length,
                )
            )
        return stories

    def get_story(self, story_id: str) -> dict[str, Any]:
        manager = ContextManager()
        item = self._find_story(manager, story_id)
        chain = manager.get_story_chain(item["filepath"])
        content, metadata = manager.load_chain_for_extension(item["filepath"])
        return {
            "id": Path(item["filepath"]).stem,
            "metadata": self._json_safe(metadata),
            "chain": self._json_safe(chain),
            "content": content,
        }

    def list_generated_stories(self) -> list[GeneratedStorySummary]:
        """List story artifacts produced in known StoryForge output directories."""
        stories: list[GeneratedStorySummary] = []
        context_ids = self._generated_story_context_ids()
        for story_path in self._generated_story_paths():
            try:
                content = story_path.read_text(encoding="utf-8")
                stat = story_path.stat()
            except OSError:
                continue
            images = self._story_images(story_path.parent)
            stories.append(
                GeneratedStorySummary(
                    id=str(story_path.resolve()),
                    title=self._story_title(content, story_path.parent.name),
                    story_path=str(story_path.resolve()),
                    generated_at=datetime.fromtimestamp(stat.st_mtime).isoformat(sep=" ", timespec="seconds"),
                    preview=self._story_preview(content),
                    image_count=len(images),
                    context_id=context_ids.get(story_path),
                )
            )
        return sorted(stories, key=lambda story: story.generated_at, reverse=True)

    def get_generated_story(self, story_id: str) -> GeneratedStory:
        """Load one story from the generated-output library."""
        summary = next((story for story in self.list_generated_stories() if story.id == story_id), None)
        if summary is None:
            raise FileNotFoundError(f"Generated story not found: {story_id}")
        story_path = Path(summary.story_path)
        images = self._story_images(story_path.parent)
        video_prompt_path = story_path.parent / "video_prompt.txt"
        return GeneratedStory(
            **summary.model_dump(),
            content=story_path.read_text(encoding="utf-8"),
            output_directory=str(story_path.parent),
            video_prompt_content=(
                video_prompt_path.read_text(encoding="utf-8") if video_prompt_path.is_file() else None
            ),
            image_paths=[str(path) for path in images],
        )

    def list_sessions(self, limit: int = 15) -> list[SessionSummary]:
        manager = CheckpointManager(auto_cleanup=False)
        result: list[SessionSummary] = []
        for path in manager.find_recent_checkpoints(limit):
            info = manager.get_checkpoint_info(path)
            result.append(
                SessionSummary(
                    session_id=str(info["session_id"]),
                    created_at=str(info["created_at"]),
                    status=str(info["status"]),
                    current_phase=str(info["current_phase"]),
                    prompt_preview=str(info["prompt_preview"]),
                    completion_percentage=int(info["completion_percentage"]),
                )
            )
        return result

    def get_session(self, session_id: str) -> dict[str, Any]:
        manager = CheckpointManager(auto_cleanup=False)
        checkpoint = self._find_checkpoint(manager, session_id)
        return dict(self._json_safe(checkpoint.__dict__))

    def resume_session(self, session_id: str) -> DraftResult:
        """Resume a checkpoint to the draft-review boundary."""
        manager = CheckpointManager(auto_cleanup=False)
        checkpoint = self._find_checkpoint(manager, session_id)
        if checkpoint.generated_content.get("story"):
            checkpoint.status = "active"
            checkpoint.last_error = None
            checkpoint.completed_phases = [
                phase for phase in checkpoint.completed_phases if phase not in POST_STORY_PHASES
            ]
            checkpoint.current_phase = ExecutionPhase.STORY_SAVE.value
            if checkpoint.progress:
                completed_count = len(checkpoint.completed_phases)
                total_phases = int(checkpoint.progress.get("total_phases") or len(ExecutionPhase) - 1)
                checkpoint.progress["completed_count"] = completed_count
                checkpoint.progress["completion_percentage"] = round(completed_count / total_phases * 100)
            manager.save_checkpoint(checkpoint)
            if ExecutionPhase.STORY_SAVE.value not in checkpoint.completed_phases:
                checkpoint.resolved_config.update({"auto_confirm": True, "defer_story_review": True})
                checkpoint = self._executor(manager).execute_existing_session(
                    checkpoint,
                    ExecutionPhase.STORY_SAVE,
                    stop_after=ExecutionPhase.STORY_SAVE,
                )
            return self._draft_result(checkpoint)

        try:
            start_phase = ExecutionPhase(checkpoint.current_phase)
        except ValueError as error:
            raise ValueError(f"Unsupported checkpoint phase: {checkpoint.current_phase}") from error
        checkpoint.status = "active"
        checkpoint.last_error = None
        checkpoint.resolved_config.update({"auto_confirm": True, "defer_story_review": True})
        checkpoint = self._executor(manager).execute_existing_session(
            checkpoint,
            start_phase,
            stop_after=ExecutionPhase.STORY_SAVE,
        )
        return self._draft_result(checkpoint)

    def create_draft(self, request: GenerationRequest) -> DraftResult:
        config = load_config(verbose=request.verbose)
        values = {
            "length": self._config_value(config, "story", "length", request.length),
            "age_range": self._config_value(config, "story", "age_range", request.age_range),
            "style": self._config_value(config, "story", "style", request.style),
            "tone": self._config_value(config, "story", "tone", request.tone),
            "voice": self._config_value(config, "story", "voice", request.voice),
            "theme": self._config_value(config, "story", "theme", request.theme),
            "learning_focus": self._config_value(config, "story", "learning_focus", request.learning_focus),
            "setting": self._config_value(config, "story", "setting", request.setting),
            "characters": self._config_value(config, "story", "characters", request.characters),
            "image_style": self._config_value(config, "images", "image_style", request.image_style),
            "image_count": self._config_value(config, "images", "image_count", request.image_count),
        }
        configured_output_dir = self._config_value(config, "output", "output_dir", request.output_dir)
        output_dir = self._output_directory(configured_output_dir)
        use_context = self._config_value(config, "output", "use_context", request.use_context)
        world_file = self._config_value(config, "output", "world_file", request.world_file)
        config_backend = config.get_field_value("system", "backend")
        cli_arguments = {
            **values,
            "output_dir": output_dir,
            "use_context": use_context,
            "world_file": world_file,
            "backend": request.backend,
            "verbose": request.verbose,
            "debug": request.debug,
        }
        resolved_config = {
            **values,
            "output_directory": output_dir,
            "use_context": use_context,
            "world_file": world_file,
            "backend": request.backend,
            "config_backend": config_backend,
            "verbose": request.verbose,
            "debug": request.debug,
            "auto_confirm": True,
            "defer_story_review": True,
        }
        manager = CheckpointManager()
        checkpoint = self._executor(manager).execute_new_session(
            request.prompt,
            cli_arguments,
            resolved_config,
            stop_after=ExecutionPhase.STORY_SAVE,
        )
        return self._draft_result(checkpoint)

    def create_extension_draft(self, request: ExtensionRequest) -> DraftResult:
        context_manager = ContextManager()
        selected = self._find_story(context_manager, request.story_id)
        story_content, metadata = context_manager.load_chain_for_extension(selected["filepath"])
        config = load_config(verbose=request.verbose)
        characters_value = metadata.get("characters", [])
        if isinstance(characters_value, str):
            characters = [value.strip() for value in characters_value.split(",") if value.strip()]
        else:
            characters = list(characters_value or [])
        prompt = Prompt(
            prompt="",
            characters=characters or None,
            theme=metadata.get("theme"),
            age_range=metadata.get("age_group") or config.get_field_value("story", "age_range") or "preschool",
            tone=metadata.get("tone") or config.get_field_value("story", "tone") or "heartwarming",
            voice=metadata.get("voice") or config.get_field_value("story", "voice") or None,
            length=metadata.get("length") or config.get_field_value("story", "length") or "short",
            style=metadata.get("style") or config.get_field_value("story", "style") or "adventure",
            image_style=metadata.get("art_style") or config.get_field_value("images", "image_style") or "chibi",
            setting=metadata.get("setting") or config.get_field_value("story", "setting") or None,
            learning_focus=metadata.get("learning_focus") or config.get_field_value("story", "learning_focus") or None,
            context=story_content,
            continuation_mode=True,
            ending_type=request.ending_type,
            continuation_direction=request.direction,
        )
        output_dir = self._output_directory(None, extended=True)
        cli_arguments = {
            "backend": request.backend,
            "verbose": request.verbose,
            "debug": request.debug,
            "continuation_mode": True,
            "ending_type": request.ending_type,
            "continuation_direction": request.direction,
            "output_dir": output_dir,
            "age_range": prompt.age_range,
            "length": prompt.length,
            "style": prompt.style,
            "tone": prompt.tone,
            "voice": prompt.voice,
            "image_style": prompt.image_style,
            "theme": prompt.theme,
            "characters": prompt.characters,
            "setting": prompt.setting,
            "learning_focus": prompt.learning_focus,
        }
        resolved_config = {
            **cli_arguments,
            "config_backend": config.get_field_value("system", "backend"),
            "output_directory": output_dir,
            "source_context_file": str(selected["filepath"]),
            "continuation_context": story_content,
            "auto_confirm": True,
            "defer_story_review": True,
        }
        manager = CheckpointManager()
        checkpoint = self._executor(manager).execute_new_session(
            f"[EXTENSION] {selected['filename']}",
            cli_arguments,
            resolved_config,
            prompt_obj=prompt,
            stop_after=ExecutionPhase.STORY_SAVE,
        )
        return self._draft_result(checkpoint)

    def refine_draft(self, session_id: str, instructions: str) -> DraftResult:
        manager = CheckpointManager(auto_cleanup=False)
        checkpoint = self._find_checkpoint(manager, session_id)
        checkpoint = self._executor(manager).refine_existing_story(checkpoint, instructions)
        return self._draft_result(checkpoint)

    def finalize_story(self, request: FinalizeRequest) -> WorkflowResult:
        manager = CheckpointManager(auto_cleanup=False)
        checkpoint = self._find_checkpoint(manager, request.session_id)
        checkpoint.resolved_config.update(
            {
                "video_scene_count": request.video_scene_count,
                "final_image_count": request.image_count,
                "final_save_context": request.save_context,
                "defer_story_review": True,
                "auto_confirm": True,
            }
        )
        checkpoint.user_decisions.update(
            {
                "story_accepted": True,
                "wants_video_prompt": None,
                "num_video_scenes": None,
                "wants_images": None,
                "num_images_requested": None,
                "save_as_context": None,
            }
        )
        checkpoint = self._executor(manager).execute_existing_session(checkpoint, ExecutionPhase.VIDEO_DECISION)
        artifacts = self._artifacts(checkpoint)
        return WorkflowResult(
            session_id=checkpoint.session_id,
            status=checkpoint.status,
            story=str(checkpoint.generated_content.get("story") or ""),
            output_directory=str(checkpoint.resolved_config.get("output_directory") or ""),
            artifacts=artifacts,
            message="Story generation completed.",
        )

    def export_chain(self, request: ExportRequest) -> WorkflowResult:
        manager = ContextManager()
        selected = self._find_story(manager, request.story_id)
        if request.output:
            output = Path(request.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = Path(f"complete_story_{selected['filename']}_{timestamp}.txt")
        result = manager.write_chain_to_file(selected["filepath"], output)
        return WorkflowResult(artifacts=[str(result)], message=f"Exported story chain to {result}")

    def get_config(self) -> ConfigResult:
        config = load_config(verbose=False)
        path = next((str(path) for path in config.get_config_paths() if path.exists()), None)
        return ConfigResult(values=config.to_dict(), path=path)

    def init_config(self, path: str | None = None, overwrite: bool = False) -> WorkflowResult:
        config = Config()
        target = Path(path) if path else config.get_default_config_path()
        if target.exists() and not overwrite:
            raise FileExistsError(f"Configuration file already exists: {target}")
        result = config.create_default_config(target)
        return WorkflowResult(artifacts=[str(result)], message=f"Created configuration file: {result}")

    def read_world(self) -> WorldResult:
        manager = ContextManager()
        path = manager._discover_world_file()
        if path is None:
            path = resolve_world_file_path()
            return WorldResult(path=str(path), exists=False)
        return WorldResult(path=str(path), exists=True, content=path.read_text(encoding="utf-8"))

    def write_world(self, content: str, overwrite: bool = False) -> WorldResult:
        path = resolve_world_file_path()
        if path.exists() and not overwrite:
            raise FileExistsError(f"World file already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return WorldResult(path=str(path), exists=True, content=content)

    def list_models(self) -> dict[str, list[dict[str, Any]]]:
        from .model_cache import ModelCache

        cache = ModelCache()
        result: dict[str, list[dict[str, Any]]] = {}
        for backend in ("gemini", "openai", "anthropic"):
            path = cache.cache_path(backend)
            if not path.exists():
                result[backend] = []
                continue
            try:
                result[backend] = list(json.loads(path.read_text(encoding="utf-8")).get("models", []))
            except (OSError, json.JSONDecodeError, TypeError):
                result[backend] = []
        return result

    def invalidate_models(self) -> WorkflowResult:
        from .model_cache import ModelCache

        cache = ModelCache()
        for backend in ("gemini", "openai", "anthropic"):
            cache.invalidate(backend)
        return WorkflowResult(message="All model caches invalidated.")

    def clear_models(self, confirmed: bool = False) -> WorkflowResult:
        if not confirmed:
            raise ValueError("Clearing model caches requires confirmed=true")
        from .model_cache import ModelCache

        ModelCache().clear_all()
        return WorkflowResult(message="All cached model data cleared.")

    @staticmethod
    def _find_story(manager: ContextManager, story_id: str) -> dict[str, Any]:
        matches = [
            story
            for story in manager.list_available_contexts()
            if story_id in {Path(story["filepath"]).stem, str(story.get("filename", "")), str(story["filepath"])}
        ]
        if not matches:
            raise FileNotFoundError(f"Saved story not found: {story_id}")
        return matches[0]

    @staticmethod
    def _generated_story_paths() -> list[Path]:
        paths = {path.resolve() for path in Path.cwd().glob("*/story.txt") if path.is_file()}
        direct_story = Path.cwd() / "story.txt"
        if direct_story.is_file():
            paths.add(direct_story.resolve())

        manager = CheckpointManager(auto_cleanup=False)
        for checkpoint_path in manager.checkpoint_dir.glob("checkpoint_*.yaml"):
            try:
                checkpoint = safe_load(checkpoint_path.read_text(encoding="utf-8")) or {}
                output_directory = checkpoint.get("resolved_config", {}).get("output_directory")
                if output_directory:
                    story_path = Path(str(output_directory)).expanduser() / "story.txt"
                    if story_path.is_file():
                        paths.add(story_path.resolve())
            except (OSError, TypeError, ValueError, YAMLError):
                continue
        return list(paths)

    @staticmethod
    def _generated_story_context_ids() -> dict[Path, str]:
        """Map generated story artifacts to their saved extension contexts."""
        context_ids: dict[Path, str] = {}
        manager = CheckpointManager(auto_cleanup=False)
        for checkpoint_path in manager.checkpoint_dir.glob("checkpoint_*.yaml"):
            try:
                checkpoint = safe_load(checkpoint_path.read_text(encoding="utf-8")) or {}
                output_directory = checkpoint.get("resolved_config", {}).get("output_directory")
                context_file = checkpoint.get("generated_content", {}).get("context_file")
                if output_directory and context_file and Path(str(context_file)).expanduser().is_file():
                    story_path = (Path(str(output_directory)).expanduser() / "story.txt").resolve()
                    context_ids[story_path] = Path(str(context_file)).stem
            except (AttributeError, OSError, TypeError, ValueError, YAMLError):
                continue
        return context_ids

    @staticmethod
    def _story_images(output_directory: Path) -> list[Path]:
        try:
            return sorted(
                path.resolve()
                for path in output_directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        except OSError:
            return []

    @staticmethod
    def _story_title(content: str, fallback: str) -> str:
        first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
        if first_line.lower().startswith("story:"):
            first_line = first_line.split(":", 1)[1].strip()
        return first_line.lstrip("# ").strip() or fallback.replace("_", " ").title()

    @staticmethod
    def _story_preview(content: str) -> str:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if lines and lines[0].lower().startswith("story:"):
            lines.pop(0)
        return " ".join(lines)[:240]

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): StoryForgeWorkflow._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [StoryForgeWorkflow._json_safe(item) for item in value]
        return value

    @staticmethod
    def _artifacts(checkpoint: CheckpointData) -> list[str]:
        artifacts: list[str] = []
        output_dir = checkpoint.resolved_config.get("output_directory")
        if output_dir:
            story_path = Path(str(output_dir)) / "story.txt"
            if story_path.exists():
                artifacts.append(str(story_path))
            video_path = Path(str(output_dir)) / "video_prompt.txt"
            if video_path.exists():
                artifacts.append(str(video_path))
        for image in checkpoint.generated_content.get("generated_images", []):
            if isinstance(image, dict) and image.get("filename"):
                artifacts.append(str(image["filename"]))
        context_file = checkpoint.generated_content.get("context_file")
        if context_file:
            artifacts.append(str(context_file))
        return artifacts
