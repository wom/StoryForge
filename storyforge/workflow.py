"""UI-independent StoryForge workflows exposed through MCP."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Callable
from configparser import Error as ConfigParserError
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from yaml import YAMLError, safe_load

from .atomic_io import atomic_write_text
from .checkpoint import CheckpointData, CheckpointManager, ExecutionPhase
from .config import Config, ConfigError, load_config, update_config_values
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
    ModelRefreshResult,
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
        self._recover_pending_replacement(checkpoint, manager)
        if checkpoint.generated_content.get("story"):
            if "superseded_artifacts" not in checkpoint.resolved_config:
                output_dir = Path(str(checkpoint.resolved_config.get("output_directory") or ""))
                checkpoint.resolved_config["superseded_artifacts"] = {
                    "images": checkpoint.generated_content.pop("generated_images", []),
                    "context_file": checkpoint.generated_content.pop("context_file", None),
                    "video_prompt": str(output_dir / "video_prompt.txt") if output_dir != Path(".") else None,
                }
                checkpoint.generated_content.pop("video_prompts", None)
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
        config = load_config(verbose=bool(request.verbose))
        verbose = self._config_value(config, "system", "verbose", request.verbose)
        debug = self._config_value(config, "system", "debug", request.debug)
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
            "verbose": verbose,
            "debug": debug,
        }
        resolved_config = {
            **values,
            "output_directory": output_dir,
            "use_context": use_context,
            "world_file": world_file,
            "backend": request.backend,
            "config_backend": config_backend,
            "verbose": verbose,
            "debug": debug,
            "auto_confirm": True,
            "defer_story_review": True,
        }
        # Migrate older checkpoint-only library records before startup cleanup
        # can prune the checkpoint that names a custom output directory.
        self._generated_story_paths()
        manager = CheckpointManager()
        checkpoint = self._executor(manager).execute_new_session(
            request.prompt,
            cli_arguments,
            resolved_config,
            stop_after=ExecutionPhase.STORY_SAVE,
        )
        self._remember_generated_story(checkpoint)
        return self._draft_result(checkpoint)

    def create_extension_draft(self, request: ExtensionRequest) -> DraftResult:
        context_manager = ContextManager()
        selected = self._find_story(context_manager, request.story_id)
        story_content, metadata = context_manager.load_chain_for_extension(selected["filepath"])
        config = load_config(verbose=bool(request.verbose))
        verbose = self._config_value(config, "system", "verbose", request.verbose)
        debug = self._config_value(config, "system", "debug", request.debug)
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
            "verbose": verbose,
            "debug": debug,
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
            "image_count": config.get_field_value("images", "image_count"),
            "theme": prompt.theme,
            "characters": prompt.characters,
            "setting": prompt.setting,
            "learning_focus": prompt.learning_focus,
        }
        resolved_config = {
            **cli_arguments,
            "world_file": config.get_field_value("output", "world_file"),
            "config_backend": config.get_field_value("system", "backend"),
            "output_directory": output_dir,
            "source_context_file": str(selected["filepath"]),
            "continuation_context": story_content,
            "auto_confirm": True,
            "defer_story_review": True,
        }
        self._generated_story_paths()
        manager = CheckpointManager()
        checkpoint = self._executor(manager).execute_new_session(
            f"[EXTENSION] {selected['filename']}",
            cli_arguments,
            resolved_config,
            prompt_obj=prompt,
            stop_after=ExecutionPhase.STORY_SAVE,
        )
        self._remember_generated_story(checkpoint)
        return self._draft_result(checkpoint)

    def refine_draft(self, session_id: str, instructions: str) -> DraftResult:
        manager = CheckpointManager(auto_cleanup=False)
        checkpoint = self._find_checkpoint(manager, session_id)
        if checkpoint.status != "active" or checkpoint.current_phase != ExecutionPhase.STORY_SAVE.value:
            raise ValueError("Refinement requires an active draft; resume the session before refining it")
        checkpoint = self._executor(manager).refine_existing_story(checkpoint, instructions)
        return self._draft_result(checkpoint)

    def finalize_story(self, request: FinalizeRequest) -> WorkflowResult:
        manager = CheckpointManager(auto_cleanup=False)
        checkpoint = self._find_checkpoint(manager, request.session_id)
        self._recover_pending_replacement(checkpoint, manager)
        previous = checkpoint.resolved_config.get("superseded_artifacts")
        if not str(checkpoint.resolved_config.get("output_directory") or ""):
            raise ValueError("Cannot finalize a story without an output directory")
        output_dir = Path(str(checkpoint.resolved_config["output_directory"])).expanduser().resolve()
        stage_dir: Path | None = None
        if isinstance(previous, dict):
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint.resolved_config["output_directory"] = str(output_dir)
            stage_dir = Path(tempfile.mkdtemp(prefix=".storyforge-finalize-", dir=output_dir))
            checkpoint.resolved_config["pending_media_stage"] = str(stage_dir)
            checkpoint.resolved_config["pending_media_promoted"] = False
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
        try:
            if stage_dir is not None:
                manager.save_checkpoint(checkpoint)
            executor = self._executor(manager)
            if stage_dir is not None:
                executor.media_output_directory = stage_dir
            checkpoint = executor.execute_existing_session(checkpoint, ExecutionPhase.VIDEO_DECISION)
            if stage_dir is not None:
                images = checkpoint.generated_content.get("generated_images") or []
                if len(images) != request.image_count:
                    raise RuntimeError("Replacement image generation did not complete; previous media was preserved")
                if request.video_scene_count and not (stage_dir / "video_prompt.txt").is_file():
                    raise RuntimeError("Replacement video prompt did not complete; previous media was preserved")
                context_path = Path(str(checkpoint.generated_content.get("context_file") or ""))
                if request.save_context and not context_path.is_file():
                    raise RuntimeError("Replacement story context did not save; previous context was preserved")
                self._commit_replacement_media(checkpoint, stage_dir, output_dir, manager)
                self._discard_superseded_artifacts(checkpoint, keep_new_video=bool(request.video_scene_count))
                checkpoint.resolved_config.pop("pending_media_stage", None)
                checkpoint.resolved_config.pop("pending_media_promoted", None)
                manager.save_checkpoint(checkpoint)
        except BaseException:
            if stage_dir is not None:
                self._recover_pending_replacement(checkpoint, manager)
            raise
        finally:
            if (
                stage_dir is not None
                and stage_dir.is_dir()
                and "pending_media_stage" not in checkpoint.resolved_config
            ):
                shutil.rmtree(stage_dir)
        self._remember_generated_story(checkpoint)
        artifacts = self._artifacts(checkpoint)
        return WorkflowResult(
            session_id=checkpoint.session_id,
            status=checkpoint.status,
            story=str(checkpoint.generated_content.get("story") or ""),
            output_directory=str(checkpoint.resolved_config.get("output_directory") or ""),
            artifacts=artifacts,
            message="Story generation completed.",
        )

    @staticmethod
    def _file_digest(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    @staticmethod
    def _remove_failed_context(checkpoint: CheckpointData) -> None:
        new_context = checkpoint.generated_content.get("context_file")
        if not new_context:
            return
        try:
            candidate = Path(str(new_context)).expanduser().resolve()
            context_manager = ContextManager()
            if candidate.parent == context_manager.get_context_directory().resolve() and candidate.name.endswith(
                f"_{checkpoint.session_id}.md"
            ):
                candidate.unlink(missing_ok=True)
                context_manager.build_character_registry()
        except OSError:
            logging.getLogger(__name__).warning("Could not clean up failed replacement context", exc_info=True)

    @staticmethod
    def _rollback_staged_media(stage_dir: Path, output_dir: Path) -> None:
        manifest_path = stage_dir / "promotion_manifest.json"
        if not manifest_path.is_file():
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Invalid replacement media manifest")
        for item in manifest.get("images", []):
            if not isinstance(item, dict):
                continue
            name, digest = item.get("name"), item.get("sha256")
            if (
                not isinstance(name, str)
                or not isinstance(digest, str)
                or Path(name).name != name
                or Path(name).suffix.lower() not in IMAGE_SUFFIXES
            ):
                continue
            target = output_dir / name
            if target.is_file() and StoryForgeWorkflow._file_digest(target) == digest:
                target.unlink()

        video_digest = manifest.get("video_sha256")
        video_path = output_dir / "video_prompt.txt"
        if isinstance(video_digest, str) and video_path.is_file():
            if StoryForgeWorkflow._file_digest(video_path) == video_digest:
                backup = stage_dir / "previous_video_prompt.txt"
                if backup.is_file():
                    os.replace(backup, video_path)
                else:
                    video_path.unlink()

    def _recover_pending_replacement(self, checkpoint: CheckpointData, manager: CheckpointManager) -> None:
        """Finish or roll back an interrupted media transaction before resuming."""
        raw_stage = checkpoint.resolved_config.get("pending_media_stage")
        if not raw_stage:
            return
        output_dir = Path(str(checkpoint.resolved_config.get("output_directory") or "")).expanduser().resolve()
        stage_dir = Path(str(raw_stage)).expanduser().resolve()
        if stage_dir.parent != output_dir or not stage_dir.name.startswith(".storyforge-finalize-"):
            raise ValueError("Invalid pending replacement staging path")

        if checkpoint.resolved_config.get("pending_media_promoted"):
            for item in checkpoint.generated_content.get("generated_images") or []:
                if not isinstance(item, dict):
                    raise RuntimeError("Promoted replacement image metadata is invalid")
                image = Path(str(item.get("filename") or "")).expanduser().resolve()
                if image.parent != output_dir or not image.is_file():
                    raise RuntimeError("Promoted replacement image is missing; previous assets were retained")
            if checkpoint.resolved_config.get("video_scene_count") and not (output_dir / "video_prompt.txt").is_file():
                raise RuntimeError("Promoted replacement video prompt is missing; previous assets were retained")
            context_file = checkpoint.generated_content.get("context_file")
            if checkpoint.resolved_config.get("final_save_context") and not (
                context_file and Path(str(context_file)).expanduser().is_file()
            ):
                raise RuntimeError("Promoted replacement context is missing; previous assets were retained")
            self._discard_superseded_artifacts(
                checkpoint,
                keep_new_video=bool(checkpoint.resolved_config.get("video_scene_count")),
            )
        else:
            if stage_dir.is_dir():
                self._rollback_staged_media(stage_dir, output_dir)
            self._remove_failed_context(checkpoint)
            checkpoint.generated_content.pop("generated_images", None)
            checkpoint.generated_content.pop("video_prompts", None)
            checkpoint.generated_content.pop("context_file", None)
            checkpoint.status = "active"
            checkpoint.current_phase = ExecutionPhase.STORY_SAVE.value
            checkpoint.completed_phases = [
                phase for phase in checkpoint.completed_phases if phase not in POST_STORY_PHASES
            ]

        checkpoint.resolved_config.pop("pending_media_stage", None)
        checkpoint.resolved_config.pop("pending_media_promoted", None)
        manager.save_checkpoint(checkpoint)
        if stage_dir.is_dir():
            shutil.rmtree(stage_dir)

    @staticmethod
    def _discard_superseded_artifacts(checkpoint: CheckpointData, *, keep_new_video: bool = False) -> None:
        """Remove only files owned by a resumed session's previous finalization."""
        previous = checkpoint.resolved_config.get("superseded_artifacts")
        if not isinstance(previous, dict):
            return
        output = Path(str(checkpoint.resolved_config.get("output_directory") or "")).expanduser().resolve()
        if str(checkpoint.resolved_config.get("output_directory") or ""):
            for item in previous.get("images", []):
                if not isinstance(item, dict) or not item.get("filename"):
                    continue
                image = Path(str(item["filename"])).expanduser().resolve()
                if image.parent == output and image.suffix.lower() in IMAGE_SUFFIXES:
                    image.unlink(missing_ok=True)
            video = previous.get("video_prompt")
            if not keep_new_video and video and Path(str(video)).expanduser().resolve() == output / "video_prompt.txt":
                (output / "video_prompt.txt").unlink(missing_ok=True)

        context_file = previous.get("context_file")
        if context_file:
            manager = ContextManager()
            context = Path(str(context_file)).expanduser().resolve()
            context_dir = manager.get_context_directory().resolve()
            if context.parent == context_dir and context.suffix.lower() == ".md":
                linked = any(
                    candidate != context
                    and manager.parse_context_metadata(candidate).get("extended_from") == context.stem
                    for candidate in manager._discover_context_files()
                )
                if not linked and context.is_file():
                    context.unlink()
                    manager.build_character_registry()
        checkpoint.resolved_config.pop("superseded_artifacts", None)

    @staticmethod
    def _commit_replacement_media(
        checkpoint: CheckpointData, stage_dir: Path, output_dir: Path, manager: CheckpointManager
    ) -> None:
        """Promote staged media with a durable manifest for crash recovery."""
        planned_images: list[tuple[dict[str, Any], Path, Path]] = []
        image_manifest: list[dict[str, str]] = []
        for item in checkpoint.generated_content.get("generated_images") or []:
            source = Path(str(item["filename"]))
            if source.parent.resolve() != stage_dir.resolve() or not source.is_file():
                raise RuntimeError("Replacement image is missing from staging")
            target = output_dir / f"{source.stem}_{uuid4().hex[:8]}{source.suffix}"
            while target.exists():
                target = output_dir / f"{source.stem}_{uuid4().hex[:8]}{source.suffix}"
            planned_images.append((item, source, target))
            image_manifest.append({"name": target.name, "sha256": StoryForgeWorkflow._file_digest(source)})

        old_video = output_dir / "video_prompt.txt"
        staged_video = stage_dir / "video_prompt.txt"
        video_backup = stage_dir / "previous_video_prompt.txt"
        video_digest = StoryForgeWorkflow._file_digest(staged_video) if staged_video.is_file() else None
        atomic_write_text(
            stage_dir / "promotion_manifest.json",
            json.dumps({"images": image_manifest, "video_sha256": video_digest}) + "\n",
        )
        if video_digest is not None and old_video.is_file():
            shutil.copy2(old_video, video_backup)
            with video_backup.open("rb") as stream:
                os.fsync(stream.fileno())
        for item, source, target in planned_images:
            os.replace(source, target)
            item["filename"] = str(target)
        if video_digest is not None:
            os.replace(staged_video, old_video)
        checkpoint.resolved_config["pending_media_promoted"] = True
        try:
            manager.save_checkpoint(checkpoint)
        except BaseException:
            # The in-memory state was not committed successfully. The caller
            # can still roll back from the durable staging manifest.
            checkpoint.resolved_config["pending_media_promoted"] = False
            raise

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
        path = str(config.config_path) if config.config_path is not None else None
        content = Path(path).read_text(encoding="utf-8") if path else ""
        return ConfigResult(values=config.to_dict(), path=path, content=content)

    def write_config(self, content: str) -> ConfigResult:
        active_config = load_config(verbose=False)
        path = active_config.config_path
        if path is None:
            raise FileNotFoundError("No StoryForge configuration file exists")

        candidate = Config()
        try:
            candidate.config.read_string(content)
        except ConfigParserError as error:
            raise ConfigError(f"Invalid configuration syntax: {error}") from error

        errors = candidate.validate_config()
        if errors:
            raise ConfigError("Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))

        atomic_write_text(path, content)
        return ConfigResult(values=candidate.to_dict(), path=str(path), content=content)

    def configure_models(self, backend: str, story_model: str, image_model: str = "") -> ConfigResult:
        if backend not in {"gemini", "openai", "anthropic"}:
            raise ValueError(f"Unsupported model backend: {backend}")

        active_config = load_config(verbose=False)
        path = active_config.config_path
        if path is None:
            path = Config().create_default_config()

        updates = {
            "backend": backend,
            f"{backend}_story_model": story_model,
        }
        if backend != "anthropic":
            updates[f"{backend}_image_model"] = image_model

        content = path.read_text(encoding="utf-8")
        return self.write_config(update_config_values(content, "system", updates))

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

    def write_world(self, content: str, overwrite: bool = False, expected_path: str | None = None) -> WorldResult:
        path = ContextManager()._discover_world_file() or resolve_world_file_path()
        if expected_path is not None and Path(expected_path).expanduser().resolve() != path.resolve():
            raise ValueError("The active world file changed; reload it before saving.")
        if path.exists() and not overwrite:
            raise FileExistsError(f"World file already exists: {path}")
        atomic_write_text(path, content)
        return WorldResult(path=str(path), exists=True, content=content)

    def list_models(self) -> dict[str, list[dict[str, Any]]]:
        from .model_cache import ModelCache

        cache = ModelCache()
        return {backend: cache.get(backend) or [] for backend in ("gemini", "openai", "anthropic")}

    def refresh_models(self) -> ModelRefreshResult:
        """Discover provider models now while preserving valid cache data on failures."""
        from .model_cache import ModelCache
        from .model_discovery import list_anthropic_models, list_gemini_models, list_openai_models
        from .model_ranking import model_supports_purpose

        providers = {
            "gemini": ("GEMINI_API_KEY", list_gemini_models),
            "openai": ("OPENAI_API_KEY", list_openai_models),
            "anthropic": ("ANTHROPIC_API_KEY", list_anthropic_models),
        }
        cache = ModelCache()
        models: dict[str, list[dict[str, Any]]] = {}
        statuses: dict[str, str] = {}

        for provider, (key_name, discover) in providers.items():
            previous = cache.get(provider) or []
            models[provider] = previous
            api_key = os.environ.get(key_name)
            if not api_key:
                statuses[provider] = f"skipped: {key_name} is not set"
                continue
            try:
                discovered = discover(api_key, raise_errors=True)
            except Exception as error:
                statuses[provider] = f"error: {error}"
                continue
            usable = [
                model
                for model in discovered
                if model_supports_purpose(model, provider, "text") or model_supports_purpose(model, provider, "image")
            ]
            if not usable:
                statuses[provider] = "empty: provider returned no usable models"
                continue
            if not cache.set(provider, usable):
                statuses[provider] = "error: could not write model cache"
                continue
            models[provider] = usable
            statuses[provider] = f"refreshed: {len(usable)} models"

        summary = "; ".join(f"{provider}: {status}" for provider, status in statuses.items())
        return ModelRefreshResult(models=models, statuses=statuses, message=f"Model refresh complete. {summary}")

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

        # Keep temporarily unavailable locations indexed (for example, a
        # removable drive) but omit them from the visible library until back.
        for stored_path in StoryForgeWorkflow._read_story_index():
            story_path = Path(stored_path)
            if story_path.is_file():
                paths.add(story_path.resolve())

        manager = CheckpointManager(auto_cleanup=False)
        for checkpoint_path in manager.checkpoint_dir.glob("checkpoint_*.yaml"):
            try:
                checkpoint = safe_load(checkpoint_path.read_text(encoding="utf-8")) or {}
                if not isinstance(checkpoint, dict):
                    continue
                resolved = checkpoint.get("resolved_config")
                generated = checkpoint.get("generated_content")
                if not isinstance(resolved, dict):
                    continue
                output_directory = resolved.get("output_directory")
                if output_directory:
                    story_path = Path(str(output_directory)).expanduser() / "story.txt"
                    if story_path.is_file():
                        paths.add(story_path.resolve())
                        context_file = generated.get("context_file") if isinstance(generated, dict) else None
                        if str(story_path.resolve()) not in StoryForgeWorkflow._read_story_index():
                            StoryForgeWorkflow._remember_story_path(story_path, context_file)
            except (AttributeError, OSError, TypeError, ValueError, YAMLError):
                continue
        return list(paths)

    @staticmethod
    def _generated_story_context_ids() -> dict[Path, str]:
        """Map generated story artifacts to their saved extension contexts."""
        context_ids: dict[Path, str] = {
            Path(story_path).resolve(): Path(context_path).stem
            for story_path, context_path in StoryForgeWorkflow._read_story_index().items()
            if context_path and Path(context_path).is_file()
        }
        manager = CheckpointManager(auto_cleanup=False)
        for checkpoint_path in manager.checkpoint_dir.glob("checkpoint_*.yaml"):
            try:
                checkpoint = safe_load(checkpoint_path.read_text(encoding="utf-8")) or {}
                if not isinstance(checkpoint, dict):
                    continue
                resolved = checkpoint.get("resolved_config")
                generated = checkpoint.get("generated_content")
                if not isinstance(resolved, dict) or not isinstance(generated, dict):
                    continue
                output_directory = resolved.get("output_directory")
                context_file = generated.get("context_file")
                if output_directory and context_file and Path(str(context_file)).expanduser().is_file():
                    story_path = (Path(str(output_directory)).expanduser() / "story.txt").resolve()
                    context_ids[story_path] = Path(str(context_file)).stem
            except (AttributeError, OSError, TypeError, ValueError, YAMLError):
                continue
        return context_ids

    @staticmethod
    def _story_index_path() -> Path | None:
        manager = CheckpointManager(auto_cleanup=False)
        checkpoint_dir = manager.checkpoint_dir
        return checkpoint_dir.parent / "generated_stories.sqlite3" if isinstance(checkpoint_dir, Path) else None

    @staticmethod
    def _read_legacy_story_index(path: Path) -> dict[str, str | None]:
        legacy_path = path.with_name("generated_stories.json")
        if not legacy_path.is_file():
            return {}
        try:
            data = json.loads(legacy_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(key): value for key, value in data.items() if value is None or isinstance(value, str)}
        except (OSError, ValueError):
            logging.getLogger(__name__).warning("Could not read legacy story index", exc_info=True)
        return {}

    @staticmethod
    def _read_story_index() -> dict[str, str | None]:
        path = StoryForgeWorkflow._story_index_path()
        if path is None:
            return {}
        records = StoryForgeWorkflow._read_legacy_story_index(path)
        if not path.is_file():
            return records
        try:
            with sqlite3.connect(path, timeout=30) as database:
                records.update(database.execute("SELECT story_path, context_path FROM generated_stories"))
        except sqlite3.DatabaseError:
            logging.getLogger(__name__).warning("Could not read generated-story index", exc_info=True)
        return records

    @staticmethod
    def _remember_story_path(story_path: Path, context_file: str | None = None) -> None:
        path = StoryForgeWorkflow._story_index_path()
        if path is None or not story_path.is_file():
            return
        key = str(story_path.resolve())
        value = str(Path(context_file).expanduser().resolve()) if context_file else None
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path, timeout=30) as database:
            # SQLite serializes the entire read/migrate/write transaction across
            # StoryForge processes, avoiding lost JSON read-modify-write updates.
            database.execute("BEGIN IMMEDIATE")
            database.execute(
                "CREATE TABLE IF NOT EXISTS generated_stories (story_path TEXT PRIMARY KEY, context_path TEXT)"
            )
            for old_path, old_context in StoryForgeWorkflow._read_legacy_story_index(path).items():
                database.execute(
                    "INSERT OR IGNORE INTO generated_stories (story_path, context_path) VALUES (?, ?)",
                    (old_path, old_context),
                )
            database.execute(
                "INSERT INTO generated_stories (story_path, context_path) VALUES (?, ?) "
                "ON CONFLICT(story_path) DO UPDATE SET context_path = excluded.context_path",
                (key, value),
            )

    @staticmethod
    def _remember_generated_story(checkpoint: CheckpointData) -> None:
        output_dir = checkpoint.resolved_config.get("output_directory")
        if output_dir:
            StoryForgeWorkflow._remember_story_path(
                Path(str(output_dir)).expanduser() / "story.txt",
                checkpoint.generated_content.get("context_file"),
            )

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
