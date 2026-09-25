"""Regression tests for MCP-backed workflow state transitions."""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from storyforge.checkpoint import CheckpointData, CheckpointManager, ExecutionPhase
from storyforge.config import Config, ConfigError
from storyforge.mcp_models import ExtensionRequest, FinalizeRequest, GenerationRequest
from storyforge.phase_executor import PhaseExecutor
from storyforge.workflow import POST_STORY_PHASES, StoryForgeWorkflow


def _checkpoint_with_story() -> CheckpointData:
    checkpoint = CheckpointData.create_new(
        "A test prompt",
        {},
        {"output_directory": "output", "verbose": False, "debug": False},
    )
    checkpoint.generated_content["story"] = "A saved draft"
    return checkpoint


def test_resume_completed_session_reopens_post_story_phases():
    checkpoint = _checkpoint_with_story()
    checkpoint.status = "completed"
    checkpoint.current_phase = ExecutionPhase.COMPLETED.value
    checkpoint.completed_phases = [phase.value for phase in ExecutionPhase]
    manager = MagicMock()
    workflow = StoryForgeWorkflow()

    with (
        patch("storyforge.workflow.CheckpointManager", return_value=manager),
        patch.object(workflow, "_find_checkpoint", return_value=checkpoint),
    ):
        result = workflow.resume_session(checkpoint.session_id)

    assert result.status == "active"
    assert checkpoint.current_phase == ExecutionPhase.STORY_SAVE.value
    assert POST_STORY_PHASES.isdisjoint(checkpoint.completed_phases)
    manager.save_checkpoint.assert_called_once_with(checkpoint)


def test_refine_completed_session_requires_resume():
    checkpoint = _checkpoint_with_story()
    checkpoint.status = "completed"
    checkpoint.current_phase = ExecutionPhase.COMPLETED.value
    workflow = StoryForgeWorkflow()
    with (
        patch("storyforge.workflow.CheckpointManager"),
        patch.object(workflow, "_find_checkpoint", return_value=checkpoint),
        patch.object(workflow, "_executor") as executor,
        pytest.raises(ValueError, match="resume the session"),
    ):
        workflow.refine_draft(checkpoint.session_id, "Change the ending")
    executor.assert_not_called()


@pytest.mark.parametrize("linked_context", [False, True])
def test_finalized_revision_discards_old_artifacts_without_breaking_chains(tmp_path, monkeypatch, linked_context):
    output = tmp_path / "output"
    output.mkdir()
    story = output / "story.txt"
    image = output / "illustration_01.png"
    unrelated = output / "personal.png"
    video = output / "video_prompt.txt"
    for path in (story, image, unrelated, video):
        path.write_text("old", encoding="utf-8")
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    monkeypatch.setenv("STORYFORGE_TEST_CONTEXT_DIR", str(context_dir))
    context = context_dir / "original.md"
    context.write_text("# Original", encoding="utf-8")
    if linked_context:
        (context_dir / "continuation.md").write_text(
            "# Continuation\n\n**Extended From:** original\n", encoding="utf-8"
        )

    checkpoint = _checkpoint_with_story()
    checkpoint.status = "completed"
    checkpoint.current_phase = ExecutionPhase.COMPLETED.value
    checkpoint.completed_phases = [phase.value for phase in ExecutionPhase]
    checkpoint.resolved_config["output_directory"] = str(output)
    checkpoint.generated_content["generated_images"] = [{"filename": str(image)}]
    checkpoint.generated_content["context_file"] = str(context)
    manager = MagicMock()
    executor = MagicMock()
    executor.execute_existing_session.return_value = checkpoint
    workflow = StoryForgeWorkflow()
    with (
        patch("storyforge.workflow.CheckpointManager", return_value=manager),
        patch.object(workflow, "_find_checkpoint", return_value=checkpoint),
        patch.object(workflow, "_executor", return_value=executor),
    ):
        workflow.resume_session(checkpoint.session_id)
        assert image.exists() and video.exists() and context.exists()
        result = workflow.finalize_story(FinalizeRequest(session_id=checkpoint.session_id))

    assert result.artifacts == [str(story)]
    assert not image.exists() and not video.exists()
    assert unrelated.exists()
    assert context.exists() is linked_context
    assert "context_file" not in checkpoint.generated_content
    assert "superseded_artifacts" not in checkpoint.resolved_config


def test_failed_replacement_preserves_previous_media(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    old_image = output / "old_01.png"
    old_video = output / "video_prompt.txt"
    old_image.write_bytes(b"original image")
    old_video.write_text("original video", encoding="utf-8")
    checkpoint = _checkpoint_with_story()
    checkpoint.resolved_config.update(
        {
            "output_directory": str(output),
            "superseded_artifacts": {
                "images": [{"filename": str(old_image)}],
                "video_prompt": str(old_video),
                "context_file": None,
            },
        }
    )
    manager = MagicMock()
    executor = MagicMock()
    executor.execute_existing_session.return_value = checkpoint
    workflow = StoryForgeWorkflow()
    with (
        patch("storyforge.workflow.CheckpointManager", return_value=manager),
        patch.object(workflow, "_find_checkpoint", return_value=checkpoint),
        patch.object(workflow, "_executor", return_value=executor),
        pytest.raises(RuntimeError, match="previous media was preserved"),
    ):
        workflow.finalize_story(FinalizeRequest(session_id=checkpoint.session_id, image_count=1))

    assert old_image.read_bytes() == b"original image"
    assert old_video.read_text(encoding="utf-8") == "original video"
    assert list(output.glob(".storyforge-finalize-*")) == []


def test_successful_replacement_promotes_media_without_overwriting_unrelated_files(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    old_image = output / "drawing_01.png"
    old_image.write_bytes(b"old")
    unrelated = output / "personal.png"
    unrelated.write_bytes(b"personal")
    checkpoint = _checkpoint_with_story()
    checkpoint.resolved_config.update(
        {
            "output_directory": str(output),
            "superseded_artifacts": {
                "images": [{"filename": str(old_image)}],
                "video_prompt": None,
                "context_file": None,
            },
        }
    )
    manager = MagicMock()
    executor = MagicMock()

    def generate(current, _phase):
        assert current.resolved_config["output_directory"] == str(output)
        staged_image = executor.media_output_directory / "drawing_01.png"
        staged_image.write_bytes(b"new")
        current.generated_content["generated_images"] = [{"filename": str(staged_image)}]
        return current

    executor.execute_existing_session.side_effect = generate
    workflow = StoryForgeWorkflow()
    with (
        patch("storyforge.workflow.CheckpointManager", return_value=manager),
        patch.object(workflow, "_find_checkpoint", return_value=checkpoint),
        patch.object(workflow, "_executor", return_value=executor),
    ):
        workflow.finalize_story(FinalizeRequest(session_id=checkpoint.session_id, image_count=1))

    promoted = Path(checkpoint.generated_content["generated_images"][0]["filename"])
    assert promoted.read_bytes() == b"new"
    assert not old_image.exists()
    assert unrelated.read_bytes() == b"personal"


@pytest.mark.parametrize("failure", [OSError("commit failed"), KeyboardInterrupt()])
def test_replacement_rolls_back_when_checkpoint_commit_fails(tmp_path, failure):
    output = tmp_path / "output"
    output.mkdir()
    old_image = output / "drawing_01.png"
    old_video = output / "video_prompt.txt"
    old_image.write_bytes(b"old image")
    old_video.write_text("old video", encoding="utf-8")
    checkpoint = _checkpoint_with_story()
    checkpoint.resolved_config.update(
        {
            "output_directory": str(output),
            "superseded_artifacts": {
                "images": [{"filename": str(old_image)}],
                "video_prompt": str(old_video),
                "context_file": None,
            },
        }
    )
    manager = MagicMock()
    manager.save_checkpoint.side_effect = [None, failure, None]
    executor = MagicMock()

    def generate(current, _phase):
        assert current.resolved_config["output_directory"] == str(output)
        stage = executor.media_output_directory
        image = stage / "drawing_01.png"
        image.write_bytes(b"new image")
        (stage / "video_prompt.txt").write_text("new video", encoding="utf-8")
        current.generated_content["generated_images"] = [{"filename": str(image)}]
        return current

    executor.execute_existing_session.side_effect = generate
    workflow = StoryForgeWorkflow()
    with (
        patch("storyforge.workflow.CheckpointManager", return_value=manager),
        patch.object(workflow, "_find_checkpoint", return_value=checkpoint),
        patch.object(workflow, "_executor", return_value=executor),
        pytest.raises(type(failure)),
    ):
        workflow.finalize_story(FinalizeRequest(session_id=checkpoint.session_id, image_count=1, video_scene_count=1))

    assert old_image.read_bytes() == b"old image"
    assert old_video.read_text(encoding="utf-8") == "old video"
    assert list(output.glob(".storyforge-finalize-*")) == []


def test_interrupted_replacement_keeps_canonical_checkpoint_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    old_image = output / "old.png"
    old_image.write_bytes(b"original")
    checkpoint = _checkpoint_with_story()
    checkpoint.resolved_config.update(
        {
            "output_directory": "output",
            "superseded_artifacts": {
                "images": [{"filename": str(old_image)}],
                "video_prompt": None,
                "context_file": None,
            },
        }
    )
    with patch("storyforge.checkpoint.user_data_dir", return_value=str(tmp_path / "data")):
        manager = CheckpointManager(auto_cleanup=False)
    manager.save_checkpoint(checkpoint)
    real_save = manager.save_checkpoint
    saved_outputs = []

    def record_save(current):
        saved_outputs.append(current.resolved_config["output_directory"])
        return real_save(current)

    def interrupt_at_image_decision(_executor, phase):
        if phase is ExecutionPhase.IMAGE_DECISION:
            raise KeyboardInterrupt

    with (
        patch("storyforge.workflow.CheckpointManager", return_value=manager),
        patch.object(manager, "save_checkpoint", side_effect=record_save),
        patch.object(PhaseExecutor, "_execute_phase", autospec=True, side_effect=interrupt_at_image_decision),
        pytest.raises(KeyboardInterrupt),
    ):
        StoryForgeWorkflow().finalize_story(FinalizeRequest(session_id=checkpoint.session_id))

    persisted = manager.load_checkpoint(manager.checkpoint_dir / f"checkpoint_{checkpoint.session_id}.yaml")
    assert saved_outputs and set(saved_outputs) == {str(output)}
    assert persisted.resolved_config["output_directory"] == str(output)
    assert "pending_media_stage" not in persisted.resolved_config
    assert persisted.current_phase == ExecutionPhase.STORY_SAVE.value
    assert old_image.read_bytes() == b"original"
    assert list(output.glob(".storyforge-finalize-*")) == []


def test_resume_rolls_back_partial_promotion_after_process_exit(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    old_image = output / "old.png"
    old_image.write_bytes(b"old image")
    old_video = output / "video_prompt.txt"
    old_video.write_bytes(b"new video")
    promoted = output / "new_123.png"
    promoted.write_bytes(b"new image")
    stage = output / ".storyforge-finalize-crashed"
    stage.mkdir()
    (stage / "previous_video_prompt.txt").write_bytes(b"old video")
    (stage / "promotion_manifest.json").write_text(
        json.dumps(
            {
                "images": [{"name": promoted.name, "sha256": hashlib.sha256(b"new image").hexdigest()}],
                "video_sha256": hashlib.sha256(b"new video").hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    checkpoint = _checkpoint_with_story()
    checkpoint.completed_phases = [ExecutionPhase.STORY_SAVE.value]
    checkpoint.resolved_config.update(
        {
            "output_directory": str(output),
            "pending_media_stage": str(stage),
            "pending_media_promoted": False,
            "superseded_artifacts": {
                "images": [{"filename": str(old_image)}],
                "video_prompt": str(old_video),
                "context_file": None,
            },
        }
    )
    checkpoint.generated_content["generated_images"] = [{"filename": str(stage / "new.png")}]
    with patch("storyforge.checkpoint.user_data_dir", return_value=str(tmp_path / "data")):
        manager = CheckpointManager(auto_cleanup=False)
    manager.save_checkpoint(checkpoint)

    with patch("storyforge.workflow.CheckpointManager", return_value=manager):
        result = StoryForgeWorkflow().resume_session(checkpoint.session_id)

    assert result.output_directory == str(output)
    assert old_image.read_bytes() == b"old image"
    assert old_video.read_bytes() == b"old video"
    assert not promoted.exists() and not stage.exists()
    assert (
        "pending_media_stage"
        not in manager.load_checkpoint(
            manager.checkpoint_dir / f"checkpoint_{checkpoint.session_id}.yaml"
        ).resolved_config
    )


def test_resume_finishes_promoted_replacement_after_process_exit(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    old_image = output / "old.png"
    old_image.write_bytes(b"old")
    new_image = output / "new_123.png"
    new_image.write_bytes(b"new")
    stage = output / ".storyforge-finalize-committed"
    stage.mkdir()
    checkpoint = _checkpoint_with_story()
    checkpoint.completed_phases = [ExecutionPhase.STORY_SAVE.value]
    checkpoint.resolved_config.update(
        {
            "output_directory": str(output),
            "pending_media_stage": str(stage),
            "pending_media_promoted": True,
            "video_scene_count": 0,
            "superseded_artifacts": {
                "images": [{"filename": str(old_image)}],
                "video_prompt": None,
                "context_file": None,
            },
        }
    )
    checkpoint.generated_content["generated_images"] = [{"filename": str(new_image)}]
    with patch("storyforge.checkpoint.user_data_dir", return_value=str(tmp_path / "data")):
        manager = CheckpointManager(auto_cleanup=False)
    manager.save_checkpoint(checkpoint)

    with patch("storyforge.workflow.CheckpointManager", return_value=manager):
        StoryForgeWorkflow()._recover_pending_replacement(checkpoint, manager)

    assert not old_image.exists() and new_image.read_bytes() == b"new"
    assert not stage.exists()
    assert (
        "pending_media_stage"
        not in manager.load_checkpoint(
            manager.checkpoint_dir / f"checkpoint_{checkpoint.session_id}.yaml"
        ).resolved_config
    )


def test_resume_failed_story_save_retries_the_failed_phase():
    checkpoint = _checkpoint_with_story()
    checkpoint.status = "failed"
    checkpoint.current_phase = ExecutionPhase.STORY_SAVE.value
    checkpoint.completed_phases = [ExecutionPhase.STORY_GENERATE.value]
    manager = MagicMock()
    executor = MagicMock()

    def execute_existing(current, start_phase, *, stop_after):
        current.completed_phases.append(ExecutionPhase.STORY_SAVE.value)
        return current

    executor.execute_existing_session.side_effect = execute_existing
    workflow = StoryForgeWorkflow()
    with (
        patch("storyforge.workflow.CheckpointManager", return_value=manager),
        patch.object(workflow, "_find_checkpoint", return_value=checkpoint),
        patch.object(workflow, "_executor", return_value=executor),
    ):
        workflow.resume_session(checkpoint.session_id)

    executor.execute_existing_session.assert_called_once_with(
        checkpoint,
        ExecutionPhase.STORY_SAVE,
        stop_after=ExecutionPhase.STORY_SAVE,
    )


def test_create_draft_uses_configured_output_directory():
    config = MagicMock()
    configured_values = {
        ("story", "length"): "short",
        ("story", "age_range"): "preschool",
        ("story", "style"): "fantasy",
        ("story", "tone"): "gentle",
        ("story", "voice"): "",
        ("story", "theme"): "kindness",
        ("story", "learning_focus"): "",
        ("story", "setting"): "",
        ("story", "characters"): [],
        ("images", "image_style"): "watercolor",
        ("images", "image_count"): 3,
        ("output", "output_dir"): "configured-output",
        ("output", "use_context"): True,
        ("output", "world_file"): "",
        ("system", "backend"): "",
        ("system", "verbose"): False,
        ("system", "debug"): False,
    }
    config.get_field_value.side_effect = lambda section, field: configured_values[(section, field)]
    executor = MagicMock()

    def execute_new(prompt, cli_arguments, resolved_config, **_kwargs):
        checkpoint = CheckpointData.create_new(prompt, cli_arguments, resolved_config)
        checkpoint.generated_content["story"] = "Draft"
        checkpoint.current_phase = ExecutionPhase.STORY_SAVE.value
        return checkpoint

    executor.execute_new_session.side_effect = execute_new
    workflow = StoryForgeWorkflow()
    with (
        patch("storyforge.workflow.load_config", return_value=config),
        patch("storyforge.workflow.CheckpointManager"),
        patch.object(workflow, "_executor", return_value=executor),
    ):
        result = workflow.create_draft(GenerationRequest(prompt="A configured story"))

    assert result.output_directory == "configured-output"


@pytest.mark.parametrize(
    ("requested", "configured", "expected"),
    [(None, True, True), (False, True, False), (True, False, True)],
)
def test_create_draft_resolves_debug_and_verbose_from_request_or_config(requested, configured, expected):
    config = MagicMock()
    config.get_field_value.side_effect = lambda section, field: (
        configured if (section, field) in {("system", "debug"), ("system", "verbose")} else None
    )
    executor = MagicMock()

    def execute_new(prompt, cli_arguments, resolved_config, **_kwargs):
        checkpoint = CheckpointData.create_new(prompt, cli_arguments, resolved_config)
        checkpoint.generated_content["story"] = "Draft"
        return checkpoint

    executor.execute_new_session.side_effect = execute_new
    workflow = StoryForgeWorkflow()
    with (
        patch("storyforge.workflow.load_config", return_value=config),
        patch("storyforge.workflow.CheckpointManager"),
        patch.object(workflow, "_executor", return_value=executor),
    ):
        workflow.create_draft(GenerationRequest(prompt="A story", debug=requested, verbose=requested))

    resolved = executor.execute_new_session.call_args.args[2]
    assert resolved["debug"] is expected
    assert resolved["verbose"] is expected


def test_create_debug_draft_does_not_initialize_provider(tmp_path, monkeypatch):
    config = Config()
    output_dir = tmp_path / "debug-output"
    data_dir = tmp_path / "data"
    monkeypatch.chdir(tmp_path)

    with (
        patch("storyforge.workflow.load_config", return_value=config),
        patch("storyforge.phase_executor.load_config", return_value=config),
        patch("storyforge.checkpoint.user_data_dir", return_value=str(data_dir)),
        patch("storyforge.phase_executor.get_backend", side_effect=AssertionError("provider initialized")) as backend,
    ):
        result = StoryForgeWorkflow().create_draft(
            GenerationRequest(
                prompt="An offline test",
                debug=True,
                use_context=False,
                output_dir=str(output_dir),
            )
        )

    assert result.story.startswith("Ethan and Isaac")
    assert (output_dir / "story.txt").is_file()
    backend.assert_not_called()


def test_create_extension_draft_preserves_saved_story_parameters():
    metadata = {
        "filepath": "/stories/story1.md",
        "filename": "story1",
        "characters": "Wizard, Dragon",
        "theme": "courage",
        "tone": "silly",
        "style": "fantasy",
        "voice": "lyrical",
        "age_group": "early_reader",
        "art_style": "watercolor",
        "length": "bedtime",
        "learning_focus": "counting",
        "setting": "enchanted forest",
    }
    context_manager = MagicMock()
    context_manager.list_available_contexts.return_value = [metadata]
    context_manager.load_chain_for_extension.return_value = ("Full story chain", metadata)
    config = MagicMock()
    config.get_field_value.side_effect = lambda section, field: (
        True
        if (section, field) in {("system", "debug"), ("system", "verbose")}
        else 5
        if (section, field) == ("images", "image_count")
        else None
    )
    executor = MagicMock()

    def execute_new(prompt, cli_arguments, resolved_config, *, prompt_obj, stop_after):
        assert cli_arguments["debug"] is True
        assert cli_arguments["verbose"] is True
        assert prompt_obj.characters == ["Wizard", "Dragon"]
        assert prompt_obj.setting == "enchanted forest"
        assert prompt_obj.learning_focus == "counting"
        assert prompt_obj.continuation_mode is True
        checkpoint = CheckpointData.create_new(prompt, cli_arguments, resolved_config)
        checkpoint.generated_content["story"] = "Continuation"
        checkpoint.current_phase = stop_after.value
        return checkpoint

    executor.execute_new_session.side_effect = execute_new
    workflow = StoryForgeWorkflow()
    with (
        patch("storyforge.workflow.ContextManager", return_value=context_manager),
        patch("storyforge.workflow.load_config", return_value=config),
        patch("storyforge.workflow.CheckpointManager"),
        patch.object(workflow, "_executor", return_value=executor),
    ):
        result = workflow.create_extension_draft(
            ExtensionRequest(story_id="story1", ending_type="wrap_up", direction="Bring everyone home")
        )

    assert result.story == "Continuation"
    assert result.metadata["image_count"] == 5


def test_init_config_create_overwrite_contract(tmp_path):
    target = tmp_path / "nested" / "storyforge.ini"
    workflow = StoryForgeWorkflow()

    created = workflow.init_config(str(target))
    with pytest.raises(FileExistsError):
        workflow.init_config(str(target))
    overwritten = workflow.init_config(str(target), overwrite=True)

    assert created.artifacts == [str(target)]
    assert overwritten.artifacts == [str(target)]
    assert target.exists()


def test_get_and_write_config_content(tmp_path, monkeypatch):
    target = tmp_path / "storyforge.ini"
    target.write_text("[story]\nlength = short\n", encoding="utf-8")
    monkeypatch.setenv("STORYFORGE_CONFIG", str(target))
    workflow = StoryForgeWorkflow()

    loaded = workflow.get_config()
    saved = workflow.write_config("[story]\nlength = bedtime\n")

    assert loaded.content == "[story]\nlength = short\n"
    assert saved.path == str(target)
    assert saved.values["story"]["length"] == "bedtime"
    assert target.read_text(encoding="utf-8") == "[story]\nlength = bedtime\n"


def test_write_config_rejects_invalid_content_without_overwriting(tmp_path, monkeypatch):
    target = tmp_path / "storyforge.ini"
    original = "[story]\nlength = short\n"
    target.write_text(original, encoding="utf-8")
    monkeypatch.setenv("STORYFORGE_CONFIG", str(target))

    with pytest.raises(ConfigError, match="Invalid configuration syntax"):
        StoryForgeWorkflow().write_config("[story\nlength = long\n")

    assert target.read_text(encoding="utf-8") == original


def test_write_config_rejects_schema_error_without_overwriting(tmp_path, monkeypatch):
    target = tmp_path / "storyforge.ini"
    original = "[story]\nlength = short\n"
    target.write_text(original, encoding="utf-8")
    monkeypatch.setenv("STORYFORGE_CONFIG", str(target))

    with pytest.raises(ConfigError, match="Configuration validation failed"):
        StoryForgeWorkflow().write_config("[story]\nlength = enormous\n")

    assert target.read_text(encoding="utf-8") == original


def test_write_config_preserves_original_when_atomic_replace_fails(tmp_path, monkeypatch):
    target = tmp_path / "storyforge.ini"
    original = "[story]\nlength = short\n"
    target.write_text(original, encoding="utf-8")
    monkeypatch.setenv("STORYFORGE_CONFIG", str(target))

    with (
        patch("storyforge.atomic_io.os.replace", side_effect=OSError("replace failed")),
        pytest.raises(OSError, match="replace failed"),
    ):
        StoryForgeWorkflow().write_config("[story]\nlength = bedtime\n")

    assert target.read_text(encoding="utf-8") == original


def test_configure_models_updates_provider_fields_and_preserves_config(tmp_path, monkeypatch):
    target = tmp_path / "storyforge.ini"
    target.write_text(
        "# User comment\n[story]\nlength = short\n\n[system]\nbackend = gemini\nverbose = true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STORYFORGE_CONFIG", str(target))

    result = StoryForgeWorkflow().configure_models("openai", "gpt-5.5", "gpt-image-1.5")

    content = target.read_text(encoding="utf-8")
    assert result.values["system"]["backend"] == "openai"
    assert "# User comment" in content
    assert "length = short" in content
    assert "backend = openai" in content
    assert "openai_story_model = gpt-5.5" in content
    assert "openai_image_model = gpt-image-1.5" in content


def test_configure_models_creates_config_when_missing(tmp_path):
    target = tmp_path / "storyforge.ini"

    with (
        patch("storyforge.config.Config.get_config_paths", return_value=[target]),
        patch("storyforge.config.Config.get_default_config_path", return_value=target),
    ):
        result = StoryForgeWorkflow().configure_models("anthropic", "claude-sonnet-4-6")

    assert result.path == str(target)
    assert result.values["system"]["backend"] == "anthropic"
    assert result.values["system"]["anthropic_story_model"] == "claude-sonnet-4-6"


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
def test_refresh_models_repopulates_available_provider_and_reports_skips():
    cache = MagicMock()
    cache.get.return_value = None
    cache.set.return_value = True
    discovered = [{"name": "gpt-6.0"}, {"name": "gpt-image-2"}]

    with (
        patch("storyforge.model_cache.ModelCache", return_value=cache),
        patch("storyforge.model_discovery.list_openai_models", return_value=discovered) as list_openai,
        patch("storyforge.model_discovery.list_gemini_models") as list_gemini,
        patch("storyforge.model_discovery.list_anthropic_models") as list_anthropic,
    ):
        result = StoryForgeWorkflow().refresh_models()

    assert result.models["openai"] == discovered
    assert result.statuses["openai"] == "refreshed: 2 models"
    assert result.statuses["gemini"] == "skipped: GEMINI_API_KEY is not set"
    assert result.statuses["anthropic"] == "skipped: ANTHROPIC_API_KEY is not set"
    list_openai.assert_called_once_with("test-key", raise_errors=True)
    list_gemini.assert_not_called()
    list_anthropic.assert_not_called()
    cache.set.assert_called_once_with("openai", discovered)


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
def test_refresh_models_preserves_valid_cache_when_provider_fails():
    previous = [{"name": "gpt-5.5"}]
    cache = MagicMock()
    cache.get.side_effect = lambda provider: previous if provider == "openai" else None

    with (
        patch("storyforge.model_cache.ModelCache", return_value=cache),
        patch(
            "storyforge.model_discovery.list_openai_models",
            side_effect=RuntimeError("provider unavailable"),
        ),
    ):
        result = StoryForgeWorkflow().refresh_models()

    assert result.models["openai"] == previous
    assert result.statuses["openai"] == "error: provider unavailable"
    cache.set.assert_not_called()


def test_list_models_only_returns_fresh_cache_entries():
    cache = MagicMock()
    cache.get.side_effect = lambda provider: [{"name": "fresh"}] if provider == "gemini" else None

    with patch("storyforge.model_cache.ModelCache", return_value=cache):
        result = StoryForgeWorkflow().list_models()

    assert result == {"gemini": [{"name": "fresh"}], "openai": [], "anthropic": []}


def test_generated_story_library_discovers_text_and_images(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "storyforge_output_20260814_120000"
    output.mkdir()
    story_path = output / "story.txt"
    story_path.write_text("Story: The Lantern Fox\n\nA fox carried a lantern home.", encoding="utf-8")
    video_prompt_path = output / "video_prompt.txt"
    video_prompt_path.write_text("A lantern glows in a moonlit forest.", encoding="utf-8")
    (output / "lantern_01.png").write_bytes(b"image")
    (output / "notes.md").write_text("not an image", encoding="utf-8")
    os.utime(story_path, (1_700_000_000, 1_700_000_000))

    checkpoint_manager = MagicMock()
    checkpoint_manager.checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_manager.checkpoint_dir.mkdir()
    with patch("storyforge.workflow.CheckpointManager", return_value=checkpoint_manager):
        workflow = StoryForgeWorkflow()
        stories = workflow.list_generated_stories()
        detail = workflow.get_generated_story(stories[0].id)
        video_prompt_path.unlink()
        detail_without_prompt = workflow.get_generated_story(stories[0].id)

    assert len(stories) == 1
    assert stories[0].title == "The Lantern Fox"
    assert stories[0].preview == "A fox carried a lantern home."
    assert stories[0].image_count == 1
    assert detail.content.startswith("Story: The Lantern Fox")
    assert detail.output_directory == str(output.resolve())
    assert detail.video_prompt_content == "A lantern glows in a moonlit forest."
    assert detail.image_paths == [str((output / "lantern_01.png").resolve())]
    assert detail_without_prompt.video_prompt_content is None


def test_generated_story_library_links_saved_extension_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "storyforge_output_test"
    output.mkdir()
    story_path = output / "story.txt"
    story_path.write_text("Story: Linked Story\n\nOnce upon a test.", encoding="utf-8")
    context_path = tmp_path / "linked_story.md"
    context_path.write_text("# Story Context: Linked Story", encoding="utf-8")

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "checkpoint_linked.yaml").write_text(
        f"""resolved_config:\n  output_directory: {output}\ngenerated_content:\n  context_file: {context_path}\n""",
        encoding="utf-8",
    )
    checkpoint_manager = MagicMock()
    checkpoint_manager.checkpoint_dir = checkpoint_dir

    with patch("storyforge.workflow.CheckpointManager", return_value=checkpoint_manager):
        story = StoryForgeWorkflow().list_generated_stories()[0]

    assert story.context_id == "linked_story"


def test_custom_output_and_extend_link_survive_checkpoint_pruning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "custom" / "nested" / "book"
    output.mkdir(parents=True)
    story_path = output / "story.txt"
    story_path.write_text("Story: A Lasting Story\n\nOnce upon a time.", encoding="utf-8")
    context_path = tmp_path / "linked.md"
    context_path.write_text("# Story Context", encoding="utf-8")
    checkpoint_dir = tmp_path / "data" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "checkpoint_old.yaml"
    checkpoint.write_text(
        f"resolved_config:\n  output_directory: {output}\ngenerated_content:\n  context_file: {context_path}\n",
        encoding="utf-8",
    )
    manager = MagicMock()
    manager.checkpoint_dir = checkpoint_dir
    with patch("storyforge.workflow.CheckpointManager", return_value=manager):
        first = StoryForgeWorkflow().list_generated_stories()
        checkpoint.unlink()
        second = StoryForgeWorkflow().list_generated_stories()
        story_path.unlink()
        unavailable = StoryForgeWorkflow().list_generated_stories()
        story_path.write_text("Story: A Lasting Story\n\nOnce upon a time.", encoding="utf-8")
        restored = StoryForgeWorkflow().list_generated_stories()

    assert first[0].context_id == "linked"
    assert len(second) == 1
    assert second[0].story_path == str(story_path.resolve())
    assert second[0].context_id == "linked"
    assert unavailable == []
    assert restored[0].context_id == "linked"


def test_story_index_preserves_concurrent_additions_and_context_updates(tmp_path):
    index_path = tmp_path / "data" / "generated_stories.sqlite3"
    stories = []
    contexts = []
    for number in range(12):
        output = tmp_path / f"book-{number}"
        output.mkdir()
        story = output / "story.txt"
        story.write_text(f"Story: Book {number}", encoding="utf-8")
        context = tmp_path / f"book-{number}.md"
        context.write_text(f"# Book {number}", encoding="utf-8")
        stories.append(story)
        contexts.append(context)

    with patch.object(StoryForgeWorkflow, "_story_index_path", return_value=index_path):
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(StoryForgeWorkflow._remember_story_path, stories))
            list(
                pool.map(
                    lambda pair: StoryForgeWorkflow._remember_story_path(*pair), zip(stories, contexts, strict=True)
                )
            )
        records = StoryForgeWorkflow._read_story_index()

    assert len(records) == len(stories)
    assert records == {
        str(story.resolve()): str(context.resolve()) for story, context in zip(stories, contexts, strict=True)
    }


def test_story_index_migrates_existing_json_records(tmp_path):
    index_path = tmp_path / "data" / "generated_stories.sqlite3"
    index_path.parent.mkdir()
    old_story = tmp_path / "old" / "story.txt"
    old_story.parent.mkdir()
    old_story.write_text("Story: Old", encoding="utf-8")
    (index_path.parent / "generated_stories.json").write_text(
        json.dumps({str(old_story.resolve()): None}), encoding="utf-8"
    )
    new_story = tmp_path / "new" / "story.txt"
    new_story.parent.mkdir()
    new_story.write_text("Story: New", encoding="utf-8")

    with patch.object(StoryForgeWorkflow, "_story_index_path", return_value=index_path):
        StoryForgeWorkflow._remember_story_path(new_story)
        records = StoryForgeWorkflow._read_story_index()

    assert set(records) == {str(old_story.resolve()), str(new_story.resolve())}


@pytest.mark.parametrize("malformed", ["- invalid checkpoint list\n", "resolved_config: invalid\n"])
def test_malformed_legacy_checkpoint_does_not_block_new_draft(tmp_path, monkeypatch, malformed):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STORYFORGE_TEST_CONTEXT_DIR", str(tmp_path / "context"))
    checkpoint_dir = tmp_path / "data" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "checkpoint_bad.yaml").write_text(malformed, encoding="utf-8")
    config = Config()
    with (
        patch("storyforge.checkpoint.user_data_dir", return_value=str(tmp_path / "data")),
        patch("storyforge.workflow.load_config", return_value=config),
        patch("storyforge.phase_executor.load_config", return_value=config),
        patch("storyforge.phase_executor.get_backend", side_effect=AssertionError("provider initialized")),
    ):
        draft = StoryForgeWorkflow().create_draft(
            GenerationRequest(
                prompt="An offline story", debug=True, use_context=False, output_dir=str(tmp_path / "book")
            )
        )

    assert draft.story.startswith("Ethan and Isaac")


def test_malformed_legacy_checkpoint_does_not_block_extension(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checkpoint_dir = tmp_path / "data" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "checkpoint_bad.yaml").write_text("- invalid checkpoint list\n", encoding="utf-8")
    selected = {"filepath": str(tmp_path / "saved.md"), "filename": "saved"}
    context_manager = MagicMock()
    context_manager.list_available_contexts.return_value = [selected]
    context_manager.load_chain_for_extension.return_value = ("Original story", {})
    executor = MagicMock()

    def create_extension(prompt, cli_arguments, resolved_config, **_kwargs):
        checkpoint = CheckpointData.create_new(prompt, cli_arguments, resolved_config)
        checkpoint.generated_content["story"] = "Continuation"
        return checkpoint

    executor.execute_new_session.side_effect = create_extension
    with (
        patch("storyforge.checkpoint.user_data_dir", return_value=str(tmp_path / "data")),
        patch("storyforge.workflow.ContextManager", return_value=context_manager),
        patch("storyforge.workflow.load_config", return_value=Config()),
        patch.object(StoryForgeWorkflow, "_executor", return_value=executor),
    ):
        draft = StoryForgeWorkflow().create_extension_draft(ExtensionRequest(story_id="saved"))

    assert draft.story == "Continuation"
