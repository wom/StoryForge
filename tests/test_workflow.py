"""Regression tests for MCP-backed workflow state transitions."""

import os
from unittest.mock import MagicMock, patch

import pytest

from storyforge.checkpoint import CheckpointData, ExecutionPhase
from storyforge.config import Config, ConfigError
from storyforge.mcp_models import ExtensionRequest, FinalizeRequest, GenerationRequest
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
        True if (section, field) in {("system", "debug"), ("system", "verbose")} else None
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
