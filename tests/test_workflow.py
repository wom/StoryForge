"""Regression tests for MCP-backed workflow state transitions."""

from unittest.mock import MagicMock, patch

import pytest

from storyforge.checkpoint import CheckpointData, ExecutionPhase
from storyforge.mcp_models import ExtensionRequest, GenerationRequest
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
    config.get_field_value.return_value = None
    executor = MagicMock()

    def execute_new(prompt, cli_arguments, resolved_config, *, prompt_obj, stop_after):
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
