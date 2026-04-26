"""
Tests for the sf extend command and related functionality.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from storyforge.context import ContextManager
from storyforge.prompt import Prompt


class TestContextManagerExtension:
    """Test ContextManager extension methods."""

    def test_list_available_contexts_empty(self, tmp_path):
        """Test listing context files when directory is empty."""
        mgr = ContextManager()
        with patch.object(mgr, "get_context_directory", return_value=tmp_path):
            contexts = mgr.list_available_contexts()

        assert len(contexts) == 0

    def test_list_available_contexts(self, tmp_path):
        """Test listing context files."""
        # Create mock context files
        context_dir = tmp_path / "context"
        context_dir.mkdir()

        ctx1 = context_dir / "story1_20251022_123456.md"
        ctx1.write_text(
            """# Story Context: Test Story 1
**Generated:** 2025-10-22 12:34:56
**Characters:** Alice, Bob
**Theme:** adventure

## Story
Once upon a time..."""
        )

        ctx2 = context_dir / "story2_20251022_234567.md"
        ctx2.write_text(
            """# Story Context: Test Story 2
**Generated:** 2025-10-22 23:45:67
**Characters:** Charlie
**Theme:** friendship

## Story
In a magical forest..."""
        )

        mgr = ContextManager()
        with patch.object(mgr, "get_context_directory", return_value=context_dir):
            contexts = mgr.list_available_contexts()

        assert len(contexts) == 2
        # Should be sorted by modification time (reverse)
        filenames = [contexts[0]["filename"], contexts[1]["filename"]]
        assert "story2_20251022_234567" in filenames
        assert "story1_20251022_123456" in filenames

    def test_parse_context_metadata(self, tmp_path):
        """Test parsing context file metadata."""
        ctx_file = tmp_path / "test.md"
        ctx_file.write_text(
            """# Story Context: Dragons
**Generated:** 2025-10-22 19:55:22
**Characters:** Spike, Flutter
**Theme:** adventure
**Age Group:** early_reader
**Tone:** exciting
**Art Style:** cartoon

## Story
Story content here..."""
        )

        mgr = ContextManager()
        metadata = mgr.parse_context_metadata(ctx_file)

        assert metadata["filename"] == "test"
        assert metadata["timestamp"] == "2025-10-22 19:55:22"
        assert "Spike" in metadata.get("characters", "")
        assert metadata.get("theme") == "adventure"
        assert metadata.get("age_group") == "early_reader"
        assert metadata.get("tone") == "exciting"
        assert metadata.get("art_style") == "cartoon"

    def test_parse_context_metadata_all_parameters(self, tmp_path):
        """Test parsing context file metadata includes all story parameters."""
        ctx_file = tmp_path / "full_params.md"
        ctx_file.write_text(
            """# Story Context: Full Params
**Generated:** 2026-04-26 00:00:00
**Original Prompt:** A wizard's quest
**Characters:** Wizard, Dragon
**Setting:** enchanted forest
**Tone:** whimsical
**Style:** fable
**Voice:** sage
**Theme:** perseverance
**Age Group:** early_reader
**Art Style:** watercolor
**Length:** bedtime
**Learning Focus:** counting

## Story
Once upon a time in an enchanted forest..."""
        )

        mgr = ContextManager()
        metadata = mgr.parse_context_metadata(ctx_file)

        assert metadata.get("characters") == "Wizard, Dragon"
        assert metadata.get("setting") == "enchanted forest"
        assert metadata.get("tone") == "whimsical"
        assert metadata.get("style") == "fable"
        assert metadata.get("voice") == "sage"
        assert metadata.get("theme") == "perseverance"
        assert metadata.get("age_group") == "early_reader"
        assert metadata.get("art_style") == "watercolor"
        assert metadata.get("length") == "bedtime"
        assert metadata.get("learning_focus") == "counting"
        assert metadata.get("prompt") == "A wizard's quest"

    def test_parse_context_metadata_with_story_preview(self, tmp_path):
        """Test that metadata includes story preview."""
        ctx_file = tmp_path / "test.md"
        ctx_file.write_text(
            """# Story Context: Test
**Generated:** 2025-10-22

## Story
Once upon a time in a magical forest, there lived a brave little dragon named Spike.
He loved to explore and make new friends. One day he discovered a mysterious cave..."""
        )

        mgr = ContextManager()
        metadata = mgr.parse_context_metadata(ctx_file)

        assert "preview" in metadata
        assert "magical forest" in metadata["preview"]

    def test_load_context_for_extension(self, tmp_path):
        """Test loading context for extension."""
        ctx_file = tmp_path / "test.md"
        content = """# Story Context
**Characters:** Alice
**Theme:** adventure

## Story
Once upon a time in a magical forest..."""
        ctx_file.write_text(content)

        mgr = ContextManager()
        loaded_content, metadata = mgr.load_context_for_extension(ctx_file)

        assert "magical forest" in loaded_content
        assert metadata["filename"] == "test"
        assert "Alice" in metadata.get("characters", "")

    def test_get_context_directory(self):
        """Test getting the context directory path."""
        mgr = ContextManager()
        context_dir = mgr.get_context_directory()

        assert context_dir.name == "context"
        assert "storyforge" in str(context_dir).lower()


class TestPromptContinuation:
    """Test Prompt continuation mode."""

    def test_continuation_mode_wrap_up(self):
        """Test continuation prompt with wrap-up ending."""
        prompt = Prompt(
            prompt="",  # Not needed in continuation mode
            characters=["Alice"],
            theme="courage",  # Valid theme value
            context="Original story content...",
            continuation_mode=True,
            ending_type="wrap_up",
        )

        story_prompt = prompt.story

        assert "CONTINUATION TASK" in story_prompt
        assert "wraps up the narrative" in story_prompt
        assert "Original story content" in story_prompt

    def test_continuation_mode_cliffhanger(self):
        """Test continuation prompt with cliffhanger ending."""
        prompt = Prompt(
            prompt="",
            characters=["Bob"],
            theme="teamwork",  # Valid theme value
            context="Previous mystery story...",
            continuation_mode=True,
            ending_type="cliffhanger",
        )

        story_prompt = prompt.story

        assert "CONTINUATION TASK" in story_prompt
        assert "cliffhanger" in story_prompt
        assert "Previous mystery story" in story_prompt

    def test_normal_mode_unchanged(self):
        """Test that normal mode is unaffected."""
        prompt = Prompt(
            prompt="A story about friendship",
            characters=["Charlie"],
            theme="kindness",  # Valid theme value
            continuation_mode=False,
        )

        story_prompt = prompt.story

        assert "CONTINUATION TASK" not in story_prompt
        assert "A story about friendship" in story_prompt

    def test_continuation_mode_preserves_parameters(self):
        """Test that continuation mode preserves original parameters."""
        prompt = Prompt(
            prompt="",
            characters=["Dragon", "Knight"],
            theme="courage",  # Valid theme value
            tone="exciting",
            age_range="early_reader",
            context="A dragon and knight became friends...",
            continuation_mode=True,
            ending_type="wrap_up",
        )

        story_prompt = prompt.story

        assert "exciting" in story_prompt
        assert "adventure" in story_prompt  # This is the style, which defaults to "adventure"

    def test_continuation_mode_includes_all_preserved_parameters(self):
        """Test that continuation prompt text includes theme, setting, characters, and learning_focus."""
        prompt = Prompt(
            prompt="",
            characters=["Wizard", "Dragon"],
            theme="courage",
            tone="silly",
            style="fantasy",
            voice="lyrical",
            age_range="early_reader",
            setting="enchanted forest",
            learning_focus="counting",
            context="Once upon a time in an enchanted forest...",
            continuation_mode=True,
            ending_type="wrap_up",
        )

        story_prompt = prompt.story

        # Core parameters (already tested)
        assert "silly" in story_prompt
        assert "fantasy" in story_prompt

        # Newly preserved parameters in continuation prompt
        assert "enchanted forest" in story_prompt
        assert "Wizard" in story_prompt
        assert "Dragon" in story_prompt
        assert "courage" in story_prompt
        assert "counting" in story_prompt

    def test_continuation_mode_omits_none_parameters(self):
        """Test that continuation prompt gracefully omits parameters that are None."""
        prompt = Prompt(
            prompt="",
            characters=None,
            theme=None,
            tone="heartwarming",
            style="adventure",
            setting=None,
            learning_focus=None,
            context="A simple story...",
            continuation_mode=True,
            ending_type="cliffhanger",
        )

        story_prompt = prompt.story

        # Should not contain the parameter labels when values are None
        assert "Maintain the setting" not in story_prompt
        assert "Keep these characters" not in story_prompt
        assert "Preserve the theme" not in story_prompt
        assert "incorporating learning" not in story_prompt
        # But core params should still be there
        assert "heartwarming" in story_prompt
        assert "adventure" in story_prompt


class TestExtendCommandIntegration:
    """Integration tests for the extend command."""

    @patch("storyforge.StoryForge.ContextManager")
    @patch("storyforge.StoryForge.PhaseExecutor")
    @patch("storyforge.StoryForge.load_config")
    @patch("typer.prompt")
    def test_extend_command_no_contexts(self, mock_prompt, mock_load_config, mock_executor, mock_context_mgr):
        """Test extend command when no context files exist."""
        # Mock context manager to return empty list
        mock_mgr = Mock()
        mock_mgr.list_available_contexts.return_value = []
        mock_context_mgr.return_value = mock_mgr

        # typer.Exit is actually click.exceptions.Exit
        from click.exceptions import Exit

        from storyforge.StoryForge import extend_story

        with pytest.raises(Exit) as exc_info:
            extend_story(backend=None, verbose=False, debug=False)

        assert exc_info.value.exit_code == 1

    @patch("storyforge.StoryForge.ContextManager")
    @patch("storyforge.StoryForge.PhaseExecutor")
    @patch("storyforge.StoryForge.load_config")
    @patch("storyforge.StoryForge.CheckpointManager")
    @patch("storyforge.StoryForge.pick_story")
    @patch("typer.prompt")
    def test_extend_command_wrap_up(
        self,
        mock_prompt,
        mock_pick_story,
        mock_checkpoint_mgr,
        mock_load_config,
        mock_executor,
        mock_context_mgr,
    ):
        """Test extend command with wrap-up ending."""
        # Mock context manager
        mock_mgr = Mock()
        mock_mgr.list_available_contexts.return_value = [
            {
                "filepath": Path("/tmp/story1.md"),
                "filename": "story1",
                "timestamp": "2025-10-22",
                "characters": "Alice",
                "theme": "courage",  # Valid theme value
                "preview": "Once upon a time...",
            }
        ]
        mock_mgr.load_context_for_extension.return_value = (
            "# Story\nFull story content...",
            {"characters": "Alice", "theme": "courage"},  # Valid theme
        )
        mock_mgr.load_chain_for_extension.return_value = (
            "# Story\nFull story content...",
            {"characters": "Alice", "theme": "courage"},  # Valid theme
        )
        # Mock get_story_chain to return a single-story chain
        mock_mgr.get_story_chain.return_value = [
            {
                "filepath": Path("/tmp/story1.md"),
                "filename": "story1",
                "timestamp": "2025-10-22",
                "prompt": "Original story",
            }
        ]
        mock_context_mgr.return_value = mock_mgr

        # Mock user input: pick_story returns 0-based index, typer.prompt for ending choice + direction
        mock_pick_story.return_value = 0
        mock_prompt.side_effect = [1, ""]  # wrap-up ending, no direction

        # Mock config
        mock_config = Mock()
        mock_config.get_field_value.return_value = None
        mock_load_config.return_value = mock_config

        # Mock checkpoint manager
        mock_checkpoint = Mock()
        mock_checkpoint_mgr.return_value = mock_checkpoint

        # Mock executor
        mock_exec = Mock()
        mock_executor.return_value = mock_exec

        # Run command
        from storyforge.StoryForge import extend_story

        extend_story(backend=None, verbose=False, debug=False)

        # Verify executor was called
        assert mock_exec.execute_new_session.called

        # Check that the prompt object was created with continuation mode
        call_args = mock_exec.execute_new_session.call_args
        prompt_obj = call_args.kwargs.get("prompt_obj")

        assert prompt_obj is not None
        assert prompt_obj.continuation_mode is True
        assert prompt_obj.ending_type == "wrap_up"
        assert "Full story content" in prompt_obj.context

    @patch("storyforge.StoryForge.ContextManager")
    @patch("storyforge.StoryForge.pick_story")
    def test_extend_command_cancelled(self, mock_pick_story, mock_context_mgr):
        """Test extend command when user cancels the picker."""
        # Mock context manager
        mock_mgr = Mock()
        mock_mgr.list_available_contexts.return_value = [
            {
                "filepath": Path("/tmp/story1.md"),
                "filename": "story1",
                "timestamp": "2025-10-22",
            }
        ]
        mock_context_mgr.return_value = mock_mgr

        # Mock picker returning None (cancelled)
        mock_pick_story.return_value = None

        from click.exceptions import Exit

        from storyforge.StoryForge import extend_story

        with pytest.raises(Exit) as exc_info:
            extend_story(backend=None, verbose=False, debug=False)

        assert exc_info.value.exit_code == 0

    @patch("storyforge.StoryForge.ContextManager")
    @patch("storyforge.StoryForge.PhaseExecutor")
    @patch("storyforge.StoryForge.load_config")
    @patch("storyforge.StoryForge.CheckpointManager")
    @patch("storyforge.StoryForge.pick_story")
    @patch("typer.prompt")
    def test_extend_preserves_all_parameters_from_metadata(
        self,
        mock_prompt,
        mock_pick_story,
        mock_checkpoint_mgr,
        mock_load_config,
        mock_executor,
        mock_context_mgr,
    ):
        """Test that extend_story creates a Prompt preserving all parameters from metadata."""
        all_metadata = {
            "filepath": Path("/tmp/story1.md"),
            "filename": "story1",
            "timestamp": "2026-04-26",
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

        mock_mgr = Mock()
        mock_mgr.list_available_contexts.return_value = [all_metadata]
        mock_mgr.load_chain_for_extension.return_value = (
            "Full story chain content...",
            all_metadata,
        )
        mock_mgr.get_story_chain.return_value = [all_metadata]
        mock_context_mgr.return_value = mock_mgr

        mock_pick_story.return_value = 0
        mock_prompt.side_effect = [1, ""]  # wrap-up ending, no direction

        mock_config = Mock()
        mock_config.get_field_value.return_value = None
        mock_load_config.return_value = mock_config

        mock_checkpoint_mgr.return_value = Mock()
        mock_exec = Mock()
        mock_executor.return_value = mock_exec

        from storyforge.StoryForge import extend_story

        extend_story(backend=None, verbose=False, debug=False)

        call_args = mock_exec.execute_new_session.call_args
        prompt_obj = call_args.kwargs.get("prompt_obj")

        # Verify every parameter was passed through from metadata
        assert prompt_obj.continuation_mode is True
        assert prompt_obj.theme == "courage"
        assert prompt_obj.tone == "silly"
        assert prompt_obj.style == "fantasy"
        assert prompt_obj.voice == "lyrical"
        assert prompt_obj.age_range == "early_reader"
        assert prompt_obj.image_style == "watercolor"
        assert prompt_obj.length == "bedtime"
        assert prompt_obj.learning_focus == "counting"
        assert prompt_obj.setting == "enchanted forest"
        assert prompt_obj.characters == ["Wizard", "Dragon"]
        assert "Full story chain content" in prompt_obj.context

        # Also verify cli_arguments and resolved_config include all params
        cli_args = call_args[0][1]  # second positional arg
        assert cli_args["setting"] == "enchanted forest"
        assert cli_args["learning_focus"] == "counting"
        assert cli_args["length"] == "bedtime"

        resolved = call_args[0][2]  # third positional arg
        assert resolved["setting"] == "enchanted forest"
        assert resolved["learning_focus"] == "counting"
        assert resolved["length"] == "bedtime"

    @patch("storyforge.StoryForge.ContextManager")
    @patch("storyforge.StoryForge.PhaseExecutor")
    @patch("storyforge.StoryForge.load_config")
    @patch("storyforge.StoryForge.CheckpointManager")
    @patch("storyforge.StoryForge.pick_story")
    @patch("typer.prompt")
    def test_extend_falls_back_to_config_for_missing_metadata(
        self,
        mock_prompt,
        mock_pick_story,
        mock_checkpoint_mgr,
        mock_load_config,
        mock_executor,
        mock_context_mgr,
    ):
        """Test extend falls back to config when metadata is missing (backward compat)."""
        # Old-format metadata missing length, learning_focus, setting
        sparse_metadata = {
            "filepath": Path("/tmp/old_story.md"),
            "filename": "old_story",
            "timestamp": "2025-01-01",
            "characters": "Alice",
            "theme": "courage",
            "tone": "heartwarming",
        }

        mock_mgr = Mock()
        mock_mgr.list_available_contexts.return_value = [sparse_metadata]
        mock_mgr.load_chain_for_extension.return_value = (
            "Old story content...",
            sparse_metadata,
        )
        mock_mgr.get_story_chain.return_value = [sparse_metadata]
        mock_context_mgr.return_value = mock_mgr

        mock_pick_story.return_value = 0
        mock_prompt.side_effect = [1, ""]

        # Config provides fallback values
        mock_config = Mock()

        def config_get(section, field):
            config_values = {
                ("story", "length"): "medium",
                ("story", "setting"): "a magical kingdom",
                ("story", "learning_focus"): "colors",
                ("images", "image_style"): "chibi",
            }
            return config_values.get((section, field))

        mock_config.get_field_value.side_effect = config_get
        mock_load_config.return_value = mock_config

        mock_checkpoint_mgr.return_value = Mock()
        mock_exec = Mock()
        mock_executor.return_value = mock_exec

        from storyforge.StoryForge import extend_story

        extend_story(backend=None, verbose=False, debug=False)

        call_args = mock_exec.execute_new_session.call_args
        prompt_obj = call_args.kwargs.get("prompt_obj")

        # Falls back to config values when metadata is missing
        assert prompt_obj.length == "medium"
        assert prompt_obj.setting == "a magical kingdom"
        assert prompt_obj.learning_focus == "colors"
        assert prompt_obj.image_style == "chibi"
        # But metadata values are still used when present
        assert prompt_obj.theme == "courage"
        assert prompt_obj.tone == "heartwarming"


class TestOutputDirectoryNaming:
    """Test output directory naming for extended stories."""

    def test_generate_default_output_dir_normal(self):
        """Test normal output directory generation."""
        from storyforge.StoryForge import generate_default_output_dir

        output_dir = generate_default_output_dir(extended=False)

        assert "storyforge_output_" in output_dir
        assert "_extended" not in output_dir

    def test_generate_default_output_dir_extended(self):
        """Test extended output directory generation."""
        from storyforge.StoryForge import generate_default_output_dir

        output_dir = generate_default_output_dir(extended=True)

        assert "storyforge_output_" in output_dir
        assert "_extended" in output_dir
