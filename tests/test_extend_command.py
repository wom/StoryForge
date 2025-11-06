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
    @patch("storyforge.StoryForge.Confirm.ask")
    @patch("storyforge.StoryForge.PhaseExecutor")
    @patch("storyforge.StoryForge.load_config")
    @patch("storyforge.StoryForge.CheckpointManager")
    @patch("typer.prompt")
    def test_extend_command_wrap_up(
        self, mock_prompt, mock_checkpoint_mgr, mock_load_config, mock_executor, mock_confirm, mock_context_mgr
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

        # Mock user input
        mock_prompt.side_effect = [1, 1]  # Select story 1, wrap-up ending
        mock_confirm.return_value = False  # Don't view full story

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
    @patch("typer.prompt")
    def test_extend_command_invalid_selection(self, mock_prompt, mock_context_mgr):
        """Test extend command with invalid story selection."""
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

        # Mock user selecting invalid option
        mock_prompt.return_value = 5  # Invalid selection

        from click.exceptions import Exit

        from storyforge.StoryForge import extend_story

        with pytest.raises(Exit) as exc_info:
            extend_story(backend=None, verbose=False, debug=False)

        assert exc_info.value.exit_code == 1


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
