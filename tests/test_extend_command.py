"""
Tests for the sf extend command and related functionality.
"""

from unittest.mock import patch

from storyforge.context import ContextManager
from storyforge.paths import create_output_directory_name
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

    def test_parse_context_metadata_normalizes_legacy_source_labels(self, tmp_path):
        """Historical source labels must not make a saved context unextendable."""
        ctx_file = tmp_path / "legacy_params.md"
        ctx_file.write_text(
            """# Story Context: Legacy
**Theme:** Random (kindness)
**Age Group:** preschool (CLI)
**Tone:** heartwarming (Config)
**Style:** adventure (Default)
**Voice:** lyrical (CLI)
**Art Style:** watercolor (CLI)
**Length:** short (CLI)
**Learning Focus:** colors (Config)

## Story

A legacy story."""
        )

        metadata = ContextManager().parse_context_metadata(ctx_file)

        assert metadata["theme"] == "kindness"
        assert metadata["age_group"] == "preschool"
        assert metadata["tone"] == "heartwarming"
        assert metadata["style"] == "adventure"
        assert metadata["voice"] == "lyrical"
        assert metadata["art_style"] == "watercolor"
        assert metadata["length"] == "short"
        assert metadata["learning_focus"] == "colors"

    def test_parse_context_metadata_does_not_read_source_line_as_blank_value(self, tmp_path):
        """A blank parameter must not consume its provenance line as the value."""
        ctx_file = tmp_path / "blank_learning_focus.md"
        ctx_file.write_text(
            """# Story Context: Blank Learning Focus
**Learning Focus:**

**Learning Focus Source:** CLI

## Story

A story without an educational focus."""
        )

        metadata = ContextManager().parse_context_metadata(ctx_file)

        assert "learning_focus" not in metadata

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


class TestOutputDirectoryNaming:
    """Test output directory naming for extended stories."""

    def test_generate_default_output_dir_normal(self):
        """Test normal output directory generation."""
        output_dir = create_output_directory_name(extended=False)

        assert "storyforge_output_" in output_dir
        assert "_extended" not in output_dir

    def test_generate_default_output_dir_extended(self):
        """Test extended output directory generation."""
        output_dir = create_output_directory_name(extended=True)

        assert "storyforge_output_" in output_dir
        assert "_extended" in output_dir
