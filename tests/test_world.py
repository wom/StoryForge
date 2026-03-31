"""Tests for the world file feature (world.md — always-included context)."""

import os
from unittest.mock import MagicMock, patch

from storyforge.context import ContextManager
from storyforge.prompt import Prompt
from storyforge.world_template import WORLD_FILENAME, WORLD_TEMPLATE


class TestStripHtmlComments:
    """Unit tests for ContextManager._strip_html_comments()."""

    def test_no_comments(self):
        """Text without comments passes through unchanged."""
        text = "# World\n\nLuna is brave."
        assert ContextManager._strip_html_comments(text) == text

    def test_single_line_comment(self):
        """Single-line comment is removed."""
        assert ContextManager._strip_html_comments("Hello <!-- hidden --> World") == "Hello  World"

    def test_multiline_comment(self):
        """Multi-line comment spanning several lines is removed."""
        text = "Before\n<!-- line1\nline2\nline3 -->\nAfter"
        result = ContextManager._strip_html_comments(text)
        assert "line1" not in result
        assert "Before" in result
        assert "After" in result

    def test_multiple_comments(self):
        """Multiple separate comments are all removed."""
        text = "A <!-- one --> B <!-- two --> C"
        assert ContextManager._strip_html_comments(text) == "A  B  C"

    def test_comment_at_start(self):
        """Comment at the start of text is removed."""
        assert ContextManager._strip_html_comments("<!-- header -->\n# World") == "# World"

    def test_comment_at_end(self):
        """Comment at the end of text is removed."""
        assert ContextManager._strip_html_comments("# World\n<!-- footer -->") == "# World"

    def test_adjacent_comments(self):
        """Back-to-back comments are both removed."""
        text = "A\n<!-- one --><!-- two -->\nB"
        result = ContextManager._strip_html_comments(text)
        assert "one" not in result
        assert "two" not in result
        assert "A" in result
        assert "B" in result

    def test_collapses_triple_newlines(self):
        """Three or more consecutive newlines collapse to double."""
        text = "A\n\n\n\nB"
        assert ContextManager._strip_html_comments(text) == "A\n\nB"

    def test_preserves_double_newlines(self):
        """Exactly two newlines (one blank line) are preserved."""
        text = "A\n\nB"
        assert ContextManager._strip_html_comments(text) == "A\n\nB"

    def test_all_comments_returns_empty(self):
        """Text that is entirely comments returns empty string."""
        assert ContextManager._strip_html_comments("<!-- everything is a comment -->") == ""

    def test_unclosed_comment_left_as_is(self):
        """An unclosed <!-- without --> is not treated as a comment."""
        text = "Hello <!-- not closed"
        assert ContextManager._strip_html_comments(text) == text.strip()

    def test_stray_closing_tag_left_as_is(self):
        """A stray --> without opening <!-- is left in text."""
        text = "Hello --> world"
        assert ContextManager._strip_html_comments(text) == text.strip()

    def test_empty_comment(self):
        """An empty comment <!----> is removed."""
        assert ContextManager._strip_html_comments("A<!---->B") == "AB"

    def test_comment_with_dashes(self):
        """Comment containing dashes is handled."""
        assert ContextManager._strip_html_comments("A <!-- -- stuff -- --> B") == "A  B"

    def test_real_template_stripping(self):
        """The actual WORLD_TEMPLATE strips cleanly to just headings."""
        result = ContextManager._strip_html_comments(WORLD_TEMPLATE)
        assert "<!--" not in result
        assert "-->" not in result
        assert "# Story World" in result
        assert "## Characters" in result
        assert "## Places" in result


class TestWorldFileDiscovery:
    """Test _discover_world_file() resolution priority."""

    def test_explicit_path_found(self, tmp_path):
        """Explicit world_file_path is used when the file exists."""
        world_file = tmp_path / "custom_world.md"
        world_file.write_text("# My World", encoding="utf-8")
        mgr = ContextManager(world_file_path=str(world_file))
        assert mgr._discover_world_file() == world_file

    def test_explicit_path_missing(self, tmp_path):
        """Returns None when explicit path doesn't exist."""
        mgr = ContextManager(world_file_path=str(tmp_path / "nonexistent.md"))
        assert mgr._discover_world_file() is None

    def test_explicit_path_with_tilde(self, tmp_path):
        """Tilde in explicit path is expanded."""
        world_file = tmp_path / "world.md"
        world_file.write_text("# World", encoding="utf-8")
        with patch("pathlib.Path.expanduser", return_value=world_file):
            mgr = ContextManager(world_file_path="~/world.md")
            assert mgr._discover_world_file() is not None

    @patch.dict(os.environ, {"STORYFORGE_TEST_CONTEXT_DIR": ""})
    def test_local_context_dir(self, tmp_path, monkeypatch):
        """Finds world.md in ./context/ directory."""
        monkeypatch.chdir(tmp_path)
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        world_file = context_dir / WORLD_FILENAME
        world_file.write_text("# World", encoding="utf-8")
        monkeypatch.delenv("STORYFORGE_TEST_CONTEXT_DIR", raising=False)

        mgr = ContextManager()
        result = mgr._discover_world_file()
        assert result is not None
        assert result.name == WORLD_FILENAME

    def test_test_context_dir_env(self, tmp_path, monkeypatch):
        """Finds world.md via STORYFORGE_TEST_CONTEXT_DIR env var."""
        world_file = tmp_path / WORLD_FILENAME
        world_file.write_text("# Test World", encoding="utf-8")
        monkeypatch.setenv("STORYFORGE_TEST_CONTEXT_DIR", str(tmp_path))

        mgr = ContextManager()
        result = mgr._discover_world_file()
        assert result == world_file

    def test_explicit_path_takes_priority_over_local(self, tmp_path, monkeypatch):
        """Explicit world_file_path is preferred over ./context/world.md."""
        monkeypatch.chdir(tmp_path)
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / WORLD_FILENAME).write_text("# Local", encoding="utf-8")

        custom = tmp_path / "custom_world.md"
        custom.write_text("# Custom", encoding="utf-8")

        mgr = ContextManager(world_file_path=str(custom))
        assert mgr._discover_world_file() == custom

    def test_no_world_file_returns_none(self, tmp_path, monkeypatch):
        """Returns None when no world file exists anywhere."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("STORYFORGE_TEST_CONTEXT_DIR", raising=False)
        mgr = ContextManager()
        with patch("storyforge.context.user_data_dir", return_value=str(tmp_path / "xdg")):
            assert mgr._discover_world_file() is None


class TestWorldFileLoading:
    """Test load_world() reads content and strips HTML comments."""

    def test_loads_content_without_comments(self, tmp_path):
        """Plain content without comments is returned as-is."""
        world_file = tmp_path / WORLD_FILENAME
        content = "# My World\n\n## Characters\n\nLuna is a brave explorer."
        world_file.write_text(content, encoding="utf-8")

        mgr = ContextManager(world_file_path=str(world_file))
        assert mgr.load_world() == content

    def test_strips_single_line_comment(self, tmp_path):
        """Single-line HTML comments are removed."""
        world_file = tmp_path / WORLD_FILENAME
        world_file.write_text("# World\n\n<!-- this is a comment -->\n\nLuna is brave.", encoding="utf-8")

        mgr = ContextManager(world_file_path=str(world_file))
        result = mgr.load_world()
        assert "<!--" not in result
        assert "this is a comment" not in result
        assert "Luna is brave." in result

    def test_strips_multiline_comment(self, tmp_path):
        """Multi-line HTML comments are removed."""
        world_file = tmp_path / WORLD_FILENAME
        world_file.write_text("# World\n\n<!-- this is\na multiline\ncomment -->\n\nLuna is brave.", encoding="utf-8")

        mgr = ContextManager(world_file_path=str(world_file))
        result = mgr.load_world()
        assert "<!--" not in result
        assert "multiline" not in result
        assert "Luna is brave." in result

    def test_collapses_blank_lines_after_stripping(self, tmp_path):
        """Runs of blank lines left after comment removal are collapsed."""
        world_file = tmp_path / WORLD_FILENAME
        world_file.write_text("# World\n\n<!-- comment -->\n\n\n\nLuna is brave.", encoding="utf-8")

        mgr = ContextManager(world_file_path=str(world_file))
        result = mgr.load_world()
        assert "\n\n\n" not in result
        assert result == "# World\n\nLuna is brave."

    def test_template_strips_to_headings(self, tmp_path):
        """The default template with only comments strips to just headings."""
        world_file = tmp_path / WORLD_FILENAME
        world_file.write_text(WORLD_TEMPLATE, encoding="utf-8")

        mgr = ContextManager(world_file_path=str(world_file))
        result = mgr.load_world()
        assert result is not None
        assert "<!--" not in result
        assert "# Story World" in result

    def test_empty_file_returns_none(self, tmp_path):
        """Empty world file returns None."""
        world_file = tmp_path / WORLD_FILENAME
        world_file.write_text("   \n  \n", encoding="utf-8")

        mgr = ContextManager(world_file_path=str(world_file))
        assert mgr.load_world() is None

    def test_comments_only_file_returns_none(self, tmp_path):
        """File with only a comment and no other text returns None."""
        world_file = tmp_path / WORLD_FILENAME
        world_file.write_text("<!-- just a comment -->", encoding="utf-8")

        mgr = ContextManager(world_file_path=str(world_file))
        assert mgr.load_world() is None

    def test_missing_file_returns_none(self, tmp_path):
        """Missing world file returns None."""
        mgr = ContextManager(world_file_path=str(tmp_path / "missing.md"))
        assert mgr.load_world() is None

    def test_unreadable_file_returns_none(self, tmp_path):
        """Unreadable world file returns None (with logged warning)."""
        world_file = tmp_path / WORLD_FILENAME
        world_file.write_text("content", encoding="utf-8")
        world_file.chmod(0o000)

        mgr = ContextManager(world_file_path=str(world_file))
        try:
            assert mgr.load_world() is None
        finally:
            world_file.chmod(0o644)


class TestWorldFileExclusion:
    """Test that world.md is excluded from _discover_context_files()."""

    def test_world_md_excluded_from_context_files(self, tmp_path, monkeypatch):
        """world.md should not appear in _discover_context_files() results."""
        monkeypatch.setenv("STORYFORGE_TEST_CONTEXT_DIR", str(tmp_path))
        (tmp_path / WORLD_FILENAME).write_text("# World", encoding="utf-8")
        (tmp_path / "story1.md").write_text("# Story 1", encoding="utf-8")
        (tmp_path / "story2.md").write_text("# Story 2", encoding="utf-8")

        mgr = ContextManager()
        files = mgr._discover_context_files()
        filenames = [f.name for f in files]

        assert WORLD_FILENAME not in filenames
        assert "story1.md" in filenames
        assert "story2.md" in filenames

    def test_world_discovered_separately(self, tmp_path, monkeypatch):
        """world.md is excluded from context files but found by _discover_world_file."""
        monkeypatch.setenv("STORYFORGE_TEST_CONTEXT_DIR", str(tmp_path))
        (tmp_path / WORLD_FILENAME).write_text("# World", encoding="utf-8")
        (tmp_path / "story.md").write_text("# Story", encoding="utf-8")

        mgr = ContextManager()
        context_names = [f.name for f in mgr._discover_context_files()]
        world_file = mgr._discover_world_file()

        assert WORLD_FILENAME not in context_names
        assert world_file is not None
        assert world_file.name == WORLD_FILENAME

    def test_context_files_work_without_world(self, tmp_path, monkeypatch):
        """Context file discovery works normally when no world.md exists."""
        monkeypatch.setenv("STORYFORGE_TEST_CONTEXT_DIR", str(tmp_path))
        (tmp_path / "story1.md").write_text("# Story", encoding="utf-8")

        mgr = ContextManager()
        files = mgr._discover_context_files()
        assert len(files) == 1
        assert files[0].name == "story1.md"


class TestWorldInPrompt:
    """Test that world content is injected into prompts."""

    def test_world_prepended_in_story_prompt(self):
        """World content appears before context in story prompt."""
        p = Prompt(
            prompt="A dragon learns to fly",
            world="## Characters\nDrago the friendly dragon",
            context="Previous story about Drago",
        )
        story = p.story
        world_pos = story.index("Drago the friendly dragon")
        context_pos = story.index("Previous story about Drago")
        assert world_pos < context_pos

    def test_world_only_no_context(self):
        """Prompt works with world but no context."""
        p = Prompt(
            prompt="A dragon learns to fly",
            world="## Characters\nDrago the friendly dragon",
        )
        story = p.story
        assert "Story world definition:" in story
        assert "Drago the friendly dragon" in story
        assert "Based on the story world above" in story

    def test_no_world_no_context(self):
        """Prompt works with neither world nor context."""
        p = Prompt(prompt="A dragon learns to fly")
        story = p.story
        assert "Story world definition:" not in story
        assert "write" in story.lower()

    def test_world_in_continuation_mode(self):
        """World content appears in continuation prompts."""
        p = Prompt(
            prompt="",
            world="## Characters\nDrago the friendly dragon",
            context="Previous adventure...",
            continuation_mode=True,
            ending_type="wrap_up",
        )
        story = p.story
        assert "Story world definition:" in story
        assert "Drago the friendly dragon" in story

    def test_world_in_refinement_prompt(self):
        """World content appears in refinement prompts."""
        p = Prompt(
            prompt="A dragon learns to fly",
            world="## Characters\nDrago the friendly dragon",
            refinement_mode=True,
            original_story="Once upon a time...",
            refinement_instructions="Make Drago bigger",
        )
        story = p.story
        assert "STORY WORLD:" in story
        assert "Drago the friendly dragon" in story

    def test_world_in_image_prompt(self):
        """World content appears in image prompts."""
        p = Prompt(
            prompt="A dragon learns to fly",
            world="## Characters\nDrago: a small green dragon with golden wings",
        )
        images = p.image(1)
        assert len(images) == 1
        assert "Drago: a small green dragon with golden wings" in images[0]

    def test_world_in_every_image_of_multi_set(self):
        """World content appears in each image prompt of a multi-image set."""
        p = Prompt(
            prompt="A dragon learns to fly",
            world="## Characters\nDrago the dragon",
        )
        images = p.image(3)
        assert len(images) == 3
        for img in images:
            assert "Drago the dragon" in img

    def test_world_and_context_both_in_image_prompt(self):
        """Both world and context appear in image prompts, world first."""
        p = Prompt(
            prompt="A dragon learns to fly",
            world="## Characters\nDrago the dragon",
            context="Previous adventure context",
        )
        images = p.image(1)
        world_pos = images[0].index("Drago the dragon")
        context_pos = images[0].index("Previous adventure context")
        assert world_pos < context_pos


class TestWorldPhaseExecutor:
    """Test world loading in the phase executor."""

    @patch("storyforge.phase_executor.ContextManager")
    def test_phase_context_load_loads_world(self, mock_context_mgr_class):
        """_phase_context_load loads world and stores it on the executor."""
        from storyforge.checkpoint import CheckpointData, CheckpointManager
        from storyforge.phase_executor import PhaseExecutor

        mock_context_mgr = MagicMock()
        mock_context_mgr.load_world.return_value = "## Characters\nLuna"
        mock_context_mgr.extract_relevant_context.return_value = "Summarized context"
        mock_context_mgr.has_old_context = False
        mock_context_mgr_class.return_value = mock_context_mgr

        checkpoint_mgr = MagicMock(spec=CheckpointManager)
        executor = PhaseExecutor(checkpoint_mgr)
        executor.checkpoint_data = MagicMock(spec=CheckpointData)
        executor.checkpoint_data.resolved_config = {"use_context": True, "verbose": False}
        executor.checkpoint_data.original_inputs = {"prompt": "Test prompt"}
        executor.checkpoint_data.context_data = {}

        executor._phase_context_load()

        assert executor.world == "## Characters\nLuna"
        assert executor.context == "Summarized context"
        mock_context_mgr.load_world.assert_called_once()

    @patch("storyforge.phase_executor.ContextManager")
    def test_phase_context_load_world_none_when_missing(self, mock_context_mgr_class):
        """_phase_context_load sets world to None when no world file exists."""
        from storyforge.checkpoint import CheckpointData, CheckpointManager
        from storyforge.phase_executor import PhaseExecutor

        mock_context_mgr = MagicMock()
        mock_context_mgr.load_world.return_value = None
        mock_context_mgr.extract_relevant_context.return_value = None
        mock_context_mgr.has_old_context = False
        mock_context_mgr_class.return_value = mock_context_mgr

        checkpoint_mgr = MagicMock(spec=CheckpointManager)
        executor = PhaseExecutor(checkpoint_mgr)
        executor.checkpoint_data = MagicMock(spec=CheckpointData)
        executor.checkpoint_data.resolved_config = {"use_context": True, "verbose": False}
        executor.checkpoint_data.original_inputs = {"prompt": "Test prompt"}
        executor.checkpoint_data.context_data = {}

        executor._phase_context_load()

        assert executor.world is None

    @patch("storyforge.phase_executor.ContextManager")
    def test_world_file_path_passed_from_config(self, mock_context_mgr_class):
        """world_file from resolved_config is passed to ContextManager."""
        from storyforge.checkpoint import CheckpointData, CheckpointManager
        from storyforge.phase_executor import PhaseExecutor

        mock_context_mgr = MagicMock()
        mock_context_mgr.load_world.return_value = None
        mock_context_mgr.extract_relevant_context.return_value = None
        mock_context_mgr.has_old_context = False
        mock_context_mgr_class.return_value = mock_context_mgr

        checkpoint_mgr = MagicMock(spec=CheckpointManager)
        executor = PhaseExecutor(checkpoint_mgr)
        executor.checkpoint_data = MagicMock(spec=CheckpointData)
        executor.checkpoint_data.resolved_config = {
            "use_context": True,
            "verbose": False,
            "world_file": "/custom/world.md",
        }
        executor.checkpoint_data.original_inputs = {"prompt": "Test"}
        executor.checkpoint_data.context_data = {}

        executor._phase_context_load()

        mock_context_mgr_class.assert_called_once_with(max_tokens=None, world_file_path="/custom/world.md")


class TestWorldTemplate:
    """Test the world.md template."""

    def test_template_is_valid_markdown(self):
        """Template contains expected markdown sections."""
        assert "# Story World" in WORLD_TEMPLATE
        assert "## Characters" in WORLD_TEMPLATE
        assert "## Places" in WORLD_TEMPLATE
        assert "## Rules & Lore" in WORLD_TEMPLATE
        assert "## Relationships" in WORLD_TEMPLATE
        assert "## Tone & Style Notes" in WORLD_TEMPLATE

    def test_template_has_guidance_comments(self):
        """Template contains HTML comment guidance."""
        assert "<!--" in WORLD_TEMPLATE
        assert "-->" in WORLD_TEMPLATE

    def test_world_filename_constant(self):
        """WORLD_FILENAME is world.md."""
        assert WORLD_FILENAME == "world.md"


class TestWorldCLICommands:
    """Test the storyforge world CLI commands."""

    def test_world_init_creates_file(self, tmp_path, monkeypatch):
        """world init creates world.md from template."""
        from typer.testing import CliRunner

        from storyforge.StoryForge import app

        runner = CliRunner()
        monkeypatch.chdir(tmp_path)
        context_dir = tmp_path / "context"
        context_dir.mkdir()

        result = runner.invoke(app, ["world", "init"])
        assert result.exit_code == 0
        assert "created" in result.output.lower() or "✨" in result.output

        world_file = context_dir / WORLD_FILENAME
        assert world_file.exists()
        content = world_file.read_text(encoding="utf-8")
        assert "# Story World" in content

    def test_world_init_no_overwrite(self, tmp_path, monkeypatch):
        """world init refuses to overwrite without --force."""
        from typer.testing import CliRunner

        from storyforge.StoryForge import app

        runner = CliRunner()
        monkeypatch.chdir(tmp_path)
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / WORLD_FILENAME).write_text("existing", encoding="utf-8")

        result = runner.invoke(app, ["world", "init"])
        assert result.exit_code == 0
        assert "already exists" in result.output.lower()
        assert (context_dir / WORLD_FILENAME).read_text(encoding="utf-8") == "existing"

    def test_world_init_force_overwrites(self, tmp_path, monkeypatch):
        """world init --force overwrites existing file."""
        from typer.testing import CliRunner

        from storyforge.StoryForge import app

        runner = CliRunner()
        monkeypatch.chdir(tmp_path)
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        (context_dir / WORLD_FILENAME).write_text("old content", encoding="utf-8")

        result = runner.invoke(app, ["world", "init", "--force"])
        assert result.exit_code == 0
        content = (context_dir / WORLD_FILENAME).read_text(encoding="utf-8")
        assert "# Story World" in content

    def test_world_show_displays_content(self, tmp_path, monkeypatch):
        """world show displays world.md content."""
        from typer.testing import CliRunner

        from storyforge.StoryForge import app

        runner = CliRunner()
        monkeypatch.setenv("STORYFORGE_TEST_CONTEXT_DIR", str(tmp_path))
        (tmp_path / WORLD_FILENAME).write_text("# My World\n\nLuna is brave.", encoding="utf-8")

        result = runner.invoke(app, ["world", "show"])
        assert result.exit_code == 0
        assert "Luna is brave" in result.output

    def test_world_show_strips_comments(self, tmp_path, monkeypatch):
        """world show strips HTML comments from display."""
        from typer.testing import CliRunner

        from storyforge.StoryForge import app

        runner = CliRunner()
        monkeypatch.setenv("STORYFORGE_TEST_CONTEXT_DIR", str(tmp_path))
        (tmp_path / WORLD_FILENAME).write_text("# World\n\n<!-- hidden -->\n\nVisible content", encoding="utf-8")

        result = runner.invoke(app, ["world", "show"])
        assert result.exit_code == 0
        assert "hidden" not in result.output
        assert "Visible content" in result.output

    def test_world_show_no_file(self, tmp_path, monkeypatch):
        """world show reports when no world file exists."""
        from typer.testing import CliRunner

        from storyforge.StoryForge import app

        runner = CliRunner()
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("STORYFORGE_TEST_CONTEXT_DIR", raising=False)

        with patch("storyforge.context.user_data_dir", return_value=str(tmp_path / "xdg")):
            result = runner.invoke(app, ["world", "show"])
        assert result.exit_code == 1
        assert "no world file" in result.output.lower()

    def test_world_path_shows_path(self, tmp_path, monkeypatch):
        """world path shows the resolved path."""
        from typer.testing import CliRunner

        from storyforge.StoryForge import app

        runner = CliRunner()
        monkeypatch.setenv("STORYFORGE_TEST_CONTEXT_DIR", str(tmp_path))
        world_file = tmp_path / WORLD_FILENAME
        world_file.write_text("# World", encoding="utf-8")

        result = runner.invoke(app, ["world", "path"])
        assert result.exit_code == 0
        assert WORLD_FILENAME in result.output

    def test_world_path_no_file(self, tmp_path, monkeypatch):
        """world path reports expected location when no file exists."""
        from typer.testing import CliRunner

        from storyforge.StoryForge import app

        runner = CliRunner()
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("STORYFORGE_TEST_CONTEXT_DIR", raising=False)

        with patch("storyforge.context.user_data_dir", return_value=str(tmp_path / "xdg")):
            result = runner.invoke(app, ["world", "path"])
        assert result.exit_code == 0
        assert "no world file" in result.output.lower()
        assert "storyforge world init" in result.output
