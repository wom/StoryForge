"""Comprehensive tests for PhaseExecutor phase methods."""

from unittest.mock import MagicMock, patch

import pytest

from storyforge.checkpoint import CheckpointData, CheckpointManager
from storyforge.phase_executor import PhaseExecutor


class TestPhaseExecutorPhases:
    """Test individual phase execution methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.checkpoint_manager = MagicMock(spec=CheckpointManager)
        self.phase_executor = PhaseExecutor(self.checkpoint_manager)

        # Create a basic checkpoint
        self.checkpoint_data = CheckpointData.create_new(
            "Test story prompt",
            {"style": "adventure", "age_range": "preschool"},
            {"backend": "gemini", "verbose": False, "debug": False},
        )
        self.phase_executor.checkpoint_data = self.checkpoint_data

    def test_phase_init_does_nothing(self):
        """Test _phase_init is a no-op (validation happens in CLI)."""
        # Should not raise any exceptions
        self.phase_executor._phase_init()

    @patch("storyforge.phase_executor.load_config")
    def test_phase_config_load(self, mock_load_config):
        """Test _phase_config_load loads configuration."""
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        self.phase_executor._phase_config_load()

        assert self.phase_executor.config == mock_config
        mock_load_config.assert_called_once()

    @patch("storyforge.phase_executor.console")
    @patch("storyforge.phase_executor.get_backend")
    def test_phase_backend_init(self, mock_get_backend, mock_console):
        """Test _phase_backend_init initializes the backend."""
        mock_backend = MagicMock()
        mock_backend.name = "gemini"
        mock_get_backend.return_value = mock_backend
        self.checkpoint_data.resolved_config["backend"] = "gemini"

        self.phase_executor._phase_backend_init()

        assert self.phase_executor.llm_backend == mock_backend
        # Check it was called with config_backend and config keyword args
        mock_get_backend.assert_called_once_with(config_backend="gemini", config=self.phase_executor.config)

    @patch("storyforge.phase_executor.console")
    @patch("storyforge.phase_executor.get_backend")
    def test_phase_backend_init_auto_detect(self, mock_get_backend, mock_console):
        """Test _phase_backend_init with auto-detection."""
        mock_backend = MagicMock()
        mock_backend.name = "gemini"
        mock_get_backend.return_value = mock_backend
        self.checkpoint_data.resolved_config["backend"] = None

        self.phase_executor._phase_backend_init()

        # Should call with None to trigger auto-detection
        mock_get_backend.assert_called_once_with(config_backend=None, config=self.phase_executor.config)

    @patch("storyforge.phase_executor.console")
    @patch("storyforge.phase_executor.get_backend")
    def test_phase_backend_init_failure(self, mock_get_backend, mock_console):
        """Test _phase_backend_init handles initialization failure."""
        mock_get_backend.side_effect = Exception("API key not found")
        self.checkpoint_data.resolved_config["backend"] = "gemini"

        with pytest.raises(RuntimeError, match="Failed to initialize.*API key"):
            self.phase_executor._phase_backend_init()

    @patch("storyforge.phase_executor.typer.Exit")
    @patch("storyforge.phase_executor.console")
    def test_phase_prompt_confirm_rejected(self, mock_console, mock_exit):
        """Test _phase_prompt_confirm when user rejects."""
        mock_exit.side_effect = SystemExit(0)
        self.checkpoint_data.original_inputs["prompt"] = "Test prompt"
        self.checkpoint_data.original_inputs["cli_arguments"] = {}

        # Mock the function imported in phase_executor
        with patch("storyforge.StoryForge.show_prompt_summary_and_confirm", return_value=False):
            with pytest.raises(SystemExit):
                self.phase_executor._phase_prompt_confirm()

    @patch("storyforge.phase_executor.console")
    def test_phase_prompt_confirm_accepted(self, mock_console):
        """Test _phase_prompt_confirm when user accepts."""
        self.checkpoint_data.original_inputs["prompt"] = "Test prompt"
        self.checkpoint_data.original_inputs["cli_arguments"] = {
            "age_range": "preschool",
            "length": "short",
            "style": "adventure",
        }

        with patch("storyforge.StoryForge.show_prompt_summary_and_confirm", return_value=True):
            # Should not raise exception
            self.phase_executor._phase_prompt_confirm()

    @patch("storyforge.phase_executor.ContextManager")
    def test_phase_context_load_with_context(self, mock_context_mgr_class):
        """Test _phase_context_load when context is loaded via summarization."""
        mock_context_mgr = MagicMock()
        mock_context_mgr.extract_relevant_context.return_value = "Summarized context content"
        mock_context_mgr_class.return_value = mock_context_mgr

        self.checkpoint_data.resolved_config["use_context"] = True
        self.checkpoint_data.context_data = {}  # Initialize context_data
        self.checkpoint_data.original_inputs["prompt"] = "Test story prompt"

        # Set up a mock backend with token budget
        mock_backend = MagicMock()
        mock_backend.get_context_token_budget.return_value = 64000
        mock_backend.CONTEXT_BUDGET_RATIO = 0.50
        mock_backend.text_input_limit = 128000
        self.phase_executor.llm_backend = mock_backend

        self.phase_executor._phase_context_load()

        assert self.phase_executor.context == "Summarized context content"
        mock_context_mgr.extract_relevant_context.assert_called_once_with(prompt="Test story prompt")
        mock_context_mgr_class.assert_called_once_with(max_tokens=64000)
        assert self.checkpoint_data.context_data["loaded_context"] == "Summarized context content"
        assert self.checkpoint_data.context_data["summarized"] is True

    @patch("storyforge.phase_executor.ContextManager")
    def test_phase_context_load_without_context(self, mock_context_mgr_class):
        """Test _phase_context_load when context is disabled."""
        self.checkpoint_data.resolved_config["use_context"] = False

        self.phase_executor._phase_context_load()

        assert self.phase_executor.context is None

    @patch("storyforge.phase_executor.ContextManager")
    def test_phase_context_load_no_prompt_falls_back_to_raw(self, mock_context_mgr_class):
        """Test _phase_context_load falls back to raw concat when no prompt available."""
        mock_context_mgr = MagicMock()
        mock_context_mgr.load_context.return_value = "Raw context content"
        mock_context_mgr_class.return_value = mock_context_mgr

        self.checkpoint_data.resolved_config["use_context"] = True
        self.checkpoint_data.context_data = {}
        self.checkpoint_data.original_inputs["prompt"] = ""  # Empty prompt

        self.phase_executor._phase_context_load()

        assert self.phase_executor.context == "Raw context content"
        mock_context_mgr.load_context.assert_called_once()
        mock_context_mgr.extract_relevant_context.assert_not_called()

    @patch("storyforge.phase_executor.ContextManager")
    def test_phase_context_load_uses_backend_budget(self, mock_context_mgr_class):
        """Test _phase_context_load passes backend token budget to ContextManager."""
        mock_context_mgr = MagicMock()
        mock_context_mgr.extract_relevant_context.return_value = "Context"
        mock_context_mgr_class.return_value = mock_context_mgr

        self.checkpoint_data.resolved_config["use_context"] = True
        self.checkpoint_data.context_data = {}
        self.checkpoint_data.original_inputs["prompt"] = "Test"

        # Set up backend with specific budget
        mock_backend = MagicMock()
        mock_backend.get_context_token_budget.return_value = 100000
        mock_backend.CONTEXT_BUDGET_RATIO = 0.50
        mock_backend.text_input_limit = 200000
        self.phase_executor.llm_backend = mock_backend

        self.phase_executor._phase_context_load()

        mock_context_mgr_class.assert_called_once_with(max_tokens=100000)

    @patch("storyforge.phase_executor.Prompt")
    def test_phase_build_prompt(self, mock_prompt_class):
        """Test _phase_build_prompt creates prompt object."""
        mock_prompt = MagicMock()
        mock_prompt_class.return_value = mock_prompt

        self.checkpoint_data.original_inputs["prompt"] = "A wizard's quest"
        self.checkpoint_data.original_inputs["cli_arguments"] = {
            "age_range": "preschool",
            "length": "short",
            "style": "adventure",
            "theme": "courage",
        }
        self.phase_executor.context = None

        self.phase_executor._phase_build_prompt()

        assert self.phase_executor.story_prompt == mock_prompt
        mock_prompt_class.assert_called_once()

    @patch("storyforge.phase_executor.Prompt")
    def test_phase_build_prompt_with_context(self, mock_prompt_class):
        """Test _phase_build_prompt with context."""
        mock_prompt = MagicMock()
        mock_prompt_class.return_value = mock_prompt

        self.checkpoint_data.original_inputs["prompt"] = "A wizard's quest"
        self.checkpoint_data.original_inputs["cli_arguments"] = {"age_range": "preschool"}
        self.phase_executor.context = "Previous story context"

        self.phase_executor._phase_build_prompt()

        # Verify context was passed to Prompt
        call_kwargs = mock_prompt_class.call_args[1]
        assert call_kwargs["context"] == "Previous story context"

    @patch("storyforge.phase_executor.Progress")
    @patch("storyforge.phase_executor.Confirm.ask")
    @patch("storyforge.phase_executor.console")
    def test_phase_story_generate(self, mock_console, mock_confirm, mock_progress):
        """Test _phase_story_generate generates a story."""
        # Mock the refinement confirmation to skip refinement
        mock_confirm.return_value = False

        # Mock Progress context manager
        mock_progress_instance = MagicMock()
        mock_progress.return_value.__enter__ = MagicMock(return_value=mock_progress_instance)
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)

        mock_backend = MagicMock()
        mock_backend.generate_story.return_value = "Generated story content"
        self.phase_executor.llm_backend = mock_backend

        mock_prompt = MagicMock()
        mock_prompt.to_backend_format.return_value = "Formatted prompt"
        self.phase_executor.story_prompt = mock_prompt

        self.phase_executor._phase_story_generate()

        assert self.phase_executor.story == "Generated story content"
        assert self.checkpoint_data.generated_content["story"] == "Generated story content"
        mock_backend.generate_story.assert_called_once()

    @patch("storyforge.phase_executor.Progress")
    @patch("storyforge.phase_executor.Confirm.ask")
    @patch("storyforge.phase_executor.console")
    def test_phase_story_generate_debug_mode(self, mock_console, mock_confirm, mock_progress):
        """Test _phase_story_generate in debug mode loads test story."""
        # Mock the refinement confirmation
        mock_confirm.return_value = False

        # Mock Progress context manager
        mock_progress_instance = MagicMock()
        mock_progress.return_value.__enter__ = MagicMock(return_value=mock_progress_instance)
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)

        self.checkpoint_data.resolved_config["debug"] = True

        # Mock the function that loads the debug story
        with patch("storyforge.StoryForge.load_story_from_file", return_value="Debug test story"):
            self.phase_executor._phase_story_generate()

        assert self.phase_executor.story == "Debug test story"

    @patch("storyforge.phase_executor.console")
    @patch("storyforge.phase_executor.Path")
    def test_phase_story_save(self, mock_path_class, mock_console):
        """Test _phase_story_save writes story to file."""
        self.phase_executor.story = "Test story content"
        self.checkpoint_data.resolved_config["output_directory"] = "/tmp/output"
        self.checkpoint_data.original_inputs["prompt"] = "A wizard's quest"

        # Mock Path operations
        mock_dir = MagicMock()
        mock_dir.exists.return_value = True
        mock_path_class.return_value = mock_dir

        written_content = None

        def capture_write(content):
            nonlocal written_content
            written_content = content

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.write = capture_write

        with patch("builtins.open", return_value=mock_file):
            self.phase_executor._phase_story_save()

        # Verify story was written
        assert written_content == "Test story content"

    @patch("storyforge.phase_executor.console")
    @patch("storyforge.phase_executor.Confirm.ask")
    @patch("storyforge.phase_executor.typer.prompt")
    def test_phase_image_decision_yes(self, mock_typer_prompt, mock_confirm, mock_console):
        """Test _phase_image_decision when user wants images."""
        mock_confirm.return_value = True
        mock_typer_prompt.return_value = 3
        self.checkpoint_data.resolved_config["continuation_mode"] = False

        self.phase_executor._phase_image_decision()

        assert self.checkpoint_data.user_decisions["wants_images"] is True
        assert self.checkpoint_data.user_decisions["num_images_requested"] == 3

    @patch("storyforge.phase_executor.console")
    @patch("storyforge.phase_executor.Confirm.ask")
    def test_phase_image_decision_no(self, mock_confirm, mock_console):
        """Test _phase_image_decision when user declines images."""
        mock_confirm.return_value = False
        self.checkpoint_data.resolved_config["continuation_mode"] = False

        self.phase_executor._phase_image_decision()

        assert self.checkpoint_data.user_decisions["wants_images"] is False
        # When user says no, num_images_requested is set to None
        num_requested = self.checkpoint_data.user_decisions.get("num_images_requested")
        assert num_requested is None or num_requested == 0

    @patch("storyforge.phase_executor.console")
    @patch("storyforge.phase_executor.Confirm.ask")
    def test_phase_image_decision_continuation_mode(self, mock_confirm, mock_console):
        """Test _phase_image_decision in continuation mode still asks user."""
        mock_confirm.return_value = False
        self.checkpoint_data.resolved_config["continuation_mode"] = True

        self.phase_executor._phase_image_decision()

        # Still asks in continuation mode, just doesn't force generation
        assert self.checkpoint_data.user_decisions["wants_images"] is False

    @patch("storyforge.phase_executor.Progress")
    @patch("storyforge.phase_executor.console")
    @patch("storyforge.phase_executor.Path")
    def test_phase_image_generate(self, mock_path_class, mock_console, mock_progress):
        """Test _phase_image_generate generates images."""
        # Mock Progress context manager
        mock_progress_instance = MagicMock()
        mock_progress.return_value.__enter__ = MagicMock(return_value=mock_progress_instance)
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)

        mock_backend = MagicMock()
        mock_backend.generate_image_prompt.return_value = ["prompt1", "prompt2"]
        # generate_image returns tuple of (image_object, image_bytes)
        mock_image = MagicMock()
        mock_image.format = "png"
        mock_backend.generate_image.return_value = (mock_image, b"fake_image_data")
        mock_backend.generate_image_name.return_value = "test_image"
        self.phase_executor.llm_backend = mock_backend
        self.phase_executor.story = "Test story"

        mock_prompt = MagicMock()
        self.phase_executor.story_prompt = mock_prompt

        self.checkpoint_data.user_decisions["wants_images"] = True  # Need this to be True
        self.checkpoint_data.user_decisions["num_images_requested"] = 2
        self.checkpoint_data.resolved_config["output_directory"] = "/tmp/output"

        # Mock Path operations
        mock_dir = MagicMock()
        mock_dir.exists.return_value = True
        mock_path_class.return_value.__truediv__ = MagicMock(return_value=mock_dir)
        mock_dir.parent.mkdir = MagicMock()

        with patch("builtins.open", MagicMock()):
            self.phase_executor._phase_image_generate()

        # Images are stored in generated_images
        assert len(self.checkpoint_data.generated_content.get("generated_images", [])) == 2
        mock_backend.generate_image_prompt.assert_called_once()

    @patch("storyforge.phase_executor.console")
    def test_phase_image_generate_skip_if_not_requested(self, mock_console):
        """Test _phase_image_generate skips when no images requested."""
        self.checkpoint_data.user_decisions["num_images_requested"] = 0

        initial_images = self.checkpoint_data.generated_content.get("images", [])
        self.phase_executor._phase_image_generate()

        # Images list should be unchanged
        assert self.checkpoint_data.generated_content.get("images", []) == initial_images

    @patch("storyforge.phase_executor.console")
    @patch("storyforge.phase_executor.Confirm.ask")
    @patch("storyforge.phase_executor.Path")
    def test_phase_context_save_with_parent_tracking(self, mock_path_class, mock_confirm, mock_console):
        """Test _phase_context_save adds Extended From metadata for chains."""
        mock_confirm.return_value = True
        self.phase_executor.story = "Test story content"
        self.checkpoint_data.original_inputs["prompt"] = "A wizard's quest"
        self.checkpoint_data.original_inputs["cli_arguments"] = {"characters": ["Wizard", "Dragon"]}
        self.checkpoint_data.resolved_config["source_context_file"] = "/path/to/parent_story.md"

        mock_dir = MagicMock()
        mock_dir.exists.return_value = True
        # Mock stem to return filename without .md
        mock_parent_path = MagicMock()
        mock_parent_path.stem = "parent_story"
        mock_path_class.return_value = mock_dir
        mock_path_class.side_effect = lambda p: mock_parent_path if "parent_story" in str(p) else mock_dir

        written_content = None

        def capture_write(content):
            nonlocal written_content
            written_content = content

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.write = capture_write

        with patch("builtins.open", return_value=mock_file):
            self.phase_executor._phase_context_save()

        # Verify Extended From field was added
        assert written_content is not None
        assert "**Extended From:**" in written_content
        assert "context_file" in self.checkpoint_data.generated_content

    @patch("storyforge.phase_executor.console")
    @patch("storyforge.phase_executor.Confirm.ask")
    @patch("storyforge.phase_executor.Path")
    def test_phase_context_save_without_parent(self, mock_path_class, mock_confirm, mock_console):
        """Test _phase_context_save without parent tracking (original story)."""
        mock_confirm.return_value = True
        self.phase_executor.story = "Original story content"
        self.checkpoint_data.original_inputs["prompt"] = "A wizard's quest"
        self.checkpoint_data.original_inputs["cli_arguments"] = {}
        self.checkpoint_data.resolved_config["source_context_file"] = None

        mock_dir = MagicMock()
        mock_dir.exists.return_value = True
        mock_path_class.return_value = mock_dir

        written_content = None

        def capture_write(content):
            nonlocal written_content
            written_content = content

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.write = capture_write

        with patch("builtins.open", return_value=mock_file):
            self.phase_executor._phase_context_save()

        # Verify Extended From field was NOT added
        assert written_content is not None
        assert "**Extended From:**" not in written_content

    @patch("storyforge.phase_executor.console")
    @patch("storyforge.phase_executor.Confirm.ask")
    def test_phase_context_save_user_declines(self, mock_confirm, mock_console):
        """Test _phase_context_save when user declines to save."""
        mock_confirm.return_value = False

        self.phase_executor._phase_context_save()

        assert self.checkpoint_data.user_decisions["save_as_context"] is False
        assert "context_file" not in self.checkpoint_data.generated_content

    @patch("storyforge.phase_executor.Progress")
    @patch("storyforge.phase_executor.Confirm.ask")
    @patch("storyforge.phase_executor.console")
    def test_error_handling_in_phase(self, mock_console, mock_confirm, mock_progress):
        """Test error handling when a phase fails."""
        mock_confirm.return_value = False  # Skip refinement

        # Mock Progress context manager
        mock_progress_instance = MagicMock()
        mock_progress.return_value.__enter__ = MagicMock(return_value=mock_progress_instance)
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)

        mock_backend = MagicMock()
        mock_backend.generate_story.side_effect = Exception("Backend error")
        self.phase_executor.llm_backend = mock_backend

        mock_prompt = MagicMock()
        self.phase_executor.story_prompt = mock_prompt

        with pytest.raises(Exception, match="Backend error"):
            self.phase_executor._phase_story_generate()


class TestDumpSessionContext:
    """Test the _dump_session_context method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.checkpoint_manager = MagicMock(spec=CheckpointManager)
        self.phase_executor = PhaseExecutor(self.checkpoint_manager)

        self.checkpoint_data = CheckpointData.create_new(
            "A brave fox story",
            {"style": "adventure", "age_range": "early_reader", "tone": "exciting"},
            {"backend": "gemini", "verbose": True, "debug": False},
        )
        self.phase_executor.checkpoint_data = self.checkpoint_data

    @patch("storyforge.phase_executor.console")
    def test_dump_creates_context_dump_file(self, mock_console, tmp_path):
        """Test that _dump_session_context creates context_dump.md."""
        output_dir = str(tmp_path)
        self.phase_executor.context = "Once upon a time, Finn the fox lived in the forest."
        self.phase_executor.story = "A great story about Finn."

        mock_prompt = MagicMock()
        mock_prompt.prompt = "A brave fox story"
        mock_prompt.length = "bedtime"
        mock_prompt.age_range = "early_reader"
        mock_prompt.style = "adventure"
        mock_prompt.tone = "exciting"
        mock_prompt.theme = "courage"
        mock_prompt.image_style = "chibi"
        mock_prompt.setting = None
        mock_prompt.characters = None
        mock_prompt.learning_focus = None
        mock_prompt.continuation_mode = False
        mock_prompt.story = "Full prompt text sent to LLM..."
        self.phase_executor.story_prompt = mock_prompt

        mock_backend = MagicMock()
        mock_backend.name = "gemini"
        self.phase_executor.llm_backend = mock_backend

        self.phase_executor._dump_session_context(output_dir)

        dump_path = tmp_path / "context_dump.md"
        assert dump_path.exists()

        content = dump_path.read_text()
        assert "# StoryForge Session Context Dump" in content
        assert "**Backend:** gemini" in content
        assert "A brave fox story" in content
        assert "Full prompt text sent to LLM..." in content
        assert "Finn the fox" in content
        assert "adventure" in content

    @patch("storyforge.phase_executor.console")
    def test_dump_without_context(self, mock_console, tmp_path):
        """Test dump when no context was loaded."""
        output_dir = str(tmp_path)
        self.phase_executor.context = None
        self.phase_executor.story = "A story."

        self.phase_executor._dump_session_context(output_dir)

        content = (tmp_path / "context_dump.md").read_text()
        assert "No context was loaded" in content

    @patch("storyforge.phase_executor.console")
    def test_dump_includes_config_table(self, mock_console, tmp_path):
        """Test dump includes resolved config as a table."""
        output_dir = str(tmp_path)
        # Update both CLI args and resolved config so CLI values take precedence
        self.checkpoint_data.original_inputs["cli_arguments"]["style"] = "fantasy"
        self.checkpoint_data.original_inputs["cli_arguments"]["tone"] = "magical"
        self.checkpoint_data.resolved_config["style"] = "fantasy"
        self.checkpoint_data.resolved_config["tone"] = "magical"
        self.phase_executor.context = None

        self.phase_executor._dump_session_context(output_dir)

        content = (tmp_path / "context_dump.md").read_text()
        assert "| Parameter | Value | Source |" in content
        assert "fantasy" in content
        assert "magical" in content

    @patch("storyforge.phase_executor.console")
    def test_dump_includes_config_file_details(self, mock_console, tmp_path):
        """Test dump includes config file details when config is loaded."""
        output_dir = str(tmp_path)
        self.phase_executor.context = None

        mock_config = MagicMock()
        mock_config.to_dict.return_value = {
            "story": {"style": "adventure", "length": "bedtime"},
            "system": {"backend": "gemini"},
        }
        self.phase_executor.config = mock_config
        self.checkpoint_data.resolved_config["config_path"] = "/home/user/.config/storyforge/storyforge.ini"

        self.phase_executor._dump_session_context(output_dir)

        content = (tmp_path / "context_dump.md").read_text()
        assert "## Config File" in content
        assert "storyforge.ini" in content
        assert "[story]" in content
        assert "style = adventure" in content

    @patch("storyforge.phase_executor.console")
    def test_dump_with_continuation_mode(self, mock_console, tmp_path):
        """Test dump shows continuation/ending_type when in continuation mode."""
        output_dir = str(tmp_path)
        self.phase_executor.context = None

        mock_prompt = MagicMock()
        mock_prompt.prompt = "Continue the story"
        mock_prompt.length = "short"
        mock_prompt.age_range = "preschool"
        mock_prompt.style = "fantasy"
        mock_prompt.tone = "magical"
        mock_prompt.theme = None
        mock_prompt.image_style = "watercolor"
        mock_prompt.setting = "enchanted forest"
        mock_prompt.characters = ["Finn", "Barnaby"]
        mock_prompt.learning_focus = "teamwork"
        mock_prompt.continuation_mode = True
        mock_prompt.ending_type = "cliffhanger"
        mock_prompt.story = "Continuation prompt..."
        self.phase_executor.story_prompt = mock_prompt

        self.phase_executor._dump_session_context(output_dir)

        content = (tmp_path / "context_dump.md").read_text()
        assert "continuation_mode" in content
        assert "True" in content
        assert "ending_type" in content
        assert "cliffhanger" in content
        assert "enchanted forest" in content
        assert "Finn, Barnaby" in content
        assert "teamwork" in content

    @patch("storyforge.phase_executor.console")
    def test_dump_not_called_without_verbose_or_debug(self, mock_console, tmp_path):
        """Test that _phase_story_save does NOT dump when verbose/debug are off."""
        self.checkpoint_data.resolved_config["verbose"] = False
        self.checkpoint_data.resolved_config["debug"] = False
        self.checkpoint_data.resolved_config["output_directory"] = str(tmp_path)
        self.checkpoint_data.original_inputs["prompt"] = "test"
        self.phase_executor.story = "A story."

        self.phase_executor._phase_story_save()

        assert not (tmp_path / "context_dump.md").exists()

    @patch("storyforge.phase_executor.console")
    def test_dump_called_with_verbose(self, mock_console, tmp_path):
        """Test that _phase_story_save dumps when --verbose is set."""
        self.checkpoint_data.resolved_config["verbose"] = True
        self.checkpoint_data.resolved_config["debug"] = False
        self.checkpoint_data.resolved_config["output_directory"] = str(tmp_path)
        self.checkpoint_data.original_inputs["prompt"] = "test"
        self.phase_executor.story = "A story."

        self.phase_executor._phase_story_save()

        assert (tmp_path / "context_dump.md").exists()

    @patch("storyforge.phase_executor.console")
    def test_dump_called_with_debug(self, mock_console, tmp_path):
        """Test that _phase_story_save dumps when --debug is set."""
        self.checkpoint_data.resolved_config["verbose"] = False
        self.checkpoint_data.resolved_config["debug"] = True
        self.checkpoint_data.resolved_config["output_directory"] = str(tmp_path)
        self.checkpoint_data.original_inputs["prompt"] = "test"
        self.phase_executor.story = "A story."

        self.phase_executor._phase_story_save()

        assert (tmp_path / "context_dump.md").exists()

    @patch("storyforge.phase_executor.console")
    def test_dump_handles_none_checkpoint(self, mock_console, tmp_path):
        """Test _dump_session_context exits gracefully with no checkpoint."""
        self.phase_executor.checkpoint_data = None
        # Should not raise
        self.phase_executor._dump_session_context(str(tmp_path))
        assert not (tmp_path / "context_dump.md").exists()
