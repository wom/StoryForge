import os
from unittest.mock import MagicMock, mock_open, patch

from typer.testing import CliRunner

from storytime.prompt import Prompt
from storytime.StoryCLI import app


class TestCLIIntegration:
    """Test CLI integration with the backend factory."""

    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()

    def test_cli_app_loads(self):
        """Test that the CLI app loads without errors."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "StoryTime CLI" in result.stdout

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True)
    @patch("storytime.gemini_backend.GeminiBackend")
    @patch(
        "rich.prompt.Confirm.ask", side_effect=[True, False]
    )  # Mock user confirmations: proceed=True, save_context=False
    @patch("os.makedirs")  # Mock directory creation
    @patch("builtins.open", new_callable=mock_open)
    def test_story_command_uses_factory(
        self, mock_open, mock_makedirs, mock_confirm, mock_gemini
    ):
        """Test that story command uses the backend factory with user confirmation."""
        # Mock the backend
        mock_backend = MagicMock()
        mock_backend.generate_story.return_value = "Test story content"
        mock_backend.generate_image.return_value = (None, None)
        mock_gemini.return_value = mock_backend

        result = self.runner.invoke(app, ["story", "test prompt"])

        # Should create backend via factory
        mock_gemini.assert_called_once()
        # Should ask for confirmation twice: once to proceed, once for context saving
        assert mock_confirm.call_count == 2
        # Now passes a Prompt object
        args, kwargs = mock_backend.generate_story.call_args
        assert len(args) >= 1  # At least prompt
        assert isinstance(args[0], Prompt)  # Should be a Prompt object
        assert args[0].prompt == "test prompt"  # Check the prompt text
        assert result.exit_code == 0
        assert "Test story content" in result.stdout
        # Should show generated directory
        assert "Generated output directory:" in result.stdout
        # Should show story saved message
        assert "Story saved as:" in result.stdout
        # Should call open to write story file
        mock_open.assert_called()

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True)
    @patch("storytime.gemini_backend.GeminiBackend")
    @patch("rich.prompt.Confirm.ask", return_value=True)  # Mock user confirmation
    @patch("os.makedirs")  # Mock directory creation
    def test_image_command_uses_factory(self, mock_makedirs, mock_confirm, mock_gemini):
        """Test that image command uses the backend factory with user confirmation."""
        # Mock the backend
        mock_backend = MagicMock()
        mock_backend.generate_image.return_value = (MagicMock(), b"fake_image_bytes")
        mock_backend.generate_image_name.return_value = "test_image"
        mock_gemini.return_value = mock_backend

        with patch("builtins.open", MagicMock()):
            result = self.runner.invoke(
                app, ["image", "test prompt", "--filename", "custom_name"]
            )

        # Should create backend via factory
        mock_gemini.assert_called_once()
        # Should ask for confirmation
        mock_confirm.assert_called_once()
        # Check that generate_image was called with a Prompt object
        args, kwargs = mock_backend.generate_image.call_args
        assert len(args) >= 1
        assert isinstance(args[0], Prompt)  # Should be a Prompt object
        assert args[0].prompt == "test prompt"  # Check the prompt text
        assert result.exit_code == 0
        assert "custom_name.png" in result.stdout
        # Should show generated directory
        assert "Generated output directory:" in result.stdout

    @patch.dict(os.environ, {}, clear=True)
    @patch("rich.prompt.Confirm.ask", return_value=True)  # Mock user confirmation
    def test_story_command_no_backend_error(self, mock_confirm):
        """Test story command fails gracefully when no backend available."""
        result = self.runner.invoke(app, ["story", "test prompt"])

        assert result.exit_code == 1
        # Should ask for confirmation before failing
        mock_confirm.assert_called_once()
        # Should show error about missing API key
        assert result.stdout != ""

    @patch.dict(os.environ, {}, clear=True)
    @patch("rich.prompt.Confirm.ask", return_value=True)  # Mock user confirmation
    def test_image_command_no_backend_error(self, mock_confirm):
        """Test image command fails gracefully when no backend available."""
        result = self.runner.invoke(app, ["image", "test prompt"])

        assert result.exit_code == 1
        # Should ask for confirmation before failing
        mock_confirm.assert_called_once()
        # Should show error about missing API key
        assert result.stdout != ""

    @patch.dict(
        os.environ, {"LLM_BACKEND": "gemini", "GEMINI_API_KEY": "test_key"}, clear=True
    )
    @patch("storytime.gemini_backend.GeminiBackend")
    @patch("rich.prompt.Confirm.ask", return_value=True)  # Mock user confirmation
    @patch("os.makedirs")  # Mock directory creation
    def test_explicit_backend_selection(self, mock_makedirs, mock_confirm, mock_gemini):
        """Test that explicit backend selection works via environment."""
        mock_backend = MagicMock()
        mock_backend.generate_image.return_value = (MagicMock(), b"fake_bytes")
        mock_backend.generate_image_name.return_value = "test_name"
        mock_gemini.return_value = mock_backend

        with patch("builtins.open", MagicMock()):
            result = self.runner.invoke(
                app, ["image", "test", "--filename", "explicit_test"]
            )

        mock_gemini.assert_called_once()
        mock_confirm.assert_called_once()
        assert result.exit_code == 0

    def test_cli_help_shows_all_commands(self):
        """Test that CLI help shows all available commands."""
        result = self.runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "story" in result.stdout
        assert "image" in result.stdout
        assert "StoryTime CLI" in result.stdout

    def test_image_command_help(self):
        """Test that image command help is correct."""
        result = self.runner.invoke(app, ["image", "--help"])

        assert result.exit_code == 0
        assert "Generate an image from a prompt" in result.stdout
        assert "--filename" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--verbose" in result.stdout

    @patch("rich.prompt.Confirm.ask", return_value=False)  # User declines
    def test_story_command_user_cancellation(self, mock_confirm):
        """Test that story command exits gracefully when user cancels."""
        result = self.runner.invoke(app, ["story", "test prompt"])

        # Should ask for confirmation
        mock_confirm.assert_called_once()
        # Should exit with code 0 (cancelled, not error)
        assert result.exit_code == 0
        # Should show cancellation message
        assert "Story generation cancelled" in result.stdout

    @patch("rich.prompt.Confirm.ask", return_value=False)  # User declines
    def test_image_command_user_cancellation(self, mock_confirm):
        """Test that image command exits gracefully when user cancels."""
        result = self.runner.invoke(app, ["image", "test prompt"])

        # Should ask for confirmation
        mock_confirm.assert_called_once()
        # Should exit with code 0 (cancelled, not error)
        assert result.exit_code == 0
        # Should show cancellation message
        assert "Image generation cancelled" in result.stdout

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True)
    @patch("storytime.gemini_backend.GeminiBackend")
    @patch(
        "rich.prompt.Confirm.ask", side_effect=[True, False]
    )  # proceed=True, save_context=False
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_story_auto_generated_directory(
        self, mock_open, mock_makedirs, mock_confirm, mock_gemini
    ):
        """Test that story command generates output directory automatically."""
        mock_backend = MagicMock()
        mock_backend.generate_story.return_value = "Test story"
        mock_backend.generate_image.return_value = (MagicMock(), b"fake_image_bytes")
        mock_backend.generate_image_name.return_value = "test_image"
        mock_gemini.return_value = mock_backend

        result = self.runner.invoke(app, ["story", "test prompt"])

        # Should show generated directory message
        assert "Generated output directory:" in result.stdout
        assert "storytime_output_" in result.stdout
        # Should call makedirs with the generated directory
        mock_makedirs.assert_called_once()
        assert result.exit_code == 0

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True)
    @patch("storytime.gemini_backend.GeminiBackend")
    @patch("rich.prompt.Confirm.ask", return_value=True)
    @patch("os.makedirs")
    def test_image_auto_generated_directory(
        self, mock_makedirs, mock_confirm, mock_gemini
    ):
        """Test that image command generates output directory automatically."""
        mock_backend = MagicMock()
        mock_backend.generate_image.return_value = (MagicMock(), b"fake_bytes")
        mock_backend.generate_image_name.return_value = "test_image"
        mock_gemini.return_value = mock_backend

        with patch("builtins.open", MagicMock()):
            result = self.runner.invoke(app, ["image", "test prompt"])

        # Should show generated directory message
        assert "Generated output directory:" in result.stdout
        assert "storytime_output_" in result.stdout
        # Should call makedirs with the generated directory
        mock_makedirs.assert_called_once()
        assert result.exit_code == 0

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True)
    @patch("storytime.gemini_backend.GeminiBackend")
    @patch(
        "rich.prompt.Confirm.ask", side_effect=[True, False]
    )  # proceed=True, save_context=False
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_story_with_explicit_output_dir(
        self, mock_open, mock_makedirs, mock_confirm, mock_gemini
    ):
        """Test that explicit output directory is used when provided."""
        mock_backend = MagicMock()
        mock_backend.generate_story.return_value = "Test story"
        mock_backend.generate_image.return_value = (MagicMock(), b"fake_image_bytes")
        mock_backend.generate_image_name.return_value = "test_image"
        mock_gemini.return_value = mock_backend

        result = self.runner.invoke(
            app, ["story", "test prompt", "--output-dir", "custom_dir"]
        )

        # Should NOT show generated directory message
        assert "Generated output directory:" not in result.stdout
        # Should call makedirs with the explicit directory
        mock_makedirs.assert_called_once_with("custom_dir", exist_ok=True)
        assert result.exit_code == 0

    def test_story_prompt_summary_format(self):
        """Test that story prompt summary shows expected format."""
        with patch("rich.prompt.Confirm.ask", return_value=False):
            result = self.runner.invoke(app, ["story", "test prompt"])

        # Should show summary with expected format
        assert "📋 Story Generation Summary:" in result.stdout
        assert "Prompt: test prompt" in result.stdout
        assert "Age Range:" in result.stdout
        assert "Length:" in result.stdout
        assert "Style:" in result.stdout
        assert "Tone:" in result.stdout

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True)
    @patch("storytime.gemini_backend.GeminiBackend")
    @patch(
        "rich.prompt.Confirm.ask", side_effect=[True, True]
    )  # proceed=True, save_context=True
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_story_context_saving(
        self, mock_open, mock_makedirs, mock_confirm, mock_gemini
    ):
        """Test that story command can save context when requested."""
        mock_backend = MagicMock()
        mock_backend.generate_story.return_value = "Test story content"
        mock_backend.generate_image.return_value = (MagicMock(), b"fake_image_bytes")
        mock_backend.generate_image_name.return_value = "test_image"
        mock_gemini.return_value = mock_backend

        result = self.runner.invoke(app, ["story", "test prompt"])

        # Should ask for confirmation twice: once to proceed, once for context saving
        assert mock_confirm.call_count == 2
        # Should show context saved message
        assert "Context saved as:" in result.stdout
        # Should create both main directory and context directory
        assert mock_makedirs.call_count >= 2
        assert result.exit_code == 0

    def test_image_prompt_summary_format(self):
        """Test that image prompt summary shows expected format."""
        with patch("rich.prompt.Confirm.ask", return_value=False):
            result = self.runner.invoke(app, ["image", "test prompt"])

        # Should show summary with expected format
        assert "📋 Image Generation Summary:" in result.stdout
        assert "Prompt: test prompt" in result.stdout
        assert "Age Range:" in result.stdout
        assert "Length:" in result.stdout
        assert "Style:" in result.stdout
        assert "Tone:" in result.stdout
