import os
from unittest.mock import MagicMock, patch

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
    def test_story_command_uses_factory(self, mock_gemini):
        """Test that story command uses the backend factory."""
        # Mock the backend
        mock_backend = MagicMock()
        mock_backend.generate_story.return_value = "Test story content"
        mock_backend.generate_image.return_value = (None, None)
        mock_gemini.return_value = mock_backend

        result = self.runner.invoke(app, ["story", "test prompt"])

        # Should create backend via factory
        mock_gemini.assert_called_once()
        # Now passes a Prompt object
        args, kwargs = mock_backend.generate_story.call_args
        assert len(args) >= 1  # At least prompt
        assert isinstance(args[0], Prompt)  # Should be a Prompt object
        assert args[0].prompt == "test prompt"  # Check the prompt text
        assert result.exit_code == 0
        assert "Test story content" in result.stdout

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True)
    @patch("storytime.gemini_backend.GeminiBackend")
    def test_image_command_uses_factory(self, mock_gemini):
        """Test that image command uses the backend factory."""
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
        # Check that generate_image was called with a Prompt object
        args, kwargs = mock_backend.generate_image.call_args
        assert len(args) >= 1
        assert isinstance(args[0], Prompt)  # Should be a Prompt object
        assert args[0].prompt == "test prompt"  # Check the prompt text
        assert result.exit_code == 0
        assert "custom_name.png" in result.stdout

    @patch.dict(os.environ, {}, clear=True)
    def test_story_command_no_backend_error(self):
        """Test story command fails gracefully when no backend available."""
        result = self.runner.invoke(app, ["story", "test prompt"])

        assert result.exit_code == 1
        # Should show error about missing API key
        assert result.stdout != ""

    @patch.dict(os.environ, {}, clear=True)
    def test_image_command_no_backend_error(self):
        """Test image command fails gracefully when no backend available."""
        result = self.runner.invoke(app, ["image", "test prompt"])

        assert result.exit_code == 1
        # Should show error about missing API key
        assert result.stdout != ""

    @patch.dict(
        os.environ, {"LLM_BACKEND": "gemini", "GEMINI_API_KEY": "test_key"}, clear=True
    )
    @patch("storytime.gemini_backend.GeminiBackend")
    def test_explicit_backend_selection(self, mock_gemini):
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
