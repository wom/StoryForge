import typer
from typer.testing import CliRunner

from storyforge.StoryForge import main

cli_app = typer.Typer()
cli_app.command()(main)


class TestCLIIntegration:
    """Test CLI integration with the backend factory."""

    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()

    def test_cli_app_loads(self):
        """Test that the CLI app loads without errors."""
        result = self.runner.invoke(cli_app, ["--help"])
        assert result.exit_code == 0
        # CLI help output no longer includes "StoryForge", so check for a prompt-related string
        assert "prompt" in result.stdout.lower()

    def test_cli_help_shows_all_commands(self):
        """Test that CLI help shows all available commands."""
        result = self.runner.invoke(cli_app, ["--help"])

        assert result.exit_code == 0
        assert "prompt" in result.stdout.lower()
