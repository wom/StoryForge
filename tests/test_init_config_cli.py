"""Tests for the config init CLI command."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from storyforge.StoryForge import app

runner = CliRunner()


@pytest.fixture
def temp_config_path(tmp_path):
    """Provide a temporary config path for testing."""
    return tmp_path / "storyforge.ini"


def test_init_config_creates_file(temp_config_path, monkeypatch):
    """Test that config init creates a config file when it doesn't exist."""
    # Mock get_default_config_path to return our temp path
    with patch("storyforge.StoryForge.Config") as MockConfig:
        mock_instance = MockConfig.return_value
        mock_instance.get_default_config_path.return_value = temp_config_path
        mock_instance.create_default_config.return_value = temp_config_path
        mock_instance.get_config_paths.return_value = [temp_config_path]

        result = runner.invoke(app, ["config", "init"])

        assert result.exit_code == 0
        assert "Configuration file created" in result.stdout
        assert str(temp_config_path) in result.stdout
        mock_instance.create_default_config.assert_called_once()


def test_init_config_does_not_overwrite_existing_file(temp_config_path, monkeypatch):
    """Test that config init does not overwrite an existing config file."""
    # Create an existing config file
    temp_config_path.write_text("[story]\nlength = short\n")

    with patch("storyforge.StoryForge.Config") as MockConfig:
        mock_instance = MockConfig.return_value
        mock_instance.get_default_config_path.return_value = temp_config_path
        mock_instance.get_config_paths.return_value = [temp_config_path]

        result = runner.invoke(app, ["config", "init"])

        assert result.exit_code == 0
        assert "already exists" in result.stdout
        assert "Use --force to overwrite" in result.stdout
        # Verify the file was not overwritten
        assert temp_config_path.read_text() == "[story]\nlength = short\n"


def test_init_config_force_overwrites_existing_file(temp_config_path, monkeypatch):
    """Test that config init --force overwrites an existing config file."""
    # Create an existing config file
    temp_config_path.write_text("[story]\nlength = short\n")

    with patch("storyforge.StoryForge.Config") as MockConfig:
        mock_instance = MockConfig.return_value
        mock_instance.get_default_config_path.return_value = temp_config_path
        mock_instance.create_default_config.return_value = temp_config_path
        mock_instance.get_config_paths.return_value = [temp_config_path]

        result = runner.invoke(app, ["config", "init", "--force"])

        assert result.exit_code == 0
        assert "Configuration file created" in result.stdout
        mock_instance.create_default_config.assert_called_once()


def test_init_config_creates_parent_directory(tmp_path, monkeypatch):
    """Test that config init creates parent directories if they don't exist."""
    nested_path = tmp_path / "nested" / "dir" / "storyforge.ini"

    with patch("storyforge.StoryForge.Config") as MockConfig:
        mock_instance = MockConfig.return_value
        mock_instance.get_default_config_path.return_value = nested_path
        mock_instance.create_default_config.return_value = nested_path
        mock_instance.get_config_paths.return_value = [nested_path]

        result = runner.invoke(app, ["config", "init"])

        assert result.exit_code == 0
        assert nested_path.parent.exists()


def test_init_config_error_handling(temp_config_path, monkeypatch):
    """Test that config init handles errors gracefully."""
    with patch("storyforge.StoryForge.Config") as MockConfig:
        mock_instance = MockConfig.return_value
        mock_instance.get_default_config_path.return_value = temp_config_path
        mock_instance.create_default_config.side_effect = OSError("Permission denied")

        result = runner.invoke(app, ["config", "init", "--force"])

        assert result.exit_code == 1
        assert "Error creating configuration file" in result.stdout
