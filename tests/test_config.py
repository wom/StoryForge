"""Comprehensive tests for config.py module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from storyforge.config import Config, ConfigError, load_config


class TestConfigPathResolution:
    """Test configuration file path resolution priority."""

    def test_config_paths_priority_order(self, tmp_path, monkeypatch):
        """Test that config paths are returned in correct priority order."""
        # Set up environment
        env_config = tmp_path / "env_config.ini"
        monkeypatch.setenv("STORYFORGE_CONFIG", str(env_config))

        config = Config()
        paths = config.get_config_paths()

        # Verify priority order
        assert len(paths) >= 4
        assert paths[0] == env_config  # Env var has highest priority
        assert paths[-1] == Path("./storyforge.ini")  # Current dir is lowest

    def test_find_config_file_env_priority(self, tmp_path, monkeypatch):
        """Test that STORYFORGE_CONFIG env var takes priority."""
        # Create configs
        env_config = tmp_path / "env.ini"
        env_config.write_text("[story]\nlength = long\n")

        home_config = tmp_path / ".storyforge.ini"
        home_config.write_text("[story]\nlength = short\n")

        # Mock paths
        monkeypatch.setenv("STORYFORGE_CONFIG", str(env_config))

        with patch.object(Path, "home", return_value=tmp_path):
            config = Config()
            found = config.find_config_file()

            assert found == env_config

    def test_find_config_file_returns_none_when_missing(self):
        """Test that find_config_file returns None when no config exists."""
        config = Config()

        with patch.object(config, "get_config_paths", return_value=[Path("/nonexistent/config.ini")]):
            found = config.find_config_file()
            assert found is None


class TestConfigLoading:
    """Test configuration loading functionality."""

    def test_load_config_success(self, tmp_path):
        """Test successfully loading a config file."""
        config_file = tmp_path / "test.ini"
        config_file.write_text("[story]\nlength = medium\nstyle = fantasy\n")

        config = Config()

        with patch.object(config, "find_config_file", return_value=config_file):
            result = config.load_config(verbose=False)

            assert result is True
            assert config.config_path == config_file

    def test_load_config_not_found(self):
        """Test loading config when no file exists."""
        config = Config()

        with patch.object(config, "find_config_file", return_value=None):
            result = config.load_config(verbose=False)

            assert result is False
            assert config.config_path is None

    def test_load_config_malformed_file(self, tmp_path):
        """Test loading a malformed config file raises ConfigError."""
        config_file = tmp_path / "bad.ini"
        config_file.write_text("[story\nthis is not valid ini")

        config = Config()

        with patch.object(config, "find_config_file", return_value=config_file):
            with pytest.raises(ConfigError):
                config.load_config()

    def test_load_config_verbose_output(self, tmp_path, capsys):
        """Test that verbose mode prints loading messages."""
        config_file = tmp_path / "test.ini"
        config_file.write_text("[story]\nlength = short\n")

        config = Config()

        with patch.object(config, "find_config_file", return_value=config_file):
            config.load_config(verbose=True)
            captured = capsys.readouterr()
            assert "Loaded configuration from" in captured.out


class TestFieldValueConversion:
    """Test field value type conversion."""

    def test_get_field_value_string(self, tmp_path):
        """Test getting string field values."""
        config_file = tmp_path / "test.ini"
        config_file.write_text("[story]\nstyle = fantasy\n")

        config = Config()

        with patch.object(config, "find_config_file", return_value=config_file):
            config.load_config()
            value = config.get_field_value("story", "style")
            assert value == "fantasy"
            assert isinstance(value, str)

    def test_get_field_value_boolean_true(self, tmp_path):
        """Test getting boolean field values (true variants)."""
        config_file = tmp_path / "test.ini"
        config_file.write_text("[system]\nverbose = true\n")

        config = Config()

        with patch.object(config, "find_config_file", return_value=config_file):
            config.load_config()
            value = config.get_field_value("system", "verbose")
            assert value is True
            assert isinstance(value, bool)

    def test_get_field_value_boolean_false(self, tmp_path):
        """Test getting boolean field values (false variants)."""
        config_file = tmp_path / "test.ini"
        config_file.write_text("[system]\nverbose = false\n")

        config = Config()

        with patch.object(config, "find_config_file", return_value=config_file):
            config.load_config()
            value = config.get_field_value("system", "verbose")
            assert value is False

    def test_get_field_value_integer(self, tmp_path):
        """Test getting integer field values."""
        config_file = tmp_path / "test.ini"
        config_file.write_text("[images]\nimage_count = 5\n")

        config = Config()

        with patch.object(config, "find_config_file", return_value=config_file):
            config.load_config()
            value = config.get_field_value("images", "image_count")
            assert value == 5
            assert isinstance(value, int)

    def test_get_field_value_list_comma_separated(self, tmp_path):
        """Test getting list field values from comma-separated string."""
        config_file = tmp_path / "test.ini"
        config_file.write_text("[story]\ncharacters = Alice, Bob, Charlie\n")

        config = Config()

        with patch.object(config, "find_config_file", return_value=config_file):
            config.load_config()
            value = config.get_field_value("story", "characters")
            assert value == ["Alice", "Bob", "Charlie"]
            assert isinstance(value, list)

    def test_get_field_value_list_empty(self, tmp_path):
        """Test getting empty list field values."""
        config_file = tmp_path / "test.ini"
        config_file.write_text("[story]\ncharacters = \n")

        config = Config()

        with patch.object(config, "find_config_file", return_value=config_file):
            config.load_config()
            value = config.get_field_value("story", "characters")
            assert value == []

    def test_get_field_value_returns_default_on_error(self):
        """Test that get_field_value returns schema default on error."""
        config = Config()

        # Invalid section/field combination should return default
        value = config.get_field_value("story", "nonexistent_field")
        assert value is None


class TestConfigValidation:
    """Test configuration validation."""

    def test_validate_config_valid(self, tmp_path):
        """Test validation passes for valid config."""
        config_file = tmp_path / "test.ini"
        config_file.write_text(
            "[story]\nlength = short\nstyle = fantasy\n[images]\nimage_style = chibi\n[output]\nverbose = true\n"
        )

        config = Config()

        with patch.object(config, "find_config_file", return_value=config_file):
            config.load_config()
            errors = config.validate_config()
            assert errors == []

    def test_validate_config_with_errors(self, tmp_path):
        """Test validation detects invalid values."""
        config_file = tmp_path / "test.ini"
        # Use invalid values that would fail schema validation
        config_file.write_text("[story]\nlength = invalid_length\n")

        config = Config()

        with patch.object(config, "find_config_file", return_value=config_file):
            config.load_config()
            errors = config.validate_config()
            # Should have validation errors for invalid length value
            assert len(errors) > 0


class TestConfigCreation:
    """Test configuration file creation."""

    def test_create_default_config(self, tmp_path):
        """Test creating default config file."""
        config = Config()
        config_path = tmp_path / "test.ini"

        created_path = config.create_default_config(config_path)

        assert created_path == config_path
        assert config_path.exists()
        assert config_path.is_file()

        # Verify content
        content = config_path.read_text()
        assert "[story]" in content
        assert "[images]" in content
        assert "[output]" in content
        assert "[system]" in content

    def test_create_default_config_creates_parent_dirs(self, tmp_path):
        """Test that creating config creates parent directories."""
        config = Config()
        config_path = tmp_path / "nested" / "dir" / "config.ini"

        created_path = config.create_default_config(config_path)

        assert created_path.parent.exists()
        assert created_path.exists()

    def test_get_default_config_path(self):
        """Test getting default config path."""
        config = Config()
        path = config.get_default_config_path()

        assert path.name == "storyforge.ini"
        assert "storyforge" in str(path).lower()


class TestConfigToDict:
    """Test configuration to dictionary conversion."""

    def test_to_dict_conversion(self, tmp_path):
        """Test converting config to dictionary."""
        config_file = tmp_path / "test.ini"
        config_file.write_text("[story]\nlength = short\nstyle = fantasy\n[images]\nimage_style = chibi\n")

        config = Config()

        with patch.object(config, "find_config_file", return_value=config_file):
            config.load_config()
            config_dict = config.to_dict()

            assert "story" in config_dict
            assert "images" in config_dict
            assert config_dict["story"]["length"] == "short"
            assert config_dict["story"]["style"] == "fantasy"
            assert config_dict["images"]["image_style"] == "chibi"

    def test_to_dict_excludes_comments(self, tmp_path):
        """Test that to_dict excludes comment keys."""
        config_file = tmp_path / "test.ini"
        config_file.write_text("[story]\n# This is a comment\nlength = short\n")

        config = Config()

        with patch.object(config, "find_config_file", return_value=config_file):
            config.load_config()
            config_dict = config.to_dict()

            # Ensure no comment keys in output
            for section_values in config_dict.values():
                for key in section_values.keys():
                    assert not key.startswith("#")


class TestLoadConfigFunction:
    """Test the load_config convenience function."""

    def test_load_config_function_success(self, tmp_path):
        """Test load_config function loads successfully."""
        config_file = tmp_path / "test.ini"
        config_file.write_text("[story]\nlength = short\n")

        with patch("storyforge.config.Config") as MockConfig:
            mock_instance = MockConfig.return_value
            mock_instance.load_config.return_value = True
            mock_instance.validate_config.return_value = []

            config = load_config(verbose=False)

            assert config is not None
            mock_instance.load_config.assert_called_once_with(verbose=False)
            mock_instance.validate_config.assert_called_once()

    def test_load_config_function_validation_error(self, tmp_path):
        """Test load_config function raises ConfigError on validation failure."""
        with patch("storyforge.config.Config") as MockConfig:
            mock_instance = MockConfig.return_value
            mock_instance.load_config.return_value = True
            mock_instance.validate_config.return_value = ["Error 1", "Error 2"]

            with pytest.raises(ConfigError) as exc_info:
                load_config()

            assert "Configuration validation failed" in str(exc_info.value)
            assert "Error 1" in str(exc_info.value)
            assert "Error 2" in str(exc_info.value)
