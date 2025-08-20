"""
Configuration management for StoryForge.

Handles loading, parsing, and merging configuration from:
1. Configuration files (INI format)
2. Environment variables
3. Command line arguments (highest priority)

Configuration file priority:
1. STORYFORGE_CONFIG environment variable path
2. XDG config directory: ~/.config/storyforge/storyforge.ini
3. Home directory: ~/.storyforge.ini
4. Current directory: ./storyforge.ini
"""

import os
from configparser import ConfigParser
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir
from rich.console import Console

# Import schema-driven components
from .schema.config_schema import STORYFORGE_SCHEMA
from .schema.validation import SchemaValidator

console = Console()

def _generate_default_config_template() -> str:
    """Generate default configuration template dynamically from schema."""
    lines = [
        "# StoryForge Configuration File",
        "# This file contains default values for story generation parameters",
        "# Command line arguments will override these settings",
        ""
    ]

    # Iterate through each section in the schema
    for section_name in ['story', 'images', 'output', 'system']:
        section = getattr(STORYFORGE_SCHEMA, section_name)
        lines.append(f"[{section_name}]")

        # Add each field in the section
        for field_name, field in section.fields.items():
            # Add comment from ini_comment or generate one
            if field.ini_comment:
                lines.append(f"# {field.ini_comment}")

            # Format default value based on field type
            default_value = field.default
            if field.field_type.value == "boolean":
                default_value = "true" if default_value else "false"
            elif field.field_type.value == "list":
                default_value = "" if not default_value else default_value
            else:
                default_value = str(default_value) if default_value is not None else ""

            lines.append(f"{field_name} = {default_value}")
            lines.append("")  # Empty line after each field

        lines.append("")  # Extra empty line after each section

    return "\n".join(lines).rstrip() + "\n"


# Generate default configuration template from schema
DEFAULT_CONFIG_TEMPLATE = _generate_default_config_template()

def _generate_default_config() -> dict[str, dict[str, Any]]:
    """Generate DEFAULT_CONFIG dictionary dynamically from schema."""
    config = {}

    for section_name in ['story', 'images', 'output', 'system']:
        section = getattr(STORYFORGE_SCHEMA, section_name)
        config[section_name] = {}

        for field_name, field in section.fields.items():
            config[section_name][field_name] = field.default

    return config


def _generate_valid_values_from_schema():
    """Generate VALID_VALUES dictionary from schema for backward compatibility."""
    valid_values = {}

    for section_name in ['story', 'images', 'output', 'system']:
        section = getattr(STORYFORGE_SCHEMA, section_name)
        for field_name, field in section.fields.items():
            if field.valid_values:
                valid_values[field_name] = field.valid_values

    return valid_values


# Generate default configuration values from schema
DEFAULT_CONFIG = _generate_default_config()

# Valid configuration values for validation (generated from schema)
VALID_VALUES = _generate_valid_values_from_schema()


class ConfigError(Exception):
    """Configuration related errors."""
    pass


class Config:
    """Configuration manager for StoryForge."""

    def __init__(self):
        self.config = ConfigParser()
        self.config_path: Path | None = None
        self.validator = SchemaValidator(STORYFORGE_SCHEMA)
        self._load_defaults()

    def _load_defaults(self):
        """Load default configuration values."""
        self.config.read_string(DEFAULT_CONFIG_TEMPLATE)

    def get_config_paths(self) -> list[Path]:
        """Return configuration file paths in priority order."""
        paths = []

        # 1. STORYFORGE_CONFIG environment variable (highest priority)
        env_config = os.environ.get("STORYFORGE_CONFIG")
        if env_config:
            paths.append(Path(env_config))

        # 2. XDG config directory
        paths.append(Path(user_config_dir("storyforge", "StoryForge")) / "storyforge.ini")

        # 3. Home directory fallback
        paths.append(Path.home() / ".storyforge.ini")

        # 4. Current directory fallback
        paths.append(Path("./storyforge.ini"))

        return paths

    def find_config_file(self) -> Path | None:
        """Find the first existing configuration file."""
        for path in self.get_config_paths():
            if path.exists() and path.is_file():
                return path
        return None

    def load_config(self, verbose: bool = False) -> bool:
        """
        Load configuration from file.
        
        Returns:
            bool: True if config file was found and loaded, False otherwise.
        """
        config_path = self.find_config_file()
        if not config_path:
            if verbose:
                console.print("[dim]No configuration file found, using defaults[/dim]")
            return False

        try:
            self.config.read(config_path)
            self.config_path = config_path
            if verbose:
                console.print(f"[dim]Loaded configuration from: {config_path}[/dim]")
            return True
        except Exception as e:
            raise ConfigError(f"Error reading configuration file {config_path}: {e}") from e

    def validate_config(self) -> list[str]:
        """
        Validate configuration values using schema.
        
        Returns:
            List[str]: List of validation errors, empty if valid.
        """
        try:
            config_dict = self.to_dict()
            validation_errors = self.validator.validate_config(config_dict)
            return [str(error) for error in validation_errors]
        except Exception as e:
            return [f"Configuration validation error: {e}"]

    def get_default_config_path(self) -> Path:
        """Get the default configuration file path (XDG config directory)."""
        return Path(user_config_dir("storyforge", "StoryForge")) / "storyforge.ini"

    def create_default_config(self, path: Path | None = None) -> Path:
        """
        Create a default configuration file.
        
        Args:
            path: Path to create config file. If None, uses default location.
            
        Returns:
            Path: The path where the config file was created.
        """
        if path is None:
            path = self.get_default_config_path()

        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write default configuration
        with open(path, 'w', encoding='utf-8') as f:
            f.write(DEFAULT_CONFIG_TEMPLATE)

        return path

    def get_value(self, section: str, key: str, fallback: str = "") -> str:
        """Get configuration value with fallback."""
        try:
            value = self.config.get(section, key, fallback=fallback)
            return value.strip() if value else ""
        except Exception:
            return fallback

    def get_bool(self, section: str, key: str, fallback: bool = False) -> bool:
        """Get boolean configuration value with fallback."""
        try:
            return self.config.getboolean(section, key, fallback=fallback)
        except Exception:
            return fallback

    def get_list(self, section: str, key: str, fallback: list[str] | None = None) -> list[str] | None:
        """Get list configuration value (comma-separated) with fallback."""
        if fallback is None:
            fallback = []

        try:
            value = self.get_value(section, key)
            if not value:
                return None if not fallback else fallback

            # Split by comma and strip whitespace
            return [item.strip() for item in value.split(',') if item.strip()]
        except Exception:
            return fallback if fallback else None

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Convert configuration to dictionary format."""
        result = {}
        for section_name in self.config.sections():
            result[section_name] = {}
            for key, value in self.config[section_name].items():
                # Skip comment keys
                if not key.startswith('#'):
                    result[section_name][key] = value
        return result


def load_config(verbose: bool = False) -> Config:
    """
    Load configuration from file system.
    
    Args:
        verbose: Enable verbose output
        
    Returns:
        Config: Loaded configuration object
        
    Raises:
        ConfigError: If configuration file is malformed
    """
    config = Config()
    config.load_config(verbose=verbose)

    # Validate configuration
    errors = config.validate_config()
    if errors:
        raise ConfigError(
            "Configuration validation failed:\n" +
            "\n".join(f"  - {error}" for error in errors)
        )

    return config
