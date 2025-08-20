"""
Core infrastructure for StoryForge configuration schema.
Provides base classes and utilities for defining configuration fields with rich metadata.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FieldType(Enum):
    """Supported configuration field types."""

    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    LIST = "list"
    PATH = "path"


@dataclass
class ConfigField:
    """
    Comprehensive metadata for a single configuration field.

    This class stores all information needed to:
    - Validate configuration values
    - Generate CLI options automatically
    - Create configuration file templates
    - Provide user documentation
    """

    name: str
    field_type: FieldType
    default: Any
    section: str
    description: str
    cli_help: str

    # Validation
    valid_values: list[str] | None = None
    required: bool = False
    validator: Callable[[Any], bool] | None = None

    # CLI Integration
    cli_short: str | None = None
    cli_long: str | None = None
    cli_flag_name: str | None = None

    # Documentation
    example_values: list[str] | None = None
    added_version: str | None = None
    deprecated: bool = False
    deprecation_message: str | None = None

    # INI file generation
    ini_comment: str | None = None
    ini_group_comment: str | None = None

    def __post_init__(self):
        """Generate derived fields after initialization."""
        if self.cli_flag_name is None:
            self.cli_flag_name = self.name.replace("_", "-")

        if self.cli_long is None:
            self.cli_long = f"--{self.cli_flag_name}"

        if self.ini_comment is None:
            comment_parts = []
            if self.description:
                comment_parts.append(self.description)
            if self.valid_values:
                comment_parts.append(f"Options: {', '.join(self.valid_values)}")
            if self.example_values:
                comment_parts.append(f"Examples: {', '.join(self.example_values)}")
            self.ini_comment = " | ".join(comment_parts) if comment_parts else ""


@dataclass
class ConfigSection:
    """Base class for configuration sections."""

    name: str
    description: str
    fields: dict[str, ConfigField] = field(default_factory=dict)

    def get_field(self, name: str) -> ConfigField | None:
        """Get a field by name."""
        return self.fields.get(name)

    def add_field(self, field_obj: ConfigField) -> None:
        """Add a field to this section."""
        field_obj.section = self.name
        self.fields[field_obj.name] = field_obj
