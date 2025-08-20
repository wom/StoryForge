"""
CLI integration utilities for automatic option generation from schema.
"""

from typing import Any

import typer

from .config_schema import STORYFORGE_SCHEMA
from .validation import SchemaValidator


def get_field_by_name(field_name: str):
    """Get a field by name from any section."""
    for section_name in ["story", "images", "output", "system"]:
        section = getattr(STORYFORGE_SCHEMA, section_name)
        if field_name in section.fields:
            return section.fields[field_name]
    return None


def generate_cli_option(field_name: str):
    """Generate a single CLI option from schema field."""
    field = get_field_by_name(field_name)
    if not field:
        raise ValueError(f"Field '{field_name}' not found in schema")

    # Create option arguments - ensure we always have cli_long
    option_args = []

    # Use cli_long if available, otherwise generate from field name
    cli_long = field.cli_long if hasattr(field, "cli_long") and field.cli_long else f"--{field.name.replace('_', '-')}"
    option_args.append(cli_long)

    # Add cli_short if available
    if hasattr(field, "cli_short") and field.cli_short:
        option_args.append(field.cli_short)

    # Create typer option
    return typer.Option(None, *option_args, help=field.cli_help)


def generate_boolean_cli_option(field_name: str, flag_format: str | None = None):
    """Generate a boolean CLI option with proper flag format."""
    field = get_field_by_name(field_name)
    if not field:
        raise ValueError(f"Field '{field_name}' not found in schema")

    if flag_format:
        # Use custom flag format (like --use-context/--no-use-context)
        return typer.Option(None, flag_format, help=field.cli_help)
    else:
        # Standard boolean flag
        option_args = []
        # Use cli_long if available, otherwise generate from field name
        cli_long = (
            field.cli_long if hasattr(field, "cli_long") and field.cli_long else f"--{field.name.replace('_', '-')}"
        )
        option_args.append(cli_long)

        # Add cli_short if available
        if hasattr(field, "cli_short") and field.cli_short:
            option_args.append(field.cli_short)
        return typer.Option(None, *option_args, help=field.cli_help)


def generate_multi_option(field_name: str, option_name: str | None = None):
    """Generate a multi-use CLI option from schema field."""
    field = get_field_by_name(field_name)
    if not field:
        raise ValueError(f"Field '{field_name}' not found in schema")

    # Use provided option name or generate from field name
    if option_name:
        return typer.Option(option_name, help=field.cli_help)
    else:
        # Generate option name from field name (singular form for multi-use)
        option_flag = f"--{field.name.rstrip('s').replace('_', '-')}"
        return typer.Option(option_flag, help=field.cli_help)


def generate_typer_options() -> dict[str, Any]:
    """
    Generate typer.Option definitions from schema.
    Returns a dictionary that can be used to dynamically create CLI commands.
    """
    options = {}

    for section_name in ["story", "images", "output", "system"]:
        section = getattr(STORYFORGE_SCHEMA, section_name)
        for _field_name, field in section.fields.items():
            # Determine the Python type for typer
            python_type: type[str] | type[bool] | type[int] | type[list[str]] = str  # Default to string
            if field.field_type.value == "boolean":
                python_type = bool
            elif field.field_type.value == "integer":
                python_type = int
            elif field.field_type.value == "list":
                python_type = list[str]

            # Create typer option
            option_kwargs = {
                "default": None,  # Always allow CLI override
                "help": field.cli_help,
            }

            if field.cli_short:
                option_args = [field.cli_long, field.cli_short]
            else:
                option_args = [field.cli_long]

            options[_field_name] = (python_type | None, typer.Option(None, *option_args, **option_kwargs))

    return options


def validate_cli_arguments(**kwargs) -> list[str]:
    """
    Validate CLI arguments using schema.

    Args:
        **kwargs: CLI argument values

    Returns:
        List of validation error messages
    """
    validator = SchemaValidator(STORYFORGE_SCHEMA)
    errors = []

    for field_name, value in kwargs.items():
        if value is not None:  # Only validate provided arguments
            field_errors = validator.validate_cli_argument(field_name, value)
            errors.extend([str(error) for error in field_errors])

    return errors


def get_field_choices(field_name: str) -> list[str] | None:
    """
    Get valid choices for a field from schema.

    Args:
        field_name: The field name to get choices for

    Returns:
        List of valid choices or None if no restrictions
    """
    for section_name in ["story", "images", "output", "system"]:
        section = getattr(STORYFORGE_SCHEMA, section_name)
        if field_name in section.fields:
            field = section.fields[field_name]
            return field.valid_values  # type: ignore[no-any-return]
    return None


def get_field_help(field_name: str) -> str | None:
    """
    Get help text for a field from schema.

    Args:
        field_name: The field name to get help for

    Returns:
        Help text or None if field not found
    """
    for section_name in ["story", "images", "output", "system"]:
        section = getattr(STORYFORGE_SCHEMA, section_name)
        if field_name in section.fields:
            field = section.fields[field_name]
            return field.cli_help  # type: ignore[no-any-return]
    return None


def generate_help_text() -> str:
    """Generate comprehensive help text from schema."""
    help_sections = []

    for section_name in ["story", "images", "output", "system"]:
        section = getattr(STORYFORGE_SCHEMA, section_name)

        section_help = [f"\n[bold]{section.name.upper()} OPTIONS[/bold]"]
        section_help.append(section.description)
        section_help.append("")

        for _field_name, field in section.fields.items():
            field_help = f"  {field.cli_long}"
            if field.cli_short:
                field_help += f", {field.cli_short}"

            field_help += f": {field.cli_help}"

            if field.valid_values and len(field.valid_values) <= 6:
                field_help += f" (choices: {', '.join(field.valid_values)})"

            section_help.append(field_help)

        help_sections.append("\n".join(section_help))

    return "\n".join(help_sections)
