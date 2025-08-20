"""
Validation framework for configuration schema.
Provides comprehensive validation with clear error messages.
"""

from typing import Any

from .core import ConfigField, FieldType


class ValidationError(Exception):
    """Configuration validation error with context."""

    def __init__(self, field_name: str, value: Any, message: str, section: str = ""):
        self.field_name = field_name
        self.value = value
        self.message = message
        self.section = section
        super().__init__(f"[{section}.{field_name}] {message}")


class SchemaValidator:
    """Comprehensive configuration validator using schema definitions."""

    def __init__(self, schema):
        self.schema = schema

    def validate_field(self, field: ConfigField, value: Any) -> list[ValidationError]:
        """Validate a single field value against its schema definition."""
        errors = []

        # Required field check
        if field.required and (value is None or value == ""):
            errors.append(ValidationError(field.name, value, "Required field cannot be empty", field.section))
            return errors

        # Skip further validation for empty optional fields
        if not field.required and (value is None or value == ""):
            return errors

        # Type validation
        type_valid, type_error = self._validate_type(field, value)
        if not type_valid:
            errors.append(ValidationError(field.name, value, type_error, field.section))
            return errors  # Don't continue if type is wrong

        # Valid values check
        if field.valid_values and str(value) not in field.valid_values:
            errors.append(
                ValidationError(
                    field.name,
                    value,
                    f"Invalid value '{value}'. Valid options: {', '.join(field.valid_values)}",
                    field.section,
                )
            )

        # Custom validator
        if field.validator and not field.validator(value):
            errors.append(
                ValidationError(field.name, value, f"Custom validation failed for value '{value}'", field.section)
            )

        return errors

    def _validate_type(self, field: ConfigField, value: Any) -> tuple[bool, str]:
        """Validate field type with comprehensive checking."""
        if field.field_type == FieldType.STRING:
            if not isinstance(value, str):
                return False, f"Expected string, got {type(value).__name__}"

        elif field.field_type == FieldType.INTEGER:
            try:
                int(value)
            except (ValueError, TypeError):
                return False, f"Expected integer, got '{value}'"

        elif field.field_type == FieldType.BOOLEAN:
            if not isinstance(value, bool) and str(value).lower() not in ["true", "false", "1", "0", "yes", "no"]:
                return False, f"Expected boolean, got '{value}'"

        elif field.field_type == FieldType.LIST:
            if not isinstance(value, list | str):  # Allow comma-separated strings
                return False, f"Expected list or comma-separated string, got {type(value).__name__}"

        elif field.field_type == FieldType.PATH:
            if not isinstance(value, str):
                return False, f"Expected path string, got {type(value).__name__}"

        return True, ""

    def validate_config(self, config_dict: dict[str, dict[str, Any]]) -> list[ValidationError]:
        """Validate entire configuration dictionary."""
        all_errors = []

        for section_name, section_config in config_dict.items():
            if hasattr(self.schema, section_name):
                section = getattr(self.schema, section_name)
                for field_name, value in section_config.items():
                    if field_name in section.fields:
                        field = section.fields[field_name]
                        field_errors = self.validate_field(field, value)
                        all_errors.extend(field_errors)

        return all_errors

    def validate_cli_argument(self, field_name: str, value: Any) -> list[ValidationError]:
        """Validate a single CLI argument by field name."""
        # Find the field across all sections
        for section_name in ["story", "images", "output", "system"]:
            if hasattr(self.schema, section_name):
                section = getattr(self.schema, section_name)
                if field_name in section.fields:
                    field = section.fields[field_name]
                    return self.validate_field(field, value)

        # Field not found in schema
        return [ValidationError(field_name, value, f"Unknown configuration field '{field_name}'", "")]
