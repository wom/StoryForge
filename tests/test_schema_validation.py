"""Comprehensive tests for schema validation module."""

from storyforge.schema.config_schema import STORYFORGE_SCHEMA
from storyforge.schema.core import ConfigField, FieldType
from storyforge.schema.validation import SchemaValidator, ValidationError


class TestValidationError:
    """Test ValidationError class."""

    def test_validation_error_creation(self):
        """Test creating a validation error."""
        error = ValidationError("length", "invalid", "Invalid length value", "story")

        assert error.field_name == "length"
        assert error.value == "invalid"
        assert error.message == "Invalid length value"
        assert error.section == "story"
        assert "[story.length]" in str(error)
        assert "Invalid length value" in str(error)

    def test_validation_error_string_representation(self):
        """Test error string formatting."""
        error = ValidationError("age_range", "999", "Invalid age range", "story")
        error_str = str(error)

        assert "story.age_range" in error_str
        assert "Invalid age range" in error_str


class TestSchemaValidator:
    """Test SchemaValidator class."""

    def test_validator_initialization(self):
        """Test validator initializes with schema."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)
        assert validator.schema is not None


class TestFieldValidation:
    """Test individual field validation."""

    def test_validate_required_field_missing(self):
        """Test that missing required fields are caught."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        field = ConfigField(
            name="test_field",
            field_type=FieldType.STRING,
            default="",
            required=True,
            description="Test field description",
            section="story",
            cli_long="--test",
            cli_help="Test field",
        )

        errors = validator.validate_field(field, None)
        assert len(errors) == 1
        assert "Required field cannot be empty" in errors[0].message

    def test_validate_required_field_empty_string(self):
        """Test that empty string for required field is caught."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        field = ConfigField(
            name="test_field",
            field_type=FieldType.STRING,
            default="",
            required=True,
            description="Test field description",
            section="story",
            cli_long="--test",
            cli_help="Test field",
        )

        errors = validator.validate_field(field, "")
        assert len(errors) == 1
        assert "Required field cannot be empty" in errors[0].message

    def test_validate_optional_field_empty(self):
        """Test that empty optional fields pass validation."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        field = ConfigField(
            name="test_field",
            field_type=FieldType.STRING,
            default="",
            required=False,
            description="Test field description",
            section="story",
            cli_long="--test",
            cli_help="Test field",
        )

        errors = validator.validate_field(field, None)
        assert len(errors) == 0

    def test_validate_field_with_valid_values(self):
        """Test validation with valid_values constraint."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        field = ConfigField(
            name="length",
            field_type=FieldType.STRING,
            default="short",
            valid_values=["short", "medium", "long"],
            description="Test field description",
            section="story",
            cli_long="--length",
            cli_help="Story length",
        )

        # Valid value
        errors = validator.validate_field(field, "short")
        assert len(errors) == 0

        # Invalid value
        errors = validator.validate_field(field, "extra-long")
        assert len(errors) == 1
        assert "Invalid value" in errors[0].message
        assert "short, medium, long" in errors[0].message

    def test_validate_field_with_custom_validator(self):
        """Test validation with custom validator function."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        def custom_validator(value):
            return len(str(value)) > 5

        field = ConfigField(
            name="test_field",
            field_type=FieldType.STRING,
            default="default",
            validator=custom_validator,
            description="Test field description",
            section="story",
            cli_long="--test",
            cli_help="Test field",
        )

        # Valid (length > 5)
        errors = validator.validate_field(field, "long_value")
        assert len(errors) == 0

        # Invalid (length <= 5)
        errors = validator.validate_field(field, "short")
        assert len(errors) == 1
        assert "Custom validation failed" in errors[0].message


class TestTypeValidation:
    """Test type validation for different field types."""

    def test_validate_string_type_valid(self):
        """Test valid string type validation."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        field = ConfigField(
            name="test",
            field_type=FieldType.STRING,
            default="",
            description="Test field description",
            section="story",
            cli_long="--test",
            cli_help="Test",
        )

        errors = validator.validate_field(field, "valid string")
        assert len(errors) == 0

    def test_validate_string_type_invalid(self):
        """Test invalid string type validation."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        field = ConfigField(
            name="test",
            field_type=FieldType.STRING,
            default="",
            description="Test field description",
            section="story",
            cli_long="--test",
            cli_help="Test",
        )

        errors = validator.validate_field(field, 123)
        assert len(errors) == 1
        assert "Expected string" in errors[0].message

    def test_validate_integer_type_valid(self):
        """Test valid integer type validation."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        field = ConfigField(
            name="count",
            field_type=FieldType.INTEGER,
            default=0,
            description="Test field description",
            section="images",
            cli_long="--count",
            cli_help="Count",
        )

        # Valid integer
        errors = validator.validate_field(field, 5)
        assert len(errors) == 0

        # Valid string representation of integer
        errors = validator.validate_field(field, "5")
        assert len(errors) == 0

    def test_validate_integer_type_invalid(self):
        """Test invalid integer type validation."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        field = ConfigField(
            name="count",
            field_type=FieldType.INTEGER,
            default=0,
            description="Test field description",
            section="images",
            cli_long="--count",
            cli_help="Count",
        )

        errors = validator.validate_field(field, "not_a_number")
        assert len(errors) == 1
        assert "Expected integer" in errors[0].message

    def test_validate_boolean_type_valid(self):
        """Test valid boolean type validation."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        field = ConfigField(
            name="verbose",
            field_type=FieldType.BOOLEAN,
            default=False,
            description="Test field description",
            section="system",
            cli_long="--verbose",
            cli_help="Verbose",
        )

        # Valid boolean
        errors = validator.validate_field(field, True)
        assert len(errors) == 0

        # Valid boolean strings
        for value in ["true", "false", "1", "0", "yes", "no"]:
            errors = validator.validate_field(field, value)
            assert len(errors) == 0

    def test_validate_boolean_type_invalid(self):
        """Test invalid boolean type validation."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        field = ConfigField(
            name="verbose",
            field_type=FieldType.BOOLEAN,
            default=False,
            description="Test field description",
            section="system",
            cli_long="--verbose",
            cli_help="Verbose",
        )

        errors = validator.validate_field(field, "maybe")
        assert len(errors) == 1
        assert "Expected boolean" in errors[0].message

    def test_validate_list_type_valid(self):
        """Test valid list type validation."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        field = ConfigField(
            name="characters",
            field_type=FieldType.LIST,
            default=[],
            description="Test field description",
            section="story",
            cli_long="--character",
            cli_help="Characters",
        )

        # Valid list
        errors = validator.validate_field(field, ["Alice", "Bob"])
        assert len(errors) == 0

        # Valid comma-separated string
        errors = validator.validate_field(field, "Alice, Bob")
        assert len(errors) == 0

    def test_validate_list_type_invalid(self):
        """Test invalid list type validation."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        field = ConfigField(
            name="characters",
            field_type=FieldType.LIST,
            default=[],
            description="Test field description",
            section="story",
            cli_long="--character",
            cli_help="Characters",
        )

        errors = validator.validate_field(field, 123)
        assert len(errors) == 1
        assert "Expected list" in errors[0].message

    def test_validate_path_type_valid(self):
        """Test valid path type validation."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        field = ConfigField(
            name="config_path",
            field_type=FieldType.PATH,
            default="",
            description="Test field description",
            section="system",
            cli_long="--config-path",
            cli_help="Config path",
        )

        errors = validator.validate_field(field, "/path/to/config")
        assert len(errors) == 0

    def test_validate_path_type_invalid(self):
        """Test invalid path type validation."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        field = ConfigField(
            name="config_path",
            field_type=FieldType.PATH,
            default="",
            description="Test field description",
            section="system",
            cli_long="--config-path",
            cli_help="Config path",
        )

        errors = validator.validate_field(field, 123)
        assert len(errors) == 1
        assert "Expected path string" in errors[0].message


class TestConfigValidation:
    """Test full configuration validation."""

    def test_validate_config_all_valid(self):
        """Test validation of valid configuration."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        config_dict = {
            "story": {
                "length": "short",
                "style": "fantasy",
                "tone": "magical",
                "age_range": "preschool",
            },
            "images": {"image_style": "chibi", "image_count": "3"},
            "output": {"verbose": "true"},
            "system": {"backend": "gemini"},
        }

        errors = validator.validate_config(config_dict)
        assert len(errors) == 0

    def test_validate_config_multiple_errors(self):
        """Test that multiple validation errors are accumulated."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        config_dict = {
            "story": {
                "length": "invalid_length",  # Invalid value
                "style": "invalid_style",  # Invalid value
            }
        }

        errors = validator.validate_config(config_dict)
        # Should have at least 2 errors
        assert len(errors) >= 2

    def test_validate_config_unknown_section(self):
        """Test that unknown sections are ignored."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        config_dict = {"unknown_section": {"field": "value"}, "story": {"length": "short"}}

        # Should not raise error for unknown section
        _ = validator.validate_config(config_dict)
        # Only check that it doesn't crash, unknown fields are silently ignored

    def test_validate_config_unknown_field(self):
        """Test that unknown fields in known sections are ignored."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        config_dict = {"story": {"length": "short", "unknown_field": "value"}}

        # Should not raise error for unknown field
        _ = validator.validate_config(config_dict)
        # Unknown fields are silently ignored


class TestCLIArgumentValidation:
    """Test CLI argument validation."""

    def test_validate_cli_argument_valid(self):
        """Test valid CLI argument validation."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        errors = validator.validate_cli_argument("length", "short")
        assert len(errors) == 0

    def test_validate_cli_argument_invalid_value(self):
        """Test invalid CLI argument value."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        errors = validator.validate_cli_argument("length", "invalid_length")
        assert len(errors) == 1
        assert "Invalid value" in errors[0].message

    def test_validate_cli_argument_unknown_field(self):
        """Test validation of unknown CLI argument."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        errors = validator.validate_cli_argument("unknown_field", "value")
        assert len(errors) == 1
        assert "Unknown configuration field" in errors[0].message


class TestValidationErrorAccumulation:
    """Test that validation errors are properly accumulated."""

    def test_type_error_stops_further_validation(self):
        """Test that type errors prevent further validation checks."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        field = ConfigField(
            name="test",
            field_type=FieldType.STRING,
            default="",
            valid_values=["a", "b", "c"],
            description="Test field description",
            section="story",
            cli_long="--test",
            cli_help="Test",
        )

        # Invalid type (integer instead of string)
        errors = validator.validate_field(field, 123)

        # Should only have type error, not valid_values error
        assert len(errors) == 1
        assert "Expected string" in errors[0].message

    def test_required_error_stops_further_validation(self):
        """Test that required field errors prevent further validation."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        field = ConfigField(
            name="test",
            field_type=FieldType.STRING,
            default="",
            required=True,
            valid_values=["a", "b", "c"],
            description="Test field description",
            section="story",
            cli_long="--test",
            cli_help="Test",
        )

        # Missing required field
        errors = validator.validate_field(field, None)

        # Should only have required error, not type/valid_values errors
        assert len(errors) == 1
        assert "Required field" in errors[0].message
