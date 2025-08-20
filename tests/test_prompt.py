"""
Tests for the Prompt class.
"""

from typing import cast

import pytest

from storyforge.prompt import Prompt
from storyforge.schema import STORYFORGE_SCHEMA, SchemaValidator


class TestPromptBasic:
    """Test basic Prompt functionality."""

    def test_story_prompt_creation(self):
        """Test creating a basic Prompt."""
        prompt = Prompt(
            prompt="A little rabbit goes on an adventure",
            age_range="preschool",
            style="adventure",
            tone="heartwarming",
        )

        assert prompt.prompt == "A little rabbit goes on an adventure"
        assert prompt.age_range == "preschool"
        assert prompt.style == "adventure"
        assert prompt.tone == "heartwarming"
        assert prompt.length == "short"  # default value
        assert prompt.context is None  # default value

    def test_story_prompt_with_all_parameters(self):
        """Test creating a Prompt with all parameters."""
        prompt = Prompt(
            prompt="Two friends solve a mystery",
            context="Maya and Sam are best friends",
            length="medium",
            age_range="early_reader",
            style="friendship",
            tone="exciting",
            theme="teamwork",
            setting="suburban neighborhood",
            characters=["Maya", "Sam"],
            learning_focus="emotions",
        )

        assert prompt.prompt == "Two friends solve a mystery"
        assert prompt.context == "Maya and Sam are best friends"
        assert prompt.length == "medium"
        assert prompt.age_range == "early_reader"
        assert prompt.style == "friendship"
        assert prompt.tone == "exciting"
        assert prompt.theme == "teamwork"
        assert prompt.setting == "suburban neighborhood"
        assert prompt.characters == ["Maya", "Sam"]
        assert prompt.learning_focus == "emotions"


class TestPromptValidation:
    """Test parameter validation in Prompt."""

    def test_valid_parameters_accepted(self):
        """Test that valid parameters are accepted."""
        # Should not raise any exception
        Prompt(
            prompt="Test",
            length="bedtime",
            age_range="toddler",
            style="fairy_tale",
            tone="magical",
            theme="kindness",
            learning_focus="colors",
        )

    def test_invalid_length_rejected(self):
        """Test that invalid length values are rejected."""
        with pytest.raises(ValueError, match="Invalid value.*invalid_length"):
            Prompt(prompt="Test", length=cast(str, "invalid_length"))

    def test_invalid_age_range_rejected(self):
        """Test that invalid age_range values are rejected."""
        with pytest.raises(ValueError, match="Invalid value.*invalid_age"):
            Prompt(prompt="Test", age_range=cast(str, "invalid_age"))

    def test_invalid_style_rejected(self):
        """Test that invalid style values are rejected."""
        with pytest.raises(ValueError, match="Invalid value.*invalid_style"):
            Prompt(prompt="Test", style=cast(str, "invalid_style"))

    def test_invalid_tone_rejected(self):
        """Test that invalid tone values are rejected."""
        with pytest.raises(ValueError, match="Invalid value.*invalid_tone"):
            Prompt(prompt="Test", tone=cast(str, "invalid_tone"))

    def test_invalid_theme_rejected(self):
        """Test that invalid theme values are rejected."""
        with pytest.raises(ValueError, match="Invalid value.*invalid_theme"):
            Prompt(prompt="Test", theme=cast(str, "invalid_theme"))

    def test_invalid_learning_focus_rejected(self):
        """Test that invalid learning_focus values are rejected."""
        with pytest.raises(ValueError, match="Invalid value.*invalid_focus"):
            Prompt(prompt="Test", learning_focus=cast(str, "invalid_focus"))


class TestPromptPromptBuilding:
    """Test prompt building methods."""

    def test_build_story_prompt_basic(self):
        """Test building a basic story prompt."""
        prompt = Prompt(
            prompt="A cat finds a magical hat",
            age_range="preschool",
            style="fantasy",
            tone="magical",
        )

        result = prompt.story

        assert "A cat finds a magical hat" in result
        assert "short (3-4 paragraphs)" in result
        assert "magical story with fantastical elements" in result
        assert "magical tone" in result
        assert "simple language" in result  # preschool guidance
        assert "safe and appropriate for children" in result

    def test_build_story_prompt_with_context(self):
        """Test building a story prompt with context."""
        prompt = Prompt(
            prompt="Adventure in the garden",
            context="Fluffy is a curious orange tabby cat who loves exploring.",
            style="adventure",
            tone="exciting",
        )

        result = prompt.story

        assert "Context for story generation:" in result
        assert "Fluffy is a curious orange tabby cat" in result
        assert "Adventure in the garden" in result
        assert "exciting journey or quest" in result
        assert "Consistent with the provided context" in result

    def test_build_story_prompt_with_all_options(self):
        """Test building a comprehensive story prompt."""
        prompt = Prompt(
            prompt="Friends help each other",
            length="bedtime",
            age_range="early_reader",
            style="friendship",
            tone="heartwarming",
            theme="kindness",
            setting="a cozy village",
            characters=["Luna", "Oliver"],
            learning_focus="emotions",
        )

        result = prompt.story

        assert "bedtime story length" in result
        assert "friendship, cooperation, and caring" in result
        assert "heartwarming tone" in result
        assert "theme of kindness" in result
        assert "set in: a cozy village" in result
        assert "Luna, Oliver" in result
        assert "learning about emotions" in result

    def test_build_image_prompt(self):
        """Test building an image generation prompt."""
        prompt = Prompt(
            prompt="A butterfly learns to fly",
            style="adventure",
            tone="gentle",
            age_range="toddler",
            setting="flower garden",
        )

        result = prompt.image(1)[0]

        assert "child-friendly illustration" in result
        assert "exciting adventure scene" in result
        assert "A butterfly learns to fly" in result
        assert "Set in: flower garden" in result
        assert "soft, calming colors" in result  # gentle tone
        assert "Simple, clear shapes with bright primary colors" in result  # toddler age
        assert "safe, positive, and appropriate for children" in result

    def test_build_image_name_prompt(self):
        """Test building an image name generation prompt."""
        prompt = Prompt(prompt="A dragon makes friends", style="friendship", age_range="preschool")

        story = (
            "Once upon a time, a lonely dragon named Spark met a kind rabbit "
            "named Pip, and they became the best of friends."
        )
        result = prompt.image_name(story)

        assert "friendship story for preschool children" in result
        assert story in result
        assert "2-4 words that capture the essence" in result
        assert "no spaces or special characters" in result
        assert "Return only the filename" in result


class TestPromptUtilities:
    """Test utility methods."""

    def test_get_valid_values(self):
        """Test the get_valid_values class method."""
        valid_values = Prompt.get_valid_values()

        assert isinstance(valid_values, dict)
        assert "length" in valid_values
        assert "age_range" in valid_values
        assert "style" in valid_values
        assert "tone" in valid_values
        assert "theme" in valid_values
        assert "learning_focus" in valid_values

        # Check some expected values
        assert "short" in valid_values["length"]
        assert "preschool" in valid_values["age_range"]
        assert "adventure" in valid_values["style"]
        assert "heartwarming" in valid_values["tone"]
        assert "kindness" in valid_values["theme"]
        assert "colors" in valid_values["learning_focus"]


class TestPromptEdgeCases:
    """Test edge cases and special scenarios."""

    def test_none_optional_parameters(self):
        """Test that None values work for optional parameters."""
        prompt = Prompt(
            prompt="Test story",
            context=None,
            theme=None,
            setting=None,
            characters=None,
            learning_focus=None,
        )

        # Should not raise any exceptions
        story_result = prompt.story
        image_result = prompt.image(1)[0]
        name_result = prompt.image_name("Sample story")

        assert isinstance(story_result, str)
        assert isinstance(image_result, str)
        assert isinstance(name_result, str)

    def test_empty_characters_list(self):
        """Test behavior with empty characters list."""
        prompt = Prompt(prompt="Test story", characters=[])

        result = prompt.story
        assert isinstance(result, str)
        # Should not include character information
        assert "Include these characters:" not in result

    def test_multiple_characters(self):
        """Test behavior with multiple characters."""
        prompt = Prompt(
            prompt="Test story",
            characters=["Alice", "Bob", "Charlie the Cat", "Mrs. Henderson"],
        )

        result = prompt.story
        assert "Alice, Bob, Charlie the Cat, Mrs. Henderson" in result


class TestBackwardCompatibility:
    """Test that the system maintains backward compatibility."""

    def test_story_prompt_can_be_used_as_string_alternative(self):
        """Test that Prompt can be used where strings were used before."""

        # This simulates how the backend would handle the prompt
        def simulate_backend_usage(prompt_input, context=None):
            if isinstance(prompt_input, Prompt):
                return prompt_input.story
            else:
                if context:
                    return f"Legacy: {prompt_input} with context: {context}"
                return f"Legacy: {prompt_input}"

        # Test with Prompt
        story_prompt = Prompt(prompt="Test adventure", style="adventure")
        result1 = simulate_backend_usage(story_prompt)
        assert isinstance(result1, str)
        assert "Test adventure" in result1

        # Test with legacy string
        result2 = simulate_backend_usage("Test adventure", "Some context")
        assert result2 == "Legacy: Test adventure with context: Some context"

        # Test with simple string
        result3 = simulate_backend_usage("Test adventure")
        assert result3 == "Legacy: Test adventure"


class TestRandomParameterFunctionality:
    """Test the new random parameter functionality."""

    def test_random_style_resolution(self):
        """Test that random style values are resolved to valid options."""
        prompt = Prompt(prompt="Test story", style="random")

        valid_styles = Prompt.get_valid_values()["style"]
        assert prompt.style in valid_styles
        assert prompt.style != "random"

    def test_random_tone_resolution(self):
        """Test that random tone values are resolved to valid options."""
        prompt = Prompt(prompt="Test story", tone="random")

        valid_tones = Prompt.get_valid_values()["tone"]
        assert prompt.tone in valid_tones
        assert prompt.tone != "random"

    def test_random_theme_resolution(self):
        """Test that random theme values are resolved to valid options."""
        prompt = Prompt(prompt="Test story", theme="random")

        valid_themes = Prompt.get_valid_values()["theme"]
        assert prompt.theme in valid_themes
        assert prompt.theme != "random"

    def test_random_learning_focus_resolution(self):
        """Test that random learning_focus values are resolved to valid options."""
        prompt = Prompt(prompt="Test story", learning_focus=cast(str, "random"))

        valid_learning = Prompt.get_valid_values()["learning_focus"]
        assert prompt.learning_focus in valid_learning
        assert prompt.learning_focus != "random"

    def test_multiple_random_parameters(self):
        """Test that multiple random parameters are all resolved."""
        prompt = Prompt(
            prompt="Test story",
            style="random",
            tone="random",
            theme="random",
            learning_focus=cast(str, "random"),
        )

        valid_values = Prompt.get_valid_values()
        assert prompt.style in valid_values["style"]
        assert prompt.tone in valid_values["tone"]
        assert prompt.theme in valid_values["theme"]
        assert prompt.learning_focus in valid_values["learning_focus"]

        # None should still be "random"
        assert prompt.style != "random"
        assert prompt.tone != "random"
        assert prompt.theme != "random"
        assert prompt.learning_focus != "random"

    def test_mixed_random_and_specific_parameters(self):
        """Test mixing random and specific parameter values."""
        prompt = Prompt(
            prompt="Test story",
            style="adventure",  # specific
            tone="random",  # random
            theme="kindness",  # specific
            learning_focus=cast(str, "random"),  # random
        )

        # Specific values should remain unchanged
        assert prompt.style == "adventure"
        assert prompt.theme == "kindness"

        # Random values should be resolved
        valid_values = Prompt.get_valid_values()
        assert prompt.tone in valid_values["tone"]
        assert prompt.learning_focus in valid_values["learning_focus"]
        assert prompt.tone != "random"
        assert prompt.learning_focus != "random"

    def test_random_parameters_in_prompts(self):
        """Test that resolved random parameters appear in generated prompts."""
        prompt = Prompt(prompt="A brave mouse adventure", style="random", tone="random")

        story_prompt = prompt.story
        image_prompt = prompt.image(1)[0]

        # The resolved values should appear in some form (as descriptive text)
        # Check that we have a valid style and tone
        valid_values = Prompt.get_valid_values()
        assert prompt.style in valid_values["style"]
        assert prompt.tone in valid_values["tone"]

        # Check that the prompts contain relevant descriptive text
        assert len(story_prompt) > 0
        assert len(image_prompt) > 0
        assert "A brave mouse adventure" in story_prompt
        assert "A brave mouse adventure" in image_prompt

        # "random" should not appear in the prompts
        assert "random" not in story_prompt
        assert "random" not in image_prompt

    def test_random_parameter_validation_still_works(self):
        """Test that validation still works properly with random parameters."""
        # Valid random parameters should work
        prompt = Prompt(prompt="Test", style="random", tone="random")
        assert isinstance(prompt.style, str)
        assert isinstance(prompt.tone, str)

        # Invalid non-random parameters should still be rejected
        with pytest.raises(ValueError, match="Invalid value.*invalid_style"):
            Prompt(prompt="Test", style="invalid_style")

        with pytest.raises(ValueError, match="Invalid value.*invalid_tone"):
            Prompt(prompt="Test", tone="invalid_tone")

    def test_randomness_varies_between_instances(self):
        """Test that random resolution actually varies between different instances."""
        # Create multiple prompts with random parameters
        prompts = [Prompt(prompt="Test story", style="random", tone="random") for _ in range(10)]

        # Collect all resolved values
        styles = [p.style for p in prompts]
        tones = [p.tone for p in prompts]

        # With 10 instances and multiple valid options, we should see some variation
        # (This is probabilistic, but very likely to pass)
        assert len(set(styles)) > 1 or len(set(tones)) > 1, "Expected some variation in random values"


class TestSchemaBasedValidation:
    """Test that the refactored schema-based validation works correctly."""

    def test_validation_uses_schema(self):
        """Test that validation now uses the configuration schema."""
        # This should work with valid schema values
        prompt = Prompt(prompt="Test story", style="adventure", tone="heartwarming", image_style="chibi")

        assert prompt.style == "adventure"
        assert prompt.tone == "heartwarming"
        assert prompt.image_style == "chibi"

    def test_schema_validator_integration(self):
        """Test that schema validator is properly integrated."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        # Test valid value
        story_section = STORYFORGE_SCHEMA.story
        style_field = story_section.fields["style"]
        errors = validator.validate_field(style_field, "adventure")
        assert len(errors) == 0

        # Test invalid value
        errors = validator.validate_field(style_field, "invalid_style")
        assert len(errors) == 1
        assert "Invalid value" in errors[0].message

    def test_get_valid_values_from_schema(self):
        """Test that get_valid_values now extracts from schema."""
        valid_values = Prompt.get_valid_values()

        # Verify it matches schema definitions
        story_section = STORYFORGE_SCHEMA.story
        images_section = STORYFORGE_SCHEMA.images

        # Check that values come from schema
        schema_style_values = [v for v in story_section.fields["style"].valid_values if v and v != "random"]
        schema_image_values = [v for v in images_section.fields["image_style"].valid_values if v and v != "random"]

        assert valid_values["style"] == schema_style_values
        assert valid_values["image_style"] == schema_image_values

    def test_validation_error_messages_from_schema(self):
        """Test that validation error messages come from schema validation."""
        with pytest.raises(ValueError) as exc_info:
            Prompt(prompt="Test", style="invalid_style")

        error_message = str(exc_info.value)
        # Should contain schema validation error format
        assert "Invalid value" in error_message
        assert "Valid options:" in error_message
        assert "adventure, comedy, fantasy, fairy_tale, friendship, random" in error_message

    def test_image_style_validation_from_schema(self):
        """Test that image_style validation also uses schema."""
        # Valid image style should work
        prompt = Prompt(prompt="Test", image_style="watercolor")
        assert prompt.image_style == "watercolor"

        # Invalid image style should be rejected
        with pytest.raises(ValueError) as exc_info:
            Prompt(prompt="Test", image_style="invalid_image_style")

        error_message = str(exc_info.value)
        assert "Invalid value" in error_message
        assert "chibi, realistic, cartoon, watercolor, sketch" in error_message


class TestContextManagerIntegration:
    """Test that context loading improvements work correctly."""

    def test_context_manager_import(self):
        """Test that ContextManager can be imported and used."""
        from storyforge.context import ContextManager

        context_manager = ContextManager()
        assert context_manager is not None

        # Should be able to call load_context without error
        context = context_manager.load_context()
        assert context is None or isinstance(context, str)

    def test_context_manager_caching(self):
        """Test that ContextManager caching works."""
        from storyforge.context import ContextManager

        context_manager = ContextManager()

        # First call
        context1 = context_manager.load_context()

        # Second call should return same result (cached)
        context2 = context_manager.load_context()

        assert context1 == context2

        # Clear cache and try again
        context_manager.clear_cache()
        context3 = context_manager.load_context()

        # Should still be same content, but freshly loaded
        assert context1 == context3


class TestTechnicalDebtCleanup:
    """Test that technical debt cleanup was successful."""

    def test_no_hardcoded_validation_arrays(self):
        """Test that hardcoded validation arrays were removed from prompt.py."""
        import inspect

        from storyforge import prompt

        # Get the source code of the prompt module
        source_lines = inspect.getsourcelines(prompt)[0]
        source_text = "".join(source_lines)

        # Should not contain hardcoded arrays like this anymore
        hardcoded_patterns = [
            "valid_lengths = [",
            "valid_age_ranges = [",
            "valid_styles = [",
            "valid_tones = [",
            "valid_themes = [",
            "valid_learning = [",
            "valid_image_styles = [",
        ]

        for pattern in hardcoded_patterns:
            assert pattern not in source_text, f"Found hardcoded validation array: {pattern}"

    def test_schema_imports_present(self):
        """Test that schema imports were added to prompt.py."""
        import inspect

        from storyforge import prompt

        source_lines = inspect.getsourcelines(prompt)[0]
        source_text = "".join(source_lines)

        # Should contain schema imports
        assert "from .schema import" in source_text
        assert "STORYFORGE_SCHEMA" in source_text
        assert "SchemaValidator" in source_text

    def test_validation_method_uses_schema(self):
        """Test that _validate_parameters method uses schema validation."""
        import inspect

        from storyforge.prompt import Prompt

        # Get source of validation method
        source_lines = inspect.getsourcelines(Prompt._validate_parameters)[0]
        source_text = "".join(source_lines)

        # Should use SchemaValidator
        assert "SchemaValidator" in source_text
        assert "STORYFORGE_SCHEMA" in source_text

        # Should not contain hardcoded validation
        assert "valid_lengths" not in source_text
        assert "valid_age_ranges" not in source_text
