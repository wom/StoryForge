"""
Tests for the StoryPrompt class.
"""

import pytest

from storytime.story_prompt import StoryPrompt


class TestStoryPromptBasic:
    """Test basic StoryPrompt functionality."""

    def test_story_prompt_creation(self):
        """Test creating a basic StoryPrompt."""
        prompt = StoryPrompt(
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
        """Test creating a StoryPrompt with all parameters."""
        prompt = StoryPrompt(
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


class TestStoryPromptValidation:
    """Test parameter validation in StoryPrompt."""

    def test_valid_parameters_accepted(self):
        """Test that valid parameters are accepted."""
        # Should not raise any exception
        StoryPrompt(
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
        with pytest.raises(ValueError, match="Invalid length"):
            StoryPrompt(prompt="Test", length="invalid_length")

    def test_invalid_age_range_rejected(self):
        """Test that invalid age_range values are rejected."""
        with pytest.raises(ValueError, match="Invalid age_range"):
            StoryPrompt(prompt="Test", age_range="invalid_age")

    def test_invalid_style_rejected(self):
        """Test that invalid style values are rejected."""
        with pytest.raises(ValueError, match="Invalid style"):
            StoryPrompt(prompt="Test", style="invalid_style")

    def test_invalid_tone_rejected(self):
        """Test that invalid tone values are rejected."""
        with pytest.raises(ValueError, match="Invalid tone"):
            StoryPrompt(prompt="Test", tone="invalid_tone")

    def test_invalid_theme_rejected(self):
        """Test that invalid theme values are rejected."""
        with pytest.raises(ValueError, match="Invalid theme"):
            StoryPrompt(prompt="Test", theme="invalid_theme")

    def test_invalid_learning_focus_rejected(self):
        """Test that invalid learning_focus values are rejected."""
        with pytest.raises(ValueError, match="Invalid learning_focus"):
            StoryPrompt(prompt="Test", learning_focus="invalid_focus")


class TestStoryPromptPromptBuilding:
    """Test prompt building methods."""

    def test_build_story_prompt_basic(self):
        """Test building a basic story prompt."""
        prompt = StoryPrompt(
            prompt="A cat finds a magical hat",
            age_range="preschool",
            style="fantasy",
            tone="magical",
        )

        result = prompt.build_story_prompt()

        assert "A cat finds a magical hat" in result
        assert "short (3-4 paragraphs)" in result
        assert "magical story with fantastical elements" in result
        assert "magical tone" in result
        assert "simple language" in result  # preschool guidance
        assert "safe and appropriate for children" in result

    def test_build_story_prompt_with_context(self):
        """Test building a story prompt with context."""
        prompt = StoryPrompt(
            prompt="Adventure in the garden",
            context="Fluffy is a curious orange tabby cat who loves exploring.",
            style="adventure",
            tone="exciting",
        )

        result = prompt.build_story_prompt()

        assert "Context for story generation:" in result
        assert "Fluffy is a curious orange tabby cat" in result
        assert "Adventure in the garden" in result
        assert "exciting journey or quest" in result
        assert "Consistent with the provided context" in result

    def test_build_story_prompt_with_all_options(self):
        """Test building a comprehensive story prompt."""
        prompt = StoryPrompt(
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

        result = prompt.build_story_prompt()

        assert "bedtime story length" in result
        assert "friendship, cooperation, and caring" in result
        assert "heartwarming tone" in result
        assert "theme of kindness" in result
        assert "set in: a cozy village" in result
        assert "Luna, Oliver" in result
        assert "learning about emotions" in result

    def test_build_image_prompt(self):
        """Test building an image generation prompt."""
        prompt = StoryPrompt(
            prompt="A butterfly learns to fly",
            style="adventure",
            tone="gentle",
            age_range="toddler",
            setting="flower garden",
        )

        result = prompt.build_image_prompt()

        assert "child-friendly illustration" in result
        assert "exciting adventure scene" in result
        assert "A butterfly learns to fly" in result
        assert "Set in: flower garden" in result
        assert "soft, calming colors" in result  # gentle tone
        assert (
            "Simple, clear shapes with bright primary colors" in result
        )  # toddler age
        assert "safe, positive, and appropriate for children" in result

    def test_build_image_name_prompt(self):
        """Test building an image name generation prompt."""
        prompt = StoryPrompt(
            prompt="A dragon makes friends", style="friendship", age_range="preschool"
        )

        story = (
            "Once upon a time, a lonely dragon named Spark met a kind rabbit "
            "named Pip, and they became the best of friends."
        )
        result = prompt.build_image_name_prompt(story)

        assert "friendship story for preschool children" in result
        assert story in result
        assert "2-4 words that capture the essence" in result
        assert "no spaces or special characters" in result
        assert "Return only the filename" in result


class TestStoryPromptUtilities:
    """Test utility methods."""

    def test_get_valid_values(self):
        """Test the get_valid_values class method."""
        valid_values = StoryPrompt.get_valid_values()

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


class TestStoryPromptEdgeCases:
    """Test edge cases and special scenarios."""

    def test_none_optional_parameters(self):
        """Test that None values work for optional parameters."""
        prompt = StoryPrompt(
            prompt="Test story",
            context=None,
            theme=None,
            setting=None,
            characters=None,
            learning_focus=None,
        )

        # Should not raise any exceptions
        story_result = prompt.build_story_prompt()
        image_result = prompt.build_image_prompt()
        name_result = prompt.build_image_name_prompt("Sample story")

        assert isinstance(story_result, str)
        assert isinstance(image_result, str)
        assert isinstance(name_result, str)

    def test_empty_characters_list(self):
        """Test behavior with empty characters list."""
        prompt = StoryPrompt(prompt="Test story", characters=[])

        result = prompt.build_story_prompt()
        assert isinstance(result, str)
        # Should not include character information
        assert "Include these characters:" not in result

    def test_multiple_characters(self):
        """Test behavior with multiple characters."""
        prompt = StoryPrompt(
            prompt="Test story",
            characters=["Alice", "Bob", "Charlie the Cat", "Mrs. Henderson"],
        )

        result = prompt.build_story_prompt()
        assert "Alice, Bob, Charlie the Cat, Mrs. Henderson" in result


class TestBackwardCompatibility:
    """Test that the system maintains backward compatibility."""

    def test_story_prompt_can_be_used_as_string_alternative(self):
        """Test that StoryPrompt can be used where strings were used before."""

        # This simulates how the backend would handle the prompt
        def simulate_backend_usage(prompt_input, context=None):
            if isinstance(prompt_input, StoryPrompt):
                return prompt_input.build_story_prompt()
            else:
                if context:
                    return f"Legacy: {prompt_input} with context: {context}"
                return f"Legacy: {prompt_input}"

        # Test with StoryPrompt
        story_prompt = StoryPrompt(prompt="Test adventure", style="adventure")
        result1 = simulate_backend_usage(story_prompt)
        assert isinstance(result1, str)
        assert "Test adventure" in result1

        # Test with legacy string
        result2 = simulate_backend_usage("Test adventure", "Some context")
        assert result2 == "Legacy: Test adventure with context: Some context"

        # Test with simple string
        result3 = simulate_backend_usage("Test adventure")
        assert result3 == "Legacy: Test adventure"
