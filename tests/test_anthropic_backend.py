"""
Tests for AnthropicBackend implementation.
Following the same testing patterns as test_gemini_backend.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from storyforge.anthropic_backend import AnthropicBackend
from storyforge.prompt import Prompt


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
@patch("storyforge.anthropic_backend.anthropic.Anthropic")
def test_anthropic_backend_init_success(mock_anthropic):
    """Test successful initialization of AnthropicBackend."""
    AnthropicBackend()
    mock_anthropic.assert_called_once_with(api_key="test_key")


def test_anthropic_backend_init_no_api_key():
    """Test AnthropicBackend initialization fails without API key."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError) as exc_info:
            AnthropicBackend()
        assert "ANTHROPIC_API_KEY environment variable not set" in str(exc_info.value)


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
@patch("storyforge.anthropic_backend.anthropic.Anthropic")
def test_generate_story_success(mock_anthropic):
    """Test successful story generation."""
    backend = AnthropicBackend()

    # Mock response structure
    mock_content_block = MagicMock()
    mock_content_block.type = "text"
    mock_content_block.text = "A wonderful story about adventure"

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]

    mock_client = mock_anthropic.return_value
    mock_client.messages.create.return_value = mock_response

    prompt = Prompt(prompt="test prompt")
    result = backend.generate_story(prompt)

    assert result == "A wonderful story about adventure"
    mock_client.messages.create.assert_called_once()


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
@patch("storyforge.anthropic_backend.anthropic.Anthropic")
def test_generate_story_no_content(mock_anthropic):
    """Test story generation when response has no content."""
    backend = AnthropicBackend()

    mock_response = MagicMock()
    mock_response.content = []

    mock_client = mock_anthropic.return_value
    mock_client.messages.create.return_value = mock_response

    prompt = Prompt(prompt="test prompt")
    result = backend.generate_story(prompt)

    assert result == "[Error: No valid response from Claude]"


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
@patch("storyforge.anthropic_backend.anthropic.Anthropic")
def test_generate_story_non_text_content(mock_anthropic):
    """Test story generation when response has non-text content blocks."""
    backend = AnthropicBackend()

    # Mock non-text content block
    mock_content_block = MagicMock()
    mock_content_block.type = "tool_use"  # Not text

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]

    mock_client = mock_anthropic.return_value
    mock_client.messages.create.return_value = mock_response

    prompt = Prompt(prompt="test prompt")
    result = backend.generate_story(prompt)

    assert result == "[Error: No valid response from Claude]"


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
@patch("storyforge.anthropic_backend.anthropic.Anthropic")
def test_generate_story_api_error(mock_anthropic):
    """Test story generation when API call fails."""

    mock_client = mock_anthropic.return_value
    mock_client.messages.create.side_effect = Exception("API Error")

    prompt = Prompt(prompt="test prompt")
    result = AnthropicBackend().generate_story(prompt)

    assert "Error generating story: API Error" in result


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
@patch("storyforge.anthropic_backend.anthropic.Anthropic")
def test_generate_image_always_returns_none(mock_anthropic):
    """Test that generate_image always returns None since Claude can't generate images."""
    backend = AnthropicBackend()

    prompt = Prompt(prompt="test prompt")
    image, image_bytes = backend.generate_image(prompt)

    assert image is None
    assert image_bytes is None


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
@patch("storyforge.anthropic_backend.anthropic.Anthropic")
def test_generate_image_with_reference_returns_none(mock_anthropic):
    """Test that generate_image returns None even with reference image."""
    backend = AnthropicBackend()

    prompt = Prompt(prompt="test prompt")
    image, image_bytes = backend.generate_image(prompt, reference_image_bytes=b"test_bytes")

    assert image is None
    assert image_bytes is None


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
@patch("storyforge.anthropic_backend.anthropic.Anthropic")
def test_generate_image_name_success(mock_anthropic):
    """Test successful image name generation."""
    backend = AnthropicBackend()

    mock_content_block = MagicMock()
    mock_content_block.type = "text"
    mock_content_block.text = "brave_mouse_adventure"

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]

    mock_client = mock_anthropic.return_value
    mock_client.messages.create.return_value = mock_response

    prompt = Prompt(prompt="test prompt")
    result = backend.generate_image_name(prompt, "A story about a brave mouse")

    assert result == "brave_mouse_adventure"


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
@patch("storyforge.anthropic_backend.anthropic.Anthropic")
def test_generate_image_name_with_extension(mock_anthropic):
    """Test image name generation removes file extensions."""
    backend = AnthropicBackend()

    mock_content_block = MagicMock()
    mock_content_block.type = "text"
    mock_content_block.text = "brave_mouse_adventure.png"

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]

    mock_client = mock_anthropic.return_value
    mock_client.messages.create.return_value = mock_response

    prompt = Prompt(prompt="test prompt")
    result = backend.generate_image_name(prompt, "A story about a brave mouse")

    assert result == "brave_mouse_adventure"


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
@patch("storyforge.anthropic_backend.anthropic.Anthropic")
def test_generate_image_name_cleans_special_chars(mock_anthropic):
    """Test image name generation cleans special characters."""
    backend = AnthropicBackend()

    mock_content_block = MagicMock()
    mock_content_block.type = "text"
    mock_content_block.text = "brave-mouse adventure@123!"

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]

    mock_client = mock_anthropic.return_value
    mock_client.messages.create.return_value = mock_response

    prompt = Prompt(prompt="test prompt")
    result = backend.generate_image_name(prompt, "A story about a brave mouse")

    assert result == "bravemouseadventure123"


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
@patch("storyforge.anthropic_backend.anthropic.Anthropic")
def test_generate_image_name_api_error(mock_anthropic):
    """Test image name generation fallback on API error."""
    backend = AnthropicBackend()

    mock_client = mock_anthropic.return_value
    mock_client.messages.create.side_effect = Exception("API Error")

    prompt = Prompt(prompt="test prompt")
    result = backend.generate_image_name(prompt, "A story about a brave mouse")

    assert result == "story_image"


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
@patch("storyforge.anthropic_backend.anthropic.Anthropic")
def test_generate_image_name_empty_response(mock_anthropic):
    """Test image name generation fallback on empty response."""
    backend = AnthropicBackend()

    mock_content_block = MagicMock()
    mock_content_block.type = "text"
    mock_content_block.text = ""

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]

    mock_client = mock_anthropic.return_value
    mock_client.messages.create.return_value = mock_response

    prompt = Prompt(prompt="test prompt")
    result = backend.generate_image_name(prompt, "A story about a brave mouse")

    assert result == "story_image"


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
@patch("storyforge.anthropic_backend.anthropic.Anthropic")
def test_generate_image_prompt_success(mock_anthropic):
    """Test successful image prompt generation."""
    backend = AnthropicBackend()

    mock_content_block = MagicMock()
    mock_content_block.type = "text"
    mock_content_block.text = (
        "1. A brave little mouse stands at the edge of a dark forest\n"
        "2. The mouse discovers a magical golden acorn glowing softly\n"
        "3. The mouse returns home triumphantly with the treasure"
    )

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]

    mock_client = mock_anthropic.return_value
    mock_client.messages.create.return_value = mock_response

    result = backend.generate_image_prompt("A brave mouse story", "forest context", 3)

    expected = [
        "A brave little mouse stands at the edge of a dark forest",
        "The mouse discovers a magical golden acorn glowing softly",
        "The mouse returns home triumphantly with the treasure",
    ]
    assert result == expected


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
@patch("storyforge.anthropic_backend.anthropic.Anthropic")
def test_generate_image_prompt_api_error_fallback(mock_anthropic):
    """Test image prompt generation uses fallback on API error."""
    backend = AnthropicBackend()

    mock_client = mock_anthropic.return_value
    mock_client.messages.create.side_effect = Exception("API Error")

    story = "A brave mouse went on an adventure.\nThe mouse found treasure.\nThe mouse returned home."
    result = backend.generate_image_prompt(story, "forest context", 2)

    assert len(result) == 2
    assert "brave mouse went on an adventure" in result[0].lower()
    assert "forest context" in result[0]


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
@patch("storyforge.anthropic_backend.anthropic.Anthropic")
def test_generate_image_prompt_wrong_number_fallback(mock_anthropic):
    """Test image prompt generation uses fallback when wrong number of prompts returned."""
    backend = AnthropicBackend()

    mock_content_block = MagicMock()
    mock_content_block.type = "text"
    mock_content_block.text = "1. Only one prompt returned"

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]

    mock_client = mock_anthropic.return_value
    mock_client.messages.create.return_value = mock_response

    story = "A brave mouse story"
    result = backend.generate_image_prompt(story, "context", 3)

    # Should use fallback and return 3 prompts
    assert len(result) == 3
    assert all("brave mouse story" in prompt.lower() for prompt in result)


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
def test_generate_fallback_image_prompts():
    """Test the fallback image prompt generation method."""
    backend = AnthropicBackend()

    story = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    context = "Forest setting"

    result = backend._generate_fallback_image_prompts(story, context, 2)

    assert len(result) == 2
    assert "First paragraph" in result[0]
    assert "Forest setting" in result[0]
    assert "Second paragraph" in result[1]
    assert "Forest setting" in result[1]


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})
def test_generate_fallback_image_prompts_fewer_paragraphs():
    """Test fallback when story has fewer paragraphs than requested prompts."""
    backend = AnthropicBackend()

    story = "Only one paragraph."
    context = "Forest setting"

    result = backend._generate_fallback_image_prompts(story, context, 3)

    assert len(result) == 3
    # First prompt uses the paragraph, others use full story
    assert "Only one paragraph" in result[0]
    assert all("Forest setting" in prompt for prompt in result)
