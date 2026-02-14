import os
from unittest.mock import MagicMock, patch

from storyforge.gemini_backend import GeminiBackend
from storyforge.prompt import Prompt


@patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
@patch("storyforge.gemini_backend.genai.Client")
def test_generate_story_success(mock_client):
    backend = GeminiBackend()
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(text="A story")]))]
    backend.client.models.generate_content.return_value = mock_response
    prompt = Prompt(prompt="test prompt")
    result = backend.generate_story(prompt)
    assert result == "A story"


@patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
@patch("storyforge.gemini_backend.genai.Client")
def test_generate_story_error(mock_client):
    backend = GeminiBackend()
    backend.client.models.generate_content.side_effect = Exception("fail")
    prompt = Prompt(prompt="test prompt")
    result = backend.generate_story(prompt)
    assert result == "[Error generating story: fail]"


@patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
@patch("storyforge.gemini_backend.genai.Client")
def test_generate_image_success(mock_client):
    backend = GeminiBackend()
    mock_part = MagicMock()
    mock_part.inline_data = MagicMock(data=b"bytes")
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
    # Ensure generated_images is not set so fallback path is used
    mock_response.generated_images = None
    backend.client.models.generate_content.return_value = mock_response
    with patch("storyforge.gemini_backend.Image.open", return_value="image_obj"):
        prompt = Prompt(prompt="test prompt")
        image, image_bytes = backend.generate_image(prompt)
        assert image == "image_obj"
        assert image_bytes == b"bytes"

        # Verify response_modalities was passed in the config
        call_kwargs = backend.client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config is not None
        assert "IMAGE" in config.response_modalities
        assert "TEXT" in config.response_modalities


@patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
@patch("storyforge.gemini_backend.genai.Client")
def test_generate_image_none(mock_client):
    backend = GeminiBackend()
    mock_part = MagicMock()
    mock_part.inline_data = None
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
    backend.client.models.generate_content.return_value = mock_response
    prompt = Prompt(prompt="test prompt")
    image, image_bytes = backend.generate_image(prompt)
    assert image is None
    assert image_bytes is None


@patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
@patch("storyforge.gemini_backend.genai.Client")
def test_generate_image_name_success(mock_client):
    backend = GeminiBackend()
    mock_part = MagicMock()
    mock_part.text = "filename.png"
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
    backend.client.models.generate_content.return_value = mock_response
    prompt = Prompt(prompt="test prompt")
    name = backend.generate_image_name(prompt, "story")
    assert name == "filename"


@patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
@patch("storyforge.gemini_backend.genai.Client")
def test_generate_image_name_error(mock_client):
    backend = GeminiBackend()
    backend.client.models.generate_content.side_effect = Exception("fail")
    prompt = Prompt(prompt="test prompt")
    name = backend.generate_image_name(prompt, "story")
    assert name == "story_image"


@patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
@patch("storyforge.gemini_backend.genai.Client")
def test_generate_image_with_override_prompt(mock_client):
    """Test that override_prompt is used instead of prompt.image() when provided."""
    backend = GeminiBackend()
    mock_part = MagicMock()
    mock_part.inline_data = MagicMock(data=b"bytes")
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
    mock_response.generated_images = None
    backend.client.models.generate_content.return_value = mock_response
    with patch("storyforge.gemini_backend.Image.open", return_value="image_obj"):
        prompt = Prompt(prompt="test prompt")
        override = "A specific scene: the fox crosses the bridge"
        image, image_bytes = backend.generate_image(prompt, override_prompt=override)
        assert image == "image_obj"
        assert image_bytes == b"bytes"

        # Verify the override prompt was passed as contents
        call_kwargs = backend.client.models.generate_content.call_args
        contents = call_kwargs.kwargs.get("contents") or call_kwargs[1].get("contents")
        assert contents == override


@patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
@patch("storyforge.gemini_backend.genai.Client")
def test_generate_image_generated_images_path(mock_client):
    """Test extraction via the generated_images response path."""
    backend = GeminiBackend()
    mock_response = MagicMock()
    mock_image_obj = MagicMock()
    mock_image_obj.image_bytes = b"gen_image_bytes"
    mock_gi = MagicMock()
    mock_gi.image = mock_image_obj
    mock_response.generated_images = [mock_gi]
    backend.client.models.generate_content.return_value = mock_response
    with patch("storyforge.gemini_backend.Image.open", return_value="gen_image"):
        prompt = Prompt(prompt="test prompt")
        image, image_bytes = backend.generate_image(prompt)
        assert image == "gen_image"
        assert image_bytes == b"gen_image_bytes"
