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
    assert result == "[Error generating story]"


@patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
@patch("storyforge.gemini_backend.genai.Client")
def test_generate_image_success(mock_client):
    backend = GeminiBackend()
    mock_part = MagicMock()
    mock_part.inline_data = MagicMock(data=b"bytes")
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
    backend.client.models.generate_content.return_value = mock_response
    with patch("storyforge.gemini_backend.Image.open", return_value="image_obj"):
        prompt = Prompt(prompt="test prompt")
        image, image_bytes = backend.generate_image(prompt)
        assert image == "image_obj"
        assert image_bytes == b"bytes"


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
