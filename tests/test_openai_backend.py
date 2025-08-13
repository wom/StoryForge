import os
from unittest.mock import MagicMock, patch

from storyforge.openai_backend import OpenAIBackend
from storyforge.prompt import Prompt


@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
@patch("storyforge.openai_backend.openai.OpenAI")
def test_generate_story_success(mock_openai):
    backend = OpenAIBackend()
    mock_message = MagicMock()
    mock_message.content = "A story"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    backend.client.chat.completions.create.return_value = mock_response
    prompt = Prompt(prompt="test prompt")
    result = backend.generate_story(prompt)
    assert result == "A story"


@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
@patch("storyforge.openai_backend.openai.OpenAI")
def test_generate_story_error(mock_openai):
    backend = OpenAIBackend()
    backend.client.chat.completions.create.side_effect = Exception("fail")
    prompt = Prompt(prompt="test prompt")
    result = backend.generate_story(prompt)
    assert result == "[Error generating story]"


@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
@patch("storyforge.openai_backend.openai.OpenAI")
@patch("requests.get")
def test_generate_image_success(mock_requests, mock_openai):
    backend = OpenAIBackend()
    mock_data = MagicMock()
    mock_data.url = "https://example.com/image.png"
    mock_response = MagicMock()
    mock_response.data = [mock_data]
    backend.client.images.generate.return_value = mock_response

    # Mock the requests.get call
    mock_img_response = MagicMock()
    mock_img_response.status_code = 200
    mock_img_response.content = b"image_bytes"
    mock_requests.return_value = mock_img_response

    with patch("storyforge.openai_backend.Image.open", return_value="image_obj"):
        prompt = Prompt(prompt="test prompt")
        image, image_bytes = backend.generate_image(prompt)
        assert image == "image_obj"
        assert image_bytes == b"image_bytes"


@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
@patch("storyforge.openai_backend.openai.OpenAI")
def test_generate_image_none(mock_openai):
    backend = OpenAIBackend()
    mock_response = MagicMock()
    mock_response.data = []
    backend.client.images.generate.return_value = mock_response
    prompt = Prompt(prompt="test prompt")
    image, image_bytes = backend.generate_image(prompt)
    assert image is None
    assert image_bytes is None


@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
@patch("storyforge.openai_backend.openai.OpenAI")
def test_generate_image_name_success(mock_openai):
    backend = OpenAIBackend()
    mock_message = MagicMock()
    mock_message.content = "filename.png"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    backend.client.chat.completions.create.return_value = mock_response
    prompt = Prompt(prompt="test prompt")
    name = backend.generate_image_name(prompt, "story")
    assert name == "filename"


@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
@patch("storyforge.openai_backend.openai.OpenAI")
def test_generate_image_name_error(mock_openai):
    backend = OpenAIBackend()
    backend.client.chat.completions.create.side_effect = Exception("fail")
    prompt = Prompt(prompt="test prompt")
    name = backend.generate_image_name(prompt, "story")
    assert name == "story_image"


@patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
@patch("storyforge.openai_backend.openai.OpenAI")
def test_generate_image_prompt_basic(mock_openai):
    backend = OpenAIBackend()
    story = "Once upon a time.\nThere was a cat.\nThe end."
    context = "test context"
    prompts = backend.generate_image_prompt(story, context, 2)
    assert len(prompts) == 2
    assert "Create a detailed, child-friendly illustration" in prompts[0]
    assert "Once upon a time." in prompts[0]
    assert "Context: test context" in prompts[0]
