import os
from unittest.mock import MagicMock, patch

from storyforge.gemini_backend import GeminiBackend
from storyforge.llm_backend import ERROR_STORY_SENTINEL
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
    assert result == f"{ERROR_STORY_SENTINEL}: fail"


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


def test_extract_text_valid_response():
    """Test _extract_text with valid response containing text."""
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(text="A story")]))]
    assert GeminiBackend._extract_text(mock_response) == "A story"


def test_extract_text_no_candidates_attribute():
    """Test _extract_text when response has no candidates attribute."""
    mock_response = MagicMock(spec=[])
    assert GeminiBackend._extract_text(mock_response) is None


def test_extract_text_empty_candidates():
    """Test _extract_text with empty candidates list."""
    mock_response = MagicMock()
    mock_response.candidates = []
    assert GeminiBackend._extract_text(mock_response) is None


def test_extract_text_content_none():
    """Test _extract_text when candidates[0].content is None."""
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock(content=None)]
    assert GeminiBackend._extract_text(mock_response) is None


def test_extract_text_empty_parts():
    """Test _extract_text with empty parts list."""
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock(content=MagicMock(parts=[]))]
    assert GeminiBackend._extract_text(mock_response) is None


def test_extract_text_text_none():
    """Test _extract_text when parts[0].text is None."""
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(text=None)]))]
    assert GeminiBackend._extract_text(mock_response) is None


def test_extract_text_strips_whitespace():
    """Test _extract_text strips leading/trailing whitespace."""
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(text="  hello  ")]))]
    assert GeminiBackend._extract_text(mock_response) == "hello"


class TestGeminiRetryIntegration:
    """Test that GeminiBackend integrates retry logic for transient errors."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @patch("storyforge.gemini_backend.genai.Client")
    @patch("storyforge.llm_backend.time.sleep")
    @patch("storyforge.llm_backend.random.uniform", return_value=0)
    @patch("storyforge.console.console")
    def test_generate_story_retries_on_503(self, mock_console, mock_random, mock_sleep, mock_client):
        """generate_story() should retry on 503 and succeed on second attempt."""
        backend = GeminiBackend()

        success_response = MagicMock()
        success_response.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(text="A story")]))]

        backend.client.models.generate_content.side_effect = [
            Exception("503 UNAVAILABLE. This model is currently experiencing high demand."),
            success_response,
        ]

        prompt = Prompt(prompt="test prompt")
        result = backend.generate_story(prompt)

        assert result == "A story"
        assert backend.client.models.generate_content.call_count == 2
        mock_sleep.assert_called_once()

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @patch("storyforge.gemini_backend.genai.Client")
    def test_generate_story_no_retry_on_permanent_error(self, mock_client):
        """generate_story() should NOT retry on non-transient errors like 400."""
        backend = GeminiBackend()
        backend.client.models.generate_content.side_effect = Exception("400 Bad Request: Invalid content")

        prompt = Prompt(prompt="test prompt")
        result = backend.generate_story(prompt)

        assert result.startswith(ERROR_STORY_SENTINEL)
        assert backend.client.models.generate_content.call_count == 1

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @patch("storyforge.gemini_backend.genai.Client")
    @patch("storyforge.llm_backend.time.sleep")
    @patch("storyforge.llm_backend.random.uniform", return_value=0)
    @patch("storyforge.console.console")
    def test_generate_story_returns_sentinel_after_retry_exhaustion(
        self, mock_console, mock_random, mock_sleep, mock_client
    ):
        """generate_story() should return error sentinel after all retries fail."""
        backend = GeminiBackend()
        backend.client.models.generate_content.side_effect = Exception("503 UNAVAILABLE")

        prompt = Prompt(prompt="test prompt")
        result = backend.generate_story(prompt)

        assert result.startswith(ERROR_STORY_SENTINEL)
        assert "503" in result
        assert backend.client.models.generate_content.call_count == 4  # 1 initial + 3 retries

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"})
    @patch("storyforge.gemini_backend.genai.Client")
    @patch("storyforge.llm_backend.time.sleep")
    @patch("storyforge.llm_backend.random.uniform", return_value=0)
    @patch("storyforge.console.console")
    def test_generate_image_retries_on_503(self, mock_console, mock_random, mock_sleep, mock_client):
        """generate_image() should retry on 503."""
        backend = GeminiBackend()

        success_response = MagicMock()
        success_response.generated_images = None
        mock_part = MagicMock()
        mock_part.inline_data = MagicMock(data=b"image_bytes")
        mock_part.text = None
        success_response.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]

        backend.client.models.generate_content.side_effect = [
            Exception("503 UNAVAILABLE"),
            success_response,
        ]

        prompt = Prompt(prompt="test prompt")
        with patch("storyforge.gemini_backend.Image.open", return_value="image_obj"):
            image, image_bytes = backend.generate_image(prompt)

        assert image == "image_obj"
        assert backend.client.models.generate_content.call_count == 2
