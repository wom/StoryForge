"""
Tests for the OpenAI backend implementation.
"""

import os
from unittest.mock import Mock, patch

import pytest

from storyforge.openai_backend import OpenAIBackend
from storyforge.prompt import Prompt


class TestOpenAIBackend:
    """Test cases for OpenAIBackend class."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_init_success(self):
        """Test successful initialization with API key."""
        backend = OpenAIBackend()
        assert backend.client is not None

    def test_init_no_api_key(self):
        """Test initialization failure without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY environment variable not set"):
                OpenAIBackend()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_generate_story_success(self, mock_openai_client):
        """Test successful story generation."""
        # Mock the OpenAI client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "A wonderful test story about friendship."

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_client.return_value = mock_client_instance

        backend = OpenAIBackend()
        prompt = Mock(spec=Prompt)
        prompt.story = "Tell me a story about friendship"

        result = backend.generate_story(prompt)

        assert result == "A wonderful test story about friendship."
        mock_client_instance.chat.completions.create.assert_called_once_with(
            model="gpt-5.2",
            messages=[{"role": "user", "content": "Tell me a story about friendship"}],
            temperature=1,
        )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_generate_story_no_response(self, mock_openai_client):
        """Test story generation with no response."""
        mock_response = Mock()
        mock_response.choices = []

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_client.return_value = mock_client_instance

        backend = OpenAIBackend()
        prompt = Mock(spec=Prompt)
        prompt.story = "Test prompt"

        result = backend.generate_story(prompt)

        assert result == "[Error: No valid response from OpenAI]"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_generate_story_exception(self, mock_openai_client):
        """Test story generation with exception."""
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_client.return_value = mock_client_instance

        backend = OpenAIBackend()
        prompt = Mock(spec=Prompt)
        prompt.story = "Test prompt"

        result = backend.generate_story(prompt)

        assert result.startswith("[Error generating story:")
        # The implementation returns error message with exception details
        assert result.startswith("[Error generating story:")
        assert "API Error" in result

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("openai.OpenAI")
    @patch("requests.get")
    def test_generate_image_success(self, mock_requests_get, mock_openai_client):
        """Test successful image generation with DALL-E."""
        # Mock the OpenAI client response
        mock_image_response = Mock()
        mock_image_response.data = [Mock()]
        mock_image_response.data[0].url = "https://example.com/image.png"

        mock_client_instance = Mock()
        mock_client_instance.images.generate.return_value = mock_image_response
        mock_openai_client.return_value = mock_client_instance

        # Mock the requests.get for downloading image
        mock_http_response = Mock()
        mock_http_response.status_code = 200
        mock_http_response.content = b"fake image data"
        mock_requests_get.return_value = mock_http_response

        # Mock PIL Image
        with patch("PIL.Image.open") as mock_image_open:
            mock_image = Mock()
            mock_image_open.return_value = mock_image

            backend = OpenAIBackend()
            prompt = Mock(spec=Prompt)
            prompt.image.return_value = ["A beautiful test image prompt"]

            image, image_bytes = backend.generate_image(prompt)

            assert image == mock_image
            assert image_bytes == b"fake image data"
            mock_client_instance.images.generate.assert_called_once_with(
                prompt="A beautiful test image prompt",
                model="gpt-image-1.5",
                size="1024x1024",
                quality="auto",
                n=1,
            )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("openai.OpenAI")
    @patch("requests.get")
    def test_generate_image_with_reference(self, mock_requests_get, mock_openai_client):
        """Test image generation with reference image bytes."""
        # Mock the OpenAI client response
        mock_image_response = Mock()
        mock_image_response.data = [Mock()]
        mock_image_response.data[0].url = "https://example.com/image.png"

        mock_client_instance = Mock()
        mock_client_instance.images.generate.return_value = mock_image_response
        mock_openai_client.return_value = mock_client_instance

        # Mock the requests.get for downloading image
        mock_http_response = Mock()
        mock_http_response.status_code = 200
        mock_http_response.content = b"fake image data"
        mock_requests_get.return_value = mock_http_response

        # Mock PIL Image
        with patch("PIL.Image.open") as mock_image_open:
            mock_image = Mock()
            mock_image_open.return_value = mock_image

            backend = OpenAIBackend()
            prompt = Mock(spec=Prompt)
            prompt.image.return_value = ["A test image prompt"]
            reference_bytes = b"reference image data"

            image, image_bytes = backend.generate_image(prompt, reference_bytes)

            assert image == mock_image
            assert image_bytes == b"fake image data"

            # Check that the prompt was modified to include style consistency
            call_args = mock_client_instance.images.generate.call_args
            assert "consistent visual style" in call_args[1]["prompt"]
            assert "A test image prompt" in call_args[1]["prompt"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_generate_image_no_response(self, mock_openai_client):
        """Test image generation with no response."""
        mock_image_response = Mock()
        mock_image_response.data = []

        mock_client_instance = Mock()
        mock_client_instance.images.generate.return_value = mock_image_response
        mock_openai_client.return_value = mock_client_instance

        backend = OpenAIBackend()
        prompt = Mock(spec=Prompt)
        prompt.image.return_value = ["Test prompt"]

        image, image_bytes = backend.generate_image(prompt)

        assert image is None
        assert image_bytes is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_generate_image_exception(self, mock_openai_client):
        """Test image generation with exception."""
        mock_client_instance = Mock()
        mock_client_instance.images.generate.side_effect = Exception("DALL-E Error")
        mock_openai_client.return_value = mock_client_instance

        backend = OpenAIBackend()
        prompt = Mock(spec=Prompt)
        prompt.image.return_value = ["Test prompt"]

        image, image_bytes = backend.generate_image(prompt)

        assert image is None
        assert image_bytes is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_generate_image_name_success(self, mock_openai_client):
        """Test successful image name generation."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "friendship_adventure.png"

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_client.return_value = mock_client_instance

        backend = OpenAIBackend()
        prompt = Mock(spec=Prompt)
        prompt.image_name.return_value = "Generate a name for this image"
        story = "A story about friendship"

        result = backend.generate_image_name(prompt, story)

        assert result == "friendship_adventure"
        mock_client_instance.chat.completions.create.assert_called_once_with(
            model="gpt-5.2",
            messages=[{"role": "user", "content": "Generate a name for this image"}],
            temperature=1,
        )

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_generate_image_name_with_special_chars(self, mock_openai_client):
        """Test image name generation with special characters cleanup."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "amazing-story@2024!.jpg"

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_client.return_value = mock_client_instance

        backend = OpenAIBackend()
        prompt = Mock(spec=Prompt)
        prompt.image_name.return_value = "Generate a name"
        story = "Test story"

        result = backend.generate_image_name(prompt, story)

        assert result == "amazingstory2024"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    @patch("openai.OpenAI")
    def test_generate_image_name_exception(self, mock_openai_client):
        """Test image name generation with exception."""
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_client.return_value = mock_client_instance

        backend = OpenAIBackend()
        prompt = Mock(spec=Prompt)
        prompt.image_name.return_value = "Test prompt"
        story = "Test story"

        result = backend.generate_image_name(prompt, story)

        assert result == "story_image"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_generate_image_prompt(self):
        """Test image prompt generation (stub implementation)."""
        backend = OpenAIBackend()

        story = "Once upon a time, there was a brave mouse.\nThe mouse went on an adventure.\nThe end."
        context = "A children's story about courage"
        num_prompts = 2

        result = backend.generate_image_prompt(story, context, num_prompts)

        assert len(result) == 2
        assert "Once upon a time, there was a brave mouse." in result[0]
        assert "The mouse went on an adventure." in result[1]
        assert "Context: A children's story about courage" in result[0]
        assert "Context: A children's story about courage" in result[1]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_generate_image_prompt_more_prompts_than_paragraphs(self):
        """Test image prompt generation with more prompts than paragraphs."""
        backend = OpenAIBackend()

        story = "A short story."
        context = "Test context"
        num_prompts = 3

        result = backend.generate_image_prompt(story, context, num_prompts)

        assert len(result) == 3
        assert "A short story." in result[0]
        # Should repeat the full story for additional prompts
        assert "A short story." in result[1]
        assert "A short story." in result[2]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False)
    def test_generate_image_prompt_no_context(self):
        """Test image prompt generation without context."""
        backend = OpenAIBackend()

        story = "Line one.\nLine two."
        context = ""
        num_prompts = 2

        result = backend.generate_image_prompt(story, context, num_prompts)

        assert len(result) == 2
        assert "Line one." in result[0]
        assert "Line two." in result[1]
        assert "Context:" not in result[0]
        assert "Context:" not in result[1]
