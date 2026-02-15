import os
from unittest.mock import MagicMock, patch

import pytest

from storyforge.llm_backend import LLMBackend, get_backend
from storyforge.prompt import Prompt


class DummyBackend(LLMBackend):
    name = "dummy"

    def generate_story(self, prompt: Prompt) -> str:
        raise NotImplementedError("Test implementation")

    def generate_image(self, prompt: Prompt):
        raise NotImplementedError("Test implementation")

    def generate_image_name(self, prompt: Prompt, story: str) -> str:
        raise NotImplementedError("Test implementation")

    def generate_image_prompt(self, prompt: Prompt) -> str:
        raise NotImplementedError("Test implementation")


# Original abstract base class tests
def test_generate_story_not_implemented():
    backend = DummyBackend()
    prompt = Prompt(prompt="test prompt")
    with pytest.raises(NotImplementedError):
        backend.generate_story(prompt)


def test_generate_image_not_implemented():
    backend = DummyBackend()
    prompt = Prompt(prompt="test prompt")
    with pytest.raises(NotImplementedError):
        backend.generate_image(prompt)


def test_generate_image_name_not_implemented():
    backend = DummyBackend()
    prompt = Prompt(prompt="test prompt")
    with pytest.raises(NotImplementedError):
        backend.generate_image_name(prompt, "story")


class TestTextInputLimit:
    """Test text_input_limit property and context budget."""

    def test_default_text_input_limit(self):
        """Test default text_input_limit when _text_input_limit is not set."""
        backend = DummyBackend()
        assert backend.text_input_limit == 8192

    def test_custom_text_input_limit(self):
        """Test text_input_limit reads from _text_input_limit."""
        backend = DummyBackend()
        backend._text_input_limit = 200000
        assert backend.text_input_limit == 200000

    def test_get_context_token_budget(self):
        """Test context budget is 50% of text_input_limit."""
        backend = DummyBackend()
        backend._text_input_limit = 128000
        assert backend.get_context_token_budget() == 64000

    def test_get_context_token_budget_default(self):
        """Test context budget with default limit."""
        backend = DummyBackend()
        # Default is 8192, budget is 50% = 4096
        assert backend.get_context_token_budget() == 4096

    def test_context_budget_ratio(self):
        """Test CONTEXT_BUDGET_RATIO is 0.50."""
        assert LLMBackend.CONTEXT_BUDGET_RATIO == 0.50

    def test_compression_trigger_ratio(self):
        """Test COMPRESSION_TRIGGER_RATIO is 0.80."""
        assert LLMBackend.COMPRESSION_TRIGGER_RATIO == 0.80

    def test_check_and_truncate_prompt_under_limit(self):
        """Test that prompts under limit are returned unchanged."""
        backend = DummyBackend()
        backend._text_input_limit = 100000
        short_prompt = "A short prompt"
        result = backend._check_and_truncate_prompt(short_prompt)
        assert result == short_prompt

    def test_check_and_truncate_prompt_over_limit(self):
        """Test that prompts over 80% of limit are truncated."""
        backend = DummyBackend()
        backend._text_input_limit = 100  # 100 tokens = 400 chars
        # 80% trigger = 80 tokens = 320 chars
        long_prompt = "x" * 400  # 100 tokens, over 80% trigger
        result = backend._check_and_truncate_prompt(long_prompt)
        assert len(result) == 320  # 80 tokens * 4 chars


# Factory function tests
class TestGetBackend:
    """Test the get_backend factory function."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True)
    @patch("storyforge.gemini_backend.GeminiBackend")
    def test_get_backend_auto_detect_gemini(self, mock_gemini):
        """Test auto-detection of Gemini backend via API key."""
        mock_instance = MagicMock()
        mock_gemini.return_value = mock_instance

        backend = get_backend()

        mock_gemini.assert_called_once()
        assert backend == mock_instance

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True)
    @patch("storyforge.openai_backend.OpenAIBackend")
    def test_get_backend_auto_detect_openai(self, mock_openai):
        """Test auto-detection of OpenAI backend via API key."""
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance

        backend = get_backend()

        mock_openai.assert_called_once()
        assert backend == mock_instance

    @patch.dict(os.environ, {"LLM_BACKEND": "gemini", "GEMINI_API_KEY": "test_key"}, clear=True)
    @patch("storyforge.gemini_backend.GeminiBackend")
    def test_get_backend_explicit_gemini(self, mock_gemini):
        """Test explicit Gemini backend selection."""
        mock_instance = MagicMock()
        mock_gemini.return_value = mock_instance

        backend = get_backend()

        mock_gemini.assert_called_once()
        assert backend == mock_instance

    @patch.dict(os.environ, {}, clear=True)
    def test_get_backend_no_api_keys(self):
        """Test error when no API keys are available."""
        with pytest.raises(RuntimeError) as exc_info:
            get_backend()

        assert "No LLM backend available" in str(exc_info.value)
        assert "GEMINI_API_KEY" in str(exc_info.value)
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}, clear=True)
    @patch("storyforge.anthropic_backend.AnthropicBackend")
    def test_get_backend_auto_detect_anthropic(self, mock_anthropic):
        """Test auto-detection of Anthropic backend via API key."""
        mock_instance = MagicMock()
        mock_anthropic.return_value = mock_instance

        backend = get_backend()

        mock_anthropic.assert_called_once()
        assert backend == mock_instance

    @patch.dict(os.environ, {"LLM_BACKEND": "anthropic", "ANTHROPIC_API_KEY": "test_key"}, clear=True)
    @patch("storyforge.anthropic_backend.AnthropicBackend")
    def test_get_backend_explicit_anthropic(self, mock_anthropic):
        """Test explicit Anthropic backend selection."""
        mock_instance = MagicMock()
        mock_anthropic.return_value = mock_instance

        backend = get_backend()

        mock_anthropic.assert_called_once()
        assert backend == mock_instance

    @patch.dict(os.environ, {"LLM_BACKEND": "openai", "OPENAI_API_KEY": "test_key"}, clear=True)
    @patch("storyforge.openai_backend.OpenAIBackend")
    def test_get_backend_explicit_openai(self, mock_openai):
        """Test explicit OpenAI backend selection."""
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance

        backend = get_backend()

        mock_openai.assert_called_once()
        assert backend == mock_instance

    @patch.dict(os.environ, {"LLM_BACKEND": "unknown"}, clear=True)
    def test_get_backend_unknown_backend(self):
        """Test error for unknown backends."""
        with pytest.raises(RuntimeError) as exc_info:
            get_backend()

        error_msg = str(exc_info.value)
        assert "Unknown backend 'unknown'" in error_msg
        assert "Supported backends: gemini, openai, anthropic" in error_msg

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True)
    @patch("storyforge.openai_backend.OpenAIBackend")
    def test_get_backend_explicit_parameter(self, mock_openai):
        """Test explicit backend parameter overrides environment."""
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance

        backend = get_backend("openai")

        mock_openai.assert_called_once()
        assert backend == mock_instance

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True)
    def test_get_backend_import_error_handling(self):
        """Test handling of import errors."""
        with patch("builtins.__import__", side_effect=ImportError("Module not found")):
            with pytest.raises(ImportError) as exc_info:
                get_backend("gemini")

            assert "Failed to import gemini backend" in str(exc_info.value)
