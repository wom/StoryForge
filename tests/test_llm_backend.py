import os
from unittest.mock import MagicMock, patch

import pytest

from storyforge.llm_backend import LLMBackend, get_backend, list_available_backends
from storyforge.prompt import Prompt


class DummyBackend(LLMBackend):
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


class TestListAvailableBackends:
    """Test the list_available_backends function."""

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True)
    def test_list_backends_gemini_available(self):
        """Test listing when Gemini is available."""
        with patch("builtins.__import__"):  # Mock the import
            backends = list_available_backends()

            assert backends["gemini"]["available"] is True
            assert backends["gemini"]["reason"] == "Ready"
            assert backends["openai"]["available"] is False
            assert "OPENAI_API_KEY not set" in backends["openai"]["reason"]
            assert backends["anthropic"]["available"] is False
            assert "ANTHROPIC_API_KEY not set" in backends["anthropic"]["reason"]

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}, clear=True)
    def test_list_backends_anthropic_available(self):
        """Test listing when Anthropic is available."""
        with patch("builtins.__import__"):  # Mock the import
            backends = list_available_backends()

            assert backends["anthropic"]["available"] is True
            assert backends["anthropic"]["reason"] == "Ready"
            assert backends["gemini"]["available"] is False
            assert backends["openai"]["available"] is False
            assert "GEMINI_API_KEY not set" in backends["gemini"]["reason"]

    @patch.dict(os.environ, {}, clear=True)
    def test_list_backends_gemini_no_key(self):
        """Test listing when Gemini package available but no API key."""
        with patch("builtins.__import__"):  # Mock the import
            backends = list_available_backends()

            assert backends["gemini"]["available"] is False
            assert "GEMINI_API_KEY not set" in backends["gemini"]["reason"]

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}, clear=True)
    def test_list_backends_anthropic_import_error(self):
        """Test listing when Anthropic package not installed."""
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            backends = list_available_backends()

            assert backends["anthropic"]["available"] is False
            assert "anthropic package not installed" in backends["anthropic"]["reason"]

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True)
    def test_list_backends_gemini_import_error(self):
        """Test listing when Gemini package not installed."""
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            backends = list_available_backends()

            assert backends["gemini"]["available"] is False
            assert "google-genai package not installed" in backends["gemini"]["reason"]

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True)
    def test_list_backends_openai_available(self):
        """Test listing when OpenAI is available."""
        with patch("builtins.__import__"):  # Mock the import
            backends = list_available_backends()

            assert backends["openai"]["available"] is True
            assert backends["openai"]["reason"] == "Ready"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True)
    def test_list_backends_openai_import_error(self):
        """Test listing when OpenAI package not installed."""
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            backends = list_available_backends()

            assert backends["openai"]["available"] is False
            assert "openai package not installed" in backends["openai"]["reason"]
