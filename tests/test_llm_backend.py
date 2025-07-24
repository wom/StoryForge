import os
from unittest.mock import MagicMock, patch

import pytest

from storytime.llm_backend import LLMBackend, get_backend, list_available_backends
from storytime.prompt import Prompt


class DummyBackend(LLMBackend):
    def generate_story(self, prompt: Prompt) -> str:
        raise NotImplementedError("Test implementation")

    def generate_image(self, prompt: Prompt):
        raise NotImplementedError("Test implementation")

    def generate_image_name(self, prompt: Prompt, story: str) -> str:
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
    @patch("storytime.gemini_backend.GeminiBackend")
    def test_get_backend_auto_detect_gemini(self, mock_gemini):
        """Test auto-detection of Gemini backend via API key."""
        mock_instance = MagicMock()
        mock_gemini.return_value = mock_instance

        backend = get_backend()

        mock_gemini.assert_called_once()
        assert backend == mock_instance

    @patch.dict(
        os.environ, {"LLM_BACKEND": "gemini", "GEMINI_API_KEY": "test_key"}, clear=True
    )
    @patch("storytime.gemini_backend.GeminiBackend")
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

    @patch.dict(os.environ, {"LLM_BACKEND": "openai"}, clear=True)
    def test_get_backend_unimplemented_backend(self):
        """Test error for unimplemented backends."""
        with pytest.raises(RuntimeError) as exc_info:
            get_backend()

        assert "OpenAI backend not yet implemented" in str(exc_info.value)

    @patch.dict(os.environ, {"LLM_BACKEND": "unknown"}, clear=True)
    def test_get_backend_unknown_backend(self):
        """Test error for unknown backends."""
        with pytest.raises(RuntimeError) as exc_info:
            get_backend()

        assert "Unknown backend 'unknown'" in str(exc_info.value)

    def test_get_backend_explicit_parameter(self):
        """Test explicit backend parameter overrides environment."""
        with pytest.raises(RuntimeError) as exc_info:
            get_backend("openai")

        assert "OpenAI backend not yet implemented" in str(exc_info.value)

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
            assert backends["anthropic"]["available"] is False

    @patch.dict(os.environ, {}, clear=True)
    def test_list_backends_gemini_no_key(self):
        """Test listing when Gemini package available but no API key."""
        with patch("builtins.__import__"):  # Mock the import
            backends = list_available_backends()

            assert backends["gemini"]["available"] is False
            assert "GEMINI_API_KEY not set" in backends["gemini"]["reason"]

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}, clear=True)
    def test_list_backends_gemini_import_error(self):
        """Test listing when Gemini package not installed."""
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            backends = list_available_backends()

            assert backends["gemini"]["available"] is False
            assert "google-genai package not installed" in backends["gemini"]["reason"]
