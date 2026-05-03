"""Tests for model discovery functionality across all providers."""

import os
from unittest.mock import MagicMock, patch

from storyforge.model_discovery import (
    find_anthropic_text_model,
    find_image_generation_model,
    find_openai_image_model,
    find_openai_text_model,
    find_text_generation_model,
    list_anthropic_models,
    list_gemini_models,
    list_openai_models,
)


def test_find_image_generation_model_with_gemini_2_5():
    """Test finding gemini-2.5-flash-image model."""
    models = [
        {
            "name": "models/gemini-2.5-flash-image",
            "display_name": "Gemini 2.5 Flash Image",
            "supported_generation_methods": ["generateContent"],
            "description": "Fast image generation model",
        },
        {
            "name": "models/gemini-2.5-pro",
            "display_name": "Gemini 2.5 Pro",
            "supported_generation_methods": ["generateContent"],
            "description": "Text generation model",
        },
    ]

    model = find_image_generation_model(models)
    assert model == "gemini-2.5-flash-image"


def test_find_image_generation_model_fallback():
    """Test fallback when no image model found."""
    models = [
        {
            "name": "models/gemini-2.5-pro",
            "display_name": "Gemini 2.5 Pro",
            "supported_generation_methods": ["generateContent"],
            "description": "Text generation model",
        }
    ]

    model = find_image_generation_model(models)
    assert model == "gemini-2.5-flash-image"  # Falls back to default


def test_find_image_generation_model_with_imagen():
    """Test finding legacy imagen model."""
    models = [
        {
            "name": "models/imagen-4.0-generate-001",
            "display_name": "Imagen 4.0",
            "supported_generation_methods": ["generateContent"],
            "description": "Image generation model",
        }
    ]

    model = find_image_generation_model(models)
    assert model == "imagen-4.0-generate-001"


def test_find_image_generation_model_skips_preview():
    """Test that preview models are skipped in favor of stable models."""
    models = [
        {
            "name": "models/gemini-2.5-flash-preview-image",
            "display_name": "Gemini 2.5 Flash Image Preview",
            "supported_generation_methods": ["generateContent"],
            "description": "Preview image generation model",
        },
        {
            "name": "models/gemini-2.5-flash-image",
            "display_name": "Gemini 2.5 Flash Image",
            "supported_generation_methods": ["generateContent"],
            "description": "Stable image generation model",
        },
    ]

    model = find_image_generation_model(models)
    assert model == "gemini-2.5-flash-image"
    assert "preview" not in model.lower()


def test_find_text_generation_model_skips_preview():
    """Test that preview models are skipped for text generation."""
    models = [
        {
            "name": "models/gemini-2.5-pro-preview",
            "display_name": "Gemini 2.5 Pro Preview",
            "supported_generation_methods": ["generateContent"],
            "description": "Preview text model",
        },
        {
            "name": "models/gemini-2.5-pro",
            "display_name": "Gemini 2.5 Pro",
            "supported_generation_methods": ["generateContent"],
            "description": "Stable text model",
        },
    ]

    model = find_text_generation_model(models)
    assert model == "gemini-2.5-pro"
    assert "preview" not in model.lower()


def test_find_text_generation_model():
    """Test finding gemini-2.5-pro text model."""
    models = [
        {
            "name": "models/gemini-2.5-pro",
            "display_name": "Gemini 2.5 Pro",
            "supported_generation_methods": ["generateContent"],
            "description": "Text generation model",
        }
    ]

    model = find_text_generation_model(models)
    assert model == "gemini-2.5-pro"


def test_find_text_generation_model_fallback():
    """Test fallback when no text model found."""
    models = []

    model = find_text_generation_model(models)
    assert model == "gemini-2.5-pro"  # Falls back to default


@patch("storyforge.model_discovery.genai.Client")
def test_list_gemini_models(mock_client_class):
    """Test listing Gemini models from API."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Mock model objects
    mock_model1 = MagicMock()
    mock_model1.name = "models/gemini-2.5-pro"
    mock_model1.display_name = "Gemini 2.5 Pro"
    mock_model1.supported_generation_methods = ["generateContent"]
    mock_model1.description = "Text model"

    mock_model2 = MagicMock()
    mock_model2.name = "models/gemini-2.5-flash-image"
    mock_model2.display_name = "Gemini 2.5 Flash Image"
    mock_model2.supported_generation_methods = ["generateContent"]
    mock_model2.description = "Image model"

    mock_client.models.list.return_value = [mock_model1, mock_model2]

    models = list_gemini_models("fake-api-key")

    assert len(models) == 2
    assert models[0]["name"] == "models/gemini-2.5-pro"
    assert models[1]["name"] == "models/gemini-2.5-flash-image"
    assert "generateContent" in models[0]["supported_generation_methods"]


@patch("storyforge.model_discovery.genai.Client")
def test_list_gemini_models_error_handling(mock_client_class):
    """Test that list_gemini_models handles errors gracefully."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.models.list.side_effect = Exception("API error")

    models = list_gemini_models("fake-api-key")

    assert models == []  # Returns empty list on error


def test_list_gemini_models_no_api_key():
    """Test that list_gemini_models raises error when no API key provided."""
    import os

    # Temporarily remove API key if it exists
    old_key = os.environ.get("GEMINI_API_KEY")
    if old_key:
        del os.environ["GEMINI_API_KEY"]

    try:
        import pytest

        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            list_gemini_models()
    finally:
        # Restore API key
        if old_key:
            os.environ["GEMINI_API_KEY"] = old_key


class TestOpenAIDiscovery:
    """Tests for OpenAI model discovery functions."""

    @patch.dict("sys.modules", {"openai": MagicMock()})
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_list_openai_models(self):
        """Mock openai client, verify filtering."""
        import sys

        mock_openai = sys.modules["openai"]
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        mock_gpt = MagicMock()
        mock_gpt.id = "gpt-4o"
        mock_gpt.owned_by = "openai"
        mock_gpt.created = 1700000000

        mock_dalle = MagicMock()
        mock_dalle.id = "dall-e-3"
        mock_dalle.owned_by = "openai"
        mock_dalle.created = 1690000000

        mock_whisper = MagicMock()
        mock_whisper.id = "whisper-1"
        mock_whisper.owned_by = "openai"
        mock_whisper.created = 1680000000

        mock_client.models.list.return_value = [mock_gpt, mock_dalle, mock_whisper]

        models = list_openai_models("test-key")
        # whisper should be filtered out
        model_names = [m["name"] for m in models]
        assert "gpt-4o" in model_names
        assert "dall-e-3" in model_names
        assert "whisper-1" not in model_names

    @patch.dict("sys.modules", {"openai": MagicMock()})
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_list_openai_models_error_handling(self):
        """Mock client to raise, verify empty list."""
        import sys

        mock_openai = sys.modules["openai"]
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.models.list.side_effect = Exception("API error")

        models = list_openai_models("test-key")
        assert models == []

    @patch.dict(os.environ, {}, clear=True)
    def test_list_openai_models_no_api_key(self, monkeypatch):
        """No env var, returns empty."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        models = list_openai_models()
        assert models == []

    def test_find_openai_text_model(self):
        """Given model list, finds best text model."""
        models = [
            {"name": "gpt-4o", "owned_by": "openai", "created": 1700000000},
            {"name": "gpt-5.2", "owned_by": "openai", "created": 1710000000},
        ]
        result = find_openai_text_model(models)
        assert result == "gpt-5.2"

    def test_find_openai_text_model_skips_mini(self):
        """Mini models are skipped."""
        models = [
            {"name": "gpt-5.2-mini", "owned_by": "openai", "created": 1710000000},
            {"name": "gpt-4o", "owned_by": "openai", "created": 1700000000},
        ]
        result = find_openai_text_model(models)
        assert result == "gpt-4o"
        assert "mini" not in result

    def test_find_openai_text_model_fallback(self):
        """Empty list returns default."""
        result = find_openai_text_model([])
        assert result == "gpt-5.2"

    def test_find_openai_image_model(self):
        """Finds best image model."""
        models = [
            {"name": "dall-e-3", "owned_by": "openai", "created": 1690000000},
            {"name": "gpt-image-1.5", "owned_by": "openai", "created": 1710000000},
        ]
        result = find_openai_image_model(models)
        assert result == "gpt-image-1.5"

    def test_find_openai_image_model_fallback(self):
        """Empty list returns default."""
        result = find_openai_image_model([])
        assert result == "gpt-image-1.5"


class TestAnthropicDiscovery:
    """Tests for Anthropic model discovery functions."""

    @patch.dict("sys.modules", {"anthropic": MagicMock()})
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_list_anthropic_models(self):
        """Mock anthropic client, verify model listing."""
        import sys

        mock_anthropic = sys.modules["anthropic"]
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        mock_model1 = MagicMock()
        mock_model1.id = "claude-3-5-sonnet-20241022"
        mock_model1.display_name = "Claude 3.5 Sonnet"
        mock_model1.created_at = "2024-10-22T00:00:00Z"

        mock_model2 = MagicMock()
        mock_model2.id = "claude-3-haiku-20240307"
        mock_model2.display_name = "Claude 3 Haiku"
        mock_model2.created_at = "2024-03-07T00:00:00Z"

        mock_client.models.list.return_value = [mock_model1, mock_model2]

        models = list_anthropic_models("test-key")
        assert len(models) == 2
        assert models[0]["name"] == "claude-3-5-sonnet-20241022"
        assert models[1]["name"] == "claude-3-haiku-20240307"

    @patch.dict("sys.modules", {"anthropic": MagicMock()})
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_list_anthropic_models_error_handling(self):
        """Mock to raise, verify empty list."""
        import sys

        mock_anthropic = sys.modules["anthropic"]
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.models.list.side_effect = Exception("API error")

        models = list_anthropic_models("test-key")
        assert models == []

    @patch.dict(os.environ, {}, clear=True)
    def test_list_anthropic_models_no_api_key(self, monkeypatch):
        """No env var, returns empty."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        models = list_anthropic_models()
        assert models == []

    def test_find_anthropic_text_model(self):
        """Given model list, finds best."""
        models = [
            {"name": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet", "created_at": None},
            {"name": "claude-3-opus-20240229", "display_name": "Claude 3 Opus", "created_at": None},
        ]
        result = find_anthropic_text_model(models)
        assert result == "claude-3-5-sonnet-20241022"

    def test_find_anthropic_text_model_skips_haiku(self):
        """Haiku models skipped."""
        models = [
            {"name": "claude-3-haiku-20240307", "display_name": "Claude 3 Haiku", "created_at": None},
            {"name": "claude-3-opus-20240229", "display_name": "Claude 3 Opus", "created_at": None},
        ]
        result = find_anthropic_text_model(models)
        assert result == "claude-3-opus-20240229"
        assert "haiku" not in result

    def test_find_anthropic_text_model_fallback(self):
        """Empty list returns default."""
        result = find_anthropic_text_model([])
        assert result == "claude-3-5-sonnet-20241022"
