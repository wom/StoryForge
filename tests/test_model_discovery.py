"""Tests for Gemini model discovery functionality."""

from unittest.mock import MagicMock, patch

from storyforge.model_discovery import find_image_generation_model, find_text_generation_model, list_gemini_models


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
