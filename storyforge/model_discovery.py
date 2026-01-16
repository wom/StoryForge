"""Dynamic model discovery for Gemini API.

Provides utilities to list available models and select appropriate models
for different tasks (text generation, image generation, etc.).
"""

import logging
import os
from typing import Any

from google import genai

logger = logging.getLogger(__name__)


def list_gemini_models(api_key: str | None = None) -> list[dict[str, Any]]:
    """List all available Gemini models.

    Args:
        api_key: Optional API key. If not provided, uses GEMINI_API_KEY env var.

    Returns:
        List of model information dictionaries with keys: name, display_name,
        supported_generation_methods, description.

    Raises:
        RuntimeError: If API key is not provided and not found in environment.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")

    client = genai.Client(api_key=key)
    models = []

    try:
        for model in client.models.list():
            model_info = {
                "name": getattr(model, "name", ""),
                "display_name": getattr(model, "display_name", ""),
                "supported_generation_methods": getattr(model, "supported_generation_methods", []),
                "description": getattr(model, "description", ""),
                "input_token_limit": getattr(model, "input_token_limit", None),
                "output_token_limit": getattr(model, "output_token_limit", None),
            }
            models.append(model_info)
    except Exception:
        # If listing fails, return empty list
        pass

    return models


def find_image_generation_model(models: list[dict[str, Any]] | None = None) -> str:
    """Find the best available image generation model.

    Searches for models that support 'generateContent' and are suitable
    for image generation. Prioritizes newer models.

    Args:
        models: Optional list of models from list_gemini_models(). If not
            provided, will call list_gemini_models() automatically.

    Returns:
        Model name string (e.g., "gemini-2.5-flash-image"). Falls back to
        "gemini-2.5-flash-image" if no suitable model is found.
    """
    if models is None:
        try:
            models = list_gemini_models()
        except Exception:
            # If we can't list models, use the known good default
            return "gemini-2.5-flash-image"

    # Priority order of model name patterns to search for
    # Prefer stable models over preview models to avoid quota issues
    # Based on Gemini API documentation showing gemini-2.5-flash-image as current
    priority_patterns = [
        "gemini-2.5-flash-image",  # Stable model (preferred)
        "gemini-2.0-flash-image",  # Potential future stable model
        "imagen-",  # Legacy imagen models
    ]

    for pattern in priority_patterns:
        for model in models:
            name = model.get("name", "")
            methods = model.get("supported_generation_methods", [])

            # Skip preview models to avoid quota issues on free tier
            if "preview" in name.lower():
                logger.debug(f"Skipping preview model for stability: {name}")
                continue

            # Check if model name matches pattern and supports generateContent
            if pattern in name and "generateContent" in methods:
                # Extract just the model ID (remove "models/" prefix if present)
                if name.startswith("models/"):
                    model_id: str = name[7:]
                    return model_id
                result: str = name
                return result

    # Fallback to documented default
    return "gemini-2.5-flash-image"


def find_text_generation_model(models: list[dict[str, Any]] | None = None) -> str:
    """Find the best available text generation model.

    Args:
        models: Optional list of models from list_gemini_models().

    Returns:
        Model name string. Falls back to "gemini-2.5-pro" if no model found.
    """
    if models is None:
        try:
            models = list_gemini_models()
        except Exception:
            return "gemini-2.5-pro"

    # Priority order for text models
    # Prefer stable models over preview models to avoid quota issues
    priority_patterns = [
        "gemini-2.5-pro",  # Current flagship model
        "gemini-2.5-flash",  # Fast stable model
        "gemini-2.0-pro",  # Potential future model
        "gemini-pro",  # Legacy fallback
    ]

    for pattern in priority_patterns:
        for model in models:
            name = model.get("name", "")
            methods = model.get("supported_generation_methods", [])

            # Skip preview models to avoid quota issues on free tier
            if "preview" in name.lower():
                logger.debug(f"Skipping preview model for stability: {name}")
                continue

            if pattern in name and "generateContent" in methods:
                if name.startswith("models/"):
                    model_id: str = name[7:]
                    return model_id
                result: str = name
                return result

    return "gemini-2.5-pro"
