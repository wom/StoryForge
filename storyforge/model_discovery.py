"""Dynamic model discovery for LLM providers.

Provides utilities to list available models and select appropriate models
for different tasks (text generation, image generation, etc.) across
Gemini, OpenAI, and Anthropic providers.
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
        logger.warning("Failed to list Gemini models", exc_info=True)

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
            logger.debug("Could not list models for image model discovery, using default")
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
            logger.debug("Could not list models for text model discovery, using default")
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


def list_openai_models(api_key: str | None = None) -> list[dict[str, Any]]:
    """List available OpenAI models.

    Args:
        api_key: Optional API key. If not provided, uses OPENAI_API_KEY env var.

    Returns:
        List of model information dictionaries with keys: name, owned_by, created.
    """
    try:
        import openai
    except ImportError:
        logger.warning("openai package not installed, cannot list models")
        return []

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        logger.warning("OPENAI_API_KEY not set, cannot list models")
        return []

    client = openai.OpenAI(api_key=key)
    models: list[dict[str, Any]] = []
    allowed_patterns = ("gpt", "dall-e", "o1", "o3", "o4")

    try:
        for model in client.models.list():
            model_id = model.id
            if not any(pattern in model_id for pattern in allowed_patterns):
                logger.debug(f"Skipping non-relevant model: {model_id}")
                continue
            models.append(
                {
                    "name": model_id,
                    "owned_by": getattr(model, "owned_by", ""),
                    "created": getattr(model, "created", None),
                }
            )
    except Exception:
        logger.warning("Failed to list OpenAI models", exc_info=True)

    return models


def find_openai_text_model(models: list[dict[str, Any]] | None = None) -> str:
    """Find the best available OpenAI text generation model.

    Args:
        models: Optional list of models from list_openai_models().

    Returns:
        Model name string. Falls back to "gpt-5.2" if no model found.
    """
    if models is None:
        try:
            models = list_openai_models()
        except Exception:
            logger.debug("Could not list models for OpenAI text model discovery, using default")
            return "gpt-5.2"

    priority_patterns = [
        "gpt-5.2",
        "gpt-5",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4",
    ]

    for pattern in priority_patterns:
        for model in models:
            name: str = model.get("name", "")
            if "mini" in name.lower() or "nano" in name.lower():
                logger.debug(f"Skipping small model: {name}")
                continue
            if pattern in name:
                return name

    return "gpt-5.2"


def find_openai_image_model(models: list[dict[str, Any]] | None = None) -> str:
    """Find the best available OpenAI image generation model.

    Args:
        models: Optional list of models from list_openai_models().

    Returns:
        Model name string. Falls back to "gpt-image-1.5" if no model found.
    """
    if models is None:
        try:
            models = list_openai_models()
        except Exception:
            logger.debug("Could not list models for OpenAI image model discovery, using default")
            return "gpt-image-1.5"

    priority_patterns = [
        "gpt-image-1.5",
        "gpt-image-1",
        "dall-e-3",
        "dall-e-2",
    ]

    for pattern in priority_patterns:
        for model in models:
            name: str = model.get("name", "")
            if pattern in name:
                return name

    return "gpt-image-1.5"


def list_anthropic_models(api_key: str | None = None) -> list[dict[str, Any]]:
    """List available Anthropic models.

    Args:
        api_key: Optional API key. If not provided, uses ANTHROPIC_API_KEY env var.

    Returns:
        List of model information dictionaries with keys: name, display_name, created_at.
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed, cannot list models")
        return []

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        logger.warning("ANTHROPIC_API_KEY not set, cannot list models")
        return []

    client = anthropic.Anthropic(api_key=key)
    models: list[dict[str, Any]] = []

    try:
        for model in client.models.list():
            models.append(
                {
                    "name": getattr(model, "id", ""),
                    "display_name": getattr(model, "display_name", ""),
                    "created_at": getattr(model, "created_at", None),
                }
            )
    except Exception:
        logger.warning("Failed to list Anthropic models", exc_info=True)

    return models


def find_anthropic_text_model(models: list[dict[str, Any]] | None = None) -> str:
    """Find the best available Anthropic text generation model.

    Args:
        models: Optional list of models from list_anthropic_models().

    Returns:
        Model name string. Falls back to "claude-3-5-sonnet-20241022" if no model found.
    """
    if models is None:
        try:
            models = list_anthropic_models()
        except Exception:
            logger.debug("Could not list models for Anthropic text model discovery, using default")
            return "claude-3-5-sonnet-20241022"

    priority_patterns = [
        "claude-4-sonnet",
        "claude-4-opus",
        "claude-3-5-sonnet",
        "claude-3-opus",
    ]

    for pattern in priority_patterns:
        for model in models:
            name: str = model.get("name", "")
            if "haiku" in name.lower():
                logger.debug(f"Skipping low-quality model: {name}")
                continue
            if pattern in name:
                return name

    return "claude-3-5-sonnet-20241022"
