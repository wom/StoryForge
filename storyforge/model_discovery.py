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

    Uses version-aware ranking when models are available, falling back to
    the cross-generation alias `gemini-flash-latest`.

    Args:
        models: Optional list of models from list_gemini_models(). If not
            provided, will call list_gemini_models() automatically.

    Returns:
        Model name string. Falls back to "gemini-flash-latest" if no
        suitable model is found via ranking.
    """
    from storyforge.model_ranking import rank_models

    if models is None:
        try:
            models = list_gemini_models()
        except Exception:
            logger.debug("Could not list models for image model discovery, using default")
            return "gemini-flash-latest"

    # Filter to models that support generateContent
    eligible = [m for m in models if "generateContent" in m.get("supported_generation_methods", [])]

    best = rank_models(eligible, "gemini", "image")
    if best:
        return best

    return "gemini-flash-latest"


def find_text_generation_model(models: list[dict[str, Any]] | None = None) -> str:
    """Find the best available text generation model.

    Uses version-aware ranking when models are available, falling back to
    the cross-generation alias `gemini-pro-latest`.

    Args:
        models: Optional list of models from list_gemini_models().

    Returns:
        Model name string. Falls back to "gemini-pro-latest" if no model found.
    """
    from storyforge.model_ranking import rank_models

    if models is None:
        try:
            models = list_gemini_models()
        except Exception:
            logger.debug("Could not list models for text model discovery, using default")
            return "gemini-pro-latest"

    # Filter to models that support generateContent
    eligible = [m for m in models if "generateContent" in m.get("supported_generation_methods", [])]

    best = rank_models(eligible, "gemini", "text")
    if best:
        return best

    return "gemini-pro-latest"


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

    Uses version-aware ranking when models are available, falling back to
    the highest known versionless alias `gpt-5.4`.

    Args:
        models: Optional list of models from list_openai_models().

    Returns:
        Model name string. Falls back to "gpt-5.4" if no model found.
    """
    from storyforge.model_ranking import rank_models

    if models is None:
        try:
            models = list_openai_models()
        except Exception:
            logger.debug("Could not list models for OpenAI text model discovery, using default")
            return "gpt-5.4"

    best = rank_models(models, "openai", "text")
    if best:
        return best

    return "gpt-5.4"


def find_openai_image_model(models: list[dict[str, Any]] | None = None) -> str:
    """Find the best available OpenAI image generation model.

    Uses version-aware ranking when models are available, falling back to
    "gpt-image-1.5".

    Args:
        models: Optional list of models from list_openai_models().

    Returns:
        Model name string. Falls back to "gpt-image-1.5" if no model found.
    """
    from storyforge.model_ranking import rank_models

    if models is None:
        try:
            models = list_openai_models()
        except Exception:
            logger.debug("Could not list models for OpenAI image model discovery, using default")
            return "gpt-image-1.5"

    best = rank_models(models, "openai", "image")
    if best:
        return best

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

    Uses version-aware ranking when models are available, falling back to
    the versionless alias `claude-sonnet-4-6`.

    Args:
        models: Optional list of models from list_anthropic_models().

    Returns:
        Model name string. Falls back to "claude-sonnet-4-6" if no model found.
    """
    from storyforge.model_ranking import rank_models

    if models is None:
        try:
            models = list_anthropic_models()
        except Exception:
            logger.debug("Could not list models for Anthropic text model discovery, using default")
            return "claude-sonnet-4-6"

    best = rank_models(models, "anthropic", "text")
    if best:
        return best

    return "claude-sonnet-4-6"
