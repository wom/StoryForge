"""
LLM Backend Factory and Abstract Base Class.

This module provides:
1. LLMBackend abstract base class defining the interface for all backends
2. get_backend() factory function for automatic backend selection
3. Backend detection logic based on environment variables

Supported backends:
- Gemini (Google AI): requires GEMINI_API_KEY
- Future: OpenAI, Anthropic, etc.
"""

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .prompt import Prompt


class LLMBackend(ABC):
    """
    Abstract base class for Large Language Model backends.

    This class defines the interface that all LLM backends must implement
    to provide story generation, image generation, and image naming capabilities.
    """

    name: str

    @abstractmethod
    def generate_story(self, prompt: "Prompt") -> str:
        """
        Generate a story based on the given Prompt object.

        Args:
            prompt (Prompt): A Prompt object containing comprehensive story
                generation parameters including context, style, tone, etc.

        Returns:
            str: The generated story text.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclass must implement generate_story method")

    @abstractmethod
    def generate_image(
        self, prompt: "Prompt", reference_image_bytes: bytes | None = None
    ) -> tuple[object | None, bytes | None]:
        """
        Generate an image based on the given Prompt object.

        Args:
            prompt (Prompt): A Prompt object containing comprehensive parameters
                for image generation including style, tone, setting, etc.
            reference_image_bytes (Optional[bytes]): Reference image bytes to maintain
                consistency with previous images.

        Returns:
            Tuple[Optional[object], Optional[bytes]]: A tuple containing:
                - The image object (PIL.Image or similar), or None if generation failed
                - The raw image bytes, or None if generation failed

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclass must implement generate_image method")

    @abstractmethod
    def generate_image_name(self, prompt: "Prompt", story: str) -> str:
        """
        Generate a descriptive filename for an image based on the Prompt and story.

        Args:
            prompt (Prompt): A Prompt object containing the original parameters.
            story (str): The generated story text.

        Returns:
            str: A suggested filename (without extension) that is descriptive
                 and filesystem-safe (no spaces or special characters).

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclass must implement generate_image_name method")

    @abstractmethod
    def generate_image_prompt(self, story: str, context: str, num_prompts: int) -> list[str]:
        """
        Break the given story into `num_prompts` image prompts that each progressively tell the story.
        Each prompt should be incredibly detailed, with a strong focus on small things
        (e.g., hair color, breed of dog, glasses).

        Args:
            story (str): The generated story to break into image prompts.
            context (str): Additional context for the story.
            num_prompts (int): The number of image prompts to return.

        Returns:
            list[str]: A list of image prompts, each describing a detailed scene from the story.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclass must implement generate_image_prompt method")


def get_backend(backend_name: str | None = None) -> LLMBackend:
    """
    Factory function to get an appropriate LLM backend instance.

    Args:
        backend_name (Optional[str]): Specific backend to use ("gemini",
                                      "openai", etc.).
                                    If None, auto-detect based on available API keys.

    Returns:
        LLMBackend: An instance of the selected backend.

    Raises:
        RuntimeError: If no suitable backend is found or API keys are missing.
        ImportError: If required dependencies for the backend are not installed.

    Environment Variables:
        LLM_BACKEND: Override automatic detection (e.g., "gemini", "openai")
        GEMINI_API_KEY: Required for Gemini backend
        OPENAI_API_KEY: Required for OpenAI backend
        ANTHROPIC_API_KEY: Required for Anthropic backend (future)
    """

    # Override backend selection if explicitly requested
    if backend_name is None:
        backend_name = os.environ.get("LLM_BACKEND", "").lower()

    # If still no explicit backend, auto-detect based on available API keys
    if not backend_name:
        if os.environ.get("GEMINI_API_KEY"):
            backend_name = "gemini"
        elif os.environ.get("OPENAI_API_KEY"):
            backend_name = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            backend_name = "anthropic"
        else:
            raise RuntimeError(
                "No LLM backend available. Please set one of the following "
                "environment variables:\n"
                "- GEMINI_API_KEY (for Google Gemini)\n"
                "- OPENAI_API_KEY (for OpenAI)\n"
                "- ANTHROPIC_API_KEY (for Anthropic - future)\n"
                "Or set LLM_BACKEND to specify which backend to use."
            )

    # Import and instantiate the requested backend
    try:
        if backend_name == "gemini":
            from .gemini_backend import GeminiBackend

            return GeminiBackend()

        elif backend_name == "openai":
            from .openai_backend import OpenAIBackend

            return OpenAIBackend()

        elif backend_name == "anthropic":
            from .anthropic_backend import AnthropicBackend

            return AnthropicBackend()

        else:
            raise RuntimeError(
                f"Unknown backend '{backend_name}'. Supported backends: gemini, openai (more coming soon)"
            )

    except ImportError as e:
        raise ImportError(
            f"Failed to import {backend_name} backend. Please ensure required dependencies are installed: {e}"
        ) from e


def list_available_backends() -> dict:
    """
    List all available backends and their status.

    Returns:
        dict: Backend name -> {"available": bool, "reason": str}
    """
    backends = {}

    # Check Gemini
    try:
        import google.genai  # noqa: F401

        has_key = bool(os.environ.get("GEMINI_API_KEY"))
        backends["gemini"] = {
            "available": has_key,
            "reason": "Ready" if has_key else "GEMINI_API_KEY not set",
        }
    except ImportError:
        backends["gemini"] = {
            "available": False,
            "reason": "google-genai package not installed",
        }

    # Check Anthropic
    try:
        import anthropic  # noqa: F401

        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        backends["anthropic"] = {
            "available": has_key,
            "reason": "Ready" if has_key else "ANTHROPIC_API_KEY not set",
        }
    except ImportError:
        backends["anthropic"] = {
            "available": False,
            "reason": "anthropic package not installed",
        }

    # Check OpenAI
    try:
        import openai  # noqa: F401

        has_key = bool(os.environ.get("OPENAI_API_KEY"))
        backends["openai"] = {
            "available": has_key,
            "reason": "Ready" if has_key else "OPENAI_API_KEY not set",
        }
    except ImportError:
        backends["openai"] = {
            "available": False,
            "reason": "openai package not installed",
        }

    return backends
