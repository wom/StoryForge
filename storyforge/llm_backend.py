"""
LLM Backend Factory and Abstract Base Class.

This module provides:
1. LLMBackend abstract base class defining the interface for all backends
2. get_backend() factory function for automatic backend selection
3. Backend detection logic based on environment variables

Supported backends:
- Gemini (Google AI): requires GEMINI_API_KEY
- Anthropic Claude: requires ANTHROPIC_API_KEY
- OpenAI: requires OPENAI_API_KEY
"""

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config
    from .prompt import Prompt


class LLMBackend(ABC):
    """
    Abstract base class for Large Language Model backends.

    This class defines the interface that all LLM backends must implement
    to provide story generation, image generation, and image naming capabilities.
    """

    name: str

    # Token budget configuration (shared across all backends)
    CONTEXT_BUDGET_RATIO = 0.50  # Reserve 50% of context window for context
    COMPRESSION_TRIGGER_RATIO = 0.80  # Warn/truncate when prompt exceeds 80% of limit

    @property
    def text_input_limit(self) -> int:
        """Maximum input token limit for the text generation model.

        Subclasses should override this to return model-specific limits.
        Falls back to a conservative default of 8192 tokens.

        Returns:
            int: Maximum number of input tokens the model accepts.
        """
        return getattr(self, "_text_input_limit", 8192)

    def get_context_token_budget(self) -> int:
        """Calculate the token budget available for context.

        Uses CONTEXT_BUDGET_RATIO of the model's text_input_limit.

        Returns:
            int: Number of tokens to allocate for context.
        """
        return int(self.text_input_limit * self.CONTEXT_BUDGET_RATIO)

    def _check_and_truncate_prompt(self, contents: str) -> str:
        """Check prompt size against model limits and truncate if necessary.

        Logs a warning if the prompt exceeds 80% of the model's input limit
        and truncates to 80% as a safety measure.

        Args:
            contents: The full prompt text to check.

        Returns:
            str: The original or truncated prompt text.
        """
        import logging

        logger = logging.getLogger(__name__)
        prompt_tokens = self.estimate_token_count(contents)
        trigger_limit = int(self.text_input_limit * self.COMPRESSION_TRIGGER_RATIO)

        if prompt_tokens > trigger_limit:
            target_chars = trigger_limit * 4
            logger.warning(
                "Prompt (~%d tokens) exceeds %d%% of %s model limit (%d tokens). Truncating to ~%d tokens.",
                prompt_tokens,
                int(self.COMPRESSION_TRIGGER_RATIO * 100),
                self.name,
                self.text_input_limit,
                trigger_limit,
            )
            contents = contents[:target_chars]

        return contents

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
        self,
        prompt: "Prompt",
        reference_image_bytes: bytes | None = None,
        override_prompt: str | None = None,
    ) -> tuple[object | None, bytes | None]:
        """
        Generate an image based on the given Prompt object.

        Args:
            prompt (Prompt): A Prompt object containing comprehensive parameters
                for image generation including style, tone, setting, etc.
            reference_image_bytes (Optional[bytes]): Reference image bytes to maintain
                consistency with previous images.
            override_prompt (Optional[str]): If provided, use this text prompt
                instead of building one from the Prompt object. Useful for
                passing scene-specific prompts from generate_image_prompt().

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

    def _generate_fallback_image_prompts(self, story: str, context: str, num_prompts: int) -> list[str]:
        """
        Generate fallback image prompts when the LLM API fails.

        Splits the story into paragraphs and creates simple illustration
        prompts from them.

        Args:
            story: The story text.
            context: Additional context.
            num_prompts: Number of prompts needed.

        Returns:
            Simple fallback image prompts.
        """
        paragraphs = [p.strip() for p in story.split("\n") if p.strip()]
        prompts = []

        for i in range(num_prompts):
            base = paragraphs[i] if i < len(paragraphs) else story
            prompt = f"Create a detailed, child-friendly illustration for this part of the story: {base}"
            if context:
                prompt += f"\nContext: {context}"
            prompts.append(prompt)

        return prompts

    @staticmethod
    def estimate_token_count(text: str) -> int:
        """
        Estimate the number of tokens in a text string.

        Uses the standard heuristic of approximately 4 characters per token,
        which works reasonably well across most languages and models.

        Args:
            text (str): The text to estimate token count for.

        Returns:
            int: Estimated number of tokens.
        """
        return len(text) // 4


def get_backend(
    backend_name: str | None = None, config_backend: str | None = None, config: "Config | None" = None
) -> LLMBackend:
    """
    Factory function to get an appropriate LLM backend instance.

    Args:
        backend_name (Optional[str]): Specific backend to use ("gemini",
                                      "openai", etc.). Highest priority.
        config_backend (Optional[str]): Backend from configuration file.
        config (Optional[Config]): Configuration object to pass to the backend.

    Returns:
        LLMBackend: An instance of the selected backend.

    Raises:
        RuntimeError: If no suitable backend is found or API keys are missing.
        ImportError: If required dependencies for the backend are not installed.

    Backend Selection Priority:
        1. backend_name parameter (explicit override)
        2. Configuration file backend setting
        3. Auto-detection based on available API keys

    Environment Variables:
        GEMINI_API_KEY: Required for Gemini backend
        OPENAI_API_KEY: Required for OpenAI backend
        ANTHROPIC_API_KEY: Required for Anthropic backend
    """

    # Priority 1: Explicit backend name parameter
    if backend_name:
        backend_name = backend_name.lower()
    # Priority 2: Configuration file backend setting
    elif config_backend:
        backend_name = config_backend.lower()
    # Priority 3: Auto-detect based on available API keys
    else:
        if os.environ.get("GEMINI_API_KEY"):
            backend_name = "gemini"
        elif os.environ.get("OPENAI_API_KEY"):
            backend_name = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            backend_name = "anthropic"
        elif os.environ.get("LLM_BACKEND"):
            # Explicit backend set but not recognized
            llm_backend_env = os.environ.get("LLM_BACKEND")
            backend_name = llm_backend_env.lower() if llm_backend_env else None
        else:
            raise RuntimeError(
                "No LLM backend available. Please set one of the following:\n"
                "- Configuration file: [system] backend = gemini/openai/anthropic\n"
                "- API keys: GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY"
            )

    # Import and instantiate the requested backend
    try:
        if backend_name == "gemini":
            from .gemini_backend import GeminiBackend

            return GeminiBackend(config=config)

        elif backend_name == "openai":
            from .openai_backend import OpenAIBackend

            return OpenAIBackend(config=config)

        elif backend_name == "anthropic":
            from .anthropic_backend import AnthropicBackend

            return AnthropicBackend(config=config)

        else:
            raise RuntimeError(
                f"Unknown backend '{backend_name}'. Supported backends: gemini, openai, anthropic (more coming soon)"
            )

    except ImportError as e:
        raise ImportError(
            f"Failed to import {backend_name} backend. Please ensure required dependencies are installed: {e}"
        ) from e
