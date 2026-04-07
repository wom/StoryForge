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

import logging
import os
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

ERROR_STORY_SENTINEL = "[Error generating story]"

# Transient HTTP status codes
TRANSIENT_STATUS_CODES = {429, 500, 502, 503}


def classify_story_error(sentinel_value: str) -> tuple[str, bool]:
    """Classify an error story sentinel into a user-friendly message.

    Parses the sentinel string to determine the error type and returns
    an appropriate user-facing message and whether the error was transient.

    Args:
        sentinel_value: The error sentinel string from generate_story().

    Returns:
        A tuple of (user_message, is_transient).
    """
    error_detail = sentinel_value.removeprefix(ERROR_STORY_SENTINEL).strip(": ")

    # Check for transient errors
    error_lower = error_detail.lower()
    for code in TRANSIENT_STATUS_CODES:
        if str(code) in error_detail:
            return (
                f"Server temporarily unavailable ({code}). Please try again later.\nDetails: {error_detail}",
                True,
            )

    transient_keywords = ["unavailable", "overloaded", "rate limit", "too many requests", "high demand"]
    if any(kw in error_lower for kw in transient_keywords):
        return (
            f"Server temporarily unavailable. Please try again later.\nDetails: {error_detail}",
            True,
        )

    # Check for auth errors
    if any(code in error_detail for code in ["401", "403", "UNAUTHENTICATED", "PERMISSION_DENIED"]):
        return (
            f"Authentication failed. Please check your API key and try again.\nDetails: {error_detail}",
            False,
        )

    # Unknown error — show the detail without misleading advice
    if error_detail:
        return f"Story generation failed: {error_detail}", False
    return "Story generation failed. Please check your configuration and try again.", False


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

    # Transient error codes that should trigger retry
    TRANSIENT_STATUS_CODES = {429, 500, 502, 503}
    # gRPC/Gemini status strings that indicate transient errors
    TRANSIENT_STATUS_NAMES = {"RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"}
    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 10.0  # seconds

    @staticmethod
    def _safe_int(value: object) -> int | None:
        """Attempt to convert a value to int, returning None on failure."""
        try:
            return int(value)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _is_transient_error(error: Exception) -> bool:
        """Check if an exception represents a transient API error worth retrying.

        Inspects the exception for HTTP status codes commonly associated with
        temporary failures: 429 (rate limit), 500, 502, 503 (server overload).
        Also recognises gRPC/Gemini status name strings like RESOURCE_EXHAUSTED.
        Checks both typed exception attributes and string representations.

        Args:
            error: The exception to classify.

        Returns:
            True if the error is transient and the request should be retried.
        """
        # Check for status_code attribute (google-genai, openai, anthropic all use this)
        status_code = getattr(error, "status_code", None) or getattr(error, "status", None)
        if status_code is not None:
            # Gemini may set status_code to a gRPC status name string
            if str(status_code) in LLMBackend.TRANSIENT_STATUS_NAMES:
                return True
            numeric = LLMBackend._safe_int(status_code)
            if numeric is not None and numeric in LLMBackend.TRANSIENT_STATUS_CODES:
                return True

        # Check for HTTP status attribute on nested response objects
        response = getattr(error, "response", None)
        if response is not None:
            resp_status = getattr(response, "status_code", None) or getattr(response, "status", None)
            if resp_status is not None:
                if str(resp_status) in LLMBackend.TRANSIENT_STATUS_NAMES:
                    return True
                numeric = LLMBackend._safe_int(resp_status)
                if numeric is not None and numeric in LLMBackend.TRANSIENT_STATUS_CODES:
                    return True

        # Fallback: check string representation for status codes
        error_str = str(error)
        for code in LLMBackend.TRANSIENT_STATUS_CODES:
            if str(code) in error_str:
                return True

        # Check for common transient error keywords
        transient_keywords = ["UNAVAILABLE", "overloaded", "rate limit", "too many requests", "high demand",
                              "RESOURCE_EXHAUSTED"]
        error_lower = error_str.lower()
        return any(kw.lower() in error_lower for kw in transient_keywords)

    def _retry_transient(self, fn: Callable[[], T], operation: str = "API call") -> T:
        """Execute a callable with retry logic for transient errors.

        On transient failures (429, 500, 502, 503), retries with exponential
        backoff starting at 10s (10s → 20s → 40s) plus random jitter.
        Prints clear status messages so the user knows what's happening.

        Args:
            fn: Zero-argument callable to execute (typically a lambda wrapping the API call).
            operation: Human-readable description for log messages (e.g., "story generation").

        Returns:
            The return value of fn on success.

        Raises:
            The original exception if the error is not transient or retries are exhausted.
        """
        from .console import console

        last_exception: Exception | None = None

        for attempt in range(self.MAX_RETRIES + 1):  # 0 = initial, 1-3 = retries
            try:
                return fn()
            except Exception as e:
                last_exception = e

                if not self._is_transient_error(e):
                    raise

                if attempt >= self.MAX_RETRIES:
                    console.print(
                        f"[red]❌ {operation.capitalize()} still failing after "
                        f"{self.MAX_RETRIES} retries. Giving up.[/red]"
                    )
                    raise

                delay = self.BASE_RETRY_DELAY * (2**attempt)
                jitter = random.uniform(0, delay * 0.25)
                total_delay = delay + jitter

                # Extract a short error description for the user
                error_brief = str(e).split("\n")[0][:120]
                console.print(
                    f"[yellow]⚠ {operation.capitalize()} failed: {error_brief}[/yellow]\n"
                    f"[yellow]  Retrying in {total_delay:.0f}s... "
                    f"(attempt {attempt + 1}/{self.MAX_RETRIES})[/yellow]"
                )
                logger.info(
                    "Transient error during %s (attempt %d/%d): %s", operation, attempt + 1, self.MAX_RETRIES, e
                )

                time.sleep(total_delay)

        # Should not reach here, but satisfy type checker
        assert last_exception is not None
        raise last_exception

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

    def _build_image_prompt_request(
        self, story: str, context: str, num_prompts: int, character_descriptions: str = ""
    ) -> str:
        """Build a standardized LLM instruction for generating image prompts.

        Constructs a detailed instruction asking the LLM to break the story into
        visually distinct, progressive image prompts with character context and
        scene differentiation.

        Args:
            story: The generated story text.
            context: Additional story context.
            num_prompts: Number of image prompts to generate.
            character_descriptions: Formatted character visual descriptions.

        Returns:
            The instruction text to send to the LLM.
        """
        scene_labels = self._get_scene_labels(num_prompts)
        scene_guidance = "\n".join(f"  - Image {i + 1}: {label}" for i, label in enumerate(scene_labels))

        parts = [
            f"Break this story into exactly {num_prompts} detailed, progressive image prompts "
            f"for generating illustrations. Each prompt must describe a DIFFERENT moment in the story, "
            f"progressing from beginning to end.\n",
        ]

        if character_descriptions:
            parts.append(
                "\n=== CHARACTER VISUAL DESCRIPTIONS ===\n"
                "Use these descriptions to ensure characters look consistent across ALL images. "
                "Include these visual details in every prompt where the character appears:\n"
                f"{character_descriptions}\n"
            )

        parts.append(f"\n=== STORY ===\n{story}\n")

        if context:
            parts.append(f"\n=== CONTEXT ===\n{context}\n")

        parts.append(
            "\n=== INSTRUCTIONS ===\n"
            f"Create {num_prompts} image prompts following this progression:\n"
            f"{scene_guidance}\n\n"
            "Each prompt MUST:\n"
            "- Depict a distinct narrative moment (different scene, setting, action)\n"
            "- Include detailed character appearances (hair color, clothing, glasses, etc.)\n"
            "- Describe the setting, lighting, colors, and mood\n"
            "- Vary composition and framing between images\n"
            "- Be child-friendly and suitable for AI image generation\n\n"
            f"Return exactly {num_prompts} prompts, each on a new line, numbered 1-{num_prompts}. "
            "Do not include any other text, headers, or explanations."
        )

        return "".join(parts)

    @staticmethod
    def _get_scene_labels(num_prompts: int) -> list[str]:
        """Get descriptive scene labels for progressive image prompts.

        Args:
            num_prompts: Number of images being generated.

        Returns:
            List of scene label strings, one per image.
        """
        if num_prompts == 1:
            return ["The key moment of the story — capture the essence of the narrative"]
        if num_prompts == 2:
            return [
                "Opening scene — establish characters, setting, and the beginning of the story",
                "Climactic or closing scene — the turning point or resolution",
            ]
        if num_prompts == 3:
            return [
                "Opening scene — establish characters and setting",
                "Mid-story turning point — rising action or key conflict",
                "Climactic or closing scene — resolution or emotional peak",
            ]
        if num_prompts == 4:
            return [
                "Opening scene — introduce characters and setting",
                "Rising action — the adventure or conflict develops",
                "Climactic moment — the peak of tension or excitement",
                "Resolution — the ending or aftermath",
            ]
        # 5+
        labels = [
            "Opening scene — introduce characters and setting",
            "Early development — the journey or conflict begins",
            "Mid-story turning point — a key moment of change",
            "Climactic moment — the peak of the story",
            "Resolution or epilogue — the ending",
        ]
        while len(labels) < num_prompts:
            labels.insert(-1, f"Scene {len(labels)} — a distinct moment in the story")
        return labels[:num_prompts]

    @staticmethod
    def _parse_numbered_prompts(text: str, num_prompts: int) -> list[str] | None:
        """Parse numbered prompts from an LLM response.

        Handles common formats: '1. ', '1) ', '1: ', '**1.** ', '**1)** '.
        Joins multi-line prompts (lines that don't start with a number are
        appended to the previous prompt). Accepts >= num_prompts results
        and truncates.

        Args:
            text: Raw LLM response text.
            num_prompts: Expected number of prompts.

        Returns:
            List of prompt strings if parsing succeeded, None otherwise.
        """
        # Pattern matches: optional bold markers, digit(s), separator, optional bold markers
        numbered_line_pattern = re.compile(r"^\s*\*{0,2}\s*(\d+)\s*[\.\)\:]\s*\*{0,2}\s*(.+)")
        prompts: list[str] = []

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            match = numbered_line_pattern.match(line)
            if match:
                prompt_text = match.group(2).strip()
                if prompt_text:
                    prompts.append(prompt_text)
            elif prompts:
                # Non-numbered line — append to previous prompt (multi-line support)
                prompts[-1] += " " + line

        if len(prompts) >= num_prompts:
            return prompts[:num_prompts]

        logger.debug(
            "Parsed %d prompts but expected %d, falling back",
            len(prompts),
            num_prompts,
        )
        return None

    @staticmethod
    def _segment_story(story: str, num_segments: int) -> list[str]:
        """Divide a story into roughly equal narrative segments.

        Splits by paragraphs and groups them into the requested number of
        segments. Used by the fallback prompt generator to assign distinct
        story portions to each image.

        Args:
            story: The full story text.
            num_segments: Number of segments to create.

        Returns:
            List of story segment strings.
        """
        paragraphs = [p.strip() for p in story.split("\n") if p.strip()]
        if not paragraphs:
            return [story] * num_segments

        if len(paragraphs) <= num_segments:
            # Fewer paragraphs than segments — pad with the last paragraph
            padded = list(paragraphs)
            while len(padded) < num_segments:
                padded.append(paragraphs[-1])
            return padded

        # Distribute paragraphs evenly across segments
        segments: list[str] = []
        chunk_size = len(paragraphs) / num_segments
        for i in range(num_segments):
            start = int(i * chunk_size)
            end = int((i + 1) * chunk_size)
            segment = "\n".join(paragraphs[start:end])
            segments.append(segment)

        return segments

    def _generate_fallback_image_prompts(self, story: str, context: str, num_prompts: int) -> list[str]:
        """Generate fallback image prompts when the LLM API fails.

        Splits the story into segments and creates scene-labeled illustration
        prompts with character context. Used as a reliable mechanical fallback
        when the LLM-based approach is unavailable.

        Args:
            story: The story text.
            context: Additional context (may include character descriptions).
            num_prompts: Number of prompts needed.

        Returns:
            Simple fallback image prompts with scene labels and context.
        """
        segments = self._segment_story(story, num_prompts)
        scene_labels = self._get_scene_labels(num_prompts)
        prompts = []

        for i in range(num_prompts):
            label = scene_labels[i]
            segment = segments[i]
            prompt = (
                f"Create a detailed, child-friendly illustration for the {label.split('—')[0].strip()} "
                f"of the story:\n{segment}"
            )
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

    @staticmethod
    def _sanitize_image_name(name: str) -> str:
        """Sanitize a generated image name to a safe filename stem."""
        name = name.split(".")[0]
        name = "".join(c for c in name if c.isalnum() or c == "_")
        return name if name else "story_image"


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
