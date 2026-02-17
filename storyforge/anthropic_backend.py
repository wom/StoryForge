"""
AnthropicBackend: Implementation of LLMBackend using Anthropic's Claude API.
Provides methods to generate stories and text-based content using Claude models.
Note: Claude does not support image generation, so image-related methods return None.
"""

import logging
import os
from typing import Any

import anthropic

from .llm_backend import LLMBackend
from .prompt import Prompt

logger = logging.getLogger(__name__)


class AnthropicBackend(LLMBackend):
    """
    LLM backend implementation using Anthropic's Claude API.
    Requires ANTHROPIC_API_KEY environment variable to be set.

    Note: This backend only supports text generation. Image generation
    methods will return None since Claude cannot generate images.
    """

    name = "anthropic"

    # Claude model context window sizes
    MODEL_TOKEN_LIMITS: dict[str, int] = {
        "claude-3-5-sonnet": 200000,
        "claude-3-5-haiku": 200000,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-4-sonnet": 200000,
        "claude-4-opus": 200000,
    }
    DEFAULT_TEXT_INPUT_LIMIT = 200000

    def __init__(self, config: Any = None) -> None:
        """
        Initialize the Anthropic client using the API key from environment variables.

        Args:
            config: Optional Config object (currently unused, for API consistency).

        Raises:
            RuntimeError: If ANTHROPIC_API_KEY is not set.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")
        self.client = anthropic.Anthropic(api_key=api_key)

        # Set text input limit (Claude models all have 200K context)
        self._text_input_limit = self.DEFAULT_TEXT_INPUT_LIMIT

    def generate_story(self, prompt: Prompt) -> str:
        """
        Generate a story based on the given Prompt object using Claude.

        Args:
            prompt (Prompt): A Prompt object containing comprehensive story
                generation parameters including context, style, tone, etc.

        Returns:
            str: The generated story, or an error message on failure.
        """
        try:
            # Use the Prompt's comprehensive prompt building
            story_prompt = prompt.story

            # Safety check: truncate if prompt exceeds model limits
            story_prompt = self._check_and_truncate_prompt(story_prompt)

            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.7,
                messages=[{"role": "user", "content": story_prompt}],
            )

            # Extract the story text from the response with proper null checking
            if response.content and len(response.content) > 0:
                # Claude returns a list of content blocks, get the first text block
                for content_block in response.content:
                    if content_block.type == "text" and hasattr(content_block, "text"):
                        return content_block.text.strip()

            return "[Error: No valid response from Claude]"
        except Exception as e:
            # Return a generic error message if generation fails
            logger.warning("Story generation failed: %s", e)
            return f"[Error generating story: {str(e)}]"

    def generate_image(
        self,
        prompt: Prompt,
        reference_image_bytes: bytes | None = None,
        override_prompt: str | None = None,
    ) -> tuple[object | None, bytes | None]:
        """
        Claude does not support image generation, so this method returns None.

        Args:
            prompt (Prompt): A Prompt object containing image generation parameters.
            reference_image_bytes (Optional[bytes]): Reference image bytes (unused).
            override_prompt (Optional[str]): Override prompt text (unused).

        Returns:
            Tuple[None, None]: Always returns None since Claude cannot generate images.
        """
        logger.warning("Claude (Anthropic) does not support image generation. Use Gemini or OpenAI for images.")
        return None, None

    def generate_image_name(self, prompt: Prompt, story: str) -> str:
        """
        Generate a short, creative, and descriptive filename for an image
        illustrating the story using Claude's text generation capabilities.

        Args:
            prompt (Prompt): A Prompt object containing the original parameters.
            story (str): The generated story.

        Returns:
            str: A suggested filename (no spaces or special characters), or
            'story_image' on failure.
        """
        try:
            # Use the Prompt's comprehensive image name prompt building
            name_prompt = prompt.image_name(story)

            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                temperature=0.3,  # Lower temperature for more consistent naming
                messages=[{"role": "user", "content": name_prompt}],
            )

            # Extract name with proper null checking
            if response.content and len(response.content) > 0:
                for content_block in response.content:
                    if content_block.type == "text" and hasattr(content_block, "text"):
                        name = content_block.text.strip()
                        # Remove file extension if present
                        name = name.split(".")[0]
                        # Clean up any unwanted characters
                        name = "".join(c for c in name if c.isalnum() or c == "_")
                        return name if name else "story_image"

            return "story_image"
        except Exception:
            logger.debug("Image name generation failed, using fallback", exc_info=True)
            return "story_image"

    def generate_image_prompt(self, story: str, context: str, num_prompts: int) -> list[str]:
        """Break the story into detailed image prompts using Claude's text model.

        Uses the shared LLM-based approach: builds a standardized instruction,
        sends it to Claude, and parses numbered prompts from the response.
        Falls back to mechanical paragraph splitting on failure.

        Args:
            story: The generated story text.
            context: Additional context (may include character descriptions).
            num_prompts: Number of image prompts to generate.

        Returns:
            List of detailed image prompts, one per requested image.
        """
        try:
            image_prompt_request = self._build_image_prompt_request(story, context, num_prompts)

            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.5,
                messages=[{"role": "user", "content": image_prompt_request}],
            )

            if response.content and len(response.content) > 0:
                for content_block in response.content:
                    if content_block.type == "text" and hasattr(content_block, "text"):
                        text = content_block.text.strip()
                        parsed = self._parse_numbered_prompts(text, num_prompts)
                        if parsed:
                            return parsed

            return self._generate_fallback_image_prompts(story, context, num_prompts)

        except Exception:
            logger.debug("Image prompt generation failed, using fallback", exc_info=True)
            return self._generate_fallback_image_prompts(story, context, num_prompts)
