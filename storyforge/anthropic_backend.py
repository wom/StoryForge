"""
AnthropicBackend: Implementation of LLMBackend using Anthropic's Claude API.
Provides methods to generate stories and text-based content using Claude models.
Note: Claude does not support image generation, so image-related methods return None.
"""

import os
from typing import Any

import anthropic

from .llm_backend import LLMBackend
from .prompt import Prompt


class AnthropicBackend(LLMBackend):
    """
    LLM backend implementation using Anthropic's Claude API.
    Requires ANTHROPIC_API_KEY environment variable to be set.

    Note: This backend only supports text generation. Image generation
    methods will return None since Claude cannot generate images.
    """

    name = "anthropic"

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
            return f"[Error generating story: {str(e)}]"

    def generate_image(
        self, prompt: Prompt, reference_image_bytes: bytes | None = None
    ) -> tuple[object | None, bytes | None]:
        """
        Claude does not support image generation, so this method returns None.

        Args:
            prompt (Prompt): A Prompt object containing image generation parameters.
            reference_image_bytes (Optional[bytes]): Reference image bytes (unused).

        Returns:
            Tuple[None, None]: Always returns None since Claude cannot generate images.
        """
        import logging

        logger = logging.getLogger(__name__)
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
            # Fallback filename
            return "story_image"

    def generate_image_prompt(self, story: str, context: str, num_prompts: int) -> list[str]:
        """
        Break the given story into detailed image prompts using Claude's understanding.
        Each prompt will be incredibly detailed for use with external image generation services.

        Args:
            story (str): The generated story to break into image prompts.
            context (str): Additional context for the story.
            num_prompts (int): The number of image prompts to return.

        Returns:
            list[str]: A list of detailed image prompts describing scenes from the story.
        """
        try:
            # Create a comprehensive prompt for Claude to generate image descriptions
            image_prompt_request = (
                f"Please break this story into {num_prompts} detailed, progressive image prompts "
                f"that would be perfect for generating illustrations. Each prompt should be "
                f"incredibly detailed, focusing on visual elements like character appearance, "
                f"setting details, colors, lighting, and mood.\n\n"
                f"Story: {story}\n\n"
            )

            if context:
                image_prompt_request += f"Context: {context}\n\n"

            image_prompt_request += (
                f"Return exactly {num_prompts} image prompts, each on a new line, numbered 1-{num_prompts}. "
                f"Each should be child-friendly, detailed, and suitable for AI image generation."
            )

            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.5,
                messages=[{"role": "user", "content": image_prompt_request}],
            )

            # Extract and parse the prompts
            if response.content and len(response.content) > 0:
                for content_block in response.content:
                    if content_block.type == "text" and hasattr(content_block, "text"):
                        text = content_block.text.strip()
                        # Parse numbered prompts from Claude's response
                        lines = text.split("\n")
                        prompts: list[str] = []

                        for line in lines:
                            line = line.strip()
                            # Look for numbered lines like "1. " or "1) "
                            if line and (line[0].isdigit() or line.startswith(str(len(prompts) + 1))):
                                # Remove the number and clean up
                                cleaned = line
                                for prefix in [
                                    f"{len(prompts) + 1}. ",
                                    f"{len(prompts) + 1}) ",
                                    f"{len(prompts) + 1}: ",
                                ]:
                                    if cleaned.startswith(prefix):
                                        cleaned = cleaned[len(prefix) :]
                                        break
                                if cleaned:
                                    prompts.append(cleaned)

                        # If we got the right number of prompts, return them
                        if len(prompts) == num_prompts:
                            return prompts

            # Fallback: Simple story-based prompts
            return self._generate_fallback_image_prompts(story, context, num_prompts)

        except Exception:
            # Fallback to simple story-based prompts
            return self._generate_fallback_image_prompts(story, context, num_prompts)

    def _generate_fallback_image_prompts(self, story: str, context: str, num_prompts: int) -> list[str]:
        """
        Generate fallback image prompts when Claude API fails.

        Args:
            story (str): The story text.
            context (str): Additional context.
            num_prompts (int): Number of prompts needed.

        Returns:
            list[str]: Simple fallback image prompts.
        """
        # Simple fallback: split story into paragraphs, or repeat the story if not enough
        paragraphs = [p.strip() for p in story.split("\n") if p.strip()]
        prompts = []

        for i in range(num_prompts):
            if i < len(paragraphs):
                base = paragraphs[i]
            else:
                base = story
            prompt = f"Create a detailed, child-friendly illustration for this part of the story: {base}"
            if context:
                prompt += f"\nContext: {context}"
            prompts.append(prompt)

        return prompts
