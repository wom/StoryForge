"""
GeminiBackend: Implementation of LLMBackend using Google Gemini APIs.
Provides methods to generate stories, images, and image filenames using Gemini models.
"""

import os
from io import BytesIO

from google import genai
from google.genai import types
from PIL import Image

from .llm_backend import LLMBackend
from .story_prompt import StoryPrompt


class GeminiBackend(LLMBackend):
    """
    LLM backend implementation using Google Gemini APIs.
    Requires GEMINI_API_KEY environment variable to be set.
    """

    def __init__(self):
        """
        Initialize the Gemini client using the API key from environment variables.
        Raises:
            RuntimeError: If GEMINI_API_KEY is not set.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set.")
        self.client = genai.Client(api_key=api_key)

    def generate_story(
        self, prompt: str | StoryPrompt, context: str | None = None
    ) -> str:
        """
        Generate a story based on the given prompt using Gemini LLM.

        Args:
            prompt (str | StoryPrompt): The prompt to base the story on, or a
                StoryPrompt object containing comprehensive story generation parameters.
            context (str, optional): Additional context like character descriptions
                and background information for more consistent stories.
                                   Ignored if prompt is a StoryPrompt object.

        Returns:
            str: The generated story, or an error message on failure.
        """
        try:
            # Determine if we're using a StoryPrompt or simple string
            if isinstance(prompt, StoryPrompt):
                # Use the StoryPrompt's comprehensive prompt building
                contents = prompt.build_story_prompt()
            else:
                # Legacy behavior for backward compatibility
                if context:
                    contents = (
                        f"Context for story generation:\n\n{context}\n\n"
                        f"Based on the above context, write a short story for this "
                        f"prompt: {prompt}\n"
                        f"Use the character descriptions and background information to "
                        f"make the story consistent with established personalities and "
                        f"relationships."
                    )
                else:
                    # Fallback to basic prompt when no context available
                    contents = f"Write a short story based on this prompt: {prompt}"

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
            )
            # Extract the story text from the response
            return response.candidates[0].content.parts[0].text.strip()
        except Exception:
            # Return a generic error message if generation fails
            return "[Error generating story]"

    def generate_image(
        self, prompt: str | StoryPrompt
    ) -> tuple[object | None, bytes | None]:
        """
        Generate an illustration image for the given story prompt using Gemini
        image model.

        Args:
            prompt (str | StoryPrompt): The story prompt to illustrate, or a
                StoryPrompt object containing comprehensive image generation parameters.

        Returns:
            Tuple[Optional[Image.Image], Optional[bytes]]: The PIL Image object
            and its raw bytes, or (None, None) on failure.
        """
        # Determine if we're using a StoryPrompt or simple string
        if isinstance(prompt, StoryPrompt):
            contents = prompt.build_image_prompt()
        else:
            # Legacy behavior for backward compatibility
            contents = (
                f"Create a detailed, beautiful illustration for this story: {prompt}"
            )

        response = self.client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
        # Iterate through response parts to find image data
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                try:
                    # Load image from bytes
                    image = Image.open(BytesIO(part.inline_data.data))
                    image_bytes = part.inline_data.data
                    return image, image_bytes
                except Exception:
                    # Skip parts that fail to decode as images
                    continue
        # Return (None, None) if no image found
        return None, None

    def generate_image_name(self, prompt: str | StoryPrompt, story: str) -> str:
        """
        Generate a short, creative, and descriptive filename for an image
        illustrating the story.

        Args:
            prompt (str | StoryPrompt): The original prompt or StoryPrompt object.
            story (str): The generated story.

        Returns:
            str: A suggested filename (no spaces or special characters), or
            'story_image' on failure.
        """
        try:
            # Determine if we're using a StoryPrompt or simple string
            if isinstance(prompt, StoryPrompt):
                contents = prompt.build_image_name_prompt(story)
            else:
                # Legacy behavior for backward compatibility
                contents = (
                    f"Given this story: {story}\n"
                    "Suggest a short, creative, and descriptive filename for an "
                    "image illustrating it (no spaces, no special characters, "
                    "just letters, numbers, and underscores)."
                )

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
            )
            name = response.candidates[0].content.parts[0].text.strip()
            # Remove file extension if present
            name = name.split(".")[0]
            return name
        except Exception:
            # Fallback filename
            return "story_image"
