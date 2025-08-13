"""
OpenAIBackend: Implementation of LLMBackend using OpenAI APIs.
Provides methods to generate stories, images, and image filenames using OpenAI models.
"""

import os
from io import BytesIO

import openai
from PIL import Image

from .llm_backend import LLMBackend
from .prompt import Prompt


class OpenAIBackend(LLMBackend):
    """
    LLM backend implementation using OpenAI APIs.
    Requires OPENAI_API_KEY environment variable to be set.
    """

    def generate_image_prompt(self, story: str, context: str, num_prompts: int) -> list[str]:
        """
        Return a list of image prompts by splitting the story into num_prompts chunks.
        This is a stub implementation to satisfy the abstract base class.
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

    def __init__(self) -> None:
        """
        Initialize the OpenAI client using the API key from environment variables.
        Raises:
            RuntimeError: If OPENAI_API_KEY is not set.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set.")
        self.client = openai.OpenAI(api_key=api_key)

    def generate_story(self, prompt: Prompt) -> str:
        """
        Generate a story based on the given Prompt object using OpenAI LLM.

        Args:
            prompt (Prompt): A Prompt object containing comprehensive story
                generation parameters including context, style, tone, etc.

        Returns:
            str: The generated story, or an error message on failure.
        """
        try:
            # Use the Prompt's comprehensive prompt building
            contents = prompt.story

            response = self.client.chat.completions.create(
                model="gpt-4", messages=[{"role": "user", "content": contents}], max_tokens=2000, temperature=0.7
            )

            # Extract the story text from the response with proper null checking
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                return "[Error: No valid response from OpenAI]"
        except Exception:
            # Return a generic error message if generation fails
            return "[Error generating story]"

    def generate_image(
        self, prompt: Prompt, reference_image_bytes: bytes | None = None
    ) -> tuple[object | None, bytes | None]:
        """
        Generate an illustration image for the given Prompt object using OpenAI
        DALL-E model, optionally using a reference image for consistency.

        Args:
            prompt (Prompt): A Prompt object containing comprehensive image
                generation parameters including style, tone, setting, etc.
            reference_image_bytes (Optional[bytes]): Reference image bytes to maintain
                consistency with previous images (currently not used by DALL-E).

        Returns:
            Tuple[Optional[Image.Image], Optional[bytes]]: The PIL Image object
            and its raw bytes, or (None, None) on failure.
        """
        try:
            # Use the Prompt's comprehensive image prompt building
            text_prompt = prompt.image(1)[0]

            # Note: DALL-E doesn't support reference images like Gemini does
            # We'll include a style consistency note if reference_image_bytes is provided
            if reference_image_bytes:
                text_prompt = (
                    "Create the next illustration in a consistent visual style "
                    "with the same characters and art style as previous images. "
                    f"{text_prompt}"
                )

            response = self.client.images.generate(
                model="dall-e-3", prompt=text_prompt, size="1024x1024", quality="standard", response_format="url", n=1
            )

            if response.data and response.data[0].url:
                # Download the image from the URL
                import requests

                img_response = requests.get(response.data[0].url)
                if img_response.status_code == 200:
                    image_bytes = img_response.content
                    image = Image.open(BytesIO(image_bytes))
                    return image, image_bytes

            return None, None

        except Exception:
            # Return (None, None) if image generation fails
            return None, None

    def generate_image_name(self, prompt: Prompt, story: str) -> str:
        """
        Generate a short, creative, and descriptive filename for an image
        illustrating the story.

        Args:
            prompt (Prompt): A Prompt object containing the original parameters.
            story (str): The generated story.

        Returns:
            str: A suggested filename (no spaces or special characters), or
            'story_image' on failure.
        """
        try:
            # Use the Prompt's comprehensive image name prompt building
            contents = prompt.image_name(story)

            response = self.client.chat.completions.create(
                model="gpt-4", messages=[{"role": "user", "content": contents}], max_tokens=50, temperature=0.5
            )

            # Extract name with proper null checking
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                name = response.choices[0].message.content.strip()
                # Remove file extension if present
                name = name.split(".")[0]
                return name
            else:
                return "story_image"
        except Exception:
            # Fallback filename
            return "story_image"
