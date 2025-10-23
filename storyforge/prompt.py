"""
Prompt Builder for Children-Friendly Content.

This module provides the Prompt class, a comprehensive prompt builder
for generating age-appropriate, educational, and engaging stories and images
for children.
It encapsulates all parameters needed for story generation, image generation,
and image naming across different LLM backends.
"""

import random
from dataclasses import dataclass
from typing import Literal

from .schema import STORYFORGE_SCHEMA, SchemaValidator


@dataclass
class Prompt:
    """
    A comprehensive prompt builder for children-friendly content generation.

    This class encapsulates all the parameters needed to generate age-appropriate,
    educational, and engaging stories and images for children. It provides properties
    to access prompts for story generation, image generation, and image naming.

    When generating an image prompt, if context is provided, it is included as a
    separate section before the image instructions. This ensures that the illustration
    is informed by the same context as the story.

    Attributes:
        prompt (str): The main story prompt or description
        context (str, optional): Background information for story consistency
        length (str): Story length - "flash", "short", "medium", "bedtime"
        age_range (str): Target age group - "toddler", "preschool", "early_reader",
            "middle_grade"
        style (str): Story genre - "adventure", "comedy", "fantasy", "fairy_tale",
            "friendship"
        tone (str): Story mood - "gentle", "exciting", "silly", "heartwarming",
            "magical"
        theme (str, optional): Moral/educational theme
        setting (str, optional): Specific story setting
        characters (list[str], optional): Character names/descriptions to include
        learning_focus (str, optional): Educational element to emphasize
        image_style (str): Visual art style for illustrations - "chibi", "realistic",
            "cartoon", "watercolor", "sketch"

    Usage:
        prompt = Prompt("A brave little mouse goes on an adventure")
        story_prompt = prompt.story  # Get story generation prompt
        image_prompt = prompt.image  # Get image generation prompt
        name_prompt = prompt.image_name(story_text)  # Get filename prompt
    """

    prompt: str
    context: str | None = None
    length: str = "short"
    age_range: str = "preschool"
    style: str = "adventure"
    tone: str = "heartwarming"
    theme: str | None = None
    setting: str | None = None
    characters: list[str] | None = None
    learning_focus: str | None = None
    image_style: str = "chibi"
    continuation_mode: bool = False
    ending_type: Literal["wrap_up", "cliffhanger"] = "wrap_up"

    def __post_init__(self) -> None:
        """Resolve random parameters and validate after initialization."""
        self._resolve_random_parameters()
        self._validate_parameters()

    def _resolve_random_parameters(self) -> None:
        """Replace 'random' values with randomly selected valid values."""
        valid_values = self.get_valid_values()

        # Resolve random style
        if self.style == "random":
            self.style = random.choice(valid_values["style"])

        # Resolve random tone
        if self.tone == "random":
            self.tone = random.choice(valid_values["tone"])

        # Resolve random theme
        if self.theme == "random":
            self.theme = random.choice(valid_values["theme"])

        # Resolve random learning_focus
        if self.learning_focus is not None and self.learning_focus == "random":
            self.learning_focus = random.choice(valid_values["learning_focus"])

    def _validate_parameters(self) -> None:
        """Validate that all parameters have acceptable values using schema validation."""
        validator = SchemaValidator(STORYFORGE_SCHEMA)

        # Map prompt parameters to their schema field names and sections
        param_validations = [
            ("length", "story", self.length),
            ("age_range", "story", self.age_range),
            ("style", "story", self.style),
            ("tone", "story", self.tone),
            ("theme", "story", self.theme),
            ("learning_focus", "story", self.learning_focus),
            ("image_style", "images", self.image_style),
        ]

        for field_name, section_name, value in param_validations:
            if hasattr(STORYFORGE_SCHEMA, section_name):
                section = getattr(STORYFORGE_SCHEMA, section_name)
                if field_name in section.fields:
                    field = section.fields[field_name]
                    errors = validator.validate_field(field, value)
                    if errors:
                        # Raise the first validation error
                        raise ValueError(errors[0].message)

    def _get_length_description(self) -> str:
        """Get description text for the story length."""
        length_descriptions = {
            "flash": "very short (1-2 paragraphs)",
            "short": "short (3-4 paragraphs)",
            "medium": "medium-length (5-7 paragraphs)",
            "bedtime": "perfect bedtime story length (2-3 paragraphs with a calming ending)",
        }
        return length_descriptions[self.length]

    def _get_age_appropriate_guidance(self) -> str:
        """Get age-appropriate content guidance."""
        age_guidance = {
            "toddler": "Use simple words, short sentences, and focus on basic "
            "concepts like colors, shapes, and familiar objects. Include "
            "repetition and sound effects.",
            "preschool": "Use clear, simple language with some descriptive words. "
            "Include basic emotions and simple problem-solving. Keep concepts "
            "concrete and relatable.",
            "early_reader": "Use engaging vocabulary with some challenging words. "
            "Include more complex emotions and relationships. Allow for simple "
            "moral lessons.",
            "middle_grade": "Use rich vocabulary and more complex sentence "
            "structures. Include deeper character development and meaningful "
            "life lessons.",
        }
        return age_guidance[self.age_range]

    def _get_style_description(self) -> str:
        """Get description for the story style."""
        style_descriptions = {
            "adventure": "an exciting journey or quest with age-appropriate challenges",
            "comedy": "a funny, lighthearted story that will make children laugh",
            "fantasy": "a magical story with fantastical elements like talking animals or fairy creatures",
            "fairy_tale": "a classic fairy tale style story with a clear moral and happy ending",
            "friendship": "a heartwarming story about friendship, cooperation, and caring for others",
        }
        return style_descriptions[self.style]

    def _build_continuation_instruction(self) -> str:
        """Build instruction for story continuation."""
        recap_instruction = (
            "\n\nIMPORTANT: Begin your continuation with a brief recap "
            "(2-3 sentences starting with a phrase like 'When last we left our friends...' "
            "or 'Previously...' or 'In our last adventure...') that reminds readers "
            "what happened in the original story before continuing with new content."
        )

        if self.ending_type == "wrap_up":
            instruction = (
                "CONTINUATION TASK: The following is an existing story. "
                "Please write a continuation that wraps up the narrative "
                "with a satisfying resolution and complete ending."
                f"{recap_instruction}"
            )
        else:  # cliffhanger
            instruction = (
                "CONTINUATION TASK: The following is an existing story. "
                "Please write a continuation that advances the plot but "
                "ends with a cliffhanger - an exciting moment that sets "
                "up the next adventure without full resolution."
                f"{recap_instruction}"
            )

        return instruction

    @property
    def story(self) -> str:
        """
        Get the complete story generation prompt.

        Returns:
            str: A comprehensive prompt for story generation
        """
        # Start with the base prompt
        prompt_parts = []

        # Add continuation instruction if in continuation mode
        if self.continuation_mode:
            continuation_instruction = self._build_continuation_instruction()
            prompt_parts.append(f"{continuation_instruction}\n\n")

        # Add context if provided
        if self.context:
            prompt_parts.append(f"Context for story generation:\n{self.context}\n")
            if not self.continuation_mode:
                prompt_parts.append("Based on the above context, ")

        # Main story instruction (skip if in continuation mode since context IS the story)
        if not self.continuation_mode:
            prompt_parts.append(f"write {self._get_length_description()} story")

            # Add style and tone
            prompt_parts.append(f" that is {self._get_style_description()}")
            prompt_parts.append(f" with a {self.tone} tone")

            # Add the main prompt
            prompt_parts.append(f" based on this prompt: {self.prompt}")

            # Add setting if specified
            if self.setting:
                prompt_parts.append(f" The story should be set in: {self.setting}")

            # Add characters if specified
            if self.characters:
                char_list = ", ".join(self.characters)
                prompt_parts.append(f" Include these characters: {char_list}")

            # Add theme if specified
            if self.theme:
                prompt_parts.append(f" The story should emphasize the theme of {self.theme}")

            # Add learning focus if specified
            if self.learning_focus:
                prompt_parts.append(f" Incorporate learning about {self.learning_focus} naturally into the story")

            # Add age-appropriate guidance
            prompt_parts.append(f"\n\nAge-appropriate guidelines: {self._get_age_appropriate_guidance()}")

            # Add safety and quality guidelines
            prompt_parts.append("\n\nEnsure the story is:")
            prompt_parts.append("- Completely safe and appropriate for children")
            prompt_parts.append("- Positive and uplifting with a happy or meaningful ending")
            prompt_parts.append("- Educational or character-building in some way")
            prompt_parts.append("- Engaging and fun to read aloud")

            if self.context:
                prompt_parts.append("- Consistent with the provided context and character descriptions")
        else:
            # In continuation mode, provide guidance for the continuation
            prompt_parts.append(f"\nWrite a {self._get_length_description()} continuation ")
            prompt_parts.append(f"that maintains the {self.tone} tone and {self.style} style.")

            # Add age-appropriate guidance
            prompt_parts.append(f"\n\nAge-appropriate guidelines: {self._get_age_appropriate_guidance()}")

            # Add safety guidelines
            prompt_parts.append("\n\nEnsure the continuation is:")
            prompt_parts.append("- Completely safe and appropriate for children")
            prompt_parts.append("- Consistent with the original story's characters and setting")
            prompt_parts.append("- Engaging and fun to read aloud")

        return "".join(prompt_parts)

    def image(self, num_images: int) -> list[str]:
        """
        Generate a detailed, progressive image prompt for illustration.

        Args:
            num_images (int): The number of images to generate prompts for. Use this to create
                detailed, progressive prompts for multiple illustrations or scenes.

        Returns:
            list[str]: A list of detailed image prompts, one for each image.
        """
        prompts = []
        for _i in range(num_images):
            image_parts = []

            # Add context as a separate section if present
            if self.context:
                image_parts.append(f"Context for illustration:\n{self.context}\n\n")

            # Base instruction
            image_parts.append("Create a detailed, beautiful, child-friendly illustration")

            # Add style guidance
            style_guidance = {
                "adventure": "showing an exciting adventure scene",
                "comedy": "showing a fun, silly, and cheerful scene",
                "fantasy": "showing a magical, whimsical fantasy scene",
                "fairy_tale": "in classic fairy tale illustration style",
                "friendship": "showing characters together in a warm, friendly scene",
            }
            image_parts.append(f" {style_guidance.get(self.style, '')}")

            # Add the main prompt
            image_parts.append(f" for this story: {self.prompt}")

            # Add setting if specified
            if self.setting:
                image_parts.append(f" Set in: {self.setting}")

            # Add tone guidance
            tone_guidance = {
                "gentle": "Use soft, calming colors and peaceful imagery",
                "exciting": "Use bright, vibrant colors and dynamic composition",
                "silly": "Use playful, cartoon-like style with expressive characters",
                "heartwarming": "Use warm colors and cozy, comfortable imagery",
                "magical": "Use sparkles, glowing effects, and enchanting details",
            }
            image_parts.append(f" {tone_guidance.get(self.tone, '')}")

            # Add age-appropriate guidance
            age_art_guidance = {
                "toddler": "Simple, clear shapes with bright primary colors",
                "preschool": "Colorful, engaging artwork with clear details",
                "early_reader": "Rich, detailed illustrations with interesting elements to discover",
                "middle_grade": "Sophisticated artwork with depth and artistic detail",
            }
            image_parts.append(f" Style: {age_art_guidance.get(self.age_range, '')}")

            # Add image style guidance
            image_style_guidance = {
                "chibi": "in cute chibi/kawaii style with oversized heads and adorable features",
                "realistic": "in realistic, ultra detailed artistic style with natural proportions",
                "cartoon": "in bright cartoon style with bold lines and expressive characters",
                "watercolor": "in soft watercolor painting style with gentle, flowing colors",
                "sketch": "in pencil sketch style with artistic shading and fine details",
            }
            image_parts.append(f" Art style: {image_style_guidance.get(self.image_style, '')}")

            # Safety guidelines
            image_parts.append(" Ensure the image is completely safe, positive, and appropriate for children.")

            prompts.append("".join(image_parts))
        return prompts

    def image_name(self, story: str) -> str:
        """
        Build a prompt for generating a descriptive filename for the story image.

        Args:
            story (str): The generated story text

        Returns:
            str: A prompt for filename generation
        """
        return (
            f"Given this {self.style} story for {self.age_range} children: {story}\n\n"
            "Suggest a short, creative, and descriptive filename for an image "
            "illustrating this story. The filename should:\n"
            "- Be 2-4 words that capture the essence of the story\n"
            "- Use only letters, numbers, and underscores (no spaces or "
            "special characters)\n"
            "- Be memorable and appealing to children and parents\n"
            "- Reflect the story's main theme or characters\n"
            "\nReturn only the filename, nothing else."
        )

    @classmethod
    def get_valid_values(cls) -> dict[str, list[str]]:
        """
        Get all valid values for each parameter from schema (excluding 'random').

        Returns:
            dict: Parameter names mapped to their valid values for random selection
        """
        valid_values = {}

        # Extract valid values from schema sections
        for section_name in ["story", "images"]:
            if hasattr(STORYFORGE_SCHEMA, section_name):
                section = getattr(STORYFORGE_SCHEMA, section_name)
                for field_name, field in section.fields.items():
                    if field.valid_values:
                        # Filter out empty strings and 'random' for random selection
                        values = [v for v in field.valid_values if v and v != "random"]
                        if values:
                            valid_values[field_name] = values

        return valid_values
