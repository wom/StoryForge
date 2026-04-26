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

# ---------------------------------------------------------------------------
# Voice archetype descriptions
#
# DISCLAIMER: Voice archetypes are original stylistic descriptions inspired
# by literary traditions. They are not affiliated with, endorsed by, or
# associated with any named authors or their estates. Author references in
# documentation are used solely as familiar touchstones to help users
# understand the writing style each archetype produces.
# ---------------------------------------------------------------------------

VOICE_DESCRIPTIONS: dict[str, str] = {
    "anapestic": (
        "Write in anapestic tetrameter — a bouncy, rhythmic verse style with a "
        "'da-da-DUM da-da-DUM' meter. Use playful invented nonsense words, "
        "strong rhythmic patterns, and repetitive refrains. Sentences should have "
        "a sing-song quality with unexpected rhymes and exuberant wordplay."
    ),
    "sardonic": (
        "Write with a sardonic, conspiratorial narrator voice. Address the reader "
        "directly with dark humor and gleeful mischief. Use vivid, slightly "
        "grotesque descriptions played for laughs. Villains should be delightfully "
        "awful and receive satisfying comeuppance. The narrator should seem to "
        "relish telling the story with knowing asides."
    ),
    "picaresque": (
        "Write as a picaresque adventure — a quest-driven narrative with rich "
        "world-building and enchanted settings. Use vivid descriptions of magical "
        "places and creatures. The protagonist should journey through a series of "
        "episodic encounters, each revealing something new about the story world. "
        "Balance wonder with forward momentum."
    ),
    "iambic": (
        "Write in iambic meter — a classic 'da-DUM da-DUM' rhythmic pattern. "
        "Use elegant, flowing verse with measured cadence. Favor rich imagery and "
        "elevated vocabulary appropriate for the age range. The rhythm should feel "
        "natural and stately, like a story told by firelight."
    ),
    "fable": (
        "Write as a fable — a concise moral tale with animal characters who embody "
        "human traits. Use a wise, measured narrator voice. Each character should "
        "represent a clear virtue or flaw. Build toward a satisfying moral lesson "
        "stated or implied at the end. Keep the narrative focused and purposeful."
    ),
    "gothic": (
        "Write as a gothic fairy tale — atmospheric, lyrical, and bittersweet. "
        "Use rich sensory descriptions: moonlit forests, ancient castles, "
        "whispering winds. The tone should be dreamy and slightly mysterious, "
        "but always safe for children. Beauty and wonder should shine through "
        "the shadows. End with warmth and resolution."
    ),
    "nonsense": (
        "Write as nonsense literature — embrace absurdist logic, impossible events, "
        "and delightful wordplay. Invent portmanteau words, use paradoxes, and let "
        "characters behave in charmingly illogical ways. The world should operate "
        "by its own whimsical rules. Include witty dialogue and playful riddles."
    ),
    "lyrical": (
        "Write as soft, repetitive bedtime prose — gentle rhythm, cozy imagery, "
        "and a soothing, almost hypnotic cadence. Use simple repetitive phrases "
        "and warm sensory details: soft blankets, twinkling stars, quiet whispers. "
        "The story should feel like a lullaby in prose form, winding down gently "
        "toward sleep."
    ),
    "epistolary": (
        "Write as an epistolary narrative — first-person journal entries, diary "
        "pages, or letters. Use a conversational, confessional tone with the "
        "immediacy of someone writing about their day. Include doodle-like "
        "asides, crossed-out words (struck through), and the candid humor of "
        "a kid narrating their own life."
    ),
}


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
        world (str, optional): World definition (characters, places, lore) always included verbatim
        length (str): Story length - "flash", "short", "medium", "bedtime"
        age_range (str): Target age group - "toddler", "preschool", "early_reader",
            "middle_grade"
        style (str): Story genre - "adventure", "comedy", "fantasy", "fairy_tale",
            "friendship"
        tone (str): Story mood - "gentle", "exciting", "silly", "heartwarming",
            "magical"
        voice (str, optional): Writing voice archetype - "anapestic", "sardonic",
            "picaresque", "iambic", "fable", "gothic", "nonsense", "lyrical",
            "epistolary"
        theme (str, optional): Moral/educational theme
        setting (str, optional): Specific story setting
        characters (list[str], optional): Character names/descriptions to include
        learning_focus (str, optional): Educational element to emphasize
        image_style (str): Visual art style for illustrations - "chibi", "realistic",
            "cartoon", "watercolor", "sketch"
        continuation_mode (bool): Whether this is a continuation of a previous story
        ending_type (str): How to end the story - "wrap_up" or "cliffhanger"
        has_old_context (bool): Whether old-era context was included via temporal sampling

    Usage:
        prompt = Prompt("A brave little mouse goes on an adventure")
        story_prompt = prompt.story  # Get story generation prompt
        image_prompt = prompt.image  # Get image generation prompt
        name_prompt = prompt.image_name(story_text)  # Get filename prompt
    """

    prompt: str
    context: str | None = None
    world: str | None = None
    length: str = "short"
    age_range: str = "preschool"
    style: str = "adventure"
    tone: str = "heartwarming"
    voice: str | None = None
    theme: str | None = None
    setting: str | None = None
    characters: list[str] | None = None
    learning_focus: str | None = None
    image_style: str = "chibi"
    continuation_mode: bool = False
    ending_type: Literal["wrap_up", "cliffhanger"] = "wrap_up"
    has_old_context: bool = False
    continuation_direction: str | None = None
    refinement_mode: bool = False
    original_story: str | None = None
    refinement_instructions: str | None = None

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

        # Resolve random voice
        if self.voice is not None and self.voice == "random":
            self.voice = random.choice(valid_values["voice"])

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
            ("voice", "story", self.voice),
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

        if self.refinement_mode:
            if not self.original_story:
                raise ValueError("refinement_mode requires original_story to be set")
            if not self.refinement_instructions:
                raise ValueError("refinement_mode requires refinement_instructions to be set")

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

    def _get_voice_instruction(self) -> str | None:
        """Get the full voice instruction for prompt assembly, or None if no voice is set."""
        if not self.voice:
            return None
        return VOICE_DESCRIPTIONS.get(self.voice)

    def _build_continuation_instruction(self) -> str:
        """Build instruction for story continuation."""
        # Detect multi-part chain from "--- Part N of M ---" separators
        multi_part_note = ""
        if self.context and "--- Part " in self.context:
            import re

            parts = re.findall(r"--- Part (\d+) of (\d+) ---", self.context)
            if parts:
                total = parts[-1][1]  # e.g. "3" from "Part 3 of 3"
                multi_part_note = (
                    f"\n\nNOTE: The context contains {total} connected story parts. "
                    f"Continue from where Part {total} ends."
                )

        recap_instruction = (
            "\n\nIMPORTANT: Begin your continuation with a brief recap "
            "(2-3 sentences starting with a phrase like 'When last we left our friends...' "
            "or 'Previously...' or 'In our last adventure...') that reminds readers "
            "what happened in the original story before continuing with new content."
        )

        direction_note = ""
        if self.continuation_direction:
            direction_note = f"\n\nDIRECTION FOR THIS CONTINUATION: {self.continuation_direction}"

        if self.ending_type == "wrap_up":
            instruction = (
                "CONTINUATION TASK: The following is an existing story. "
                "Please write a continuation that wraps up the narrative "
                "with a satisfying resolution and complete ending."
                f"{multi_part_note}"
                f"{direction_note}"
                f"{recap_instruction}"
            )
        else:  # cliffhanger
            instruction = (
                "CONTINUATION TASK: The following is an existing story. "
                "Please write a continuation that advances the plot but "
                "ends with a cliffhanger - an exciting moment that sets "
                "up the next adventure without full resolution."
                f"{multi_part_note}"
                f"{direction_note}"
                f"{recap_instruction}"
            )

        return instruction

    def _build_refinement_prompt(self) -> str:
        """Build a prompt for refining an existing story.

        Creates a focused editing prompt that instructs the LLM to modify
        the existing story based on specific requested changes, rather than
        generating a completely new story.
        """
        parts = []

        parts.append(
            "STORY REFINEMENT TASK: You are editing an existing children's story. "
            "Make ONLY the requested changes below. Keep everything else — plot, characters, "
            "structure, pacing, and wording — as close to the original as possible.\n"
        )

        # Include world definition if available (always verbatim)
        if self.world:
            parts.append(f"\nSTORY WORLD:\n{self.world}\n")

        # Include context if available (character descriptions, etc.)
        if self.context:
            parts.append(f"\nSTORY CONTEXT:\n{self.context}\n")

        # Include the original user prompt for thematic reference
        if self.prompt:
            parts.append(f"\nORIGINAL PROMPT: {self.prompt}\n")

        parts.append(f"\nORIGINAL STORY:\n{self.original_story}\n")
        parts.append(f"\nREQUESTED CHANGES:\n{self.refinement_instructions}\n")

        # Constraints framed as "maintain these" not "write with these"
        parts.append(f"\nMaintain the {self.tone} tone and {self.style} style throughout.")
        voice_instruction = self._get_voice_instruction()
        if voice_instruction:
            parts.append(f"\nMaintain the voice / narrator style: {voice_instruction}")
        if self.theme:
            parts.append(f"\nPreserve the theme of {self.theme}.")
        if self.characters:
            parts.append(f"\nKeep these characters consistent: {', '.join(self.characters)}.")
        if self.setting:
            parts.append(f"\nMaintain the setting: {self.setting}.")
        parts.append(f"\nAge-appropriate constraints: {self._get_age_appropriate_guidance()}")
        parts.append("\n\nThe refined story must remain:")
        parts.append("\n- Completely safe and appropriate for children")
        parts.append("\n- Consistent with the original characters and setting")
        parts.append("\n- The same length and structure as the original unless the changes require otherwise")

        parts.append("\n\nOutput ONLY the complete refined story text. Do not include commentary or explanations.")

        return "".join(parts)

    @property
    def story(self) -> str:
        """
        Get the complete story generation prompt.

        Returns:
            str: A comprehensive prompt for story generation
        """
        # Start with the base prompt
        prompt_parts = []

        # Refinement mode: dedicated path for modifying an existing story
        if self.refinement_mode:
            return self._build_refinement_prompt()

        # Add continuation instruction if in continuation mode
        if self.continuation_mode:
            continuation_instruction = self._build_continuation_instruction()
            prompt_parts.append(f"{continuation_instruction}\n\n")

        # Add world definition (always included verbatim, before context)
        if self.world:
            prompt_parts.append(f"Story world definition:\n{self.world}\n\n")

        # Add context if provided
        if self.context:
            prompt_parts.append(f"Context for story generation:\n{self.context}\n")
            if self.has_old_context and not self.continuation_mode:
                prompt_parts.append(
                    "If you notice characters or events from older stories in the context above, "
                    "feel free to include natural callbacks — have characters reminisce about "
                    "past adventures or reference earlier events. This creates a richer, "
                    "connected story world.\n\n"
                )
            if not self.continuation_mode:
                prompt_parts.append("Based on the above context, ")
        elif self.world and not self.continuation_mode:
            prompt_parts.append("Based on the story world above, ")

        # Main story instruction (skip if in continuation mode since context IS the story)
        if not self.continuation_mode:
            prompt_parts.append(f"write {self._get_length_description()} story")

            # Add style and tone
            prompt_parts.append(f" that is {self._get_style_description()}")
            prompt_parts.append(f" with a {self.tone} tone")

            # Add voice archetype instruction (prominent placement for LLM attention)
            voice_instruction = self._get_voice_instruction()
            if voice_instruction:
                prompt_parts.append(f"\n\nVOICE / NARRATOR STYLE: {voice_instruction}")

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

            # Add voice instruction in continuation mode
            voice_instruction = self._get_voice_instruction()
            if voice_instruction:
                prompt_parts.append(f"\n\nVOICE / NARRATOR STYLE: {voice_instruction}")

            # Preserve story parameters in continuation
            if self.setting:
                prompt_parts.append(f"\n\nMaintain the setting: {self.setting}")
            if self.characters:
                char_list = ", ".join(self.characters)
                prompt_parts.append(f"\nKeep these characters consistent: {char_list}")
            if self.theme:
                prompt_parts.append(f"\nPreserve the theme of {self.theme}")
            if self.learning_focus:
                prompt_parts.append(
                    f"\nContinue incorporating learning about {self.learning_focus} naturally into the story"
                )

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
        for i in range(num_images):
            image_parts = []

            # Add world definition for visual consistency
            if self.world:
                image_parts.append(f"Story world (character and setting reference):\n{self.world}\n\n")

            # Add context as a separate section if present
            if self.context:
                image_parts.append(f"Context for illustration:\n{self.context}\n\n")

            # Base instruction with scene differentiation for multi-image sets
            if num_images > 1:
                scene_labels = ["opening", "middle", "climactic", "closing", "epilogue"]
                label = scene_labels[i] if i < len(scene_labels) else f"scene {i + 1}"
                image_parts.append(f"Create a detailed, beautiful, child-friendly illustration for the {label} scene")
            else:
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
