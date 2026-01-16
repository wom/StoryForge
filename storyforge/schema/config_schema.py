"""
Complete StoryForge configuration schema definition.
Centralizes all configuration options with comprehensive metadata.
"""

from .core import ConfigField, ConfigSection, FieldType


def _create_story_section() -> ConfigSection:
    """Create the story configuration section with all fields."""
    section = ConfigSection(name="story", description="Story generation parameters")

    # Story Length
    section.add_field(
        ConfigField(
            name="length",
            field_type=FieldType.STRING,
            default="bedtime",
            section="story",
            description="Controls the target length of the generated story",
            cli_help="Story length (flash, short, medium, bedtime)",
            cli_short="-l",
            valid_values=["flash", "short", "medium", "bedtime"],
            example_values=["bedtime", "short"],
            ini_comment=(
                "Story length options: flash (~100 words), short (~300 words), "
                "medium (~600 words), bedtime (~1000 words)"
            ),
        )
    )

    # Age Range
    section.add_field(
        ConfigField(
            name="age_range",
            field_type=FieldType.STRING,
            default="early_reader",
            section="story",
            description="Target age group for story complexity and themes",
            cli_help="Target age group (toddler, preschool, early_reader, middle_grade)",
            cli_short="-a",
            valid_values=["toddler", "preschool", "early_reader", "middle_grade"],
            example_values=["early_reader", "preschool"],
            ini_comment=(
                "Target age group options: toddler (1-3 years), preschool (3-5 years), "
                "early_reader (5-8 years), middle_grade (8-12 years)"
            ),
        )
    )

    # Story Style
    section.add_field(
        ConfigField(
            name="style",
            field_type=FieldType.STRING,
            default="random",
            section="story",
            description="Overall narrative style and genre of the story",
            cli_help="Story style (adventure, comedy, fantasy, fairy_tale, friendship, random)",
            cli_short="-s",
            valid_values=["adventure", "comedy", "fantasy", "fairy_tale", "friendship", "random"],
            example_values=["fantasy", "adventure"],
            ini_comment="Story style options: adventure, comedy, fantasy, fairy_tale, friendship, random",
        )
    )

    # Story Tone
    section.add_field(
        ConfigField(
            name="tone",
            field_type=FieldType.STRING,
            default="random",
            section="story",
            description="Emotional tone and mood of the story",
            cli_help="Story tone (gentle, exciting, silly, heartwarming, magical, random)",
            cli_short="-t",
            valid_values=["gentle", "exciting", "silly", "heartwarming", "magical", "random"],
            example_values=["heartwarming", "magical"],
            ini_comment="Story tone options: gentle, exciting, silly, heartwarming, magical, random",
        )
    )

    # Story Theme
    section.add_field(
        ConfigField(
            name="theme",
            field_type=FieldType.STRING,
            default="random",
            section="story",
            description="Core message or lesson of the story",
            cli_help="Story theme (courage, kindness, teamwork, problem_solving, creativity, random)",
            valid_values=["courage", "kindness", "teamwork", "problem_solving", "creativity", "random"],
            example_values=["kindness", "courage"],
            ini_comment="Story theme options: courage, kindness, teamwork, problem_solving, creativity, random",
        )
    )

    # Learning Focus
    section.add_field(
        ConfigField(
            name="learning_focus",
            field_type=FieldType.STRING,
            default="",
            section="story",
            description="Educational element to incorporate into the story",
            cli_help="Learning focus (counting, colors, letters, emotions, nature). Leave empty for none",
            valid_values=["counting", "colors", "letters", "emotions", "nature", ""],
            example_values=["colors", "emotions"],
            ini_comment="Learning focus options: counting, colors, letters, emotions, nature (leave empty for none)",
        )
    )

    # Setting
    section.add_field(
        ConfigField(
            name="setting",
            field_type=FieldType.STRING,
            default="",
            section="story",
            description="Where the story takes place",
            cli_help="Story setting (free text)",
            example_values=["enchanted forest", "space station", "underwater kingdom"],
            ini_comment="Default story setting (leave empty for none)",
        )
    )

    # Characters
    section.add_field(
        ConfigField(
            name="characters",
            field_type=FieldType.LIST,
            default="",
            section="story",
            description="Main characters in the story",
            cli_help="Character names/descriptions (multi-use flag)",
            example_values=["Luna the wise owl", "Max the brave mouse"],
            ini_comment="Default characters, comma-separated (leave empty for none)",
        )
    )

    return section


def _create_images_section() -> ConfigSection:
    """Create the images configuration section."""
    section = ConfigSection(name="images", description="Image generation parameters")

    section.add_field(
        ConfigField(
            name="image_style",
            field_type=FieldType.STRING,
            default="chibi",
            section="images",
            description="Visual art style for generated images",
            cli_help="Image art style (chibi, realistic, cartoon, watercolor, sketch)",
            valid_values=["chibi", "realistic", "cartoon", "watercolor", "sketch"],
            example_values=["chibi", "watercolor"],
            ini_comment="Image art style options: chibi, realistic, cartoon, watercolor, sketch",
        )
    )

    section.add_field(
        ConfigField(
            name="image_count",
            field_type=FieldType.INTEGER,
            default=3,
            section="images",
            description="Number of images to generate for the story",
            cli_help="Number of images to generate (1-5)",
            cli_short="-n",
            example_values=["3", "5"],
            ini_comment="Number of images to generate (default: 3, range: 1-5)",
        )
    )

    return section


def _create_output_section() -> ConfigSection:
    """Create the output configuration section."""
    section = ConfigSection(name="output", description="Output and file handling options")

    section.add_field(
        ConfigField(
            name="output_dir",
            field_type=FieldType.PATH,
            default="",
            section="output",
            description="Directory where generated stories and images are saved",
            cli_help="Directory to save the output (default: auto-generated)",
            cli_short="-o",
            example_values=["./my_stories", "/home/user/stories"],
            ini_comment="Default output directory (leave empty for auto-generated timestamp)",
        )
    )

    section.add_field(
        ConfigField(
            name="use_context",
            field_type=FieldType.BOOLEAN,
            default=True,
            section="output",
            description="Whether to use context files from context/ directory for story generation",
            cli_help="Use context files for story consistency",
            ini_comment="Whether to use context files by default: true, false",
        )
    )

    return section


def _create_system_section() -> ConfigSection:
    """Create the system configuration section."""
    section = ConfigSection(name="system", description="System-level configuration options")

    section.add_field(
        ConfigField(
            name="backend",
            field_type=FieldType.STRING,
            default="",
            section="system",
            description="LLM backend for story and image generation",
            cli_help="LLM backend (gemini, openai, anthropic). Leave empty for auto-detection",
            valid_values=["gemini", "openai", "anthropic", ""],
            example_values=["gemini", "anthropic"],
            ini_comment="LLM backend options: gemini, openai, anthropic (leave empty for auto-detection)",
        )
    )

    section.add_field(
        ConfigField(
            name="openai_story_model",
            field_type=FieldType.STRING,
            default="gpt-5.2",
            section="system",
            description="OpenAI model to use for story generation",
            cli_help="OpenAI story model (e.g., gpt-5.2, gpt-4o)",
            example_values=["gpt-5.2", "gpt-4o"],
            ini_comment="OpenAI model for story generation (default: gpt-5.2)",
        )
    )

    section.add_field(
        ConfigField(
            name="openai_image_model",
            field_type=FieldType.STRING,
            default="gpt-image-1.5",
            section="system",
            description="OpenAI model to use for image generation",
            cli_help="OpenAI image model (e.g., gpt-image-1.5, dall-e-3)",
            example_values=["gpt-image-1.5", "dall-e-3"],
            ini_comment="OpenAI model for image generation (default: gpt-image-1.5)",
        )
    )

    section.add_field(
        ConfigField(
            name="verbose",
            field_type=FieldType.BOOLEAN,
            default=False,
            section="system",
            description="Enable detailed output during story generation",
            cli_help="Enable verbose output",
            cli_short="-v",
            ini_comment="Enable verbose output by default: true, false",
        )
    )

    section.add_field(
        ConfigField(
            name="debug",
            field_type=FieldType.BOOLEAN,
            default=False,
            section="system",
            description="Enable debug mode (uses local test story instead of AI generation)",
            cli_help="Enable debug mode (use local file instead of backend for story generation)",
            ini_comment="Enable debug mode by default: true, false",
        )
    )

    return section


# Global schema instance
STORYFORGE_SCHEMA = type(
    "StoryForgeSchema",
    (),
    {
        "story": _create_story_section(),
        "images": _create_images_section(),
        "output": _create_output_section(),
        "system": _create_system_section(),
    },
)()
