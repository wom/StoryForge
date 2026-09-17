# StoryForge Configuration

This document describes how to configure StoryForge, where the configuration file is located, and what each configuration option means.

## Generating a default configuration file

StoryForge provides a convenient CLI command to create a default config file:

```bash
# Create config file (won't overwrite an existing file)
sf config init

# Force overwrite an existing config file
sf config init --force

# Specify a custom path
sf config init --path /path/to/custom/storyforge.ini
```

The default location is the XDG config directory (typically `~/.config/storyforge/storyforge.ini`). You can override the location by setting the `STORYFORGE_CONFIG` environment variable:

```bash
export STORYFORGE_CONFIG=/path/to/custom/storyforge.ini
```

The generated file is derived from the project schema and contains sensible defaults and helpful inline comments.

## Config file format
- StoryForge uses an INI-style config file (key-value pairs grouped into sections).
- Sections correspond to configuration areas: `[story]`, `[images]`, `[output]`, and `[system]`.
- You can create a default file with the built-in template using the CLI or programmatically via `Config.create_default_config()`.

## Config file priority (highest to lowest)

StoryForge follows the [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html) and common cascading configuration patterns used by many CLI tools (similar to Git, Vim, etc.):

1. Path set in the `STORYFORGE_CONFIG` environment variable (explicit override)
2. XDG config directory: `~/.config/storyforge/storyforge.ini` (modern standard)
3. Home directory: `~/.storyforge.ini` (legacy/traditional location)
4. Current directory: `./storyforge.ini` (project-specific settings)

The first configuration file found in this order is used. Higher priority locations override lower ones.

## How values are resolved
- Values in the config file are loaded and validated against a schema defined in `storyforge/schema/config_schema.py`.
- Command line arguments override config file values.
- Environment variables are not generally supported for individual fields. `STORYFORGE_CONFIG` selects the file,
  `LLM_BACKEND` selects a backend, and `GEMINI_IMAGE_MODEL` remains a provider-specific image-model override.
- Backend selection priority is: explicit command/request value → `LLM_BACKEND` → `system.backend` → API-key auto-detection.
- Explicit model fields take precedence over discovery. An empty model field means automatic selection from the fresh
  provider cache or provider API. `GEMINI_IMAGE_MODEL` is a legacy environment override with precedence over
  `system.gemini_image_model`.

## Sections and fields
Below is a concise reference of available configuration options, their defaults, and acceptable values.

### [story]
- `length` (string) — Default: `bedtime`
  - Options: `flash`, `short`, `medium`, `bedtime`
  - Description: Controls approximate story length (word-count target).
- `age_range` (string) — Default: `early_reader`
  - Options: `toddler`, `preschool`, `early_reader`, `middle_grade`
  - Description: Target age group for story complexity and themes.
- `style` (string) — Default: `random`
  - Options: `adventure`, `comedy`, `fantasy`, `fairy_tale`, `friendship`, `random`
  - Description: Narrative style or genre.
- `tone` (string) — Default: `random`
  - Options: `gentle`, `exciting`, `silly`, `heartwarming`, `magical`, `random`
  - Description: Emotional tone/mood of the story.
- `theme` (string) — Default: `random`
  - Options: `courage`, `kindness`, `teamwork`, `problem_solving`, `creativity`, `random`
  - Description: Core lesson or thematic focus.
- `learning_focus` (string) — Default: `` (empty)
  - Options: `counting`, `colors`, `letters`, `emotions`, `nature` or empty for none
  - Description: Optional educational topic to include.
- `setting` (string) — Default: ``
  - Free-text setting description (e.g., `enchanted forest`).
- `characters` (list) — Default: empty
  - Comma-separated list of character descriptions or names.
- `voice` (string) — Default: `` (empty, no voice applied)
  - Options: `anapestic`, `sardonic`, `picaresque`, `iambic`, `fable`, `gothic`, `nonsense`, `lyrical`, `epistolary`, `random` or empty for none
  - Description: Writing voice archetype that shapes the narrator's style and rhythm. Each voice produces a distinct literary feel (e.g., anapestic = whimsical rhyming verse, sardonic = darkly humorous narrator).

### [images]
- `image_style` (string) — Default: `chibi`
  - Options: `chibi`, `realistic`, `cartoon`, `watercolor`, `sketch`
  - Description: Art style used for generated illustrations.
- `image_count` (integer) — Default: `3`
  - Range: 1–5
  - Description: Number of images to generate per story. When generating multiple images, StoryForge automatically creates scene-specific prompts with story progression (opening, rising action, climax, resolution).

### [output]
- `output_dir` (path) — Default: auto-generated timestamped directory
  - Description: Directory where stories and images are saved.
- `use_context` (boolean) — Default: `true`
  - Description: Whether to load context files from the `context/` directory and include them in prompt generation.
- `world_file` (path) — Default: `` (auto-discover)
  - Description: Explicit world-definition file included verbatim in each story prompt. When empty, StoryForge checks
    the local context directory and then its platform user-data directory.

### [system]
- `backend` (string) — Default: `` (auto-detect)
  - Options: `gemini`, `openai`, `anthropic` or empty for auto-detection.
- `openai_story_model` (string) — Default: `gpt-5.5`
  - Description: OpenAI model used for story generation (e.g., `gpt-5.5`, `gpt-4o`).
- `openai_image_model` (string) — Default: `gpt-image-1.5`
  - Description: OpenAI model used for image generation (e.g., `gpt-image-1.5`, `dall-e-3`).
- `anthropic_story_model` (string) — Default: `` (automatic)
  - Description: Anthropic model used for story and prompt generation. Anthropic does not render images.
- `gemini_story_model` (string) — Default: `` (automatic)
  - Description: Gemini model used for story and prompt generation.
- `gemini_image_model` (string) — Default: `` (automatic)
  - Description: Gemini model used for image rendering. `GEMINI_IMAGE_MODEL` overrides this field when set.
- `verbose` (boolean) — Default: `false`
  - Description: Enable verbose output for debugging and more detailed logs.
- `debug` (boolean) — Default: `false`
  - Description: Enable debug mode which uses a local test story file instead of calling LLM backends.

## Example config file
```
[story]
# Story length options: flash (~100 words), short (~300 words), medium (~600 words), bedtime (~1000 words)
length = bedtime

# Target age group options: toddler (1-3 years), preschool (3-5 years), early_reader (5-8 years), middle_grade (8-12 years)
age_range = early_reader

style = fantasy

tone = heartwarming

# Writing voice archetype (leave empty for none): anapestic, sardonic, picaresque, iambic, fable, gothic, nonsense, lyrical, epistolary, random
voice =

theme = kindness

learning_focus = emotions

setting = enchanted forest

characters = Luna the wise owl, Max the brave mouse

[images]
# Image art style options: chibi, realistic, cartoon, watercolor, sketch
image_style = chibi

[output]
# Default output directory (leave empty for auto-generated timestamp)
output_dir =

# Whether to use context files by default: true, false
use_context = true

# Explicit world file (leave empty for auto-discovery)
world_file =

[system]
# LLM backend options: gemini, openai, anthropic (leave empty for auto-detection)
backend = openai

# Provider-specific model choices. Empty values use automatic discovery.
openai_story_model = gpt-5.5
openai_image_model = gpt-image-1.5
anthropic_story_model =
gemini_story_model =
gemini_image_model =

# Enable verbose output by default: true, false
verbose = false

# Enable debug mode by default: true, false
debug = false
```

## Validation and errors
- When loading configuration, StoryForge validates values against the schema and will raise a `ConfigError` if validation fails.
- Use `Config.validate_config()` to programmatically get a list of validation errors.
- Run `sf config` to edit the active file in the TUI. StoryForge validates the complete candidate before replacing the
  file atomically. Failed saves return to the editor with the unsaved text intact.

## Model discovery cache

Run `sf models` to open the provider/model picker. The automatic option stores an empty model value and lets
StoryForge choose a suitable model. Anthropic has no image-model selection because it cannot render images.

Provider model lists are cached for seven days under the platform user-data directory, normally
`~/.local/share/storyforge/model_cache/` on Linux. `sf models refresh` immediately queries each provider with an
available API key and reports refreshed, skipped, empty, or failed status separately. A failed refresh preserves any
still-valid previous cache. `sf models clear` deletes only cached discovery metadata and does not remove saved model
choices from this configuration file.

## Tips
- Prefer using the XDG config path or set `STORYFORGE_CONFIG` to keep your project-level settings separate.
- Use `debug = true` to load the bundled story without a provider key. Refinement, video prompts, and image generation
  still require a provider if requested.
- If you need reproducible runs, enable context and consider tracking the files used (the application stores context metadata in the session checkpoint).
