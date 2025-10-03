# StoryForge Configuration

This document describes how to configure StoryForge, where the configuration file is located, and what each configuration option means.

## Generating a default configuration file

StoryForge provides a convenient CLI command to create a default config file:

```bash
# Create config file (won't overwrite an existing file)
storyforge config init

# Force overwrite an existing config file
storyforge config init --force

# Specify a custom path
storyforge config init --path /path/to/custom/storyforge.ini
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
- Environment variables are not directly supported for individual fields (except `STORYFORGE_CONFIG` to choose file), but config is validated after loading.

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

### [images]
- `image_style` (string) — Default: `chibi`
  - Options: `chibi`, `realistic`, `cartoon`, `watercolor`, `sketch`
  - Description: Art style used for generated illustrations.

### [output]
- `output_dir` (path) — Default: auto-generated timestamped directory
  - Description: Directory where stories and images are saved.
- `use_context` (boolean) — Default: `true`
  - Description: Whether to load context files from the `context/` directory and include them in prompt generation.

### [system]
- `backend` (string) — Default: `` (auto-detect)
  - Options: `gemini`, `openai`, `anthropic` or empty for auto-detection.
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

[system]
# LLM backend options: gemini, openai, anthropic (leave empty for auto-detection)
backend = openai

# Enable verbose output by default: true, false
verbose = false

# Enable debug mode by default: true, false
debug = false
```

## Validation and errors
- When loading configuration, StoryForge validates values against the schema and will raise a `ConfigError` if validation fails.
- Use `Config.validate_config()` to programmatically get a list of validation errors.

## Tips
- Prefer using the XDG config path or set `STORYFORGE_CONFIG` to keep your project-level settings separate.
- Use `debug = true` while developing to avoid calling external APIs.
- If you need reproducible runs, enable context and consider tracking the files used (the application stores context metadata in the session checkpoint).