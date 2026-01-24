# StoryForge

StoryForge is a command-line tool that generates illustrated children's stories using AI language models. Simply provide a story prompt, and StoryForge will create both a short story and accompanying AI-generated images.

## Supported AI Backends

- **Google Gemini** - Fully supported for story and image generation
- **OpenAI** - Fully supported for story and image generation
- **Anthropic** - Supported for story (text) generation only; image generation is not currently supported

## Features

- 📖 Generate custom children's stories from simple prompts
- 🎨 Create AI illustrations with multiple art styles (chibi, realistic, cartoon, watercolor, sketch)
- ⚙️ Flexible story customization (age range, length, tone, theme, learning focus)
- 💾 Save stories and images with organized output directories
- 🖥️ Interactive terminal interface or direct CLI usage
- 📚 Context system for character consistency across stories
- ⏯️ **Checkpoint system** for resuming interrupted sessions
- 📝 **Story extension** - Continue existing stories with wrap-up or cliffhanger endings
- 🔄 **Intelligent context summarization** for efficient token usage with large context files

## Configuration

For detailed configuration options, defaults, and examples see the full configuration reference: [Configuration Documentation](docs/CONFIGURATION.md)

StoryForge supports model configuration for OpenAI backends via the config file:
- `openai_story_model` (default: `gpt-5.2`) - Model for story generation
- `openai_image_model` (default: `gpt-image-1.5`) - Model for image generation
- `image_count` (default: `3`) - Number of images to generate per story (1-5)

See the [Configuration Documentation](docs/CONFIGURATION.md) for all available options.

### Generate a default config file

```bash
# Create config file in default location
storyforge config init

# Force overwrite an existing config file
storyforge config init --force

# Create config at custom location
storyforge config init --path /path/to/config.ini
```

The config file will be created at `~/.config/storyforge/storyforge.ini` by default. You can override the location using the `STORYFORGE_CONFIG` environment variable.

### Command Alias

For convenience, `sf` can be used as a shorter alias for all `storyforge` commands:

```bash
# These are equivalent
storyforge "A brave knight befriends a dragon"
sf "A brave knight befriends a dragon"

# Works with all commands
sf config init
sf continue
sf extend
sf --help
```

## Requirements

## Checkpoint System

StoryForge automatically saves your progress during story generation, allowing you to resume from any point if the process is interrupted or if you want to retry different options.

### Resume from Previous Sessions

```bash
# Resume from a previous session (interactive selection)
storyforge continue

# Or use the --continue flag with main command
storyforge main --continue
```

This will show you the last 5 sessions and let you choose:

- **For interrupted sessions**: Resume from where you left off
- **For completed sessions**: Choose to:
  - Generate new images with the same story
  - Modify and regenerate the story
  - Save the story as context for future use
  - Start completely over with the same parameters

### Checkpoint Storage

Checkpoint files are automatically stored in:
- **Linux/macOS**: `~/.local/share/storyforge/checkpoints/`
- **Windows**: `%APPDATA%\storyforge\storyforge\checkpoints\`

The system automatically cleans up old checkpoints, keeping the 15 most recent sessions. Stale active sessions (older than 24 hours) are automatically marked as failed/abandoned.

### Example Workflow

```bash
# Start a story generation
storyforge "A dragon learning to dance"

# If interrupted, resume later with:
storyforge continue
# Select your session and choose where to resume
```

## Story Extension

Create continuations of previously generated stories with the `extend` command. This is perfect for creating sequels or adding new chapters to existing stories.

### Extend an Existing Story

```bash
# Interactive story selection from recent stories
storyforge extend

# The extend command will:
# 1. Show you a list of recently generated stories
# 2. Let you select which story to continue
# 3. Ask if you want to wrap up or leave a cliffhanger
# 4. Generate a continuation based on your choice
```

### Example Extend Workflow

```bash
# First, generate a story
storyforge "A brave mouse named Max finds a magic acorn"

# Later, extend it with a continuation
storyforge extend

# Output:
# Recent stories:
#   1. "A brave mouse named Max finds a magic acorn" (2025-10-26 14:30)
#   2. "Two robots become friends" (2025-10-25 10:15)
#   ...
# 
# Select story to extend [1-5]: 1
# 
# How should the continuation end?
#   1. Wrap up the story (satisfying conclusion)
#   2. Leave a cliffhanger (sets up next adventure)
# 
# Select ending type [1/2]: 2
#
# Generating continuation...
```

The extended story will:
- Continue from where the original story left off
- Maintain character consistency and story context
- Be saved in a new timestamped output directory
- Include the original story context for reference

## Story Chain Tracking

When you extend stories multiple times, StoryForge automatically tracks the complete "chain" of related stories. This makes it easy to see the full lineage and export complete story sagas.

### View Story Chains

```bash
# When extending a story, StoryForge shows the full chain
storyforge extend

# Example output:
# Story Chain (3 parts):
#   1. "A wizard finds a mysterious artifact" (2025-11-05 10:00)
#   2. "The artifact reveals its power" (2025-11-05 12:00)  
#   3. "The final confrontation" (2025-11-05 14:00)  ← You are here
```

### Export Complete Story Chains

Combine all parts of a story chain into a single readable file:

```bash
# Interactive selection from available chains
storyforge export-chain

# Export a specific chain by name match
storyforge export-chain -c wizard_artifact

# Specify custom output location
storyforge export-chain -c wizard_story -o my_complete_saga.txt
```

The exported file will contain:
- All story parts in chronological order
- Clear section dividers between parts
- Metadata about when each part was created

For detailed information about story chain tracking, see: [Story Chain Tracking Documentation](docs/STORY_CHAIN_TRACKING.md)

## Installation

### Recommended: Using uv

```bash
uv tool install StoryForge
```

If you don't have uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Alternative: Using pipx

```bash
pipx install StoryForge
```

If you don't have pipx:

```bash
# macOS: brew install pipx
# Ubuntu/Debian: sudo apt install pipx
# Or: pip install pipx
```

## Setup

Choose one of the supported AI backends and configure the corresponding API key:

### Google Gemini

1. Visit [Google AI Studio](https://aistudio.google.com/) to get your free Gemini API key
2. Set the environment variable:

```bash
export GEMINI_API_KEY=your_api_key_here
```

### OpenAI

1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set the environment variable:

```bash
export OPENAI_API_KEY=your_api_key_here
```

### Anthropic (Experimental)

1. Get your API key from [Anthropic Console](https://console.anthropic.com/)
2. Set the environment variable:

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

### Environment Variables

| Variable | Backend | Status | Description |
|----------|---------|---------|-------------|
| `GEMINI_API_KEY` | Google Gemini | ✅ Fully Supported | Required for Gemini backend |
| `GEMINI_IMAGE_MODEL` | Google Gemini | 🔧 Optional | Override image model (e.g., `gemini-2.5-flash-image`) |
| `OPENAI_API_KEY` | OpenAI | ✅ Fully Supported | Required for OpenAI backend |
| `ANTHROPIC_API_KEY` | Anthropic | 🧪 Experimental | Required for Anthropic backend |
| `LLM_BACKEND` | All | Optional | Force specific backend (`gemini`, `openai`, `anthropic`) |

**Note**: StoryForge will automatically detect which backend to use based on available API keys. If multiple keys are set, you can specify which backend to use with the `LLM_BACKEND` environment variable.

Add environment variables to your shell profile (`.bashrc`, `.zshrc`, etc.) to make them permanent:

```bash
# Example for Gemini
echo 'export GEMINI_API_KEY=your_api_key_here' >> ~/.bashrc
source ~/.bashrc

# Example for OpenAI
echo 'export OPENAI_API_KEY=your_api_key_here' >> ~/.bashrc
source ~/.bashrc
```

## Usage

### Basic Story Generation

```bash
# Generate a story from a simple prompt
storyforge "Tell me a story about a robot learning to make friends"
```

### Continue an Existing Story

```bash
# Extend a previously generated story
storyforge extend
```

### Resume a Previous Session

```bash
# Resume from an interrupted or completed session
storyforge continue
```

### Interactive prompts

The CLI is interactive and will ask for confirmation and decisions during the run (for images, story refinements, etc.).

### Advanced Options

```bash
storyforge "A brave mouse goes on an adventure" \
  --age-range preschool \
  --length short \
  --tone exciting \
  --image-style cartoon \
  --output-dir my_story \
  -n 3
```

#### Available Options

- **Age Range**: `toddler`, `preschool`, `early_reader`, `middle_grade`
- **Length**: `flash`, `short`, `medium`, `bedtime`
- **Style**: `adventure`, `comedy`, `fantasy`, `fairy_tale`, `friendship`, `random`
- **Tone**: `gentle`, `exciting`, `silly`, `heartwarming`, `magical`, `random`
- **Theme**: `courage`, `kindness`, `teamwork`, `problem_solving`, `creativity`, `random`
- **Image Style**: `chibi`, `realistic`, `cartoon`, `watercolor`, `sketch`

### All Available Commands

```bash
# Generate a new story
storyforge "Your story prompt here" [options]

# Continue/extend an existing story
storyforge extend

# Export a complete story chain
storyforge export-chain [-c CONTEXT_NAME] [-o OUTPUT_FILE]

# Resume a previous session
storyforge continue

# Initialize configuration file
storyforge config init [--force] [--path PATH]

# Show help
storyforge --help
storyforge extend --help
storyforge export-chain --help
storyforge continue --help
storyforge config --help
```

## Tab Completion

Enable tab completion for easier CLI usage:

```bash
storyforge --install-completion
```

Or manually for bash/zsh:

```bash
eval "$(storyforge --show-completion)"
```

## Output

StoryForge creates timestamped directories containing:
- `story.txt` - The generated story
- `*.png` - AI-generated illustrations
- Organized by creation date/time

## Development

For development setup, testing, and contributing guidelines, see [`DEV.md`](DEV.md).

## License

MIT License - see [`LICENSE`](LICENSE) file for details.
