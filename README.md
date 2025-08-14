# StoryForge

StoryForge is a command-line tool that generates illustrated children's stories using AI language models. Simply provide a story prompt, and StoryForge will create both a short story and accompanying AI-generated images.

## Supported AI Backends

- **Google Gemini** - Fully supported for story and image generation
- **OpenAI** - Fully supported for story and image generation
- **Anthropic** - Experimental (coming soon)

## Features

- 📖 Generate custom children's stories from simple prompts
- 🎨 Create AI illustrations with multiple art styles (chibi, realistic, cartoon, watercolor, sketch)
- ⚙️ Flexible story customization (age range, length, tone, theme, learning focus)
- 💾 Save stories and images with organized output directories
- 🖥️ Interactive terminal interface or direct CLI usage
- 📚 Context system for character consistency across stories

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
| `OPENAI_API_KEY` | OpenAI | ✅ Fully Supported | Required for OpenAI backend |
| `ANTHROPIC_API_KEY` | Anthropic | 🧪 Experimental | Required for Anthropic backend (coming soon) |
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
storyforge "Tell me a story about a robot learning to make friends"
```

### Generate Just an Image

```bash
storyforge image "A friendly robot in a colorful playground"
```

### Interactive Mode

```bash
storyforge tui
```

### Advanced Options

```bash
storyforge "A brave mouse goes on an adventure" \
  --age-range preschool \
  --length short \
  --tone exciting \
  --image-style cartoon \
  --output-dir my_story
```

#### Available Options

- **Age Range**: `toddler`, `preschool`, `early_reader`, `middle_grade`
- **Length**: `flash`, `short`, `medium`, `bedtime`
- **Style**: `adventure`, `comedy`, `fantasy`, `fairy_tale`, `friendship`
- **Tone**: `gentle`, `exciting`, `silly`, `heartwarming`, `magical`
- **Theme**: `courage`, `kindness`, `teamwork`, `problem_solving`, `creativity`
- **Image Style**: `chibi`, `realistic`, `cartoon`, `watercolor`, `sketch`

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
