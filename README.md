# StoryForge

StoryForge is a command-line tool that generates illustrated children's stories using AI language models. Simply provide a story prompt, and StoryForge will create both a short story and accompanying AI-generated images. Supports OpenAI, Google's Gemini AI and Anthropic's Claude AI backends.

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
    
### Choose Your AI Backend

StoryForge supports multiple AI backends. Choose one or set up both:

#### Option 1: Google Gemini (Full Features)
**Supports:** Story generation + Image generation

1. Visit [Google AI Studio](https://aistudio.google.com/) to get your free Gemini API key
2. Set the environment variable:
```bash
export GEMINI_API_KEY=your_api_key_here
```

#### Option 2: Anthropic Claude (Text Only)
**Supports:** Story generation only (excellent quality)

1. Visit [Anthropic Console](https://console.anthropic.com/) to get your Claude API key
2. Set the environment variable:
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

## Option 3: OpenAI (Text Only)
**Supports:** Story generation only (excellent quality)

1. Visit [OpenAI Platform](https://platform.openai.com/) to get your OpenAI API key
2. Set the environment variable:
```bash
export OPENAI_API_KEY=your_api_key_here
```

#### Option 4: Hybrid Setup (Best of Both)
Set up both backends for maximum flexibility:
```bash
export GEMINI_API_KEY=your_gemini_key_here
export ANTHROPIC_API_KEY=your_anthropic_key_here
```

Add these to your shell profile (`.bashrc`, `.zshrc`, etc.) to make them permanent.

**Backend Selection:**
- StoryForge automatically detects available backends
- Prefers Gemini for full features, falls back to others
- Use `LLM_BACKEND=anthropic` to force Claude for text generation

## Usage

### Basic Story Generation

```bash
storyforge "Tell me a story about a robot learning to make friends"
```

### Generate Just an Image

```bash
storyforge image "A friendly robot in a colorful playground"
```

=======
### 1. Get a Gemini API Key

Visit [Google AI Studio](https://aistudio.google.com/) to get your free Gemini API key.

### 2. Set Environment Variable

```bash
export GEMINI_API_KEY=your_api_key_here
```

Add this to your shell profile (`.bashrc`, `.zshrc`, etc.) to make it permanent.

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

### Backend-Specific Notes

- **Gemini**: Supports both story and image generation in one tool
- **Claude**: Excellent story quality, but requires Gemini for images
- **OpenAI**
- **Hybrid**: Use `LLM_BACKEND=anthropic` for Claude stories + Gemini for images

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
