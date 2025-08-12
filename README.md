# StoryForge

StoryForge is a command-line tool that generates illustrated children's stories using Google's Gemini AI. Simply provide a story prompt, and StoryForge will create both a short story and accompanying AI-generated images.

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
