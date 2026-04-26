# StoryForge

A CLI/TUI that generates illustrated children's stories using AI. Provide a prompt, get a story and AI-generated images.

## Features

- 📖 Story generation from simple prompts with customizable age range, length, tone, theme, and style
- 🎨 AI illustrations in multiple art styles (chibi, realistic, cartoon, watercolor, sketch)
- 🗣️ **Voice archetypes** — narrator styles (anapestic, sardonic, picaresque, gothic, lyrical, and more)
- 🌍 **World definitions** — persistent `world.md` for characters, places, and lore across stories
- 📚 **Story extension** — continue stories with an interactive TUI picker; chain tracking and export
- 👤 **Character registry** — tracks appearances and injects visual descriptions into image prompts
- ⏯️ **Checkpoint system** — resume interrupted sessions with `sf continue`
- 🔄 Context summarization with temporal sampling, sentence deduplication, and token budget management
- 🔁 Automatic retry with exponential backoff for transient API errors

**Backends:** [Google Gemini](https://aistudio.google.com/apikey) ✅ | [OpenAI](https://platform.openai.com/api-keys) ✅ | [Anthropic](https://console.anthropic.com/) (text only)

## Installation

Requires **Python 3.12+** and at least one API key from a supported backend.

```bash
# Install with uv (recommended)
uv tool install StoryForge

# Or with pipx
pipx install StoryForge
```

Set up an API key (add to your shell profile to persist):

```bash
export GEMINI_API_KEY=your_key    # or OPENAI_API_KEY or ANTHROPIC_API_KEY
```

StoryForge auto-detects the backend from available keys. Override with `LLM_BACKEND=gemini|openai|anthropic`.

## Quick Start

```bash
# Generate a story
sf "A brave mouse named Max finds a magic acorn"

# With options
sf "A dragon learns to fly" \
  --age-range preschool --length short --tone exciting \
  --voice anapestic --image-style watercolor \
  --setting "enchanted forest" --character "Luna the owl" -n 3

# Resume an interrupted session
sf continue

# Extend a previous story (interactive TUI picker)
sf extend

# Export a multi-part story chain to one file
sf export-chain
```

> **Tip:** `sf` is a shorthand alias for `storyforge`. All commands work with either.

### Story Options

| Option | Values |
|--------|--------|
| `--age-range` | `toddler`, `preschool`, `early_reader`, `middle_grade` |
| `--length` | `flash`, `short`, `medium`, `bedtime` |
| `--style` | `adventure`, `comedy`, `fantasy`, `fairy_tale`, `friendship`, `random` |
| `--tone` | `gentle`, `exciting`, `silly`, `heartwarming`, `magical`, `random` |
| `--voice` | `anapestic`, `sardonic`, `picaresque`, `iambic`, `fable`, `gothic`, `nonsense`, `lyrical`, `epistolary`, `random` |
| `--theme` | `courage`, `kindness`, `teamwork`, `problem_solving`, `creativity`, `random` |
| `--image-style` | `chibi`, `realistic`, `cartoon`, `watercolor`, `sketch` |
| `--setting` | Free text (e.g., `"enchanted forest"`) |
| `--character` | Repeatable (e.g., `--character "Max the mouse" --character "Luna the owl"`) |
| `-n` | Image count (1–5, default: 3) |

### All Commands

```bash
sf "prompt" [options]           # Generate a new story
sf continue                     # Resume a previous session
sf extend                       # Extend a previous story
sf export-chain [-c NAME] [-o FILE]  # Export story chain
sf config init [--force]        # Generate default config file
sf world init                   # Create world.md template
sf world edit                   # Open world.md in $EDITOR
sf world show                   # Display world.md contents
sf world path                   # Show world.md location
sf --help                       # Full help
```

## Configuration

StoryForge can be configured via CLI flags, a config file, or both (CLI flags take priority).

```bash
# Generate a default config file
sf config init
```

Config file location (first found wins): `$STORYFORGE_CONFIG` → `~/.config/storyforge/storyforge.ini` → `~/.storyforge.ini` → `./storyforge.ini`

See [**docs/CONFIGURATION.md**](docs/CONFIGURATION.md) for the full reference of all options, defaults, and examples.

## World Definitions

Define your story universe in a persistent `world.md` file — characters, places, lore, and tone notes. This content is injected into every story prompt, giving the LLM consistent world knowledge across all generations.

```bash
sf world init     # Create from template
sf world edit     # Open in $EDITOR (creates if missing)
sf world show     # Display current contents
sf world path     # Show file location
```

The template includes sections for **Characters**, **Places**, **Rules & Lore**, **Relationships**, and **Tone & Style Notes**. You fill in what matters for your stories — keep it concise since it counts against the token budget.

**HTML comments** (`<!-- ... -->`) are stripped before prompt injection, so you can leave yourself notes that won't reach the LLM:

```markdown
## Characters

### Luna
A curious 7-year-old with curly red hair and bright green eyes.
Always wears purple rain boots, even on sunny days.
<!-- TODO: decide if she has a pet yet -->
```

**File location:** `./context/world.md` if a local `context/` directory exists, otherwise `~/.local/share/storyforge/context/world.md`.

## Story Chains

When you extend stories multiple times, StoryForge tracks the full chain. During extension, the chain lineage is displayed. Use `sf export-chain` to combine all parts into a single file.

See [**docs/STORY_CHAIN_TRACKING.md**](docs/STORY_CHAIN_TRACKING.md) for details.

## Output

Stories are saved to timestamped directories containing `story.txt` and `*.png` illustrations.

## Tips

- **Tab completion:** `sf --install-completion` (or `eval "$(sf --show-completion)"` for manual setup)
- **Offline dev mode:** `sf "any prompt" --debug` loads a test story instead of calling APIs
- **Verbose output:** `sf "prompt" --verbose` for detailed generation logs

## Development

See [**DEV.md**](DEV.md) for setup, testing, and contributing.

## License

MIT — see [LICENSE](LICENSE).
