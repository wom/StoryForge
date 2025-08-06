# StoryForge

StoryForge is a Textual-based terminal application that uses Google's Gemini API to generate short stories and AI-generated images from user prompts. The app provides a simple TUI (Text User Interface) for entering prompts, confirming actions, and viewing results. When a user submits a prompt, the app generates a story and a corresponding image, saving the image to disk with a descriptive filename. The project is designed for creative exploration, rapid prototyping, and as a demonstration of integrating LLMs and generative image models into a modern Python TUI.

**Features:**
- Enter a custom story prompt and generate a short story using LLM.
- Generate AI image illustrations for your story with flexible options:
  - Use the story as context and describe the image yourself
  - Break the story into logical chunks (paragraphs) and generate an image for each, with consistent style
  - Skip image generation if desired
- Save generated images to disk with creative, context-aware filenames.
- All interactions are handled in a modern, responsive terminal UI.
- Includes unit tests for core logic and easy setup with uv and pytest.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python virtual environment and dependency management.

### 1. Create and activate a virtual environment

```
uv venv .venv
source .venv/bin/activate
```

### 2. Install all dependencies (including dev dependencies) and the package in editable mode

```
pip install -e .
```

Or, if you want to use uv:

```
uv pip install .[dev]
```

## Running the Unit Tests

```
pytest
```

Or, if not in an activated venv:

```
.venv/bin/pytest
```

## Running the Program

After installing in editable mode, run the app with:

```
storyforge
```

This command is available as long as your virtual environment is activated.

### CLI Usage

You can use the command-line interface directly with the unified entry point:

```bash
storyforge story "Tell me a story about a robot"
```

After generating a story, you will be prompted to choose how to generate illustrations:
- Use the story as context and describe the image yourself
- Break the story into logical chunks (paragraphs) and generate an image for each, with consistent style
- Skip image generation

You can also generate a standalone image:
```bash
storyforge image "A beautiful sunset over mountains"
```

Or launch the interactive TUI:
```bash
storyforge tui
```

## Tab Completion

StoryForge CLI supports tab completion for commands, options, and arguments.

### Quick Setup (Automatic)

Enable tab completion with:

```bash
storyforge --install-completion
```

**Note:** If you get an error about existing files, use the manual setup below.

### Manual Setup

If automatic installation doesn't work, you can set up completion manually for your shell:

**Bash** (add to `~/.bashrc`):
```bash
eval "$(storyforge --show-completion)"
```

**Zsh** (add to `~/.zshrc`):
```bash
eval "$(storyforge --show-completion)"
```

**Fish** (add to `~/.config/fish/config.fish`):
```bash
storyforge --show-completion | source
```

After adding the line to your shell configuration, restart your terminal or run:
```bash
source ~/.bashrc  # for bash
source ~/.zshrc   # for zsh
```

### Using Tab Completion

Once enabled, you can use tab completion for:

- **Commands**: `storyforge <TAB>` shows available commands (`story`, `image`, `tui`)
- **Options**: `storyforge story --<TAB>` shows options like `--output-dir`, `--verbose`, etc.
- **Help**: Use `storyforge <command> --help` to see all available options

## Editing and Debugging

- All source code is in the `storyforge/` package. When editing or adding files, use package imports (e.g., `from storyforge.gemini_backend import GeminiBackend`).
- To add a new backend, create a new file in `storyforge/` and implement the `LLMBackend` interface.
- To debug, you can add print statements or use a debugger in any module in the `storyforge/` package.
- After making changes, rerun the tests with `pytest` to ensure everything works.

## Adding New Dependencies

- To add a new main (runtime) dependency:

```
uv pip install <package>
```

- To add a new dev (test/tooling) dependency:

```
uv pip install --extra dev <package>
```

- After adding dependencies, uv will update your `pyproject.toml` and `uv.lock` automatically.

## Notes
- Make sure you have a valid `GEMINI_API_KEY` in your environment before running the program.
- The TUI requires a terminal that supports Textual applications.
