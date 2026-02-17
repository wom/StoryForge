# Development Guide

This document contains development setup instructions, testing procedures, and contribution guidelines for StoryForge.

## Quick Start

The easiest way to set up the development environment is using the provided Makefile:

```bash
# Set up everything (creates venv, installs dependencies, installs package in editable mode)
make install

# Run the application
make run

# Run tests
make test
```

## Prerequisites

The project requires [uv](https://github.com/astral-sh/uv) for fast Python virtual environment and dependency management. See README.md for installation instructions.

## Makefile Commands

The Makefile handles all development tasks:

```bash
make install    # Complete setup (creates .venv, installs dependencies)
make venv       # Create virtual environment only
make test       # Run all tests
make run        # Run the application
make debug      # Run with Textual debugging console
make console    # Start Textual debug server
make lint       # Lint and format code (auto-fixes issues)
make lint-check # Check linting without auto-fixes
make typecheck  # Run type checking only
make clean      # Clean up build artifacts, caches, and remove .venv
```

## Code Quality Tools

The project uses:
- **ruff** - Fast Python linter and formatter
- **mypy** - Static type checking
- **pre-commit** - Git hooks for automated checks

### Pre-commit Hooks

After `make install`, enable git hooks:

```bash
pre-commit install
```

## Debug Mode

Use the `--debug` flag to load a test story from [`storyforge/test_story.txt`](storyforge/test_story.txt) instead of calling the API:

```bash
storyforge "Any prompt here" --debug
```

## Project Structure

```
storyforge/
├── __init__.py
├── StoryForge.py        # Main CLI application (Typer)
├── anthropic_backend.py # Anthropic Claude API integration
├── checkpoint.py        # Session persistence and resume
├── config.py            # Configuration loading and validation
├── console.py           # Shared Rich console instance
├── context.py           # Context management, character registry, summarization
├── gemini_backend.py    # Gemini API integration
├── llm_backend.py       # LLM backend interface and shared helpers
├── model_discovery.py   # Dynamic Gemini model discovery
├── openai_backend.py    # OpenAI API integration (GPT + image models)
├── phase_executor.py    # Phase-based execution engine
├── prompt.py            # Prompt handling and validation
├── story_picker.py      # Interactive TUI story picker (Textual)
├── test_story.txt       # Test story for debug mode
└── schema/              # Validation schema and CLI integration
    ├── __init__.py
    ├── cli_integration.py
    ├── config_schema.py
    └── core.py

tests/                   # Test suite (pytest)
docs/                    # Documentation
Makefile                 # Development automation
pyproject.toml           # Project configuration
.pre-commit-config.yaml  # Pre-commit hooks
```

## Code Architecture

### Backend System

- All backends implement the `LLMBackend` interface in [`llm_backend.py`](storyforge/llm_backend.py)
- Shared image prompt helpers in base class: `_build_image_prompt_request()`, `_parse_numbered_prompts()`, `_segment_story()`, `_get_scene_labels()`
- Current backends:
  - **Gemini** ([`gemini_backend.py`](storyforge/gemini_backend.py)) - Full features (text + image generation)
  - **Anthropic** ([`anthropic_backend.py`](storyforge/anthropic_backend.py)) - Text + image prompt generation (no image rendering)
  - **OpenAI** ([`openai_backend.py`](storyforge/openai_backend.py)) - Full features (GPT + image models)
- To add a new backend: implement `LLMBackend` interface in a new module

### Package Organization

- All source code in the `storyforge/` package
- Use package imports: `from storyforge.module import Class`
- Entry points in `pyproject.toml`: `storyforge` and `sf` (short alias)

## Contributing Workflow

1. **Setup**: `make install && pre-commit install`
2. **Development**: `make test && make lint`
3. **Commit**: Pre-commit hooks run automatically

### Code Style Guidelines

- Follow PEP 8 (enforced by ruff)
- Use type hints (checked by mypy)
- Write docstrings for public functions and classes
- Line length: 119 characters
- Use double quotes for strings

## Environment Variables

### Required for Runtime
Choose one or more backends by setting the corresponding API keys:
- `GEMINI_API_KEY` - Your Gemini API key (see README.md for setup)
- `ANTHROPIC_API_KEY` - Your Anthropic Claude API key
- `OPENAI_API_KEY` - Your OpenAI API key

### Backend Selection
- Set `LLM_BACKEND=gemini|anthropic|openai` to force a specific backend
- Default: Auto-detects available backends (prefers Gemini)

## Manual Setup (Alternative)

Without the Makefile:

```bash
uv venv .venv && source .venv/bin/activate && uv pip install .[dev]
```

## Configuration Files

- **`pyproject.toml`** - Project metadata, dependencies, tool configuration
- **`Makefile`** - Development automation
- **`.pre-commit-config.yaml`** - Git hook configuration

## Troubleshooting

### Common Issues

- **"uv not found"**: Install uv first, then run `make install`
- **Import errors**: Run `make clean` then `make install`
- **Test failures**: Ensure at least one API key is set (see README.md)
- **Pre-commit issues**: Run `make lint` to fix formatting

### Development Tips

- Use `--verbose` flag for detailed output
- Use `--debug` flag for offline development
- Use `make clean` if you encounter dependency issues
- Check generated output directories for saved files