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
make mcp        # Run the bundled stdio MCP server
make mcp-dev    # Open the MCP development inspector
make lint       # Lint and format code (auto-fixes issues)
make lint-check # Check linting without auto-fixes
make dead-code  # Check production code for unused symbols
make typecheck  # Run type checking only
make coverage   # Run tests with coverage report (xml + html + term)
make clean      # Clean up build artifacts, caches, and remove .venv
```

## Code Quality Tools

The project uses:
- **ruff** - Fast Python linter and formatter
- **mypy** - Static type checking
- **Vulture** - Dead-code detection with an explicit framework-entry-point whitelist
- **pre-commit** - Git hooks for automated checks

### Pre-commit Hooks

After `make install`, enable git hooks:

```bash
pre-commit install
```

## Debug Mode

Use the `--debug` flag to load the packaged test story from [`storyforge/test_story.txt`](storyforge/test_story.txt)
without initializing a provider:

```bash
sf "Any prompt here" --debug
```

Accepting and saving that draft remains API-free. Refinement, video prompts, and image generation initialize a provider
only when requested and therefore require the corresponding API key.

## Project Structure

```
storyforge/
├── __init__.py
├── anthropic_backend.py # Anthropic Claude API integration
├── atomic_io.py         # Atomic replacement for user-authored text files
├── checkpoint.py        # Session persistence and resume
├── classic_cli.py       # Rich MCP client
├── cli.py               # argparse entry point and UI routing
├── config.py            # Configuration loading and validation
├── console.py           # Shared Rich console instance
├── context.py           # Context management, character registry, summarization
├── external_viewer.py   # Platform-aware external file/image opening
├── gemini_backend.py    # Gemini API integration
├── llm_backend.py       # LLM backend interface and shared helpers
├── model_cache.py       # Atomic, TTL-based provider model cache
├── model_discovery.py   # Provider model-list API adapters
├── model_ranking.py     # Provider-aware model filtering and ranking
├── mcp_client.py        # Shared typed stdio MCP client
├── mcp_models.py        # MCP request/result schemas
├── mcp_server.py        # Bundled StoryForge MCP server
├── openai_backend.py    # OpenAI API integration (GPT + image models)
├── paths.py             # Shared output and world-file path helpers
├── phase_executor.py    # Phase-based execution engine
├── portable_text.py     # Clipboard/export ASCII normalization
├── prompt.py            # Prompt handling and validation
├── py.typed             # PEP 561 type marker
├── test_story.txt       # Test story for debug mode
├── tui.py               # Unified full-screen Textual application
├── workflow.py          # UI-independent staged workflow services
├── world_template.py    # World definition template for world.md
└── schema/              # Configuration and prompt validation schema
    ├── __init__.py
    ├── config_schema.py
    ├── core.py
    └── validation.py

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
- Entry points in `pyproject.toml`: `storyforge`, `sf` (short alias), and `storyforge-mcp`

### Client / Server Boundary

- The Textual and Rich interfaces are presentation-only MCP clients.
- `storyforge-mcp` owns workflow execution, checkpoints, provider access, and file operations.
- Long mutations run on one bounded server worker; MCP progress notifications are bridged back to the active client.
- Workflow services contain no Textual, Rich, or MCP imports and should be tested independently.
- MCP tools never prompt. Clients collect review, refinement, media, overwrite, and destructive-operation decisions explicitly.

### Test Strategy

- Test workflow behavior directly without starting Textual or a subprocess MCP server.
- Use the in-process MCP server for protocol contract tests.
- Use Textual pilot tests for screen transitions, focus behavior, and failed-operation recovery.
- Mock every provider API. Unit and CI tests must never require credentials or perform provider network calls.
- Run filesystem-producing tests under `tmp_path`; a clean temporary working-directory run must leave no generated
  story or `MagicMock/` artifacts behind.

### Distribution Smoke Test

Source-checkout tests do not prove that runtime data is packaged. Build and inspect both artifacts, then install the
wheel into a fresh environment:

```bash
DIST_AUDIT_DIR="$(mktemp -d)"
UV_CACHE_DIR=/tmp/storyforge-uv-cache uv build --out-dir "$DIST_AUDIT_DIR"
uv venv "$DIST_AUDIT_DIR/venv"
uv pip install --python "$DIST_AUDIT_DIR/venv/bin/python" "$DIST_AUDIT_DIR"/*.whl
```

Verify that the wheel and sdist contain `storyforge/py.typed` and `storyforge/test_story.txt`, run `sf --help` and
`sf --version` from the fresh environment, and exercise debug draft creation with all provider keys unset. CI performs
the same check in its `package-smoke` job.

## Contributing Workflow

1. **Setup**: `make install && pre-commit install`
2. **Development**: `make test && make lint-check`
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
- **Pre-commit issues**: Run `make lint` to fix formatting

### Development Tips

- Use `--verbose` flag for detailed output
- Use `--debug` flag for offline development
- Use `make clean` if you encounter dependency issues
- Check generated output directories for saved files

## Release Process

1. **Update version** in `storyforge/__init__.py` (`pyproject.toml` reads it dynamically)
2. **Update CHANGELOG.md** — move `[Unreleased]` items into a new versioned section with today's date
3. **Validate**: `make lint && make lint-check && make test && make coverage`
4. **Commit**: `git commit -am "release: v0.0.X"`
5. **Tag**: `git tag v0.0.X`
6. **Push**: `git push origin main --tags`
