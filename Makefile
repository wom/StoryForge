.PHONY: all test run install lint lint-check clean coverage debug console typecheck mcp mcp-dev release

# Use .venv/bin/uv consistently
VENV_ACTIVATE = . .venv/bin/activate &&

# Default target
all: install


venv:
	test -d .venv || uv venv .venv

install: venv
	uv pip install -e .[dev]


# Run tests
test: install
	$(VENV_ACTIVATE) PYTHONPATH=. pytest

# Run tests with coverage
coverage: install
	$(VENV_ACTIVATE) PYTHONPATH=. pytest --cov=storyforge --cov-report=xml --cov-report=html --cov-report=term

# Run the app
run: install
	$(VENV_ACTIVATE) storyforge --help

# Run the bundled StoryForge MCP server over stdio.
mcp: install
	$(VENV_ACTIVATE) storyforge-mcp

# Open the bundled server in the official MCP development inspector.
mcp-dev: install
	$(VENV_ACTIVATE) mcp dev storyforge/mcp_server.py

# [venv] Run attaching to debug console
debug:
	$(VENV_ACTIVATE) textual run --dev `which storyforge`
# [venv] Run textual debug server
console:
	$(VENV_ACTIVATE) textual console 

# Lint the code (mirrors pre-commit hook behavior)
lint: install
	$(VENV_ACTIVATE) ruff check --fix storyforge tests
	$(VENV_ACTIVATE) ruff format storyforge tests
	$(VENV_ACTIVATE) mypy storyforge

# Lint check only (no auto-fixes)
lint-check: install
	$(VENV_ACTIVATE) ruff check storyforge tests
	$(VENV_ACTIVATE) ruff format --check storyforge tests
	$(VENV_ACTIVATE) mypy storyforge

# Type check only
typecheck: install
	$(VENV_ACTIVATE) mypy storyforge

# Clean up Python cache and test artifacts
clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -type f -delete
	rm -rf *.egg-info \
		.mypy_cache \
		.ruff_cache \
		.venv \
		build/ \
		dist \
		.pytest_cache

# Pre-release validation gate
release: lint test coverage
	@echo "All checks passed. Ready to tag and release."
