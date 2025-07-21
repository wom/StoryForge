.PHONY: all test run install lint lint-check clean

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

# Run the app
run: install
	$(VENV_ACTIVATE) storytime

# [venv] Run attaching to debug console
debug:
	$(VENV_ACTIVATE) textual run --dev `which storytime`
# [venv] Run textual debug server
console:
	$(VENV_ACTIVATE) textual console 

# Lint the code (mirrors pre-commit hook behavior)
lint: install
	$(VENV_ACTIVATE) ruff check --fix storytime tests
	$(VENV_ACTIVATE) ruff format storytime tests
	$(VENV_ACTIVATE) mypy storytime

# Lint check only (no auto-fixes)
lint-check: install
	$(VENV_ACTIVATE) ruff check storytime tests
	$(VENV_ACTIVATE) mypy storytime

# Type check only
typecheck: install
	$(VENV_ACTIVATE) mypy storytime

# Clean up Python cache and test artifacts
clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -type f -delete
	rm -rf .pytest_cache \
		.mypy_cache \
		.ruff_cache \
		StoryTime.egg-info \
		build/ \
		.venv \
		uv.lock
