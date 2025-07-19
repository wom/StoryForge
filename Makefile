.PHONY: all test run install lint clean

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

# Lint the code
lint: install
	$(VENV_ACTIVATE) ruff check storytime tests

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
