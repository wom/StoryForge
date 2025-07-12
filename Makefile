.PHONY: test run install lint clean

# Variable for venv activation
VENV_ACTIVATE = . .venv/bin/activate &&

# [venv] Run tests with pytest (requires venv)
test:
	$(VENV_ACTIVATE) PYTHONPATH=. pytest

# [venv] Run the app, activating venv if needed
run:
	$(VENV_ACTIVATE) storytime

# Install all dependencies (including dev) and the package in editable mode
install:
	uv pip install .[dev]

# [venv] Lint the code (requires venv)
lint:
	$(VENV_ACTIVATE) ruff check storytime tests

# Clean up Python cache and test artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf StoryTime.egg-info build/
