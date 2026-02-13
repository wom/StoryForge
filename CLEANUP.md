# StoryForge Pre-Release Cleanup Checklist

## HIGH — Fix Before Public Release

- [x] **Gemini backend: no filename sanitization on LLM-generated image names**
  - `storyforge/gemini_backend.py` L283 does `name.split(".")[0]` but doesn't strip slashes or special chars. An LLM returning `"../../etc/passwd"` would write outside the output dir. The Anthropic and OpenAI backends properly strip to alphanumeric — Gemini should too.

- [x] **`.vscode/` directory is tracked in git**
  - `.vscode/settings.json` and `.vscode/launch.json` are committed. The launch.json references a non-existent path (`storytime/StoryCLI.py` — old project name). Add `.vscode/` to `.gitignore` and `git rm --cached -r .vscode/`.

- [x] **`htmlcov/` not in `.gitignore`**
  - `coverage.xml` is gitignored but `htmlcov/` is not. Add it.

- [x] **Makefile has duplicate `coverage` target**
  - `coverage:` is defined at line 1 and again at line 28. Remove the duplicate at the top.

## MEDIUM — Should Fix

- [x] **Outdated docstrings claim fewer backends than exist**
  - `storyforge/StoryForge.py` L3: says "Supports Google Gemini and Anthropic Claude" — missing OpenAI.
  - `storyforge/llm_backend.py` L10-11: says "Future: OpenAI, Anthropic, etc." — both are implemented.

- [x] **Legacy environment variable name**
  - `storyforge/context.py` L92 uses `STORYTIME_TEST_CONTEXT_DIR` — references old project name "Storytime" instead of "StoryForge".

- [x] **`[DEBUG]` prints show up in non-debug verbose mode**
  - `storyforge/phase_executor.py` L615-619 and L716-722: outputs labeled `[DEBUG]` are gated on `verbose`, not `debug`. Users running with `--verbose` see confusing `[DEBUG]` labels.

- [x] **TODO comments left in shipped code**
  - `storyforge/openai_backend.py` L6-9: 4 TODO comments at the top of the file.
  - `storyforge/phase_executor.py` L550: `# TODO: Track which files were used`.

- [x] **`assert` used for runtime validation (stripped by `-O`)**
  - `storyforge/phase_executor.py` uses `assert self.checkpoint_data is not None` at ~12 locations. These are removed when Python runs optimized. Replace with `if not ... raise RuntimeError(...)`.

- [x] **Document how to generate API keys  - in --help and/or README, whatever makes the most sense.**
  - for gemini - https://aistudio.google.com/app/api-keys
  - also for openai and anthropi

- [x] **Silent error swallowing everywhere**
  - Many `except Exception: pass` or `except Exception: return fallback` with no logging across all backends, `model_discovery.py` L47, `config.py` L188-192, `checkpoint.py` L283-289. Users will have no idea why things fail.

- [x] **Dead code — unused functions tracked in git**
  - `_get_completed_phases_before()` and `_clear_phases_from()` in `phase_executor.py`
  - `_resolve_context_path()` in `context.py` L435 (also hardcodes `"family.md"`)
  - `list_available_backends()` in `llm_backend.py` L213
  - `generate_typer_options()`, `get_field_choices()`, `get_field_help()`, `generate_help_text()` in `schema/cli_integration.py`
  - Dead `TYPE_CHECKING` block in `openai_backend.py` L48-51

- [x] **`list_gemini_models.py` in repo root**
  - Standalone utility/debug script in the project root — doesn't belong in a public release.

## LOW — Nice to Fix

- [x] **`requires-python = ">=3.8"` but `target-version = "py312"`**
  - `pyproject.toml` L10 claims Python 3.8+ support, but ruff targets 3.12 and the code uses `str | None` union syntax (requires 3.10+). Make claims consistent.

- [x] **Duplicate context-discovery logic**
  - `storyforge/context.py` L80-112 and L342-366 have identical file-finding logic copy-pasted. Extract to a helper.

- [x] **Identical image prompts in loop**
  - `Prompt.image()` takes `num_images` and loops but produces identical prompt text each iteration (loop variable `_i` is unused). All images get the same prompt.

- [x] **Recursive refinement could hit stack limit**
  - `storyforge/phase_executor.py` L739: `_handle_story_refinement()` calls itself recursively. Should be a `while` loop.

- [x] **Inconsistent `print()` vs `console.print()`**
  - `storyforge/StoryForge.py` L106 uses bare `print()` while everything else uses Rich `console.print()`.

- [x] **Duplicate fallback image-prompt code across backends**
  - `_generate_fallback_image_prompts` is copy-pasted identically in `anthropic_backend.py`, `openai_backend.py`, and `gemini_backend.py`. Should be in the base class `LLMBackend`.

- [x] **Inconsistent error message format across backends**
  - Gemini returns `"[Error generating story]"` (hides exception), Anthropic and OpenAI return `f"[Error generating story: {str(e)}]"` (includes it). Make consistent.

- [x] **Multiple `Console()` instances created independently**
  - `checkpoint.py`, `StoryForge.py`, `config.py`, `phase_executor.py` each create their own `console = Console()`. Consider sharing one instance.
