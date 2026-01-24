# Changelog

All notable changes to this project will be documented in this file.

## [0.0.6] - 2026-01-24

### Added
- Story chain tracking system automatically recording parent-child relationships across extended stories for lineage tracing. (commits: 6a101c9, 4567e11)
- `export-chain` command combining multiple story parts into single readable files with chronological ordering and metadata. (commits: 6a101c9, 4567e11)
- Full story preview display in extend command showing complete story chain before continuation. (commit: 6a101c9)
- Dynamic Gemini model discovery with automatic selection of optimal text/image models from available API models. (commit: 209ef1c)
- Token limit management for Gemini backend with dynamic detection from API and automatic prompt compression at 80% threshold. (commit: 4f80ea8)
- OpenAI model configuration support via config file (`openai_story_model`, `openai_image_model`) allowing custom model selection. (commit: 64e675d)
- `GEMINI_IMAGE_MODEL` environment variable for runtime image model override. (commit: 209ef1c)
- Gemini model discovery CLI utility (`list_gemini_models.py`) for inspecting available models and capabilities. (commit: 35be24b)

### Changed
- Backend initialization now receives config object enabling model selection and configuration access. (commits: aa941a6, a5f5cc7)
- Gemini backend refactored to use cached model discovery reducing API calls and improving reliability. (commit: 209ef1c)
- Error handling improved across all backends with proper logging and user-friendly messages. (commit: 209ef1c)
- Default OpenAI models updated: story generation to `gpt-5.2`, image generation to `gpt-image-1.5`. (commit: a81bd31)

### Test
- Comprehensive PhaseExecutor phase method test suite covering all phase transitions and error conditions. (commit: 93ce28b)
- Token compression test suite validating prompt compression logic and token estimation accuracy. (commit: 4f80ea8)
- Model discovery test suite ensuring correct Gemini API interaction and fallback handling. (commit: 35be24b)
- Story chain tracking tests verifying parent tracking, lineage reconstruction, and export functionality. (commit: 4567e11)
- OpenAI backend tests updated for configurable model defaults. (commit: a81bd31)
- Extend command tests updated for new story preview functionality. (commit: 6a101c9)

### Docs
- Comprehensive story chain tracking documentation at `docs/STORY_CHAIN_TRACKING.md` with usage examples and implementation details. (commit: 4567e11)
- README updated with export-chain command documentation and story chain tracking overview. (commit: fc6ddd5)
- README updated with model configuration capabilities and environment variable reference. (this release)
- Environment variables table expanded to include `GEMINI_IMAGE_MODEL` override option. (this release)

### Chores
- Added `storyforge_output*` pattern to `.gitignore` for excluding generated output directories from version control. (commit: e786bcd)
- Removed trailing whitespace from backend docstrings for Ruff linting compliance. (commit: 4a9af25)

## [0.0.5] - 2025-10-26

### Added
- Story extension command (`storyforge extend`) allowing users to select and continue previously generated stories with wrap-up or cliffhanger endings. (commit: bbf8286)
- Dedicated `continue` subcommand for resuming previous sessions, providing same functionality as `--continue` flag. (commit: e672719)
- `config init` subcommand to replace `--init-config` flag with support for `--force` and `--path` options. (commits: 889f4d6, ff37350)
- Intelligent context summarization with extractive compression, keyword-based relevance scoring, and token budget management for efficient LLM context usage. (commits: fa8228e, 2eb7989)
- Configurable `image_count` field in images section (1-5 range) with CLI flag support. (commit: b0579ce)
- Cleanup of stale active sessions (older than 24 hours) automatically marked as failed/abandoned on startup. (commit: 04aced6)
- Parent session tracking in resumed checkpoints for better debugging and traceability. (commit: 04aced6)
- Comprehensive test suites for config module, extend command, context management, and CLI integration. (commits: b0579ce, a0162c7, ff37350)

### Changed
- Checkpoint resumption logic now starts with fresh phase state instead of pre-marking phases as completed, preventing skip logic issues. (commit: 04aced6)
- Critical initialization phases (CONFIG_LOAD, BACKEND_INIT, CONTEXT_LOAD, PROMPT_BUILD) are now always executed before resume point to ensure proper system state. (commits: 2eb7989, 04aced6)
- Application directories normalized to lowercase 'storyforge' for cross-platform consistency. (commit: 889f4d6)
- CLI shows comprehensive help with usage examples by default when no arguments are provided. (commit: 9fb1f44)
- Context loading replaced with intelligent extractive summarization that preserves semantic content within token limits. (commit: fa8228e)

### Fixed
- Backend initialization error on resume: critical phases now properly initialize before resuming to prevent `'NoneType' object has no attribute 'generate_story'` errors. (commit: 04aced6)
- Corrupted YAML checkpoint files now handled gracefully with option to move to `.corrupt` extension instead of crashing. (commit: 908fb61)
- Backend initialization errors now provide clearer messages suggesting API key verification. (commit: 04aced6)
- Checkpoint tests updated to match new phase skip logic and resume behavior. (commit: 8e940d1)

### Refactor
- Phase skip logic simplified to only check current session completion status. (commit: 04aced6)
- Separate tracking of initialized phases to support idempotent critical phase execution. (commit: 04aced6)

### Docs
- Added comprehensive configuration documentation at `docs/CONFIGURATION.md`. (commit: 889f4d6)
- README updated with new config command examples and CLI usage patterns. (commit: 889f4d6)
- CLI help text enhanced with common usage examples in epilog. (commit: 9fb1f44)

## [0.0.4] - 2025-10-01

### Added
- Session resume and phase recovery for checkpoints (ability to resume a session and continue from a specific phase). (commit: ca16ae9)
- Test coverage reporting support with pytest-cov added to the test tooling. (commit: 4492650)

### Changed
- Refinement flow and checkpoint interaction reworked: when resuming a session that already has a generated story, StoryForge now skips duplicate generation and moves directly to the refinement flow. (commit: af8d974)
- Prompt and config refactors: prompt parameter validation moved to a schema-based flow and config access was unified via schema-driven code. (commits: 51329c1, 4cb6b32, 7a8c54b)
- CLI docs and usage updated to reflect interactive prompts and current behavior. (commit: fe6985c)

### Fixed
- Checkpoint compatibility: handle incompatible/older checkpoint formats gracefully during resume. (commit: fd85d17)
- Image generation error handling and filename formatting improvements. (commit: baa882b)
- CLI behavior fixes: show help when no arguments are provided and add `-h/--help` flags. (commits: 9db3a3b, 0a6da95)
- Linting and typechecking issues fixed. (commits: 1acf4bd, 877cb95)

### Refactor
- Refactored prompt code to use a validation schema for parameters. (commit: 51329c1)
- Refactored config code to centralize and validate config values via a schema. (commit: 4cb6b32)

### Docs
- README updated to reflect the current CLI behavior and backend capabilities (notes about Anthropic text-only backend, unified CLI usage). (commit: fe6985c)
- Added a comprehensive configuration reference at `docs/CONFIGURATION.md` and a pointer in `README.md` to help users create and customize a `storyforge.ini` file.


## [0.0.3] - 2025-08-14
### Added
- OpenAI backend with DALL-E image generation support. (commit: bdbe547, 71d034b)
- Anthropic Claude backend support added. (commit: 554b948)
- Typer-based CLI application entrypoint and improved CLI wiring. (commit: 3db8c51)

### Changed
- Project metadata updated and version bumped to 0.0.3. (commit: 45790f4)
- OpenAI backend prompt handling and error reporting improved. (commit: 0380dca)

### Fixed
- Fixes to the OpenAI story backend and environment variable usage in Anthropic tests. (commits: edbbec4, 0380dca)

### Docs
- Added multi-backend setup instructions and development guide to the README. (commits: e0b73b4, 35a95b5, 34bbd10)


## [0.0.2] - 2025-09-15
### Added
- Story refinement loop allowing iterative edits and debug output for refinement. (commit: 3de1fbd)
- Multi-image generation with consistency handling and an image style option for story illustrations. (commits: c025944, bc4c5ba)
- Debug mode and improved developer-focused type/refactor changes. (commit: 6a9ca20)

### Changed
- Simplified CLI to a single story command and split image prompts for clarity. (commit: 07fd902)

### Refactor
- Renamed project/namespace from StoryTime to StoryForge across the codebase. (commits: f3a737c, 459c7ae)
- General refactors and code formatting improvements. (commits: afd4c37, e8c0c7d)


## [0.0.1] - 2025-09-01
### Added
- Initial Release.