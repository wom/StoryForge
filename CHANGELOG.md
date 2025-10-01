# Changelog

All notable changes to this project will be documented in this file.

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