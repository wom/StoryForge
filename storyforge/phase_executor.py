"""
Phase-based execution engine for StoryForge with checkpoint support.

This module provides a structured way to execute StoryForge phases with
automatic checkpointing and recovery capabilities.
"""

import logging
import os
from collections.abc import Callable
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Any
from uuid import uuid4

from rich.progress import Progress, SpinnerColumn, TextColumn

from .checkpoint import CheckpointData, CheckpointManager, ExecutionPhase
from .config import Config, load_config
from .console import console
from .context import ContextManager
from .llm_backend import ERROR_STORY_SENTINEL, classify_story_error, get_backend
from .paths import create_output_directory_name
from .portable_text import to_portable_ascii
from .prompt import Prompt


def _load_debug_story() -> str:
    """Load the bundled deterministic story used by debug workflows."""
    return files("storyforge").joinpath("test_story.txt").read_text(encoding="utf-8").strip()


class PhaseExecutor:
    """Phase-based execution engine with checkpoint support."""

    MAX_FILENAME_PREFIX_LENGTH: int = 30

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        reporter: Callable[[str, str, float | None], None] | None = None,
    ) -> None:
        """Initialize the phase executor."""
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_data: CheckpointData | None = None
        self.config: Config | None = None
        self.llm_backend: Any = None  # LLMBackend type not available in this scope
        self.context: str | None = None
        self.world: str | None = None
        self.story_prompt: Any = None  # Prompt type not available in this scope
        self.story: str | None = None
        self.refinements: str | None = None
        self._initialized_phases: set[ExecutionPhase] = set()  # Track which phases have been initialized
        self.reporter = reporter
        # An in-memory override for replacement media. The canonical output
        # directory in the checkpoint must never point at temporary staging.
        self.media_output_directory: Path | None = None

    def _report(self, kind: str, message: str, progress: float | None = None) -> None:
        """Publish a transport-neutral workflow event when a reporter is configured."""
        if self.reporter is not None:
            self.reporter(kind, message, progress)

    def execute_new_session(
        self,
        prompt: str,
        cli_arguments: dict[str, Any],
        resolved_config: dict[str, Any],
        prompt_obj: Prompt | None = None,
        stop_after: ExecutionPhase | None = None,
    ) -> CheckpointData:
        """
        Execute a new StoryForge session with checkpointing.

        Args:
            prompt: The story prompt string
            cli_arguments: CLI arguments dictionary
            resolved_config: Resolved configuration dictionary
            prompt_obj: Optional pre-built Prompt object (for extend command)
        """
        # Validate inputs
        if not prompt or not prompt.strip():
            if not prompt_obj or not prompt_obj.continuation_mode:
                raise ValueError("Story prompt cannot be empty")

        # Create new checkpoint data
        self.checkpoint_data = CheckpointData.create_new(prompt, cli_arguments, resolved_config)

        # Store the pre-built prompt object if provided
        if prompt_obj:
            self.story_prompt = prompt_obj

        console.print(f"[bold cyan]Starting new StoryForge session:[/bold cyan] {self.checkpoint_data.session_id}")

        try:
            # Save initial checkpoint
            self.checkpoint_manager.save_checkpoint(self.checkpoint_data)

            # Start execution from the beginning
            if stop_after is None:
                self._execute_phase_sequence(ExecutionPhase.INIT)
            else:
                self._execute_phase_sequence(ExecutionPhase.INIT, stop_after=stop_after)

            if stop_after is not None:
                self.checkpoint_manager.save_checkpoint(self.checkpoint_data)
                return self.checkpoint_data

            # Mark session as completed
            self.checkpoint_data.mark_completed()
            self.checkpoint_manager.save_checkpoint(self.checkpoint_data)
            console.print("[bold green]✅ StoryForge session completed successfully![/bold green]")
            return self.checkpoint_data

        except KeyboardInterrupt:
            # User interrupted - save current state
            if self.checkpoint_data:
                console.print("\n[yellow]Session interrupted by user. Progress saved.[/yellow]")
                self.checkpoint_manager.save_checkpoint(self.checkpoint_data)
            raise
        except Exception as e:
            # Mark session as failed and save checkpoint
            # Don't re-wrap the error — _execute_phase() already adds phase context
            error_msg = str(e)
            console.print(f"[red]Session failed:[/red] {error_msg}")

            if self.checkpoint_data:
                self.checkpoint_data.mark_failed(error_msg)
                try:
                    self.checkpoint_manager.save_checkpoint(self.checkpoint_data)
                    console.print(f"[dim]Failed session saved as:[/dim] {self.checkpoint_data.session_id}")
                except Exception as save_error:
                    console.print(f"[red]Could not save failed session:[/red] {save_error}")
            raise

    def execute_existing_session(
        self,
        checkpoint_data: CheckpointData,
        start_phase: ExecutionPhase,
        stop_after: ExecutionPhase | None = None,
    ) -> CheckpointData:
        """Continue an existing checkpoint without creating a resumed-session copy."""
        self.checkpoint_data = checkpoint_data
        self.story = str(checkpoint_data.generated_content.get("story") or "")
        self.refinements = checkpoint_data.generated_content.get("refinements")
        self._execute_phase_sequence(start_phase, stop_after=stop_after)
        if stop_after is not None:
            self.checkpoint_manager.save_checkpoint(checkpoint_data)
            return checkpoint_data
        checkpoint_data.mark_completed()
        self.checkpoint_manager.save_checkpoint(checkpoint_data)
        return checkpoint_data

    def refine_existing_story(
        self,
        checkpoint_data: CheckpointData,
        instructions: str,
    ) -> CheckpointData:
        """Refine a staged draft without invoking terminal interaction."""
        if not instructions.strip():
            raise ValueError("Refinement instructions cannot be empty")

        self.checkpoint_data = checkpoint_data
        self.story = str(checkpoint_data.generated_content.get("story") or "")
        if not self.story:
            raise ValueError("Checkpoint does not contain a draft story")

        for phase in (
            ExecutionPhase.CONFIG_LOAD,
            ExecutionPhase.BACKEND_INIT,
            ExecutionPhase.CONTEXT_LOAD,
            ExecutionPhase.PROMPT_BUILD,
        ):
            self._execute_phase(phase)

        self.refinements = instructions.strip()
        if self.story_prompt is None:
            raise RuntimeError("Story prompt could not be rebuilt")
        self.story_prompt.refinement_mode = True
        self.story_prompt.original_story = self.story
        self.story_prompt.refinement_instructions = self.refinements
        self._report("phase", "story_refine", None)
        backend = self._require_backend("Story refinement")
        revised_story = backend.generate_story(self.story_prompt)
        if revised_story is None or revised_story.startswith(ERROR_STORY_SENTINEL):
            error_msg, _ = classify_story_error(revised_story or ERROR_STORY_SENTINEL)
            raise RuntimeError(error_msg)

        self.story = revised_story
        checkpoint_data.generated_content["story"] = revised_story
        checkpoint_data.generated_content["refinements"] = self.refinements
        checkpoint_data.user_decisions["story_accepted"] = None
        # STORY_SAVE ran before the review boundary. Keep the canonical artifact
        # synchronized with the draft that the user is now reviewing.
        self._phase_story_save()
        self.checkpoint_manager.save_checkpoint(checkpoint_data)
        self._report("phase", "story_refine_complete", 1.0)
        return checkpoint_data

    def _execute_phase_sequence(
        self,
        start_phase: ExecutionPhase,
        stop_after: ExecutionPhase | None = None,
    ) -> None:
        """Execute the phase sequence starting from the specified phase."""
        # Define the phase execution order
        phase_order = [
            ExecutionPhase.INIT,
            ExecutionPhase.CONFIG_LOAD,
            ExecutionPhase.BACKEND_INIT,
            ExecutionPhase.PROMPT_CONFIRM,
            ExecutionPhase.CONTEXT_LOAD,
            ExecutionPhase.PROMPT_BUILD,
            ExecutionPhase.STORY_GENERATE,
            ExecutionPhase.STORY_SAVE,
            ExecutionPhase.VIDEO_DECISION,
            ExecutionPhase.IMAGE_DECISION,
            ExecutionPhase.VIDEO_PROMPT_GENERATE,
            ExecutionPhase.IMAGE_GENERATE,
            ExecutionPhase.CONTEXT_SAVE,
        ]

        # Always execute critical initialization phases BEFORE the start phase
        # These are idempotent and required for execution environment
        critical_init_phases = [
            ExecutionPhase.CONFIG_LOAD,
            ExecutionPhase.BACKEND_INIT,
            ExecutionPhase.CONTEXT_LOAD,
            ExecutionPhase.PROMPT_BUILD,
        ]

        start_index = phase_order.index(start_phase)

        # Execute critical init phases that come BEFORE start_phase
        for phase in critical_init_phases:
            phase_index = phase_order.index(phase)

            # Only execute if this phase is before our start phase and hasn't been initialized yet
            if phase_index < start_index and phase not in self._initialized_phases:
                console.print(f"[dim]Initializing required phase:[/dim] {phase.value}")
                self._execute_phase(phase)
                self._initialized_phases.add(phase)
                # Don't add to checkpoint.completed_phases - these are initialization only

        # Execute phases in sequence from start_phase
        phases = phase_order[start_index:]
        if stop_after is not None:
            phases = phases[: phases.index(stop_after) + 1]

        for phase_index, phase in enumerate(phases):
            if self._should_skip_phase(phase):
                continue

            # Update current_phase BEFORE executing so error messages reference the correct phase
            if self.checkpoint_data is not None:
                self.checkpoint_data.current_phase = phase.value

            console.print(f"[dim]Executing phase:[/dim] {phase.value}")
            self._report("phase", phase.value, phase_index / max(len(phases), 1))
            self._execute_phase(phase)

            # Mark phase completed and save checkpoint after success
            if self.checkpoint_data is not None:
                self.checkpoint_data.update_phase(phase)
                self.checkpoint_manager.save_checkpoint(self.checkpoint_data)
        self._report("phase", "complete", 1.0)

    def _should_skip_phase(self, phase: ExecutionPhase) -> bool:
        """Determine if a phase should be skipped based on checkpoint state."""
        if not self.checkpoint_data:
            return False

        # Simplified logic - only skip if completed in THIS session
        # Critical phases are handled by _execute_phase_sequence initialization
        if phase.value in self.checkpoint_data.completed_phases:
            console.print(f"[dim]Skipping completed phase:[/dim] {phase.value}")
            return True

        return False

    def _execute_phase(self, phase: ExecutionPhase) -> None:
        """Execute a specific phase with error handling."""
        try:
            verbose = self.checkpoint_data and self.checkpoint_data.resolved_config.get("verbose", False)

            if verbose:
                console.print(f"[dim]Starting phase: {phase.value}[/dim]")

            if phase == ExecutionPhase.INIT:
                self._phase_init()
            elif phase == ExecutionPhase.CONFIG_LOAD:
                self._phase_config_load()
            elif phase == ExecutionPhase.BACKEND_INIT:
                self._phase_backend_init()
            elif phase == ExecutionPhase.PROMPT_CONFIRM:
                self._phase_prompt_confirm()
            elif phase == ExecutionPhase.CONTEXT_LOAD:
                self._phase_context_load()
            elif phase == ExecutionPhase.PROMPT_BUILD:
                self._phase_build_prompt()
            elif phase == ExecutionPhase.STORY_GENERATE:
                self._phase_story_generate()
            elif phase == ExecutionPhase.STORY_SAVE:
                self._phase_story_save()
            elif phase == ExecutionPhase.VIDEO_DECISION:
                self._phase_video_decision()
            elif phase == ExecutionPhase.VIDEO_PROMPT_GENERATE:
                self._phase_video_prompt_generate()
            elif phase == ExecutionPhase.IMAGE_DECISION:
                self._phase_image_decision()
            elif phase == ExecutionPhase.IMAGE_GENERATE:
                self._phase_image_generate()
            elif phase == ExecutionPhase.CONTEXT_SAVE:
                self._phase_context_save()
            else:
                raise ValueError(f"Unknown execution phase: {phase}")

            if verbose:
                console.print(f"[dim]Completed phase: {phase.value}[/dim]")

        except KeyboardInterrupt:
            # User interrupted - propagate up
            raise
        except Exception as e:
            # Add context to error message
            phase_error = f"Failed during {phase.value} phase: {str(e)}"
            console.print(f"[red]Phase Error:[/red] {phase_error}")

            # Log verbose error details if enabled
            if self.checkpoint_data and self.checkpoint_data.resolved_config.get("verbose", False):
                import traceback

                console.print(f"[dim]Traceback:[/dim] {traceback.format_exc()}")

            raise RuntimeError(phase_error) from e

    def _phase_init(self) -> None:
        """Initialize phase - validate CLI arguments and setup."""

    def _phase_config_load(self) -> None:
        """Load configuration phase."""
        if self.checkpoint_data is None:
            raise RuntimeError("Checkpoint data must be initialized")
        verbose = self.checkpoint_data.resolved_config.get("verbose", False)
        self.config = load_config(verbose=verbose)
        if self.config.config_path is not None:
            self.checkpoint_data.resolved_config["config_path"] = str(self.config.config_path)

    def _phase_backend_init(self) -> None:
        """Initialize LLM backend phase."""
        if self.checkpoint_data is None:
            raise RuntimeError("Checkpoint data must be initialized")
        if self.checkpoint_data.resolved_config.get("debug", False):
            if self.checkpoint_data.resolved_config.get("verbose", False):
                console.print("[dim]Offline debug mode: deferring AI backend initialization.[/dim]")
            self.llm_backend = None
            return

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Initialize the configured backend, including for a lazily requested feature."""
        if self.checkpoint_data is None:
            raise RuntimeError("Checkpoint data must be initialized")
        backend_name = self.checkpoint_data.resolved_config.get("backend")
        config_backend = self.checkpoint_data.resolved_config.get("config_backend")
        verbose = self.checkpoint_data.resolved_config.get("verbose", False)

        if verbose:
            console.print("[dim]Initializing AI backend...[/dim]")

        # Better error handling for backend initialization
        try:
            self.llm_backend = get_backend(
                backend_name=backend_name,
                config_backend=config_backend,
                config=self.config,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize {backend_name or config_backend or 'default'} backend. "
                f"Please check that your API key is set correctly in environment variables. "
                f"Error: {e}"
            ) from e

        if not self.llm_backend:
            backend_display = backend_name or config_backend or "auto-detected backend"
            raise RuntimeError(
                f"Backend initialization returned None for '{backend_display}'. "
                f"Please verify your API key is set and valid."
            )

        if verbose and self.llm_backend:
            console.print(f"[dim]Using {self.llm_backend.name} backend[/dim]")

    def _require_backend(self, feature: str) -> Any:
        """Return an initialized backend for an optional provider-dependent feature."""
        if self.llm_backend is None:
            try:
                self._initialize_backend()
            except RuntimeError as error:
                raise RuntimeError(
                    f"{feature} requires a configured AI provider and valid API key. "
                    "The debug story itself remains available offline."
                ) from error
        return self.llm_backend

    def _phase_prompt_confirm(self) -> None:
        """Validate that confirmation was handled by the calling client."""
        if self.checkpoint_data is None:
            raise RuntimeError("Checkpoint data must be initialized")
        if not self.checkpoint_data.resolved_config.get("auto_confirm"):
            raise RuntimeError("Prompt confirmation must be handled by the MCP client")

    def _phase_context_load(self) -> None:
        """Load context files and world definition phase.

        Uses extractive summarization to compress context to fit within
        the backend's token budget (50% of model context window).
        Falls back to raw concatenation if no prompt is available.
        World file (world.md) is always loaded verbatim, bypassing token budgets.
        """
        if self.checkpoint_data is None:
            raise RuntimeError("Checkpoint data must be initialized")
        use_context = self.checkpoint_data.resolved_config.get("use_context", True)
        verbose = self.checkpoint_data.resolved_config.get("verbose", False)
        world_file = self.checkpoint_data.resolved_config.get("world_file") or None

        # Determine token budget from backend (if available)
        max_tokens: int | None = None
        if self.llm_backend is not None:
            max_tokens = self.llm_backend.get_context_token_budget()
            if verbose:
                console.print(
                    f"[dim]Context token budget: {max_tokens} tokens "
                    f"({int(self.llm_backend.CONTEXT_BUDGET_RATIO * 100)}% of "
                    f"{self.llm_backend.text_input_limit} model limit)[/dim]"
                )

        context_manager = ContextManager(max_tokens=max_tokens, world_file_path=world_file)

        # Load world file (always verbatim, no budget limit)
        self.world = context_manager.load_world()
        if self.world:
            word_count = len(self.world.split())
            estimated_tokens = word_count * 4 // 3  # rough word-to-token ratio
            if verbose:
                console.print(f"[dim]Loaded world file: {word_count} words (verbatim)[/dim]")
            if estimated_tokens > 5000:
                console.print(
                    f"[yellow]⚠ World file is large (~{estimated_tokens} tokens). "
                    f"This is included verbatim in every prompt and may consume "
                    f"significant context window. Consider trimming it.[/yellow]"
                )
            self.checkpoint_data.context_data = {
                **(self.checkpoint_data.context_data or {}),
                "world_content": self.world,
            }

        # Story context is optional; the world definition is not. Extensions
        # already carry their story chain in the pre-built prompt.
        if self.story_prompt is not None:
            self.story_prompt.world = self.world
            self.context = None  # The chain is already in the pre-built story prompt.
            if verbose:
                console.print("[dim]Using pre-built prompt context[/dim]")
            return
        if not use_context:
            self.context = None
            if verbose:
                console.print("[dim]Story context loading skipped due to --no-use-context[/dim]")
            return

        # Use extractive summarization when we have a prompt for relevance scoring
        prompt_text = self.checkpoint_data.original_inputs.get("prompt", "")
        if prompt_text:
            self.context = context_manager.extract_relevant_context(prompt=str(prompt_text))
            if verbose and self.context:
                raw_context = context_manager.load_context()
                raw_words = len(raw_context.split()) if raw_context else 0
                summarized_words = len(self.context.split())
                console.print(
                    f"[dim]Context summarized: {raw_words} words → {summarized_words} words "
                    f"({int(summarized_words / raw_words * 100) if raw_words else 0}% of original)[/dim]"
                )
        else:
            self.context = context_manager.load_context()
            if verbose and self.context:
                word_count = len(self.context.split())
                console.print(f"[dim]Loaded raw context: {word_count} words (no prompt for scoring)[/dim]")

        if verbose and not self.context and not self.world:
            console.print("[dim]No context or world files found[/dim]")

        # Store context in checkpoint
        if self.checkpoint_data:
            context_data: dict[str, Any] = {}
            if self.context:
                context_data.update(
                    {
                        "loaded_context": self.context,
                        "context_files_used": [],
                        "summarized": bool(prompt_text),
                        "max_tokens": max_tokens,
                        "has_old_context": context_manager.has_old_context,
                    }
                )
            if self.world:
                context_data["world_content"] = self.world
            if context_data:
                self.checkpoint_data.context_data = context_data

    def _phase_build_prompt(self) -> None:
        """Build the story prompt from inputs."""
        if self.checkpoint_data is None:
            raise RuntimeError("Checkpoint data must be initialized")

        # If prompt is already built (e.g., from extend command), skip this phase
        if self.story_prompt is not None:
            if self.checkpoint_data.resolved_config.get("verbose"):
                console.print("[dim]Using pre-built prompt object[/dim]")
            return

        original_inputs = self.checkpoint_data.original_inputs
        resolved_config = self.checkpoint_data.resolved_config
        cli_args = original_inputs.get("cli_arguments", {})

        prompt = str(original_inputs.get("prompt", ""))

        # Get continuation mode parameters if present
        continuation_mode = cli_args.get("continuation_mode", False)
        ending_type = cli_args.get("ending_type", "wrap_up")
        continuation_direction = cli_args.get("continuation_direction") or resolved_config.get(
            "continuation_direction"
        )
        prompt_context = resolved_config.get("continuation_context") or self.context

        self.story_prompt = Prompt(
            prompt=prompt,
            context=prompt_context,
            world=self.world,
            length=str(cli_args.get("length") or resolved_config.get("length") or ""),
            age_range=str(cli_args.get("age_range") or resolved_config.get("age_range") or ""),
            style=str(cli_args.get("style") or resolved_config.get("style") or ""),
            tone=str(cli_args.get("tone") or resolved_config.get("tone") or ""),
            voice=cli_args.get("voice") or resolved_config.get("voice") or None,
            theme=cli_args.get("theme") or resolved_config.get("theme"),
            setting=cli_args.get("setting"),
            characters=cli_args.get("characters"),
            learning_focus=cli_args.get("learning_focus"),
            image_style=str(cli_args.get("image_style") or resolved_config.get("image_style") or ""),
            continuation_mode=continuation_mode,
            ending_type=ending_type,
            continuation_direction=continuation_direction,
            has_old_context=bool(
                self.checkpoint_data.context_data.get("has_old_context")
                if self.checkpoint_data.context_data
                else False
            ),
        )

    def _phase_story_generate(self) -> None:
        """Story generation and refinement phase."""
        if self.checkpoint_data is None:
            raise RuntimeError("Checkpoint data must be initialized")
        debug = self.checkpoint_data.resolved_config.get("debug", False)
        verbose = self.checkpoint_data.resolved_config.get("verbose", False)

        # Check if we're resuming and already have a story
        existing_story = self.checkpoint_data.generated_content.get("story")

        # If we're regenerating from checkpoint (story exists but we're back at this phase)
        # then we need to apply refinements
        if existing_story and ExecutionPhase.STORY_GENERATE.value in self.checkpoint_data.completed_phases:
            self.story = str(existing_story)
            console.print("[cyan]Using existing story from checkpoint[/cyan]")
            # Don't generate, just move to refinement
            self._handle_story_refinement()
            return

        # Generate new story (first time)
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Generating story..."),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("story", total=None)

            if debug:
                self.story = _load_debug_story()
                console.print("[dim]Loaded debug story from test file.[/dim]")
            else:
                self.story = self.llm_backend.generate_story(self.story_prompt)
                if verbose:
                    console.print("[dim]Story generation complete.[/dim]")

        if self.story is None or self.story.startswith(ERROR_STORY_SENTINEL):
            error_msg, _ = classify_story_error(self.story or ERROR_STORY_SENTINEL)
            raise RuntimeError(error_msg)

        # Store story in checkpoint
        self.checkpoint_data.generated_content["story"] = self.story

        # Handle story refinement
        self._handle_story_refinement()

    def _handle_story_refinement(self) -> None:
        """Ensure story review is delegated to an MCP client."""
        if self.checkpoint_data is None:
            raise RuntimeError("Checkpoint data must be initialized")
        if not self.checkpoint_data.resolved_config.get("defer_story_review"):
            raise RuntimeError("Story review must be handled by the MCP client")

    def _phase_story_save(self) -> None:
        """Save story to file phase."""
        if self.checkpoint_data is None:
            raise RuntimeError("Checkpoint data must be initialized")
        output_dir = self.checkpoint_data.resolved_config.get("output_directory")
        if not output_dir:
            continuation_mode = self.checkpoint_data.resolved_config.get("continuation_mode", False)
            output_dir = create_output_directory_name(extended=continuation_mode)
            self.checkpoint_data.resolved_config["output_directory"] = output_dir

        story_filename = "story.txt"
        story_path = os.path.join(output_dir, story_filename)
        os.makedirs(output_dir, exist_ok=True)

        prompt_text = str(self.checkpoint_data.original_inputs.get("prompt", ""))
        portable_story = to_portable_ascii(f"Story: {prompt_text}\n\n{self.story or ''}")
        with open(story_path, "w", encoding="ascii", newline="\n") as f:
            f.write(portable_story)

        console.print(f"[bold green]✅ Story saved as:[/bold green] {story_path}")

        # Always write generation metadata to output directory
        self._write_generation_metadata(output_dir)

        # Dump all context and session info when --verbose or --debug is set
        verbose = self.checkpoint_data.resolved_config.get("verbose", False)
        debug = self.checkpoint_data.resolved_config.get("debug", False)
        if verbose or debug:
            self._dump_session_context(output_dir)

    def _get_parameter_source(self, field_name: str) -> tuple[str, str | None]:
        """Determine the source and value of a generation parameter.

        Checks in priority order: random resolution, CLI argument, config file,
        schema default.

        Args:
            field_name: The parameter field name (e.g., 'voice', 'style').

        Returns:
            Tuple of (source_label, value) where source_label is one of
            'Random', 'CLI', 'Config', 'Default'. Value may be None.
        """
        if self.checkpoint_data is None:
            return ("Default", None)

        cli_args = self.checkpoint_data.original_inputs.get("cli_arguments", {}) or {}
        resolved_config = self.checkpoint_data.resolved_config or {}

        # Check if this field was randomly resolved
        if self.story_prompt and hasattr(self.story_prompt, "random_resolved"):
            random_resolved = self.story_prompt.random_resolved
            if field_name in random_resolved:
                return ("Random", random_resolved[field_name])

        # Check CLI arguments
        cli_val = cli_args.get(field_name)
        if cli_val not in (None, ""):
            return ("CLI", str(cli_val))

        # Check config file value
        cfg_val = resolved_config.get(field_name)
        if cfg_val not in (None, ""):
            return ("Config", str(cfg_val))

        # Fall back to default (get from prompt object if available)
        if self.story_prompt:
            val = getattr(self.story_prompt, field_name, None)
            if val not in (None, ""):
                return ("Default", str(val))

        return ("Default", None)

    def _build_generation_metadata(self) -> str:
        """Build generation metadata content as markdown.

        Returns:
            Markdown string with backend/model info and parameter sources.
        """
        if self.checkpoint_data is None:
            return ""

        sections: list[str] = []
        sections.append("# Generation Metadata\n\n")
        sections.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Backend and model info
        if self.llm_backend:
            sections.append(f"**Backend:** {self.llm_backend.name}\n")
            model_info = self.llm_backend.get_model_info()
            if model_info.get("story_model"):
                sections.append(f"**Story Model:** {model_info['story_model']}\n")
            if model_info.get("image_model"):
                sections.append(f"**Image Model:** {model_info['image_model']}\n")
            sections.append("\n")

        # Parameters with source tracking
        sections.append("## Parameters\n\n")
        sections.append("| Parameter | Value | Source |\n")
        sections.append("|-----------|-------|--------|\n")

        param_fields = [
            ("length", "Length"),
            ("age_range", "Age Range"),
            ("style", "Style"),
            ("tone", "Tone"),
            ("voice", "Voice"),
            ("theme", "Theme"),
            ("setting", "Setting"),
            ("learning_focus", "Learning Focus"),
            ("image_style", "Image Style"),
            ("ending_type", "Ending Type"),
        ]

        for field_name, display_name in param_fields:
            source, value = self._get_parameter_source(field_name)
            if value is not None:
                if source == "Random":
                    sections.append(f"| {display_name} | {value} | Random ({value}) |\n")
                else:
                    sections.append(f"| {display_name} | {value} | {source} |\n")

        # Characters (special handling - list type)
        cli_args = self.checkpoint_data.original_inputs.get("cli_arguments", {}) or {}
        if cli_args.get("characters"):
            sections.append(f"| Characters | {', '.join(cli_args['characters'])} | CLI |\n")

        sections.append("\n")
        return "".join(sections)

    def _write_generation_metadata(self, output_dir: str) -> None:
        """Write generation_metadata.md to the story output directory.

        Args:
            output_dir: The story output directory path.
        """
        if self.checkpoint_data is None:
            return

        metadata_path = os.path.join(output_dir, "generation_metadata.md")
        content = self._build_generation_metadata()

        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write(content)
            console.print(f"[dim]📋 Generation metadata saved: {metadata_path}[/dim]")
        except Exception as e:
            console.print(f"[dim]Warning: Could not save generation metadata: {e}[/dim]")

    def _dump_session_context(self, output_dir: str) -> None:
        """Dump all context, prompt details, and config to the output directory.

        Creates a human-readable context_dump.md file containing everything
        that went into generating the story. Triggered by --verbose or --debug.

        Args:
            output_dir: The story output directory path.
        """
        if self.checkpoint_data is None:
            return

        dump_path = os.path.join(output_dir, "context_dump.md")
        resolved_config = self.checkpoint_data.resolved_config
        original_inputs = self.checkpoint_data.original_inputs
        cli_args = original_inputs.get("cli_arguments", {})

        sections: list[str] = []

        # Header
        sections.append("# StoryForge Session Context Dump\n\n")
        sections.append(f"**Session ID:** {self.checkpoint_data.session_id}\n")
        sections.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if self.llm_backend:
            sections.append(f"**Backend:** {self.llm_backend.name}\n")
        sections.append("\n")

        # Original Inputs
        sections.append("## Original Inputs\n\n")
        sections.append(f"**Prompt:** {original_inputs.get('prompt', 'N/A')}\n\n")

        # Resolved Configuration
        sections.append("## Resolved Configuration\n\n")
        sections.append("| Parameter | Value | Source |\n")
        sections.append("|-----------|-------|--------|\n")
        config_keys = [
            "backend",
            "length",
            "age_range",
            "style",
            "tone",
            "theme",
            "setting",
            "learning_focus",
            "image_style",
            "use_context",
            "continuation_mode",
            "ending_type",
            "debug",
            "verbose",
        ]
        for key in config_keys:
            cli_val = cli_args.get(key) if isinstance(cli_args, dict) else None
            cfg_val = resolved_config.get(key)
            value = cli_val if cli_val is not None else cfg_val
            if value is not None:
                source = "cli" if cli_val is not None else "config"
                sections.append(f"| {key} | {value} | {source} |\n")

        characters = cli_args.get("characters") if isinstance(cli_args, dict) else None
        if characters:
            sections.append(f"| characters | {', '.join(characters)} | cli |\n")
        sections.append("\n")

        # Prompt Object Details (after random resolution)
        if self.story_prompt:
            prompt = self.story_prompt
            sections.append("## Prompt Object (after random resolution)\n\n")
            sections.append(f"- **prompt:** {prompt.prompt}\n")
            sections.append(f"- **length:** {prompt.length}\n")
            sections.append(f"- **age_range:** {prompt.age_range}\n")
            sections.append(f"- **style:** {prompt.style}\n")
            sections.append(f"- **tone:** {prompt.tone}\n")
            sections.append(f"- **theme:** {prompt.theme}\n")
            sections.append(f"- **image_style:** {prompt.image_style}\n")
            if prompt.setting:
                sections.append(f"- **setting:** {prompt.setting}\n")
            if prompt.characters:
                sections.append(f"- **characters:** {', '.join(prompt.characters)}\n")
            if prompt.learning_focus:
                sections.append(f"- **learning_focus:** {prompt.learning_focus}\n")
            sections.append(f"- **continuation_mode:** {prompt.continuation_mode}\n")
            if prompt.continuation_mode:
                sections.append(f"- **ending_type:** {prompt.ending_type}\n")
            sections.append("\n")

        # Full Story Prompt Sent to LLM
        if self.story_prompt:
            sections.append("## Full Story Prompt (sent to LLM)\n\n")
            sections.append("```text\n")
            sections.append(self.story_prompt.story)
            sections.append("\n```\n\n")

        # Context Used
        sections.append("## Context\n\n")
        if self.context:
            word_count = len(self.context.split())
            sections.append(f"**Length:** {len(self.context)} chars, ~{word_count} words\n\n")
            sections.append("### Full Context\n\n")
            sections.append(self.context)
            sections.append("\n\n")
        else:
            sections.append("*No context was loaded for this session.*\n\n")

        # Config file details
        if self.config:
            sections.append("## Config File\n\n")
            config_path = resolved_config.get("config_path")
            if config_path:
                sections.append(f"**Loaded from:** `{config_path}`\n\n")
            try:
                config_dict = self.config.to_dict()
                for section_name, section_values in config_dict.items():
                    sections.append(f"### [{section_name}]\n\n")
                    if isinstance(section_values, dict):
                        for k, v in section_values.items():
                            sections.append(f"- {k} = {v}\n")
                    sections.append("\n")
            except Exception:
                sections.append("*Could not serialize config.*\n\n")

        try:
            with open(dump_path, "w", encoding="utf-8") as f:
                f.write("".join(sections))
            console.print(f"[dim]📋 Context dump saved: {dump_path}[/dim]")
        except Exception as e:
            console.print(f"[dim]Warning: Could not save context dump: {e}[/dim]")

    def _phase_video_decision(self) -> None:
        """Video prompt generation decision phase."""
        if self.checkpoint_data is None:
            raise RuntimeError("Checkpoint data must be initialized")
        # Check if decision already made
        if self.checkpoint_data.user_decisions.get("wants_video_prompt") is not None:
            return

        configured_scenes = self.checkpoint_data.resolved_config.get("video_scene_count")
        if configured_scenes is None:
            raise RuntimeError("Video decision must be supplied by the MCP client")
        wants_video = int(configured_scenes) > 0
        self.checkpoint_data.user_decisions["wants_video_prompt"] = wants_video
        self.checkpoint_data.user_decisions["num_video_scenes"] = int(configured_scenes)

    def _phase_video_prompt_generate(self) -> None:
        """Video prompt generation phase."""
        if self.checkpoint_data is None:
            raise RuntimeError("Checkpoint data must be initialized")
        wants_video = self.checkpoint_data.user_decisions.get("wants_video_prompt")
        if wants_video is None:
            raise RuntimeError("Video prompt decision is missing; resume from the video decision phase.")
        if not wants_video:
            console.print("[yellow]Video prompt generation skipped by user.[/yellow]")
            return

        num_scenes = self.checkpoint_data.user_decisions.get("num_video_scenes", 3)
        if num_scenes <= 0:
            console.print("[yellow]No video scenes requested.[/yellow]")
            return

        output_dir = self.media_output_directory or self.checkpoint_data.resolved_config.get("output_directory")
        if not output_dir:
            console.print("[red]No output directory configured.[/red]")
            return

        backend = self._require_backend("Video prompt generation")
        verbose = self.checkpoint_data.resolved_config.get("verbose", False)

        # Build context for video prompts (same as image prompts)
        video_context = ""
        if self.context:
            video_context = self.context

        # Include world definition for setting/lore context
        if self.world:
            video_context = f"Story World:\n{self.world}\n\n{video_context}".strip()

        # Build character descriptions from world file and registry
        char_descriptions_parts: list[str] = []

        # Extract character descriptions from world file (authoritative source)
        if self.world:
            world_chars = ContextManager.extract_world_characters(self.world)
            if world_chars:
                char_descriptions_parts.append(world_chars)

        # Inject character descriptions from registry
        try:
            max_tokens = backend.get_context_token_budget()
            ctx_mgr = ContextManager(max_tokens=max_tokens)
            registry_descriptions = ctx_mgr.format_registry_for_image_prompt()
            if registry_descriptions:
                char_descriptions_parts.append(registry_descriptions)
                if verbose:
                    console.print("[dim]Injected character descriptions into video prompts[/dim]")
        except Exception:
            logging.getLogger(__name__).debug("Could not load character descriptions for video prompts", exc_info=True)

        char_descriptions = "\n".join(char_descriptions_parts)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Generating video prompt..."),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("video_prompt", total=None)

            video_prompts = backend.generate_video_prompt(
                story=self.story or "",
                context=video_context,
                num_scenes=num_scenes,
                character_descriptions=char_descriptions,
            )

        if not video_prompts:
            console.print("[yellow]Failed to generate video prompts.[/yellow]")
            return

        # Format and save video_prompt.txt
        from .llm_backend import LLMBackend

        prompt_text = str(self.checkpoint_data.original_inputs.get("prompt", ""))
        formatted = LLMBackend.format_video_prompt_file(video_prompts, prompt_text)

        video_path = os.path.join(output_dir, "video_prompt.txt")
        os.makedirs(output_dir, exist_ok=True)

        with open(video_path, "w", encoding="ascii", newline="\n") as f:
            f.write(to_portable_ascii(formatted))

        console.print(f"[bold green]✅ Video prompt saved as:[/bold green] {video_path}")

        # Store in checkpoint
        self.checkpoint_data.generated_content["video_prompts"] = video_prompts

    def _phase_image_decision(self) -> None:
        """Image generation decision phase."""
        if self.checkpoint_data is None:
            raise RuntimeError("Checkpoint data must be initialized")
        # Check if decision already made
        if self.checkpoint_data.user_decisions.get("wants_images") is not None:
            return

        configured_images = self.checkpoint_data.resolved_config.get("final_image_count")
        if configured_images is None:
            raise RuntimeError("Image decision must be supplied by the MCP client")
        wants_images = int(configured_images) > 0
        self.checkpoint_data.user_decisions["wants_images"] = wants_images
        self.checkpoint_data.user_decisions["num_images_requested"] = int(configured_images)

    def _phase_image_generate(self) -> None:
        """Image generation phase."""
        if self.checkpoint_data is None:
            raise RuntimeError("Checkpoint data must be initialized")
        wants_images = self.checkpoint_data.user_decisions.get("wants_images")
        if wants_images is None:
            raise RuntimeError("Image generation decision is missing; resume from the image decision phase.")
        if not wants_images:
            console.print("[yellow]Image generation skipped by user.[/yellow]")
            return

        num_images = self.checkpoint_data.user_decisions.get("num_images_requested", 1)
        if num_images <= 0:
            console.print("[yellow]No images will be generated.[/yellow]")
            return

        output_dir = self.media_output_directory or self.checkpoint_data.resolved_config.get("output_directory")
        if not output_dir:
            console.print("[yellow]No output directory specified for image generation.[/yellow]")
            return

        msg = f"Generating {num_images} image{'s' if num_images > 1 else ''}..."
        console.print(f"[bold blue]{msg}[/bold blue]")

        backend = self._require_backend("Image generation")
        try:
            # Generate image prompts from story
            verbose = self.checkpoint_data.resolved_config.get("verbose", False)
            if verbose:
                console.print("[dim]Generating image prompts...[/dim]")

            # Enrich context with world file and character visual descriptions
            image_context = self.context or ""

            # Include world definition for setting/lore/character context
            if self.world:
                image_context = f"Story World:\n{self.world}\n\n{image_context}".strip()

            # Build character descriptions from world file and registry
            char_descriptions_parts: list[str] = []

            # Extract character descriptions from world file (authoritative source)
            if self.world:
                world_chars = ContextManager.extract_world_characters(self.world)
                if world_chars:
                    char_descriptions_parts.append(world_chars)

            try:
                max_tokens = backend.get_context_token_budget()
                ctx_mgr = ContextManager(max_tokens=max_tokens)
                registry_descriptions = ctx_mgr.format_registry_for_image_prompt()
                if registry_descriptions:
                    char_descriptions_parts.append(registry_descriptions)
                    if verbose:
                        console.print("[dim]Injected character descriptions into image prompts[/dim]")
            except Exception:
                logging.getLogger(__name__).debug("Could not load character descriptions for images", exc_info=True)

            char_descriptions = "\n".join(char_descriptions_parts)

            image_prompts = backend.generate_image_prompt(
                story=self.story or "",
                context=image_context,
                num_prompts=num_images,
                character_descriptions=char_descriptions,
            )

            if not image_prompts:
                console.print("[yellow]Failed to generate image prompts.[/yellow]")
                return

            # Generate images for each prompt
            for i, image_prompt in enumerate(image_prompts[:num_images], 1):
                if verbose:
                    console.print(f"[dim]Generating image {i}: {image_prompt[:50]}...[/dim]")

                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"[bold blue]Generating image {i}..."),
                    console=console,
                    transient=True,
                ) as progress:
                    progress.add_task("image", total=None)

                    # Generate image - backends return (image_object, image_bytes)
                    try:
                        image_object, image_bytes = backend.generate_image(
                            self.story_prompt,
                            reference_image_bytes=None,
                            override_prompt=image_prompt,
                        )
                    except Exception as e:
                        console.print(f"[red]Failed to generate image {i}: {e}[/red]")
                        if self.checkpoint_data.resolved_config.get("verbose", False):
                            import traceback

                            console.print(f"[dim]{traceback.format_exc()}[/dim]")
                        image_bytes = None
                        image_object = None

                    if image_bytes:
                        # Determine image format from the image object or default to png
                        image_format = "png"  # Default format
                        if image_object and hasattr(image_object, "format") and image_object.format:
                            image_format = image_object.format.lower()

                        # Generate filename
                        image_name = backend.generate_image_name(self.story_prompt, self.story)
                        image_filename = f"{image_name}_{i:02d}.{image_format}"
                        image_path = Path(output_dir) / image_filename

                        # Ensure output directory exists
                        image_path.parent.mkdir(parents=True, exist_ok=True)

                        # Save image
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)

                        console.print(f"[bold green]✅ Image {i} saved:[/bold green] {image_path}")

                        # Store in checkpoint
                        if "generated_images" not in self.checkpoint_data.generated_content:
                            self.checkpoint_data.generated_content["generated_images"] = []

                        self.checkpoint_data.generated_content["generated_images"].append(
                            {
                                "prompt": image_prompt,
                                "filename": str(image_path),
                                "format": image_format,
                            }
                        )
                    else:
                        error_msg = f"[red]Failed to generate image {i}[/red]"
                        if self.checkpoint_data.resolved_config.get("verbose", False):
                            error_msg += " (backend returned None - check logs above for details)"
                        console.print(error_msg)

        except Exception as e:
            # Sanitize the error message to prevent binary data corruption
            error_msg = str(e)
            sanitized_error = "".join(c if c.isprintable() or c.isspace() else "?" for c in error_msg)
            console.print(f"[red]Error during image generation:[/red] {sanitized_error}")
            if self.checkpoint_data.resolved_config.get("verbose", False):
                import traceback

                # Also sanitize traceback
                tb = traceback.format_exc()
                sanitized_tb = "".join(c if c.isprintable() or c.isspace() else "?" for c in tb)
                console.print(f"[dim]Traceback:[/dim] {sanitized_tb}")

    def _phase_context_save(self) -> None:
        """Context saving phase."""
        if self.checkpoint_data is None:
            raise RuntimeError("Checkpoint data must be initialized")
        # Check if decision already made
        if self.checkpoint_data.user_decisions.get("save_as_context") is not None:
            return

        configured_save = self.checkpoint_data.resolved_config.get("final_save_context")
        if configured_save is None:
            raise RuntimeError("Context-save decision must be supplied by the MCP client")
        save_as_context = bool(configured_save)
        self.checkpoint_data.user_decisions["save_as_context"] = save_as_context

        if save_as_context:
            try:
                # Use the same root used for discovery, extension, and registry.
                context_dir = ContextManager().get_context_directory()
                context_dir.mkdir(parents=True, exist_ok=True)

                # Generate context filename based on story prompt
                prompt_summary = str(self.checkpoint_data.original_inputs.get("prompt", "story"))

                # Create a safe filename from prompt (alphanumeric only, truncated)
                safe_name = "".join(
                    c for c in prompt_summary[: self.MAX_FILENAME_PREFIX_LENGTH] if c.isalnum() or c in " -_"
                )
                safe_name = safe_name.replace(" ", "_").strip("_")
                if not safe_name:
                    safe_name = "story"

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # A resumed session may save a replacement within the same second.
                # Never overwrite the context that its previous finalization owns.
                context_filename = f"{safe_name}_{timestamp}_{uuid4().hex[:8]}_{self.checkpoint_data.session_id}.md"
                context_path = context_dir / context_filename

                # Create context content
                context_content = f"# Story Context: {prompt_summary}\n\n"
                context_content += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                context_content += f"**Original Prompt:** {prompt_summary}\n\n"

                # Add parent story tracking for extensions
                source_file = self.checkpoint_data.resolved_config.get("source_context_file")
                if source_file:
                    parent_file = Path(source_file).stem
                    context_content += f"**Extended From:** {parent_file}\n\n"

                # Add backend and model information
                if self.llm_backend:
                    context_content += f"**Backend:** {self.llm_backend.name}\n\n"
                    model_info = self.llm_backend.get_model_info()
                    if model_info.get("story_model"):
                        context_content += f"**Story Model:** {model_info['story_model']}\n\n"
                    if model_info.get("image_model"):
                        context_content += f"**Image Model:** {model_info['image_model']}\n\n"

                # Add story parameters with source tracking
                cli_args = self.checkpoint_data.original_inputs.get("cli_arguments", {})
                param_fields = [
                    ("characters", "Characters"),
                    ("setting", "Setting"),
                    ("tone", "Tone"),
                    ("style", "Style"),
                    ("voice", "Voice"),
                    ("theme", "Theme"),
                    ("age_range", "Age Group"),
                    ("image_style", "Art Style"),
                    ("length", "Length"),
                    ("learning_focus", "Learning Focus"),
                ]

                for field_name, display_name in param_fields:
                    if field_name == "characters":
                        if cli_args and cli_args.get("characters"):
                            context_content += f"**{display_name}:** {', '.join(cli_args['characters'])}\n\n"
                        continue

                    source, value = self._get_parameter_source(field_name)
                    if value is not None:
                        # Keep metadata machine-readable.  Source information is
                        # stored separately so saved contexts can be extended.
                        context_content += f"**{display_name}:** {value}\n\n"
                        context_content += f"**{display_name} Source:** {source}\n\n"

                context_content += "## Story\n\n"
                context_content += self.story or ""
                context_content += "\n\n"

                # Add refinements if any
                if self.refinements:
                    context_content += "## Refinements Applied\n\n"
                    context_content += self.refinements
                    context_content += "\n\n"

                # Save context file
                with open(context_path, "w", encoding="utf-8") as f:
                    f.write(context_content)

                console.print(f"[bold green]✅ Story saved as context:[/bold green] {context_path}")

                # Update character registry with new story
                try:
                    registry_cm = ContextManager()
                    registry_metadata: dict[str, Any] = {}
                    if cli_args and cli_args.get("characters"):
                        registry_metadata["characters"] = ", ".join(cli_args["characters"])
                    registry_cm.update_character_registry(context_content, registry_metadata, context_path.stem)
                except Exception:
                    logging.getLogger(__name__).warning("Character registry update failed", exc_info=True)

                # Store in checkpoint
                self.checkpoint_data.generated_content["context_file"] = str(context_path)

            except Exception as e:
                console.print(f"[red]Error saving story as context:[/red] {e}")
                if self.checkpoint_data.resolved_config.get("verbose", False):
                    import traceback

                    console.print(traceback.format_exc())
