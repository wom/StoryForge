"""
Phase-based execution engine for StoryForge with checkpoint support.

This module provides a structured way to execute StoryForge phases with
automatic checkpointing and recovery capabilities.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from platformdirs import user_data_dir
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

from .checkpoint import CheckpointData, CheckpointManager, ExecutionPhase
from .config import Config, load_config
from .context import ContextManager
from .llm_backend import get_backend
from .prompt import Prompt

console = Console()


class PhaseExecutor:
    """Phase-based execution engine with checkpoint support."""

    def __init__(self, checkpoint_manager: CheckpointManager) -> None:
        """Initialize the phase executor."""
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_data: CheckpointData | None = None
        self.config: Config | None = None
        self.llm_backend: Any = None  # LLMBackend type not available in this scope
        self.context: str | None = None
        self.story_prompt: Any = None  # Prompt type not available in this scope
        self.story: str | None = None
        self.refinements: str | None = None
        self._initialized_phases: set[ExecutionPhase] = set()  # Track which phases have been initialized

    def execute_from_checkpoint(self, checkpoint_data: CheckpointData, resume_phase: ExecutionPhase) -> None:
        """Execute StoryForge starting from a checkpoint and specific phase."""
        console.print(f"[bold cyan]Resuming from session:[/bold cyan] {checkpoint_data.session_id}")
        console.print(f"[dim]Creating new session starting from phase:[/dim] {resume_phase.value}")

        try:
            # Validate checkpoint data before proceeding
            self._validate_checkpoint_data(checkpoint_data)

            # Create a new checkpoint session that inherits from the old one
            self.checkpoint_data = self._create_resumed_session(checkpoint_data, resume_phase)

            # Save the new session checkpoint
            self.checkpoint_manager.save_checkpoint(self.checkpoint_data)

            # Start execution from the specified phase
            self._execute_phase_sequence(resume_phase)

            # Mark session as completed
            self.checkpoint_data.mark_completed()
            self.checkpoint_manager.save_checkpoint(self.checkpoint_data)
            console.print("[bold green]✅ StoryForge session completed successfully![/bold green]")

        except typer.Exit:
            # User cancelled - don't mark as failed
            raise
        except KeyboardInterrupt:
            # User interrupted - save current state
            if self.checkpoint_data:
                console.print("\n[yellow]Session interrupted by user. Progress saved.[/yellow]")
                self.checkpoint_manager.save_checkpoint(self.checkpoint_data)
            raise typer.Exit(130) from None  # Standard exit code for SIGINT
        except Exception as e:
            # Mark session as failed and save checkpoint
            current_phase = self.checkpoint_data.current_phase if self.checkpoint_data else "unknown"
            error_msg = f"Error in phase {current_phase}: {str(e)}"
            console.print(f"[red]Session failed:[/red] {error_msg}")

            if self.checkpoint_data:
                self.checkpoint_data.mark_failed(error_msg)
                try:
                    self.checkpoint_manager.save_checkpoint(self.checkpoint_data)
                    resumed_from = (
                        self.checkpoint_data.progress.get("resumed_from_session")
                        if self.checkpoint_data.progress
                        else None
                    )
                    if resumed_from:
                        console.print(
                            f"[dim]Failed resumed session saved as:[/dim] {self.checkpoint_data.session_id} "
                            f"[dim](resumed from {resumed_from})[/dim]"
                        )
                    else:
                        console.print(f"[dim]Failed session saved as:[/dim] {self.checkpoint_data.session_id}")
                except Exception as save_error:
                    console.print(f"[red]Could not save failed session:[/red] {save_error}")
            raise

    def _create_resumed_session(
        self, original_checkpoint: CheckpointData, resume_phase: ExecutionPhase
    ) -> CheckpointData:
        """Create a new checkpoint session based on original session, starting from resume_phase."""
        now = datetime.now().isoformat() + "Z"
        new_session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_sf_resumed"

        # Don't pre-mark phases as completed - start fresh to avoid skip logic issues
        # Only preserve generated content and decisions up to resume point
        new_checkpoint = CheckpointData(
            session_id=new_session_id,
            created_at=now,
            updated_at=now,
            status="active",
            current_phase=resume_phase.value,
            completed_phases=[],  # Start fresh - phases will be marked as we execute them
            original_inputs=original_checkpoint.original_inputs.copy(),
            resolved_config=original_checkpoint.resolved_config.copy(),
            generated_content=self._get_content_up_to_phase(original_checkpoint, resume_phase),
            user_decisions=self._get_decisions_up_to_phase(original_checkpoint, resume_phase),
            context_data=original_checkpoint.context_data.copy() if original_checkpoint.context_data else None,
            progress={
                "total_phases": len(ExecutionPhase) - 1,  # Exclude COMPLETED
                "completed_count": 0,
                "completion_percentage": 0,
                "resumed_from_session": original_checkpoint.session_id,  # Track parent session
                "resumed_at_phase": resume_phase.value,  # Track resume point
            },
        )

        return new_checkpoint

    def _validate_checkpoint_data(self, checkpoint_data: CheckpointData) -> None:
        """Validate checkpoint data for consistency and completeness."""
        if not checkpoint_data:
            raise ValueError("Checkpoint data is None")

        if not checkpoint_data.session_id:
            raise ValueError("Checkpoint missing session_id")

        if not checkpoint_data.original_inputs.get("prompt"):
            raise ValueError("Checkpoint missing original prompt")

        if not checkpoint_data.resolved_config:
            raise ValueError("Checkpoint missing resolved configuration")

        # Validate phase is known
        # Only validate ExecutionPhase values - ignore old incompatible checkpoints
        try:
            ExecutionPhase(checkpoint_data.current_phase)
        except ValueError as e:
            raise ValueError(
                f"Incompatible checkpoint format - please start a new session: {checkpoint_data.current_phase}"
            ) from e

        # Validate completed phases - skip invalid ones from old checkpoints
        valid_completed_phases = []
        for phase_name in checkpoint_data.completed_phases:
            try:
                ExecutionPhase(phase_name)
                valid_completed_phases.append(phase_name)
            except ValueError:
                # Skip invalid phases from old checkpoint format
                continue

        # Update checkpoint with only valid phases
        checkpoint_data.completed_phases = valid_completed_phases

    def _get_completed_phases_before(self, resume_phase: ExecutionPhase) -> list[str]:
        """Get list of phases that should be marked as completed before the resume phase."""
        phase_order = [
            ExecutionPhase.INIT,
            ExecutionPhase.CONFIG_LOAD,
            ExecutionPhase.BACKEND_INIT,
            ExecutionPhase.PROMPT_CONFIRM,
            ExecutionPhase.CONTEXT_LOAD,
            ExecutionPhase.PROMPT_BUILD,
            ExecutionPhase.STORY_GENERATE,
            ExecutionPhase.STORY_SAVE,
            ExecutionPhase.IMAGE_DECISION,
            ExecutionPhase.IMAGE_GENERATE,
            ExecutionPhase.CONTEXT_SAVE,
        ]

        try:
            resume_index = phase_order.index(resume_phase)
            return [phase.value for phase in phase_order[:resume_index]]
        except ValueError:
            return []

    def _get_content_up_to_phase(
        self, original_checkpoint: CheckpointData, resume_phase: ExecutionPhase
    ) -> dict[str, Any]:
        """Get generated content that should be preserved up to the resume phase."""
        content: dict[str, Any] = {
            "story": None,
            "refinements": None,
            "images": [],
        }

        # Preserve story if resuming from a phase after story generation
        story_phases = [
            ExecutionPhase.STORY_SAVE,
            ExecutionPhase.IMAGE_DECISION,
            ExecutionPhase.IMAGE_GENERATE,
            ExecutionPhase.CONTEXT_SAVE,
        ]

        if resume_phase in story_phases:
            content["story"] = original_checkpoint.generated_content.get("story")
            content["refinements"] = original_checkpoint.generated_content.get("refinements")

        # Don't preserve images or context files - let user make new decisions
        return content

    def _get_decisions_up_to_phase(
        self, original_checkpoint: CheckpointData, resume_phase: ExecutionPhase
    ) -> dict[str, Any]:
        """Get user decisions that should be preserved up to the resume phase."""
        decisions = {
            "story_accepted": None,
            "wants_images": None,
            "num_images_requested": None,
            "save_as_context": None,
        }

        # Only preserve story_accepted if resuming from phases after story generation
        # but before image decision (so we don't re-ask for story refinement)
        if resume_phase in [ExecutionPhase.STORY_SAVE]:
            decisions["story_accepted"] = original_checkpoint.user_decisions.get("story_accepted")

        # Don't preserve any other decisions - let user make fresh choices
        return decisions

    def execute_new_session(
        self,
        prompt: str,
        cli_arguments: dict[str, Any],
        resolved_config: dict[str, Any],
        prompt_obj: Prompt | None = None,
    ) -> None:
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
            self._execute_phase_sequence(ExecutionPhase.INIT)

            # Mark session as completed
            self.checkpoint_data.mark_completed()
            self.checkpoint_manager.save_checkpoint(self.checkpoint_data)
            console.print("[bold green]✅ StoryForge session completed successfully![/bold green]")

        except typer.Exit:
            # User cancelled - don't mark as failed
            raise
        except KeyboardInterrupt:
            # User interrupted - save current state
            if self.checkpoint_data:
                console.print("\n[yellow]Session interrupted by user. Progress saved.[/yellow]")
                self.checkpoint_manager.save_checkpoint(self.checkpoint_data)
            raise typer.Exit(130) from None  # Standard exit code for SIGINT
        except Exception as e:
            # Mark session as failed and save checkpoint
            current_phase = self.checkpoint_data.current_phase if self.checkpoint_data else "unknown"
            error_msg = f"Error in phase {current_phase}: {str(e)}"
            console.print(f"[red]Session failed:[/red] {error_msg}")

            if self.checkpoint_data:
                self.checkpoint_data.mark_failed(error_msg)
                try:
                    self.checkpoint_manager.save_checkpoint(self.checkpoint_data)
                    console.print(f"[dim]Failed session saved as:[/dim] {self.checkpoint_data.session_id}")
                except Exception as save_error:
                    console.print(f"[red]Could not save failed session:[/red] {save_error}")
            raise

    def _clear_phases_from(self, start_phase: ExecutionPhase) -> None:
        """Clear completion status for all phases starting from the given phase."""
        if not self.checkpoint_data:
            return

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
            ExecutionPhase.IMAGE_DECISION,
            ExecutionPhase.IMAGE_GENERATE,
            ExecutionPhase.CONTEXT_SAVE,
        ]

        # Find the starting index
        try:
            start_index = phase_order.index(start_phase)
        except ValueError:
            return

        # Remove completion status for phases from start_phase onwards
        phases_to_clear = [phase.value for phase in phase_order[start_index:]]

        # Filter out phases that shouldn't be cleared
        self.checkpoint_data.completed_phases = [
            phase for phase in self.checkpoint_data.completed_phases if phase not in phases_to_clear
        ]

        # Update current phase
        self.checkpoint_data.current_phase = start_phase.value
        self.checkpoint_data.updated_at = datetime.now().isoformat() + "Z"

    def _execute_phase_sequence(self, start_phase: ExecutionPhase) -> None:
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
            ExecutionPhase.IMAGE_DECISION,
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
            try:
                phase_index = phase_order.index(phase)
            except ValueError:
                continue

            # Only execute if this phase is before our start phase and hasn't been initialized yet
            if phase_index < start_index and phase not in self._initialized_phases:
                console.print(f"[dim]Initializing required phase:[/dim] {phase.value}")
                self._execute_phase(phase)
                self._initialized_phases.add(phase)
                # Don't add to checkpoint.completed_phases - these are initialization only

        # Execute phases in sequence from start_phase
        for phase in phase_order[start_index:]:
            if self._should_skip_phase(phase):
                continue

            console.print(f"[dim]Executing phase:[/dim] {phase.value}")
            self._execute_phase(phase)

            # Update checkpoint after each phase
            if self.checkpoint_data is not None:
                self.checkpoint_data.update_phase(phase)
                self.checkpoint_manager.save_checkpoint(self.checkpoint_data)

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

        except typer.Exit:
            # User cancelled - propagate up
            raise
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
        # CLI arguments are already validated in main(), nothing to do here
        pass

    def _phase_config_load(self) -> None:
        """Load configuration phase."""
        assert self.checkpoint_data is not None, "Checkpoint data must be initialized"
        verbose = self.checkpoint_data.resolved_config.get("verbose", False)
        self.config = load_config(verbose=verbose)

    def _phase_backend_init(self) -> None:
        """Initialize LLM backend phase."""
        assert self.checkpoint_data is not None, "Checkpoint data must be initialized"
        backend_name = self.checkpoint_data.resolved_config.get("backend")
        verbose = self.checkpoint_data.resolved_config.get("verbose", False)

        if verbose:
            console.print("[dim]Initializing AI backend...[/dim]")

        # Better error handling for backend initialization
        try:
            self.llm_backend = get_backend(config_backend=backend_name, config=self.config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize {backend_name or 'default'} backend. "
                f"Please check that your API key is set correctly in environment variables. "
                f"Error: {e}"
            ) from e

        if not self.llm_backend:
            backend_display = backend_name or "auto-detected backend"
            raise RuntimeError(
                f"Backend initialization returned None for '{backend_display}'. "
                f"Please verify your API key is set and valid."
            )

        if verbose and self.llm_backend:
            console.print(f"[dim]Using {self.llm_backend.name} backend[/dim]")

    def _phase_prompt_confirm(self) -> None:
        """Prompt confirmation phase."""
        assert self.checkpoint_data is not None, "Checkpoint data must be initialized"
        # Extract parameters from checkpoint
        original_inputs = self.checkpoint_data.original_inputs
        resolved_config = self.checkpoint_data.resolved_config

        prompt = str(original_inputs.get("prompt", ""))
        cli_args = original_inputs.get("cli_arguments", {})

        # Show summary and get confirmation
        from .StoryForge import show_prompt_summary_and_confirm

        if not show_prompt_summary_and_confirm(
            prompt=prompt,
            age_range=cli_args.get("age_range", resolved_config.get("age_range")),
            style=cli_args.get("style", resolved_config.get("style")),
            tone=cli_args.get("tone", resolved_config.get("tone")),
            theme=cli_args.get("theme", resolved_config.get("theme")),
            length=cli_args.get("length", resolved_config.get("length")),
            setting=cli_args.get("setting"),
            characters=cli_args.get("characters"),
            learning_focus=cli_args.get("learning_focus"),
            image_style=cli_args.get("image_style", resolved_config.get("image_style")),
            generation_type="story",
            backend_name=self.llm_backend.name if self.llm_backend else None,
        ):
            console.print("[yellow]Story generation cancelled.[/yellow]")
            raise typer.Exit(0)

    def _phase_context_load(self) -> None:
        """Load context files phase."""
        assert self.checkpoint_data is not None, "Checkpoint data must be initialized"
        use_context = self.checkpoint_data.resolved_config.get("use_context", True)
        verbose = self.checkpoint_data.resolved_config.get("verbose", False)

        context_manager = ContextManager()
        if use_context:
            self.context = context_manager.load_context()
            if verbose and self.context:
                console.print(f"[dim]Loaded context from {len(self.context.split())} words[/dim]")
            elif verbose:
                console.print("[dim]No context files found[/dim]")

            # Store context in checkpoint
            if self.context and self.checkpoint_data:
                self.checkpoint_data.context_data = {
                    "loaded_context": self.context,
                    "context_files_used": [],  # TODO: Track which files were used
                }
        else:
            self.context = None
            if verbose:
                console.print("[dim]Context loading skipped due to --no-use-context[/dim]")

    def _phase_build_prompt(self) -> None:
        """Build the story prompt from inputs."""
        assert self.checkpoint_data is not None, "Checkpoint data must be initialized"

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

        self.story_prompt = Prompt(
            prompt=prompt,
            context=self.context,
            length=str(cli_args.get("length") or resolved_config.get("length") or ""),
            age_range=str(cli_args.get("age_range") or resolved_config.get("age_range") or ""),
            style=str(cli_args.get("style") or resolved_config.get("style") or ""),
            tone=str(cli_args.get("tone") or resolved_config.get("tone") or ""),
            theme=cli_args.get("theme") or resolved_config.get("theme"),
            setting=cli_args.get("setting"),
            characters=cli_args.get("characters"),
            learning_focus=cli_args.get("learning_focus"),
            image_style=str(cli_args.get("image_style") or resolved_config.get("image_style") or ""),
            continuation_mode=continuation_mode,
            ending_type=ending_type,
        )

    def _phase_story_generate(self) -> None:
        """Story generation and refinement phase."""
        assert self.checkpoint_data is not None, "Checkpoint data must be initialized"
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
                from .StoryForge import load_story_from_file

                self.story = load_story_from_file("storyforge/test_story.txt")
                console.print("[cyan][DEBUG] load_story_from_file was called and returned.[/cyan]")
            else:
                self.story = self.llm_backend.generate_story(self.story_prompt)
                if verbose:
                    console.print("[cyan][DEBUG] generate_story was called and returned.[/cyan]")

        if self.story is None or self.story == "[Error generating story]":
            raise RuntimeError("Failed to generate story. Please check your API key and try again.")

        # Store story in checkpoint
        self.checkpoint_data.generated_content["story"] = self.story

        # Handle story refinement
        self._handle_story_refinement()

    def _handle_story_refinement(self) -> None:
        """Handle story refinement loop."""
        assert self.checkpoint_data is not None, "Checkpoint data must be initialized"
        debug = self.checkpoint_data.resolved_config.get("debug", False)
        verbose = self.checkpoint_data.resolved_config.get("verbose", False)

        # Display the generated story
        console.print("\n[bold green]Generated Story:[/bold green]")
        prompt_preview = self.checkpoint_data.original_inputs.get("prompt", "")
        console.print(f"[dim]Prompt:[/dim] {prompt_preview}")
        if self.refinements:
            console.print(f"[dim]Refinements:[/dim] {self.refinements}")
        console.print()
        console.print(self.story or "")
        console.print()

        # Ask if user wants to refine
        if Confirm.ask(
            "[bold yellow]Would you like to refine the story?[/bold yellow]",
            default=False,
            show_default=True,
        ):
            self.refinements = typer.prompt("Refinements:")

            # Build refinement prompt that includes the existing story
            refinement_instruction = (
                "I have an existing story that needs refinement. "
                "Please keep the story as similar as possible to the original, "
                "but apply the following specific changes. "
                "Maintain all the good storytelling elements, structure, and flow of the original story.\n\n"
                f"ORIGINAL STORY:\n{self.story}\n\n"
                f"REQUESTED CHANGES:\n{self.refinements}\n\n"
                "Please generate the refined version of this story, incorporating the requested changes "
                "while preserving everything else about the original story."
            )

            # Store refinements in checkpoint
            if self.checkpoint_data:
                self.checkpoint_data.generated_content["refinements"] = self.refinements
                self.checkpoint_manager.save_checkpoint(self.checkpoint_data)

            # Create a new prompt with the refinement instructions
            # Save the original prompt first
            original_prompt_text = self.story_prompt.prompt if self.story_prompt else ""

            # Temporarily update the prompt for refinement
            if self.story_prompt:
                self.story_prompt.prompt = refinement_instruction

            # Regenerate story with refinement
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Refining story..."),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("refining", total=None)

                if debug:
                    # In debug mode, show what would be sent but use test story
                    console.print("[cyan][DEBUG] Refinement prompt would be sent to LLM[/cyan]")
                    console.print(f"[dim]{refinement_instruction[:200]}...[/dim]")
                    from .StoryForge import load_story_from_file

                    self.story = load_story_from_file("storyforge/test_story.txt")
                else:
                    self.story = self.llm_backend.generate_story(self.story_prompt)
                    if verbose:
                        console.print("[cyan][DEBUG] Refinement story generated.[/cyan]")

            # Restore original prompt
            if self.story_prompt:
                self.story_prompt.prompt = original_prompt_text

            if self.story is None or self.story == "[Error generating story]":
                raise RuntimeError("Failed to refine story. Please check your API key and try again.")

            # Update story in checkpoint
            self.checkpoint_data.generated_content["story"] = self.story

            # Clear this phase from completed so we can run refinement again
            if ExecutionPhase.STORY_GENERATE.value in self.checkpoint_data.completed_phases:
                self.checkpoint_data.completed_phases.remove(ExecutionPhase.STORY_GENERATE.value)

            self.checkpoint_manager.save_checkpoint(self.checkpoint_data)

            # Recursively ask for more refinements
            self._handle_story_refinement()
        else:
            # Story accepted
            if self.checkpoint_data:
                self.checkpoint_data.user_decisions["story_accepted"] = True

    def _phase_story_save(self) -> None:
        """Save story to file phase."""
        assert self.checkpoint_data is not None, "Checkpoint data must be initialized"
        output_dir = self.checkpoint_data.resolved_config.get("output_directory")
        if not output_dir:
            from .StoryForge import generate_default_output_dir

            # Check if this is an extended story
            continuation_mode = self.checkpoint_data.resolved_config.get("continuation_mode", False)
            output_dir = generate_default_output_dir(extended=continuation_mode)
            self.checkpoint_data.resolved_config["output_directory"] = output_dir

        story_filename = "story.txt"
        story_path = os.path.join(output_dir, story_filename)
        os.makedirs(output_dir, exist_ok=True)

        with open(story_path, "w", encoding="utf-8") as f:
            prompt_text = str(self.checkpoint_data.original_inputs.get("prompt", ""))
            f.write(f"Story: {prompt_text}\n\n")
            f.write(self.story or "")

        console.print(f"[bold green]✅ Story saved as:[/bold green] {story_path}")

    def _phase_image_decision(self) -> None:
        """Image generation decision phase."""
        assert self.checkpoint_data is not None, "Checkpoint data must be initialized"
        # Check if decision already made
        if self.checkpoint_data.user_decisions.get("wants_images") is not None:
            return

        wants_images = Confirm.ask("Would you like to generate illustrations for the story?")
        self.checkpoint_data.user_decisions["wants_images"] = wants_images

        if wants_images:
            num_images = typer.prompt("How many images would you like to generate?", type=int, default=1)
            self.checkpoint_data.user_decisions["num_images_requested"] = num_images

    def _phase_image_generate(self) -> None:
        """Image generation phase."""
        assert self.checkpoint_data is not None, "Checkpoint data must be initialized"
        wants_images = self.checkpoint_data.user_decisions.get("wants_images", False)
        if not wants_images:
            console.print("[yellow]Image generation skipped by user.[/yellow]")
            return

        num_images = self.checkpoint_data.user_decisions.get("num_images_requested", 1)
        if num_images <= 0:
            console.print("[yellow]No images will be generated.[/yellow]")
            return

        output_dir = self.checkpoint_data.resolved_config.get("output_directory")
        if not output_dir:
            console.print("[yellow]No output directory specified for image generation.[/yellow]")
            return

        msg = f"Generating {num_images} image{'s' if num_images > 1 else ''}..."
        console.print(f"[bold blue]{msg}[/bold blue]")

        try:
            # Generate image prompts from story
            verbose = self.checkpoint_data.resolved_config.get("verbose", False)
            if verbose:
                console.print("[dim]Generating image prompts...[/dim]")

            if not self.llm_backend:
                console.print("[red]No LLM backend available for image generation.[/red]")
                return

            image_prompts = self.llm_backend.generate_image_prompt(
                story=self.story or "",
                context=self.context or "",
                num_prompts=num_images,
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
                        image_object, image_bytes = self.llm_backend.generate_image(
                            self.story_prompt, reference_image_bytes=None
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
                        image_name = self.llm_backend.generate_image_name(self.story_prompt, self.story)
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
        assert self.checkpoint_data is not None, "Checkpoint data must be initialized"
        # Check if decision already made
        if self.checkpoint_data.user_decisions.get("save_as_context") is not None:
            return

        save_as_context = Confirm.ask(
            "[bold blue]Save this story as future context for character development?[/bold blue]"
        )
        self.checkpoint_data.user_decisions["save_as_context"] = save_as_context

        if save_as_context:
            try:
                # Get context directory (normalized to lowercase 'storyforge')
                context_dir = Path(user_data_dir("storyforge", "storyforge")) / "context"
                context_dir.mkdir(parents=True, exist_ok=True)

                # Generate context filename based on story prompt
                prompt_summary = str(self.checkpoint_data.original_inputs.get("prompt", "story"))

                # Create a safe filename from prompt (first 30 chars, alphanumeric only)
                safe_name = "".join(c for c in prompt_summary[:30] if c.isalnum() or c in " -_")
                safe_name = safe_name.replace(" ", "_").strip("_")
                if not safe_name:
                    safe_name = "story"

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                context_filename = f"{safe_name}_{timestamp}.md"
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

                # Add story parameters if available
                cli_args = self.checkpoint_data.original_inputs.get("cli_arguments", {})
                if cli_args and cli_args.get("characters"):
                    context_content += f"**Characters:** {', '.join(cli_args['characters'])}\n\n"
                if cli_args and cli_args.get("setting"):
                    context_content += f"**Setting:** {cli_args['setting']}\n\n"

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

                # Store in checkpoint
                self.checkpoint_data.generated_content["context_file"] = str(context_path)

            except Exception as e:
                console.print(f"[red]Error saving story as context:[/red] {e}")
                if self.checkpoint_data.resolved_config.get("verbose", False):
                    import traceback

                    console.print(traceback.format_exc())
