"""
StoryForge: Simplified CLI for generating illustrated stories using multiple LLM backends.
Supports Google Gemini and Anthropic Claude backends.
"""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import typer
from rich.console import Console
from rich.prompt import Confirm

from .checkpoint import CheckpointManager
from .client import display_error, display_success, get_client, poll_session_until_complete, run_sync
from .config import Config, ConfigError, load_config
from .context import ContextManager
from .phase_executor import PhaseExecutor
from .schema.cli_integration import (
    generate_boolean_cli_option,
    generate_cli_option,
    generate_multi_option,
    validate_cli_arguments,
)

console = Console()

# Create Typer app instance for entrypoint
app = typer.Typer(
    help="StoryForge: Generate illustrated stories using AI language models.\n\n"
    "Configuration: Use 'storyforge config init' to create a config file with default values.\n"
    "Environment: Set STORYFORGE_CONFIG to use a custom config file location.\n\n"
    "Checkpoint System: StoryForge automatically saves progress during execution.\n"
    "Use --continue to resume from previous sessions or retry from any completed phase.\n"
    "Checkpoint files are stored in ~/.local/share/StoryForge/checkpoints/",
    epilog="Examples:\n\n"
    "  # Generate a story with a simple prompt\n"
    "  sf 'Moe and Curly play a joke'\n\n"
    "  # Generate a longer story with specific parameters\n"
    "  sf 'Moe and Curly play a joke' --length long --age-range '8-10' --tone humorous\n\n"
    "  # Resume from a previous session\n"
    "  sf continue\n\n"
    "  # Initialize configuration file\n"
    "  sf config init\n\n"
    "  # Generate with custom image style\n"
    "  sf 'A dragon learns to fly' --image-style anime --theme adventure",
)
config_app = typer.Typer(help="Configuration management commands")
app.add_typer(config_app, name="config")


def generate_default_output_dir(extended: bool = False) -> str:
    """Generate a timestamped output directory name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_extended" if extended else ""
    output_dir = f"storyforge_output_{timestamp}{suffix}"
    return output_dir


def show_prompt_summary_and_confirm(
    prompt: str,
    age_range: str,
    style: str,
    tone: str,
    theme: str | None,
    length: str,
    setting: str | None,
    characters: list[str] | None,
    learning_focus: str | None,
    image_style: str = "chibi",
    generation_type: str = "story",
    backend_name: str | None = None,
) -> bool:
    """Display a summary of the prompt and ask for user confirmation."""
    console.print(f"\n[bold cyan]📋 {generation_type.title()} Generation Summary:[/bold cyan]")
    console.print(f"[bold]Prompt:[/bold] {prompt}")
    console.print(f"[bold]Age Range:[/bold] {age_range}")
    console.print(f"[bold]Length:[/bold] {length}")
    console.print(f"[bold]Style:[/bold] {style}")
    console.print(f"[bold]Tone:[/bold] {tone}")
    if theme and theme != "random":
        console.print(f"[bold]Theme:[/bold] {theme}")
    if learning_focus:
        console.print(f"[bold]Learning Focus:[/bold] {learning_focus}")
    if setting:
        console.print(f"[bold]Setting:[/bold] {setting}")
    if characters:
        console.print(f"[bold]Characters:[/bold] {', '.join(characters)}")
    console.print(f"[bold]Image Style:[/bold] {image_style}")
    if backend_name:
        console.print(f"[bold]Backend:[/bold] {backend_name}")
    console.print()
    return Confirm.ask(f"[bold green]Proceed with {generation_type} generation?[/bold green]")


def load_story_from_file(rel_path: str) -> str | None:
    """
    Load story content from the default story file.
    Returns:
        str | None: The story content, or None if the file doesn't exist.
    """
    story_file = Path(rel_path)
    if not story_file.exists():
        print(f"[yellow]Warning:[/yellow] Story file {story_file} does not exist.")
        return None
    with open(story_file, encoding="utf-8") as f:
        story = f.read().strip()
    return story


@config_app.command(name="init", help="Create a default configuration file in the XDG config directory.")
def init_config(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config file"),
    config_path: str | None = typer.Option(None, "--path", "-p", help="Custom config file path"),
) -> None:
    """Create a default configuration file."""
    try:
        config = Config()
        # Use provided config path or default
        target_path = config.get_default_config_path() if config_path is None else Path(config_path)

        if target_path.exists() and not force:
            console.print(f"[yellow]Configuration file already exists:[/yellow] {target_path}")
            console.print("[dim]Use --force to overwrite the existing configuration file[/dim]")
            raise typer.Exit(0)

        # Create parent directory if needed
        target_path.parent.mkdir(parents=True, exist_ok=True)
        created_path = config.create_default_config(target_path)
        console.print(f"[bold green]✅ Configuration file created:[/bold green] {created_path}")
        console.print()
        console.print(
            "[bold]You can override the configuration location with the STORYFORGE_CONFIG environment variable.[/bold]"
        )
        console.print()
        console.print("[bold]Configuration file locations (in priority order):[/bold]")
        for i, search_path in enumerate(config.get_config_paths(), 1):
            if search_path == created_path:
                console.print(f"  {i}. {search_path} [bold green](created here)[/bold green]")
            else:
                console.print(f"  {i}. {search_path}")
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error creating configuration file:[/red] {e}", style="bold")
        raise typer.Exit(1) from None


@app.command(
    "main",  # Keep it named "main" but we'll make it default via entry point
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Generate an illustrated story from a prompt",
)
def main(
    ctx: typer.Context,
    prompt: str | None = typer.Argument(
        None,
        help="The story prompt to generate from (positional, required unless using --init-config or --continue)",
    ),
    length: str | None = generate_cli_option("length"),
    age_range: str | None = generate_cli_option("age_range"),
    style: str | None = generate_cli_option("style"),
    tone: str | None = generate_cli_option("tone"),
    theme: str | None = generate_cli_option("theme"),
    learning_focus: str | None = generate_cli_option("learning_focus"),
    setting: str | None = generate_cli_option("setting"),
    characters: Annotated[
        list[str] | None,
        generate_multi_option("characters", "--character"),
    ] = None,
    image_style: str | None = generate_cli_option("image_style"),
    output_dir: str | None = generate_cli_option("output_dir"),
    use_context: bool | None = generate_boolean_cli_option("use_context", "--use-context/--no-use-context"),
    verbose: bool | None = generate_cli_option("verbose"),
    debug: bool | None = generate_cli_option("debug"),
    backend: str | None = generate_cli_option("backend"),
    continue_session: bool = typer.Option(
        False,
        "--continue",
        help="Resume execution from a previous StoryForge session. "
        "Displays the last 5 sessions and allows you to choose which phase to resume from. "
        "For completed sessions, you can generate new images, modify the story, "
        "save as context, or restart completely with the same parameters.",
    ),
):
    # Validate CLI arguments using schema before processing
    cli_args = {
        "length": length,
        "age_range": age_range,
        "style": style,
        "tone": tone,
        "theme": theme,
        "learning_focus": learning_focus,
        "setting": setting,
        "characters": characters,
        "image_style": image_style,
        "output_dir": output_dir,
        "use_context": use_context,
        "verbose": verbose,
        "debug": debug,
        "backend": backend,
    }

    # Only validate provided CLI arguments (not None values)
    provided_args = {k: v for k, v in cli_args.items() if v is not None}
    validation_errors = validate_cli_arguments(**provided_args)

    if validation_errors:
        console.print("[red]CLI Argument Validation Errors:[/red]", style="bold")
        for error in validation_errors:
            console.print(f"  - {error}", style="red")
        raise typer.Exit(1)

    # Handle --continue option
    if continue_session:
        try:
            checkpoint_manager = CheckpointManager()
            checkpoint_data = checkpoint_manager.prompt_checkpoint_selection()

            if checkpoint_data is None:
                console.print("[yellow]No checkpoint selected. Exiting.[/yellow]")
                raise typer.Exit(0)

            # Get the phase to resume from
            resume_phase = checkpoint_manager.prompt_phase_selection(checkpoint_data)
            if resume_phase is None:
                console.print("[yellow]No phase selected. Exiting.[/yellow]")
                raise typer.Exit(0)

            # Execute using phase executor
            phase_executor = PhaseExecutor(checkpoint_manager)
            phase_executor.execute_from_checkpoint(checkpoint_data, resume_phase)
            raise typer.Exit(0)

        except typer.Exit:
            # Let typer.Exit propagate normally
            raise
        except Exception as e:
            console.print(f"[red]Error handling checkpoint continuation:[/red] {e}", style="bold")
            raise typer.Exit(1) from e

    try:
        # Load configuration file
        config = load_config(verbose=bool(verbose))

        # Merge configuration with CLI arguments (CLI takes precedence)
        # Only use config values if CLI argument is None (not provided)
        length = length if length is not None else config.get_field_value("story", "length")
        age_range = age_range if age_range is not None else config.get_field_value("story", "age_range")
        style = style if style is not None else config.get_field_value("story", "style")
        tone = tone if tone is not None else config.get_field_value("story", "tone")
        theme = theme if theme is not None else config.get_field_value("story", "theme")
        learning_focus = (
            learning_focus
            if learning_focus is not None
            else (config.get_field_value("story", "learning_focus") or None)
        )
        setting = setting if setting is not None else (config.get_field_value("story", "setting") or None)

        # Handle characters - merge config and CLI
        if characters is None:
            config_characters = config.get_field_value("story", "characters")
            characters = config_characters

        image_style = image_style if image_style is not None else config.get_field_value("images", "image_style")
        output_dir = output_dir if output_dir is not None else (config.get_field_value("output", "output_dir") or None)
        use_context = use_context if use_context is not None else config.get_field_value("output", "use_context")
        verbose = verbose if verbose is not None else config.get_field_value("system", "verbose")
        debug = debug if debug is not None else config.get_field_value("system", "debug")

        # Get backend from CLI or configuration (CLI takes precedence)
        config_backend = backend if backend is not None else config.get_field_value("system", "backend")

    except ConfigError as e:
        console.print(f"[red]Configuration Error:[/red] {e}", style="bold")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Error loading configuration:[/red] {e}", style="bold")
        raise typer.Exit(1) from None

    if debug:
        verbose = True  # Ensure verbose is enabled in debug mode

    # Check if prompt is required (not needed for --continue)
    if not continue_session and (prompt is None or not str(prompt).strip()):
        # Check if user provided no arguments at all - if so, show help instead of error
        import sys

        # Get command line args, excluding the script name
        args = sys.argv[1:]

        # If no arguments provided, show help
        if not args:
            # Use the context to show help
            console.print(ctx.get_help())
            raise typer.Exit(0)

        console.print("[red]Error:[/red] Please provide a non-empty story prompt.", style="bold")
        raise typer.Exit(1)

    # Generate output directory if not provided
    if output_dir is None:
        output_dir = generate_default_output_dir()
        console.print(f"[bold blue]📁 Generated output directory:[/bold blue] {output_dir}")

    # Use phase executor for both new sessions and checkpoint resumption
    try:
        # Execute new session via MCP client
        client = get_client()

        try:
            # Start story generation
            result = run_sync(
                client.generate_story(
                    prompt=prompt or "",
                    backend=str(config_backend) if config_backend is not None else None,
                    length=length,
                    age_range=age_range,
                    style=style,
                    tone=tone,
                    theme=theme,
                    learning_focus=learning_focus,
                    setting=setting,
                    characters=characters,
                    image_style=image_style,
                    output_directory=output_dir,
                    use_context=use_context,
                )
            )

            session_id = result["session_id"]
            console.print(f"\n[green]✓[/green] Story generation started (Session: {session_id})\n")

            # Poll for completion with progress display
            final_status = poll_session_until_complete(session_id, client)

            # Display results
            if final_status["status"] == "completed":
                display_success("Story generation completed successfully!")

                # Show output files
                story_file = final_status.get("story_file")
                if story_file:
                    console.print(f"\n[bold cyan]Story saved to:[/bold cyan] {story_file}")

                images = final_status.get("images", [])
                if images:
                    console.print(f"\n[bold cyan]Generated {len(images)} images:[/bold cyan]")
                    for img in images:
                        console.print(f"  • {img}")

            elif final_status["status"] == "failed":
                error_msg = final_status.get("error", "Unknown error")
                display_error(Exception(error_msg))
                raise typer.Exit(1)

        finally:
            # Disconnect client
            run_sync(client.disconnect())

    except Exception as e:
        if verbose:
            # Sanitize the error message to prevent binary data corruption
            error_msg = str(e)
            # Replace any non-printable characters with a placeholder
            sanitized_error = "".join(c if c.isprintable() or c.isspace() else "?" for c in error_msg)
            console.print(f"[red]Unexpected error:[/red] {sanitized_error}", style="bold")
        else:
            console.print(
                "[red]Error:[/red] An unexpected error occurred. Use --verbose for details.",
                style="bold",
            )
        raise typer.Exit(1) from e


@app.command(
    "continue",
    help="Resume execution from a previous StoryForge session. "
    "Displays the last 5 sessions and allows you to choose which phase to resume from. "
    "Same as 'storyforge main --continue'.",
)
def continue_session():
    """Resume execution from a previous checkpoint session."""
    from .client import get_client, run_sync
    from .client.formatters import display_error, poll_session_until_complete

    try:
        client = get_client()

        # List recent sessions
        console.print("\n[bold cyan]📋 Recent Sessions:[/bold cyan]\n")
        sessions = run_sync(client.list_sessions())

        if not sessions:
            console.print("[yellow]No sessions found.[/yellow]")
            raise typer.Exit(0)

        # Display last 5 sessions
        recent_sessions = sessions[:5]
        for idx, session in enumerate(recent_sessions, 1):
            session_id = session["session_id"]
            status = session.get("status", "unknown")
            phase = session.get("current_phase", "unknown")
            created = session.get("created_at", "unknown")

            console.print(f"[bold]{idx}.[/bold] {session_id}")
            console.print(f"   Status: {status}")
            console.print(f"   Phase: {phase}")
            console.print(f"   Created: {created}")
            if "prompt" in session:
                prompt_preview = session["prompt"][:60]
                console.print(f"   Prompt: {prompt_preview}...")
            console.print()

        # Get user selection
        selection = typer.prompt(f"Select session (1-{len(recent_sessions)})", type=int)

        if selection < 1 or selection > len(recent_sessions):
            console.print("[red]Invalid selection[/red]")
            raise typer.Exit(1)

        selected_session = recent_sessions[selection - 1]
        session_id = selected_session["session_id"]

        # Get current status
        status_info = run_sync(client.get_session_status(session_id))

        # Show available phases to resume from
        console.print(f"\n[bold cyan]Session: {session_id}[/bold cyan]")
        console.print(f"Current status: {status_info.get('status')}")
        console.print(f"Current phase: {status_info.get('current_phase')}")

        # Ask for confirmation
        if not typer.confirm("\nResume this session?"):
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

        # Continue the session
        console.print(f"\n[bold green]Resuming session {session_id}...[/bold green]\n")
        run_sync(client.continue_session(session_id))

        # Poll for completion
        final_status = poll_session_until_complete(session_id, client)

        # Display result
        if final_status["status"] == "completed":
            console.print("\n[bold green]✅ Session resumed successfully![/bold green]")
            if "output_directory" in final_status:
                console.print(f"Output: {final_status['output_directory']}")
        else:
            console.print(f"\n[yellow]Session status: {final_status['status']}[/yellow]")

        raise typer.Exit(0)

    except typer.Exit:
        # Let typer.Exit propagate normally
        raise
    except Exception as e:
        display_error(e)
        raise typer.Exit(1) from e


@app.command("extend", help="Extend an existing story with a continuation")
def extend_story(
    backend: Annotated[str | None, typer.Option(help="LLM backend to use")] = None,
    verbose: Annotated[bool, typer.Option(help="Enable verbose output")] = False,
    debug: Annotated[bool, typer.Option(help="Enable debug mode")] = False,
):
    """Extend an existing story from saved context files."""
    from .client import get_client, run_sync
    from .client.formatters import display_error, poll_session_until_complete

    try:
        client = get_client()

        # List available stories
        console.print("\n[bold cyan]📚 Available Stories to Extend:[/bold cyan]\n")
        stories = run_sync(client.list_extendable_stories())

        if not stories:
            console.print("[yellow]No saved stories found to extend.[/yellow]")
            console.print("Generate a story first with: [bold]sf generate --use-context[/bold]")
            console.print("Or save an existing story as context.")
            raise typer.Exit(1)

        # Display stories
        for idx, story in enumerate(stories, 1):
            console.print(f"[bold]{idx}.[/bold] {story['filename']}")
            console.print(f"   [dim]Generated: {story.get('timestamp', 'Unknown')}[/dim]")
            if "characters" in story:
                console.print(f"   Characters: {story['characters']}")
            if "theme" in story:
                console.print(f"   Theme: {story['theme']}")
            if "preview" in story:
                preview = story["preview"][:100]
                console.print(f"   [dim]Preview: {preview}...[/dim]")
            console.print()

        # Get user selection
        selection = typer.prompt(f"Select story (1-{len(stories)})", type=int)

        if selection < 1 or selection > len(stories):
            console.print("[red]Invalid selection[/red]")
            raise typer.Exit(1)

        selected_story = stories[selection - 1]
        context_file = selected_story["filepath"]

        # Show story preview
        console.print("\n[bold cyan]📖 Story Preview:[/bold cyan]")
        preview_text = selected_story.get("preview", "")[:200]
        console.print(f"[dim]{preview_text}...[/dim]\n")

        # Ask if user wants to see the full story
        view_full = Confirm.ask("📜 View full story before extending?", default=False)

        if view_full:
            from rich.markdown import Markdown
            from rich.panel import Panel

            full_content = selected_story.get("content", "")
            if full_content:
                console.print("\n")
                full_story_panel = Panel(
                    Markdown(full_content.strip()),
                    title=f"[bold cyan]Complete Story: {selected_story['filename']}[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                )
                console.print(full_story_panel)
                console.print()

        # Ask continuation preference
        console.print("[bold cyan]🎬 How should this story continue?[/bold cyan]")
        console.print("1. Wrap up the story (resolution, complete ending)")
        console.print("2. Leave a cliffhanger (setup for next adventure)")

        ending_choice = typer.prompt("\nChoice (1-2)", type=int)

        if ending_choice not in [1, 2]:
            console.print("[red]Invalid choice[/red]")
            raise typer.Exit(1)

        ending_type_str = "wrap_up" if ending_choice == 1 else "cliffhanger"
        ending_type = cast(Literal["wrap_up", "cliffhanger"], ending_type_str)

        console.print(
            f"\n[bold green]✨ Generating continuation with {ending_type.replace('_', ' ')} ending...[/bold green]\n"
        )

        # Generate output directory with _extended suffix
        output_dir = generate_default_output_dir(extended=True)

        # Start extension
        result = run_sync(
            client.extend_story(context_file=context_file, ending_type=ending_type, output_directory=output_dir)
        )

        session_id = result["session_id"]

        # Poll for completion
        final_status = poll_session_until_complete(session_id, client)

        # Display result
        if final_status["status"] == "completed":
            console.print("\n[bold green]✅ Story extended successfully![/bold green]")
            if "output_directory" in final_status:
                console.print(f"Output: {final_status['output_directory']}")
        else:
            console.print(f"\n[yellow]Extension status: {final_status['status']}[/yellow]")

        raise typer.Exit(0)

    except typer.Exit:
        # Let typer.Exit propagate normally
        raise
    except Exception as e:
        display_error(e)
        raise typer.Exit(1) from e


# =============================================================================
# EXPORT CHAIN COMMAND
# =============================================================================


@app.command()
def export_chain(
    context: Annotated[
        str | None,
        typer.Option("--context", "-c", help="Name or path of context file to export (if not provided, will prompt)"),
    ] = None,
    output: Annotated[
        str | None, typer.Option("--output", "-o", help="Output file path (default: complete_story_TIMESTAMP.txt)")
    ] = None,
    config_file: Annotated[str | None, typer.Option("--config", help="Path to configuration file")] = None,
) -> None:
    """
    Export a complete story chain to a single file.

    This command reconstructs the full lineage of an extended story by tracing
    back through all parent stories, then combines them into a single file
    in chronological order.

    Examples:
        # Export with interactive selection (shows only chains)
        sf export-chain

        # Export specific context
        sf export-chain -c wizard_story_extended

        # Export to specific file
        sf export-chain -c wizard_story -o my_complete_story.txt
    """
    try:
        console.print("[bold cyan]📚 Export Story Chain[/bold cyan]\n")

        # Create context manager
        context_mgr = ContextManager()

        # Get available contexts
        available_contexts = context_mgr.list_available_contexts()
        if not available_contexts:
            console.print("[yellow]No context files found. Generate and save a story first.[/yellow]")
            raise typer.Exit(1)

        # Select context file
        if context:
            # Find matching context
            matches = [ctx for ctx in available_contexts if context.lower() in ctx["filename"].lower()]
            if not matches:
                console.print(f"[red]No context file matching '{context}' found.[/red]")
                raise typer.Exit(1)
            selected_context = matches[0]
        else:
            # Interactive selection - filter to only extended stories
            contexts_with_chains: list[dict[str, Any]] = []
            for ctx in available_contexts:
                chain = context_mgr.get_story_chain(ctx["filepath"])
                if len(chain) > 1:  # Only include chains (2+ parts)
                    contexts_with_chains.append({"context": ctx, "chain": chain})

            if not contexts_with_chains:
                console.print("[yellow]No extended story chains found. Only single stories available.[/yellow]")
                console.print("[dim]Extend a story first with 'sf extend' to create a chain.[/dim]")
                raise typer.Exit(1)

            # Display chains with lineage
            console.print("[bold]Available story chains to export:[/bold]\n")
            for idx, ctx_chain in enumerate(contexts_with_chains, 1):
                story_ctx: dict[str, Any] = ctx_chain["context"]
                story_chain: list[dict[str, Any]] = ctx_chain["chain"]
                console.print(f"{idx}. [cyan]{story_ctx['filename']}[/cyan]")
                console.print(f"   [bold yellow]📚 Chain: {len(story_chain)} parts[/bold yellow]")
                for chain_idx, story in enumerate(story_chain, 1):
                    label = " (original)" if chain_idx == 1 else (" (latest)" if chain_idx == len(story_chain) else "")
                    console.print(f"      └─ Part {chain_idx}: {story['filename']}{label}")
                if "prompt" in story_ctx:
                    console.print(f"   Prompt: {story_ctx['prompt'][:60]}...")
                console.print()

            selection = typer.prompt("\nSelect a story chain to export", type=int)
            if selection < 1 or selection > len(contexts_with_chains):
                console.print("[red]Invalid selection.[/red]")
                raise typer.Exit(1)

            selected_context = contexts_with_chains[selection - 1]["context"]

        # Load and preview the complete chain
        story_chain = context_mgr.get_story_chain(selected_context["filepath"])

        # Show chain preview
        console.print("\n[bold cyan]📖 Chain Preview:[/bold cyan]")
        console.print(f"Total parts: {len(story_chain)}")

        # Show preview of each part (first 50 words)
        for idx, story_meta in enumerate(story_chain, 1):
            story_path = story_meta.get("filepath")
            if story_path and Path(story_path).exists():
                with open(story_path, encoding="utf-8") as f:
                    content = f.read()
                    # Extract story content
                    story_text = content
                    if "## Story" in content:
                        story_text = content.split("## Story", 1)[1]
                    preview = " ".join(story_text.split()[:50])
                    console.print(f"\n[bold]Part {idx}:[/bold] {story_meta['filename']}")
                    console.print(f"[dim]{preview}...[/dim]")

        console.print()

        # Ask if user wants to see the full chain
        view_full = Confirm.ask("📜 View complete chain before exporting?", default=False)

        if view_full:
            from rich.markdown import Markdown
            from rich.panel import Panel

            console.print("\n")

            # Display each part in the chain
            for idx, story_meta in enumerate(story_chain, 1):
                story_path = story_meta.get("filepath")
                if story_path and Path(story_path).exists():
                    with open(story_path, encoding="utf-8") as f:
                        content = f.read()
                        # Extract story content
                        story_text = content
                        if "## Story" in content:
                            story_text = content.split("## Story", 1)[1]

                        # Create panel for this part
                        part_label = " (original)" if idx == 1 else (" (latest)" if idx == len(story_chain) else "")
                        title = (
                            f"[bold cyan]Part {idx} of {len(story_chain)}: "
                            f"{story_meta['filename']}{part_label}[/bold cyan]"
                        )
                        panel = Panel(
                            Markdown(story_text.strip()),
                            title=title,
                            border_style="cyan",
                            padding=(1, 2),
                        )
                        console.print(panel)
                        if idx < len(story_chain):
                            console.print()  # Add spacing between parts

            console.print()

        # Determine output path
        if output:
            output_path = Path(output)
        else:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = selected_context["filename"].replace("_context", "").replace(".md", "")
            output_path = Path(f"complete_story_{base_name}_{timestamp}.txt")

        # Export the chain
        try:
            result_path = context_mgr.write_chain_to_file(selected_context["filepath"], output_path)
            story_chain = context_mgr.get_story_chain(selected_context["filepath"])
            console.print(f"\n[bold green]✓ Exported to:[/bold green] {result_path}")
            console.print(f"[dim]Total parts: {len(story_chain)}[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from e

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    import sys

    # If first argument doesn't look like a subcommand or flag, assume it's a prompt for main command
    if (
        len(sys.argv) > 1
        and not sys.argv[1].startswith("-")
        and sys.argv[1]
        not in [
            "main",
            "config",
            "continue",
            "extend",
            "export-chain",
        ]
    ):
        sys.argv.insert(1, "main")
    app()


def cli_entry() -> None:
    """Entry point for the CLI that handles default command routing."""
    import sys

    # Show help if no arguments provided
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    # If first argument doesn't look like a subcommand or flag, assume it's a prompt for main command
    elif not sys.argv[1].startswith("-") and sys.argv[1] not in [
        "main",
        "config",
        "continue",
        "extend",
        "export-chain",
    ]:
        sys.argv.insert(1, "main")
    app()
