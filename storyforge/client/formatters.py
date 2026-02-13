"""Rich display formatters and helpers for StoryForge CLI."""

import time
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .mcp_client import MCPClient, run_sync

console = Console()


def poll_session_until_complete(session_id: str, client: MCPClient) -> dict[str, Any]:
    """
    Poll session status until generation is complete.

    Args:
        session_id: Session ID to monitor
        client: MCP client instance

    Returns:
        Final session status dict

    Raises:
        Exception: If generation fails
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Initializing...", total=100)

        while True:
            status = run_sync(client.get_session_status(session_id))

            # Debug: check what we got
            if not isinstance(status, dict):
                raise Exception(f"get_session_status returned non-dict: {type(status)} = {repr(status)}")
            if "status" not in status:
                raise Exception(f"get_session_status returned dict without 'status' key: {status}")

            # Check if complete or failed
            if status["status"] == "completed":
                progress.update(task, description="[bold green]✅ Generation complete!", completed=100)
                break
            elif status["status"] == "failed":
                error_msg = status.get("error_message", "Unknown error")
                progress.update(task, description=f"[bold red]❌ Failed: {error_msg}")
                raise Exception(f"Story generation failed: {error_msg}")

            # Update progress
            phase = status.get("current_phase", "unknown")
            progress_pct = status.get("progress_percent", 0)
            progress.update(task, description=f"{phase}", completed=progress_pct)

            # Check queue status if waiting
            if status["status"] == "queued":
                queue_status = run_sync(client.get_queue_status())
                queue_pos = queue_status.get("queue_length", 0)
                if queue_pos > 0:
                    progress.update(task, description=f"⏳ Queued (position {queue_pos})")

            # Sleep before next poll
            time.sleep(3)

    return status


def format_session_list(sessions: list[dict[str, Any]]) -> Table:
    """
    Format session list as a Rich table.

    Args:
        sessions: List of session dicts

    Returns:
        Rich Table object
    """
    table = Table(title="StoryForge Sessions", show_header=True, header_style="bold magenta")

    table.add_column("Session ID", style="cyan", no_wrap=True, width=12)
    table.add_column("Prompt", style="white", width=40)
    table.add_column("Status", style="yellow", width=12)
    table.add_column("Phase", style="blue", width=15)
    table.add_column("Progress", style="green", width=8)
    table.add_column("Created", style="dim", width=16)

    for session in sessions:
        session_id = session.get("session_id", "")[:12]
        prompt = session.get("prompt", "")[:40]
        status = session.get("status", "unknown")
        phase = session.get("current_phase", "N/A")
        progress = f"{session.get('progress_percent', 0)}%"
        created = session.get("created_at", "")[:16]

        # Color-code status
        if status == "completed":
            status = f"[green]{status}[/green]"
        elif status == "failed":
            status = f"[red]{status}[/red]"
        elif status == "running":
            status = f"[yellow]{status}[/yellow]"

        table.add_row(session_id, prompt, status, phase, progress, created)

    return table


def format_backend_list(backends: list[dict[str, Any]]) -> Table:
    """
    Format backend list as a Rich table.

    Args:
        backends: List of backend dicts

    Returns:
        Rich Table object
    """
    table = Table(title="Available LLM Backends", show_header=True, header_style="bold magenta")

    table.add_column("Backend", style="cyan", no_wrap=True)
    table.add_column("Available", style="yellow", width=10)
    table.add_column("API Key", style="blue", width=10)
    table.add_column("Story Gen", style="green", width=10)
    table.add_column("Image Gen", style="green", width=10)
    table.add_column("Status", style="dim", width=30)

    for backend in backends:
        name = backend.get("name", "unknown")
        available = "✅ Yes" if backend.get("available") else "❌ No"
        api_key = "✅ Set" if backend.get("api_key_set") else "❌ Not Set"
        capabilities = backend.get("capabilities", {})
        story_gen = "✅" if capabilities.get("story_generation") else "❌"
        image_gen = "✅" if capabilities.get("image_generation") else "❌"
        reason = backend.get("reason", "")

        table.add_row(name, available, api_key, story_gen, image_gen, reason)

    return table


def format_story_chain(chain: list[dict[str, Any]]) -> None:
    """
    Display story chain with previews.

    Args:
        chain: List of chain part dicts
    """
    console.print("\n[bold cyan]📚 Story Chain[/bold cyan]\n")

    for part in chain:
        part_num = part.get("part_number", 0)
        preview = part.get("preview", "")
        metadata = part.get("metadata", {})
        prompt = metadata.get("prompt", "Unknown")

        console.print(f"[bold]Part {part_num}:[/bold] {prompt}")
        console.print(f"[dim]{preview}...[/dim]\n")


def display_error(error: Exception, recovery_hint: str | None = None) -> None:
    """
    Display error message with recovery hint.

    Args:
        error: Exception that occurred
        recovery_hint: Optional hint for recovery
    """
    console.print(f"\n[bold red]❌ Error:[/bold red] {str(error)}")

    if recovery_hint:
        console.print(f"[yellow]💡 Hint:[/yellow] {recovery_hint}\n")


def display_success(message: str, details: dict[str, Any] | None = None) -> None:
    """
    Display success message with optional details.

    Args:
        message: Success message
        details: Optional details dict
    """
    console.print(f"\n[bold green]✅ {message}[/bold green]")

    if details:
        for key, value in details.items():
            console.print(f"  [cyan]{key}:[/cyan] {value}")
    console.print()


def display_session_list(sessions: list[dict[str, Any]]) -> None:
    """
    Display sessions in a formatted table.

    Args:
        sessions: List of session dicts
    """
    table = format_session_list(sessions)
    console.print(table)


def display_backends(backends: list[dict[str, Any]]) -> None:
    """
    Display backends in a formatted table.

    Args:
        backends: List of backend dicts
    """
    table = format_backend_list(backends)
    console.print(table)
