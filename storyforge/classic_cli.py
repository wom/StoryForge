"""Rich/Typer-compatible fallback implemented as an MCP client."""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, cast

from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt

from .console import console
from .mcp_client import StoryForgeMCPClient
from .mcp_models import ExportRequest, ExtensionRequest, FinalizeRequest, GenerationRequest, RefinementRequest


class ClassicCLI:
    """Script-friendly presentation over the bundled MCP server."""

    def __init__(self) -> None:
        self._last_progress = ""

    def _progress(self, progress: float, total: float | None, message: str | None) -> None:
        if message and message != self._last_progress:
            self._last_progress = message
            suffix = f" ({round(progress / total * 100)}%)" if total else ""
            console.print(f"[dim]{message.replace('_', ' ').title()}{suffix}[/dim]")

    async def run(self, args: list[str], generation_request: GenerationRequest | None = None) -> int:
        command = (
            args[0]
            if args
            and args[0]
            in {
                "generate",
                "continue",
                "extend",
                "export-chain",
                "config",
                "world",
                "models",
            }
            else "generate"
        )
        async with StoryForgeMCPClient(progress_callback=self._progress) as client:
            if command == "generate":
                if generation_request is None:
                    raise ValueError("A story prompt is required")
                return await self._generate(client, generation_request)
            if command == "continue":
                return await self._continue(client)
            if command == "extend":
                return await self._extend(client)
            if command == "export-chain":
                return await self._export(client, args[1:])
            if command == "config":
                return await self._config(client, args[1:])
            if command == "world":
                return await self._world(client, args[1:])
            return await self._models(client, args[1:])

    async def _generate(self, client: StoryForgeMCPClient, request: GenerationRequest) -> int:
        console.print(Panel.fit(request.prompt, title="[bold cyan]Story Generation[/bold cyan]", border_style="cyan"))
        if not Confirm.ask("Proceed with story generation?", default=True):
            return 0
        draft = await client.create_draft(request)
        while True:
            console.print(Panel(draft.story, title="[bold green]Generated Story[/bold green]", border_style="green"))
            if not Confirm.ask("Would you like to refine the story?", default=False):
                break
            instructions = Prompt.ask("Refinements").strip()
            if instructions:
                draft = await client.refine_draft(
                    RefinementRequest(session_id=draft.session_id, instructions=instructions)
                )
        video_count = 0
        if Confirm.ask("Generate a video prompt?", default=False):
            video_count = IntPrompt.ask("Number of scenes", default=2)
        image_count = 0
        if Confirm.ask("Generate illustrations?", default=False):
            image_count = IntPrompt.ask("Number of images", default=3)
        save_context = Confirm.ask("Save this story as future context?", default=False)
        result = await client.finalize_story(
            FinalizeRequest(
                session_id=draft.session_id,
                video_scene_count=video_count,
                image_count=image_count,
                save_context=save_context,
            )
        )
        self._print_result(result.message, result.artifacts)
        return 0

    async def _continue(self, client: StoryForgeMCPClient) -> int:
        sessions = await client.list_sessions()
        if not sessions:
            console.print("[yellow]No previous StoryForge sessions found.[/yellow]")
            return 1
        for index, session in enumerate(sessions, 1):
            console.print(
                f"{index}. {session.prompt_preview} "
                f"[dim]({session.status} at {session.current_phase}, {session.completion_percentage}%)[/dim]"
            )
        choice = IntPrompt.ask("Select session", choices=[str(index) for index in range(1, len(sessions) + 1)])
        draft = await client.resume_session(sessions[choice - 1].session_id)
        console.print(Panel(draft.story, title="[bold cyan]Resumed Story[/bold cyan]", border_style="cyan"))
        result = await client.finalize_story(FinalizeRequest(session_id=draft.session_id))
        self._print_result(result.message, result.artifacts)
        return 0

    async def _extend(self, client: StoryForgeMCPClient) -> int:
        stories = await client.list_stories()
        if not stories:
            console.print("[yellow]No saved stories found to extend.[/yellow]")
            return 1
        for index, story in enumerate(stories, 1):
            console.print(f"{index}. [cyan]{story.filename}[/cyan] [dim]{story.preview[:80]}[/dim]")
        choice = IntPrompt.ask("Select story", choices=[str(index) for index in range(1, len(stories) + 1)])
        ending = cast(
            Literal["wrap_up", "cliffhanger"],
            Prompt.ask("Ending", choices=["cliffhanger", "wrap_up"], default="cliffhanger"),
        )
        direction = Prompt.ask("Continuation direction (optional)", default="").strip() or None
        draft = await client.create_extension_draft(
            ExtensionRequest(story_id=stories[choice - 1].id, ending_type=ending, direction=direction)
        )
        console.print(Panel(draft.story, title="[bold cyan]Continuation Draft[/bold cyan]", border_style="cyan"))
        result = await client.finalize_story(
            FinalizeRequest(
                session_id=draft.session_id,
                save_context=Confirm.ask("Save continuation as context?", default=True),
            )
        )
        self._print_result(result.message, result.artifacts)
        return 0

    async def _export(self, client: StoryForgeMCPClient, args: list[str]) -> int:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--context", "-c")
        parser.add_argument("--output", "-o")
        options, _ = parser.parse_known_args(args)
        stories = await client.list_stories(chain_only=True)
        if not stories:
            console.print("[yellow]No extended story chains found.[/yellow]")
            return 1
        selected = next(
            (story for story in stories if options.context and options.context.lower() in story.filename.lower()),
            None,
        )
        if selected is None:
            for index, story in enumerate(stories, 1):
                console.print(f"{index}. {story.filename} [dim]({story.chain_length} parts)[/dim]")
            choice = IntPrompt.ask("Select chain", choices=[str(index) for index in range(1, len(stories) + 1)])
            selected = stories[choice - 1]
        result = await client.export_chain(ExportRequest(story_id=selected.id, output=options.output))
        self._print_result(result.message, result.artifacts)
        return 0

    async def _config(self, client: StoryForgeMCPClient, args: list[str]) -> int:
        action = args[0] if args else "show"
        if action == "init":
            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument("--path", "-p")
            parser.add_argument("--force", "-f", action="store_true")
            options, _ = parser.parse_known_args(args[1:])
            result = await client.init_config(options.path, options.force)
            self._print_result(result.message, result.artifacts)
        else:
            console.print(await client.get_config())
        return 0

    async def _world(self, client: StoryForgeMCPClient, args: list[str]) -> int:
        action = args[0] if args else "show"
        world = await client.read_world()
        if action == "path":
            console.print(world.get("path", ""))
        elif action == "show":
            console.print(Panel(str(world.get("content", "")), title="Story World", border_style="cyan"))
        elif action == "init":
            from .world_template import WORLD_TEMPLATE

            result = await client.write_world(WORLD_TEMPLATE, overwrite="--force" in args)
            console.print(f"[green]World file created:[/green] {result.get('path', '')}")
        elif action == "edit":
            content = str(world.get("content", ""))
            with tempfile.NamedTemporaryFile("w+", suffix=".md", encoding="utf-8", delete=False) as temporary:
                temporary.write(content)
                temporary_path = Path(temporary.name)
            try:
                editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "nano"))
                subprocess.run([*shlex.split(editor), str(temporary_path)], check=True)  # noqa: S603
                await client.write_world(temporary_path.read_text(encoding="utf-8"), overwrite=True)
            finally:
                temporary_path.unlink(missing_ok=True)
        return 0

    async def _models(self, client: StoryForgeMCPClient, args: list[str]) -> int:
        action = args[0] if args else "list"
        if action == "refresh":
            result = await client.invalidate_models()
            console.print(f"[green]{result.message}[/green]")
        elif action == "clear":
            result = await client.clear_models(confirmed=True)
            console.print(f"[green]{result.message}[/green]")
        else:
            models = await client.list_models()
            for backend, entries in models.items():
                console.print(f"[bold]{backend}[/bold] ({len(entries)} cached)")
                for entry in entries:
                    console.print(f"  • {entry.get('name', entry.get('id', 'unknown'))}")
        return 0

    @staticmethod
    def _print_result(message: str, artifacts: list[str]) -> None:
        console.print(f"[bold green]{message}[/bold green]")
        for artifact in artifacts:
            console.print(f"  • {artifact}")


def run_classic(args: list[str], generation_request: GenerationRequest | None = None) -> int:
    """Run the MCP-backed classic interface and return an exit code."""
    return asyncio.run(ClassicCLI().run(args, generation_request))
