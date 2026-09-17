"""Public command-line entry point and TUI/classic MCP routing."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__
from .mcp_models import GenerationRequest
from .schema import STORYFORGE_SCHEMA, ConfigField

COMMANDS = frozenset({"generate", "continue", "extend", "export-chain", "config", "world", "models"})
RESERVED_COMMANDS = COMMANDS | {"main"}
TUI_ROUTES = {
    "generate": "generate",
    "continue": "continue",
    "extend": "extend",
    "export-chain": "export",
    "config": "config",
    "world": "world",
    "models": "models",
}


class StoryForgeArgumentParser(argparse.ArgumentParser):
    """ArgumentParser variant that never accepts abbreviated options."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)


def normalize_argv(argv: Sequence[str]) -> list[str]:
    """Route a bare prompt to ``generate`` without modifying caller-owned arguments."""
    args = list(argv)
    if not args:
        return ["--help"]
    if args[0].startswith("-") or args[0] in RESERVED_COMMANDS:
        return args
    return ["generate", *args]


def terminal_supports_tui() -> bool:
    """Return whether the current terminal can safely host Textual."""
    return bool(
        sys.stdin.isatty()
        and sys.stdout.isatty()
        and os.environ.get("TERM", "").lower() != "dumb"
        and os.environ.get("STORYFORGE_NO_TUI", "").lower() not in {"1", "true", "yes"}
    )


def _add_schema_argument(
    parser: argparse.ArgumentParser,
    field: ConfigField,
    *flags: str,
    **kwargs: Any,
) -> None:
    option_flags = list(flags) or [value for value in (field.cli_long, field.cli_short) if value]
    parser.add_argument(*option_flags, help=field.cli_help, **kwargs)


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    story = STORYFORGE_SCHEMA.story.fields
    images = STORYFORGE_SCHEMA.images.fields
    output = STORYFORGE_SCHEMA.output.fields
    system = STORYFORGE_SCHEMA.system.fields
    parser.add_argument("prompt", nargs="?", help="Story prompt")
    parser.add_argument("--continue", dest="continue_session", action="store_true", help="Resume a saved session")
    for name in ("length", "age_range", "style", "tone", "voice", "theme", "learning_focus", "setting"):
        _add_schema_argument(parser, story[name])
    _add_schema_argument(parser, story["characters"], "--character", action="append", dest="characters")
    _add_schema_argument(parser, images["image_style"])
    _add_schema_argument(parser, images["image_count"], type=int, choices=range(1, 6), metavar="1-5")
    _add_schema_argument(parser, output["output_dir"])
    _add_schema_argument(parser, output["world_file"])
    _add_schema_argument(parser, system["backend"])
    _add_schema_argument(parser, system["verbose"], action="store_true")
    _add_schema_argument(parser, system["debug"], action="store_true")
    context = parser.add_mutually_exclusive_group()
    context.add_argument("--use-context", action="store_true", default=None, help=output["use_context"].cli_help)
    context.add_argument("--no-use-context", action="store_false", dest="use_context")


def build_parser() -> argparse.ArgumentParser:
    """Build the single strict parser used by both terminal presentations."""
    parser = StoryForgeArgumentParser(
        prog="sf",
        description="Create and continue illustrated stories through the bundled MCP server.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--tui", action="store_true", help="Force the full-screen Textual interface")
    parser.add_argument("--no-tui", action="store_true", help="Force the classic terminal interface")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    generate = subparsers.add_parser("generate", help="Generate a story", description="Generate a new story draft.")
    _add_generation_arguments(generate)
    subparsers.add_parser("continue", help="Resume a checkpoint")
    subparsers.add_parser("extend", help="Continue a saved story")

    export = subparsers.add_parser("export-chain", help="Export a complete story chain")
    export.add_argument("--context", "-c")
    export.add_argument("--output", "-o")

    config = subparsers.add_parser("config", help="Manage configuration")
    config_commands = config.add_subparsers(dest="config_action", metavar="ACTION")
    config_commands.add_parser("show", help="Show resolved configuration")
    config_init = config_commands.add_parser("init", help="Create a configuration file")
    config_init.add_argument("--path", "-p")
    config_init.add_argument("--force", "-f", action="store_true")

    world = subparsers.add_parser("world", help="Manage the story world file")
    world_commands = world.add_subparsers(dest="world_action", metavar="ACTION")
    world_commands.add_parser("show", help="Show the world file")
    world_commands.add_parser("path", help="Show the world file path")
    world_commands.add_parser("edit", help="Edit the world file")
    world_init = world_commands.add_parser("init", help="Create a world file")
    world_init.add_argument("--force", "-f", action="store_true")

    models = subparsers.add_parser("models", help="Manage the model cache")
    model_commands = models.add_subparsers(dest="models_action", metavar="ACTION")
    model_commands.add_parser("list", help="List cached models")
    model_commands.add_parser("refresh", help="Query providers and refresh cached models")
    model_commands.add_parser("clear", help="Clear cached model data")
    return parser


def _request_from_namespace(namespace: argparse.Namespace) -> GenerationRequest:
    data = {
        key: value
        for key, value in vars(namespace).items()
        if key
        in {
            "prompt",
            "length",
            "age_range",
            "style",
            "tone",
            "voice",
            "theme",
            "learning_focus",
            "setting",
            "characters",
            "image_style",
            "image_count",
            "output_dir",
            "world_file",
            "backend",
            "verbose",
            "debug",
            "use_context",
        }
    }
    data["prompt"] = data.get("prompt") or " "
    return GenerationRequest(**data)


def _parse_args(args: Sequence[str]) -> tuple[list[str], argparse.Namespace]:
    normalized = normalize_argv(args)
    return normalized, build_parser().parse_args(normalized)


def _requires_direct_execution(namespace: argparse.Namespace) -> bool:
    """Return whether routing to a general TUI screen would discard an explicit action."""
    action_fields = {
        "config": "config_action",
        "world": "world_action",
        "models": "models_action",
    }
    action_field = action_fields.get(namespace.command)
    if action_field is not None and getattr(namespace, action_field, None) is not None:
        return True
    return bool(
        namespace.command == "export-chain"
        and (getattr(namespace, "context", None) is not None or getattr(namespace, "output", None) is not None)
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run StoryForge from a console script or ``python -m storyforge``."""
    args = list(sys.argv[1:] if argv is None else argv)
    force_tui = "--tui" in args
    no_tui = "--no-tui" in args
    args = [arg for arg in args if arg not in {"--tui", "--no-tui"}]

    if not args and not no_tui and (force_tui or terminal_supports_tui()):
        from .tui import run_tui

        run_tui("home", None)
        return

    local_only = not args or any(arg in {"--help", "-h", "--version"} for arg in args)
    if local_only:
        _parse_args(args)
        return

    normalized, namespace = _parse_args(args)
    route = (
        "continue" if namespace.command == "generate" and namespace.continue_session else TUI_ROUTES[namespace.command]
    )
    generation_request = _request_from_namespace(namespace) if namespace.command == "generate" else None
    direct_execution = _requires_direct_execution(namespace)

    if not direct_execution and not no_tui and (force_tui or terminal_supports_tui()):
        from .tui import run_tui

        run_tui(route, generation_request if route == "generate" else None)
        return

    from .classic_cli import run_classic
    from .console import console

    classic_args = ["continue"] if route == "continue" and namespace.command == "generate" else normalized
    try:
        raise SystemExit(run_classic(classic_args, generation_request if route == "generate" else None))
    except KeyboardInterrupt:
        console.print("[yellow]StoryForge operation cancelled.[/yellow]")
        raise SystemExit(130) from None
    except SystemExit:
        raise
    except Exception as error:
        console.print(f"[bold red]StoryForge MCP error:[/bold red] {error}")
        raise SystemExit(1) from None
