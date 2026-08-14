"""Public command-line entry point and TUI/CLI routing."""

import argparse
import os
import sys
from collections.abc import Sequence
from typing import Any

from .StoryForge import app

COMMANDS = frozenset({"generate", "continue", "extend", "export-chain", "config", "world", "models"})
# Do not reinterpret removed command names as story prompts. This lets Typer
# return its normal "No such command" error instead of silently accepting a
# legacy invocation.
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


def normalize_argv(argv: Sequence[str]) -> list[str]:
    """Route a bare prompt to ``generate`` without modifying process arguments."""
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


def _generation_request(argv: Sequence[str]):
    """Build a TUI prefill request without replacing Typer's validation contract."""
    from .mcp_models import GenerationRequest

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--length", "-l")
    parser.add_argument("--age-range", "-a")
    parser.add_argument("--style", "-s")
    parser.add_argument("--tone", "-t")
    parser.add_argument("--voice")
    parser.add_argument("--theme")
    parser.add_argument("--learning-focus")
    parser.add_argument("--setting")
    parser.add_argument("--character", action="append", dest="characters")
    parser.add_argument("--image-style")
    parser.add_argument("--image-count", "-n", type=int)
    parser.add_argument("--output-dir", "-o")
    parser.add_argument("--world-file")
    parser.add_argument("--backend")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use-context", action="store_true", default=None)
    parser.add_argument("--no-use-context", action="store_false", dest="use_context")
    values, _unknown = parser.parse_known_args(list(argv))
    data: dict[str, Any] = vars(values)
    data["prompt"] = data.get("prompt") or " "
    return GenerationRequest(**data)


def _tui_route(args: list[str]) -> tuple[str, Any | None]:
    """Resolve a command line into a direct TUI route and optional form values."""
    if not args:
        return "home", None
    if args[0] not in COMMANDS and not args[0].startswith("-"):
        return "generate", _generation_request(args)
    route = TUI_ROUTES.get(args[0], "home")
    if route == "generate" and "--continue" in args[1:]:
        return "continue", None
    initial = _generation_request(args[1:]) if route == "generate" else None
    return route, initial


def main(argv: Sequence[str] | None = None) -> None:
    """Run StoryForge from a console script or ``python -m storyforge``."""
    args = list(sys.argv[1:] if argv is None else argv)
    force_tui = "--tui" in args
    no_tui = "--no-tui" in args
    args = [arg for arg in args if arg not in {"--tui", "--no-tui"}]

    local_only = any(arg in {"--help", "-h", "--version", "--install-completion", "--show-completion"} for arg in args)
    if not local_only and not no_tui and (force_tui or terminal_supports_tui()):
        from .tui import run_tui

        route, initial = _tui_route(args)
        run_tui(route, initial)
        return

    # Programmatic callers retain Typer's direct adapter. The installed console
    # entrypoints use the MCP-backed fallback for actual operations.
    if argv is None and args and not local_only:
        from .classic_cli import run_classic
        from .console import console

        route, initial = _tui_route(args)
        generation_request = initial if route == "generate" else None
        try:
            raise SystemExit(run_classic(args, generation_request))
        except KeyboardInterrupt:
            console.print("[yellow]StoryForge operation cancelled.[/yellow]")
            raise SystemExit(130) from None
        except SystemExit:
            raise
        except Exception as error:
            console.print(f"[bold red]StoryForge MCP error:[/bold red] {error}")
            raise SystemExit(1) from None

    app(args=normalize_argv(args))
