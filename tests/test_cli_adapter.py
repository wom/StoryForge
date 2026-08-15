"""Tests for public CLI argument routing."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from storyforge.cli import build_parser, main, normalize_argv, terminal_supports_tui

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_module_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the public ``python -m storyforge`` surface without generation."""
    return subprocess.run(
        [sys.executable, "-m", "storyforge", *args],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )


def test_normalize_argv_routes_bare_prompt_to_generate():
    assert normalize_argv(["A brave mouse"]) == ["generate", "A brave mouse"]


def test_normalize_argv_preserves_commands_and_options():
    assert normalize_argv(["generate", "A brave mouse"]) == ["generate", "A brave mouse"]
    assert normalize_argv(["continue"]) == ["continue"]
    assert normalize_argv(["main", "A brave mouse"]) == ["main", "A brave mouse"]
    assert normalize_argv(["--help"]) == ["--help"]


def test_normalize_argv_shows_help_without_arguments():
    assert normalize_argv([]) == ["--help"]


def test_main_passes_normalized_copy_to_classic_client_without_mutating_input():
    arguments = ["A brave mouse"]
    with (
        patch("storyforge.cli.terminal_supports_tui", return_value=False),
        patch("storyforge.classic_cli.run_classic", return_value=0) as run_classic,
        pytest.raises(SystemExit) as error,
    ):
        main(arguments)

    assert arguments == ["A brave mouse"]
    assert error.value.code == 0
    request = run_classic.call_args.args[1]
    assert run_classic.call_args.args[0] == ["generate", "A brave mouse"]
    assert request.prompt == "A brave mouse"


def test_main_builds_generation_request_from_schema_options():
    with (
        patch("storyforge.cli.terminal_supports_tui", return_value=False),
        patch("storyforge.classic_cli.run_classic", return_value=0) as run_classic,
        pytest.raises(SystemExit),
    ):
        main(["generate", "A brave mouse", "-w", "world.md", "-v", "--character", "Max"])

    request = run_classic.call_args.args[1]
    assert request.world_file == "world.md"
    assert request.verbose is True
    assert request.characters == ["Max"]


def test_force_tui_opens_home_route():
    with patch("storyforge.tui.run_tui") as run_tui:
        main(["--tui"])

    run_tui.assert_called_once_with("home", None)


def test_generation_parser_rejects_unknown_options():
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["generate", "A brave mouse", "--tonne", "gentle"])

    assert error.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["continue"],
        ["extend"],
        ["export-chain", "--context", "story", "--output", "chain.txt"],
        ["config", "init", "--path", "storyforge.ini", "--force"],
        ["world", "init", "--force"],
        ["models", "refresh"],
    ],
)
def test_parser_accepts_supported_command_surfaces(arguments):
    assert build_parser().parse_args(arguments).command == arguments[0]


def test_parser_rejects_unknown_subcommand_options():
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["config", "init", "--froce"])

    assert error.value.code == 2


def test_parser_rejects_out_of_range_image_count():
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["generate", "A story", "--image-count", "6"])

    assert error.value.code == 2


def test_parser_rejects_abbreviated_options():
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["generate", "A story", "--verb"])

    assert error.value.code == 2


def test_classic_generate_continue_routes_to_continue_command():
    with (
        patch("storyforge.cli.sys.argv", ["sf", "generate", "--continue"]),
        patch("storyforge.cli.terminal_supports_tui", return_value=False),
        patch("storyforge.classic_cli.run_classic", return_value=0) as run_classic,
        pytest.raises(SystemExit) as error,
    ):
        main()

    assert error.value.code == 0
    run_classic.assert_called_once_with(["continue"], None)


def test_terminal_capability_rejects_redirected_streams():
    with patch("storyforge.cli.sys.stdin.isatty", return_value=False):
        assert terminal_supports_tui() is False


def test_module_entrypoint_lists_generate_command():
    result = run_module_cli("--help")

    assert result.returncode == 0, result.stderr
    assert "generate" in result.stdout


def test_module_entrypoint_routes_bare_prompt_to_generate():
    result = run_module_cli("A brave mouse", "--help")

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "storyforge generate" in result.stdout


def test_module_entrypoint_supports_explicit_generate_command():
    result = run_module_cli("generate", "--help")

    assert result.returncode == 0, result.stderr
    assert "storyforge generate" in result.stdout
