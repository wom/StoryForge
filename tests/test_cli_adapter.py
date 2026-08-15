"""Tests for public CLI argument routing."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from storyforge.cli import _generation_request, main, normalize_argv, terminal_supports_tui

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


def test_main_passes_normalized_copy_without_mutating_input():
    arguments = ["A brave mouse"]
    with patch("storyforge.cli.app") as mock_app:
        main(arguments)

    assert arguments == ["A brave mouse"]
    mock_app.assert_called_once_with(args=["generate", "A brave mouse"])


def test_force_tui_opens_home_route():
    with patch("storyforge.tui.run_tui") as run_tui:
        main(["--tui"])

    run_tui.assert_called_once_with("home", None)


def test_generation_arguments_prefill_tui_request():
    request = _generation_request(["A brave mouse", "--tone", "gentle", "--character", "Max", "--character", "Luna"])

    assert request.prompt == "A brave mouse"
    assert request.tone == "gentle"
    assert request.characters == ["Max", "Luna"]


def test_generation_adapter_supports_advertised_short_options():
    request = _generation_request(["A brave mouse", "-w", "world.md", "-v"])

    assert request.world_file == "world.md"
    assert request.verbose is True


def test_generation_adapter_rejects_unknown_options():
    with pytest.raises(SystemExit) as error:
        _generation_request(["A brave mouse", "--tonne", "gentle"])

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
    assert "Usage:" in result.stdout
    assert "generate [OPTIONS] [prompt]" in result.stdout


def test_module_entrypoint_supports_explicit_generate_command():
    result = run_module_cli("generate", "--help")

    assert result.returncode == 0, result.stderr
    assert "generate [OPTIONS] [prompt]" in result.stdout
