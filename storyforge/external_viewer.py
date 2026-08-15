"""Open generated artifacts with the host platform's default application."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path


def open_path_externally(path: str | Path) -> str:
    """Open an existing file and return a user-facing launcher description."""
    target = Path(path).expanduser().resolve(strict=True)
    if not target.is_file():
        raise FileNotFoundError(f"Image is not a file: {target}")

    if platform.system() == "Windows":  # pragma: no cover - exercised through mocks on non-Windows CI
        startfile = getattr(os, "startfile", None)
        if startfile is None:
            raise RuntimeError("Windows file associations are unavailable")
        startfile(str(target))
        return "the default Windows viewer"

    command, description = _launcher_command(target)
    subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )
    return description


def _launcher_command(target: Path) -> tuple[list[str], str]:
    if _is_wsl():
        explorer = shutil.which("explorer.exe")
        if explorer:
            return [explorer, _windows_path(target)], "Windows Explorer"

    if platform.system() == "Darwin":
        opener = shutil.which("open")
        if opener:
            return [opener, str(target)], "the default macOS viewer"

    linux_openers = (
        ("xdg-open", [str(target)], "the default desktop viewer"),
        ("gio", ["open", str(target)], "the default desktop viewer"),
        ("kde-open5", [str(target)], "the KDE viewer"),
        ("gnome-open", [str(target)], "the GNOME viewer"),
    )
    for executable, arguments, description in linux_openers:
        opener = shutil.which(executable)
        if opener:
            return [opener, *arguments], description

    raise RuntimeError(
        "No external image viewer was found (tried explorer.exe, open, xdg-open, gio, and desktop fallbacks)"
    )


def _is_wsl() -> bool:
    return bool(os.environ.get("WSL_DISTRO_NAME")) or "microsoft" in platform.release().lower()


def _windows_path(target: Path) -> str:
    converter = shutil.which("wslpath")
    if converter is None:
        return str(target)
    try:
        result = subprocess.run(
            [converter, "-w", str(target)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return str(target)
    return result.stdout.strip() or str(target)
