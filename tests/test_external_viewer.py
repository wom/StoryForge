"""Platform launcher selection for generated image artifacts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from storyforge.external_viewer import open_path_externally


def test_wsl_uses_explorer_with_a_translated_windows_path(tmp_path):
    image = tmp_path / "story image.png"
    image.write_bytes(b"image")
    converted = MagicMock(stdout=r"C:\Users\reader\story image.png" + "\n")

    def executable(name):
        return {"explorer.exe": "/mnt/c/Windows/explorer.exe", "wslpath": "/usr/bin/wslpath"}.get(name)

    with (
        patch.dict("os.environ", {"WSL_DISTRO_NAME": "Ubuntu"}),
        patch("storyforge.external_viewer.shutil.which", side_effect=executable),
        patch("storyforge.external_viewer.subprocess.run", return_value=converted) as run,
        patch("storyforge.external_viewer.subprocess.Popen") as popen,
    ):
        viewer = open_path_externally(image)

    assert viewer == "Windows Explorer"
    run.assert_called_once_with(
        ["/usr/bin/wslpath", "-w", str(image.resolve())],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert popen.call_args.args[0] == ["/mnt/c/Windows/explorer.exe", converted.stdout.strip()]


@pytest.mark.parametrize(
    ("system", "available", "expected"),
    [
        ("Linux", "xdg-open", ["/usr/bin/xdg-open"]),
        ("Darwin", "open", ["/usr/bin/open"]),
    ],
)
def test_desktop_platform_uses_native_default_opener(tmp_path, system, available, expected):
    image = tmp_path / "story.png"
    image.write_bytes(b"image")

    def executable(name):
        return f"/usr/bin/{name}" if name == available else None

    with (
        patch.dict("os.environ", {}, clear=True),
        patch("storyforge.external_viewer.platform.system", return_value=system),
        patch("storyforge.external_viewer.platform.release", return_value="generic"),
        patch("storyforge.external_viewer.shutil.which", side_effect=executable),
        patch("storyforge.external_viewer.subprocess.Popen") as popen,
    ):
        open_path_externally(image)

    assert popen.call_args.args[0] == [*expected, str(image.resolve())]


def test_missing_external_viewer_reports_a_clear_error(tmp_path):
    image = tmp_path / "story.png"
    image.write_bytes(b"image")
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("storyforge.external_viewer.platform.system", return_value="Linux"),
        patch("storyforge.external_viewer.platform.release", return_value="generic"),
        patch("storyforge.external_viewer.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="No external image viewer"),
    ):
        open_path_externally(image)
