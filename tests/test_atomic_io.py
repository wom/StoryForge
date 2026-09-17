"""Tests for crash-safe user-file replacement."""

import stat
from unittest.mock import patch

import pytest

from storyforge.atomic_io import atomic_write_text


def test_atomic_write_text_replaces_content_and_preserves_mode(tmp_path):
    target = tmp_path / "storyforge.ini"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)

    atomic_write_text(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert list(tmp_path.glob(".storyforge.ini.*.tmp")) == []


def test_atomic_write_text_preserves_original_and_cleans_temp_when_replace_fails(tmp_path):
    target = tmp_path / "storyforge.ini"
    target.write_text("original", encoding="utf-8")

    with (
        patch("storyforge.atomic_io.os.replace", side_effect=OSError("replace failed")),
        pytest.raises(OSError, match="replace failed"),
    ):
        atomic_write_text(target, "candidate")

    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.glob(".storyforge.ini.*.tmp")) == []
