"""Tests for portable reader text conversion."""

from storyforge.portable_text import to_portable_ascii


def test_to_portable_ascii_transliterates_text_and_normalizes_whitespace():
    source = "“Café”—北京…\tLine\r\nNext\x00😀"

    result = to_portable_ascii(source)

    assert result == '"Cafe"-BeiJing...    Line\nNext:grinning:'
    assert result.encode("ascii").decode("ascii") == result


def test_to_portable_ascii_preserves_plain_ascii():
    source = "Story: A simple tale\n\nOnce upon a time..."

    assert to_portable_ascii(source) == source
