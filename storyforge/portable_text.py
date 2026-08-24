"""Portable plain-text conversion for reader artifacts and clipboards."""

from anyascii import anyascii


def to_portable_ascii(text: str) -> str:
    """Return readable ASCII with stable whitespace for plain-text consumers."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    transliterated = anyascii(normalized)
    portable = "".join(character for character in transliterated if character == "\n" or " " <= character <= "~")
    portable.encode("ascii")
    return portable
