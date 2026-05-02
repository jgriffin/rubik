"""Tests for the viz color palette invariants."""

import re

from rubik.viz.colors import FACE_COLORS

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def test_six_colors() -> None:
    assert len(FACE_COLORS) == 6


def test_hex_format() -> None:
    for value in FACE_COLORS.values():
        assert _HEX_RE.match(value), f"not a 7-char hex: {value!r}"


def test_keys_are_face_indices() -> None:
    assert set(FACE_COLORS.keys()) == {0, 1, 2, 3, 4, 5}
