"""Tests for the viz color palette invariants."""

import re

from rubik.viz.colors import SPEEDCUBING_PALETTE

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def test_six_colors() -> None:
    assert len(SPEEDCUBING_PALETTE) == 6


def test_hex_format() -> None:
    for value in SPEEDCUBING_PALETTE.values():
        assert _HEX_RE.match(value), f"not a 7-char hex: {value!r}"


def test_keys_are_face_indices() -> None:
    assert set(SPEEDCUBING_PALETTE.keys()) == {0, 1, 2, 3, 4, 5}


def test_palette_type_is_dict_int_str() -> None:
    assert isinstance(SPEEDCUBING_PALETTE, dict)
    for key, value in SPEEDCUBING_PALETTE.items():
        assert isinstance(key, int), (
            f"key {key!r} is {type(key).__name__}, expected int"
        )
        assert isinstance(value, str), (
            f"value {value!r} is {type(value).__name__}, expected str"
        )
        assert _HEX_RE.match(value), f"value {value!r} is not a 7-char hex string"
