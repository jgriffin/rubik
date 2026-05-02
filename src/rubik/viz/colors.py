"""Face → hex color palette for the viz layer.

`SPEEDCUBING_PALETTE` is the default — keyed by face index matching
``spec.faces`` ordering (``("U", "L", "F", "R", "B", "D")``), with the
speedcubing-standard scheme: white / orange / green / red / blue / yellow.

The ``Palette`` type alias lets callers plug in alternative palettes
(monochrome for print, accessible/CVD-safe variants, custom themes)
without editing this module — the SVG renderers accept a ``palette``
keyword that defaults to ``SPEEDCUBING_PALETTE``.
"""

type Palette = dict[int, str]

SPEEDCUBING_PALETTE: Palette = {
    0: "#f5f5f5",  # U  white
    1: "#ff8c00",  # L  orange
    2: "#1aa64a",  # F  green
    3: "#dd2222",  # R  red
    4: "#1144cc",  # B  blue
    5: "#ffdd00",  # D  yellow
}
