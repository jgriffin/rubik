"""Face → hex color palette — single source of truth for the viz layer.

Keyed by face index matching `spec.faces` ordering
(``("U", "L", "F", "R", "B", "D")``). Speedcubing-standard scheme:
white / orange / green / red / blue / yellow.
"""

FACE_COLORS: dict[int, str] = {
    0: "#f5f5f5",  # U  white
    1: "#ff8c00",  # L  orange
    2: "#1aa64a",  # F  green
    3: "#dd2222",  # R  red
    4: "#1144cc",  # B  blue
    5: "#ffdd00",  # D  yellow
}
