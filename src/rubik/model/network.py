"""Value network for DAVI training.

MLP value network parameterized on ``CubeSpec`` so the same class drops
in for 3x3 at M8 by changing the spec only — input dim derives from
``spec.n_stickers × spec.n_colors``.

Pipeline:
    int state ``(B, n_stickers)`` → one-hot ``(B, n_stickers, n_colors)``
    → flat ``(B, n_stickers·n_colors)`` → MLP body → scalar cost-to-go.

Body shape:
    - Linear(in_dim, h1) + Norm + ReLU
    - Linear(h1, h2) + Norm + ReLU
    - n × residual blocks: ``[Linear(h2, h2) + Norm + ReLU] × 2`` with skip
    - Linear(h2, 1) — scalar output, no activation (regression target)

``body_widths=(h1, h2)``, ``n_residual_blocks=n``, and ``normalization``
are required kwargs. ``normalization`` is one of ``"bn"`` (BatchNorm1d),
``"none"`` (Identity, i.e. no normalization), or ``"ln"`` (LayerNorm).
The class ships with no opinion on the right architecture — that's an
empirical question for tier 1+ experimentation. Module-level placeholder
defaults exist only so the unit tests can construct a tiny instance for
shape/gradient checks; they are placeholders, not a recommended config.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rubik.cube.spec import CubeSpec

_VALID_NORMALIZATIONS = ("bn", "none", "ln")


def _make_norm(kind: str, dim: int) -> nn.Module:
    """Build a normalization layer of the requested kind, or Identity."""
    if kind == "bn":
        return nn.BatchNorm1d(dim)
    if kind == "ln":
        return nn.LayerNorm(dim)
    if kind == "none":
        return nn.Identity()
    raise ValueError(
        f"normalization must be one of {_VALID_NORMALIZATIONS!r}, got {kind!r}"
    )


class ResidualBlock(nn.Module):
    """[Linear + Norm + ReLU] × 2 with skip connection. Preserves dimension."""

    def __init__(self, dim: int, normalization: str):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = _make_norm(normalization, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = _make_norm(normalization, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm1(self.fc1(x)))
        h = self.norm2(self.fc2(h))
        return F.relu(x + h)


class ValueNet(nn.Module):
    """MLP value network for the cube; parameterized on ``CubeSpec``."""

    # Test-only placeholder defaults. Tiny on purpose — they exist so unit
    # tests can spin up an instance without committing to architecture
    # choices. No real training run should rely on them.
    _PLACEHOLDER_BODY_WIDTHS: tuple[int, int] = (64, 32)
    _PLACEHOLDER_N_RESIDUAL_BLOCKS: int = 1
    _PLACEHOLDER_NORMALIZATION: str = "bn"

    def __init__(
        self,
        spec: CubeSpec,
        *,
        body_widths: tuple[int, int] = _PLACEHOLDER_BODY_WIDTHS,
        n_residual_blocks: int = _PLACEHOLDER_N_RESIDUAL_BLOCKS,
        normalization: str = _PLACEHOLDER_NORMALIZATION,
    ):
        super().__init__()
        if normalization not in _VALID_NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {_VALID_NORMALIZATIONS!r}, "
                f"got {normalization!r}"
            )
        self.spec = spec
        self.input_dim = spec.n_stickers * spec.n_colors
        self.body_widths = body_widths
        self.n_residual_blocks = n_residual_blocks
        self.normalization = normalization

        h1, h2 = body_widths
        self.fc_in = nn.Linear(self.input_dim, h1)
        self.norm_in = _make_norm(normalization, h1)
        self.fc_proj = nn.Linear(h1, h2)
        self.norm_proj = _make_norm(normalization, h2)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(h2, normalization) for _ in range(n_residual_blocks)]
        )
        self.head = nn.Linear(h2, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Run the network. states: ``(B, n_stickers)`` int (any int dtype).

        Returns ``(B,)`` float32 scalar cost-to-go estimate.
        """
        if states.shape[-1] != self.spec.n_stickers:
            raise ValueError(
                f"states last dim {states.shape[-1]} does not match"
                f" spec.n_stickers ({self.spec.n_stickers})"
            )
        one_hot = F.one_hot(states.long(), num_classes=self.spec.n_colors).float()
        x = one_hot.flatten(start_dim=-2)  # (..., n_stickers · n_colors)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # BN requires (B, F) shape
            squeeze_batch = True
        else:
            squeeze_batch = False

        x = F.relu(self.norm_in(self.fc_in(x)))
        x = F.relu(self.norm_proj(self.fc_proj(x)))
        for block in self.residual_blocks:
            x = block(x)
        out = self.head(x).squeeze(-1)
        return out.squeeze(0) if squeeze_batch else out
