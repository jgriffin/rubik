"""Value network for DAVI training.

MLP value network parameterized on ``CubeSpec`` so the same class drops
in for 3x3 at M8 by changing the spec only — input dim derives from
``spec.n_stickers × spec.n_colors``.

Pipeline:
    int state ``(B, n_stickers)`` → one-hot ``(B, n_stickers, n_colors)``
    → flat ``(B, n_stickers·n_colors)`` → MLP body → scalar cost-to-go.

Body shape:
    - Linear(in_dim, h1) + BN + ReLU
    - Linear(h1, h2) + BN + ReLU
    - n × residual blocks: ``[Linear(h2, h2) + BN + ReLU] × 2`` with skip
    - Linear(h2, 1) — scalar output, no activation (regression target)

``body_widths=(h1, h2)`` and ``n_residual_blocks=n`` are required kwargs.
The class ships with no opinion on the right size — that's an empirical
question for tier 1+ experimentation. Module-level defaults exist only
so the unit tests can construct a tiny instance for shape/gradient
checks; they are placeholders, not a recommended config.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rubik.cube.spec import CubeSpec


class ResidualBlock(nn.Module):
    """[Linear + BN + ReLU] × 2 with skip connection. Preserves dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.bn2(self.fc2(h))
        return F.relu(x + h)


class ValueNet(nn.Module):
    """MLP value network for the cube; parameterized on ``CubeSpec``."""

    # Test-only placeholder defaults. Tiny on purpose — they exist so unit
    # tests can spin up an instance without committing to architecture
    # choices. No real training run should rely on them.
    _PLACEHOLDER_BODY_WIDTHS: tuple[int, int] = (64, 32)
    _PLACEHOLDER_N_RESIDUAL_BLOCKS: int = 1

    def __init__(
        self,
        spec: CubeSpec,
        *,
        body_widths: tuple[int, int] = _PLACEHOLDER_BODY_WIDTHS,
        n_residual_blocks: int = _PLACEHOLDER_N_RESIDUAL_BLOCKS,
    ):
        super().__init__()
        self.spec = spec
        self.input_dim = spec.n_stickers * spec.n_colors
        self.body_widths = body_widths
        self.n_residual_blocks = n_residual_blocks

        h1, h2 = body_widths
        self.fc_in = nn.Linear(self.input_dim, h1)
        self.bn_in = nn.BatchNorm1d(h1)
        self.fc_proj = nn.Linear(h1, h2)
        self.bn_proj = nn.BatchNorm1d(h2)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(h2) for _ in range(n_residual_blocks)]
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

        x = F.relu(self.bn_in(self.fc_in(x)))
        x = F.relu(self.bn_proj(self.fc_proj(x)))
        for block in self.residual_blocks:
            x = block(x)
        out = self.head(x).squeeze(-1)
        return out.squeeze(0) if squeeze_batch else out
