"""Value network for DAVI training.

Architecture follows the DeepCubeA / draft-spec MLP, parameterized on
``CubeSpec`` so the same class drops in for 3x3 at M8 by changing the
spec only — input dim derives from ``spec.n_stickers × spec.n_colors``.

Pipeline:
    int8 state ``(B, n_stickers)`` → one-hot ``(B, n_stickers, n_colors)``
    → flat ``(B, n_stickers·n_colors)`` → MLP body → scalar cost-to-go.

Body (per draft spec):
    - Linear(in_dim, 5000) + BN + ReLU
    - Linear(5000, 1000) + BN + ReLU
    - 4 × residual blocks: ``[Linear(1000, 1000) + BN + ReLU] × 2`` with skip
    - Linear(1000, 1) — scalar output, no activation (regression target)

For 2x2 (in_dim = 144), this is ~10M params; for 3x3 (in_dim = 288)
it's the canonical DeepCubeA size. Body widths and residual count are
parametric so M7 hyperparam sweep can vary them without touching the
architecture.
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

    def __init__(
        self,
        spec: CubeSpec,
        *,
        body_widths: tuple[int, int] = (5000, 1000),
        n_residual_blocks: int = 4,
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
