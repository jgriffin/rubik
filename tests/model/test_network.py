"""Tests for ValueNet — shape, dtype, gradient flow, parameterization."""

import pytest
import torch

from rubik.cube.spec import CUBE_2X2
from rubik.model.network import ResidualBlock, ValueNet


def test_input_dim_derives_from_spec():
    net = ValueNet(CUBE_2X2)
    assert net.input_dim == CUBE_2X2.n_stickers * CUBE_2X2.n_colors == 144


def test_forward_batched_shape_and_dtype():
    net = ValueNet(CUBE_2X2).eval()
    states = CUBE_2X2.solved_state.unsqueeze(0).expand(4, -1).clone()
    out = net(states)
    assert out.shape == (4,)
    assert out.dtype == torch.float32


def test_forward_single_state_returns_scalar():
    net = ValueNet(CUBE_2X2).eval()
    out = net(CUBE_2X2.solved_state)
    assert out.shape == ()


def test_gradient_flow_to_all_params():
    net = ValueNet(CUBE_2X2)
    states = CUBE_2X2.solved_state.unsqueeze(0).expand(8, -1).clone()
    out = net(states)
    loss = out.sum()
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"no gradient on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient on {name}"


def test_residual_block_preserves_dim():
    block = ResidualBlock(128)
    x = torch.randn(4, 128)
    y = block(x)
    assert y.shape == x.shape


def test_deterministic_forward_in_eval_mode():
    """Same seed + eval mode + same input → same output."""
    torch.manual_seed(42)
    net1 = ValueNet(CUBE_2X2).eval()
    torch.manual_seed(42)
    net2 = ValueNet(CUBE_2X2).eval()
    states = CUBE_2X2.solved_state.unsqueeze(0).expand(4, -1).clone()
    assert torch.equal(net1(states), net2(states))


def test_rejects_wrong_sticker_dim():
    net = ValueNet(CUBE_2X2).eval()
    bad = torch.zeros(4, CUBE_2X2.n_stickers - 1, dtype=torch.int8)
    with pytest.raises(ValueError, match="last dim"):
        net(bad)


def test_accepts_int8_input():
    """States from env are int8; network must accept without explicit cast."""
    net = ValueNet(CUBE_2X2).eval()
    states = CUBE_2X2.solved_state.unsqueeze(0).expand(4, -1).clone()
    assert states.dtype == torch.int8
    out = net(states)
    assert out.shape == (4,)


def test_body_widths_parameterizable():
    """Width and depth must be parametric so any caller can pick its own size."""
    net = ValueNet(CUBE_2X2, body_widths=(512, 128), n_residual_blocks=2).eval()
    states = CUBE_2X2.solved_state.unsqueeze(0).expand(4, -1).clone()
    out = net(states)
    assert out.shape == (4,)
    # Sanity: a bigger network has more params than the placeholder default.
    placeholder = ValueNet(CUBE_2X2)
    assert sum(p.numel() for p in net.parameters()) > sum(
        p.numel() for p in placeholder.parameters()
    )
