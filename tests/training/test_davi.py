"""Tests for DAVI primitives — target computation, step, target-sync, config."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from rubik.cube.env import apply_all_moves
from rubik.cube.spec import CUBE_2X2
from rubik.model.network import ValueNet
from rubik.training.config import DAVIConfig
from rubik.training.davi import compute_targets, davi_step, sync_target
from rubik.training.scrambles import generate_adi_batch


def test_compute_targets_one_move_from_solved_bounded_by_one():
    """One child IS solved (inverse move) → target ≤ 1 always.

    Equality only when V_target ≥ 0 on every non-solved child. With
    random init this isn't guaranteed (output is unconstrained scalar),
    so we test the upper bound here and exact equality in the
    zeroed-head test below.
    """
    spec = CUBE_2X2
    net = ValueNet(spec).eval()
    states = apply_all_moves(spec.solved_state, spec)  # (12, 24)
    targets = compute_targets(states, net, spec)
    assert (targets <= 1.0 + 1e-5).all()


def test_compute_targets_with_zero_v_target_is_exactly_one():
    """Forcing V_target ≡ 0 (zero out the head) → target for one-move-
    from-solved states = min_a (1 + 0) = 1 exactly.
    """
    spec = CUBE_2X2
    net = ValueNet(spec).eval()
    with torch.no_grad():
        net.head.weight.zero_()
        net.head.bias.zero_()
    states = apply_all_moves(spec.solved_state, spec)
    targets = compute_targets(states, net, spec)
    assert torch.allclose(targets, torch.ones(12), atol=1e-5)


def test_compute_targets_shape_and_dtype():
    spec = CUBE_2X2
    net = ValueNet(spec).eval()
    states, _, _ = generate_adi_batch(spec, batch_size=32, max_depth=5)
    targets = compute_targets(states, net, spec)
    assert targets.shape == (32,)
    assert targets.dtype == torch.float32


def test_compute_targets_no_grad_on_target_net():
    """compute_targets uses torch.no_grad on V_target — params accumulate
    no gradient even if the calling context allows backward."""
    spec = CUBE_2X2
    target_net = ValueNet(spec).eval()
    states, _, _ = generate_adi_batch(spec, batch_size=8, max_depth=3)
    targets = compute_targets(states, target_net, spec)
    # Targets should not require grad
    assert not targets.requires_grad


def test_sync_target_copies_parameters_bit_exactly():
    spec = CUBE_2X2
    net = ValueNet(spec)
    target = ValueNet(spec)
    # Different init → some params differ
    differ_before = any(
        not torch.equal(p1, p2)
        for p1, p2 in zip(net.parameters(), target.parameters(), strict=True)
    )
    assert differ_before
    sync_target(net, target)
    for p1, p2 in zip(net.parameters(), target.parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_sync_target_copies_buffers():
    """BN running stats are buffers, not params — must also sync."""
    spec = CUBE_2X2
    net = ValueNet(spec)
    target = ValueNet(spec)
    # Drive net's BN running stats away from default by training briefly.
    net.train()
    for _ in range(5):
        states, _, _ = generate_adi_batch(spec, batch_size=64, max_depth=5)
        net(states)
    sync_target(net, target)
    for (n1, b1), (n2, b2) in zip(
        net.named_buffers(), target.named_buffers(), strict=True
    ):
        assert n1 == n2
        assert torch.equal(b1, b2), f"buffer mismatch on {n1}"


def test_davi_step_reduces_loss_over_iterations():
    """Repeated davi_step on the same batch with a fixed target should
    monotonically reduce loss (within optimizer noise)."""
    spec = CUBE_2X2
    torch.manual_seed(0)
    net = ValueNet(spec)
    target = ValueNet(spec).eval()
    sync_target(net, target)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    states, _, _ = generate_adi_batch(
        spec,
        batch_size=64,
        max_depth=5,
        generator=torch.Generator().manual_seed(0),
    )

    initial_loss = davi_step(net, target, optimizer, states, spec)
    for _ in range(20):
        final_loss = davi_step(net, target, optimizer, states, spec)
    assert final_loss < initial_loss


def test_davi_step_returns_scalar_float():
    spec = CUBE_2X2
    net = ValueNet(spec)
    target = ValueNet(spec).eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    states, _, _ = generate_adi_batch(spec, batch_size=32, max_depth=5)
    loss = davi_step(net, target, optimizer, states, spec)
    assert isinstance(loss, float)
    assert loss >= 0


def _full_davi_config_kwargs() -> dict:
    """Every field DAVIConfig requires. Values are illustrative — the test
    suite asserts shape + round-trip + frozen, not value semantics."""
    return dict(
        max_scramble_depth=14,
        max_scramble_depth_initial=0,
        max_scramble_depth_ramp_steps=0,
        batch_size=512,
        n_steps=10_000,
        learning_rate=1e-4,
        target_sync_interval=5_000,
        body_widths=(2048, 512),
        n_residual_blocks=4,
        normalization="bn",
        log_every=100,
        eval_every=5_000,
        checkpoint_every=10_000,
        seed=42,
        device="cpu",
        early_stop_enabled=False,
        early_stop_metric="macro_v_star_mae",
        early_stop_patience_evals=0,
        early_stop_min_evals=0,
        early_stop_min_delta=0.0,
    )


def test_davi_config_requires_all_fields_explicit():
    """No defaults — calling DAVIConfig() must fail loudly.

    The contract: every YAML must specify every field; you cannot
    accidentally inherit a hyperparameter from a code-default. Missing
    fields surface as TypeError at construction.
    """
    with pytest.raises(TypeError):
        DAVIConfig()  # type: ignore[call-arg]


def test_davi_config_yaml_roundtrip(tmp_path):
    cfg = DAVIConfig(**_full_davi_config_kwargs())
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    assert path.exists()
    loaded = DAVIConfig.from_yaml(path)
    assert loaded == cfg
    # Tuple preservation is the trickiest part — call out explicitly.
    assert isinstance(loaded.body_widths, tuple)


def test_davi_config_from_yaml_missing_field_raises(tmp_path):
    """Missing keys in the YAML → loud TypeError, not silent default."""
    cfg = DAVIConfig(**_full_davi_config_kwargs())
    path = tmp_path / "incomplete.yaml"
    cfg.to_yaml(path)
    # Strip a field from the on-disk yaml.
    import yaml as _yaml

    d = _yaml.safe_load(path.read_text())
    d.pop("learning_rate")
    path.write_text(_yaml.safe_dump(d))
    with pytest.raises(TypeError):
        DAVIConfig.from_yaml(path)


def test_davi_config_frozen():
    """Frozen dataclass — accidental mutation should raise."""
    cfg = DAVIConfig(**_full_davi_config_kwargs())
    with pytest.raises(FrozenInstanceError):
        cfg.batch_size = 200  # type: ignore[misc]
