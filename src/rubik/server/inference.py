"""Inference glue between FastAPI handlers and the rubik core.

Loads a trained ValueNet from a run dir (sibling config.yaml supplies
arch fields), generates scrambles via rubik.cube.env.random_scrambles,
and solves via rubik.search.beam.beam_solve_batch with N=1. The single-
row path is the production-first exercise of beam_solve_batch's batched
shape — keep N=1 for consistency with future multi-state requests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from rubik.cube.env import random_scrambles
from rubik.cube.spec import CUBE_3X3, CubeSpec
from rubik.model.network import ValueNet
from rubik.notation.facelet import facelet_to_state, state_to_facelet
from rubik.notation.moves import move_to_str
from rubik.search.beam import beam_solve_batch


@dataclass
class LoadedModel:
    """Bundle of net + spec + provenance carried through the request lifecycle."""

    net: torch.nn.Module
    spec: CubeSpec
    model_path: str
    arch: dict  # {body_widths, n_residual_blocks, normalization}


def _load_checkpoint(path: Path, device: torch.device) -> tuple[dict, bool]:
    """Load checkpoint, returning ``(raw, is_bundle)``.

    Mirrors scripts/beam_eval_model.py:_load_checkpoint structurally.
    Bundle dicts (training format) carry weights under ``net_state``;
    bare state-dicts (legacy) are themselves the weights.
    """
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "net_state" in raw:
        return raw, True
    return raw, False


def _read_arch(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return {
        "body_widths": tuple(cfg["body_widths"]),
        "n_residual_blocks": int(cfg["n_residual_blocks"]),
        "normalization": str(cfg["normalization"]),
    }


def load_model(
    checkpoint_path: Path, device: torch.device | None = None
) -> LoadedModel:
    """Load the 3x3 ValueNet from a run dir's ``net_final.pt``.

    Also accepts any ``net_step_*.pt`` from the same dir. Sibling
    ``config.yaml`` supplies architecture; bundle dict's ``net_state``
    is preferred over a bare state-dict. Defaults to MPS when
    available, CPU otherwise.
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    config_path = checkpoint_path.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"expected sibling config.yaml at {config_path}; not found"
        )
    arch = _read_arch(config_path)
    spec = CUBE_3X3
    net = ValueNet(
        spec,
        body_widths=arch["body_widths"],
        n_residual_blocks=arch["n_residual_blocks"],
        normalization=arch["normalization"],
    ).to(device)
    ckpt, is_bundle = _load_checkpoint(checkpoint_path, device)
    state_dict = ckpt["net_state"] if is_bundle else ckpt
    net.load_state_dict(state_dict)
    net.eval()
    return LoadedModel(net=net, spec=spec, model_path=str(checkpoint_path), arch=arch)


class _StubNet(torch.nn.Module):
    """Zero-returning stub used when ``RUBIK_USE_STUB_NET=1``.

    Solve loop will not actually solve (every state's V is identical),
    but the wire shape is real — Playwright/TestClient can exercise the
    full handler path without paying the 234M model load + MPS warmup.

    Carries a dummy parameter so ``next(net.parameters())`` (used by
    ``beam_solve_batch`` to discover the net's device) returns a tensor
    instead of raising ``StopIteration``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._device_marker = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)


def load_stub_model(device: torch.device | None = None) -> LoadedModel:
    if device is None:
        device = torch.device("cpu")  # stub doesn't need MPS
    spec = CUBE_3X3
    net = _StubNet().to(device).eval()
    return LoadedModel(
        net=net,
        spec=spec,
        model_path="<stub-net>",
        arch={"body_widths": (), "n_residual_blocks": 0, "normalization": "stub"},
    )


def scramble_facelet(
    spec: CubeSpec, length: int, seed: int | None = None
) -> tuple[str, list[str]]:
    """Generate a random scramble of ``length`` QTM moves, return (facelet, moves).

    Wraps ``random_scrambles`` (batch_size=1) and converts the resulting
    state to a kociemba facelet string and the move-index sequence to
    Singmaster strings.
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(int(seed))
    states, move_seqs = random_scrambles(
        spec, batch_size=1, depth=length, generator=gen
    )
    facelet = state_to_facelet(states[0], spec)
    moves = [move_to_str(int(m)) for m in move_seqs[0].tolist()]
    return facelet, moves


def solve_facelet(
    loaded: LoadedModel,
    facelet: str,
    *,
    beam_width: int,
    max_steps: int,
) -> dict:
    """Solve a single facelet via ``beam_solve_batch`` (N=1) and return wire dict.

    ``stats.steps_searched`` is the per-row solve length when solved
    (``solve_lens[0]``); when unsolved it's the full ``max_steps`` budget
    that was actually exhausted. ``stats.final_value`` is not currently
    surfaced by ``BeamSearchResult`` — left at 0.0 as a future enhancement.
    """
    state = facelet_to_state(facelet, loaded.spec).unsqueeze(0)
    t0 = time.perf_counter()
    result = beam_solve_batch(
        loaded.net,
        loaded.spec,
        state,
        beam_width=beam_width,
        max_steps=max_steps,
    )
    dt_ms = int((time.perf_counter() - t0) * 1000)
    solve_len = int(result.solve_lens[0].item())
    solved = solve_len >= 0
    move_idxs = result.solve_paths[0] if solved else []
    moves = [move_to_str(int(m)) for m in move_idxs]
    steps_searched = solve_len if solved else max_steps
    return {
        "solved": solved,
        "moves": moves,
        "stats": {
            "time_ms": dt_ms,
            "beam_width": beam_width,
            "steps_searched": steps_searched,
            "final_value": 0.0,  # not exposed by BeamSearchResult; future enhancement
        },
    }
