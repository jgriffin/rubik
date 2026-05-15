"""Numerical parity: ValueNet PyTorch forward vs ONNX (onnxruntime CPU EP).

Runs N diverse random 3x3 scrambles through both paths and asserts max and
mean absolute deltas are within FP32 numerical drift. Marked ``slow`` because
loading the 234MB champion checkpoint takes ~10s. Skips automatically if the
``.onnx`` artifact isn't present — run ``scripts/export_onnx_3x3.py`` first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rubik.cube.env import random_scrambles
from rubik.cube.spec import CUBE_3X3
from rubik.server.inference import load_model

CHECKPOINT = (
    Path(__file__).resolve().parent.parent
    / "experiments/davi-3x3/runs/20260508T084940Z_ln_kmax30_100k/net_final.pt"
)
ONNX_PATH = CHECKPOINT.with_suffix(".onnx")

N_PARITY_STATES = 1000
PARITY_SCRAMBLE_DEPTH = 14
MAX_ABS_DELTA = 1e-4
MAX_MEAN_DELTA = 1e-5


@pytest.mark.slow
def test_onnx_pytorch_parity_3x3():
    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint missing: {CHECKPOINT}")
    if not ONNX_PATH.exists():
        pytest.skip(f"onnx missing: {ONNX_PATH} — run scripts/export_onnx_3x3.py first")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(0xCAFE)
    states, _ = random_scrambles(
        CUBE_3X3,
        batch_size=N_PARITY_STATES,
        depth=PARITY_SCRAMBLE_DEPTH,
        generator=gen,
    )

    loaded = load_model(CHECKPOINT, device=torch.device("cpu"))
    with torch.no_grad():
        pt_out = loaded.net(states).cpu().numpy()

    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["value"], {"states": states.cpu().numpy().astype(np.int64)})[0]

    delta = np.abs(pt_out - ort_out)
    max_d = float(delta.max())
    mean_d = float(delta.mean())
    print(
        f"\nparity: N={N_PARITY_STATES}, depth={PARITY_SCRAMBLE_DEPTH}, "
        f"max|Δ|={max_d:.3e}, mean|Δ|={mean_d:.3e}"
    )
    assert max_d < MAX_ABS_DELTA, f"max|Δ|={max_d:.3e} exceeds {MAX_ABS_DELTA}"
    assert mean_d < MAX_MEAN_DELTA, f"mean|Δ|={mean_d:.3e} exceeds {MAX_MEAN_DELTA}"
