"""Export the 3x3 ValueNet champion checkpoint to ONNX.

Loads the checkpoint via ``rubik.server.inference.load_model`` (which reads
the sibling ``config.yaml`` for arch fields), runs ``torch.onnx.export`` with
a dynamic batch dim at opset 18 (PyTorch 2.11's dynamo exporter targets 18
natively; LayerNormalization is a native ONNX op from opset 17 onward), and
verifies the artifact via ``onnx.checker.check_model``.

Large models (≥2GB raw, or whenever the dynamo exporter chooses to) are
split into ``<name>.onnx`` (graph) plus ``<name>.onnx.data`` (weights).
``onnxruntime`` loads them transparently if co-located. For browser
deployment Block C may want a single-file artifact — handle that in the
loader on the JS side or via an inlining pass at packaging time.

Numerical parity vs the PyTorch forward is the responsibility of
``tests/onnx_parity_test_3x3.py`` — run that after export.

Usage:
    uv run python scripts/export_onnx_3x3.py
    uv run python scripts/export_onnx_3x3.py --checkpoint <path>
    uv run python scripts/export_onnx_3x3.py --opset 18 --output /tmp/x.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rubik.server.inference import load_model

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent.parent
    / "experiments/davi-3x3/runs/20260508T084940Z_ln_kmax30_100k/net_final.pt"
)


def export(checkpoint: Path, output: Path, opset: int) -> None:
    loaded = load_model(checkpoint, device=torch.device("cpu"))
    net = loaded.net
    spec = loaded.spec

    n_params = sum(p.numel() for p in net.parameters())
    print(f"Loaded {checkpoint}")
    print(f"  arch: {loaded.arch}")
    print(f"  n_stickers: {spec.n_stickers}, n_colors: {spec.n_colors}")
    print(f"  params: {n_params:,} ({n_params * 4 / 1e6:.1f} MB FP32)")

    dummy = torch.zeros((1, spec.n_stickers), dtype=torch.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        (dummy,),
        str(output),
        input_names=["states"],
        output_names=["value"],
        dynamic_axes={"states": {0: "batch"}, "value": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )

    graph_size = output.stat().st_size
    data_path = output.with_suffix(output.suffix + ".data")
    data_size = data_path.stat().st_size if data_path.exists() else 0
    total_mb = (graph_size + data_size) / 1e6
    print(f"Wrote {output} (graph {graph_size / 1e3:.1f} KB, opset {opset})")
    if data_size:
        print(f"  + external data {data_path.name} ({data_size / 1e6:.1f} MB)")
    print(f"  total: {total_mb:.1f} MB")

    import onnx

    model = onnx.load(str(output))
    onnx.checker.check_model(model)
    print("onnx.checker.check_model: PASSED")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: net_final.onnx next to the checkpoint.",
    )
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()
    out = args.output or args.checkpoint.with_suffix(".onnx")
    export(args.checkpoint, out, args.opset)


if __name__ == "__main__":
    main()
