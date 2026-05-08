"""Backward-compat shim — delegates to ``beam_eval_run.py``.

The idempotent / ensure semantics this script used to add as a wrapper
now live inside ``beam_eval_run.py`` as the **default** behavior (skip
already-evaluated steps; pass ``--force`` to recompute). This file is
kept so external callers — notably the ``/tmp/ln_kmax30_chain.sh``
overnight chainer — keep working without modification.

Prefer ``beam_eval_run.py`` directly for new work; this shim is a thin
forwarder and may be removed once no callers reference it.

Translation:

    ensure_run_results.py <run-dir> [--eval-config NAME] [--force]

    →  beam_eval_run.py <run-dir> --config NAME --render-html [--force]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Training run directory.")
    parser.add_argument(
        "--eval-config",
        default="fast",
        help=(
            "Beam-eval config name (resolves to "
            "scripts/eval-configs/<name>.yaml). Default: fast."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forwarded to beam_eval_run.py — recompute every checkpoint.",
    )
    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"error: {run_dir} is not a directory", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "beam_eval_run.py"),
        str(run_dir),
        "--config",
        args.eval_config,
        "--render-html",
    ]
    if args.force:
        cmd.append("--force")
    print(f"-> {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
