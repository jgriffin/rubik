#!/usr/bin/env bash
uv run python experiments/beam-search-2x2/run.py --config experiments/beam-search-2x2/configs/sweep.yaml --out-dir experiments/beam-search-2x2/runs/sync500_kmax20-baseline
