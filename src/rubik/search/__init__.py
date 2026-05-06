"""Search primitives — beam search policy.

`beam_solve_batch` runs batched beam search using V_θ as the scorer with
within-beam dedup via raw-bytes state packing. Spec-agnostic — same code
path serves 2x2 and 3x3.

`BeamEvalConfig` is the YAML-serialized config consumed by
`experiments/beam-search-2x2/run.py`.
"""

from rubik.search.beam import BeamSearchResult, beam_solve_batch
from rubik.search.config import BeamEvalConfig

__all__ = ["BeamEvalConfig", "BeamSearchResult", "beam_solve_batch"]
