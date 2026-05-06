"""Search primitives — beam search policy.

`beam_solve_batch` runs batched beam search using V_θ as the scorer with
within-beam dedup via raw-bytes state packing. Spec-agnostic — same code
path serves 2x2 and 3x3.
"""

from rubik.search.beam import BeamSearchResult, beam_solve_batch

__all__ = ["BeamSearchResult", "beam_solve_batch"]
