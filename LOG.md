# LOG

Backward-looking. Newest blocks on top. See `ROADMAP.md` for what's
ahead, `SPEC.md` for the full project spec. Process docs at
`@~/.claude/cc-process.md`.

## 2026-05-01 — M0 skeleton ✅ done
**Goal:** Land the package skeleton, `CubeSpec` abstraction, and notation hub stubs (move + state converters with round-trip tests). Acceptance: `uv run pytest` green, `uv run ruff check` clean, `CubeSpec(CUBE_2X2)` round-trips through every notation converter.
**Milestone:** [plans/m0-skeleton.md](plans/m0-skeleton.md)
**Approach:** Create the 7 package subdirs as `__init__.py` placeholders. Implement `cube/spec.py` (CubeSpec dataclass + `CUBE_2X2`), `notation/moves.py` (CubeSpec-independent QTM converters), `notation/state.py` (CubeSpec-parameterized tensor↔face-dict). Add ruff + pytest config to `pyproject.toml`. Tests under `tests/cube/` and `tests/notation/`. Branch: `m0-skeleton`.
**Outcome:**
- Acceptance gate green: `42 passed`, `ruff check` clean, `ruff format --check` clean.
- 17 source files added (7 skeleton `__init__.py`, 3 content modules, 5 test files + 2 empty test `__init__.py`), 2 modified (`pyproject.toml`, `src/rubik/__init__.py`), plan committed to `plans/m0-skeleton.md`.
- **Decision (recorded):** Face order locked at `("U","L","F","R","B","D")` — inherited from archived draft (lines 50–55) so the future encoding doc / 3x3 extension stays consistent. Move index mapping locked at `face_idx * 2 + direction` (CW=0, CCW=1) so same-face pruning is `move >> 1`.
- **Decision (recorded):** Within-face geometric ordering deliberately NOT pinned at M0. M0 commits only to face-block ordering and contiguity; M1's cubie oracle pins the within-face mapping when it generates physical-rotation move tables.
- **Decision (recorded):** `CubeSpec` kept lean — no `move_table`, no `face_to_color_map`, no corner/edge bookkeeping. Fields earn their way in by being needed downstream (move_table at M2, edge bookkeeping at M5 if oracle architecture demands).
- **Decision (recorded):** `solved_state` is a `@property` returning a fresh tensor each call (no caching). Avoids aliasing surprises if a caller mutates; cost is negligible (24-element int8 tensor). Locked in via a `test_solved_state_returns_fresh_tensor`.
- Drive-by: `state_to_dict` validation extended to also cover bad-shape inputs (plan only specified bad-input cases for `dict_to_state`).
**Commits:** 164e0da

## 2026-05-01 — Bootstrap project ✅ done
**Goal:** Initialize the repo, package skeleton, foundational docs, and first commit.
**Milestone:** drive-by (pre-M0)
**Approach:** `git init`, `uv init --lib`, add deps, write SPEC / ROADMAP / CLAUDE / LOG, archive the original draft spec, commit.
**Outcome:**
- `git init -b main`; default branch `main`.
- `.gitignore` covering Python, macOS, IDE, env, and project-specific (model checkpoints, training logs, `experiments/*/runs/`, `experiments/*/results/`).
- `uv init --lib --no-readme --python 3.12` produced `src/rubik/` package with `py.typed`. Trimmed auto-generated `authors` and updated `description` in `pyproject.toml`.
- Deps: runtime `torch`, `numpy`; dev `pytest`, `ruff`, `pytest-snapshot`. No pycuber (3x3-only, unmaintained); no tensorboard / wandb / matplotlib (deferred).
- Authored `CLAUDE.md` (opts into cc-process via `@~/.claude/cc-process.md`, codifies tooling + key conventions), `SPEC.md` (project goal, 8 key decisions with rationale, M0–M9 acceptance criteria, package layout, references), `ROADMAP.md` (cc-process format), `LOG.md` (this file).
- Original draft moved to `plans/archive/llm-draft-spec.md` — retained for technical detail (sticker indexing, network architecture, hyperparam starting points), no longer authoritative.
- **Decision (recorded):** 2x2 first, parameterized via `CubeSpec` — single code path, no 2x2/3x3 forks. Two-witness correctness (cubie oracle + tensor); pycuber dropped. QTM only at first. Notation hub as a first-class module. Visualization is its own milestone (M3) with HTML/SVG preferred over matplotlib. "Make it work, then make it fast" — perf bar moves to M4. Experiment-loop pattern established at M4, reused at M8.
**Commits:** 9522ce0
