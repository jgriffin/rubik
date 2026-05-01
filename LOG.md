# LOG

Backward-looking. Newest blocks on top. See `ROADMAP.md` for what's
ahead, `SPEC.md` for the full project spec. Process docs at
`@~/.claude/cc-process.md`.

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
