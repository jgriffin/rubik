# M9 — Web UI + solver demo

Forward-looking. See `ROADMAP.md` for milestone status, `LOG.md` for
in-progress / closed blocks, `SPEC.md` for the project spec.

## Goal

A web app that wraps the trained 3x3 ValueNet to demonstrate it solving
real cube states. User clicks **Scramble** → sees the cube → clicks
**Solve** → watches the network's solution play out. First in our flat
sideways-cross renderer (M9.1), then in 3D via cubing.js's
`<twisty-player>` (M9.2). Eventually accepts arbitrary cube states via
paste-notation or manual color entry (M9.3). Photo / CV input is its own
milestone (M10).

## Locked decisions

- **Inference.** FastAPI Python backend, calling existing
  `beam_solve_batch` from `src/rubik/search/beam.py`. Backend loads
  `experiments/davi-3x3/runs/20260508T084940Z_ln_kmax30_100k/net_final.pt`
  at startup; warms one solve before serving so the first user-triggered
  solve doesn't pay MPS kernel-compile cost. ONNX-in-browser parked.
- **Frontend.** TypeScript + React + Vite, with pnpm. `vite build`
  produces a static `dist/` deployable to Vercel. three.js available for
  any custom 3D, but `<twisty-player>` carries its own.
- **Cubing primitives.** cubing.js npm package owns notation parsing,
  scramble generation, and `<twisty-player>` (the animated 3D cube web
  component). We do not reimplement these.
- **Internal state form.** 54-char facelet string (kociemba's format).
  Both server and client canonicalize to facelet; cubing.js handles
  conversion to/from move sequences.
- **Move notation on the wire.** Singmaster (`R U R' U2 F'`). Our
  training is QTM-only, so the server emits only quarter turns; the UI
  may visually group consecutive same-face turns.
- **2D rendering (M9.1).** Our own flat sideways-cross SVG renderer in
  React.
- **3D rendering (M9.2+).** cubing.js `<twisty-player>` web component.

## Out of scope (entire M9)

- Photo / camera-based cube state detection — own milestone (M10).
- ONNX export and browser-side inference — backlog. Park until Vercel
  deploy is seriously pursued; would also need a JS port of the beam
  search loop.
- Solution-trace analysis (V\* trajectory, branching factor, policy
  entropy, kociemba-comparison overlay). Useful but not the M9 demo.
  Kociemba comparison runs as a Python eval (separate ROADMAP backlog
  item).
- 2x2 in the UI. Backend stays parameterized by `CubeSpec`, but only 3x3
  is exposed.
- Authentication, multi-user, persistence — local single-user dev only.

## Phases

### M9.1 — Solver demo MVP

**Goal.** Random-scramble button → flat 2D render → "Solve" button →
network's moves displayed → click-through animation between states.

**Scope.**
- FastAPI app at `src/rubik/server/`. Three endpoints:
  - `POST /api/scramble` body `{length:int=20, seed?:int}` → `{moves:[…], state:"…"}`
  - `POST /api/solve` body `{state:"…", beam_width?:int=256}` → `{solved:bool, moves:[…], stats:{time_ms, beam_width, …}}`
  - `GET /api/health` → `{model_loaded, model_path, warmup_done}`
- Vite + React + TS + cubing.js scaffolding under `web/`.
- Flat sideways-cross SVG renderer (React component).
- Move-list with prev / next / play-all controls; cubing.js applies
  moves client-side from the canonical state.
- CORS in dev (Vite :5173 → FastAPI :8000).

**Acceptance.**
- `uv run rubik-serve` starts the backend; `pnpm --prefix web dev`
  starts the frontend; the page loads at localhost:5173 and renders.
- Click *Scramble* → cube renders the scrambled state.
- Click *Solve* → spinner during solve → moves list appears, cube
  returns to solved when stepped through fully.
- Step controls (prev / next / play) work.
- `/api/health` reports `warmup_done=true` only after the warm solve
  completes.

**Out of M9.1 scope.** 3D, paste-notation, manual color entry, mobile
layout, share URL, polish.

### M9.2 — 3D rendering via `<twisty-player>`

**Goal.** Drop in cubing.js `<twisty-player>`; toggle 2D ↔ 3D; sync the
move-step state between the two views.

**Scope.**
- Add `<twisty-player>` next to (or replacing, via toggle) the flat
  renderer.
- View-mode toggle: "Flat" | "3D".
- Single source of truth: app state holds `{scramble, solution, stepIdx}`.
  Both renderers consume it.
- Solution alg (Singmaster string) drives `<twisty-player>` directly.

### M9.3 — Alt input + polish

**Goal.** Multiple input formats; copy-out shareable state; mobile-friendly.

**Scope.**
- Paste-notation field. cubing.js parses both move sequences and 54-char
  facelet strings (it can convert between).
- Manual color-grid entry (six 3x3 sticker grids, click to recolor).
- Random-seed control + "copy this scramble" link.
- URL-encoded shareable state (e.g. `/#state=…` or `/#scramble=…`).
- Mobile-friendly layout (touch-friendly controls; renderer scales).

## Repo layout

```
src/rubik/server/         # FastAPI backend
  __init__.py
  app.py                  # FastAPI app, lifespan = load + warm model
  schemas.py              # pydantic request/response models
  inference.py            # thin wrapper around beam_solve_batch
web/                      # frontend (pnpm-managed)
  package.json
  pnpm-lock.yaml
  vite.config.ts
  tsconfig.json
  index.html
  src/
    main.tsx
    App.tsx
    components/
      FlatCubeRenderer.tsx       # M9.1 — SVG sideways-cross
      ScrambleButton.tsx
      SolveButton.tsx
      MoveList.tsx
      StepControls.tsx
      TwistyPlayerWrapper.tsx    # M9.2
      ColorGridEntry.tsx         # M9.3
    api/
      client.ts                  # fetch wrappers, types
    state/
      cubeState.ts               # facelet helpers, cubing.js bridge
plans/m9-ui.md             # this file
plans/m9.1-solver-demo.md  # per-block plan, written when block work starts
```

`pyproject.toml` gets a `[project.optional-dependencies] server` group
with `fastapi`, `uvicorn`. The `rubik-serve` command lives under
`[project.scripts]`.

## Server contract details

All endpoints return JSON; errors as `{error: string}` with the
appropriate HTTP status.

```
POST /api/scramble
  request:  {length: int = 20, seed?: int}
  response: {moves: [string], state: string}    # state is 54-char facelet

POST /api/solve
  request:  {state: string, beam_width?: int = 256, max_steps?: int = 30}
  response: {solved: bool, moves: [string],
             stats: {time_ms: int, beam_width: int, steps_searched: int,
                     final_value: float}}

GET  /api/health
  response: {model_loaded: bool, model_path: string,
             warmup_done: bool, cube_size: 3}
```

Move strings are Singmaster QTM (`R`, `R'`, `U`, `U'`, etc.) — no double
turns since training is QTM-only.

## Risks / unknowns

1. **MPS kernel compile on first solve** — probably 5–10 s. Mitigated by
   warmup at server start (`/api/health` reports `warmup_done=false`
   until the warm solve completes; UI can wait or display a banner).
2. **CORS in dev** — Vite :5173 → FastAPI :8000. Standard
   `fastapi.middleware.cors` allowlist for `http://localhost:5173`.
3. **Beam search latency** — recent eval data shows d=14 at width=256
   BF16 ≈ 0.3 s per scramble in batched form. Single-scramble interactive
   should feel snappy; watch this in M9.1 and investigate before chasing
   fancier UX if it disappoints.
4. **cubing.js npm under Vite + TS** — verify clean ESM import at
   scaffold time, not later.
5. **uvicorn autoreload** reloads the PyTorch model on every code
   change (~2 s). Document or accept; not blocking.

## Backlog adjacent to M9

- **Kociemba comparison** — Python eval, see ROADMAP backlog. Not UI work.
- **ONNX-in-browser inference** — for fully static Vercel deploy. Park
  until that goal is concrete.
- **Solution-trace analysis sidebar** — V\* per step, branching factor,
  policy entropy. Lands inside the UI later, not M9.

## References

- cubing.js: https://js.cubing.net/cubing/
- cubing.js source: https://github.com/cubing/cubing.js
- `<twisty-player>` explorer: https://experiments.cubing.net/cubing.js/twisty-player-explorer/
- Twizzle (full app on cubing.js): https://alpha.twizzle.net/edit/
- Kociemba 54-char facelet format: https://kociemba.org/cube.htm
