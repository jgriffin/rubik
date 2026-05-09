# rubik web demo

Vite + React + TypeScript frontend for the M9.1 solver demo. The
backend FastAPI server lives at `src/rubik/server/`.

## Setup (one-time)

From the repo root:

```
uv sync --extra server
cd web
pnpm install
pnpm exec playwright install chromium
```

## Run dev

Two terminals:

```
# terminal A: backend
uv run rubik-serve

# terminal B: frontend
cd web && pnpm dev
```

Then open <http://localhost:5173>.

## Backend env vars

- `RUBIK_MODEL_PATH` — path to a `net_final.pt` (default: the canonical LN run).
- `RUBIK_USE_STUB_NET=1` — install a zero-returning stub `nn.Module` and skip MPS warmup. Fast startup, doesn't actually solve. Used by Playwright's default e2e config.

## E2E tests

```
pnpm test:e2e          # stub-net backend, fast
pnpm test:e2e:real     # real LN model, slower (~3s warmup)
```

Playwright's `webServer` config boots both processes automatically — no manual terminal juggling.
