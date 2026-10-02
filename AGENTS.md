# Working on Galton's Hoard

A local test bench for language and vision models (package `galton_hoard`, service `galton-hoard`, app id `galton`, port 5201). It runs checkable tasks on the models of this computer, stores the answers, computes statistics and publishes a routing table. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first.

## Commands

```bash
python -m pytest                          # the whole suite; no network, no GPU, no llama-server needed
python scripts/gen_api_doc.py             # regenerate docs/API.md after changing any tool (a test fails when it is stale)
npx vite build                            # rebuild the UI into galton_hoard/static (run from the repository root)
python -m galton_hoard                    # run on http://127.0.0.1:5201 (GALTON_DATA_DIR=<folder> for a scratch copy)
GALTON_FAKE=1 python -m galton_hoard      # demo: fake GPUs and models that answer from a table; touches no hardware
python scripts/make_icon.py               # regenerate the icon and the copies in client/public and galton_hoard/static
```

Run Python from the repository root or with absolute paths.

## Rules of the code

- The UI, the REST agent route and MCP all call the same handlers in `agent_tools.py`: add a capability there once and the three surfaces get it. The bundled UI calls tools through `POST /api/ui/call`.
- A new tool needs: a pydantic argument model with descriptions, a first description line of at most 110 characters (English, then a Spanish phrase), synonyms, annotations, a test in `tests/test_tools.py` (there is a test that fails when a tool is never called), and a regenerated `docs/API.md`. The README tool lists must stay in step.
- `checkers/` is pure: output in, `{score, passed, detail}` out. The judge and the family call arrive in the check context; when they are missing the result is `unavailable`, never a failure. Every deterministic built-in case needs a `reference` that passes its own checker (`tests/test_suites.py`).
- Built-in suites are JSON files in `galton_hoard/suites/`; changing a case changes its content hash and the suite version, so old results can be told apart. Generated suites (long context, vision) are built by `generators/` with fixed seeds.
- GPUs: only the ones in `gpus.allowed` are ever used (default 2 and 3); 0 and 1 are the owner's. Every load takes a lease per GPU first, and the index granted is checked against the allowed list again. Never add a code path that picks a GPU without going through `GpuManager`.
- Galton starts llama-server only on ports 8091 to 8099; 8080 to 8090 belong to servers that other apps scan. Children are recorded in `servers.json` and killed as a process tree.
- A result is stored with the digest of the model it was measured on; every statistic uses only the digest the model has now. Never mix stale results into a ranking silently.
- Every statistic reads results through `store.scoring_rows` / the `LIVE` fragment, which leaves out discarded runs; a new query over results must use it. Cancelling a run only ever stops llama-server children that Galton started (ports 8091-8099); never call anything that stops, kills or unloads a shared server.
- Honest errors: when something cannot be done (no GPU has room, no binary, a model without a projector) return a clear message with a hint, or store a skip reason. Never report success for work that was not done.
- `hoard_link/` is vendored and byte-identical to upstream: never edit it here. What it carries is not re-implemented in the app: the request guard, the port search and the launcher (`service.run_main`), the error envelope, the PWA files and the SPA (`service`), the SQLite layer (`sqlkit.Database`: settings are JSON values), the lane scheduler (`lanes`, wrapped by `scheduler.py`), the process helpers (`proc`), atomic writes (`atomic`), ULIDs (`ids`), accent folding and slugs (`text`), the quiet-hours window (`notify_channels`), file sniffing (`docs.sniff`), the agent kit and router (`agentkit`) and the MCP bridge (`bridge`). `GaltonError` is an `AppError`.
- User-facing text goes in `client/src/i18n.js` (Spanish first, English second). Spanish is castellano de España. Keep the UI and the docs plain: no marketing lines, no names of other products.
- Tests build everything from invented data and a fake world (`fakes.py`, `tests/helpers.py`); time is injected (`clock` fixture). Never add real names, paths or personal data.
- Tests must not depend on the computer: `hermetic_host` (autouse, `tests/conftest.py`) clears the GALTON_/OLLAMA_/HOARD_ variables, gives `HOME` an empty folder, and points the Windows default folders, llama-server on the PATH, `OLLAMA_MODELS` and the checkout's `.env` at nowhere; `tests/test_hermetic.py` checks that the real defaults still apply when something is there. A test that needs one of them sets it itself.

## Where things are

| Need | Look at |
|---|---|
| A case is scored wrongly | `checkers/*`, the case's `checker` JSON, `tests/test_checkers.py` |
| A model is not found or is duplicated | `discovery.py`, `ollama_models.py`, `gguf_meta.py`, `adhoc.py` |
| A run fails, hangs or leaves a server | `runner.py`, `servers.py`, `gpus.py`, `procs.py` (kill and process name; starting is `hoard_link.proc`) |
| A shared server changed its model, or a contestant says "no se sirve ahora" | `identity.py`, `Runner` guard (`IdentityGuard`, held results), `discovery._check_served`, `tests/test_server_identity.py` |
| A run must not count (discard, restore) or cancel says the wrong thing | `store.LIVE` / `scoring_rows`, `Services.discard_run`, `Runner.cancel`, `tests/test_tools.py`, `tests/test_cancel.py` |
| A run waits for (or interrupts) a shared server | `idle.py`, `Runner._wait_for_server` / `yield_to_others`, `tests/test_idle.py` |
| A number in the ranking looks wrong | `stats.py`, `board.py`, `store.scoring_rows` |
| The routing table | `routes.py`, the setting `routes.policy`, `watch.py` for regressions |
| A tool or its arguments | `agent_tools.py`, `docs/API.md` (a tool that waits for a run or the judge sets `timeout_s=BRIDGE_WAIT_S`) |
| Start-up, the bridge, the error envelope, the token | `__main__.py` and `main.py` (wiring only), `api/agent.py`, `mcp_server.py`, `tests/test_shared_commons.py` |
| The UI | `client/src/pages/*.jsx`, `client/src/components/*`, `client/src/i18n.js` |

## Before finishing a change

1. `python -m pytest` is green.
2. `python scripts/gen_api_doc.py` leaves `docs/API.md` unchanged.
3. If the UI changed: `npx vite build`, then open the running app (demo mode is enough) and look at every page in Spanish and English, at desktop and phone width.
4. README.md, README.es.md and docs/ARCHITECTURE.md say what the app can and cannot do now.
