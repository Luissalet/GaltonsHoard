# Galton's Hoard

A local test bench for the language and vision models on your computer. It gives each model tasks whose answers can be checked automatically (reasoning, maths, Python code with hidden tests, JSON extraction, tool calling, instruction following, retrieval in long text, answers with citations, images, translation, summaries and Spanish writing, plus cases you add yourself), stores every answer with its timing and memory use, ranks and compares the models with confidence intervals and paired tests, notices when a model is new or has changed, and publishes a routing table that says which measured model is best for each kind of task.

Part of the Hoard family of local apps: it runs on your PC, keeps its data in `data/`, works on its own in the browser and can be driven by an assistant over MCP.

[Versión en español](README.es.md)

## What it does

- **Finds your models.** Models already served by Ollama, by llama-server (ports 8080 to 8090) or listed in the Faustus registry, the GGUF files in the folders you configure (default `D:\LocalAI\models` when it exists) and the Ollama models whose file is on disk. For an Ollama model both ways of running are offered: through the Ollama API, or on Galton's own llama-server using the same file (the default when the file is found). On Windows the Ollama store is found through `OLLAMA_MODELS` from the environment or, when the app was not started from a shell that has it, from the user and machine variables in the registry; the folder whose `manifests` really hold models wins. Aliases are only names a server or an app can report (Ollama tags, the llama-server alias, the id from `/v1/models`, a GGUF file stem); file paths and `sha256-…` blob names stay in their own fields and are removed from aliases stored by earlier versions on start. A model whose file changed is detected by its digest and its old results are marked stale, never mixed in. A llama-server that loads an Ollama blob runs the same weights as the Ollama tag: both entries answer to each other's names and the model page lists them as the same weights. A server that answers `/v1/models` but cannot chat (no chat template, or an error to one test request) is marked as not a chat model once, switched off and never announced as a new model. The identity of a contestant on a shared server is its address plus what the server says it serves (ids of `/v1/models`, alias and model file of `/props`, or the Ollama digest). A run checks it before the contestant starts and again before every case (the look is cached about 20 s) and compares the `model` field of each reply; if somebody restarted the server with another model the contestant stops with an error and nothing measured after the change is stored. `models_refresh` marks a contestant whose address now serves another model as not served ("no se sirve ahora"): it keeps its history, run forms exclude it with the reason, and it comes back when the server serves it again; the new model gets its own contestant.
- **Measures with checkable tasks.** 225 cases in 13 built-in suites, in Spanish of Spain unless the task is about another language. Fifteen kinds of checker: exact text, contains, regular expression, multiple choice, number (Spanish and English formats, tolerance; after an answer marker such as «Respuesta:» it reads the result of a calculation on that line, so `Respuesta: 2^10 = 1024` is 1024; when that line has no digits it reads numbers written with words and ordinals in Spanish and English, so `Respuesta: El octavo día` is 8), mathematical equivalence (LaTeX such as `\(3x^2\ln(x)+x^2\)` is normalised before the symbolic comparison, and only the part after the last `=` of the answer line counts), JSON with a schema, tool call, Python with hidden tests, instruction constraints, needle in a long text, citations, a judge model with a rubric, a call to another Hoard app, and none (kept for the arena). A word or constraint check can be limited to the final answer (`scope: "answer"`: only the text after the last «Respuesta:» or «Answer:» line), and a field of an extraction case can be compared as `contains` or `norm` (ignoring a leading generic word such as sala, calle or avda.), so a model that says `sala Magallanes` is not failed for `Magallanes`. Every deterministic built-in case carries a reference answer and a test checks that it passes its own checker.
- **Gives reasoning models room to think.** Whether a model reasons is read from the chat template of llama-server (`enable_thinking` or `<think>`), from the capabilities Ollama reports (`thinking`), or learned when an answer arrives with reasoning content. For such a model, unless the run's effort is `off`, the output limit is the answer budget of the case plus `runner.reasoning_tokens` (8192 by default, 0 switches it off; a bigger effort asks for more), and the timeout grows with it. Effort `off` sends `enable_thinking: false` to llama-server and `think: false` to Ollama. A result whose visible answer is empty because the limit was reached is stored as `truncated`; a model not known to reason that shows it is asked once more with the room and remembered. The run page and the leaderboard show how many results were truncated per model, and the leaderboard, the routing table and `recommend` warn when more than 5 % of a model's results in a category are (the budget was too small, so the score understates the model).
- **Runs models on the GPUs you allow.** A GGUF file runs on a llama-server started by Galton on a free port in 8091 to 8099, under a lease for the estimated memory (file size plus KV cache plus headroom), on one allowed GPU or split across several, and the process tree is killed afterwards. GPUs 0 and 1 are reserved by default and allowed GPUs are 2 and 3; reserved GPUs cannot be allowed without an explicit confirmation. Running servers are called as they are, but never while somebody else is using them: a busy server puts that model in the state `waiting_server` (the run page and `run_status` show how long it has waited), and the run starts once the server has been quiet for `runner.idle_grace_s` (20 s by default); it gives up only after `runner.wait_idle_max_s` (an hour by default, 0 waits for ever; a run's `wait_s` overrides it). Between two questions it looks again and pauses the same way if someone else started a chat (llama-server shows its slots; for Ollama the activity is inferred from the `expires_at` of `/api/ps`, so a request still in flight cannot be seen); an Ollama model that is not loaded is only loaded if you enable it.
- **Falls back to the CPU for small files.** When a GGUF fits no allowed GPU right now (every GPU taken by another workload, or none of the allowed ones present) and its file is at most `runner.cpu_max_gb` (4 GB by default), Galton starts its own llama-server with `-ngl 0`, `CUDA_VISIBLE_DEVICES=""` (it never touches a GPU), `-t` = physical cores minus 2 (at least 2) and no lease, instead of waiting for a GPU: a model that could take a GPU is still given one when it is free. The run setting `device` is `auto` (this behaviour), `gpu` (never the CPU: exactly the old behaviour, including the wait and the error) or `cpu` (always; the size limit does not apply). `runner.cpu_fallback` (on by default) switches the automatic choice off. The CPU needs the memory too: with too little free RAM the old GPU error stays. The plan says where each model would run ("CPU (no hay GPU libre)"), the run card and the results are flagged as CPU, and a CPU speed is never mixed with GPU speeds: for one model the leaderboard, `compare` and the routes use its GPU measurements when it has any (the CPU figure is shown apart), otherwise its CPU ones with a "CPU" badge; a CPU-only speed does not count in the speed weight of the ranking. The watch never puts a background run on the CPU.
- **Says how sure it is.** Pass rate with a Wilson 95 % interval; mean score with a seeded bootstrap 95 % interval, weighted by case weight; two models are compared on the cases they share with an exact McNemar test and a paired bootstrap, and the verdict is one of better, worse, no clear difference or no data. Speed (tokens per second, time to first token, load time) and memory come from the runs, not from guesses.
- **Publishes the routing table.** For each task category the candidates are the enabled models with recent results for their current file; remote endpoints are left out unless allowed; too slow or too large models are excluded; the rest are ranked by the lower bound of the 95 % interval, with decode speed as tie-break. Each choice has a sentence that explains it. The table is written atomically to `routes.json` (`HOARD_ROUTES_FILE` or `~/.hoard/routes.json`) and Hoard Link reads it, so the other apps of the family use the measured best model. A category with fewer checked cases than the policy asks for is never published. See what would change before publishing.
- **Watches.** New or changed models are measured with the quick suite when that disturbs nobody (a loaded server idle for 10 minutes, or a GGUF that fits on a free allowed GPU; never in quiet hours, 01:00 to 08:00 by default). After each run it compares a model with the results it had before its file changed and raises a notice and the family event `galton.regression` when it got significantly worse.
- **Reports its work to the family.** Every run emits the canonical job events (queued, started, progress with the GPU and an estimate, done, failed, cancelled), so the hub shows it next to the jobs of the other apps. `galton_run {models}` queues the quick suite on the named models (a name it has not seen yet triggers one rediscovery first; names it still cannot find are listed, never ignored); it is what the hub calls after a model is published.
- **Arena.** Two anonymous answers to the same prompt, taken from stored runs (no model time spent), you vote, and Bradley-Terry ratings are computed per category.
- **Your own suites.** Create a suite, add cases by hand, or import JSONL or CSV (columns may be in Spanish or English; a one-line checker such as `contains:uno|dos` or `number:42` is enough). Built-in suites are read-only and can be duplicated. `case_try` runs one case on one model without saving anything.
- **Judge.** A model you choose grades open answers against a rubric on a 0 to 10 scale. It never silently grades itself (such results are marked `self_judged`), grades are cached by what the judge saw, a judge that is the model being measured grades only after that model's last case (so grading never delays a timed question), and answers wait as pending, not failed, while the judge is not available.

## Screens

Spanish by default, English with one click, dark.

- **Panel**: routing table with score, interval, speed and memory; the run in progress; notices; the GPUs with their leases; buttons to measure what is new and to publish the routes.
- **Modelos**: every model with kind, family, size, quantisation, context, whether it fits in 16 GB, last measurement and stale marks; enable, rename, add aliases, add a file or a server.
- **Pruebas**: suites and cases with their checkers; create, duplicate, import and try a case.
- **Ejecutar**: choose suites and models, see the plan (cases, memory, where each model would run, problems), start, follow the progress case by case, cancel (the message says what is stopped: only a llama-server that Galton started itself; a shared server is never touched), continue a run that failed (Galton was restarted) or was cancelled, asking only the cases it did not measure, discard a finished run with a reason (it stays in the history with a badge but no statistic, ranking, route, comparison or regression check uses it; restore undoes it), read every answer with the checker detail.
- **Clasificación**: ranking by category or suite with interval bars, speed and memory columns, and the paired comparison of two models with the list of cases where they differ.
- **Arena**: blind votes and ratings.
- **Rutas**: the routing table, the difference with what is published, the history of publications.
- **Ajustes**: allowed GPUs, llama-server path, judge, runner limits, regression watch, routing policy, language.

## What it cannot do

- It measures what its cases check. A high score on a suite says little about tasks that are not in the suite; add cases of your own for the work you care about.
- Galton's own llama-server runs on the CPU only for files up to `runner.cpu_max_gb` (or when a run says `device: cpu`), and its speed is not comparable with GPU speeds. A GGUF over that size that does not fit in the allowed GPUs together is reported as impossible with the numbers, not attempted.
- It needs `llama-server` (llama.cpp) installed to run GGUF files itself; without it only servers that are already running can be measured.
- Judge grades are as good as the judge model you choose, and an answer that is still waiting for the judge does not count until it is graded. A request that fails because the server broke is stored with its error and left out of the score; it is never counted as a wrong answer.
- Python cases run model-written code in a separate interpreter with a timeout and a cleaned environment. That is not a security sandbox; switch it off with the setting `checks.allow_code_execution` if you do not want it.
- Vision cases need a model with its projector file; models without one are skipped with the reason stored.
- It does not train, quantise or download models, and it does not call remote endpoints unless you enabled that model one by one.
- The room given to thinking is a limit, not a measurement of how much a model thinks; a model that needs more than `runner.reasoning_tokens` is still cut and shown as truncated.
- Small samples give wide intervals. The app shows them instead of hiding them and never publishes a category with too few cases.

## Install

Requirements: Python 3.11 or newer (tested on 3.11 and 3.13), Node 22 only to rebuild the UI, a llama-server binary for GGUF files and Ollama if you use it.

```bash
python -m venv venv
venv/Scripts/python -m pip install -r requirements.txt      # Windows; venv/bin/python elsewhere
```

The UI is prebuilt in `galton_hoard/static`. To rebuild it: `npm install && npx vite build`.

## Run

```bash
venv/Scripts/python -m galton_hoard                           # http://127.0.0.1:5201
python scripts/launch.py                                      # free port, opens the browser
```

Demo mode shows the whole app without touching hardware: `GALTON_FAKE=1 python -m galton_hoard` uses invented GPUs and three models that answer from a table, and publishes to `data/routes-demo.json`.

The UI is Spanish or English; the messages of the backend (errors with their hints, plan lines, run warnings, notices) reach it as a stable key plus parameters and are formatted in the chosen language, while assistants always get the English sentence. Without a console (`pythonw`) the start-up messages go to the log only.

Environment: `GALTON_PORT` (5201), `GALTON_DATA_DIR`, `PORT_STRICT=1`, `GALTON_ALLOWED_HOSTS`, `GALTON_HTTP_TIMEOUT_S`, `GALTON_SCHEDULER=0` (no background jobs), `GALTON_OFFLINE=1` (no discovery), `GALTON_FAKE=1` (demo), `HOARD_ROUTES_FILE`, and `GALTON_FAUSTUS_TOKEN` in `.env` (see `.env.example`). Everything else is a setting stored in `data/galton.db` and changed in the app.

## Assistants (MCP)

`mcp_server.py` is a stdio MCP bridge named `galton-hoard`. It never opens the database: it proxies every call to the running app with the token in `data/mcp-token`, and starts the app when it is not answering. `faustus-plugin.json` describes the app, its health check and the bridge for Faustus and the Hoard Hub.

Tools (45): `galton_overview`, `galton_status`, `models_list`, `models_refresh`, `model_get`, `model_add`, `model_update`, `model_remove`, `suites_list`, `suite_get`, `case_get`, `suite_create`, `suite_update`, `suite_duplicate`, `suite_remove`, `case_add`, `case_update`, `case_remove`, `cases_import`, `case_try`, `run_plan`, `run_start`, `run_status`, `run_cancel`, `run_discard`, `run_restore`, `run_resume`, `runs_list`, `run_results`, `measure_new`, `galton_run`, `judge_run`, `leaderboard`, `compare`, `recommend`, `routes_get`, `routes_publish`, `arena_next`, `arena_vote`, `arena_ratings`, `settings_get`, `settings_set`, `gpu_status`, `notices_list`, `housekeeping_run`. Arguments in [docs/API.md](docs/API.md).

Events on the family bus: `galton.routes.updated` (the published table changed), `galton.regression` (a model got worse), `galton.models.updated` (new, changed or missing models) and the canonical job events of every run, `galton.job.queued|started|progress|done|failed|cancelled` with `{job_id, title, kind: "run", progress, gpu, eta_s, url, error}` (progress events at most every 5 seconds). The hub's Work tab shows them, and a rule of the hub calls `galton_run {models}` after a model is published, so a freshly published model is measured without asking.

## Data and privacy

Everything lives in `data/` (or `GALTON_DATA_DIR`): `galton.db` (models, suites, runs, results, settings), `images/` (images attached to your cases), `logs/` (one llama-server log per model, and `galton.log`, a rotating log of the app itself, which is what remains when it is started without a console), `cache/`, `servers.json` (the llama-server processes Galton started, so a crash cannot leave a model loaded: they are stopped at the next start), `mcp-token` and `url`. Nothing leaves the computer: the app listens on 127.0.0.1 only, rejects cross-site requests, and calls only local servers unless you enable a remote endpoint for a model. The prompts and answers of your own cases are stored as they are; do not put secrets in them. Back up the folder to back up the app.

## Development

```bash
python -m pytest                           # no network, no GPU, no llama-server needed
python scripts/gen_api_doc.py              # regenerate docs/API.md after changing a tool (a test fails when it is stale)
npx vite build                             # rebuild the UI into galton_hoard/static
```

See [AGENTS.md](AGENTS.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The tests use a fake world (fake GPUs and leases, models that answer from the reference answers, a fake launcher) and mocked HTTP; nothing in them touches real hardware.

## License

MIT
