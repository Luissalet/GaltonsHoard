# Galton's Hoard — agent tools

Every tool is served by the app at `GET /api/agent/tools` and `POST /api/agent/call` (Bearer token from `data/mcp-token`), by the stdio bridge `mcp_server.py`, and to the bundled UI through `POST /api/ui/call`. The argument tables are generated from the code (`python scripts/gen_api_doc.py`).

## `galton_overview`

Models, routing table, running run, notices, GPUs and next steps. Estado general del banco de pruebas.

Start here: what was measured, what is new or stale, what the routing table says, whether a run is in progress.
Sinónimos: resumen, qué modelos tengo, qué modelo uso, mediciones, pruebas, novedades, regresiones

Annotations: readOnlyHint, idempotentHint.

## `galton_status`

Health: scheduler, watch, llama-server, GPUs, settings. Estado de Galton.

Sinónimos: configuración, planificador, vigilancia, ajustes, servidor llama.cpp

Annotations: readOnlyHint, idempotentHint.

## `models_list`

List models (servers, Ollama, GGUF files) with measured/stale flags. Lista de modelos.

Filter by text, kind, vision, enabled, measured (never/stale/fresh).
Sinónimos: qué modelos hay, instalados, cuáles no he medido, cuantizaciones, visión

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `text` (string) | no | Matches name, family, quantisation, provider or alias. |
| `kind` ( \| server \| gguf) | no |  |
| `enabled` (boolean/null) | no |  |
| `vision` (boolean/null) | no |  |
| `measured` (any \| never \| stale \| fresh) | no | never: no results; stale: the file changed since it was measured; fresh: measured at its current version. |
| `include_missing` (boolean) | no | Also list models that are no longer installed. |
| `include_adhoc` (boolean) | no | Also list evaluation-only entries created from specs. |
| `limit` (integer) | no |  |

## `models_refresh`

Rediscover models: Ollama, llama-server ports, GGUF folders. Buscar modelos nuevos.

Detects new models, changed files (stale results), models that disappeared and servers that now serve another model (the old entry is marked not served, never deleted).
Sinónimos: escanear, actualizar lista, detectar modelos, descubrir

Annotations: idempotentHint, openWorldHint.

## `model_get`

One model: aliases, metadata, where it would run, scores, speed, memory. Detalle de un modelo.

Sinónimos: cuánto ocupa, qué tal rinde, alias, nombres, cuántos tokens por segundo

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `model` (string) | yes | Model id (c_…), name or any alias. |

## `model_add`

Register a server URL or a GGUF file to measure. Añadir un modelo.

A GGUF is run by Galton with its own llama-server on an allowed GPU; a server is called as it is. Remote endpoints stay off until remote_ok.
Sinónimos: añadir servidor, añadir gguf, endpoint, ruta del modelo

Annotations: idempotentHint, openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `url` (string) | no | Base URL of a running server (llama-server style or Ollama), e.g. http://127.0.0.1:8081. |
| `model` (string) | no | Model name on that server (needed when the server offers several). |
| `api` ( \| chat \| ollama) | no | chat: llama-server and other chat-completions servers. Empty: detected. |
| `path` (string) | no | Absolute path of a .gguf file Galton will run itself with llama-server (instead of url). |
| `mmproj` (string) | no | Absolute path of the vision projector for that file. |
| `name` (string) | no |  |
| `remote_ok` (boolean) | no | Remote endpoints cost money and send prompts off this computer: true only if the user explicitly said so. |

## `model_update`

Enable or disable a model, rename it, add aliases, allow a remote endpoint. Editar un modelo.

Sinónimos: activar, desactivar, renombrar, alias, permitir remoto, proyector de visión

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `model` (string) | yes |  |
| `enabled` (boolean/null) | no |  |
| `name` (string/null) | no |  |
| `aliases` (array/null) | no | Replace the list of aliases. |
| `add_aliases` (array/null) | no | Names to add: every name other apps see this model under. |
| `remote_ok` (boolean/null) | no | Allow measuring a remote endpoint (cost warning: only if the user asked). |
| `vision` (boolean/null) | no |  |
| `mmproj` (string/null) | no | Absolute path of the vision projector ('' clears it). |

## `model_remove`

Remove a model and its stored results (confirm=true). Quitar un modelo.

Installed models come back on the next refresh; disabling is the gentle way.
Sinónimos: borrar modelo, eliminar

Annotations: destructiveHint.

| Argument | Required | Description |
|---|---|---|
| `model` (string) | yes |  |
| `confirm` (boolean) | no |  |

## `suites_list`

List test suites with category and case count. Lista de suites de pruebas.

Built-in: escritura-es, razonamiento, matematicas, codigo-python, extraccion, herramientas, instrucciones, contexto-largo, rag-citas, vision, traduccion, resumen, rapida.
Sinónimos: baterías, conjuntos de pruebas, benchmarks, tests

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `category` (string) | no | One of writing_es, reasoning, math, code, extraction, tool_use, instruction, long_context, rag, vision, translation, summary, custom. |
| `builtin` (boolean/null) | no | true: only the built-in suites; false: only yours. |

## `suite_get`

One suite with its cases (prompt preview, checker, weight). Detalle de una suite.

Sinónimos: casos, preguntas, pruebas de una suite

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `suite` (string) | yes | Suite id (s_…) or name. |
| `include_cases` (boolean) | no |  |
| `limit` (integer) | no |  |
| `offset` (integer) | no |  |

## `case_get`

One case in full: prompt, tools, checker, reference answer, result count. Detalle de un caso.

Sinónimos: ver caso, pregunta completa, respuesta esperada

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `case` (string) | yes |  |

## `suite_create`

Create your own suite, optionally with cases. Crear una suite propia.

Categories other than custom feed the routing table once measured.
Sinónimos: nueva suite, mis pruebas, conjunto propio

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `name` (string) | yes |  |
| `description` (string) | no |  |
| `category` (string) | no | One of writing_es, reasoning, math, code, extraction, tool_use, instruction, long_context, rag, vision, translation, summary, custom. 'custom' suites never feed the routing table. |
| `max_tokens` (integer) | no | Default answer length for its cases. |
| `cases` (array) | no |  |

## `suite_update`

Rename or change the description, category or answer length of your suite. Editar suite.

Sinónimos: renombrar suite, cambiar categoría

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `suite` (string) | yes |  |
| `name` (string/null) | no |  |
| `description` (string/null) | no |  |
| `category` (string/null) | no |  |
| `max_tokens` (integer/null) | no |  |

## `suite_duplicate`

Copy any suite (also a built-in one) into an editable suite of your own. Duplicar una suite.

Sinónimos: copiar suite, partir de una suite

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `suite` (string) | yes |  |
| `name` (string) | no | Default: «<name> (copia)». |

## `suite_remove`

Delete one of your suites and its cases (confirm=true). Borrar suite propia.

Sinónimos: eliminar suite

Annotations: destructiveHint.

| Argument | Required | Description |
|---|---|---|
| `suite` (string) | yes |  |
| `confirm` (boolean) | no |  |

## `case_add`

Add a case (prompt + expected answer or checker) to your suite. Añadir un caso de prueba.

Save a case from a conversation when the user asks: the prompt, what a right answer contains, and how to check it.
Sinónimos: guardar esta pregunta, nuevo caso, prueba propia, respuesta esperada, comprobador

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `title` (string) | no | Empty: taken from the prompt. |
| `prompt` (string) | no | What the model is asked. |
| `messages` (array/null) | no | A whole conversation [{role, content}] instead of prompt. |
| `system` (string) | no |  |
| `expected` (string) | no | The right answer. Without a checker: a number is compared as a number, text must be contained in the answer. |
| `checker` (object/string/null) | no | A checker object ({type: exact\|contains\|regex\|choice\|number\|math_equiv\|json\|tool_call\|python_tests\|constraints\|needle\|citations\|judge\|family\|none\|all\|any, ...}) or a shorthand: exact:Madrid, contains:uno\|dos, any:a\|b, number:42, choice:B, regex:^\d+$, math:x**2, judge:rubric, none. |
| `weight` (number) | no |  |
| `tags` (array) | no |  |
| `notes` (string) | no |  |
| `tools` (array) | no | Function definitions (chat-completions tools format) offered to the model (for tool_call cases). |
| `max_tokens` (integer/null) | no |  |
| `min_context` (integer/null) | no | Skip models whose context is smaller. |
| `image_paths` (array) | no | Absolute paths of images to attach (PNG, JPEG, WebP, GIF): the case then needs a vision model. |
| `suite` (string) | yes | One of your suites (built-in suites are read-only: duplicate them first). |

## `case_update`

Change a case of your suite: prompt, expected answer, checker, weight. Editar un caso.

If the task changed, earlier results measured the old version (forget_results deletes them).
Sinónimos: corregir caso, cambiar respuesta esperada, peso

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `case` (string) | yes | Case id (k_…). |
| `title` (string/null) | no |  |
| `prompt` (string/null) | no |  |
| `messages` (array/null) | no |  |
| `system` (string/null) | no |  |
| `expected` (string/null) | no |  |
| `checker` (object/string/null) | no |  |
| `weight` (number/null) | no |  |
| `tags` (array/null) | no |  |
| `notes` (string/null) | no |  |
| `tools` (array/null) | no |  |
| `max_tokens` (integer/null) | no |  |
| `min_context` (integer/null) | no |  |
| `image_paths` (array/null) | no |  |
| `forget_results` (boolean) | no | Delete the earlier results of this case when its prompt or checker changed (they measured the old version). |

## `case_remove`

Delete a case of your suite (confirm=true). Borrar un caso.

Sinónimos: eliminar caso, quitar pregunta

Annotations: destructiveHint.

| Argument | Required | Description |
|---|---|---|
| `case` (string) | yes |  |
| `confirm` (boolean) | no |  |

## `cases_import`

Import cases from JSONL or CSV text or an absolute path. Importar casos.

Columns: prompt (required), title, system, expected, checker (full object or exact:/contains:/number:/regex:/choice:/judge: shorthand), weight, tags, notes.
Sinónimos: cargar preguntas, csv, jsonl, importar pruebas

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `suite` (string) | yes | One of your suites. |
| `text` (string) | no | JSONL (one object per line) or CSV with a header row. Columns: prompt (required), title, system, expected, checker, weight, tags, notes. |
| `path` (string) | no | Absolute path of a .jsonl, .json or .csv file instead of text. |
| `format` (auto \| jsonl \| csv) | no |  |
| `default_checker` (object/string/null) | no | Used for rows without checker and without expected answer. |

## `benchmark_import`

Import IFMTBench (instruction-following translation) as your suite. Importar benchmark de traducción.

Downloads the data from a pinned commit into the data folder, checks its SHA-256 and builds a seeded stratified sample (default 30 per constraint type plus 30 multi-constraint). Glossary rule, layout, structured data and code/tag checks are exact; style, context and a glossary the rule rejects use the judge model (settings judge.contestant) and stay pending without one. Data CC BY 4.0, scoring ported from the benchmark's Apache-2.0 code.
Sinónimos: traducción, glosario, terminología, formato, benchmark externo, seguir instrucciones al traducir, estilo, etiquetas, código

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `source` (string) | no | ifmtbench: instruction-following translation (glossary, style, context, layout, structured data, code and tags), CC BY 4.0 data. |
| `name` (string) | no | Name of the new suite (default IFMTBench). It must not exist yet. |
| `per_type` (integer) | no | Cases taken from each single-constraint type (6 types). 0: none. The whole file holds up to 1,644 per type. |
| `multi` (integer) | no | Cases taken from the multi-constraint file, spread evenly over its 5 combinations. 0: none. |
| `seed` (integer) | no | Fixes the sample: the same seed and sizes give the same cases. |
| `category` (string) | no | One of writing_es, reasoning, math, code, extraction, tool_use, instruction, long_context, rag, vision, translation, summary, custom. 'custom' keeps the suite out of the routing table; 'translation' lets it feed the translation route. |
| `refresh` (boolean) | no | Download the data again even if the verified copy is on disk. |
| `keep_unsatisfiable` (boolean) | no | Also sample items whose own reference translation fails their rule checks (about 3 in 10 of the multi-constraint structured-data ones: Markdown that is not a table). Off: they are left out and counted. |

## `case_try`

Run one case (saved or draft) on one model and show answer + checker detail. Probar un caso.

Nothing is stored. Uses the same session rules as a run (leases, allowed GPUs).
Sinónimos: probar con, prueba rápida, ver qué responde, depurar un comprobador

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `model` (string/object) | yes | A model id/name, or a spec {kind: gguf\|ollama\|server, ...}. |
| `case` (string) | no | A saved case id (k_…). Or describe a draft with prompt and checker/expected below. |
| `prompt` (string) | no |  |
| `system` (string) | no |  |
| `expected` (string) | no |  |
| `checker` (object/string/null) | no |  |
| `tools` (array) | no |  |
| `settings` (object) | no | Run settings: temperature, top_p, max_tokens, effort, repeats, seed, context, timeout_s, wait_s, device, load_local. |

## `run_plan`

Preview a run: cases per model, memory, where each would run, problems. Plan de una ejecución.

Nothing starts. GGUF files run on an allowed GPU under a lease; servers are called as they are.
Sinónimos: qué pasaría, cuánta memoria, dónde correría, dry run

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `suites` (array) | yes | Suite ids, slugs or names; ['all'] means every suite but the quick one. |
| `contestants` (array) | yes | Model ids or names, or specs: {kind: 'gguf', path: 'D:\\m\\x.gguf', mmproj?}, {kind: 'ollama', model: 'qwen3:8b'}, {kind: 'server', url: 'http://127.0.0.1:8081', model: '…'}. |
| `settings` (object) | no | Run settings: temperature, top_p, max_tokens, effort, repeats, seed, context, timeout_s, wait_s, device, load_local. effort is off\|low\|medium\|high\|max. device is auto\|gpu\|cpu: auto runs a small GGUF (file up to runner.cpu_max_gb) on the CPU when no allowed GPU is free, gpu never does, cpu always does. A model that reasons gets runner.reasoning_tokens more tokens than the answer budget, unless effort is off. load_local=false makes the run refuse to start llama-server or load a model into Ollama (it only uses what is already served). |

## `run_start`

Start a run: suites x models (ids, names or specs of a file/server). Lanzar una medición.

Queued on the GPU lane; returns the run id (use wait_s to wait). Another app can pass {kind:'gguf', path} to evaluate a file it just made.
Sinónimos: medir, evaluar, benchmark, probar modelo, ejecutar pruebas, comparar base y afinado

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `suites` (array) | yes | Suite ids, slugs or names; ['all'] means every suite but the quick one. |
| `contestants` (array) | yes | Model ids or names, or specs: {kind: 'gguf', path: 'D:\\m\\x.gguf', mmproj?}, {kind: 'ollama', model: 'qwen3:8b'}, {kind: 'server', url: 'http://127.0.0.1:8081', model: '…'}. |
| `settings` (object) | no | Run settings: temperature, top_p, max_tokens, effort, repeats, seed, context, timeout_s, wait_s, device, load_local. effort is off\|low\|medium\|high\|max. device is auto\|gpu\|cpu: auto runs a small GGUF (file up to runner.cpu_max_gb) on the CPU when no allowed GPU is free, gpu never does, cpu always does. A model that reasons gets runner.reasoning_tokens more tokens than the answer budget, unless effort is off. load_local=false makes the run refuse to start llama-server or load a model into Ollama (it only uses what is already served). |
| `label` (string) | no |  |
| `wait_s` (number) | no | Wait this long for the run to finish before answering (0: return at once with the run id). |

## `run_status`

Progress of a run: per model state, cases done, memory, errors. Estado de una ejecución.

Sinónimos: cómo va, progreso, ejecución en curso

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | no | Run id (r_…). Empty: the running one, else the latest. |

## `run_cancel`

Cancel a run: the request in flight is dropped, results kept, shared servers untouched. Cancelar ejecución.

Only a llama-server that Galton started itself for a GGUF file is stopped. A server somebody else runs (llama-server, Ollama, an endpoint) is never stopped, killed or unloaded.
Sinónimos: parar, detener medición

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | no | Run id (r_…). Empty: the running one, else the latest. |

## `run_discard`

Discard a run's results from all statistics and routes (confirm=true). Descartar una ejecución.

The run stays in the history with the reason and a discarded badge. Use it when a run measured the wrong thing, for example when the model behind a shared server changed during it. run_restore undoes it.
Sinónimos: anular resultados, invalidar medición, resultados erróneos, ignorar ejecución

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | yes | Run id (r_…) of a run that has finished, failed or been cancelled. |
| `reason` (string) | yes | Why its results cannot be trusted (for example: the model behind a shared server changed during the run). Shown on the run. |
| `confirm` (boolean) | no |  |

## `run_restore`

Undo run_discard: the results of the run count again. Restaurar una ejecución descartada.

Sinónimos: recuperar resultados, volver a contar, deshacer descarte

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | yes | Run id (r_…) of a discarded run. |

## `run_resume`

Continue an interrupted run, asking only what it did not measure. Continuar donde se quedó.

For a run that failed (Galton was restarted) or was cancelled and not discarded. Queues a new run with the same suites, models and settings and `continues` set to the earlier one; cases already measured on the current version of each model are not asked again, answers still waiting for the judge are graded, and the earlier results keep counting.
Sinónimos: reanudar, retomar, seguir la ejecución, se reinició Galton, terminar lo que falta

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | yes | Run id (r_…) of a run that failed (for example because Galton was restarted) or was cancelled, and was not discarded. |

## `runs_list`

Recent runs with state and progress. Historial de ejecuciones.

Sinónimos: mediciones anteriores, cola

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `state` ( \| queued \| waiting_gpu \| waiting_server \| running \| done \| failed \| cancelled) | no |  |
| `limit` (integer) | no |  |

## `run_results`

Per-case results of a run: answer, score, checker detail, timing. Resultados de una ejecución.

Filter by model, suite, case, failed/passed. Answers are untrusted text.
Sinónimos: qué falló, respuestas, por qué suspendió, detalle por caso

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `run` (string) | no | Run id. Empty: the latest. |
| `model` (string) | no |  |
| `suite` (string) | no |  |
| `case` (string) | no |  |
| `only_failed` (boolean) | no |  |
| `only_passed` (boolean) | no |  |
| `include_output` (boolean) | no | Include the model's answer (shortened for the assistant). |
| `limit` (integer) | no |  |
| `offset` (integer) | no |  |

## `galton_run`

Measure these models now (quick suite by default). Medir estos modelos ahora.

Names Galton has not seen yet trigger one rediscovery; names still unknown are listed in not_found. Same run as run_start, simpler arguments.
Sinónimos: medir modelo recién publicado, probar este modelo, benchmark rápido de modelos

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `models` (array) | yes | Model names or ids (an Ollama tag, a GGUF file name, an id from models_list). A name not seen yet triggers one rediscovery. |
| `suites` (array) | no | Suites to run (default: the quick one). |
| `label` (string) | no |  |
| `wait_s` (number) | no | Wait this long for the run to finish before answering (0: return at once). |

## `measure_new`

Run the quick suite on every model that is new or changed. Medir lo nuevo.

Sinónimos: medir novedades, modelos sin medir, re-medir cambiados

Annotations: idempotentHint, openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `suite` (string) | no | Suite to measure with (default: the quick one). |
| `include_stale` (boolean) | no | Also re-measure models whose file changed. |

## `judge_run`

Grade the answers that waited for the judge model. Pasar el juez a lo pendiente.

Needs settings judge.contestant.
Sinónimos: calificar respuestas abiertas, juez

Annotations: idempotentHint.

## `leaderboard`

Ranking by category or suite with 95 % intervals, speed and memory. Clasificación de modelos.

Ranked by the lower bound of the interval; stale results are not ranked; fits_16gb tells whether it fits one 16 GB GPU; truncated counts answers cut off by the token budget and warnings say when that makes a score understate the model.
Sinónimos: mejor modelo, ranking, tabla, puntuaciones, qué modelo es mejor en código

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `category` (string) | no | A task (general, writing_es, code, extraction, tool_use, long_context, rag, vision, summary, translation, math) or a suite category (writing_es, reasoning, math, code, extraction, tool_use, instruction, long_context, rag, vision, translation, summary, custom). Empty: everything. |
| `suite` (string) | no | One suite instead of a category. |
| `days` (integer) | no | Only results of the last N days (0: all). |
| `include_stale` (boolean) | no |  |
| `include_disabled` (boolean) | no |  |
| `limit` (integer) | no |  |

## `compare`

Paired comparison of two models: wins/losses, McNemar, bootstrap, verdict. Comparar dos modelos.

Verdict better / worse / no clear difference with p-value and n; a warning under 20 shared cases.
Sinónimos: es mejor, mejora la nueva cuantización, diferencia significativa, A contra B

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `a` (string) | yes | First model (id, name or alias). |
| `b` (string) | yes | Second model. |
| `category` (string) | no |  |
| `suite` (string) | no |  |
| `include_stale` (boolean) | no |  |
| `max_cases` (integer) | no | How many of the most different cases to list. |

## `gates_correlation`

Do the checks of a case fail together? phi, lift, kappa, redundant gates. Correlación entre comprobaciones.

Per pair of gates: phi, lift, kappa, P(A fails|B fails) with bootstrap 95 % intervals and a minimum-sample guard; findings in Spanish and English and which gate to drop.
Sinónimos: comprobaciones redundantes, fallos correlacionados, puertas, verificadores, juez redundante, kappa, phi

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `runs` (array) | no | Run ids to analyse together (they should share cases). Empty: every run that still counts. |
| `category` (string) | no | A task (general, writing_es, code, extraction, tool_use, long_context, rag, vision, summary, translation, math) or a suite category (writing_es, reasoning, math, code, extraction, tool_use, instruction, long_context, rag, vision, translation, summary, custom). Empty: everything. |
| `suite` (string) | no | One suite instead of a category. |
| `include_stale` (boolean) | no |  |
| `min_samples` (integer) | no | Answers a pair of gates must share before any conclusion is drawn about it. |
| `strong_phi` (number) | no | phi from which two gates count as carrying the same signal (the interval must also stay above 0.5). |
| `resamples` (integer) | no | Bootstrap resamples for the intervals. |

## `recommend`

Best measured model for a task described in words, with why and runner-up. Qué modelo usar.

Guesses the category from keywords (es/en) or takes category; says so when there is not enough evidence.
Sinónimos: recomienda un modelo, cuál uso para, el mejor para traducir

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `task` (string) | yes | What the model will be used for, in plain words (any language). |
| `category` (string) | no | Force a task category: general, writing_es, code, extraction, tool_use, long_context, rag, vision, summary, translation, math. |

## `routes_get`

The routing table: winner per task, what changes if published now. Tabla de rutas.

Other Hoard apps read the published file through Hoard Link.
Sinónimos: rutas, qué modelo usa cada app, diferencias con lo publicado

Annotations: readOnlyHint, idempotentHint.

## `routes_publish`

Publish the routing table to routes.json for Hoard Link. Publicar rutas.

Atomic write; categories with too few checked cases are left out; emits galton.routes.updated.
Sinónimos: aplicar, publicar tabla, actualizar rutas

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `note` (string) | no |  |

## `arena_next`

A blind pair of answers to the same prompt to vote on. Siguiente par de la arena.

Sinónimos: comparar a ciegas, votar, preferencia humana

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `category` (string) | no |  |

## `arena_vote`

Vote a|b|tie|both_bad on an arena pair. Votar en la arena.

Sinónimos: elegir respuesta, empate, ambas malas

Annotations: none.

| Argument | Required | Description |
|---|---|---|
| `pair` (string) | yes | Pair id from arena_next. |
| `vote` (a \| b \| tie \| both_bad) | yes |  |

## `arena_ratings`

Bradley-Terry ratings from the arena votes, per category. Puntuaciones de la arena.

Sinónimos: elo, ranking humano

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `category` (string) | no |  |

## `settings_get`

All settings with values, types and defaults, plus the GPU state. Ver ajustes.

Sinónimos: configuración, GPUs permitidas, política de rutas, juez, horas de silencio

Annotations: readOnlyHint, idempotentHint.

## `settings_set`

Change settings: allowed GPUs, llama-server path, judge, runner, watch, routes policy. Cambiar ajustes.

Allowing a GPU reserved for the owner needs confirm_reserved=true and an explicit request from the user.
Sinónimos: permitir GPU, ruta de llama-server, política, horas de silencio, ejecución de código

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `values` (object) | yes | Setting key -> value. See settings_get for the keys and types. |
| `merge` (boolean) | no | For object settings (routes.policy): change only the fields given and keep the others as they are now. Without it the fields not given go back to their defaults. |
| `confirm_reserved` (boolean) | no | Needed to allow a GPU that is reserved for the owner of this computer: only if the user explicitly said so. |

## `gpu_status`

Per GPU: total/used/free, allowed or reserved, leases and queue. Estado de las GPU.

Sinónimos: memoria de vídeo, VRAM libre, quién usa la GPU, reservas

Annotations: readOnlyHint, idempotentHint.

## `notices_list`

Notices: regressions, improvements, new or changed models, failed runs. Avisos.

Sinónimos: regresión, novedades, qué ha pasado

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `unseen` (boolean) | no |  |
| `kind` (string) | no | regression, improvement, new_model, changed_model, run_failed… |
| `limit` (integer) | no |  |
| `mark_seen` (boolean) | no | Mark the listed notices as seen. |

## `housekeeping_run`

Tidy caches, unused images and old unvoted arena pairs; stop leftover servers. Mantenimiento.

Sinónimos: limpiar, ordenar

Annotations: idempotentHint.

## `rulebench_tasks`

Rule bench tasks and the exact prompts for variant A/B/C and a pressure. Tareas del banco de reglas.

Twenty small repositories (bugs, small features, refactor traps with dead code and other people's comments) with hidden tests and a reference patch.
Sinónimos: reglas de código, prompt de sistema, variantes, alcance de cambios, presión

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `task_id` (string) | no | One task: its system and user prompt for the variant and pressure. Empty: the list. |
| `variant` (A \| B \| C) | no | A no rules, B the four code rules in their original wording, C positive and short. |
| `pressure` (neutral \| favor \| contra) | no | contra adds 'do it fast and tidy up the file' to the request. |

## `rulebench_score`

Score an answer to a rule-bench task: hidden tests, lines beyond the reference, lost comments. Puntuar.

Applies the answer's file blocks in a scratch folder and runs the hidden tests (needs checks.allow_code_execution). Also reports test-first order.
Sinónimos: evaluar cambio, líneas fuera de alcance, código muerto borrado, comentarios ajenos, test antes que código

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `task_id` (string) | yes |  |
| `answer` (string) | yes | The model's answer: one ```file:<path> block per changed file. |
| `variant` (string) | no | Label stored with the sample. |
| `pressure` (string) | no | Label stored with the sample. |

## `harness_catalog`

Harnesses, variants, tasks and tools of the harness bench. Catálogo de arneses y tareas.

Four base harnesses (native_tools, json_text, xml_text, react_text), the .long and .trim variants, command-line harnesses (off by default) and the 24 tasks with their kinds.
Sinónimos: arnés, bucle de agente, tareas con herramientas, function calling, ReAct, XML, JSON

Annotations: readOnlyHint, idempotentHint.

## `harness_matrix_start`

Run a model x harness x task matrix: model or harness at fault. Lanzar matriz de arneses.

Same model under each harness, same token cap and turn limit in every cell. Reports pass@1, format failures apart from wrong answers, needless tool calls, turns, tokens; a trial whose verifier never ran is None, not 0. Queued on the GPU lane; use wait_s to wait, dry_run to count the trials only.
Sinónimos: medir arneses, mismo modelo distinto arnés, fallo de formato, function calling nativo, ReAct, evaluar agente

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `models` (array) | yes | Model ids or names, or specs {kind: 'server', url, model} / {kind: 'ollama', model} / {kind: 'gguf', path}. |
| `harnesses` (array) | no | Harness ids: native_tools, json_text, xml_text, react_text, each with optional .long (long system prompt) and/or .trim (older observations replaced by a note), or cli:<name> for a configured command-line harness. Empty: the four base harnesses. |
| `tasks` (array) | no | Task ids (see harness_catalog). Empty: all 24, or those of `kinds`. |
| `kinds` (array) | no | Only tasks of these kinds: calc, convert, docs, files, chain, notool, recovery. |
| `k` (integer) | no | Samples per task and cell. With k above 1 the temperature defaults to 0.7 and pass@k and pass^k are reported. |
| `settings` (object) | no | The same for every cell: max_tokens (per reply, default 1024), max_turns (default 8), max_format_errors (default 3), plus the run settings temperature, top_p, max_tokens, effort, seed, context, timeout_s, wait_s, device, load_local. |
| `label` (string) | no |  |
| `wait_s` (number) | no | Wait this long for the matrix to finish before answering (0: return at once with its id). |
| `dry_run` (boolean) | no | Only count the trials and show the settings every cell will share; start nothing. |

## `harness_matrix_status`

Progress of a harness matrix: per model state, trials done, errors. Estado de la matriz.

Sinónimos: cómo va la matriz de arneses, progreso

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `matrix` (string) | no | Matrix id (m_…). Empty: the running one, else the latest. |

## `harness_matrix_result`

Harness matrix as a table with a short conclusion: harness vs model. Resultado de la matriz de arneses.

Per cell: pass@1 with interval, format failure rate, wrong-answer rate, extra tool calls, turns, tokens, None count. Conclusion: whether the difference between harnesses is larger or smaller than between models, and the best harness per model.
Sinónimos: tabla modelo por arnés, mapa de calor, mejor arnés por modelo, qué falla, el modelo o el arnés

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `matrix` (string) | no | Matrix id (m_…). Empty: the running one, else the latest. |

## `harness_cell`

One matrix cell: its tasks and, for one task, the traces and the reason of each failure. Detalle de una celda.

Every turn with the tool calls, results, refused calls and format errors.
Sinónimos: trazas, por qué falló el formato, llamadas sobrantes, depurar un arnés

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `matrix` (string) | no | Matrix id. Empty: the running one, else the latest. |
| `model` (string) | yes | Model name or id as shown in the matrix. |
| `harness` (string) | yes | Harness id of the column, for example json_text.long. |
| `task` (string) | no | A task id: adds the full traces (every turn, call, result and format failure) of its trials. |

## `harness_matrix_cancel`

Cancel a harness matrix: results kept, shared servers untouched. Cancelar matriz de arneses.

Sinónimos: parar matriz, detener

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `matrix` (string) | no | Matrix id (m_…). Empty: the running one, else the latest. |

## `harness_matrix_list`

Recent harness matrices with state and conclusion. Historial de matrices de arneses.

Sinónimos: matrices anteriores, mediciones de arneses

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `limit` (integer) | no |  |

## `agentbench_catalog`

Agent bench: benchmarks, their task subsets and the configured agent harnesses. Catálogo del banco de agentes.

Published agentic benchmarks (terminal work, repository fixes, multi-language edits, dialogues with tools, algorithmic problems), the fixed subset Galton uses of each with its dev and test split, and the harness lines of the setting agentbench.harnesses.
Sinónimos: benchmarks de agentes, arneses reales, terminal, swe, polyglot, tau, livecodebench

Annotations: readOnlyHint, idempotentHint.

## `agentbench_start`

Run real agent harnesses on published benchmarks with one model. Lanzar campaña de agentes.

Each task runs in its own container: the harness is installed, works on the task, then the task's hidden tests decide. Same tasks, same model, same limits for every harness. Trials whose verifier never ran are None, not 0, and are retried. Runs in the background; dry_run shows the subsets.
Sinónimos: comparar arneses, qué agente es mejor con este modelo, campaña, benchmark de agentes

Annotations: openWorldHint.

| Argument | Required | Description |
|---|---|---|
| `benchmarks` (array) | no | Benchmark ids from agentbench_catalog (tb2, swev, polyglot, tau3, lcb). Empty: all. |
| `harnesses` (array) | no | Harness names from the setting agentbench.harnesses. Empty: every line that is switched on. |
| `model` (string) | no | Model id the server knows. Empty: the setting agentbench.model. |
| `base_url` (string) | no | Address of the server that serves it (common chat-completions protocol, ending in /v1). Empty: the setting agentbench.base_url. |
| `k` (integer) | no | Attempts per task and harness. |
| `splits` (array) | no | dev: the tasks a harness may be tuned on; test: only measured. |
| `label` (string) | no |  |
| `slots` (integer) | no | Trials in flight at once (0: the setting). |
| `adopt` (string) | no | Name of a campaign whose jobs another process runs: only read its job folders, launch nothing. |
| `dry_run` (boolean) | no | Only show the task subsets of each benchmark; start nothing. |

## `agentbench_status`

Progress of an agent-bench campaign. Estado de la campaña de agentes.

Sinónimos: cómo va la campaña, progreso del banco de agentes

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `campaign` (string) | no | Campaign id or name. Empty: the running one, else the latest. |

## `agentbench_result`

Agent-bench board: pass rate per benchmark and harness, with ranking. Resultado de la campaña.

Per cell: verified trials, pass rate with its 95 % interval, None trials apart, failure reasons, tokens, time. Per benchmark: ranking, the focus harness's paired wins and losses against each other harness with an exact sign test.
Sinónimos: clasificación de arneses, quién gana, mapa de calor, comparación por tarea

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `campaign` (string) | no | Campaign id or name. Empty: the running one, else the latest. |
| `split` ( \| dev \| test) | no | Only the trials of that split (empty: all). |
| `focus` (string) | no | Harness compared, task by task, against every other one. |

## `agentbench_cell`

One cell of a campaign: every trial of a harness on a benchmark. Detalle de una celda.

Sinónimos: qué tareas falló, por qué falló

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `campaign` (string) | no |  |
| `benchmark` (string) | yes |  |
| `harness` (string) | yes |  |

## `agentbench_trial`

One trial with the logs the harness and the verifier left. Detalle de una prueba con sus registros.

Sinónimos: log del agente, salida de los tests, depurar arnés

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `trial` (integer) | yes | Trial id from agentbench_cell. |
| `limit` (integer) | no | Characters kept from the end of each log file. |

## `agentbench_sync`

Read a campaign's job folders again into the database. Releer las carpetas de la campaña.

Sinónimos: actualizar resultados

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `campaign` (string) | no | Campaign id or name. Empty: the running one, else the latest. |

## `agentbench_cancel`

Cancel an agent-bench campaign: trials already finished are kept. Cancelar campaña de agentes.

Sinónimos: parar campaña

Annotations: idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `campaign` (string) | no | Campaign id or name. Empty: the running one, else the latest. |

## `agentbench_list`

Recent agent-bench campaigns. Historial de campañas de agentes.

Sinónimos: campañas anteriores

Annotations: readOnlyHint, idempotentHint.

| Argument | Required | Description |
|---|---|---|
| `limit` (integer) | no |  |

## REST routes for the UI

- `GET /api/health`, `GET /api/status`
- `GET /api/dashboard` — the routing table as it stands, the run in progress, the queue, notices, what was never measured or went stale, the GPUs and the scheduler.
- `POST /api/dashboard/visit` — marks the notices seen.
- `GET /api/runs/{id}/events?after=N` — the run (state, progress per model) and the results written after result id `N`, without the answer texts; the run page polls it while the run is live.
- `GET /api/vision/{case}/{n}.png` — image `n` (from 1) of a case: a stored image or the one a vision case generates at run time; same-origin only.
- `POST /api/ui/call` `{name, arguments}` — any tool above, uncapped (the UI is local).
- `GET /manifest.webmanifest`, `GET /sw.js` — installable app.
