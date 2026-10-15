# Harness bench: model x harness x task

Measures the same model under several ways of talking to it so that it uses tools, and tells a failure of the model from a failure of the harness
that drives it. The same model can move many points depending on the harness: one that only emits the server's native function calls fails every
task under a harness that expects `<tool_call>` text, and the reverse. A leaderboard that measures each model under one harness mixes the two.

## Harnesses

Four interchangeable adapters around one agent loop. Each says how the tools are offered, how a call is read from the reply and how a result is
sent back.

| id | The model calls a tool by | Notes |
| --- | --- | --- |
| `native_tools` | the server's function calling (`tools` in the request, `tool_calls` in the reply, `tool` messages back) | A server that refuses the `tools` field is a format failure of this harness (`tools_unsupported`), not a server error, and there is no silent fallback to a prompt. |
| `json_text` | writing a JSON object `{"tool": ..., "args": {...}}` in its text | The tools are described in the system prompt. |
| `xml_text` | writing `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` | The convention of the Qwen and Hermes families. |
| `react_text` | `Thought` / `Action` / `Action Input` lines, ending with `Final Answer` | `Action` and `Final Answer` in one reply is a format failure; a reply with neither is taken as the final answer. |

Variants are suffixes of the id and combine: `json_text.long.trim` is valid.

- `.long`: a long system prompt (rules about when to use a tool, how to read errors, how to answer) instead of the short protocol only.
- `.trim`: history compaction. Before each request every tool result except the last two is replaced by a note with its length.

A reply without a tool call is the final answer. A reply that looks like a call and cannot be read, or a call the harness refuses before running it
(unknown tool, missing or mistyped argument, arguments that are not valid JSON), is a **format failure**: the model is told why and may try again,
up to `max_format_errors` (default 3), and the trial is counted apart from a call that ran and was merely wrong.
An empty reply (no call and no answer, as when a server ends the turn on a special token such as a model's own tool-call token) is
also a format failure, `empty reply` in the trace, with the model's reasoning when the server returned it.

### Command-line harnesses

Any agent that can be started with a command can be measured as one more column, with id `cli:<name>`. They are configured in the setting
`harnessbench.external`, one line per harness, `name = command and arguments`; a line that starts with `#` is switched off. Nothing is installed and
nothing runs by default: the shipped line is a commented example and `harnessbench.allow_external` is off.

```
# example = terminal-agent --base-url {base_url} --model {model} --prompt-file {prompt_file} --workdir {workdir}
```

The command runs without a shell in a scratch folder that holds the task's files and the documentation store (`docs/<id>.txt`). In its arguments
`{prompt}`, `{prompt_file}`, `{base_url}` (ending in `/v1`), `{model}` and `{workdir}` are replaced; the same values are exported as `GALTON_PROMPT`,
`GALTON_PROMPT_FILE`, `GALTON_BASE_URL`, `GALTON_MODEL` and `GALTON_WORKDIR`. The last five lines it prints are the final answer and the files it
leaves in the folder are what the verifier reads. Only tasks that can be checked that way take part (`external` in the catalogue). The bench cannot see
the agent's own turns, calls or format failures: those metrics are `None` for these cells, never zero. A command that cannot be started, or exits with an
error and prints nothing, gives `None` for the trial; one that does not finish within `harnessbench.external_timeout_s` is a failed answer.

## Tasks

24 deterministic tasks over eight tools (`calculator`, `convert`, `search_docs`, `read_doc`, `list_dir`, `read_file`, `write_file`, `lookup_record`)
that act on an in-memory world: an invented documentation store, a virtual file system and a record table. Nothing touches the disk or the network.

| kind | tasks | what it needs |
| --- | --- | --- |
| `calc` | 4 | the calculator (the expression language has no `^`, and refuses anything that is not arithmetic) |
| `convert` | 3 | unit conversion: length, mass, time, volume, temperature |
| `docs` | 4 | search by words (ids and titles only), then read the document |
| `files` | 4 | list, read and write files; some are judged on the final state of the files, not on the words |
| `chain` | 4 | two to four dependent calls |
| `notool` | 2 | no tool at all; the tools are offered anyway and a call is an extra |
| `recovery` | 3 | a tool error to recover from: a failing first call, a wrong path, a wrong document id |

Every task carries a programmatic verifier and an oracle path (the calls that solve it); a test checks that the oracle passes every task in every
harness variant with no extras. Every cell shares the same output-token cap per reply (`max_tokens`, default 1024), turn cap (`max_turns`, default 8),
format-error cap and temperature (0 for k=1, 0.7 above), and the same task list.

## What a cell reports

| field | meaning |
| --- | --- |
| `pass_at_1`, `ci` | share of verified trials that passed, with the Wilson 95 % interval |
| `pass_at_k`, `pass_hat_k` | per task, with k > 1: some of the k samples pass, and all of them pass |
| `format_fail_rate` | share of trials that ended as a format failure (`format_limit`, `tools_unsupported`, or out of turns on an unreadable call) |
| `format_error_rate` | share of trials with at least one unreadable or refused call, even if the model recovered |
| `answer_fail_rate` | share of trials with a wrong final answer, a truncated reply or no answer within the turn cap |
| `extra_calls_mean`, `extra_rate` | calls beyond what was needed: made after the task was already solved, or identical to an earlier successful call |
| `turns_mean`, `calls_mean`, `tokens_in_mean`, `tokens_out_mean`, `latency_ms_mean` | means over verified trials |
| `none_n`, `none_reasons` | trials without a result |

### None is not zero

A trial whose verifier never ran (server timeout, connection error, a model that stopped being the one that was asked for, a command that could not
start) has outcome `none` and `passed = null`. It is excluded from every rate and every denominator and reported apart: a cell shows `n` (verified
trials) and `None N`. A cell with nothing left shows no number at all, and it does not enter the margins or the conclusion. After four infrastructure
errors in a row the rest of that model's trials are stored as `None` and the matrix moves on.

## The automatic conclusion

Computed from the cells that have a number, in English and Spanish:

- how much a model's pass@1 moves when only the harness changes (the spread between its best and worst harness, averaged over models) against how much
  the models differ under one harness (the spread between best and worst model, averaged over harnesses). The verdict is `harness_larger`,
  `model_larger`, `similar` (within 2 points) or `not_comparable` (one model or one harness only);
- the best harness per model (ties go to fewer format failures, then fewer tokens) and, per model, whether it `depends on the harness` (15 points or
  more), is `stable`, or `fails whatever the harness` (best pass@1 under 50 %);
- a warning of weak evidence when a cell has fewer than 20 verified trials, and the number of `None` trials left out.

The text is a reading of the numbers, not a test of significance: look at the intervals before acting on a small difference.

## Run it

From the app: screen Arneses (`#/arneses`), or the tools `harness_catalog`, `harness_matrix_start`, `harness_matrix_status`, `harness_matrix_result`,
`harness_cell`, `harness_matrix_cancel`, `harness_matrix_list`. A matrix is a job on the GPU lane: it waits for the models' servers and GPU leases like a
run, runs one model at a time, reports the canonical job events and is stored in the database (`hb_matrices`, `hb_trials`). A matrix that was running
when the app stopped is marked failed with its stored trials kept; one that was still queued is queued again.

A small real matrix against a server that is already running (replace the model name with one the endpoint knows):

```
harness_matrix_start  {"models": ["MODEL"], "harnesses": ["native_tools", "json_text"], "kinds": ["calc", "notool"], "k": 1, "dry_run": true}
harness_matrix_start  {"models": ["MODEL"], "harnesses": ["native_tools", "json_text"], "kinds": ["calc", "notool"], "k": 1, "wait_s": 300}
harness_matrix_result {}
harness_cell          {"model": "MODEL", "harness": "json_text", "task": "calc-sum-product"}
```

Without the app, against any chat-completions compatible endpoint (`--url` has no `/v1`; point it at a server nobody else is using):

```
python scripts/harnessbench_eval.py --dry-run
python scripts/harnessbench_eval.py --url http://127.0.0.1:8081 --model NAME --harnesses native_tools,json_text,xml_text --kinds calc,notool --out matrix.json
python scripts/harnessbench_eval.py --model A@http://127.0.0.1:8081 --model B@http://127.0.0.1:8082 --k 3 --out matrix.json
```

The report has the matrix, the conclusion and every trial with its trace; the table and the conclusion are also printed to stderr.

## Limits

- 24 tasks over invented data are a probe of how a model and a harness fit together, not a measure of general ability; a cell of 24 trials has a wide
  interval. Use `k` above 1 and more cells before drawing conclusions.
- The harnesses are Galton's own adapters. They are close to the common conventions but are not the code of any particular client, so a result
  describes the convention, not a product.
- Trials of one model run one after another, and the models of a matrix one after another; the matrix is not a speed benchmark.
- A model that reasons before answering uses the reasoning allowance of the runner settings, like in a run; the cap that is the same in every cell is
  the visible output.
- The command-line harness path was written against the interface above and checked with harmless commands; it has not been run against a real
  third-party agent.
