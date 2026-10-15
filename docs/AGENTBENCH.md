# Agent bench: real harnesses on published benchmarks

The harness bench ([HARNESSBENCH.md](HARNESSBENCH.md)) measures Galton's own adapters on 24 invented tasks. The agent bench measures the real thing:
complete agent programs (the coding agents people use, and Faustus) driving the same model on published agentic benchmarks, each task in its own
Linux container, decided by the task's own hidden tests. It answers one question: with this model, which harness gets the most work done?

## How a trial runs

Galton does not run agents itself. It drives a container trial runner (`harbor`, the runner of Terminal-Bench, Apache-2.0) that knows how to install
dozens of agents and how to read dozens of benchmarks. For each task the runner:

1. builds or pulls the task's container image and starts it;
2. installs the harness inside (each harness its own way: a package manager, an installer script, or a prepared bundle);
3. gives the harness the task text and lets it work until it stops or the task's time runs out;
4. copies the task's hidden tests in and runs them; they write a reward (1 is a pass);
5. writes `result.json` and the logs of the harness and of the tests in the trial's folder.

Galton reads those folders. Nothing it stores comes from inside the container.

## Benchmarks

Each benchmark is a dataset of the runner's registry plus a fixed recipe for the subset Galton uses: filters on the task metadata, a seed, how
many tasks, and which of them form the **dev** split. The subset depends only on the task list and the recipe, so every harness and every later
run gets the same tasks.

| id | dataset | tasks (dev + test) | what the tasks are |
| --- | --- | --- | --- |
| `tb2` | terminal-bench/terminal-bench-2 | 8 + 16 | terminal work: build, configure, debug, data; easy and medium tasks whose own time limit is at most 30 min |
| `swev` | swe-bench/swe-bench-verified | 8 + 16 | a real issue of a real Python repository, fixes estimated under an hour |
| `polyglot` | aider/aider-polyglot | 10 + 20 | Exercism exercises in C++, Go, Java, JavaScript, Python and Rust, five per language |
| `tau3` | sierra-research/tau3-bench | 8 + 16 | retail and airline customer-service dialogues with a simulated user, a policy and tools served over MCP |
| `lcb` | livecodebench/livecodebench | 8 + 16 | competitive programming problems with hidden tests |

The dev split is where a harness may be tuned: its failures can be read and its code changed. The test split is only ever measured. A harness that
gains on dev but not on test was tuned to the tasks, not improved.

## Harnesses

Nothing about any particular agent is built in. A harness is one line of the setting `agentbench.harnesses`:

```
name = agent [model=TEMPLATE] [path=FOLDER] [setup=N] [mcp=no] [ak.KEY=VALUE ...] [env.KEY=VALUE ...]
```

- `agent`: an agent the runner knows by name, or an import path `package.module:Class` of an agent written for it;
- `model`: how that agent wants the model written; `{model}` is replaced by the campaign's model (by default the runner's provider prefix for any
  server that speaks the common chat-completions protocol, then `/{model}`);
- `path`: a folder put in front of `PYTHONPATH`, so an agent given by import path can be found;
- `setup`: multiplies the time the runner gives the agent to install itself;
- `mcp=no`: the agent cannot use tools served over MCP; benchmarks that need them (`tau3`) skip it instead of scoring it zero;
- `ak.KEY=VALUE`: a keyword argument of the agent (a JSON value is read as JSON); `env.KEY=VALUE`: an environment variable for the job.

In values, `{model}`, `{base_url}` (ending in `/v1`), `{root_url}` (the same without `/v1`) and `{api_key}` are replaced. A line that starts with
`#` is switched off. Every harness runs with its own defaults; only what it needs to reach the model is set.

## Outcomes, and why None is not zero

| outcome | when | counted |
| --- | --- | --- |
| `pass` | the tests ran and gave reward 1 | yes |
| `fail` (`tests_failed`) | the tests ran and gave less | yes |
| `fail` (`timeout`) | the harness was still working when the task's time ran out (the tests decide on what it left, when they ran) | yes |
| `fail` (`agent_error`) | the harness started working and then stopped with an error of its own | yes |
| `none` | the trial says nothing about the harness: the container could not be built, the harness could not install itself, the model server did not answer | no |

A `none` trial is left out of every rate and launched again, up to `agentbench.retries` times. A verified trial is never run twice.

## What the board says

- per cell (benchmark x harness): verified trials, pass rate and its Wilson 95 % interval, tasks solved, `None` count and reasons, failure reasons,
  mean tokens and mean agent time;
- per benchmark: the ranking, and the focus harness (Faustus by default) against every other one task by task: tasks it solves and the other does
  not, the reverse, and the exact two-sided sign-test p-value of that split;
- a verdict per benchmark for the focus harness: `first`, `tied_first`, `behind` (with its rank) and the lead in points, with `weak evidence` when
  a lead rests on a split whose p-value is above 0.05.

Read a small lead as a lead on this sample, not as a law.

## Campaigns

A campaign is benchmarks x harnesses x k attempts with one model. It runs in a background thread of the app, not on the GPU lane: the model is
served elsewhere. The thread keeps `agentbench.slots` trials in flight (`agentbench.per_cell` per cell), benchmark by benchmark and harness after
harness, and copies every finished trial into the database (`ab_campaigns`, `ab_trials`). The job folders under `agentbench.jobs_dir` are the
source of truth: a campaign interrupted by a restart is started again and only launches what is missing.

`adopt` registers a campaign whose jobs another process runs (the standalone script): Galton only reads its folders.

## Use it

From the app: screen Agentes (`#/agentes`), settings in Ajustes, section Banco de agentes. From an assistant: `agentbench_catalog`,
`agentbench_start`, `agentbench_status`, `agentbench_result`, `agentbench_cell`, `agentbench_trial`, `agentbench_sync`, `agentbench_cancel`,
`agentbench_list`.

Without the app:

```
python scripts/agentbench_run.py --plan --bench tb2,swev --datasets D:/bench/datasets
python scripts/agentbench_run.py --name c1 --model MODEL --base-url http://HOST:PORT/v1 --harnesses harnesses.txt \
    --bench tb2,swev,polyglot --slots 8 --jobs D:/bench/jobs --datasets D:/bench/datasets --report c1.json
```

## Limits

- A trial needs Docker with Linux containers and network access for the harness's own installer; image pulls for SWE-bench tasks are large.
- Time limits are wall-clock: a slow model server or many trials at once lower every harness's results together, so compare harnesses inside one
  campaign, not across campaigns run under different loads.
- Subsets of 24 to 30 tasks give intervals of about 20 points; differences smaller than that need more attempts (`k`) or more tasks.
- The runner's adapter for a harness can fall behind the harness's own releases; when one cannot install itself, every trial ends `none` and the
  cell shows it instead of a zero.
