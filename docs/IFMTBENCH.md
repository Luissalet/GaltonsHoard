# IFMTBench: translation that obeys instructions

IFMTBench is a benchmark of instruction-following machine translation. Each item is a translation request whose instruction carries a constraint: a glossary to apply, a style to write in, a background that settles an ambiguity, a layout to keep, structured data to leave intact, or code and tags to leave untouched. Galton imports a sample of it as an ordinary suite, asks the models, and scores each answer the way the benchmark does.

Credit and licence: IFMTBench is Copyright (C) 2026 Tencent. The data is under CC BY 4.0, the code under Apache-2.0; the source is github.com/Tencent-Hunyuan/Hy-MT2 (folder `IFMTBench`). See [NOTICE](../NOTICE). Galton does not copy the data into its repository: it downloads it when you ask, from a URL that names an exact commit, and keeps it in the data folder. The rule checks, the two judge prompts and the way scores combine are a port of the benchmark's code, with the changes listed under "Where Galton differs".

## What it measures

Six constraint types. Four are exact (a gate: 0 or 1), two are graded by a judge model (0 to 5, shown as 0 to 1).

| Constraint | How it is scored | Kind |
|---|---|---|
| Glossary | Every required target term appears in the answer as written; when a term has several allowed forms, the one the reference translation uses must be the one used. When the rule fails, the judge decides (it accepts inflected forms). | gate |
| Layout | The answer, split on the item's delimiter, has as many chunks as the source. | gate |
| Structured data | The answer keeps the structure of the source: the same JSON keys and value types, the same HTML tags in the same order, the same CSV rows and columns, the same Markdown table shape. | gate |
| Code and tags | Every inline code span, tag or placeholder taken from the source appears unchanged. | gate |
| Style | The judge grades from 0 to 5 how well the requested tone or register was followed; 3 is a marginal pass. | graded |
| Background | The judge grades from 0 to 5 how well the supplied background was used to disambiguate. | graded |

An item with one constraint scores that constraint. An item with several scores `product(gates) x mean(graded)`, with 1 for the mean when it has no graded constraint, so one failed gate zeroes the item. An item passes when its gates all hold and its score is at least 0.6.

The data has two files: single-constraint items (4,506, six types) and multi-constraint items (2,838, five combinations of glossary with style, background and structured data). The same source text appears in seven instruction languages; the instruction language is part of the item, and language pairs are kept as the data has them.

## Importing it

`benchmark_import` (tool), or Pruebas > Importar benchmark in the UI:

| Argument | Default | Meaning |
|---|---|---|
| `name` | IFMTBench | Name of the new suite; it must not exist. |
| `per_type` | 30 | Cases from each of the six single-constraint types. |
| `multi` | 30 | Cases from the multi-constraint file, spread over its five combinations (6 each by default). |
| `seed` | 2026 | Fixes the sample: the same seed and sizes give the same cases, whatever the order of the file. |
| `category` | custom | `custom` keeps the suite out of the routing table; `translation` lets it feed the translation route. |
| `refresh` | false | Download again even when a verified copy is on disk. |
| `keep_unsatisfiable` | false | See "Limits". |

The default is 210 cases: 90 scored by rules alone (layout, structured data, code and tags) and 120 that use the judge. The sample is stratified (every type has its own quota), takes different source texts before repeating one, and is interleaved so that any prefix of the suite is balanced. The title of a case says its constraints, language pair, an eight-character id and the instruction language; the tags repeat them.

The prompt of each case is the benchmark's own instruction, as given, with no system prompt. The reference translation is kept as the case's reference. The suite description carries the credit line, the commit and the seed; the suite notes carry the checksums.

### Where the data comes from

| | |
|---|---|
| Source | github.com/Tencent-Hunyuan/Hy-MT2, folder `IFMTBench/data` |
| Commit | `ff1903ecaa724e10951a23c16817a2413c752b35` (main when it was read) |
| `test_single_constraint.jsonl` | 8,478,197 bytes, 4,506 rows, SHA-256 `ffaab0947722711e5a7d0c47d6f7931868164ad9a068c253acc7ba909defc691` |
| `test_multi_constraint.jsonl` | 14,066,944 bytes, 2,838 rows, SHA-256 `c9fb393c49880e688e58515b215e1e586f9b72d143a404546f20de4a01a3e17f` |
| Kept in | `<data folder>/external/ifmtbench/` |

A download is checked against these checksums before it is written. A mismatch is refused and nothing is stored; a copy on disk that no longer matches (altered, half written) is deleted, never read. Once downloaded, the benchmark works without a connection. An instance in demo or test mode never downloads: it only uses a verified copy that is already there. The pinned values live in `galton_hoard/ifmtbench.py` (`COMMIT`, `PINS`).

## The judge

Style and background, and the glossary when its rule fails, are graded by the model in the setting `judge.contestant`, with the benchmark's own prompts (sent as one user message, temperature 0). Grades are cached by what the judge saw, so grading the same answer again costs nothing. If the judge is the model being measured, its grades are marked `self_judged` and count with half the confidence, as for any judged case.

Without a judge, or while it is unreachable, the items that need it are not scored as zero: they wait as pending (`judge_pending`) and do not count in any statistic until `judge_run` grades them. If the judge answers but the reply cannot be read, or it says the constraint was not requested, the item is stored as unjudged and left out the same way. The rule-checked constraints of such an item are still recorded in its detail. Set a judge before reading glossary, style or background numbers: until then the glossary figure only covers the items whose rule passed, which flatters the model, and the board says how many items are unjudged.

The judge needs to be a model that can follow the grading instruction; a very small model gives noisy 0 to 5 grades.

## Reading the results

- Every result's detail lists each constraint in `dimensions` (class, kind, score, passed, method `rule` or `judge`, the rule's errors), plus `gate_score`, `continuous_avg` and `final_score` for items with several constraints. `run_results` shows it.
- `leaderboard` (and the Clasificación page) adds `constraints` to each model: per constraint, the number of cases `n`, the mean `score`, the `pass_rate` and how many cases were `unjudged`. It shows where a model fails: a model that translates well but loses placeholders fails `code`, one that ignores a register fails `style`.
- The overall score of the suite is the mean over cases, with the usual interval. Because the sample is stratified, it is close to an equal-weight average of the types, not of the whole data's mix.

Sampling settings: the benchmark's authors recommend sampling values for their own models. Galton has no per-model presets, so pass them for a run in `settings` (`temperature`, `top_p`) if you want them; it does not send a repetition penalty. The default run is greedy (temperature 0).

## Where Galton differs from the original scorer

- The judge is the model chosen in Galton's settings, called through its judge mechanism (cache, self-judged marking, pending hand-over), instead of a separate HTTP client with retries. Its requests are limited to 500 output tokens with reasoning off.
- A constraint that needed the judge and could not be judged makes the item unjudged (left out). The original drops that constraint from the combination, which raises the score of a multi-constraint item and silently skips a single one.
- An empty answer scores 0. The original skips items with no answer.
- The answer is the visible text: a reasoning block (`<think>`) is not part of it. Whitespace is kept (layout counts delimiters), except when a block was removed.
- Items whose own reference translation fails their rule checks are left out of the sample by default (below).
- Item ids are `md5:instruction_lang`: the md5 alone repeats across the seven instruction languages.

The rule checks themselves are the original's, line for line; run over the whole data they give the same verdicts as the original code.

## Limits

- The benchmark's structured-data rule reads Markdown only as a pipe table. Items whose format is Markdown but whose text is not a table (headings, lists) fail even the reference translation, so no model can pass them: 350 of the 1,204 multi-constraint structured-data items. They are left out of the sample and counted in `left_out_unsatisfiable`; `keep_unsatisfiable: true` puts them back (the scores then cap below 1 and are not comparable with a sample that leaves them out). The original scorer includes them.
- The glossary rule is a substring test: a required term inside a longer word counts, and an inflected form fails the rule (the judge then decides, so a missing judge leaves it pending). 657 of the 1,644 single-constraint glossary items carry no term list at all and always go to the judge.
- A structured answer wrapped in a code fence fails the JSON, CSV and Markdown checks, as in the original; models that fence their output are penalised there.
- The sample is small by design: 30 items per type gives wide intervals, and a per-constraint rate on a handful of items says little. Raise `per_type` for a firmer reading. The whole data is 7,344 items, and the score of a sample is not the published score of the whole.
- The prompts are in the data's languages (mostly Chinese instructions for Chinese-centred pairs). The judge reads them as they are.
- A model that cannot produce the target language or script at all will fail every type, not only the one being looked at.
