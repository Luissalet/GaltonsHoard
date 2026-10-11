# Rule bench

Measures what a set of code rules in the system prompt does to a model's edits of a small repository.

## Run it

```
python scripts/rulebench_eval.py --url http://HOST:PORT --model NAME --k 5 --pressures neutral,favor,contra --concurrency 6 --out report.json
python scripts/rulebench_eval.py --url ... --model ... --rules D=path/to/rule.md --variants A,D
python scripts/rulebench_eval.py --resummarize report.json
```

`--url` is a chat-completions compatible server (without `/v1`). Each call sends the variant's system prompt and the request
with every file of the task; the answer must carry one ```` ```file:<path> ```` block per changed file. The two tools
`rulebench_tasks` and `rulebench_score` do the same one answer at a time from an assistant.

## What is measured

| Measure | Meaning |
| --- | --- |
| `pass@1`, `pass@k` | Hidden tests pass (one sample; at least one of k, unbiased estimator). |
| `pass^k` | All k samples pass: how reliable the model is, not how lucky. |
| `mean_excess_lines` | Lines changed beyond the reference patch, plus every changed line in source files the reference does not touch. Test files the model adds do not count. |
| `outside_file_rate` | Share of answers that change a source file the reference leaves alone. |
| `kept_missing_rate` | Share of answers that remove a protected snippet: someone else's comment or dead code the request did not mention. |
| `test_written_rate`, `test_first_rate` | The answer brought a test file; of those, the test block came before the first source file. |

Verdicts compare each variant with A (no rules), pooled and per pressure; the target is 30 % fewer excess lines without a
lower `pass@k`.

## First measurement · 11-10-2026

GLM-5.3-Flash NVFP4 on the three Sparks (vLLM, temperature 0.7, reasoning off), 20 tasks x A/B/C x neutral/favor/contra
x k=5 = 900 calls, 6 in parallel, 27 minutes. Variants: A none, B the four rules in their original wording, C the
positive short wording (the texts of Faustus's `common/karpathy-code` and `common/karpathy-code-positive`).

| Variant / pressure | pass^5 | excess lines | lost snippets | wrote a test |
| --- | --- | --- | --- | --- |
| A neutral | 1.00 | 0.40 | 0 % | 7 % |
| B neutral | 1.00 | 0.21 | 1 % | 42 % |
| C neutral | 1.00 | 0.21 | 0 % | 45 % |
| A favor | 1.00 | 0.16 | 1 % | 0 % |
| B favor | 1.00 | 0.10 | 0 % | 6 % |
| C favor | 1.00 | 0.17 | 1 % | 7 % |
| A contra | 1.00 | 3.37 | 92 % | 1 % |
| B contra | 1.00 | 3.02 | 86 % | 18 % |
| C contra | 1.00 | 3.20 | 86 % | 9 % |

Reading:

- Without pressure both rule sets cut the excess lines by 47.5 % and keep every test passing: both meet the target there.
  They also make the model add a test six times as often. B also meets it under "favor" (−37.5 %); C does not: it changes 6 % more lines than A there.
- Under "contra" (the user asks to tidy up) no variant holds: the model follows the user and removes comments and dead
  code in 86-92 % of answers. That is the user's request winning over a system rule, not a failure of the rule text.
- Pooled over the three pressures B cuts 15 % and C 9 %, so neither meets the target pooled: the pooled number is
  dominated by "contra".
- Test-first order is 0 % everywhere: in a one-shot answer the model writes the code block first even when it adds a
  test.
- GLM solves every task in every variant (pass^5 = 1.00), so this set cannot show a rule costing correctness with this
  model; harder tasks or a smaller model are needed for that side.

Recommendation for the main model: B (original wording) as the default code rule: same gain as C without pressure, the
only one that also helps when the user asks for a minimal change, and the higher test-writing rate under pressure.
