#!/usr/bin/env python3
"""Run the harness bench (model x harness x task) against chat-completions compatible endpoints, without the Galton server.

    python scripts/harnessbench_eval.py --dry-run
    python scripts/harnessbench_eval.py --url http://127.0.0.1:8081 --model qwen3-8b --harnesses native_tools,json_text --kinds calc,notool
    python scripts/harnessbench_eval.py --model qwen3-8b@http://127.0.0.1:8081 --model llama3.1-8b@http://127.0.0.1:8082 --k 3 --out matrix.json
    python scripts/harnessbench_eval.py --url ... --model m --harnesses native_tools,xml_text.long.trim --tasks calc-sum,fs-write-greeting
    python scripts/harnessbench_eval.py --url ... --model m --cli "agent = terminal-agent --base-url {base_url} --model {model} --prompt-file {prompt_file} --workdir {workdir}"

Every cell gets the same output-token cap, turn cap and temperature. The report is JSON on stdout (or --out FILE): the matrix, the automatic conclusion
(is the difference between harnesses larger than between models) and every trial with its trace; the markdown table goes to stderr. A trial whose
server failed (timeout, connection) has no result: it is counted apart and left out of the percentages. Point --url at a server that is free, never
at one that someone else is using.

Harness ids: native_tools, json_text, xml_text, react_text, each with the optional suffixes .long (long system prompt) and .trim (old results are
shortened), for example json_text.long.trim. --cli adds a command-line agent; it is started without a shell and receives the task and the endpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from galton_hoard.backends import ChatRequest, make_backend  # noqa: E402
from galton_hoard.errors import GaltonError  # noqa: E402
from galton_hoard.harnessbench import (  # noqa: E402
    DEFAULT_HARNESSES, KINDS, build_matrix, conclude, parse_external, parse_harness_id, run_external, run_trial, select_tasks,
)
from galton_hoard.harnessbench.external import ExternalSpec  # noqa: E402
from galton_hoard.harnessbench.runner import markdown_table  # noqa: E402


def _csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _models(args, ap) -> list[tuple[str, str]]:
    out = []
    for item in args.model:
        name, sep, url = item.partition("@")
        url = url if sep else args.url
        if not name or not url:
            ap.error(f"{item!r} needs a server: give --url or write name@http://host:port")
        out.append((name, url))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="", help="chat-completions compatible server (no /v1)")
    ap.add_argument("--model", action="append", default=[], help="model name, or name@url when it lives on another server (repeatable)")
    ap.add_argument("--api", default="openai", choices=("openai", "ollama"))
    ap.add_argument("--harnesses", default=",".join(DEFAULT_HARNESSES))
    ap.add_argument("--cli", action="append", default=[], help="command-line harness as 'name = command {base_url} {model} {prompt_file} {workdir}' (repeatable)")
    ap.add_argument("--tasks", default="", help="comma separated task ids; default all")
    ap.add_argument("--kinds", default="", help=f"comma separated, from {', '.join(KINDS)}")
    ap.add_argument("--k", type=int, default=1, help="samples per task (pass@k and pass^k when above 1)")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-format-errors", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=None, help="default 0 for k=1, 0.7 otherwise")
    ap.add_argument("--seed", type=int, default=None, help="base seed; sample i uses seed + i")
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds per model call")
    ap.add_argument("--cli-timeout", type=float, default=600.0, help="seconds per command-line trial")
    ap.add_argument("--no-traces", action="store_true", help="leave the traces out of the report")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit without calling any model")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    external = {s.name: s for s in parse_external(args.cli)}
    refs = _csv(args.harnesses) + [f"cli:{n}" for n in external]
    harnesses = []
    for ref in refs:
        if ref.startswith("cli:"):
            if ref[4:] not in external:
                ap.error(f"unknown command-line harness {ref!r}")
            harnesses.append(external[ref[4:]])
            continue
        try:
            harnesses.append(parse_harness_id(ref))
        except GaltonError as exc:
            ap.error(str(exc.message))
    tasks = select_tasks(_csv(args.tasks), _csv(args.kinds))
    if not tasks:
        print("no task matches", file=sys.stderr)
        return 1
    models = _models(args, ap)
    k = max(1, args.k)
    temperature = args.temperature if args.temperature is not None else (0.0 if k <= 1 else 0.7)
    per_model = sum(1 for h in harnesses for t in tasks if t["external"] or not isinstance(h, ExternalSpec)) * k
    plan = f"{len(models) or 1} models x {len(harnesses)} harnesses x {len(tasks)} tasks x k={k} = about {per_model * (len(models) or 1)} trials"
    if args.dry_run:
        print(plan)
        print("harnesses: " + ", ".join(h.id for h in harnesses))
        print("tasks: " + ", ".join(t["id"] for t in tasks))
        return 0
    if not models:
        ap.error("--model is required unless --dry-run")
    print(plan, file=sys.stderr)

    rows: list[dict] = []
    ids = []
    for name, url in models:
        mid = f"{name}@{url}" if len(models) > 1 and "@" not in name else name
        ids.append(mid)
        backend = make_backend(args.api, url, name)
        fails_in_a_row = 0
        for h in harnesses:
            for task in tasks:
                if isinstance(h, ExternalSpec) and not task["external"]:
                    continue
                for sample in range(k):
                    if isinstance(h, ExternalSpec):
                        trial = run_external(h, task, base_url=url, model=name, timeout_s=args.cli_timeout, sample=sample)
                    else:
                        seed = None if args.seed is None else args.seed + sample

                        def chat(messages, tools):
                            return backend.chat(ChatRequest(messages=messages, tools=tools, temperature=temperature, max_tokens=args.max_tokens, seed=seed, timeout_s=args.timeout,
                                                            tools_fallback=False))
                        trial = run_trial(task, h, chat, api=args.api, max_turns=args.max_turns, max_format_errors=args.max_format_errors, sample=sample)
                    d = trial.as_dict() | {"contestant_id": mid}
                    if args.no_traces:
                        d.pop("trace", None)
                    rows.append(d)
                    fails_in_a_row = fails_in_a_row + 1 if trial.outcome == "none" else 0
                    print(f"{mid} {h.id} {task['id']}#{sample}: {trial.outcome}{' (' + trial.reason + ')' if trial.reason else ''}", file=sys.stderr)
                    if fails_in_a_row >= 4:
                        print(f"{mid}: four trials in a row without a result, the server looks down; stopping this model", file=sys.stderr)
                        break
                if fails_in_a_row >= 4:
                    break
            if fails_in_a_row >= 4:
                break
    harness_ids = [h.id for h in harnesses]
    matrix = build_matrix(rows, ids, harness_ids, k)
    conclusion = conclude(matrix, {i: i for i in ids}, harness_ids)
    print(markdown_table([{"id": i, "name": i} for i in ids], harness_ids, matrix), file=sys.stderr)
    print(conclusion["summary"]["en"], file=sys.stderr)
    report = {"models": ids, "harnesses": harness_ids, "k": k, "settings": {"max_turns": args.max_turns, "max_format_errors": args.max_format_errors, "max_tokens": args.max_tokens,
                                                                          "temperature": temperature}, **matrix, "conclusion": conclusion, "trials": rows}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
