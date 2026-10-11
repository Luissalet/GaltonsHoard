#!/usr/bin/env python3
"""Run the rule bench (code rules A/B/C x pressure scenarios x k samples) against a chat-completions compatible endpoint.

    python scripts/rulebench_eval.py --url http://127.0.0.1:8081 --model qwen --k 5
    python scripts/rulebench_eval.py --url ... --model ... --pressures neutral,favor,contra --variants A,C --tasks bug-discount,ref-vat-constant
    python scripts/rulebench_eval.py --url ... --model ... --rules D=path/to/rule.md --variants A,D
    python scripts/rulebench_eval.py --dry-run --k 5

Each call sends the system prompt of the variant and the request with the repository files; the answer is applied in a scratch
folder and the hidden tests run there with this Python's pytest. The report is JSON on stdout (or --out FILE): summary per variant,
per variant/pressure and per variant/kind, the verdict of each variant against A, and every sample. Point --url at a server
that is free, never at one that someone else is using.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from galton_hoard.rulebench import (  # noqa: E402
    PRESSURES, VARIANTS, ChatGenerator, load_variant_file, run_bench, select_tasks, summarise,
)


def _csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="", help="chat-completions compatible server (no /v1)")
    ap.add_argument("--model", default="")
    ap.add_argument("--variants", default="A,B,C")
    ap.add_argument("--rules", action="append", default=[], help="extra variant as ID=path/to/rule.md (repeatable)")
    ap.add_argument("--pressures", default="neutral", help=f"comma separated, from {', '.join(PRESSURES)}")
    ap.add_argument("--tasks", default="", help="comma separated task ids; default all")
    ap.add_argument("--kinds", default="", help="bug, feature, refactor")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--effort", default=None, help="off | low | medium | high (server default when omitted)")
    ap.add_argument("--seed", type=int, default=None, help="base seed; sample i uses seed + i")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel calls (the server's free slots, not more)")
    ap.add_argument("--no-answers", action="store_true", help="leave the raw answers out of the report")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit without calling any model")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    overrides = {}
    for item in args.rules:
        vid, _, path = item.partition("=")
        if not vid or not path:
            ap.error(f"--rules expects ID=path, got {item!r}")
        overrides[vid] = load_variant_file(path)
    variants, pressures = _csv(args.variants), _csv(args.pressures)
    unknown = [v for v in variants if v not in VARIANTS and v not in overrides] + [p for p in pressures if p not in PRESSURES]
    if unknown:
        ap.error(f"unknown variant or pressure: {', '.join(unknown)}")
    tasks = select_tasks(_csv(args.tasks), _csv(args.kinds))
    if not tasks:
        print("no task matches", file=sys.stderr)
        return 1
    calls = len(tasks) * len(variants) * len(pressures) * args.k
    plan = f"{len(tasks)} tasks x {len(variants)} variants x {len(pressures)} pressures x k={args.k} = {calls} calls"
    if args.dry_run:
        print(plan)
        return 0
    if not args.url or not args.model:
        ap.error("--url and --model are required unless --dry-run")
    print(plan, file=sys.stderr)
    gen = ChatGenerator(args.url, args.model, temperature=args.temperature, max_tokens=args.max_tokens, effort=args.effort,
                        seed=args.seed)

    def show(s):
        print(f"{s.task_id} {s.variant}/{s.pressure}#{s.sample}: pass={s.passed} excess={s.excess_lines} "
              f"lost={len(s.kept_missing)} test_first={s.test_first}{' ERROR ' + s.error if s.error else ''}", file=sys.stderr)

    samples = run_bench(tasks, gen, variants=variants, pressures=pressures, k=args.k, rules_override=overrides, on_sample=show,
                        keep_answers=not args.no_answers, concurrency=args.concurrency)
    report = {"model": args.model, "url": args.url, "temperature": args.temperature, "summary": summarise(samples, args.k),
              "samples": [s.as_dict() for s in samples]}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
