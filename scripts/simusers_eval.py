#!/usr/bin/env python3
"""Run the simulated-user family against Faustus (default) or any chat-completions compatible chat endpoint.

    python scripts/simusers_eval.py --simulator-url http://127.0.0.1:8081 --simulator-model qwen --cases vague-savings,hurry-speed
    python scripts/simusers_eval.py --target chat --target-url http://127.0.0.1:8081 --target-model m --simulator-url ... --simulator-model ...

Faustus sign-in: FAUSTUS_URL (default http://127.0.0.1:7000), FAUSTUS_USER / FAUSTUS_PASS or FAUSTUS_API_TOKEN; nothing is needed when
Faustus runs with authentication disabled. The report is JSON on stdout (or --out FILE). It uses the simulator and the target you name:
point them at a server that is free, never at one that someone else is using.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from galton_hoard.simusers import FaustusTarget, OpenAISimulator, OpenAITarget, load_cases, run_family, summarise  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=("faustus", "chat"), default="faustus")
    ap.add_argument("--target-url", default=os.environ.get("FAUSTUS_URL", "http://127.0.0.1:7000"))
    ap.add_argument("--target-model", default=os.environ.get("FAUSTUS_MODEL", ""))
    ap.add_argument("--endpoint-url", default=os.environ.get("FAUSTUS_ENDPOINT_URL", ""), help="Faustus: model endpoint of the new chats")
    ap.add_argument("--simulator-url", required=True, help="chat-completions compatible server that plays the user (no /v1)")
    ap.add_argument("--simulator-model", required=True)
    ap.add_argument("--cases", default="", help="comma separated case ids; default all")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    cases = load_cases(only=[c for c in args.cases.split(",") if c])
    if not cases:
        print("no case matches", file=sys.stderr)
        return 1
    if args.target == "faustus":
        def factory():
            return FaustusTarget(args.target_url, model=args.target_model, endpoint_url=args.endpoint_url,
                                 username=os.environ.get("FAUSTUS_USER", ""), password=os.environ.get("FAUSTUS_PASS", ""),
                                 token=os.environ.get("FAUSTUS_API_TOKEN", ""))
    else:
        def factory():
            return OpenAITarget(args.target_url, args.target_model)
    sim = OpenAISimulator(args.simulator_url, args.simulator_model)
    results = run_family(cases, factory, sim, on_result=lambda r: print(f"{r.case_id}: goal={r.goal_reached} turns={r.turns} asked={r.asked_clarifying} score={r.score} ({r.ended})", file=sys.stderr))
    report = {"summary": summarise(results), "results": [r.as_dict() for r in results]}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
