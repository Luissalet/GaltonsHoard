#!/usr/bin/env python
"""Run an agent-bench campaign without the app: real harnesses on published benchmarks, one model.

    python scripts/agentbench_run.py --plan --bench tb2,swev --datasets D:/bench/datasets
    python scripts/agentbench_run.py --name c1 --model my-model --base-url http://host:8002/v1 \
        --harnesses harnesses.txt --bench tb2,swev,polyglot --k 1 --slots 8 --per-cell 2 \
        --jobs D:/bench/jobs --datasets D:/bench/datasets --report c1.json

``--harnesses`` is a text file with one harness line per row, in the format of the setting ``agentbench.harnesses`` (see
galton_hoard/agentbench/spec.py). The job folders under ``--jobs`` are the state: running the same command again only launches what is missing.
The report has every cell, the ranking per benchmark and, with ``--focus``, the paired comparison of that harness against each other one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from galton_hoard.agentbench.campaign import Campaign, CampaignConfig, dataset_folder, download, save_json  # noqa: E402
from galton_hoard.agentbench.harbor import runner_exe  # noqa: E402
from galton_hoard.agentbench.scoring import board  # noqa: E402
from galton_hoard.agentbench.spec import CATALOGUE, benchmark, parse_harnesses, read_dataset, select  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--api-key", default="sk-local")
    ap.add_argument("--harnesses", default="", help="file with one harness line per row")
    ap.add_argument("--only", default="", help="comma list of harness names to keep from the file")
    ap.add_argument("--bench", default=",".join(b.id for b in CATALOGUE))
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--time-scale", type=float, default=1.0, help="multiply every task's agent time limit (same for all harnesses)")
    ap.add_argument("--jobs", default="jobs")
    ap.add_argument("--datasets", default="datasets")
    ap.add_argument("--exe", default="")
    ap.add_argument("--focus", default="faustus")
    ap.add_argument("--report", default="")
    ap.add_argument("--plan", action="store_true", help="print the subsets and exit")
    ap.add_argument("--report-only", action="store_true", help="read the job folders and write the report, launch nothing")
    args = ap.parse_args(argv)

    exe = runner_exe(args.exe)
    benches = [benchmark(b) for b in args.bench.split(",") if b.strip()]
    datasets = Path(args.datasets)
    for b in benches:
        folder = dataset_folder(datasets, b)
        if not folder.is_dir():
            print(f"downloading {b.dataset} ...", file=sys.stderr)
            download(exe, b, datasets)
    if args.plan:
        for b in benches:
            sub = select(b, read_dataset(dataset_folder(datasets, b)))
            print(json.dumps({"benchmark": b.id, "pool": sub["pool"], "dev": sub["dev"], "test": sub["test"]}, indent=1))
        return 0
    harnesses = parse_harnesses(Path(args.harnesses).read_text(encoding="utf-8").splitlines()) if args.harnesses else []
    harnesses = [h for h in harnesses if h.enabled]
    if args.only:
        keep = {x.strip() for x in args.only.split(",") if x.strip()}
        harnesses = [h for h in harnesses if h.name in keep]
    if not harnesses:
        ap.error("no harness to run (see --harnesses / --only)")
    cfg = CampaignConfig(name=args.name or time.strftime("ab%Y%m%d%H%M"), model=args.model, base_url=args.base_url, api_key=args.api_key, k=args.k,
                         slots=args.slots, per_cell=args.per_cell, retries=args.retries, splits=tuple(s for s in args.splits.split(",") if s),
                         exe=exe, jobs_dir=Path(args.jobs), datasets_dir=datasets, time_scale=args.time_scale)
    camp = Campaign(cfg, benches, harnesses, log=lambda m: print(time.strftime("%H:%M:%S"), m, file=sys.stderr, flush=True))
    camp.plan()
    report_path = Path(args.report) if args.report else None

    def write_report() -> dict:
        rows = camp.all_trials()
        data = {"campaign": cfg.name, "model": cfg.model, "harnesses": [h.info() for h in harnesses],
                "benchmarks": [b.info() for b in benches], "all": board(rows, focus=args.focus), "test": board(rows, focus=args.focus, split="test"),
                "dev": board(rows, focus=args.focus, split="dev"), "trials": rows,
                "skipped": [{"cell": c.key, "why": c.skipped} for c in camp.cells if c.skipped]}
        if report_path:
            save_json(report_path, data)
        return data

    if not args.report_only:
        camp.on_trials = lambda cell, rows: write_report()
        camp.run()
    data = write_report()
    for b in data["all"]["benchmarks"]:
        print(f"\n{b}")
        for r in data["all"]["ranking"][b]:
            rate = "-" if r["rate"] is None else f"{r['rate'] * 100:5.1f}%"
            print(f"  {r['rank'] or '-':>2} {r['harness']:<14} {rate}  n={r['n']:<3} none={r['none_n']:<2} ci={r['ci']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
