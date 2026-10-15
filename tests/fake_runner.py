"""Stand-in for the container trial runner in the agent-bench tests: writes the folders and ``result.json`` files the real one writes.

The outcome follows the task name: ``*pass*`` passes, ``*fail*`` fails its tests, ``*flaky*`` cannot install the harness on a first job and passes on a
retry, ``*slow*`` runs out of the agent's time. A harness whose agent is ``bad`` fails everything it gets to verify.
"""

import json
import re
import sys
import uuid
from pathlib import Path


def main(argv):
    if argv[:1] == ["datasets"]:
        return 0
    args = {"-i": []}
    i = 1
    while i < len(argv):
        key = argv[i]
        if key in ("-y", "-q"):
            i += 1
            continue
        value = argv[i + 1] if i + 1 < len(argv) else ""
        if key == "-i":
            args["-i"].append(value)
        else:
            args[key] = value
        i += 2
    job = Path(args["-o"]) / args["--job-name"]
    job.mkdir(parents=True, exist_ok=True)
    retry = re.search(r"-r\d+$", args["--job-name"]) is not None
    k = int(args.get("-k", "1"))
    span = {"started_at": "2026-10-15T00:00:00Z", "finished_at": "2026-10-15T00:01:40Z"}
    for task in args["-i"]:
        for _ in range(k):
            trial = job / f"{task}__{uuid.uuid4().hex[:7]}"
            trial.mkdir()
            (trial / "agent").mkdir()
            (trial / "agent" / "agent.txt").write_text(f"worked on {task}\n", encoding="utf-8")
            result = {"trial_name": trial.name, "task_name": f"bench/{task}", "agent_info": {"version": "1.0"},
                      "agent_result": {"n_input_tokens": 1000, "n_output_tokens": 100, "n_cache_tokens": 0},
                      "started_at": span["started_at"], "finished_at": span["finished_at"], "agent_setup": span, "exception_info": None}
            if "flaky" in task and not retry:
                result["exception_info"] = {"exception_type": "AgentSetupTimeoutError", "exception_message": "setup timed out"}
            elif "slow" in task:
                result["agent_execution"] = span
                result["exception_info"] = {"exception_type": "AgentTimeoutError", "exception_message": "agent timed out"}
            else:
                result["agent_execution"] = span
                ok = "fail" not in task and args["-a"] != "bad"
                result["verifier_result"] = {"rewards": {"reward": 1.0 if ok else 0.0}}
            (trial / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (job / "result.json").write_text(json.dumps({"finished_at": "2026-10-15T00:02:00Z"}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
