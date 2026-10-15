"""Agent bench: harness lines, task subsets, reading the runner's folders, scoring, and a whole campaign against a stand-in runner."""

from __future__ import annotations

import collections
import json
import threading
from pathlib import Path

import pytest

from galton_hoard.agentbench import harbor, scoring, spec
from galton_hoard.agentbench.campaign import Campaign, CampaignConfig, dataset_folder

FAKE = str(Path(__file__).with_name("fake_runner.py"))


def _dataset(root: Path, bench: spec.Benchmark, names: list[str]) -> Path:
    folder = dataset_folder(root, bench)
    for name in names:
        (folder / name).mkdir(parents=True, exist_ok=True)
        (folder / name / "instruction.md").write_text("do it", encoding="utf-8")
        (folder / name / "task.toml").write_text('[metadata]\ndifficulty = "medium"\n[agent]\ntimeout_sec = 900.0\n[environment]\ncpus = 1\nmemory_mb = 2048\n',
                                                 encoding="utf-8")
    return folder


# ------------------------------------------------------------------ harness lines
def test_a_harness_line_reads_agent_model_options_kwargs_and_env():
    h = spec.parse_harness_line(r"mine = pkg.mod:Agent model=openai/{model} path=D:\code\x setup=3 mcp=no ak.effort=high env.MY_URL={root_url}")
    assert h and h.name == "mine" and h.agent == "pkg.mod:Agent" and h.path == r"D:\code\x" and h.setup == 3 and not h.mcp
    assert dict(h.kwargs) == {"effort": "high"} and h.model_arg("m1") == "openai/m1"
    assert h.render("{root_url}|{base_url}|{model}", model="m1", base_url="http://h:8002/v1", api_key="k") == "http://h:8002|http://h:8002/v1|m1"
    off = spec.parse_harness_line("# other = some-agent")
    assert off and not off.enabled
    for bad in ("", "no equals sign", "Bad Name = x", "x = agent unknown=1", "x = agent setup=lots"):
        assert spec.parse_harness_line(bad) is None, bad
    assert [h.name for h in spec.parse_harnesses(["a = x", "a = y", "b = z"])] == ["a", "b"]


# ------------------------------------------------------------------ subsets
def test_the_subset_is_deterministic_stratified_and_split_with_the_same_mix():
    bench = spec.benchmark("polyglot")
    tasks = [spec.TaskMeta(f"{lang}_{i}") for lang in ("python", "rust", "go", "java", "javascript", "cpp") for i in range(10)]
    a, b = spec.select(bench, tasks), spec.select(bench, list(reversed(tasks)))
    assert a == b and len(a["dev"]) == bench.dev and len(a["dev"]) + len(a["test"]) == bench.n
    assert not set(a["dev"]) & set(a["test"])
    per_lang = collections.Counter(n.split("_")[0] for n in a["dev"] + a["test"])
    assert set(per_lang.values()) == {5}
    dev_lang = collections.Counter(n.split("_")[0] for n in a["dev"])
    assert max(dev_lang.values()) - min(dev_lang.values()) <= 1 and len(dev_lang) == 6


def test_filters_on_the_task_metadata_are_applied(tmp_path):
    bench = spec.benchmark("tb2")
    folder = dataset_folder(tmp_path, bench)
    for name, diff, secs in (("easy-one", "easy", 900), ("hard-one", "hard", 900), ("long-one", "medium", 7200), ("ok-one", "medium", 1200)):
        (folder / name).mkdir(parents=True)
        (folder / name / "instruction.md").write_text("x", encoding="utf-8")
        (folder / name / "task.toml").write_text(f'[metadata]\ndifficulty = "{diff}"\n[agent]\ntimeout_sec = {secs}\n', encoding="utf-8")
    sub = spec.select(bench, spec.read_dataset(folder))
    assert sorted(sub["dev"] + sub["test"]) == ["easy-one", "ok-one"] and sub["pool"] == 2


# ------------------------------------------------------------------ reading trials
@pytest.mark.parametrize("result,outcome,reason", [
    ({"verifier_result": {"rewards": {"reward": 1.0}}, "agent_execution": {"started_at": "x"}}, "pass", ""),
    ({"verifier_result": {"rewards": {"reward": 0.0}}, "agent_execution": {"started_at": "x"}}, "fail", "tests_failed"),
    ({"exception_info": {"exception_type": "AgentTimeoutError"}, "agent_execution": {"started_at": "x"}}, "fail", "timeout"),
    ({"exception_info": {"exception_type": "NonZeroAgentExitCodeError"}, "agent_execution": {"started_at": "x"}}, "fail", "agent_error"),
    ({"exception_info": {"exception_type": "NonZeroAgentExitCodeError"}}, "none", "setup:NonZeroAgentExitCodeError"),
    ({"exception_info": {"exception_type": "AgentSetupTimeoutError"}}, "none", "setup:AgentSetupTimeoutError"),
    ({"exception_info": {"exception_type": "DockerError"}, "agent_execution": {"started_at": "x"}}, "none", "DockerError"),
])
def test_a_trial_without_a_verdict_is_none_unless_the_harness_itself_failed(result, outcome, reason):
    row = harbor.read_trial({"trial_name": "t", "task_name": "org/task-a", **result})
    assert row["outcome"] == outcome and row["reason"] == reason and row["task"] == "task-a"
    assert row["passed"] is (None if outcome == "none" else outcome == "pass")


def test_job_command_carries_tasks_model_kwargs_env_and_path(tmp_path):
    h = spec.parse_harness_line("x = pkg.mod:Agent model=openai/{model} path=/opt/agent setup=4 ak.mode={model} env.EXTRA_URL={root_url}")
    job = harbor.job_command(exe="runner", bench=spec.benchmark("tb2"), harness=h, tasks=["a", "b"], dataset_dir=tmp_path / "ds", jobs_dir=tmp_path / "jobs",
                             job_name="c1-tb2-x", model="m", base_url="http://h:1/v1", k=2, concurrent=3, base_env={"PYTHONPATH": "/before"})
    argv = job.argv
    assert argv[:2] == ["runner", "run"] and argv[argv.index("-m") + 1] == "openai/m" and argv[argv.index("-k") + 1] == "2"
    assert [argv[i + 1] for i, v in enumerate(argv) if v == "-i"] == ["a", "b"] and "mode=m" in argv
    assert argv[argv.index("--agent-setup-timeout-multiplier") + 1] == "4"
    assert job.env["OPENAI_BASE_URL"] == "http://h:1/v1" and job.env["EXTRA_URL"] == "http://h:1" and job.env["PYTHONPATH"].startswith("/opt/agent")


# ------------------------------------------------------------------ scoring
def _t(bench, harness, task, ok, split="test"):
    return {"benchmark": bench, "harness": harness, "task": task, "split": split, "outcome": "pass" if ok else ("fail" if ok is False else "none"),
            "passed": ok, "reason": "" if ok else "tests_failed"}


def test_the_board_ranks_harnesses_leaves_none_out_and_compares_task_by_task():
    trials = [_t("b", "me", f"t{i}", i < 7) for i in range(10)] + [_t("b", "other", f"t{i}", i < 4) for i in range(10)]
    trials += [_t("b", "other", "t0", None), _t("b", "me", "t0", None, split="dev")]
    out = scoring.board(trials, focus="me")
    rank = out["ranking"]["b"]
    assert [r["harness"] for r in rank] == ["me", "other"] and rank[0]["rate"] == 0.7 and rank[1]["none_n"] == 1
    duel = out["duels"]["b"][0]
    assert duel["vs"] == "other" and duel["wins"] == 3 and duel["losses"] == 0 and duel["shared"] == 10
    assert out["verdict"]["b"]["state"] == "first" and out["verdict"]["b"]["weak_evidence"] is True     # 3-0 is not significant
    assert scoring.board(trials, focus="me", split="dev")["trials"] == 1
    assert scoring.sign_test(10, 0) < 0.01 and scoring.wilson(0, 0) == (None, None)


# ------------------------------------------------------------------ a campaign end to end
def test_a_campaign_runs_every_cell_retries_none_and_never_reruns_a_verdict(tmp_path):
    bench = spec.benchmark("lcb")
    _dataset(tmp_path / "ds", bench, ["a-pass", "b-fail", "c-flaky", "d-slow"])
    good, bad = spec.parse_harness_line("good = good-agent"), spec.parse_harness_line("bad = bad")
    cfg = CampaignConfig(name="c1", model="m", base_url="http://h:1/v1", slots=4, per_cell=1, retries=1, exe=FAKE, jobs_dir=tmp_path / "jobs",
                         datasets_dir=tmp_path / "ds")
    camp = Campaign(cfg, [bench], [good, bad], log=lambda m: None)
    camp.plan()
    summary = camp.run(poll_s=0.05)
    rows = camp.all_trials()
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["harness"], r["task"])].append(r["outcome"])
    assert by[("good", "a-pass")] == ["pass"] and by[("good", "b-fail")] == ["fail"] and by[("good", "d-slow")] == ["fail"]
    assert sorted(by[("good", "c-flaky")]) == ["none", "pass"]            # first job could not install, the retry verified it
    assert by[("bad", "a-pass")] == ["fail"]
    ranking = summary["board"]["ranking"][bench.id]
    assert ranking[0]["harness"] == "good" and ranking[0]["rate"] == 0.5 and ranking[0]["none_n"] == 1
    before = len(rows)
    Campaign(cfg, [bench], [good, bad], log=lambda m: None).run(poll_s=0.05)     # nothing is missing: nothing is launched again
    assert len(camp.all_trials()) == before


def test_a_runner_that_dies_before_starting_is_given_up_after_a_few_launches(tmp_path):
    bench = spec.benchmark("lcb")
    _dataset(tmp_path / "ds", bench, ["a-pass"])
    dead = tmp_path / "dead_runner.py"
    dead.write_text("import sys\nprint(\"Failed to import module 'x'\")\nsys.exit(3)\n", encoding="utf-8")
    logs: list[str] = []
    cfg = CampaignConfig(name="c3", model="m", base_url="http://h:1/v1", slots=1, per_cell=1, retries=1, exe=str(dead),
                         jobs_dir=tmp_path / "jobs", datasets_dir=tmp_path / "ds")
    camp = Campaign(cfg, [bench], [spec.parse_harness_line("broken = x")], log=logs.append)
    camp.plan()
    done = threading.Event()
    threading.Thread(target=lambda: (camp.run(poll_s=0.02), done.set()), daemon=True).start()
    assert done.wait(20), "the campaign kept relaunching a runner that never starts"
    assert sum(1 for m in logs if m.startswith("start ")) == 3
    assert camp.cells[0].skipped.startswith("launch_failed")


def test_a_benchmark_that_needs_mcp_skips_a_harness_without_it(tmp_path):
    bench = spec.benchmark("tau3")
    _dataset(tmp_path / "ds", bench, ["retail-1-pass"])
    cfg = CampaignConfig(name="c2", model="m", base_url="http://h:1/v1", exe=FAKE, jobs_dir=tmp_path / "jobs", datasets_dir=tmp_path / "ds")
    camp = Campaign(cfg, [bench], [spec.parse_harness_line("plain = x mcp=no")], log=lambda m: None)
    assert [c.skipped for c in camp.plan()] == ["needs_mcp"]


# ------------------------------------------------------------------ inside the app
def test_the_app_runs_a_campaign_in_the_background_and_the_tools_read_it(svc, tmp_path):
    from conftest import tool
    bench = spec.benchmark("lcb")
    _dataset(tmp_path / "ds", bench, ["a-pass", "b-fail"])
    svc.settings.set_many({"agentbench.harnesses": ["good = good-agent", "bad = bad", "# off = x"], "agentbench.runner": FAKE,
                           "agentbench.jobs_dir": str(tmp_path / "jobs"), "agentbench.datasets_dir": str(tmp_path / "ds"),
                           "agentbench.model": "m", "agentbench.base_url": "http://h:1/v1"})
    cat = tool(svc, "agentbench_catalog")
    assert cat["runner_found"] and [h["name"] for h in cat["harnesses"]] == ["good", "bad", "off"]
    assert next(b for b in cat["benchmarks"] if b["id"] == "lcb")["downloaded"]
    plan = tool(svc, "agentbench_start", benchmarks=["lcb"], dry_run=True)
    assert sorted(plan["subsets"]["lcb"]["dev"] + plan["subsets"]["lcb"]["test"]) == ["a-pass", "b-fail"]
    import os
    os.environ["GALTON_AB_POLL_S"] = "0.05"
    try:
        card = tool(svc, "agentbench_start", benchmarks=["lcb"], label="t")
        assert card["harnesses"] == ["good", "bad"]
        svc.agentbench.threads[card["id"]][0].join(30)
    finally:
        os.environ.pop("GALTON_AB_POLL_S", None)
    status = tool(svc, "agentbench_status", campaign=card["id"])["campaign"]
    assert status["state"] == "done" and status["trials"]["verified"] == 4 and status["trials"]["passed"] == 1
    result = tool(svc, "agentbench_result")
    assert result["board"]["ranking"]["lcb"][0]["harness"] == "good"
    cell = tool(svc, "agentbench_cell", benchmark="lcb", harness="good")
    trial = tool(svc, "agentbench_trial", trial=cell["trials"][0]["id"])
    assert any("worked on" in text for text in trial["logs"].values())
    assert tool(svc, "agentbench_list")["count"] == 1
    with pytest.raises(Exception):
        tool(svc, "agentbench_start", benchmarks=["nope"])
