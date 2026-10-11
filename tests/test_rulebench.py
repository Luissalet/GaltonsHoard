"""Rule bench: task integrity, deterministic scoring, pass@k / pass^k and the A/B/C verdict with a fake model."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from galton_hoard.rulebench import (
    TASKS, apply_patch, changed_lines, load_variant_file, parse_answer, pass_at_k, pass_hat_k, run_bench, run_tests,
    score_answer, select_tasks, summarise, system_prompt, user_prompt,
)

ROOT = Path(__file__).resolve().parent.parent


def _answer(files: dict[str, str], *, test_first: str = "") -> str:
    parts = [f"```file:tests/test_mine.py\n{test_first}```"] if test_first else []
    parts += [f"```file:{p}\n{c}```" for p, c in files.items()]
    return "Assumption: the simplest reading.\n\n" + "\n\n".join(parts)


def _reference_answer(task: dict) -> str:
    ref = apply_patch(task["files"], task["patch"])
    return _answer({p: c for p, c in ref.items() if c != task["files"].get(p)})


def _sloppy_answer(task: dict) -> str:
    """The reference change plus a 'tidy up': every comment line and the protected snippets' lines are dropped."""
    ref = apply_patch(task["files"], task["patch"])
    out = {}
    for path, content in ref.items():
        lines, skip_indent = [], None
        for ln in content.splitlines(keepends=True):
            indent = len(ln) - len(ln.lstrip())
            if skip_indent is not None and (not ln.strip() or indent > skip_indent):
                continue
            skip_indent = None
            if ln.lstrip().startswith("#") or any(s in ln for s in task.get("keep", [])):
                if ln.lstrip().startswith("def "):
                    skip_indent = indent
                continue
            lines.append(ln)
        out[path] = "".join(lines)
    return _answer(out)


def test_twenty_tasks_with_unique_ids_and_all_kinds():
    ids = [t["id"] for t in TASKS]
    assert len(ids) >= 20 and len(set(ids)) == len(ids)
    assert {t["kind"] for t in TASKS} == {"bug", "feature", "refactor"}
    for t in TASKS:
        assert t["keep"], t["id"]
        for snippet in t["keep"]:
            assert any(snippet in c for c in t["files"].values()), (t["id"], snippet)


@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_original_fails_and_reference_passes(task):
    assert run_tests(task["files"], task["tests"])[0] is False
    ok, out = run_tests(apply_patch(task["files"], task["patch"]), task["tests"])
    assert ok, out
    ref = apply_patch(task["files"], task["patch"])
    assert all(s in "\n".join(ref.values()) for s in task["keep"]), "the reference must keep every protected snippet"


def test_pass_at_k_and_pass_hat_k():
    assert pass_at_k(5, 0, 5) == 0.0 and pass_at_k(5, 1, 5) == 1.0
    assert pass_at_k(5, 1, 1) == pytest.approx(0.2)
    assert pass_at_k(10, 3, 2) == pytest.approx(1 - math.comb(7, 2) / math.comb(10, 2))
    assert pass_hat_k(5, 5, 5) == 1.0 and pass_hat_k(5, 4, 5) == 0.0
    assert pass_hat_k(10, 8, 2) == pytest.approx(math.comb(8, 2) / math.comb(10, 2))
    assert math.isnan(pass_at_k(3, 1, 4))


def test_parse_answer_order_and_paths():
    text = "x\n```file:./tests/test_a.py\nA\n```\n```file:src\\b.py\nB\n```\n```file:../evil.py\nC\n```\n```python\nD\n```"
    assert parse_answer(text) == [("tests/test_a.py", "A\n"), ("src/b.py", "B\n")]


def test_changed_lines_counts_a_replacement_once():
    assert changed_lines("a\nb\nc\n", "a\nB\nc\n") == 1
    assert changed_lines("a\n", "a\nb\nc\n") == 2
    assert changed_lines("a\nb\n", "a\n") == 1


def test_reference_answer_scores_clean():
    task = next(t for t in TASKS if t["id"] == "ref-rename-total")
    s = score_answer(task, _reference_answer(task))
    assert s.passed and s.excess_lines == 0 and not s.outside_files and not s.kept_missing
    assert s.lines_changed == s.reference_lines == 3
    assert not s.wrote_test and not s.test_first


def test_tidy_up_is_measured_as_excess_and_lost_snippets():
    task = next(t for t in TASKS if t["id"] == "ref-rename-total")
    s = score_answer(task, _sloppy_answer(task))
    assert s.passed
    assert s.excess_lines > 0
    assert "def _legacy_total" in s.kept_missing and "NOTE(marta)" in s.kept_missing


def test_test_first_outside_files_and_hidden_tests_cannot_be_replaced():
    task = next(t for t in TASKS if t["id"] == "bug-discount")
    ref = apply_patch(task["files"], task["patch"])
    test = "from pricing import apply_discount\n\n\ndef test_it():\n    assert apply_discount(100, 20) == 80.0\n"
    answer = _answer({"pricing.py": ref["pricing.py"], "other.py": "x = 1\n"}, test_first=test)
    s = score_answer(task, answer)
    assert s.passed and s.wrote_test and s.test_first
    assert s.outside_files == ["other.py"] and s.excess_lines == 1
    cheat = _answer({"tests/test_hidden.py": "def test_ok():\n    pass\n"})
    s2 = score_answer(task, cheat)
    assert not s2.passed and s2.no_files


def test_errors_and_empty_answers_are_failed_samples():
    task = TASKS[0]
    assert score_answer(task, "", error="RuntimeError: down").error
    s = score_answer(task, "I would change the multiplication.")
    assert not s.passed and s.no_files and "no file blocks" in s.test_output


def test_prompts_carry_rules_pressure_and_files():
    assert "Rules for code changes" not in system_prompt("A")
    assert "Surgical changes" in system_prompt("B") and "smallest code" in system_prompt("C")
    assert system_prompt("A", "- custom rule") .endswith("- custom rule")
    task = TASKS[0]
    u = user_prompt(task["request"], task["files"], "contra")
    assert "tidy up" in u and "```file:pricing.py" in u
    assert "tidy up" not in user_prompt(task["request"], task["files"], "neutral")


def test_load_variant_file_drops_frontmatter(tmp_path):
    f = tmp_path / "r.md"
    f.write_text("---\nid: x\n---\n\n- rule one\n", encoding="utf-8")
    assert load_variant_file(f) == "- rule one"


def test_run_bench_and_verdict_with_a_fake_model():
    tasks = select_tasks(only=["ref-rename-total", "bug-weekend"])
    by_request = {t["request"]: t for t in tasks}

    def fake(system: str, user: str, sample: int) -> str:
        task = next(t for r, t in by_request.items() if user.startswith(r))
        if "Rules for code changes" in system:
            return _reference_answer(task)
        if sample == 0:
            raise RuntimeError("server busy")
        return _sloppy_answer(task)

    seen = []
    samples = run_bench(tasks, fake, variants=("A", "C"), pressures=("neutral", "contra"), k=3, on_sample=seen.append)
    assert len(samples) == len(seen) == 2 * 2 * 2 * 3
    report = summarise(samples, k=3)
    a, c = report["by_variant"]["A"], report["by_variant"]["C"]
    assert a["errors"] == 4 and c["errors"] == 0
    assert c["pass@1"] == 1.0 and c["pass^3"] == 1.0 and c["mean_excess_lines"] == 0
    assert a["pass@3"] == 1.0 and a["pass^3"] == 0.0 and a["mean_excess_lines"] > 0
    assert report["verdict"]["C"] == {"excess_cut_vs_A": 1.0, "pass_at_k_held": True, "meets_target": True}
    assert set(report["by_variant_pressure"]) == {"A/contra", "A/neutral", "C/contra", "C/neutral"}
    assert report["verdict_by_pressure"]["contra"]["C"]["meets_target"] and set(report["verdict_by_pressure"]) == {"neutral", "contra"}
    from galton_hoard.rulebench import samples_from_report
    again = summarise(samples_from_report({"samples": [s.as_dict() for s in samples]}), k=3)
    assert again["by_variant"] == report["by_variant"]
    with pytest.raises(ValueError):
        run_bench(tasks, fake, variants=("Z",), k=1)


def test_cli_dry_run_lists_the_plan():
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "rulebench_eval.py"), "--dry-run", "--k", "5",
                           "--pressures", "neutral,contra", "--tasks", "bug-discount,feat-clamp"],
                          capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "2 tasks x 3 variants x 2 pressures x k=5 = 60 calls" in done.stdout


def test_parallel_run_keeps_the_sequential_order():
    tasks = select_tasks(kinds=["feature"])[:2]
    by_request = {t["request"]: t for t in tasks}

    def fake(system: str, user: str, sample: int) -> str:
        return _reference_answer(next(t for r, t in by_request.items() if user.startswith(r)))

    seq = run_bench(tasks, fake, variants=("A", "B"), k=2)
    par = run_bench(tasks, fake, variants=("A", "B"), k=2, concurrency=4)
    key = lambda s: (s.task_id, s.variant, s.pressure, s.sample, s.passed)  # noqa: E731
    assert [key(s) for s in par] == [key(s) for s in seq] and all(s.passed for s in par)
