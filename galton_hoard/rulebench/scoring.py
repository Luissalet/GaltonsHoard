"""Deterministic scoring of one answer: parse the file blocks, apply them in a scratch copy, run the hidden tests, measure scope.

Measures per sample:

* ``passed``: the hidden tests pass (pytest exit 0);
* ``lines_changed`` and ``reference_lines``: changed lines (difflib, a replaced line counts once) against the reference patch;
* ``excess_lines``: lines changed beyond the reference in the files the reference touches, plus every changed line in other
  source files. Test files the model adds are not counted: writing a test is what the rules ask for;
* ``outside_files``: source files changed that the reference does not touch;
* ``kept_missing``: the ``keep`` snippets (other people's comments, dead code) that disappeared;
* ``wrote_test`` and ``test_first``: whether the answer brought a test file, and whether it came before the first source file.
"""

from __future__ import annotations

import difflib
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .tasks import apply_patch

_BLOCK = re.compile(r"```file:([^\n`]+)\n(.*?)```", re.S)


def parse_answer(text: str) -> list[tuple[str, str]]:
    """``[(path, content), ...]`` in the order they appear; paths normalised to forward slashes, no leading ``./``."""
    out = []
    for m in _BLOCK.finditer(text or ""):
        path = m.group(1).strip().replace("\\", "/")
        while path.startswith("./"):
            path = path[2:]
        if not path or path.startswith("/") or ".." in path.split("/"):
            continue
        out.append((path, m.group(2)))
    return out


def is_test_path(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return path.startswith("tests/") or name.startswith("test_") or name.endswith("_test.py")


def changed_lines(old: str, new: str) -> int:
    a, b = old.splitlines(), new.splitlines()
    total = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != "equal":
            total += max(i2 - i1, j2 - j1)
    return total


def run_tests(files: dict[str, str], tests: dict[str, str], *, timeout_s: float = 60.0) -> tuple[bool, str]:
    """Write the repository plus the hidden tests in a scratch folder and run pytest there."""
    with tempfile.TemporaryDirectory(prefix="galton-rulebench-") as tmp:
        root = Path(tmp)
        for path, content in {**files, **tests}.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(root)}
        cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--rootdir", str(root), *sorted(tests)]
        try:
            done = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False, f"pytest timed out after {timeout_s:.0f}s"
        tail = (done.stdout + done.stderr).strip().splitlines()[-12:]
        return done.returncode == 0, "\n".join(tail)


@dataclass
class Sample:
    task_id: str
    kind: str
    variant: str
    pressure: str
    sample: int
    passed: bool
    lines_changed: int
    reference_lines: int
    excess_lines: int
    outside_files: list[str] = field(default_factory=list)
    kept_missing: list[str] = field(default_factory=list)
    wrote_test: bool = False
    test_first: bool = False
    no_files: bool = False
    error: str = ""
    test_output: str = ""
    answer: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_answer(task: dict, answer: str, *, variant: str = "", pressure: str = "", sample: int = 0, error: str = "",
                 timeout_s: float = 60.0, keep_answer: bool = True) -> Sample:
    original = task["files"]
    reference = apply_patch(original, task["patch"])
    ref_files = {p for p in reference if reference.get(p) != original.get(p)}
    blocks = parse_answer(answer) if not error else []
    hidden = set(task["tests"])
    edits = {p: c for p, c in blocks if p not in hidden}
    after = {**original, **edits}

    total = excess = 0
    outside, ref_lines = [], 0
    for path in sorted(set(after) | set(original) | ref_files):
        old, new = original.get(path, ""), after.get(path, "")
        n = changed_lines(old, new) if old != new else 0
        r = changed_lines(old, reference.get(path, "")) if path in ref_files else 0
        ref_lines += r
        if is_test_path(path) and path not in original:
            continue
        total += n
        if path in ref_files:
            excess += max(0, n - r)
        elif n:
            excess += n
            outside.append(path)
    joined_after = "\n".join(after.get(p, "") for p in original)
    missing = [s for s in task.get("keep", []) if s not in joined_after]

    order = [p for p, _ in blocks]
    first_test = next((i for i, p in enumerate(order) if is_test_path(p)), None)
    first_src = next((i for i, p in enumerate(order) if not is_test_path(p)), None)
    wrote_test = first_test is not None
    test_first = wrote_test and (first_src is None or first_test < first_src)

    if error or not edits:
        passed, out = False, error or "the answer has no file blocks"
    else:
        passed, out = run_tests(after, task["tests"], timeout_s=timeout_s)
    return Sample(task["id"], task["kind"], variant, pressure, sample, passed, total, ref_lines, excess, outside, missing,
                  wrote_test, test_first, no_files=not edits, error=error, test_output=out, answer=answer if keep_answer else "")


# --------------------------------------------------------------------------------------------------- pass@k and pass^k
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: chance that at least one of k samples drawn from n (c of them correct) passes."""
    if n <= 0 or k <= 0 or k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k: chance that k samples drawn from n (c of them correct) all pass; the reliability side of pass@k."""
    if n <= 0 or k <= 0 or k > n:
        return float("nan")
    return math.comb(c, k) / math.comb(n, k)
