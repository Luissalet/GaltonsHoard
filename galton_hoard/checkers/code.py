"""The ``python_tests`` checker: run the code block of an answer against hidden asserts in a separate, isolated Python process.

``python -I`` in a fresh temporary directory, a minimal environment, a timeout, capped output and (on POSIX) memory and CPU limits set
by the harness itself. The model's code runs with the user's own privileges and no network sandbox: that is why it can be switched
off (setting ``checks.allow_code_execution``); a case that needs it then reports ``unavailable`` instead of failing.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from ..hoard_link import proc as hl_proc
from ..procs import kill_tree
from .textutil import split_reasoning
from .types import CheckContext, ModelOutput, bad_spec, unavailable, verdict

OUTPUT_CAP = 4000

HARNESS = r'''
import json, sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3, 2 * 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 ** 2, 10 * 1024 ** 2))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
except Exception:
    pass
spec = json.load(open(sys.argv[1], encoding="utf-8"))
result = {"load_error": "", "missing_entry": False, "tests": []}

def save():
    with open(spec["result_file"], "w", encoding="utf-8") as fh:
        json.dump(result, fh)

namespace = {"__name__": "solution"}
try:
    exec(compile(open(spec["code_file"], encoding="utf-8").read(), "solution.py", "exec"), namespace)
except BaseException as exc:
    result["load_error"] = (type(exc).__name__ + ": " + str(exc))[:300]
if spec.get("entry") and spec["entry"] not in namespace:
    result["missing_entry"] = True
save()
for number, test in enumerate(spec["tests"]):
    try:
        exec(compile(test, "hidden test", "exec"), namespace)
        result["tests"].append({"ok": True})
    except BaseException as exc:
        result["tests"].append({"ok": False, "error": (type(exc).__name__ + ": " + str(exc))[:200]})
    save()
'''

# a fence with its info string (```python, ```Python3, ```py title="x.py", ``` alone): the language decides whether the block is Python
_BLOCK = re.compile(r"```[ \t]*([\w+.#-]*)[^\n`]*\r?\n(.*?)```", re.S)
_OPEN_BLOCK = re.compile(r"```[ \t]*([\w+.#-]*)[^\n`]*\r?\n(.*)$", re.S)
_PYTHON_LANGS = ("", "python", "python3", "py", "py3", "pycon", "ipython")


def _python_blocks(pattern: re.Pattern, text: str) -> list[str]:
    """The bodies of the fenced blocks that are Python (or say no language), leaving out ```bash, ```json and the like."""
    return [body for lang, body in pattern.findall(text) if lang.lower() in _PYTHON_LANGS]


def extract_code(text: str, entry: str = "") -> str:
    """The Python of an answer: the last fenced block that defines ``entry`` (else all blocks joined); unfenced code when it looks like code."""
    text = split_reasoning(text)[0]
    blocks = _python_blocks(_BLOCK, text)
    if not blocks and not _BLOCK.search(text):
        blocks = _python_blocks(_OPEN_BLOCK, text)
    if blocks:
        if entry:
            defining = [b for b in blocks if re.search(rf"\bdef\s+{re.escape(entry)}\b|\b{re.escape(entry)}\s*=", b)]
            if defining:
                return defining[-1]
        return "\n\n".join(blocks)
    if re.search(r"^\s*(def |class |import |from \w+ import )", text, re.M):
        return text
    return ""


def _env() -> dict[str, str]:
    env = {"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
    for key in ("SYSTEMROOT", "SystemRoot", "TEMP", "TMP"):
        if os.environ.get(key) and sys.platform.startswith("win"):
            env[key] = os.environ[key]
    return env


def run_asserts(code: str, tests: list[str], entry: str = "", timeout_s: float = 10.0) -> dict[str, Any]:
    """Run ``tests`` (statements) after ``code``. Returns ``{passed, total, load_error, missing_entry, timed_out, stdout, stderr}``."""
    # ignore_cleanup_errors: on Windows a killed child (or one it started) can still hold a file for a moment; a folder left behind
    # in the temp folder must not turn a measured answer into an internal error
    with tempfile.TemporaryDirectory(prefix="galton-code-", ignore_cleanup_errors=True) as tmp:
        work = Path(tmp)
        (work / "solution.py").write_text(code, encoding="utf-8")
        (work / "harness.py").write_text(HARNESS, encoding="utf-8")
        (work / "spec.json").write_text(json.dumps({"code_file": str(work / "solution.py"), "tests": tests, "entry": entry,
                                                     "result_file": str(work / "result.json")}), encoding="utf-8")
        out_path, err_path = work / "stdout.txt", work / "stderr.txt"
        timed_out = False
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            proc = hl_proc.popen([sys.executable, "-I", str(work / "harness.py"), str(work / "spec.json")], cwd=work, env=_env(),
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=err)
            deadline = time.monotonic() + timeout_s
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    timed_out = True
                    kill_tree(proc)
                    break
                time.sleep(0.02)
        try:
            data = json.loads((work / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {"load_error": "the process produced no result", "missing_entry": False, "tests": []}

        def tail(path: Path) -> str:
            try:
                return path.read_bytes()[:OUTPUT_CAP].decode("utf-8", "replace")
            except OSError:
                return ""

        results = data.get("tests", [])
        return {"passed": sum(1 for t in results if t.get("ok")), "total": len(tests), "load_error": data.get("load_error", ""),
                "missing_entry": bool(data.get("missing_entry")), "timed_out": timed_out, "stdout": tail(out_path), "stderr": tail(err_path),
                "errors": [t.get("error", "") for t in results if not t.get("ok")][:3]}


def check_python_tests(spec: dict[str, Any], output: ModelOutput, ctx: CheckContext) -> dict[str, Any]:
    tests = spec.get("tests")
    tests = [tests] if isinstance(tests, str) else list(tests or [])
    if not tests:
        return bad_spec("python_tests needs `tests`")
    if not ctx.allow_code:
        return unavailable("code execution is switched off (checks.allow_code_execution)")
    entry = str(spec.get("entry", ""))
    code = extract_code(output.text, entry)
    if not code.strip():
        return verdict(0.0, False, {"reason": "no Python code in the answer"})
    run = run_asserts(code, tests, entry, float(spec.get("timeout_s", ctx.code_timeout_s)))
    score = run["passed"] / run["total"] if run["total"] else 0.0
    return verdict(score, run["passed"] == run["total"] and not run["timed_out"], {k: v for k, v in run.items() if v not in ("", [], False)} | {"tests": run["total"]})
