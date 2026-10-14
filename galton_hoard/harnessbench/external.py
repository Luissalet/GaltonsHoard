"""Command-line harnesses: any agent that can be started with a command, told the task and pointed at a chat-completions endpoint.

The command is configured by the person (setting ``harnessbench.external``, one line per harness, ``name = command and arguments``; a line that
starts with ``#`` is switched off) and it is only ever run when ``harnessbench.allow_external`` is on. Nothing is installed or enabled by default: the
one example line ships commented out.

How a trial runs: the task's files and the documentation store are written to a scratch folder (``docs/<id>.txt``); the command runs there, without a
shell, with these placeholders replaced in its arguments and exported as environment variables:

    {prompt}       the task text, plus a line about the scratch folder       GALTON_PROMPT
    {prompt_file}  a text file that holds the same text                      GALTON_PROMPT_FILE
    {base_url}     the endpoint, ending in /v1                               GALTON_BASE_URL (and the usual base-URL variable)
    {model}        the model name the endpoint knows                         GALTON_MODEL
    {workdir}      the scratch folder                                        GALTON_WORKDIR

The last lines the command prints are taken as the final answer; the files it left in the folder are the state the verifier checks. The agent calls
its own tools, so the bench cannot see its turns, calls or format failures: those are ``None`` for these trials, never zero. A command that cannot be
started, or exits with an error and prints nothing, gives ``none`` (the verifier did not run). A command that does not finish within its time is
a failed answer: the agent did not get there.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..hoard_link import proc
from .loop import Trial
from .tasks import external_files
from .world import World, _norm_path

ID_PREFIX = "cli:"
ANSWER_LINES = 5
EXAMPLE_LINE = "# example = terminal-agent --base-url {base_url} --model {model} --prompt-file {prompt_file} --workdir {workdir}"
SUFFIX = ("\n\nWork inside the current folder. An absolute path in the task such as /data/x.txt means ./data/x.txt here. The documentation store is the folder ./docs, "
          "one text file per document. Print the final answer as the last line of your output.")


@dataclass(frozen=True)
class ExternalSpec:
    name: str
    argv: tuple[str, ...]
    enabled: bool = True

    @property
    def id(self) -> str:
        return ID_PREFIX + self.name

    def info(self) -> dict[str, Any]:
        return {"id": self.id, "base": "cli", "name": self.name, "command": list(self.argv), "enabled": self.enabled, "external": True}


def _split(command: str) -> list[str]:
    if os.name == "nt":
        return [p[1:-1] if len(p) > 1 and p[0] == p[-1] and p[0] in "\"'" else p for p in shlex.split(command, posix=False)]
    return shlex.split(command)


def parse_external(lines: list[str]) -> list[ExternalSpec]:
    """The harnesses written in the setting. A line that cannot be read is skipped (the settings page shows the raw text)."""
    out = []
    for raw in lines or []:
        text = str(raw).strip()
        enabled = not text.startswith("#")
        text = text.lstrip("# ").strip()
        name, sep, command = text.partition("=")
        name, command = name.strip().lower(), command.strip()
        if not sep or not name or not command or not all(c.isalnum() or c in "-_" for c in name):
            continue
        try:
            argv = _split(command)
        except ValueError:
            continue
        if argv:
            out.append(ExternalSpec(name, tuple(argv), enabled))
    return out


def read_answer(stdout: str) -> str:
    return "\n".join([ln for ln in str(stdout or "").splitlines() if ln.strip()][-ANSWER_LINES:])


def _files_back(workdir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in workdir.rglob("*"):
        rel = path.relative_to(workdir).as_posix()
        if path.is_file() and not rel.startswith("docs/") and path.stat().st_size <= 200_000:
            try:
                files["/" + rel] = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    return files


def run_external(spec: ExternalSpec, task: dict[str, Any], *, base_url: str, model: str, timeout_s: float = 600.0, sample: int = 0,
                 cancel: Optional[threading.Event] = None) -> Trial:
    trial = Trial(task_id=task["id"], harness=spec.id, sample=sample, turns=None, calls=None, format_failures=None, extra_calls=None, extra_repeat=None,
                  extra_after_solved=None, tokens_in=None, tokens_out=None)
    root = Path(tempfile.mkdtemp(prefix="galton-hb-"))
    work = root / "work"
    try:
        for rel, content in external_files(task).items():
            target = work / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        work.mkdir(exist_ok=True)
        prompt = task["prompt"] + SUFFIX
        prompt_file = root / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        v1 = base_url.rstrip("/") + ("" if base_url.rstrip("/").endswith("/v1") else "/v1")
        values = {"{prompt}": prompt, "{prompt_file}": str(prompt_file), "{base_url}": v1, "{model}": model, "{workdir}": str(work)}
        argv = []
        for token in spec.argv:
            for key, value in values.items():
                token = token.replace(key, value)
            argv.append(token)
        env = proc.build_env(None, GALTON_PROMPT=prompt, GALTON_PROMPT_FILE=prompt_file, GALTON_BASE_URL=v1, GALTON_MODEL=model, GALTON_WORKDIR=work,
                             GALTON_API_KEY="none", OPENAI_BASE_URL=v1, OPENAI_API_BASE=v1, OPENAI_API_KEY="none", OPENAI_MODEL=model)
        started = time.monotonic()
        try:
            done = proc.run(argv, timeout=timeout_s, cwd=work, env=env, cancel=cancel)
        except subprocess.TimeoutExpired as exc:
            trial.latency_ms = round((time.monotonic() - started) * 1000, 1)
            out = exc.output if isinstance(exc.output, str) else ""
            trial.outcome, trial.passed, trial.reason, trial.detail = "fail_answer", False, "timeout", f"the command did not finish within {timeout_s:.0f} s"
            trial.final = read_answer(out)
            trial.trace = [{"turn": 1, "text": read_answer(out), "exit": None, "timeout": True}]
            return trial
        except OSError as exc:
            trial.outcome, trial.passed, trial.reason, trial.detail = "none", None, "cannot_start", f"{type(exc).__name__}: {exc}"[:300]
            return trial
        trial.latency_ms = round((time.monotonic() - started) * 1000, 1)
        answer = read_answer(done.stdout)
        trial.final = answer
        trial.trace = [{"turn": 1, "text": answer, "stderr": proc.tail_lines(done.stderr), "exit": done.returncode}]
        if done.returncode != 0 and not answer:
            trial.outcome, trial.passed, trial.reason = "none", None, "command_failed"
            trial.detail = f"exit {done.returncode}: {proc.tail_lines(done.stderr, 4, 300)}"[:300]
            return trial
        world = World(files={_norm_path(p): c for p, c in _files_back(work).items()})
        ok, why = task["verify"](answer, world)
        trial.outcome, trial.passed, trial.reason, trial.detail = ("pass", True, "", "") if ok else ("fail_answer", False, "wrong_answer", why[:300])
        return trial
    finally:
        shutil.rmtree(root, ignore_errors=True)
