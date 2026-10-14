"""The tasks of the harness bench: 24 small, deterministic jobs that need tools, each with a programmatic verifier.

A task gives the model a prompt, the tools it may use and the starting state of the virtual world. It is solved when the verifier accepts the
final answer (and the files the model wrote). Each task also carries:

* ``evidence`` and ``state``: what has to be on the table for the task to count as already solved. A tool call made after that point, or an
  identical call repeated after it succeeded, is an extra call (the bench reports them apart from wrong answers). ``evidence`` are strings that
  must appear in successful tool results; ``state`` checks the virtual files. A task with neither is solved from the start (the ones that need
  no tool): any call is extra.
* ``oracle``: the shortest honest path (tool calls, then the final answer). Tests replay it through every harness; fake models read it to
  speak their dialect. Recovery tasks include the failing first call.
* ``external``: whether an agent that has only a shell and real files can do it (the command-line harnesses).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from .world import DOCS, World

Verify = Callable[[str, World], tuple[bool, str]]

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    out = []
    for raw in _NUM.findall(str(text)):
        cleaned = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", raw)
        try:
            out.append(float(cleaned.replace(",", ".")))
        except ValueError:
            continue
    return out


def has_number(text: str, value: float, tol: float = 0.0) -> bool:
    return any(abs(n - value) <= tol + 1e-9 for n in numbers_in(text))


def say_number(value: float, tol: float = 0.0) -> Verify:
    def verify(final: str, world: World) -> tuple[bool, str]:
        ok = has_number(final, value, tol)
        return ok, "" if ok else f"expected {value:g} in the answer, found {numbers_in(final)[:6]}"
    return verify


def say_numbers(*values: float) -> Verify:
    def verify(final: str, world: World) -> tuple[bool, str]:
        missing = [v for v in values if not has_number(final, v, 0.005)]
        return not missing, "" if not missing else f"the answer lacks {', '.join(f'{v:g}' for v in missing)}"
    return verify


def say_words(*words: str) -> Verify:
    def verify(final: str, world: World) -> tuple[bool, str]:
        low = final.lower()
        missing = [w for w in words if w.lower() not in low]
        return not missing, "" if not missing else f"the answer lacks {', '.join(missing)}"
    return verify


def say_exactly(word: str) -> Verify:
    def verify(final: str, world: World) -> tuple[bool, str]:
        got = final.strip().strip(".!\"'` ").lower()
        return got == word.lower(), "" if got == word.lower() else f"expected exactly '{word}', got '{final.strip()[:60]}'"
    return verify


def wrote_number(path: str, value: float, tol: float = 0.0) -> Callable[[World], bool]:
    return lambda world: path in world.files and has_number(world.files[path], value, tol)


def wrote_text(path: str, text: str) -> Callable[[World], bool]:
    return lambda world: world.files.get(path, "").strip().lower() == text.lower()


def on_state(check: Callable[[World], bool], what: str) -> Verify:
    def verify(final: str, world: World) -> tuple[bool, str]:
        ok = check(world)
        return ok, "" if ok else f"the file system does not hold the right result ({what})"
    return verify


def _task(task_id: str, kind: str, prompt: str, tools: list[str], verify: Verify, oracle: list[tuple[str, dict[str, Any]]], final: str, *,
          evidence: tuple[str, ...] = (), state: Optional[Callable[[World], bool]] = None, world: Optional[dict[str, Any]] = None, external: bool = True) -> dict[str, Any]:
    return {"id": task_id, "kind": kind, "prompt": prompt, "tools": tools, "verify": verify, "oracle": oracle, "final": final, "evidence": list(evidence),
            "state": state, "world": world or {}, "external": external}


CALC, CONV, DOC, FS = ["calculator"], ["calculator", "convert"], ["search_docs", "read_doc"], ["list_dir", "read_file", "write_file"]

TASKS: list[dict[str, Any]] = [
    # -- calculator ------------------------------------------------------------------------------------------------
    _task("calc-sum-product", "calc", "What is (17 + 25) * 38? Use the calculator and answer with the number.", CALC, say_number(1596),
          [("calculator", {"expression": "(17 + 25) * 38"})], "1596", evidence=("1596",)),
    _task("calc-tax", "calc", "A shop sells 7 items at 12.50 each and adds 8% tax to the total. How much does the customer pay? Use the calculator.", CALC,
          say_number(94.5, 0.01), [("calculator", {"expression": "7 * 12.5 * 1.08"})], "94.5", evidence=("94.5",)),
    _task("calc-power", "calc", "What is 2 to the power of 20, minus 1000? Use the calculator.", CALC, say_number(1047576),
          [("calculator", {"expression": "2 ** 20 - 1000"})], "1047576", evidence=("1047576",)),
    _task("calc-average", "calc", "What is the average of 84, 91, 77, 96 and 102? Use the calculator.", CALC, say_number(90, 0.01),
          [("calculator", {"expression": "(84 + 91 + 77 + 96 + 102) / 5"})], "90", evidence=("90",)),
    # -- conversion ------------------------------------------------------------------------------------------------
    _task("conv-km-miles", "convert", "How many miles are 42 kilometres? Round to one decimal.", CONV, say_number(26.1, 0.06),
          [("convert", {"value": 42, "from_unit": "km", "to_unit": "mi"})], "26.1", evidence=("26.09",)),
    _task("conv-fahrenheit", "convert", "Convert 100 degrees Fahrenheit to Celsius. Round to one decimal.", CONV, say_number(37.8, 0.06),
          [("convert", {"value": 100, "from_unit": "f", "to_unit": "c"})], "37.8", evidence=("37.77",)),
    _task("conv-hours", "convert", "How many minutes are there in 3.5 hours?", CONV, say_number(210),
          [("convert", {"value": 3.5, "from_unit": "h", "to_unit": "min"})], "210", evidence=("210",)),
    # -- documentation store ---------------------------------------------------------------------------------------
    _task("doc-port", "docs", "Which port does the Orchid service listen on? Look it up in the documentation and answer with the number.", DOC, say_number(8443),
          [("search_docs", {"query": "Orchid service port"}), ("read_doc", {"id": "doc-001"})], "8443", evidence=("8443",)),
    _task("doc-cache", "docs", "For how many seconds does Basalt keep cache entries by default? Check the documentation.", DOC, say_number(90),
          [("search_docs", {"query": "Basalt cache"}), ("read_doc", {"id": "doc-002"})], "90", evidence=("90 seconds",)),
    _task("doc-bucket", "docs", "What is the name of the bucket used for Delta backups? Check the documentation.", DOC, say_words("delta-archive-eu"),
          [("search_docs", {"query": "Delta backups bucket"}), ("read_doc", {"id": "doc-004"})], "delta-archive-eu", evidence=("delta-archive-eu",)),
    _task("doc-replicas", "docs", "How many replicas does the Fjord database have and how many GB does each one hold? Check the documentation and give both numbers.", DOC,
          say_numbers(6, 250), [("search_docs", {"query": "Fjord database replicas"}), ("read_doc", {"id": "doc-006"})], "6 replicas, 250 GB each", evidence=("250 GB",)),
    # -- virtual file system ---------------------------------------------------------------------------------------
    _task("fs-count-lines", "files", "Read /notes/todo.txt and tell me how many lines it has.", FS, say_number(5),
          [("read_file", {"path": "/notes/todo.txt"})], "5", evidence=("call the plumber",),
          world={"files": {"/notes/todo.txt": "buy milk\nwrite report\nreply to Marta\nrenew the licence\ncall the plumber\n"}}),
    _task("fs-write-greeting", "files", "Write the text hello galton (lowercase, nothing else) to the file /out/greeting.txt.", FS,
          on_state(wrote_text("/out/greeting.txt", "hello galton"), "/out/greeting.txt"),
          [("write_file", {"path": "/out/greeting.txt", "content": "hello galton"})], "Done.", state=wrote_text("/out/greeting.txt", "hello galton")),
    _task("fs-count-files", "files", "How many files are in the folder /data/logs?", FS, say_number(4),
          [("list_dir", {"path": "/data/logs"})], "4", evidence=("app-4.log",),
          world={"files": {"/data/logs/app-1.log": "a", "/data/logs/app-2.log": "b", "/data/logs/app-3.log": "c", "/data/logs/app-4.log": "d", "/data/readme.txt": "x"}}),
    _task("fs-copy-port", "files", 'Read /config/app.json and write the value of its "port" field, and nothing else, to /out/port.txt.', FS,
          on_state(wrote_number("/out/port.txt", 7312), "/out/port.txt"),
          [("read_file", {"path": "/config/app.json"}), ("write_file", {"path": "/out/port.txt", "content": "7312"})], "Done.",
          state=wrote_number("/out/port.txt", 7312), world={"files": {"/config/app.json": '{"name": "demo", "port": 7312, "debug": false}'}}),
    # -- chains of two to four calls -------------------------------------------------------------------------------
    _task("chain-doc-calc", "chain", "According to the documentation, how many jobs can the Cinder queue process in 2.5 hours? Use the calculator for the arithmetic.",
          ["search_docs", "read_doc", "calculator"], say_number(6000),
          [("search_docs", {"query": "Cinder queue jobs"}), ("read_doc", {"id": "doc-003"}), ("calculator", {"expression": "40 * 60 * 2.5"})], "6000",
          evidence=("40 jobs per minute", "6000")),
    _task("chain-file-convert-write", "chain", "/data/trip.txt holds a distance in kilometres. Convert it to miles and write the result, rounded to one decimal and nothing else, "
          "to /out/trip_miles.txt.", ["read_file", "convert", "calculator", "write_file"], on_state(wrote_number("/out/trip_miles.txt", 74.6, 0.06), "/out/trip_miles.txt"),
          [("read_file", {"path": "/data/trip.txt"}), ("convert", {"value": 120, "from_unit": "km", "to_unit": "mi"}),
           ("write_file", {"path": "/out/trip_miles.txt", "content": "74.6"})], "Done.", state=wrote_number("/out/trip_miles.txt", 74.6, 0.06),
          world={"files": {"/data/trip.txt": "distance_km: 120\n"}}),
    _task("chain-sales-total", "chain", "Add up the four sales in /data/sales.csv, then compute 15% of that total. Give both numbers. Use the calculator for the arithmetic.",
          ["read_file", "calculator"], say_numbers(4400, 660),
          [("read_file", {"path": "/data/sales.csv"}), ("calculator", {"expression": "1200 + 950 + 1430 + 820"}), ("calculator", {"expression": "4400 * 0.15"})],
          "Total 4400; 15% is 660.", evidence=("4400", "660"), world={"files": {"/data/sales.csv": "north,1200\nsouth,950\neast,1430\nwest,820\n"}}),
    _task("chain-two-docs", "chain", "How many more days does Harbor keep its logs than Delta keeps its backups? Check the documentation for both and use the calculator for the difference.",
          ["search_docs", "read_doc", "calculator"], say_number(16),
          [("search_docs", {"query": "Delta backups"}), ("read_doc", {"id": "doc-004"}), ("search_docs", {"query": "Harbor logs"}), ("read_doc", {"id": "doc-008"}),
           ("calculator", {"expression": "30 - 14"})], "16", evidence=("14 days", "30 days", "16")),
    # -- no tool is needed -----------------------------------------------------------------------------------------
    _task("none-capital", "notool", "What is the capital of France? Answer with one word.", ["calculator", "search_docs", "read_doc"], say_words("paris"), [], "Paris"),
    _task("none-ready", "notool", "Reply with exactly the word READY and nothing else.", ["calculator", "read_file"], say_exactly("ready"), [], "READY"),
    # -- a tool fails and the model has to recover -----------------------------------------------------------------
    _task("recover-flaky-record", "recovery", "Look up the record REC-204 and tell me its status.", ["lookup_record"], say_words("shipped"),
          [("lookup_record", {"record_id": "REC-204"}), ("lookup_record", {"record_id": "REC-204"})], "shipped", evidence=("shipped",),
          world={"records_fail_first": 1}, external=False),
    _task("recover-wrong-path", "recovery", "Read /reports/q3.txt and tell me the total revenue it states.", FS, say_number(482),
          [("read_file", {"path": "/reports/q3.txt"}), ("read_file", {"path": "/reports/q3-summary.txt"})], "482 thousand euros", evidence=("482 thousand",),
          world={"files": {"/reports/q3-summary.txt": "Q3 total revenue: 482 thousand euros.\n", "/reports/q2-summary.txt": "Q2 total revenue: 455 thousand euros.\n"}}, external=False),
    _task("recover-doc-id", "recovery", "Read document 3 of the documentation store and tell me how many seconds pass between retries of a failed job.", DOC, say_number(20),
          [("read_doc", {"id": "3"}), ("read_doc", {"id": "doc-003"})], "20", evidence=("20 second",), external=False),
]
TASKS_BY_ID = {t["id"]: t for t in TASKS}
KINDS = ("calc", "convert", "docs", "files", "chain", "notool", "recovery")
assert len(TASKS_BY_ID) == len(TASKS)


def select_tasks(only: Any = (), kinds: Any = ()) -> list[dict[str, Any]]:
    only, kinds = set(only or ()), set(kinds or ())
    return [t for t in TASKS if (not only or t["id"] in only) and (not kinds or t["kind"] in kinds)]


def new_world(task: dict[str, Any]) -> World:
    return World.build(task["world"])


def is_solved(task: dict[str, Any], world: World, observations: list[str]) -> bool:
    """Is everything the task needs on the table? ``observations`` are the texts of the tool calls that succeeded."""
    if task["state"] is not None and not task["state"](world):
        return False
    joined = "\n".join(observations).lower()
    return all(e.lower() in joined for e in task["evidence"])


def public(task: dict[str, Any]) -> dict[str, Any]:
    return {"id": task["id"], "kind": task["kind"], "prompt": task["prompt"], "tools": task["tools"], "external": task["external"], "oracle_calls": len(task["oracle"])}


def external_files(task: dict[str, Any]) -> dict[str, str]:
    """What a command-line agent finds on disk for this task: the task's files and the documentation store (one text file per document)."""
    files = {p.lstrip("/"): c for p, c in new_world(task).files.items()}
    for doc in DOCS.values():
        files[f"docs/{doc['id']}.txt"] = f"{doc['title']}\n{doc['text']}\n"
    return files
