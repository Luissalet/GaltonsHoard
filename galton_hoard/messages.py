"""Messages the UI shows, as a stable ``key`` plus parameters.

The API keeps an English sentence in every message: assistants read it as it is. The UI is Spanish first, so each message also carries a
``key`` and its ``params``, and the client formats them through its own translation table (``msg_<key>``, ``err_<key>``, ``hint_<key>`` and
``notice_title_<kind>`` / ``notice_body_<kind>`` in ``client/src/i18n.js``). A test checks that every key defined here exists in both languages
with the same placeholders.

Three catalogues:

* ``ERRORS``: ``key -> (message, hint)`` for ``GaltonError``. ``GaltonError("invalid", "no_suite", ref="x")`` formats both from the keyword arguments.
* ``TEXTS``: ``key -> sentence`` for warnings, notes, plan lines, "runs on" lines and stored run errors.
* ``NOTICES``: ``kind -> (title, body)`` for the notices of the Panel.

Parameters are plain values: numbers are rounded where they are shown, lists are joined into one string. Placeholders are bare names (no
format specs), so that the client can fill them in without Python.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional

KEY = re.compile(r"^[a-z][a-z0-9_]*$")

# ------------------------------------------------------------------------------------------------- errors: key -> (message, hint)
ERRORS: dict[str, tuple[str, str]] = {
    # suites, cases and the importer
    "rapida_pick_missing": ("rapida picks a case that does not exist: {suite} / {title}", ""),
    "no_suite": ("No suite '{ref}'.", "List suites with suites_list."),
    "suite_builtin": ("{name} is a built-in suite and cannot be changed.", "Copy it with suite_duplicate and edit the copy."),
    "suite_empty": ("Suite {name} has no cases.", "Add cases with case_add or import some."),
    "no_suite_chosen": ("Choose at least one suite.", "List suites with suites_list."),
    "no_case": ("No case {id}.", "List the cases of a suite with suite_get."),
    "case_images": ("The case has {n} image(s).", ""),
    "case_invalid": ("The case is not valid: {problems}", ""),
    "case_duplicate_title": ("The suite already has a case titled '{title}'.", "Pick another title, or change the existing case with case_update."),
    "give_case_or_prompt": ("Give a saved `case` or a draft `prompt`.", ""),
    "shorthand_needs_value": ("The checker shorthand '{text}' needs a value after the colon.",
                              "Examples: exact:Madrid, contains:uno|dos, number:42, regex:^\\d+$, judge:Es claro y correcto."),
    "shorthand_unknown": ("Unknown checker shorthand '{kind}'.", "Use exact, contains, any, number, choice, regex, math, judge or none."),
    "not_a_number": ("'{value}' is not a number.", ""),
    "regex_invalid": ("The regular expression is not valid: {detail}", ""),
    "row_no_prompt": ("The row has no prompt (columns: prompt, pregunta, question, input or text).", ""),
    "checker_json": ("The checker is not valid JSON: {detail}", ""),
    "weight_number": ("The weight must be a number.", ""),
    "import_empty": ("There is nothing to import.", "Give JSONL (one JSON object per line) or CSV with a header row."),
    "json_array_invalid": ("The JSON array is not valid: {detail}", ""),
    "import_format": ("format must be auto, jsonl or csv.", ""),
    "import_rows": ("At most {max} rows per import.", "Split the file."),
    "give_text_or_path": ("Give either `text` or `path`.", ""),
    "import_file_missing": ("No such file: {path}", "Give the absolute path of an existing file."),
    "import_file_big": ("Import files can be at most 5 MB.", ""),
    "image_not_absolute": ("The image path must be absolute: {path}", ""),
    "image_missing": ("No such image: {path}", ""),
    "image_too_large": ("Images can be at most 10 MB.", ""),
    "image_type": ("{name} is not a PNG, JPEG, WebP or GIF image.", ""),
    # models
    "no_model": ("No model matches '{ref}'.", "List models with models_list."),
    "no_model_refresh": ("No model matches '{ref}'.", "List models with models_list, or run models_refresh."),
    "no_model_spec": ("No model matches '{ref}'.", "List the models with models_list, or give a spec like {{kind: 'gguf', path: ...}}."),
    "no_model_chosen": ("Choose at least one model.", "List models with models_list."),
    "model_disabled": ("{name} is disabled.", "Enable it with model_update."),
    "model_gone": ("{name} is no longer installed.", "Refresh the models, or pick another."),
    "model_busy_run": ("{name} is being measured right now.", "Cancel the run first."),
    "model_file_gone": ("The model file is gone: {path}", "Refresh the models."),
    "size_unknown": ("The size of {name} is unknown.", "Refresh the models, or check that the file still exists."),
    "remote_ok_local": ("{name} is a local model: remote_ok only applies to remote endpoints.", ""),
    "remote_not_allowed": ("{name} is a remote endpoint and was not enabled for measuring.",
                           "Remote calls can cost money and send prompts off this computer: enable it explicitly with model_update remote_ok=true."),
    "compare_same_model": ("Pick two different models.", ""),
    "contestant_spec": ("A contestant is an id, a name or a spec object.", ""),
    "unknown_kind": ("Unknown contestant kind '{kind}'.", "Use gguf, ollama or server."),
    "gguf_needs_path": ("A gguf contestant needs `path`.", "Give the absolute path of the .gguf file."),
    "path_not_absolute": ("`path` must be absolute: {path}", "Use the full path, for example D:\\models\\model.gguf."),
    "file_missing": ("No such file: {path}", "Check the path."),
    "file_unreadable": ("Cannot open {path}: {detail}", "Check the path and that the file is readable."),
    "is_projector": ("{name} is a vision projector, not a model.", "Give the model file; the projector goes in `mmproj`."),
    "no_projector": ("No such projector file: {path}", ""),
    "ollama_needs_model": ("An ollama contestant needs `model`.", ""),
    "ollama_not_installed": ("Ollama model '{model}' is not installed (no manifest found).", "Run models_refresh, or check the name with `ollama list`."),
    "server_needs_url_model": ("A server contestant needs `url` and `model`.", ""),
    "add_model_args": ("Give either `url` (a running server) or `path` (a .gguf file).", ""),
    "url_scheme": ("`url` must start with http:// or https://.", ""),
    "offline_no_probe": ("This instance runs offline (demo or test mode): servers cannot be probed.", ""),
    "server_no_answer_kind": ("{url} does not answer as a chat-completions server (llama-server and similar) or as Ollama.", "Check the address and that the server is running."),
    "server_pick_model": ("Say which `model` to use.", "The server offers: {options}."),
    "server_lacks_model": ("The server does not list '{model}'.", "It offers: {options}."),
    "unknown_api": ("Unknown API kind '{api}'.", "Use 'openai' or 'ollama'."),
    "no_vision": ("{name} has no vision and the case has an image.", "Pick a model with vision."),
    # servers and runs
    "server_no_answer": ("The server {url} does not answer.", "Start it, or refresh the models; it may have moved to another port."),
    "server_changed": ("The server at {url} now serves {model}; refresh the models.",
                       "This entry measures {expected}. models_refresh gives the model that is served now its own entry; this one keeps its history and comes back when the server serves it again."),
    "server_unverified": ("The server at {url} did not answer when the last {n} result(s) were to be checked, so they were not stored.", "Start the server and run it again."),
    "model_not_served": ("{name} is not served now: the server at {url} serves {model}.",
                         "Refresh the models, start {name} on that server again, or pick the model that is served now."),
    "server_busy": ("The server {url} stayed busy with somebody else's requests for {limit} s, so the run stopped waiting for it.",
        "Run it again later, or raise runner.wait_idle_max_s (0 waits for ever); Galton does not compete with your chats."),
    "ollama_not_loaded": ("{name} is not loaded in Ollama and loading it is switched off.",
                          "Enable runner.allow_ollama_load, or measure it through llama.cpp (the file-based entry of the same model)."),
    "ollama_no_answer": ("Ollama at {url} does not answer.", "Start Ollama, or refresh the models."),
    "no_llama_server": ("llama-server was not found (looked for {where} and for llama-server on PATH).",
                        "Set the path of llama-server in Settings > Runner, or run the model on a server that is already up."),
    "ports_busy": ("Ports 8091-8099 are all in use.", "Stop the llama-server processes that Galton left running (see servers.json) or wait for a run to end."),
    "server_start_failed": ("Could not start llama-server: {detail}", "Check the llama-server path in Settings."),
    "cancelled_loading": ("Cancelled while the model was loading.", ""),
    "server_exited": ("llama-server exited with code {exit_code} while loading.", "Last log lines: {log_tail}"),
    "server_not_ready": ("llama-server did not become ready within {seconds} s.", "Last log lines: {log_tail}"),
    "no_run": ("No run {id}.", "List runs with run_status."),
    "run_not_finished": ("Run {id} is still {state}.", "Cancel it with run_cancel and discard it once it has stopped."),
    "no_run_contestant": ("No contestant {contestant} in run {run}.", ""),
    "no_runs": ("There are no runs yet.", "Start one with run_start."),
    "run_unknown_settings": ("Unknown run settings: {names}.", "Known: {options}."),
    # GPUs
    "no_nvidia": ("No NVIDIA GPU was found (nvidia-smi gave no inventory).",
                  "Install the NVIDIA driver or run the model on a server that is already up (a server contestant)."),
    "allowed_missing": ("None of the allowed GPUs ({allowed}) exists on this computer.", "Choose the GPUs to use in Settings > GPUs."),
    "gpu_capacity": ("The model needs about {need_gb} GB and the allowed GPUs {present} hold {capacity_gb} GB in total.",
                     "Use a smaller quantisation or a shorter context, or allow another GPU in Settings (the owner's GPUs are reserved)."),
    "lease_refused": ("The GPU hub refused the lease: {reason}", "The model cannot fit on that GPU even when it is empty."),
    "lease_not_allowed": ("The lease landed on GPU {granted}, which is not in the allowed list {allowed}.", "Galton never uses a GPU outside Settings > GPUs; nothing was loaded."),
    "cancelled_waiting_gpu": ("Cancelled while waiting for a GPU.", ""),
    "no_gpu_free": ("No allowed GPU has {need_gb} GB free.", "Wait for the other job to finish, free memory, or try again later."),
    "cpu_no_ram": ("The model needs about {need_gb} GB of memory to run on the CPU and only {free_gb} GB are free.", "Close other programs, or wait until memory is free."),
    "no_gpu_free_waited": ("No allowed GPU has {need_gb} GB free after waiting.", "Wait for the other job to finish, free memory, or try again later."),
    "gpu_reserved": ("GPU {gpus} is reserved for the owner of this computer.", "Repeat the call with confirm_reserved=true only if the user explicitly allowed it."),
    # settings
    "setting_bool": ("{setting} must be true or false.", ""),
    "setting_number": ("{setting} must be a number.", ""),
    "setting_range": ("{setting} must be between {low} and {high}.", ""),
    "setting_choice": ("{setting} must be one of {options}.", ""),
    "setting_intlist": ("{setting} must be a list of whole numbers.", ""),
    "setting_intlist_range": ("{setting}: {number} is out of range.", ""),
    "setting_object": ("{setting} must be an object.", ""),
    "setting_unknown_field": ("Unknown field {name} in {setting}.", "Known: {options}."),
    "setting_field_number": ("{setting}.{name} must be a number.", ""),
    "setting_field_range": ("{setting}.{name} must be between {low} and {high}.", ""),
    "setting_type": ("Unsupported setting type for {setting}.", ""),
    "setting_unknown": ("Unknown setting {setting}.", "Known: {options}."),
    # arena, board, routes, tools
    "arena_bad_vote": ("vote must be one of {options}.", ""),
    "arena_no_pair": ("No arena pair {pair}.", "Ask for a pair with arena_next."),
    "arena_already_voted": ("That pair already has a vote.", "Ask for a new pair with arena_next."),
    "unknown_category": ("Unknown category '{category}'.", "Use one of: {options}."),
    "unknown_task_category": ("Unknown task category '{category}'.", "Use one of: {options}."),
    "nothing_to_publish": ("There is nothing to publish: no task category has enough checked cases.",
                           "Run suites on your models first (see recommend or run_start); a category needs at least routes.min_cases checked cases."),
    "confirm_delete": ("Deleting {what} is permanent.", "Repeat the call with confirm=true if the user asked for it."),
    "confirm_discard": ("Discarding {what} removes its results from every statistic, ranking, route and comparison.",
                        "Repeat the call with confirm=true if the user asked for it; run_restore brings them back."),
}

# ------------------------------------------------------------------------------------------------- texts: key -> sentence
TEXTS: dict[str, str] = {
    # where and how a model would run (plan, model detail, run cards)
    "where_server": "server {url}",
    "where_server_resident": "server {url} (resident)",
    "where_gpus": "GPU {gpus}",
    "where_wait": "would wait for room on the allowed GPUs",
    "where_cpu": "CPU (no GPU is free)",
    "where_cpu_forced": "CPU (chosen in the run)",
    "runs_on_own": "llama.cpp (own) on GPU {gpus}",
    "runs_on_cpu": "llama.cpp (own) on the CPU, {threads} threads",
    "runs_on_server": "llama-server {url}",
    "runs_on_ollama": "ollama {url}",
    "runs_on_ollama_loaded": "ollama {url} (loaded by Galton)",
    # notes and warnings of a plan or a run
    "note_vision_dropped": "{n} vision case(s) were not run: {name} has no vision",
    "warn_split": "split across GPUs {gpus} ({split})",
    "warn_cpu": "Measured on the CPU ({threads} threads), not on a GPU: its speed is not comparable with GPU numbers",
    "warn_ollama_gpu": "Ollama decides which GPU it loads the model on; the lease only reserves the memory",
    "warn_spill": "{pct} % of the model is in system RAM, not on the GPU (speeds are lower than a full GPU load)",
    "warn_lease_hub": "GPU lease hub not reachable; local check only: {detail}",
    "warn_will_fail": "{name} will fail: {problem}",
    # a request to a model
    "skip_context": "skipped: context ({needed} tokens needed, {available} available)",
    "chat_timeout": "timeout: no answer within {seconds} s",
    "chat_timeout_detail": "timeout: {detail}",
    "chat_connection": "connection error: {detail}",
    "chat_rejected": "the server kept rejecting the request",
    "chat_error": "{detail}",
    "internal_error": "internal error: {detail}",
    "note_reasoning_retry": "the server rejected the reasoning fields; retried without them",
    "note_tools_in_prompt_server": "tools_in_prompt: the server does not take the tools field; they were described in the prompt",
    "note_think_retry": "the model does not take the think option; retried without it",
    "note_tools_in_prompt_model": "tools_in_prompt: the model does not support tools; they were described in the prompt",
    "note_reasoning_budget_retry": "the answer was cut while the model was still reasoning; asked again with {tokens} more tokens",
    "warn_reasoning_budget": "This model reasons: {tokens} tokens are added to the answer budget of each case so that its thinking does not use up the answer",
    # runs
    "run_stopped": "Galton was stopped while this run was in progress.",
    "rc_interrupted": "interrupted",
    "run_all_failed": "Every model failed; the reason is on each model below.",
    "run_finished": "the run had already finished",
    "run_stopping_own": "stopping: the request in flight is dropped and the llama-server that Galton started is stopped",
    "run_stopping_shared": "stopping: the request in flight is dropped; the shared server at {urls} is not touched and keeps running",
    "run_stopping_none": "stopping: the request in flight is dropped",
    "run_stopping_both": "stopping: the request in flight is dropped and the llama-server that Galton started is stopped; the shared server at {urls} is not touched and keeps running",
    "warn_results_dropped": "{n} result(s) answered since the server was last checked were not stored",
    "run_discard_note": "Its results no longer count in any statistic, ranking, route, comparison or regression check; the run stays in the history. run_restore brings them back.",
    "run_restore_note": "Its results count again.",
    "run_not_discarded_note": "the run was not discarded",
    "judge_none": "no judge model is chosen (setting judge.contestant)",
    "judge_unavailable": "the judge {name} is not available",
    "judge_not_started": "the judge could not be started: {reason}",
    # comparisons
    "cmp_no_shared": "The two models share no checked case: run both on the same suites.",
    "cmp_few_cases": "Only {n} cases in common (fewer than {min}): the verdict is weak, measure more cases before deciding.",
    "cmp_disagree": "The pass/fail test and the score bootstrap disagree, so no difference is claimed.",
    "board_stale_left_out": "Results of {names} are for an older version of the file and were left out (pass include_stale to use them).",
    "board_truncated": "{name}: {n} of {total} results in {category} ({pct} %) ended with no answer because the token budget ran out: the budget was too small, so the score understates the model",
    "board_unequal_results": "{a} has {who_a} and {b} has {who_b} checked results in {label}.",
    # routes
    "label_watch": "watch: {name}",
    "label_measure_new": "Measure what is new ({n})",
    "measure_new_none": "Every enabled model already has results on {name} for its current version.",
    "model_removed_note": "It is back after the next refresh if it is still installed; use model_update enabled=false to keep it out of the way instead.",
    "row_not_object": "not an object",
    "row_bad_json": "not valid JSON ({detail})",
    "arena_no_pair_note": "No pair to compare: the arena needs two models with stored answers to the same case, and every pair so far has a vote.",
    "suite_removed_note": "Results already measured on its cases stay in the history of the runs.",
    "refresh_file_error": "{name}: {detail}",
    "excl_disabled": "disabled or no longer installed",
    "excl_adhoc": "evaluation-only model (keep it to route to it)",
    "excl_remote": "remote endpoint (routes.include_remote is off)",
    "excl_no_results": "no recent results on these suites",
    "excl_stale": "results are for another version of the model",
    "excl_too_few": "fewer checked cases than routes.min_cases",
    "excl_too_slow": "slower than routes.min_tok_s",
    "excl_too_big": "needs more memory than routes.max_vram_gb",
    # servers that are not chat models
    "not_chat_no_template": "This server has no chat template: it is not a chat model.",
    "not_chat_failed": "The server answered the chat probe with an error ({status}): it is not a chat model.",
    # notes of the tools
    "case_no_checker": "This case has no checker: its answers are recorded (and usable in the arena) but it does not count towards any score.",
    "case_judge_pending": "This case is graded by the judge model (setting judge.contestant); until one is chosen its results stay pending.",
    "model_remote_note": "This is a remote endpoint: it will not be called until remote_ok is enabled (it can cost money and sends prompts off this computer).",
    "case_results_deleted": "{n} earlier result(s) of this case were deleted.",
    "case_results_old": "{n} earlier result(s) of this case measured the old version. Pass forget_results=true to delete them.",
    # problems with a case
    "prob_messages": "prompt.messages: must be a list of {{role, content}}",
    "prob_prompt_empty": "prompt: empty",
    "prob_weight_positive": "weight: must be positive",
    "prob_weight_number": "weight: must be a number",
    "prob_tools": "tools: must be a list of function definitions in the chat-completions tools format",
    "prob_checker_object": "{path}: must be an object with a `type`",
    "prob_checker_checks": "{path}: `{kind}` needs a non-empty `checks` list",
    "prob_checker_unknown": "{path}: unknown checker type '{kind}' (known: {options})",
}

# ------------------------------------------------------------------------------------------------- notices: kind -> (title, body)
NOTICES: dict[str, tuple[str, str]] = {
    "new_model": ("New model: {name}", "Measure it to see where it fits."),
    "changed_model": ("{name} changed", "The model file is different from the one that was measured: earlier results are stale."),
    "regression": ("{name} got worse", "On {n} shared cases it scores {diff} lower than before the model changed (p = {p})."),
    "improvement": ("{name} improved", "On {n} shared cases it scores {diff} higher than before the model changed."),
    "run_failed": ("Run failed: {label}", "{error}"),
    "not_served": ("{name} is not served now", "The server at {url} serves {model}. Refresh the models; this entry keeps its history and comes back when the server serves it again."),
}


def fields_of(template: str) -> set[str]:
    """The placeholder names of a template (``{{`` is a literal brace)."""
    return set(re.findall(r"(?<!\{)\{([a-z_][a-z0-9_]*)\}", template.replace("{{", "\0\0")))


def fill(template: str, params: dict[str, Any]) -> str:
    """``template`` with ``params`` filled in; a missing parameter leaves its placeholder, so a message is never lost to a typo."""
    class _Safe(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(_Safe(params))


def scalar(value: Any) -> Any:
    """A parameter value the client can print: numbers and text stay, lists are joined, everything else becomes text."""
    if isinstance(value, bool) or value is None:
        return "" if value is None else str(value).lower()
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(scalar(v)) for v in value)
    return str(value)


class CodedText(str):
    """An English sentence that remembers its ``key`` and ``params``: it behaves as a ``str`` everywhere, and ``item()`` gives the wire form."""

    key: str
    params: dict[str, Any]

    def __new__(cls, text: str, key: str = "", params: Optional[dict[str, Any]] = None) -> "CodedText":
        obj = super().__new__(cls, text)
        obj.key = key
        obj.params = dict(params or {})
        return obj

    def item(self) -> dict[str, Any]:
        return {"key": self.key, "params": {k: wire(v) for k, v in self.params.items()}, "text": str(self)}


def text(key: str, **params: Any) -> CodedText:
    """The coded sentence ``TEXTS[key]`` filled in with ``params``."""
    template = TEXTS[key]
    clean = {k: scalar(v) for k, v in params.items()}
    return CodedText(fill(template, clean), key, clean)


def item(value: Any) -> dict[str, Any]:
    """The wire form of a message that may be a ``CodedText``, a plain string (a legacy row, free text) or already an item."""
    if isinstance(value, CodedText) and value.key:
        return value.item()
    if isinstance(value, dict) and "text" in value:
        return {"key": value.get("key", ""), "params": value.get("params", {}), "text": str(value["text"])}
    return {"key": "", "params": {}, "text": str(value)}


def items(values: Iterable[Any]) -> list[dict[str, Any]]:
    return [item(v) for v in values]


def error_item(code: str, key: str, **params: Any) -> dict[str, Any]:
    """The wire form of an error from the catalogue: ``{key, params, text, hint}``."""
    message, hint = ERRORS[key]
    clean = {k: scalar(v) for k, v in params.items()}
    return {"code": code, "key": key, "params": clean, "text": fill(message, clean), "hint": fill(hint, clean)}


def hint_of(value: Any) -> str:
    """The hint that goes with a coded error message (what to do about it), or an empty string."""
    key = getattr(value, "key", "")
    if key in ERRORS:
        return fill(ERRORS[key][1], getattr(value, "params", {}))
    return ""


def notice_text(kind: str, params: dict[str, Any]) -> tuple[str, str]:
    """The English title and body of a notice."""
    title, body = NOTICES.get(kind, (kind, ""))
    clean = {k: scalar(v) for k, v in params.items()}
    return fill(title, clean), fill(body, clean)


# ------------------------------------------------------------------------------------------------- recognising stored sentences
# A run keeps its sentences as plain English text (what an assistant reads). To show them in the language of the UI they are matched back to
# the catalogue: every template becomes an anchored pattern whose placeholders take the parameter values. Rows written before the keys
# existed are recognised the same way.
GENERIC = {"chat_error"}  # a template that is only a placeholder would match anything
MIN_LETTERS = 3  # letters of fixed text a template needs before it is used to recognise a stored sentence


def _letters(key: str) -> int:
    template = TEXTS.get(key) or ERRORS.get(key, ("",))[0]
    fixed = re.sub(r"\{[a-z_][a-z0-9_]*\}", "", template)
    return sum(ch.isalpha() for ch in fixed)


def _pattern(template: str) -> tuple["re.Pattern[str]", tuple[str, ...], int]:
    pieces: list[str] = []
    names: list[str] = []
    literal = 0
    for part in re.split(r"(\{\{|\}\}|\{[a-z_][a-z0-9_]*\})", template):
        if part in ("{{", "}}"):
            pieces.append(re.escape(part[0]))
            literal += 1
        elif part.startswith("{") and part.endswith("}"):
            names.append(part[1:-1])
            pieces.append(f"(?P<p{len(names) - 1}>.+?)")
        elif part:
            pieces.append(re.escape(part))
            literal += len(part)
    return re.compile("^" + "".join(pieces) + "$", re.DOTALL), tuple(names), literal


_PATTERNS: list[tuple["re.Pattern[str]", str, tuple[str, ...]]] = []


def _patterns() -> list[tuple["re.Pattern[str]", str, tuple[str, ...]]]:
    if not _PATTERNS:
        found = []
        for key, template in TEXTS.items():
            if key not in GENERIC:
                found.append((*_pattern(template), key))
        for key, (template, _hint) in ERRORS.items():
            found.append((*_pattern(template), key))
        # a template whose fixed text has no words ("{name}: {detail}") would claim any short sentence
        found = [f for f in found if _letters(f[3]) >= MIN_LETTERS]
        found.sort(key=lambda f: -f[2])
        _PATTERNS.extend((rx, key, names) for rx, names, _literal, key in found)
    return _PATTERNS


def recognise(value: Any, prefix: str = "") -> Any:
    """``value`` as a ``CodedText`` when it is a sentence of the catalogue (a stored run keeps plain text), otherwise unchanged.

    ``prefix`` limits the keys that may match (a run label is free text written by a person or another app: only ``label_*`` keys apply)."""
    if not isinstance(value, str) or isinstance(value, CodedText) or len(value) < 6:
        return value
    for rx, key, names in _patterns():
        if prefix and not key.startswith(prefix):
            continue
        found = rx.match(value)
        if found:
            return CodedText(value, key, {name: _number(found.group(f"p{i}")) for i, name in enumerate(names)})
    return value


def recognise_all(values: Iterable[Any]) -> list[Any]:
    return [recognise(v) for v in values]


def notice_params(kind: str, title: str, body: str) -> dict[str, Any]:
    """The parameters of a notice stored before notices had them, read back from its English title and body."""
    out: dict[str, Any] = {}
    spec = NOTICES.get(kind)
    if spec is None:
        return out
    for template, source in zip(spec, (title, body)):
        if not fields_of(template):
            continue
        rx, names, _literal = _pattern(template)
        found = rx.match(source or "")
        if found:
            out.update({name: _number(found.group(f"p{i}")) for i, name in enumerate(names)})
    return out


def _number(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def wire(value: Any) -> Any:
    """``value`` as the bundled UI receives it: every ``CodedText`` becomes ``{key, params, text}`` so the client can translate it. The REST agent
    route and MCP never go through this: assistants get the plain English sentences."""
    if isinstance(value, CodedText):
        return value.item() if value.key else str(value)
    if isinstance(value, dict):
        return {k: wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire(v) for v in value]
    return value
