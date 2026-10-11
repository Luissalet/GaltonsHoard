"""Rule bench: measures how a set of code rules in the system prompt changes what a model does to a small repository.

Twenty starter tasks (bugs, small features, refactor traps with dead code and other people's comments) each carry hidden tests
and a reference patch. Every task runs under rule variants (A none, B original wording, C positive and short) and pressure
scenarios (neutral, favor, contra), k samples each. Scoring is deterministic: hidden tests, lines changed against the reference,
files outside the reference, protected snippets removed, and whether a test came before the code. The summary reports pass@1,
pass@k, pass^k and a verdict per variant against the no-rules baseline.
"""

from .runner import TARGET_EXCESS_CUT, ChatGenerator, run_bench, samples_from_report, select_tasks, summarise
from .scoring import Sample, changed_lines, parse_answer, pass_at_k, pass_hat_k, run_tests, score_answer
from .tasks import TASKS, apply_patch
from .variants import PRESSURES, VARIANTS, load_variant_file, system_prompt, user_prompt

__all__ = [
    "PRESSURES", "TARGET_EXCESS_CUT", "TASKS", "VARIANTS", "ChatGenerator", "Sample", "apply_patch", "changed_lines",
    "load_variant_file", "parse_answer", "pass_at_k", "pass_hat_k", "run_bench", "run_tests", "samples_from_report", "score_answer", "select_tasks",
    "summarise", "system_prompt", "user_prompt",
]
