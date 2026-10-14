"""Harness bench: the same model, driven by different harnesses, on the same small tool-using tasks.

A harness is the way a program talks to a model so that it uses tools: the server's function calling, a JSON block in the text, XML tags, or a
Thought / Action loop. The same model can score very differently under each, so a failure may belong to the model or to how it is spoken to. This
package runs a minimal agent loop per harness over 24 deterministic tasks with a virtual world (calculator, converter, document store, file system,
a service that fails once), separates format failures from wrong answers, counts needless tool calls, keeps a trial that never got to run as ``None``
(not zero), and reads the model x harness matrix. The pieces that need no database are importable on their own; ``runner.py`` and ``store.py`` plug
it into Galton's runs, GPU leases and scheduler.
"""

from .external import EXAMPLE_LINE, ExternalSpec, parse_external, run_external
from .harnesses import BASES, DEFAULT_HARNESSES, Harness, parse_harness_id
from .loop import OUTCOMES, Trial, run_trial
from .scoring import build_matrix, cell_stats, conclude
from .tasks import KINDS, TASKS, TASKS_BY_ID, select_tasks
from .world import TOOL_SPECS, World

__all__ = ["BASES", "DEFAULT_HARNESSES", "EXAMPLE_LINE", "KINDS", "OUTCOMES", "TASKS", "TASKS_BY_ID", "TOOL_SPECS", "ExternalSpec", "Harness", "Trial", "World",
           "build_matrix", "cell_stats", "conclude", "parse_external", "parse_harness_id", "run_external", "run_trial", "select_tasks"]
