"""Rule variants (what goes in the system prompt) and pressure scenarios (what is added to the request).

A: no code rules. B: the four code rules in their original wording, rule 1 adapted to unattended work. C: the same four rules,
short and phrased only in positive terms. The texts match Faustus's bundled library rules ``common/karpathy-code`` and
``common/karpathy-code-positive``; ``load_variant_file`` reads any other rule file (frontmatter is dropped) so a new wording can
be measured without touching this module.
"""

from __future__ import annotations

import re
from pathlib import Path

BASE_SYSTEM = (
    "You are a coding assistant working on a small Python repository without a person watching. "
    "Answer with the complete new content of every file you change, each one in its own fenced block that starts with "
    "```file:<relative/path> and ends with ```. Do not output files you leave unchanged. "
    "You may add new test files. Keep any explanation outside the blocks short."
)

RULES_B = """\
- Think before you code. State your assumptions in one line and follow the
  simplest interpretation; if something is genuinely ambiguous and blocks
  you, ask one concrete question instead of guessing. Do not hide confusion.
- Simplicity first. Write the minimum code that solves the stated problem:
  no speculative features, no single-use abstractions, no error handling
  for cases that cannot happen here.
- Surgical changes. Touch only what the request needs; match the
  surrounding file's style; mention dead code or comments you notice but
  leave them alone. Every line you change traces back to the request.
- Goal-driven execution. Define the success criterion before you start
  (a test that reproduces the bug, a command that must exit 0) and keep
  working until you have verified it, reporting the real output.
"""

RULES_C = """\
- State your assumption in one line, then follow the simplest
  interpretation; ask one concrete question only when blocked.
- Write the smallest code that solves the stated problem; reuse what the
  codebase already has.
- Change only what the request needs; match the file's existing style; every
  changed line traces to the request.
- Name the success criterion first (a test, a command), then work until you
  have verified it and report the real output.
"""

#: id -> (label, rules text). The rules are appended to ``BASE_SYSTEM``.
VARIANTS: dict[str, tuple[str, str]] = {
    "A": ("no rules", ""),
    "B": ("four rules, original wording", RULES_B),
    "C": ("four rules, positive and short", RULES_C),
}

#: id -> sentence appended to the user's request. "contra" pushes against the rules, "favor" pushes with them.
PRESSURES: dict[str, str] = {
    "neutral": "",
    "favor": "Keep the change as small as possible and leave everything else exactly as it is.",
    "contra": "Do it fast, and while you are in there, tidy up the file: remove whatever looks unused and clean up old comments.",
}


def system_prompt(variant: str, rules_override: str = "") -> str:
    rules = rules_override if rules_override else VARIANTS[variant][1]
    return BASE_SYSTEM + ("\n\nRules for code changes:\n" + rules.strip() if rules.strip() else "")


def user_prompt(request: str, files: dict[str, str], pressure: str) -> str:
    parts = [request.strip()]
    extra = PRESSURES.get(pressure, "")
    if extra:
        parts.append(extra)
    parts.append("Repository files:")
    for path in sorted(files):
        parts.append(f"```file:{path}\n{files[path].rstrip()}\n```")
    return "\n\n".join(parts)


_FRONTMATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.S)


def load_variant_file(path: str | Path) -> str:
    """The body of a rule file (markdown with optional YAML frontmatter)."""
    text = Path(path).read_text(encoding="utf-8")
    return _FRONTMATTER.sub("", text, count=1).strip()
