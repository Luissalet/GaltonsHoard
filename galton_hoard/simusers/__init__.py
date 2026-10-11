"""Simulated users: a model plays a persona with a hidden goal and talks to the system under test for a few turns.

The family scores three things per case: whether the goal was reached (an exact checker on the final answer), how many turns it took,
and whether the assistant asked a clarifying question when the request was ambiguous. Nothing here needs a GPU: the simulator and the
target are plain callables, and the real ones are thin wrappers over an chat-completions compatible endpoint or the Faustus HTTP API.
"""

from .cases import PERSONAS, Case, load_cases, persona_prompt
from .scoring import CaseResult, asked_question, score_case, summarise
from .session import run_case, run_family
from .targets import FaustusTarget, OpenAISimulator, OpenAITarget

__all__ = [
    "PERSONAS", "Case", "CaseResult", "FaustusTarget", "OpenAISimulator", "OpenAITarget",
    "asked_question", "load_cases", "persona_prompt", "run_case", "run_family", "score_case", "summarise",
]
