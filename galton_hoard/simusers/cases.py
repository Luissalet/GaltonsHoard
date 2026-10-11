"""The simulated-user cases (``cases.json``) and the prompt that makes a model play a persona."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..checkers import run_checker
from ..checkers.types import ModelOutput

CASES_FILE = Path(__file__).with_name("cases.json")

#: How each persona behaves. The text is what the simulator model is told.
PERSONAS: dict[str, str] = {
    "vague": ("Eres una persona sin conocimientos tecnicos. Pides las cosas de forma vaga y incompleta, sin dar cifras por tu cuenta. "
              "Solo cuentas un dato concreto cuando el asistente te lo pregunta, y solo ese dato. Hablas con frases cortas y coloquiales."),
    "hurry": ("Eres un experto con mucha prisa. Pides lo justo, en telegrafico, y te molestan las explicaciones largas. "
              "Si el asistente te pregunta algo imprescindible, contestas en una palabra o una cifra."),
    "selfcorrect": ("Eres una persona que se corrige a mitad de la conversacion. Tras la primera respuesta del asistente dices que te "
                    "habias equivocado en un dato y das el dato correcto, sin repetir la pregunta entera."),
}

END_MARKER = "[FIN]"


@dataclass
class Case:
    id: str
    persona: str
    opening: str
    hidden_goal: str
    checker: dict[str, Any]
    reference: str
    ambiguous: bool = False
    facts: list[str] = field(default_factory=list)
    max_turns: int = 5

    def check(self, final_answer: str) -> dict[str, Any]:
        return run_checker(self.checker, ModelOutput(text=final_answer))


def load_cases(path: Optional[Path] = None, *, only: Optional[list[str]] = None) -> list[Case]:
    data = json.loads((path or CASES_FILE).read_text(encoding="utf-8"))
    cases = [Case(**{k: v for k, v in raw.items() if k in Case.__dataclass_fields__}) for raw in data["cases"]]
    if only:
        wanted = set(only)
        cases = [c for c in cases if c.id in wanted]
    return cases


def persona_prompt(case: Case) -> str:
    """The system prompt of the simulator: who it is, what it knows and when to stop."""
    facts = "\n".join(f"- {f}" for f in case.facts) or "- (no tienes datos extra que dar)"
    return (
        f"{PERSONAS[case.persona]}\n\n"
        f"Tu objetivo oculto (el asistente no lo conoce, no lo recites): {case.hidden_goal}\n"
        f"Datos que conoces y solo das si te los piden o si encajan en tu papel:\n{facts}\n\n"
        "Escribes siempre como el usuario, en castellano de Espana, un solo mensaje corto por turno. "
        f"Cuando el asistente ya te haya dado el resultado que buscabas, o no veas que vaya a dartelo, responde solo con {END_MARKER}."
    )
