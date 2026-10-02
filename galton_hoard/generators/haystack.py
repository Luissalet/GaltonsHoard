"""Long-context cases: a needle (one fact, or two linked facts) hidden at a given depth in generated Spanish filler prose.

Everything derives from a seed, so a case is a few numbers in the database and the (large) prompt is built when it runs.
The filler has no digits and no proper names, so the only figures and names in the document are the planted ones and the decoys.
"""

from __future__ import annotations

import math
import random
from typing import Any

from ..util import stable_hash

TOKEN_CHARS = 3.5  # characters per token: the approximation the specification asks for
OVERHEAD_TOKENS = 400  # instructions + question
LENGTHS = (4_000, 8_000, 16_000, 32_000, 64_000)
DEPTHS = (0.10, 0.50, 0.90)

_SUBJECTS = ["*los vecinos del barrio", "la cooperativa agrícola", "el viejo molino", "*las lavanderas del río", "el carpintero del pueblo", "*los niños de la escuela",
             "la panadera de la esquina", "el tren de la tarde", "*los pastores del valle", "el farmacéutico jubilado", "*las golondrinas de la ermita", "el herrero de la calle larga",
             "la asociación de vecinos", "*los pescadores del embalse", "el cartero de la sierra", "la maestra rural", "*los jubilados de la plaza", "el alcalde del municipio",
             "*las costureras del taller", "el mercado de los jueves"]
_VERBS = ["recordaban", "comentaban", "preparaban", "esperaban", "revisaban", "celebraban", "discutían", "recogían", "arreglaban", "organizaban", "vigilaban", "repartían",
          "guardaban", "limpiaban", "anunciaban", "estudiaban"]
_OBJECTS = ["la cosecha del año anterior", "las reformas del lavadero", "una fiesta para el fin de semana", "el calendario de las lluvias", "las herramientas del taller",
            "el reparto del agua entre las huertas", "la vieja campana de la iglesia", "los caminos que bajan al río", "las cuentas de la cooperativa", "una receta de la abuela",
            "el estado de la carretera comarcal", "los árboles que dan sombra a la plaza", "las historias de los antepasados", "el precio de la leña", "la limpieza de las acequias",
            "los horarios del autobús", "una exposición de fotografías antiguas", "el huerto comunitario", "las normas de la biblioteca", "los planes para la romería"]
_WHEN = ["durante el otoño", "al caer la tarde", "cada primavera", "en las mañanas frías", "después de la cosecha", "antes de que llegara el invierno", "con las primeras lluvias",
         "a la hora del café", "los domingos por la mañana", "sin ninguna prisa", "mientras el sol se ponía", "tras una larga jornada", "en las noches de verano", "al final de la semana"]
_LINKS = ["Con el tiempo,", "Según contaban los más mayores,", "Como siempre,", "A pesar del cansancio,", "Sin que nadie lo pidiera,", "Aquel año,", "Poco a poco,", "Por esa razón,", "De todos modos,", ""]
_TAILS = ["y nadie pareció tener inconveniente", "aunque a veces surgían pequeñas discusiones", "con la calma de quien no tiene nada urgente", "como había ocurrido durante generaciones",
          "y así se mantuvo la costumbre", "entre risas y algún que otro refrán", "mientras el humo de las chimeneas subía despacio", "sin que cambiara demasiado el ritmo del lugar", ""]

_PEOPLE = ["Aurelio Zuloaga", "Beatriz Montenegro", "Casimiro Ledesma", "Dolores Quintana", "Eustaquio Barroso", "Fabiola Cordero", "Gaspar Villanueva", "Herminia Saavedra",
           "Ildefonso Peñalver", "Jacinta Roldán", "Leocadio Maldonado", "Macarena Toledano", "Nicanor Belmonte", "Olegaria Fontecha", "Prudencio Arteaga", "Remedios Alcántara"]
_PLACES = ["la bodega Almenara", "el archivo de Cañamar", "la biblioteca de Roblehondo", "la caja fuerte de Valdelinares", "el laboratorio de Peñarroya", "la taquilla de Sotoverde",
           "el almacén de Fuenlabrada Alta", "el observatorio de Cerrolargo", "la consigna de Torrequemada", "la cámara de Altamira Baja"]
_THINGS = ["el código de acceso", "la clave de la caja", "el número de la taquilla", "la contraseña del archivo", "el número de serie", "el código del candado"]
_PROJECTS = ["Faro", "Colmena", "Aurora", "Meridiano", "Brújula", "Cometa", "Sendero", "Marea"]


def approx_tokens(chars: int) -> int:
    return math.ceil(chars / TOKEN_CHARS)


def plan(params: dict[str, Any]) -> dict[str, Any]:
    """The planted facts, the question, the expected answer and the decoys. Cheap: it does not build the filler."""
    rng = random.Random(stable_hash(f"plan:{params.get('seed', 0)}:{params.get('variant', 0)}")[:12])
    hops = int(params.get("hops", 1))
    codes = rng.sample(range(10_000, 99_999), 6)
    if hops == 1:
        thing, place = rng.choice(_THINGS), rng.choice(_PLACES)
        decoys_places = [p for p in rng.sample(_PLACES, 4) if p != place][:3]
        fact = f"{thing[0].upper() + thing[1:]} de {place} es {codes[0]}."
        decoys = [f"{thing[0].upper() + thing[1:]} de {p} es {c}." for p, c in zip(decoys_places, codes[1:])]
        return {"facts": [fact], "decoys": decoys, "question": f"¿Cuál es {thing} de {place}? Responde solo con el número.", "answer": str(codes[0]),
                "forbid": [str(c) for c in codes[1:1 + len(decoys)]]}
    project, other = rng.sample(_PROJECTS, 2)
    boss, other_boss, third = rng.sample(_PEOPLE, 3)
    fact1 = f"La persona responsable del proyecto {project} es {boss}."
    fact2 = f"{boss} guarda la llave de la taquilla número {codes[0]}."
    decoys = [f"La persona responsable del proyecto {other} es {other_boss}.", f"{other_boss} guarda la llave de la taquilla número {codes[1]}.",
              f"{third} guarda la llave de la taquilla número {codes[2]}."]
    return {"facts": [fact1, fact2], "decoys": decoys, "question": f"¿Qué número de taquilla guarda la llave de la persona responsable del proyecto {project}? Responde solo con el número.",
            "answer": str(codes[0]), "forbid": [str(codes[1]), str(codes[2])]}


def _sentence(rng: random.Random) -> str:
    link = rng.choice(_LINKS)
    subject, verb = rng.choice(_SUBJECTS), rng.choice(_VERBS)
    subject, verb = (subject[1:], verb) if subject.startswith("*") else (subject, verb[:-1])  # every verb ends in -aban / -ían
    body = f"{subject} {verb} {rng.choice(_OBJECTS)} {rng.choice(_WHEN)}"
    tail = rng.choice(_TAILS)
    text = f"{link} {body}" if link else body
    text = text[0].upper() + text[1:] if not link else text
    return (f"{text} {tail}." if tail else f"{text}.").replace("  ", " ")


def _paragraph(rng: random.Random) -> str:
    return " ".join(_sentence(rng) for _ in range(rng.randint(3, 6)))


def filler(rng: random.Random, chars: int) -> list[str]:
    """Paragraphs of neutral prose until ``chars`` characters are reached."""
    out, total = [], 0
    while total < chars:
        p = _paragraph(rng)
        out.append(p)
        total += len(p) + 2
    return out


def build(params: dict[str, Any]) -> dict[str, Any]:
    """The full prompt: ``{"text", "answer", "chars", "tokens"}``. ``params``: ``tokens``, ``depth`` (0..1), ``hops`` (1 or 2), ``seed``, ``variant``."""
    p = plan(params)
    rng = random.Random(stable_hash(f"fill:{params.get('seed', 0)}:{params.get('variant', 0)}")[:12])
    target_chars = int((int(params["tokens"]) - OVERHEAD_TOKENS) * TOKEN_CHARS)
    paragraphs = filler(rng, target_chars)
    n = len(paragraphs)
    depth = float(params.get("depth", 0.5))
    slots: dict[int, list[str]] = {}

    def place(position: float, sentence: str) -> None:
        index = min(n - 1, max(0, int(position * n)))
        slots.setdefault(index, []).append(sentence)

    if len(p["facts"]) == 1:
        place(depth, p["facts"][0])
        decoy_positions = [0.05, 0.35, 0.65, 0.95]
        decoys_rng = random.Random(rng.random())
        decoy_positions = [x for x in decoy_positions if abs(x - depth) > 0.12]
        for pos, sentence in zip(decoys_rng.sample(decoy_positions, min(len(p["decoys"]), len(decoy_positions))), p["decoys"]):
            place(pos, sentence)
    else:
        place(max(0.02, depth - 0.3), p["facts"][0])
        place(min(0.98, depth + 0.3), p["facts"][1])
        for pos, sentence in zip([0.2, 0.5, 0.8], p["decoys"]):
            if abs(pos - max(0.02, depth - 0.3)) > 0.05 and abs(pos - min(0.98, depth + 0.3)) > 0.05:
                place(pos, sentence)
            else:
                place(min(0.97, pos + 0.07), sentence)
    out = []
    for i, para in enumerate(paragraphs):
        if i in slots:
            sentences = [x.rstrip(".") for x in para.split(". ")]
            cut = len(sentences) // 2
            sentences[cut:cut] = [x.rstrip(".") for x in slots[i]]
            para = ". ".join(sentences) + "."
        out.append(para)
    document = "\n\n".join(out)
    text = ("Lee el documento siguiente y responde a la pregunta usando solo la información que aparece en él.\n\n=== DOCUMENTO ===\n" + document +
            "\n=== FIN DEL DOCUMENTO ===\n\nPregunta: " + p["question"])
    return {"text": text, "answer": p["answer"], "chars": len(text), "tokens": approx_tokens(len(text))}


def suite_def(seed: int = 20251) -> dict[str, Any]:
    """The cases of the generated suite ``contexto-largo``: needle at 3 depths in 5 lengths (15) plus 5 two-fact cases."""
    cases: list[dict[str, Any]] = []
    for ti, tokens in enumerate(LENGTHS):
        for di, depth in enumerate(DEPTHS):
            params = {"kind": "haystack", "tokens": tokens, "depth": depth, "hops": 1, "seed": seed + ti * 10 + di, "variant": ti * 3 + di}
            cases.append(_case(params, f"Aguja al {int(depth * 100)} % en {tokens // 1000}k tokens"))
    for ti, tokens in enumerate(LENGTHS):
        params = {"kind": "haystack", "tokens": tokens, "depth": 0.5, "hops": 2, "seed": seed + 500 + ti, "variant": 100 + ti}
        cases.append(_case(params, f"Dos datos enlazados en {tokens // 1000}k tokens"))
    return {"id": "contexto-largo", "name": "Contexto largo", "category": "long_context", "version": 1, "max_tokens": 200, "position": 11,
            "description": "Una aguja (o dos datos enlazados) escondida a distinta profundidad en textos generados de 4k a 64k tokens. Los casos que no caben en el contexto del modelo se omiten.",
            "cases": cases}


def _case(params: dict[str, Any], title: str) -> dict[str, Any]:
    p = plan(params)
    tokens = params["tokens"]
    return {"title": title, "prompt": {"system": "Responde con el dato pedido y nada más.", "generate": params}, "checker": {"type": "needle", "needles": [p["answer"]], "forbid": p["forbid"]},
            "reference": p["answer"], "tags": ["contexto", f"{tokens // 1000}k", "multi-salto" if params["hops"] == 2 else "aguja"],
            "min_context": tokens + 200, "weight": 1.0 + 0.25 * LENGTHS.index(tokens)}
