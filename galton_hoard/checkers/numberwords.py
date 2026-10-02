"""Numbers written with words, Spanish and English: cardinals (``treinta y dos``, ``twenty-one``, ``dos mil quinientos``) and ordinals
(``octavo``, ``vigésimo primero``, ``third``). Used by the ``number`` checker when the answer line after «Respuesta:» has no digits at all.

The text is read as accent-free lowercase words. The first run of number words is evaluated; connectors (``y``, ``and``, hyphens) join the
words of one number, anything else ends the run. ``un`` and ``una`` count only before ``mil`` or ``cien`` (otherwise they are an article).
"""

from __future__ import annotations

import re
from typing import Optional

from ..util import fold

_UNITS_ES = {"cero": 0, "uno": 1, "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9}
_TEENS_ES = {"diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18,
             "diecinueve": 19, "veinte": 20}
_TENS_ES = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90}
_HUNDREDS_ES = {"cien": 100, "ciento": 100, "doscientos": 200, "doscientas": 200, "trescientos": 300, "trescientas": 300, "cuatrocientos": 400,
                "cuatrocientas": 400, "quinientos": 500, "quinientas": 500, "seiscientos": 600, "seiscientas": 600, "setecientos": 700,
                "setecientas": 700, "ochocientos": 800, "ochocientas": 800, "novecientos": 900, "novecientas": 900}
_VEINTI = {"veintiun": 21, "veintiuno": 21, "veintiuna": 21, "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25,
           "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29}

_UNITS_EN = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS_EN = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
             "nineteen": 19, "twenty": 20}
_TENS_EN = {"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}

_ORD_UNITS_ES = {"primero": 1, "primer": 1, "primera": 1, "segundo": 2, "segunda": 2, "tercero": 3, "tercer": 3, "tercera": 3, "cuarto": 4, "cuarta": 4,
                 "quinto": 5, "quinta": 5, "sexto": 6, "sexta": 6, "septimo": 7, "septima": 7, "octavo": 8, "octava": 8, "noveno": 9, "novena": 9}
_ORD_TEENS_ES = {"decimo": 10, "decima": 10, "undecimo": 11, "undecima": 11, "duodecimo": 12, "duodecima": 12}
_ORD_TENS_ES = {"vigesimo": 20, "vigesima": 20, "trigesimo": 30, "trigesima": 30, "cuadragesimo": 40, "cuadragesima": 40,
                "quincuagesimo": 50, "quincuagesima": 50}
_ORD_EN = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
           "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17,
           "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50, "sixtieth": 60,
           "seventieth": 70, "eightieth": 80, "ninetieth": 90, "hundredth": 100, "thousandth": 1000}

def _kind(value: int) -> str:
    if value < 10:
        return "unit"
    if value < 20:
        return "teen"
    if value < 100:
        return "ten"
    return "hundred" if value < 1000 else "thousand"


#: word -> (value, kind); kinds: unit (0-9), teen (10-19), ten (20-90), hundred (100-900), thousand
WORDS: dict[str, tuple[int, str]] = {}
for _table in (_UNITS_ES, _TEENS_ES, _TENS_ES, _HUNDREDS_ES, _VEINTI, _UNITS_EN, _TEENS_EN, _TENS_EN, _ORD_UNITS_ES, _ORD_TEENS_ES, _ORD_TENS_ES, _ORD_EN):
    for _word, _value in _table.items():
        WORDS[_word] = (_value, _kind(_value))
WORDS.update({"decimo": (10, "ten"), "decima": (10, "ten")})        # "décimo tercero" is 13: the ordinal ten takes a unit like the other tens
WORDS.update({"hundred": (100, "hundred"), "mil": (1000, "thousand"), "thousand": (1000, "thousand")})
# "decimotercero", "decimo tercero", "vigesimoprimero": a tens ordinal glued to a unit ordinal
_GLUED = re.compile(r"^(decimo|vigesimo|trigesimo|cuadragesimo|quincuagesimo)(primero|primer|segundo|tercero|tercer|cuarto|quinto|sexto|septimo|octavo|noveno)$")
_CONNECTORS = {"y", "and", "-", "–"}
_ARTICLES = {"un", "una"}
_TOKEN = re.compile(r"[^\W_]+|[-–]", re.UNICODE)


def _expand(token: str) -> list[str]:
    glued = _GLUED.match(token)
    if glued:
        return [glued.group(1), glued.group(2)]
    # veintiuno... is one word already; "treintaydos" and "treinta-y-dos" are rare enough to leave out
    return [token]


def _evaluate(run: list[str]) -> Optional[int]:
    """The value of a run of number words, or None when the words do not make a number (``dos tres``)."""
    total = current = 0
    last_kind = ""
    for word in run:
        if word in _CONNECTORS:
            continue
        value, kind = WORDS[word]
        if kind == "thousand":
            total += (current or 1) * value
            current = 0
        elif kind == "hundred":
            if word in ("hundred", "cien", "ciento") and last_kind == "unit":
                current *= value                                # "two hundred"
            else:
                current += value
        else:
            small = ("unit", "teen", "ten")
            # a number never has two units in a row, nor a teen or a ten after another small word ("dos tres", "treinta diez");
            # only "treinta y dos" / "twenty-one" (a ten, then a unit) is one number
            if last_kind in small and not (last_kind == "ten" and kind == "unit"):
                return None
            current += value
        last_kind = kind
    return total + current if last_kind else None


def first_number_words(text: str) -> Optional[tuple[int, str]]:
    """``(value, words)`` of the first number written with words in ``text``, or None."""
    tokens = [t for piece in _TOKEN.findall(fold(text or "")) for t in _expand(piece)]
    i = 0
    while i < len(tokens):
        word = tokens[i]
        usable = word in WORDS and (word not in _ARTICLES or (i + 1 < len(tokens) and tokens[i + 1] in ("mil", "cien", "ciento")))
        if not usable:
            i += 1
            continue
        run = [word]
        j = i + 1
        while j < len(tokens):
            token = tokens[j]
            if token in WORDS and token not in _ARTICLES:
                run.append(token)
                j += 1
            elif token in _CONNECTORS and j + 1 < len(tokens) and tokens[j + 1] in WORDS and tokens[j + 1] not in _ARTICLES:
                run.append(token)
                j += 1
            else:
                break
        # trim words from the end until the run is a number ("dos tres" -> "dos")
        while run:
            while run and run[-1] in _CONNECTORS:
                run.pop()
            value = _evaluate(run)
            if value is not None:
                return value, " ".join(run)
            run.pop()
        i = j if j > i else i + 1
    return None
