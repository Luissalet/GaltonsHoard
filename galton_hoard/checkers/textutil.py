"""Text helpers shared by the checkers: normalisation, counting, a stopword language detector, reasoning removal, JSON extraction."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Optional

from ..util import fold, squash

_EMPHASIS = re.compile(r"[*`]+")
_QUOTES = "\"'“”‘’«»`"


def normalise(text: str, *, ignore_accents: bool = False, strip_markup: bool = True) -> str:
    """Trim, casefold, collapse whitespace and drop one final full stop (and, by default, markdown emphasis and wrapping quotes)."""
    value = squash(text or "")
    if strip_markup:
        value = _EMPHASIS.sub("", value).strip()
        while len(value) >= 2 and value[0] in _QUOTES and value[-1] in _QUOTES:
            value = value[1:-1].strip()
    value = value.rstrip()
    if value.endswith(".") or value.endswith("。"):
        value = value[:-1].rstrip()
    value = value.casefold()
    return fold(value) if ignore_accents else value


def last_line(text: str) -> str:
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def first_line(text: str) -> str:
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    return lines[0] if lines else ""


_ANSWER_LINE = re.compile(r"(?:^|\n)[ \t>#*_`-]*(?:respuesta(?:\s+final)?|final\s+answer|answer|resultado(?:\s+final)?|soluci[oó]n)[ \t*_`]*[:：]", re.I)
SCOPES = ("", "all", "answer")


def answer_part(text: str) -> str:
    """The final answer of a reply: what follows the last «Respuesta:» / «Answer:» line marker (to the end), else the last non-empty line.
    A reply that explains its premises before answering is judged on its answer, not on every word it used on the way."""
    text = (text or "").strip()
    found = list(_ANSWER_LINE.finditer(text))
    if found:
        return text[found[-1].end():].strip(" \t*_`\n")
    return last_line(text)


# ---------------------------------------------------------------------------------------------------- reasoning
_THINK = re.compile(r"<(think|thinking|reasoning)>(.*?)</\1>", re.S | re.I)
_THINK_CLOSE_ONLY = re.compile(r"^(.*?)</(think|thinking|reasoning)>", re.S | re.I)


def split_reasoning(text: str) -> tuple[str, str]:
    """``(visible, reasoning)``: removes <think> blocks (also a lone closing tag, which chat templates that open the block themselves produce)."""
    text = text or ""
    thoughts: list[str] = []

    def take(match: re.Match) -> str:
        thoughts.append(match.group(2).strip())
        return ""

    visible = _THINK.sub(take, text)
    if not thoughts:
        lone = _THINK_CLOSE_ONLY.match(visible)
        if lone:
            thoughts.append(lone.group(1).strip())
            visible = visible[lone.end():]
    # an unfinished block (the answer was cut by max_tokens while still thinking): everything after it is reasoning,
    # also when it is the only block, so a half-finished thought is never graded as the answer
    open_tag = re.search(r"<(think|thinking|reasoning)>", visible, re.I)
    if open_tag:
        thoughts.append(visible[open_tag.end():].strip())
        visible = visible[: open_tag.start()]
    return visible.strip(), "\n\n".join(t for t in thoughts if t)


# ---------------------------------------------------------------------------------------------------- counting
_WORD = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)
_ABBREVIATION = re.compile(r"\b(?:sr|sra|srta|dr|dra|etc|art|núm|num|aprox|pág|págs|vs|ud|uds|ej|avda|fig|tel|cap|mr|mrs|ms|prof)\.", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•–—▪◦]|\d{1,3}[.)])\s+\S")


def words(text: str) -> list[str]:
    return _WORD.findall(text or "")


def word_count(text: str) -> int:
    return len(words(strip_list_markers(text)))


def strip_list_markers(text: str) -> str:
    return re.sub(r"(?m)^\s*(?:[-*•–—▪◦]|\d{1,3}[.)])\s+", "", re.sub(r"[*_`#]+", "", text or ""))


def split_sentences(text: str) -> list[str]:
    value = strip_list_markers(text)
    value = _ABBREVIATION.sub(lambda m: m.group(0).replace(".", "․"), value)
    value = re.sub(r"(?<=\d)\.(?=\d)", "․", value)
    value = re.sub(r"\b([A-Za-z])\.(?=[A-Za-z]\.)", lambda m: m.group(1) + "․", value)
    parts = re.split(r"(?<=[.!?…])[\"')\]»”]*\s+|\n+", value)
    return [p.strip() for p in parts if re.search(r"\w", p)]


def bullet_lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines() if _BULLET.match(ln)]


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]


# ---------------------------------------------------------------------------------------------------- language
_ES_STOP = frozenset("""de la que el en y los se del las un por con una su para es al lo como más pero sus le ya o este sí porque esta entre cuando muy sin
sobre también hasta hay donde quien desde todo nos durante todos uno les ni contra otros ese eso ante ellos esto antes algunos qué unos yo otro otras otra
él tanto esa estos mucho quienes nada muchos cual poco ella estar estas algunas algo nosotros mi mis tú te ti tu tus ellas vosotros os mío mía fue ser
son está están tiene tienen era han había puede pueden hacer sido así donde cada aquí allí nuestro nuestra vuestro usted ustedes""".split())
_EN_STOP = frozenset("""the of and to in is you that it was for are as with his they at be this have from or one had by but not what all were we when your
can said there use an each which she do how their if will up other about out many then them these some her would make like him into time has look two
more write see number could people my than first water been who its now find long down day did get made may part over new only little work know place
year live back give most very after thing our just name good sentence man think say great where help through much before line right too mean old any same
tell boy follow came want show also around form three small set put end does another well large must big even such because turn here why ask went men
read need land different home move try kind hand picture again change off play spell air away animal house point page letter mother answer found study
still learn should world high every near add food between own below country plant last school father keep tree never start city earth eye light thought
head under story saw left don't few while along might close something seem next hard open example begin life always those both paper together got group
often run important until children side feet car mile night walk white sea began grow took river four carry state once book hear stop without second
later miss idea enough eat face watch far really almost let above girl sometimes mountain cut young talk soon list song being leave family it's""".split())
_AMBIGUOUS = frozenset({"a", "no", "me", "he", "come", "so", "on", "an", "as", "can", "most", "pan", "e", "en", "son", "had", "set", "red", "sin"})
_ES_STOP = _ES_STOP - _AMBIGUOUS
_EN_STOP = _EN_STOP - _AMBIGUOUS
# English function words that never occur in Spanish text: one of them in a "Spanish only" answer is a leak.
ENGLISH_ONLY = frozenset("""the and is are was were with this that you your for from have has will would could should they their there what which
when where who how not but his her its our about into over more than then them these those just also because been being do does did""".split())
_SPANISH_CHARS = re.compile(r"[ñáéíóúü¿¡]", re.I)


def language_scores(text: str) -> dict[str, float]:
    tokens = [w.casefold() for w in words(text)]
    if not tokens:
        return {"es": 0.0, "en": 0.0, "words": 0}
    es = sum(1 for t in tokens if t in _ES_STOP) / len(tokens)
    en = sum(1 for t in tokens if t in _EN_STOP) / len(tokens)
    if _SPANISH_CHARS.search(text or ""):
        es += 0.05
    return {"es": round(es, 4), "en": round(en, 4), "words": len(tokens)}


def detect_language(text: str) -> str:
    """``es``, ``en`` or ``unknown``, from stopword ratios (and Spanish diacritics). Good enough to tell the two apart, nothing more."""
    s = language_scores(text)
    if s["words"] == 0:
        return "unknown"
    if s["es"] >= 0.12 and s["es"] > s["en"] * 1.5:
        return "es"
    if s["en"] >= 0.12 and s["en"] > s["es"] * 1.5:
        return "en"
    if s["words"] < 6:
        return "es" if s["es"] > s["en"] else ("en" if s["en"] > s["es"] else "unknown")
    return "unknown"


def english_leaks(text: str) -> list[str]:
    found: list[str] = []
    for token in words(text):
        low = token.casefold()
        if low in ENGLISH_ONLY and low not in found:
            found.append(low)
    return found


# ---------------------------------------------------------------------------------------------------- JSON extraction
_FENCE = re.compile(r"```(?:json|JSON|javascript|js)?\s*\n?(.*?)```", re.S)


def parse_loose(raw: str) -> tuple[Any, bool]:
    """Parse JSON; tolerate trailing commas and Python literals. Returns ``(value, repaired)``; raises ValueError."""
    try:
        return json.loads(raw), False
    except ValueError:
        pass
    cleaned = re.sub(r",(\s*[}\]])", r"\1", raw)
    try:
        return json.loads(cleaned), True
    except ValueError:
        pass
    try:
        value = ast.literal_eval(cleaned)
    except (ValueError, SyntaxError, MemoryError, RecursionError) as exc:
        raise ValueError("not JSON") from exc
    if not isinstance(value, (dict, list)):
        raise ValueError("not an object or array")
    return json.loads(json.dumps(value)), True


def extract_json(text: str, *, expect: Optional[str] = None) -> Optional[tuple[Any, bool]]:
    """The first JSON object (or array) in ``text``, also inside a fenced block. ``expect``: ``object`` or ``array`` to skip the other kind.
    Returns ``(value, repaired)`` or None."""
    text = text or ""
    sources = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for source in sources:
        for match in re.finditer(r"[{\[]", source):
            kind = "object" if match.group() == "{" else "array"
            if expect and kind != expect:
                continue
            try:
                value, end = decoder.raw_decode(source[match.start():])
                return value, False
            except ValueError:
                pass
            # a repaired parse needs the closing bracket: try the balanced slice
            piece = balanced_slice(source, match.start())
            if piece:
                try:
                    value, repaired = parse_loose(piece)
                    return value, repaired
                except ValueError:
                    continue
    return None


def balanced_slice(text: str, start: int) -> str:
    """From the bracket at ``start`` to its match, respecting strings; empty when unbalanced."""
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return ""
