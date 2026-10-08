"""The ``ifmt`` checker: instruction-following machine translation (the scoring of the IFMTBench benchmark).

Origin and changes. The rule checks (glossary, layout, structured data, code and tags), the two judge prompts, the parsing of the judge's
reply and the way constraint scores are combined are ported from ``IFMTBench/eval/rule_validators.py``, ``eval/llm_judge.py`` and
``eval/scoring.py`` of github.com/Tencent-Hunyuan/Hy-MT2 (commit ff1903ecaa724e10951a23c16817a2413c752b35), Copyright (C) 2026 Tencent,
licensed under the Apache License 2.0. See NOTICE. Changes made here: the code is typed and split into small functions; the judge is not an HTTP
client of its own but the judge model chosen in Galton's settings, called through ``ctx.judge`` (so grades are cached, marked when the judge is the
model under test and left pending when no judge is live); the answer is the visible text (no <think> block); an empty answer scores 0 instead of
being skipped; and a constraint that needs the judge and could not be judged makes the item *unjudged* (``unavailable``: left out of every
statistic) instead of being dropped from the combination, which would have raised the score.

Constraint classes (``classes`` in the spec):

* gates, score 0 or 1: ``glossary`` (rule; when the rule fails the judge decides), ``layout``, ``structured``, ``code``
* continuous, judge grade 0..5 mapped to 0..1: ``style``, ``background``

An item with several constraints scores ``product(gates) x mean(continuous)`` (1 when there are no continuous ones); with one constraint it scores
that constraint. The verdict detail lists every constraint in ``dimensions`` so that a board can show where a model fails.
"""

from __future__ import annotations

import csv
import io
import json
import re
from html.parser import HTMLParser
from typing import Any, Optional

from .textutil import split_reasoning
from .types import CheckContext, ModelOutput, bad_spec, verdict

GATES = ("glossary", "layout", "structured", "code")
CONTINUOUS = ("style", "background")
CLASSES = (*GATES, *CONTINUOUS)
#: classes that may call the judge (glossary only when its rule fails)
JUDGED = ("glossary", "style", "background")
#: "Marginal Pass" on the 0..5 scale of the judge's rubric
PASS_SCORE = 0.6

RUBRIC_GLOSSARY = "ifmt:glossary:v1"
RUBRIC_STYLE = "ifmt:style:v1"


# ================================================================================================= rule checks
def validate_glossary(response: str, term_dict: Any, ground_truth: str = "") -> dict[str, Any]:
    """Every source term has a required target in the response; with several candidates the one the reference uses must be the one used."""
    result: dict[str, Any] = {"valid": True, "errors": [], "matched": 0, "total": 0}
    if not term_dict:
        return {**result, "valid": False, "errors": ["term_dict is empty, cannot perform rule check"]}
    if not response:
        return {**result, "valid": False, "errors": ["model_response is empty"]}
    try:
        terms = json.loads(term_dict) if isinstance(term_dict, str) else term_dict
    except (ValueError, TypeError):
        return {**result, "valid": False, "errors": [f"term_dict parse failed: {str(term_dict)[:100]}"]}
    if not isinstance(terms, dict) or not terms:
        return {**result, "valid": False, "errors": ["term_dict is empty or has invalid format"]}
    for source, targets in terms.items():
        if not isinstance(targets, list):
            targets = [targets]
        result["total"] += 1
        hits = [t for t in targets if t and t in response]
        if not hits:
            result["valid"] = False
            result["errors"].append(f"Term not matched: {source} -> {targets}")
            continue
        if len(targets) > 1 and ground_truth:
            expected = [t for t in targets if t and t in ground_truth]
            if any(t in expected for t in hits):
                result["matched"] += 1
            else:
                result["valid"] = False
                result["errors"].append(f"Wrong term choice: {source} -> model used {hits}, but correct term in ground_truth is {expected}")
        else:
            result["matched"] += 1
    return result


def validate_layout(response: str, meta: dict[str, Any], origin_text: str) -> dict[str, Any]:
    """Splitting the source and the response on the primary delimiter gives as many chunks as the source has."""
    delimiter = meta.get("primary_delimiter", "")
    source_chunks = meta.get("source_chunks", [])
    result: dict[str, Any] = {"valid": False, "errors": []}
    if not delimiter:
        result["errors"].append("primary_delimiter is empty")
    elif not origin_text:
        result["errors"].append("origin_text is empty")
    elif not response:
        result["errors"].append("model_response is empty")
    else:
        origin, answer = len(origin_text.split(delimiter)), len(response.split(delimiter))
        if origin == answer == len(source_chunks):
            result["valid"] = True
        else:
            result["errors"].append(f"Chunk count mismatch: origin={origin}, response={answer}, source_chunks={len(source_chunks)}")
    return result


class _TagCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, str, list[str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        self.tags.append(("start", tag, sorted(k for k, _ in attrs)))

    def handle_endtag(self, tag: str) -> None:
        self.tags.append(("end", tag, []))


def _json_keys(origin: Any, output: Any, path: str = "$") -> list[str]:
    if type(origin) is not type(output):
        return [f"Type mismatch @ {path}"]
    errors: list[str] = []
    if isinstance(origin, dict):
        missing, extra = set(origin) - set(output), set(output) - set(origin)
        if missing:
            errors.append(f"Missing keys @ {path}: {missing}")
        if extra:
            errors.append(f"Extra keys @ {path}: {extra}")
        for key in set(origin) & set(output):
            errors += _json_keys(origin[key], output[key], f"{path}.{key}")
    elif isinstance(origin, list):
        if len(origin) != len(output):
            errors.append(f"Array length mismatch @ {path}")
        for i in range(min(len(origin), len(output))):
            errors += _json_keys(origin[i], output[i], f"{path}[{i}]")
    return errors


def _check_json(origin_text: str, output_text: str) -> list[str]:
    errors: list[str] = []
    parsed: list[Any] = []
    for label, text in (("Origin", origin_text), ("Output", output_text)):
        try:
            parsed.append(json.loads(text))
        except ValueError:
            errors.append(f"{label} JSON parse failed")
            parsed.append(None)
    if parsed[0] is not None and parsed[1] is not None:
        errors += _json_keys(parsed[0], parsed[1])
    return errors


def _check_html(origin_text: str, output_text: str) -> list[str]:
    collected = []
    for label, text in (("Origin", origin_text), ("Output", output_text)):
        parser = _TagCollector()
        try:
            parser.feed(text)
            parser.close()
        except Exception:  # noqa: BLE001 — the parser's own failures are a failed structure, as in the benchmark
            return [f"{label} HTML parse failed"]
        collected.append(parser.tags)
    origin, answer = collected
    if len(origin) != len(answer):
        return ["Tag count mismatch"]
    return [f"Tag #{i + 1} mismatch" for i, (a, b) in enumerate(zip(origin, answer)) if a[0] != b[0] or a[1] != b[1]]


def _check_csv(origin_text: str, output_text: str) -> list[str]:
    try:
        origin = list(csv.reader(io.StringIO(origin_text)))
    except csv.Error:
        return ["Origin CSV parse failed"]
    try:
        answer = list(csv.reader(io.StringIO(output_text)))
    except csv.Error:
        return ["Output CSV parse failed"]
    if len(origin) != len(answer):
        return ["Row count mismatch"]
    return [f"Column count mismatch at row {i + 1}" for i, (a, b) in enumerate(zip(origin, answer)) if len(a) != len(b)]


def _md_table(text: str) -> tuple[Optional[list[str]], Optional[list[list[str]]], str]:
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if len(lines) < 2:
        return None, None, "Insufficient table rows"

    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    if not re.match(r"^[\|\s\-:]+$", lines[1]):
        return None, None, "Row 2 is not a valid separator row"
    return cells(lines[0]), [cells(line) for line in lines[2:]], ""


def _check_markdown(origin_text: str, output_text: str) -> list[str]:
    errors: list[str] = []
    o_head, o_rows, o_err = _md_table(origin_text)
    a_head, a_rows, a_err = _md_table(output_text)
    if o_err:
        errors.append(f"origin: {o_err}")
    if a_err:
        errors.append(f"output: {a_err}")
    if o_head and a_head:
        if len(o_head) != len(a_head):
            errors.append("Header column count mismatch")
        if o_rows and a_rows and len(o_rows) != len(a_rows):
            errors.append("Data row count mismatch")
    return errors


STRUCTURED_FORMATS = {
    "JSON": _check_json, "json": _check_json,
    "HTML片段": _check_html, "HTML": _check_html, "html": _check_html,
    "CSV": _check_csv, "csv": _check_csv,
    "Markdown表格": _check_markdown, "Markdown": _check_markdown, "markdown": _check_markdown,
}


def validate_structured(origin_text: str, response: str, data_format: str) -> dict[str, Any]:
    """The translation keeps the structure of the source: JSON keys and types, HTML tags, CSV rows and columns, Markdown table shape."""
    if not origin_text or not response:
        return {"valid": False, "errors": ["origin_text or model_response is empty"]}
    check = STRUCTURED_FORMATS.get(data_format)
    if check is None:
        return {"valid": False, "errors": [f"Unknown data_format: {data_format}"]}
    errors = check(origin_text, response)
    return {"valid": not errors, "errors": errors}


def validate_code_assets(response: str, assets: list[str]) -> dict[str, Any]:
    """Every inline code span, tag or placeholder taken out of the source appears unchanged in the response."""
    result: dict[str, Any] = {"valid": True, "errors": [], "matched": 0, "total": len(assets)}
    if not assets:
        return result
    if not response:
        return {**result, "valid": False, "errors": ["model_response is empty"]}
    for asset in assets:
        if asset in response:
            result["matched"] += 1
        else:
            result["valid"] = False
            result["errors"].append(f"Not matched: {asset}")
    return result


# ================================================================================================= judge prompts and replies
GLOSSARY_PROMPT = '''
# ROLE
You are an expert Linguistic and Morphological Evaluator for a translation Reward Model. Your SOLE objective is to determine if the specified terminology from the instruction was correctly integrated into the target translation, accounting for complex morphological adaptations (declension, pluralization, tense, etc.).

# EVALUATION DATA
<instruction>
{user_instruction}
</instruction>

<ground_truth>
{ground_truth}
</ground_truth>

<model_output>
{target_translation}
</model_output>

# RUBRICS
### Glossary Compliance - [BINARY SCORING: 0 or 1]
Evaluate if the translation accurately incorporates the specific terminology provided in the instruction/background.
- [1] Perfect Adherence: Flawlessly integrated the required terms. Morphological adaptations (e.g., plurals, tense, part-of-speech, conjugations) are grammatically natural in the target language. It is acceptable if the term underwent necessary morphological changes compared to its base dictionary form.
- [0] Fatal Violation (Veto): Instant 0 if ANY of the following occur: unauthorized synonym substitution, fallback to generic dictionary translation, omission of the core concept, or severe grammatical corruption caused by forcing the term.

# OUTPUT FORMAT
Output ONLY a single integer: `1` or `0`.
Do NOT wrap it in JSON, Markdown, or any other formatting. Do NOT output any explanatory text.
'''

STYLE_PROMPT = '''
# ROLE
You are an advanced Reward Model designed for Reinforcement Learning (RL) of Large Language Models. Your primary function is to evaluate **Instruction Tracking and Constraint Satisfaction**.
Do NOT evaluate basic translation fluency. Your SOLE objective is to score whether the model executed the specific holistic [Constraints] (Style and Background).

# EVALUATION DATA
<instruction>
{user_instruction}
</instruction>

<ground_truth>
{ground_truth}
</ground_truth>

<model_output>
{target_translation}
</model_output>

# RUBRICS
Analyze the <instruction>. If a constraint is NOT requested, output `null`. If activated, evaluate against the rubrics.

### 1. Style & Register (Style) - [0-5 SCALE]
- [Activation Condition]: Activate if the instruction requests a specific tone, persona, register, or formatting style.
- [5] Perfect Alignment: Tone and register are exceptionally distinct and consistent throughout.
- [4] Strong Alignment: Generally fits the required style, but 1-2 lexical choices feel slightly generic.
- [3] Marginal Pass: Follows the basic directional constraint, but leans heavily on standard, flavorless translation.
- [2] Default/Generic: Ignored the stylistic constraint, reverting to a safe, bland machine translation tone.
- [1] Severe Deviation: Noticeable conflict with the requested style.
- [0] Rule Break: Wrong style AND included conversational filler/hallucinations, breaking the fourth wall.

### 2. Contextual Cohesion (Background) - [0-5 SCALE]
- [Activation Condition]: Activate if the instruction provides ANY preceding context, a background summary, or asks the translation to consider the "context" or "background".
- [5] Perfect Disambiguation: Masterfully leveraged the background summary to resolve potential ambiguities. Flawless logical cohesion.
- [4] Strong Utilization: Correctly used the summary to guide the translation, but feels slightly rigid when referencing the background.
- [3] Logically Consistent: Does not contradict the summary, but disambiguation is mediocre (literal translation).
- [2] Total Ignorance: Ignored the summary entirely, resulting in a disjointed literal translation.
- [1] Logical Contradiction: Directly contradicts the core logic or established facts in the background summary.
- [0] Severe Hallucination (Prompt Bleeding): Mistakenly translated the background summary itself as part of the target text.

# OUTPUT FORMAT
Output ONLY a valid JSON object. Do NOT wrap the JSON in Markdown code blocks (e.g., no ```json).
{{
  "scores": {{
    "style": [0, 1, 2, 3, 4, 5, or null],
    "background": [0, 1, 2, 3, 4, 5, or null]
  }}
}}'''

PARSERS = ("ifmt_glossary", "ifmt_style")


def glossary_messages(instruction: str, ground_truth: str, answer: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": GLOSSARY_PROMPT.format(user_instruction=instruction, ground_truth=ground_truth, target_translation=answer)}]


def style_messages(instruction: str, ground_truth: str, answer: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": STYLE_PROMPT.format(user_instruction=instruction, ground_truth=ground_truth, target_translation=answer)}]


def parse_glossary_reply(text: str) -> dict[str, Any]:
    """``{score: 0|1}`` from a judge reply that should be a lone ``1`` or ``0``; ``{error}`` when it holds neither."""
    visible = split_reasoning(text)[0].strip()
    if visible in ("0", "1"):
        return {"score": float(visible), "glossary": int(visible)}
    found = re.search(r"\b([01])\b", visible)
    if found:
        return {"score": float(found.group(1)), "glossary": int(found.group(1))}
    return {"error": "the judge's reply has no 0 or 1"}


def _json_object(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    return None


def _grade(value: Any) -> Optional[float]:
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def parse_style_reply(text: str) -> dict[str, Any]:
    """``{style, background}`` (each 0..5 or None when the judge found the constraint not requested) from the judge's JSON; ``{error}`` when unreadable.

    ``score`` is always None: which of the two grades counts depends on the constraint being scored, so the reply is cached whole."""
    parsed = _json_object(split_reasoning(text)[0])
    if parsed is None:
        return {"error": "the judge's reply is not JSON"}
    scores = parsed.get("scores", parsed) if isinstance(parsed, dict) else parsed
    if isinstance(scores, dict):
        return {"score": None, "style": _grade(scores.get("style")), "background": _grade(scores.get("background"))}
    value = _grade(scores)                       # a bare number: the original reads it as the grade of the constraint being scored
    return {"score": None, "style": value, "background": value, "bare": True}


REPLY_PARSERS = {"ifmt_glossary": parse_glossary_reply, "ifmt_style": parse_style_reply}


# ================================================================================================= combination
def needs_judge(spec: dict[str, Any]) -> bool:
    """Could scoring this item call the judge? (the glossary only when its rule fails, so the answer is yes for it)"""
    return any(c in JUDGED for c in spec.get("classes") or [])


def compose(dimensions: list[dict[str, Any]]) -> dict[str, float]:
    """``gate x mean(continuous)``: the product of the gate scores times the mean of the continuous ones (1 when there are none)."""
    gate = 1.0
    for d in dimensions:
        if d["kind"] == "gate":
            gate *= d["score"]
    continuous = [d["score"] for d in dimensions if d["kind"] == "continuous"]
    mean = sum(continuous) / len(continuous) if continuous else 1.0
    return {"gate": round(gate, 4), "continuous": round(mean, 4), "final": round(gate * mean, 4)}


def answer_text(output: ModelOutput) -> str:
    """The answer as the benchmark reads it: the text without any <think> block, with its whitespace when nothing was removed (layout counts delimiters)."""
    raw = output.text or ""
    visible = split_reasoning(raw)[0]
    return raw if visible == raw.strip() else visible


def _dim(cls: str, kind: str, score: Optional[float], **extra: Any) -> dict[str, Any]:
    out = {"class": cls, "kind": kind, "score": score, **extra}
    out["passed"] = None if score is None else (score >= 1.0 if kind == "gate" else score >= PASS_SCORE)
    return out


def _judge(ctx: CheckContext, parse: str, instruction: str, truth: str, answer: str) -> dict[str, Any]:
    """Ask the judge with the benchmark's own prompt (``messages``) and reader (``parse``); the rubric names the template for the grade cache."""
    glossary = parse == "ifmt_glossary"
    if ctx.judge is None:
        return {"error": "no judge model is configured (setting judge.contestant)"}
    messages = (glossary_messages if glossary else style_messages)(instruction, truth, answer)
    return ctx.judge({"prompt": instruction, "rubric": RUBRIC_GLOSSARY if glossary else RUBRIC_STYLE, "answer": answer, "reference": truth, "messages": messages, "parse": parse})


def _judged_dimension(cls: str, reply: dict[str, Any], **extra: Any) -> dict[str, Any]:
    if reply.get("error"):
        return _dim(cls, "gate" if cls == "glossary" else "continuous", None, unjudged=True, reason=str(reply["error"])[:200], **extra)
    flag = {"self_judged": True} if reply.get("self_judged") else {}
    if cls == "glossary":
        return _dim(cls, "gate", float(reply["score"]), method="judge", judge_score=reply.get("glossary"), judge=reply.get("judge", ""), **flag, **extra)
    grade = reply.get(cls)
    if grade is None:
        return _dim(cls, "continuous", None, unjudged=True, reason="the judge found this constraint not requested", **flag, **extra)
    return _dim(cls, "continuous", max(0.0, min(5.0, grade)) / 5.0, method="judge", judge_score=grade, judge=reply.get("judge", ""), **flag, **extra)


def check_ifmt(spec: dict[str, Any], output: ModelOutput, ctx: CheckContext) -> dict[str, Any]:
    classes = spec.get("classes")
    if not isinstance(classes, list) or not classes or not all(c in CLASSES for c in classes) or len(set(classes)) != len(classes):
        return bad_spec(f"ifmt needs `classes`, a list without repeats of {', '.join(CLASSES)}")
    pass_score = float(spec.get("pass_score", PASS_SCORE))
    origin = str(spec.get("origin_text") or "")
    meta = spec.get("meta") if isinstance(spec.get("meta"), dict) else {}
    truth = str(ctx.case.get("reference") or spec.get("ground_truth") or "")
    instruction = str(ctx.case.get("prompt_text") or "")
    answer = answer_text(output)
    multi = len(classes) > 1
    dimensions: list[dict[str, Any]] = []
    styled: Optional[dict[str, Any]] = None                # the style/background judge reply is shared by both constraints
    for cls in classes:
        kind = "continuous" if cls in CONTINUOUS else "gate"
        if not answer.strip():
            dimensions.append(_dim(cls, kind, 0.0, method="rule", errors=["model_response is empty"]))
        elif cls == "glossary":
            check = validate_glossary(answer, spec.get("term_dict"), truth)
            if check["valid"]:
                dimensions.append(_dim(cls, kind, 1.0, method="rule", matched=check["matched"], total=check["total"]))
            else:
                reply = _judge(ctx, "ifmt_glossary", instruction, truth, answer)
                dimensions.append(_judged_dimension(cls, reply, rule_errors=check["errors"][:5]))
        elif cls == "structured":
            fmt = spec.get("data_format") or meta.get("data_format", "")
            check = validate_structured(origin, answer, fmt)
            dimensions.append(_dim(cls, kind, 1.0 if check["valid"] else 0.0, method="rule", data_format=fmt, errors=check["errors"][:5]))
        elif cls == "layout":
            check = validate_layout(answer, meta, origin)
            dimensions.append(_dim(cls, kind, 1.0 if check["valid"] else 0.0, method="rule", errors=check["errors"][:5]))
        elif cls == "code":
            check = validate_code_assets(answer, list(meta.get("extracted_assets") or []))
            dimensions.append(_dim(cls, kind, 1.0 if check["valid"] else 0.0, method="rule", matched=check["matched"], total=check["total"], errors=check["errors"][:5]))
        else:
            if styled is None:
                styled = _judge(ctx, "ifmt_style", instruction, truth, answer)
            dimensions.append(_judged_dimension(cls, styled))
    detail: dict[str, Any] = {"multi": multi, "dimensions": dimensions}
    if any(d.get("self_judged") for d in dimensions):
        detail["self_judged"] = True
        detail["confidence"] = 0.5
    unjudged = [d["class"] for d in dimensions if d["score"] is None]
    if unjudged:
        detail["unjudged"] = unjudged
        detail["reason"] = "not judged: " + ", ".join(unjudged)
        return verdict(0.0, False, detail, unavailable=True)
    combined = compose(dimensions)
    detail.update(gate_score=combined["gate"], continuous_avg=combined["continuous"], final_score=combined["final"])
    return verdict(combined["final"], combined["gate"] >= 1.0 and combined["final"] >= pass_score, detail)
