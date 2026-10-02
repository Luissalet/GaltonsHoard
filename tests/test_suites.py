"""Built-in suites: structure, uniqueness and, above all, that every reference answer passes its own checker."""

from __future__ import annotations

import json

import pytest

from galton_hoard import generators, suites
from galton_hoard.checkers import COMBINATORS, NON_DETERMINISTIC, run_checker
from galton_hoard.checkers.types import CheckContext, ModelOutput
from galton_hoard.generators import haystack, vision

DEFS = suites.load_definitions()
CASES = [(d["id"], c) for d in DEFS for c in d["cases"]]


def deterministic(spec) -> bool:
    if spec.get("type") in COMBINATORS:
        return all(deterministic(s) for s in spec["checks"])
    return spec.get("type") not in NON_DETERMINISTIC


def test_the_catalogue_has_the_planned_suites():
    names = {d["id"] for d in DEFS}
    assert len(DEFS) == 13 and "s_rapida" in names and "s_vision" in names and "s_contexto-largo" in names
    assert len(CASES) >= 200


@pytest.mark.parametrize("suite", DEFS, ids=lambda d: d["id"])
def test_every_suite_is_well_formed(suite):
    assert suite["category"] in suites.CATEGORIES and suite["cases"] and suite["name"] and suite["description"]
    titles = [c["title"] for c in suite["cases"]]
    assert len(set(titles)) == len(titles), "titles must be unique inside a suite"
    ids = [c["id"] for c in suite["cases"]]
    assert len(set(ids)) == len(ids)
    for case in suite["cases"]:
        assert suites.validate_case(case) == [], case["title"]


def test_case_ids_are_unique_across_suites():
    ids = [c["id"] for _, c in CASES]
    assert len(ids) == len(set(ids))


def test_ids_are_stable_between_loads():
    again = {c["id"] for d in suites.load_definitions() for c in d["cases"]}
    assert again == {c["id"] for _, c in CASES}


def test_no_banned_words_in_the_content():
    text = json.dumps(DEFS, ensure_ascii=False).lower()
    for word in ("chatgpt", "openai", "claude", "anthropic", "gemini", "copilot", "lm studio", "odysseus"):
        assert word not in text


def test_rapida_only_reuses_other_cases():
    rapida = next(d for d in DEFS if d["id"] == "s_rapida")
    others = {c["title"] for d in DEFS if d["id"] != "s_rapida" for c in d["cases"]}
    assert len(rapida["cases"]) == 12 and all(c["title"] in others for c in rapida["cases"])


STATIC = [(sid, c) for sid, c in CASES if not (c["prompt"].get("generate"))]


@pytest.mark.parametrize("sid,case", [(s, c) for s, c in STATIC if deterministic(c["checker"])], ids=lambda x: x if isinstance(x, str) else x["title"][:40])
def test_reference_answer_passes_its_own_checker(sid, case):
    assert case["reference"], f"{case['title']} has no reference"
    ctx = CheckContext(case=case, allow_code=True, code_timeout_s=10)
    tool_calls = []
    text = case["reference"]
    if case["checker"].get("type") == "tool_call":
        try:
            parsed = json.loads(text)
            tool_calls = parsed if isinstance(parsed, list) else [parsed]
            text = ""
        except ValueError:
            pass
    verdict = run_checker(case["checker"], ModelOutput(text=text, tool_calls=tool_calls), ctx)
    assert verdict["passed"] and verdict["score"] >= 0.999, (case["title"], verdict["detail"])


@pytest.mark.parametrize("sid,case", [(s, c) for s, c in STATIC if deterministic(c["checker"]) and c["checker"].get("type") in ("exact", "number", "choice")], ids=lambda x: x if isinstance(x, str) else x["title"][:40])
def test_a_clearly_wrong_answer_does_not_pass(sid, case):
    assert not run_checker(case["checker"], ModelOutput(text="No lo sé."), CheckContext())["passed"]


def test_generated_long_context_references_pass_and_wrong_answers_do_not():
    suite = next(d for d in DEFS if d["id"] == "s_contexto-largo")
    assert len(suite["cases"]) == 20
    for case in suite["cases"][:6]:
        built = generators.materialize(case["prompt"])
        assert case["reference"] in built["text"]
        assert run_checker(case["checker"], ModelOutput(text=f"El dato es {case['reference']}"))["passed"]
        assert not run_checker(case["checker"], ModelOutput(text="No aparece."))["passed"]


def test_vision_cases_render_png_and_reference_passes():
    suite = next(d for d in DEFS if d["id"] == "s_vision")
    assert len(suite["cases"]) == 20
    for case in suite["cases"]:
        png = vision.render(case["prompt"]["generate"])
        assert png.startswith(b"\x89PNG")
        verdict = run_checker(case["checker"], ModelOutput(text=case["reference"]), CheckContext())
        assert verdict["passed"], (case["title"], verdict["detail"])


def test_generators_are_deterministic():
    p = {"kind": "haystack", "tokens": 4000, "depth": 0.5, "hops": 1, "seed": 11, "variant": 1}
    assert haystack.build(p)["text"] == haystack.build(p)["text"]
    assert haystack.build(p)["text"] != haystack.build({**p, "seed": 12})["text"]
    v = {"kind": "vision", "task": "count", "color": "rojo", "shape": "círculo", "n_target": 3, "n_other": 5, "seed": 1}
    assert vision.render(v) == vision.render(v)
    assert vision.render(v) != vision.render({**v, "seed": 2})


def test_haystack_hides_the_needle_at_the_requested_depth():
    p = {"kind": "haystack", "tokens": 4000, "depth": 0.1, "hops": 1, "seed": 5, "variant": 0}
    plan = haystack.plan(p)
    text = haystack.build(p)["text"]
    position = text.find(plan["answer"]) / len(text)
    assert plan["answer"] in text and position < 0.35
    assert haystack.approx_tokens(len(text)) == pytest.approx(4000, rel=0.35)


def test_unknown_generator_is_an_error():
    with pytest.raises(ValueError):
        generators.materialize({"generate": {"kind": "nope"}})


def test_materialize_a_plain_case_builds_chat_messages():
    built = suites.materialize_case({"prompt": {"text": "hola", "system": "sé breve"}})
    assert built["messages"] == [{"role": "system", "content": "sé breve"}, {"role": "user", "content": "hola"}] and built["images"] == []


def test_materialize_a_vision_case_has_an_image():
    case = next(c for d in DEFS if d["id"] == "s_vision" for c in d["cases"])
    built = suites.materialize_case(case)
    assert len(built["images"]) == 1 and suites.needs_vision(case)


def test_sync_is_idempotent_and_repairs(tmp_path):
    from galton_hoard.db import Database
    from galton_hoard.store import Store
    store = Store(Database(tmp_path / "t.db"))
    first = suites.sync(store)
    assert first["created"] == 13 and first["updated"] == 0
    again = suites.sync(store)
    assert again == {"created": 0, "updated": 0, "unchanged": 13, "removed": 0}
    victim = store.cases("s_razonamiento")[0]
    store.delete_case(victim["id"])
    store.update_suite("s_razonamiento", content_hash="stale")
    repaired = suites.sync(store)
    assert repaired["updated"] == 1 and store.find_case(victim["id"]) is not None


def test_sync_removes_builtin_suites_that_no_longer_exist(tmp_path):
    from galton_hoard.db import Database
    from galton_hoard.store import Store
    store = Store(Database(tmp_path / "t.db"))
    suites.sync(store)
    store.create_suite(id="s_old", name="Old", description="", category="custom", builtin=True, max_tokens=100, position=99)
    assert suites.sync(store)["removed"] >= 1 and store.find_suite("s_old") is None


@pytest.mark.parametrize("case,fragment", [
    ({"prompt": "", "checker": {"type": "none"}}, "prompt"),
    ({"prompt": "x", "checker": {"type": "bogus"}}, "unknown checker"),
    ({"prompt": "x", "checker": "contains:a"}, "must be an object"),
    ({"prompt": "x", "checker": {"type": "all", "checks": []}}, "non-empty"),
    ({"prompt": "x", "checker": {"type": "none"}, "weight": 0}, "weight"),
    ({"prompt": "x", "checker": {"type": "none"}, "weight": "a"}, "weight"),
    ({"prompt": {"messages": [{"role": "robot", "content": "x"}]}, "checker": {"type": "none"}}, "messages"),
    ({"prompt": "x", "checker": {"type": "none"}, "tools": [{"name": "x"}]}, "tools"),
])
def test_validate_case_rejects(case, fragment):
    assert any(fragment in p for p in suites.validate_case(case))


def test_normalise_case_accepts_loose_input():
    case = suites.normalise_case({"prompt": "¿Cuánto es 2+2?\nResponde.", "checker": {"type": "number", "expected": 4}})
    assert case["title"] == "¿Cuánto es 2+2?" and case["prompt"] == {"text": "¿Cuánto es 2+2?\nResponde."} and case["weight"] == 1.0


# ---- measurement defects seen on a real model ---------------------------------------------------------------------------------------------------

def case_named(suite_id: str, title: str) -> dict:
    return next(c for d in DEFS if d["id"] == suite_id for c in d["cases"] if c["title"] == title)


def test_reasoning_cases_that_forbid_words_judge_only_the_final_answer():
    weekday = case_named("s_razonamiento", "Día de la semana de una fecha futura")
    seating = case_named("s_razonamiento", "Cinco amigos en una fila")
    for case in (weekday, seating):
        assert case["checker"]["scope"] == "answer"
    restated_weekday = "Partimos de que el 1 de enero de 2025 fue miércoles; no es lunes ni jueves. Cinco años después...\nRespuesta: martes"
    restated_seating = "Ana está en un extremo y Beto justo a su derecha; Carla y Elena no pueden ir en el medio.\nRespuesta: Diego"
    assert run_checker(weekday["checker"], ModelOutput(text=restated_weekday), CheckContext())["passed"]
    assert run_checker(seating["checker"], ModelOutput(text=restated_seating), CheckContext())["passed"]
    assert not run_checker(weekday["checker"], ModelOutput(text="Respuesta: miércoles"), CheckContext())["passed"]
    assert not run_checker(seating["checker"], ModelOutput(text="Respuesta: Ana"), CheckContext())["passed"]


def test_no_builtin_case_with_forbidden_words_leaves_a_final_answer_unscoped():
    """A case whose prompt asks for a «Respuesta:» line and that forbids words must read that line only."""
    for sid, case in CASES:
        spec = case["checker"]
        if spec.get("type") not in ("contains", "constraints", "regex") or "Respuesta" not in (case["prompt"].get("text") or ""):
            continue
        forbids = spec.get("none") or spec.get("exclude") or spec.get("forbid") or spec.get("forbid_regex")
        assert not forbids or spec.get("scope") == "answer", (sid, case["title"])


def test_extraction_accepts_a_label_before_a_place_name_and_not_a_wrong_place():
    case = case_named("s_extraccion", "Correo de reunión")
    assert case["checker"]["rules"]["sala"] == "norm"
    answer = json.loads(case["reference"])
    assert run_checker(case["checker"], ModelOutput(text=json.dumps({**answer, "sala": "sala Magallanes"})), CheckContext())["passed"]
    assert not run_checker(case["checker"], ModelOutput(text=json.dumps({**answer, "sala": "sala Azul"})), CheckContext())["passed"]


def test_changing_a_case_changes_the_suite_hash_and_the_edited_suites_got_a_new_version():
    by_id = {d["id"]: d for d in DEFS}
    assert by_id["s_razonamiento"]["version"] >= 2 and by_id["s_extraccion"]["version"] >= 2
    before = suites.load_definitions()
    raz = next(d for d in before if d["id"] == "s_razonamiento")
    edited = json.loads(json.dumps(raz))
    edited["cases"][2]["checker"]["scope"] = "all"
    from galton_hoard.util import stable_hash
    assert stable_hash({k: v for k, v in edited.items() if k not in ("cases", "content_hash")}, edited["cases"]) != raz["content_hash"]
