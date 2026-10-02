"""Importing cases (JSONL and CSV, shorthand checkers, aliases, errors) and the small pure helpers."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from galton_hoard import importer
from galton_hoard.board import guess_category
from galton_hoard.errors import GaltonError
from galton_hoard.util import (clamp_text, fold, human_bytes, median, new_id, percentile, round_or_none, slugify, squash,
                               stable_hash)


# ---- shorthand -------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("exact:Madrid", {"type": "exact", "expected": "Madrid", "ignore_accents": True}),
    ("exacto:Madrid|madrid", {"type": "exact", "expected": ["Madrid", "madrid"], "ignore_accents": True}),
    ("contains:uno|dos", {"type": "contains", "all": ["uno", "dos"], "ignore_accents": True}),
    ("any:uno|dos", {"type": "contains", "any": ["uno", "dos"], "ignore_accents": True}),
    ("number:42", {"type": "number", "expected": 42.0}),
    ("numero:3,5", {"type": "number", "expected": 3.5}),
    ("choice:b", {"type": "choice", "answer": "B"}),
    ("regex:^\\d+$", {"type": "regex", "pattern": "^\\d+$"}),
    ("math:x^2", {"type": "math_equiv", "expected": "x^2"}),
    ("judge:Es claro.", {"type": "judge", "rubric": "Es claro."}),
    ("none", {"type": "none"}),
    ("", {"type": "none"}),
])
def test_shorthand_forms(text, expected):
    assert importer.shorthand(text) == expected


@pytest.mark.parametrize("text", ["contains:", "number:abc", "regex:(", "bogus:x"])
def test_shorthand_errors_are_honest(text):
    with pytest.raises(GaltonError) as exc:
        importer.shorthand(text)
    assert exc.value.code == "invalid" and exc.value.message


def test_derive_checker_picks_number_or_contains():
    assert importer.derive_checker("42") == {"type": "number", "expected": 42.0}
    assert importer.derive_checker("-3,5") == {"type": "number", "expected": -3.5}
    assert importer.derive_checker("Madrid")["type"] == "contains"
    assert importer.derive_checker("   ") == {"type": "none"}


# ---- rows ------------------------------------------------------------------------------------------------------------------------------------

def test_a_row_needs_a_prompt():
    with pytest.raises(GaltonError) as exc:
        importer.row_to_case({"answer": "x"})
    assert "prompt" in exc.value.message


def test_spanish_and_english_column_names_are_accepted():
    es = importer.row_to_case({"Título": "Capital", "Pregunta": "Capital de Francia?", "Respuesta": "París", "Peso": "2", "Etiquetas": "geo; facil"})
    en = importer.row_to_case({"title": "Capital", "question": "Capital de Francia?", "answer": "París", "weight": 2, "tags": ["geo", "facil"]})
    assert es["title"] == en["title"] == "Capital" and es["weight"] == en["weight"] == 2.0
    assert es["tags"] == en["tags"] == ["geo", "facil"]
    assert es["checker"]["type"] == "contains" and es["reference"] == "París"


def test_system_prompt_moves_into_the_prompt_object():
    case = importer.row_to_case({"prompt": "Hola", "system": "Eres breve.", "expected": "hola"})
    assert case["prompt"]["system"] == "Eres breve." and case["prompt"]["text"] == "Hola"


def test_a_checker_can_be_json_text_a_dict_or_shorthand():
    from_text = importer.row_to_case({"prompt": "2+2", "checker": '{"type": "number", "expected": 4}'})
    from_dict = importer.row_to_case({"prompt": "2+2", "checker": {"type": "number", "expected": 4}})
    short = importer.row_to_case({"prompt": "2+2", "checker": "number:4"})
    assert from_text["checker"] == from_dict["checker"] == short["checker"]


def test_a_row_without_expected_or_checker_is_kept_unscored():
    case = importer.row_to_case({"prompt": "Cuéntame algo."})
    assert case["checker"]["type"] == "none"


def test_default_checker_applies_when_the_row_has_none():
    case = importer.row_to_case({"prompt": "Cuéntame algo."}, default_checker={"type": "judge", "rubric": "Claro."})
    assert case["checker"]["type"] == "judge"


def test_bad_weight_and_bad_json_checker_are_errors():
    with pytest.raises(GaltonError):
        importer.row_to_case({"prompt": "x", "weight": "mucho"})
    with pytest.raises(GaltonError):
        importer.row_to_case({"prompt": "x", "checker": "{no json"})


# ---- whole files -----------------------------------------------------------------------------------------------------------------------------

def test_jsonl_reports_the_bad_lines_and_keeps_the_good_ones():
    text = "\n".join([json.dumps({"prompt": "uno", "expected": "1"}), "esto no es json", json.dumps({"nada": 1}), json.dumps(["lista"]),
                      json.dumps({"prompt": "dos", "expected": "2"})])
    out = importer.parse_cases(text)
    assert [c["reference"] for c in out["cases"]] == ["1", "2"]
    assert [e["row"] for e in out["errors"]] == [2, 3, 4]


def test_a_json_array_is_accepted():
    out = importer.parse_cases(json.dumps([{"prompt": "a", "expected": "1"}, {"prompt": "b", "expected": "2"}]))
    assert len(out["cases"]) == 2 and not out["errors"]


def test_csv_with_comma_semicolon_and_tab():
    for sep in (",", ";", "\t"):
        text = sep.join(["pregunta", "respuesta"]) + "\n" + sep.join(["Capital de España?", "Madrid"]) + "\n" + sep.join(["2+2", "4"]) + "\n"
        out = importer.parse_cases(text)
        assert len(out["cases"]) == 2, sep
        assert [c["checker"]["type"] for c in out["cases"]] == ["contains", "number"]


def test_csv_quoted_fields_with_commas_and_newlines():
    text = 'prompt,expected\n"Explica, con detalle\nen dos líneas",ok\n'
    out = importer.parse_cases(text)
    assert out["cases"][0]["prompt"]["text"].startswith("Explica, con detalle\nen dos")


def test_csv_row_errors_carry_the_line_number():
    out = importer.parse_cases("prompt,expected\nbien,1\n,2\n")
    assert len(out["cases"]) == 1 and out["errors"][0]["row"] == 3


def test_empty_input_and_unknown_format_raise():
    with pytest.raises(GaltonError):
        importer.parse_cases("   \n ")
    with pytest.raises(GaltonError):
        importer.parse_cases("prompt\nx", fmt="xml")


def test_a_byte_order_mark_is_ignored():
    out = importer.parse_cases("﻿" + json.dumps({"prompt": "a", "expected": "1"}))
    assert len(out["cases"]) == 1


def test_too_many_rows_are_refused():
    text = "\n".join(json.dumps({"prompt": f"p{i}"}) for i in range(importer.MAX_ROWS + 1))
    with pytest.raises(GaltonError) as exc:
        importer.parse_cases(text)
    assert exc.value.code == "too_large"


def test_imported_text_is_never_executed():
    out = importer.parse_cases(json.dumps({"prompt": "__import__('os').system('echo hacked')", "expected": "{{7*7}}"}))
    assert out["cases"][0]["prompt"]["text"].startswith("__import__")


# ---- util ------------------------------------------------------------------------------------------------------------------------------------

def test_fold_keeps_length_and_strips_accents():
    text = "ÁÉÍÓÚ ñandú ÜBER ß"
    assert len(fold(text)) == len(text)
    assert fold("Canción") == "cancion" and fold("PEÑA") == "pena"


def test_squash_and_clamp():
    assert squash("  a \n\t b  c ") == "a b c"
    assert clamp_text("uno dos tres cuatro", 10).endswith("…") and len(clamp_text("uno dos tres cuatro", 10)) <= 10
    assert clamp_text("corto", 10) == "corto"


def test_slugify():
    assert slugify("Qwen3 8B – Q4_K_M") == "qwen3-8b-q4-k-m"
    assert slugify("¿¿??") == "x"


def test_stable_hash_is_order_independent_for_keys_and_sensitive_to_values():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
    assert len(stable_hash("x", length=8)) == 8


def test_new_id_sorts_by_time_and_is_unique():
    a, b = new_id("run", 1_000.0), new_id("run", 2_000.0)
    assert a < b and a.startswith("run_")
    assert len({new_id("x", 5.0) for _ in range(50)}) == 50
    later = [new_id("r") for _ in range(200)]
    assert later == sorted(later) and len(set(later)) == 200 and all(len(i) == 28 and i == i.lower() for i in later)   # strictly ordered, lowercase ULIDs


def test_quiet_hours_wrap_over_midnight(svc):
    def at(hour, minute=30):
        return datetime(2026, 3, 4, hour, minute).timestamp()
    def quiet(start, end, ts):
        svc.settings.set_many({"watch.quiet_from": start, "watch.quiet_to": end})
        return svc.watch.in_quiet_hours(ts)
    assert quiet(22, 7, at(23)) and quiet(22, 7, at(3))
    assert not quiet(22, 7, at(12))
    assert quiet(9, 17, at(10)) and not quiet(9, 17, at(18))
    assert not quiet(5, 5, at(3))                                     # the same hour twice: no window
    assert quiet(7.5, 8, at(7, 45)) and not quiet(7.5, 8, at(7, 15))   # hours may have a fraction


def test_median_and_percentile():
    assert median([]) is None and median([3]) == 3 and median([1, 2, 3, 4]) == 2.5
    assert median([1, None, 3]) == 2
    assert percentile([], 50) is None and percentile([7], 90) == 7
    assert percentile([1, 2, 3, 4, 5], 50) == 3 and percentile([0, 10], 25) == 2.5


def test_round_or_none_and_human_bytes():
    assert round_or_none(None) is None and round_or_none(1.23456, 2) == 1.23
    assert human_bytes(0) == "" and human_bytes(512) == "512 B"
    assert human_bytes(1536) == "1.5 KB" and human_bytes(5 * 1024 ** 3) == "5.0 GB"


# ---- task guessing ---------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("task,category", [
    ("necesito escribir una función en Python y depurar un bug", "code"),
    ("resolver una ecuación e integral", "math"),
    ("traduce este texto al inglés", "translation"),
    ("resume este informe", "summary"),
    ("extraer campos de una factura en JSON", "extraction"),
    ("redacta un correo formal", "writing_es"),
    ("describe esta imagen y haz ocr de la captura", "vision"),
])
def test_guess_category(task, category):
    assert guess_category(task)["category"] == category


def test_guess_category_falls_back_to_general_and_says_so():
    out = guess_category("hablar de cosas sin más")
    assert out["category"] == "general" and out["guessed"] is False and out["note"]
