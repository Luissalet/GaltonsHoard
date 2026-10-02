"""Checkers: every type with right answers, wrong answers and the edge cases that bit during development."""

from __future__ import annotations

import pytest

from galton_hoard.checkers import CHECKERS, NON_DETERMINISTIC, checker_types, run_checker
from galton_hoard.checkers.types import CheckContext, ModelOutput


def check(spec, text="", ctx=None, **kw):
    return run_checker(spec, ModelOutput(text=text, **kw), ctx or CheckContext())


def passed(spec, text="", **kw):
    return check(spec, text, **kw)["passed"]


# ------------------------------------------------------------------ exact
@pytest.mark.parametrize("spec,text,ok", [
    ({"type": "exact", "expected": "Madrid"}, " madrid. ", True),
    ({"type": "exact", "expected": "Madrid"}, "La capital es Madrid", False),
    ({"type": "exact", "expected": ["Madrid", "Villa"]}, "villa", True),
    ({"type": "exact", "expected": "Pérez"}, "Perez", False),
    ({"type": "exact", "expected": "Pérez", "ignore_accents": True}, "Perez", True),
    ({"type": "exact", "expected": "42", "extract": "last_line"}, "Pienso un poco\n42", True),
    ({"type": "exact", "expected": "42", "extract": "marker"}, "Pienso\nRespuesta: 42", True),
    ({"type": "exact", "expected": "42", "extract": "first_line"}, "42\nporque sí", True),
])
def test_exact(spec, text, ok):
    assert passed(spec, text) is ok


def test_exact_ignores_think_blocks():
    assert passed({"type": "exact", "expected": "si"}, "<think>quizá no</think>si")


def test_exact_without_expected_is_a_bad_spec_not_a_pass():
    v = check({"type": "exact"}, "x")
    assert not v["passed"] and "invalid checker" in v["detail"]["error"]


# ------------------------------------------------------------------ contains
def test_contains_all_reports_partial_credit_and_what_is_missing():
    v = check({"type": "contains", "all": ["uno", "dos"]}, "solo uno")
    assert v["score"] == 0.5 and not v["passed"] and v["detail"]["missing"] == ["dos"]


def test_contains_is_case_insensitive_by_default():
    assert passed({"type": "contains", "all": ["uno", "dos"]}, "Uno y DOS")


def test_contains_any():
    assert passed({"type": "contains", "any": ["a", "b"]}, "b")
    assert not passed({"type": "contains", "any": ["a", "b"]}, "c")


def test_contains_none_forbids():
    v = check({"type": "contains", "none": ["malo"]}, "esto es malo")
    assert not v["passed"] and v["detail"]["forbidden_present"] == ["malo"]


def test_contains_accent_and_word_options():
    assert passed({"type": "contains", "all": ["canción"], "ignore_accents": True}, "una cancion")
    assert not passed({"type": "contains", "all": ["gato"], "words": True}, "gatos")
    assert passed({"type": "contains", "all": ["gato"], "words": True}, "un gato.")


def test_contains_case_sensitive():
    assert not passed({"type": "contains", "all": ["NASA"], "case_sensitive": True}, "la nasa")


# ------------------------------------------------------------------ regex
def test_regex_matches_and_flags():
    assert passed({"type": "regex", "pattern": r"^\d{3}$"}, "123")
    assert passed({"type": "regex", "pattern": "hola", "flags": "i"}, "HOLA")
    assert not passed({"type": "regex", "pattern": "hola"}, "HOLA")
    assert not passed({"type": "regex", "pattern": "hola", "flags": "i"}, "adiós")


def test_regex_forbid_lowers_the_score():
    v = check({"type": "regex", "pattern": "abc", "forbid": "x"}, "abc x")
    assert not v["passed"] and v["score"] == 0.5


def test_regex_invalid_pattern_is_reported():
    v = check({"type": "regex", "pattern": "(["}, "x")
    assert not v["passed"]


# ------------------------------------------------------------------ choice
@pytest.mark.parametrize("answer,text,ok", [
    ("B", "Respuesta: B", True),
    ("B", "**B)** porque es la única", True),
    ("B", "La respuesta correcta es la (B).", True),
    ("C", "Opción C", True),
    ("B", "Respuesta: A", False),
    ("B", "No lo sé", False),
])
def test_choice(answer, text, ok):
    assert passed({"type": "choice", "answer": answer}, text) is ok


# ------------------------------------------------------------------ number
@pytest.mark.parametrize("spec,text,ok", [
    ({"expected": 1234.5}, "Son 1.234,5 euros", True),
    ({"expected": 1234.5}, "It is 1,234.5", True),
    ({"expected": 1234}, "1.234", True),
    ({"expected": 3}, "1.234", False),
    ({"expected": 10, "tolerance": {"abs": 0.5}}, "10.4", True),
    ({"expected": 10, "tolerance": {"abs": 0.5}}, "11", False),
    ({"expected": 100, "tolerance": {"rel": 0.01}}, "101", True),
    ({"expected": 100, "tolerance": {"rel": 0.01}}, "102", False),
    ({"expected": 7}, "Primero 3 luego 5; Respuesta: 7", True),
    ({"expected": 7, "which": "last"}, "3 y luego 7", True),
    ({"expected": 7, "which": "first"}, "7 y luego 3", True),
    ({"expected": 0.5, "fractions": True}, "la mitad: 1/2", True),
    ({"expected": 0.5}, "la mitad: 1/2", False),
    ({"expected": -3}, "El resultado es -3", True),
])
def test_number(spec, text, ok):
    assert passed({"type": "number", **spec}, text) is ok


@pytest.mark.parametrize("spec,text,ok", [
    # a written-out calculation after the marker: the value after its final "=", not the first number of the line
    ({"expected": 1024}, "Respuesta: 2^10 = 1024", True),
    ({"expected": 2}, "Respuesta: 2^10 = 1024", False),
    ({"expected": 10}, "Respuesta: 2^10 = 1024", False),
    ({"expected": 1024}, "Answer: 2^10 = 1,024", True),
    ({"expected": 1024}, "Final answer: 2**10 = 1024.", True),
    ({"expected": 60}, "Respuesta final: 15 × 4 = 60 euros", True),
    ({"expected": 15}, "Respuesta final: 15 × 4 = 60 euros", False),
    ({"expected": 2469}, "Resultado: 1.234,5 × 2 = 2.469", True),
    ({"expected": 2469}, "Result: 1,234.5 * 2 = 2,469", True),
    ({"expected": 7}, "Answer: 3 + 4 = 7", True),
    ({"expected": 7}, "Solución: 12 - 5 = 7 ≈ 7", True),
    ({"expected": 11}, "Respuesta: 2x + 3 = 11", True),
    ({"expected": 6}, "Respuesta: 2 x 3 = 6", True),
    ({"expected": 6}, "**Respuesta:** (1 + 2) * 2 = 6", True),
    ({"expected": 1024}, "\\boxed{2^{10} = 1024}", True),
    ({"expected": 1024}, "Primero 2^10 = 1024 y luego\nRespuesta: 2^10 = 1024\nGracias, 5 de nada", True),
    ({"expected": 1024}, "Respuesta: 2^5 = 32; 2^10 = 1024", False),                 # the line is read up to its first foreign word or sign
    ({"expected": 32}, "Respuesta: 2^5 = 32; 2^10 = 1024", True),
    # what must keep working: no calculation, so the first number after the marker
    ({"expected": 42}, "Respuesta: 42 euros por 3 meses", True),
    ({"expected": 7}, "Respuesta: 7 (porque x = 3 + 4)", True),
    ({"expected": 3}, "Respuesta: 7 (porque x = 3 + 4)", False),
    ({"expected": 3}, "Respuesta: 3 + 4", True),
    ({"expected": 5}, "Respuesta: 5\n= 7", True),                                      # a line break ends the calculation
    # no marker: the last number of the answer, as before
    ({"expected": 1024}, "Calculo 2^10 = 1024", True),
])
def test_number_after_a_marker_takes_the_result_of_a_calculation_on_that_line(spec, text, ok):
    assert passed({"type": "number", **spec}, text) is ok


def test_the_reported_value_is_the_result_not_the_first_number():
    v = check({"type": "number", "expected": 1024}, "Respuesta: 2^10 = 1024")
    assert v["passed"] and v["detail"]["got"] == 1024.0 and v["detail"]["raw"] == "1024"


def test_number_without_any_number_reports_it():
    v = check({"type": "number", "expected": 4}, "cuatro")
    assert not v["passed"]


# ------------------------------------------------------------------ math_equiv
@pytest.mark.parametrize("expected,text,ok", [
    ("x**2+2*x+1", "(x+1)^2", True),
    ("2*x", "x*2", True),
    ("2*x", "3*x", False),
    ("1/2", "0.5", True),
    ("sqrt(2)", "2**(1/2)", True),
])
def test_math_equiv(expected, text, ok):
    assert passed({"type": "math_equiv", "expected": expected}, text) is ok


DERIVATIVE = "3*x**2*log(x) + x**2"


@pytest.mark.parametrize("text", [
    r"Respuesta: \(3x^2\ln(x)+x^2\)",                       # the form that failed on a real model: \ln glued to the 2 before it
    r"Respuesta: \(3x^{2}\ln x + x^{2}\)",
    r"Respuesta: $3x^2\ln(x)+x^2$.",
    r"**Respuesta:** $$3 x^{2} \ln x + x^{2}$$",
    r"Respuesta: \[ 3x^2\log(x) + x^2 \]",
    r"Respuesta: \(x^2(3\ln x+1)\)",
    r"Respuesta: \(x^2 + 3 \cdot x^2 \cdot \ln(x)\)",
    r"Respuesta: \(x^2\left(3\ln\left(x\right)+1\right)\)",
    r"Respuesta: \(\frac{3x^3\ln(x) + x^3}{x}\)",
    r"La derivada es f'(x) = 3x^2 \ln(x) + x^2",
    r"f'(x) = 3x^2 \ln(x) + x^2 = x^2(3\ln x + 1)",             # the expression after the last "="
    "Respuesta: 3x^2ln(x)+x^2",
    "Respuesta: 3x² ln(x) + x²",
    "Respuesta: 3*x**2*log(x) + x**2",
    "Calculo...\nRespuesta: f'(x) = 3x^2*ln(x) + x^2",
])
def test_math_equiv_reads_latex_and_plain_forms_of_the_same_expression(text):
    v = check({"type": "math_equiv", "expected": DERIVATIVE}, text)
    assert v["passed"], (text, v["detail"])


@pytest.mark.parametrize("text", [
    r"Respuesta: \(3x^2\ln(x)\)",
    r"Respuesta: \(3x^2\ln(x)-x^2\)",
    r"Respuesta: \(3x^2\log(x)+x\)",
    "Respuesta: no lo sé",
    "",
])
def test_math_equiv_still_rejects_other_expressions(text):
    assert not check({"type": "math_equiv", "expected": DERIVATIVE}, text)["passed"]


@pytest.mark.parametrize("expected,text", [
    ("sqrt(3)/2", r"Respuesta: \(\frac{\sqrt{3}}{2}\)"),
    ("(x**2+1)/2", r"Respuesta: \(\frac{1}{2}\left(x^{2}+1\right)\)"),
    ("pi*r**2", r"Respuesta: \pi r^2"),
    ("exp(-x)", r"Respuesta: \(e^{-x}\)"),
    ("exp(2*x)/2", r"Respuesta: $\frac{e^{2x}}{2}$"),
    ("exp(x**2)", r"Respuesta: \(e^{x^{2}}\)"),
    ("1/(x+1)", r"Respuesta: \(\frac{1}{x+1}\)"),
    ("(1/2)/(1/3)", r"Respuesta: \(\frac{\frac{1}{2}}{\frac{1}{3}}\)"),             # nested fractions
    ("3/2", r"Respuesta: x = \frac{3}{2}"),
    ("sqrt(3)", r"\boxed{\sqrt3}"),
    ("2*pi", r"Respuesta: \(2\pi\)"),
    ("6", r"Respuesta: \(2 \times 3\)"),
    ("log(2)", r"Respuesta: \ln 2"),
    ("sin(x)**2", r"Respuesta: \sin^2(x)" if False else r"Respuesta: \(\sin(x)^{2}\)"),
    ("x**2 + y**2 = 25", "Respuesta: x^2 + y^2 = 25"),                     # an equation is compared whole when the expected value is one
])
def test_math_equiv_latex_commands(expected, text):
    v = check({"type": "math_equiv", "expected": expected}, text)
    assert v["passed"], (text, v["detail"])


def test_math_equiv_set_mode_reads_latex_items():
    spec = {"type": "math_equiv", "mode": "set", "expected": ["2", "3"]}
    assert passed(spec, r"Respuesta: \(x = 2\) y \(x = 3\)")
    assert passed(spec, r"Respuesta: $x_1 = 2, x_2 = 3$")


# ------------------------------------------------------------------ number words
@pytest.mark.parametrize("expected,text", [
    (8, "El caracol llega tras 10 intentos fallidos.\nRespuesta: El octavo día"),          # the 10 earlier in the text is not the answer
    (8, "Respuesta: el octavo"),
    (8, "Respuesta: Octava"),
    (1, "Respuesta: primero"),
    (1, "Respuesta: El primer día"),
    (3, "Respuesta: tercero"),
    (10, "Respuesta: décimo"),
    (11, "Respuesta: undécimo"),
    (13, "Respuesta: decimotercero"),
    (13, "Respuesta: décimo tercero"),
    (20, "Respuesta: vigésimo"),
    (21, "Respuesta: vigésimo primero"),
    (0, "Respuesta: cero"),
    (4, "Respuesta: cuatro"),
    (15, "Respuesta: quince días"),
    (16, "Respuesta: dieciséis"),
    (20, "Respuesta: veinte"),
    (21, "Respuesta: veintiuno"),
    (21, "Respuesta: veintiún días"),
    (22, "Respuesta: veintidós"),
    (29, "Respuesta: veintinueve"),
    (30, "Respuesta: treinta"),
    (32, "Respuesta: treinta y dos"),
    (45, "Respuesta: cuarenta y cinco años"),
    (99, "Respuesta: noventa y nueve"),
    (100, "Respuesta: cien"),
    (120, "Respuesta: ciento veinte"),
    (200, "Respuesta: doscientos"),
    (531, "Respuesta: quinientos treinta y uno"),
    (1000, "Respuesta: mil"),
    (2000, "Respuesta: dos mil"),
    (2500, "Respuesta: dos mil quinientos"),
    (1, "**Respuesta:** uno"),
    (5, "**Respuesta:**\nCinco"),
    (7, "Respuesta final: siete."),
    (2, "Answer: two"),
    (3, "Answer: the third day"),
    (10, "Final answer: tenth"),
    (12, "Answer: twelve"),
    (13, "Answer: thirteen"),
    (20, "Answer: twenty"),
    (21, "Answer: twenty-one"),
    (21, "Answer: twenty one"),
    (21, "Answer: twenty-first"),
    (90, "Answer: ninety"),
    (100, "Answer: one hundred"),
    (105, "Answer: one hundred and five"),
    (2300, "Answer: two thousand three hundred"),
    (1, "Result: first"),
])
def test_number_reads_words_and_ordinals_on_the_answer_line(expected, text):
    v = check({"type": "number", "expected": expected}, text)
    assert v["passed"], (text, v["detail"])
    assert v["detail"]["from_words"] is True


@pytest.mark.parametrize("expected,text,ok", [
    (10, "Respuesta: El octavo día", False),                     # the number read is 8, not an earlier 10
    (8, "Respuesta: 8 (el octavo día)", True),                   # digits on the line win
    (10, "Respuesta: 10 (el octavo día)", True),
    (8, "Respuesta: El día 8", True),
    (4, "cuatro", False),                                        # no marker: no number words are guessed
    (2, "Respuesta: un día", False),                             # «un» alone is an article
    (1000, "Respuesta: un mil", True),
    (3, "Respuesta: dos tres", False),                           # not one number: the first word is read
    (2, "Respuesta: dos tres", True),
    (5, "Respuesta: no sé\nHabía 5 gatos", True),               # no number word on the line: the old rule (first number after the marker) applies
])
def test_number_words_only_when_the_answer_line_has_no_digit(expected, text, ok):
    assert passed({"type": "number", "expected": expected}, text) is ok


def test_number_words_with_a_tolerance_and_other_options_keep_working():
    assert passed({"type": "number", "expected": 8.3, "tolerance": {"abs": 0.5}}, "Respuesta: ocho")
    assert passed({"type": "number", "expected": 3, "which": "last"}, "dos y luego 3")


# ------------------------------------------------------------------ scope: answer
RESTATES = ("Sabemos que el 1 de enero de 2025 fue miércoles. Cada año normal avanza un día y los bisiestos dos; de 2025 a 2030 hay un bisiesto, así que "
            "el avance es 6 días: miércoles + 6 es martes. Ni lunes ni jueves.\nRespuesta: martes")


def test_scope_answer_judges_only_the_final_answer():
    base = {"type": "contains", "all": ["martes"], "none": ["lunes", "miércoles", "jueves"], "words": True, "ignore_accents": True}
    assert not passed(base, RESTATES), "the whole reply names the premises"
    assert passed({**base, "scope": "answer"}, RESTATES)
    assert not passed({**base, "scope": "answer"}, "Respuesta: miércoles")
    assert not passed({**base, "scope": "answer"}, "El 1 de enero de 2030 es martes, no lunes.\nRespuesta: lunes o martes")


def test_scope_answer_reads_the_text_after_the_last_marker_or_the_last_line():
    wrong = {"type": "contains", "none": ["Ana"], "scope": "answer"}
    assert passed(wrong, "Ana está en el extremo.\nRespuesta: Diego")
    assert passed(wrong, "Primero: Respuesta: Ana\nAl final\n**Respuesta final:** Diego")           # the last marker wins
    assert passed(wrong, "Ana se sienta primero y Beto luego\nDiego")                                # no marker: the last line
    assert not passed(wrong, "Diego se sienta ahí\nRespuesta: Ana")
    assert passed({"type": "contains", "all": ["Diego"], "scope": "answer"}, "Answer: Diego\nbecause of the clues")
    assert passed({"type": "contains", "all": ["Diego"], "scope": "answer"}, "Ana, Beto...\nANSWER:\nDiego")


def test_scope_answer_in_constraints_and_regex():
    spec = {"type": "constraints", "exclude": ["usted"], "include": ["tú"], "scope": "answer"}
    assert passed(spec, "No uses usted; usa el tuteo.\nRespuesta: ¿Tú puedes venir?")
    assert not passed({**spec, "scope": "all"}, "No uses usted; usa el tuteo.\nRespuesta: ¿Tú puedes venir?")
    assert passed({"type": "regex", "forbid": ["miércoles"], "scope": "answer"}, "fue miércoles.\nRespuesta: martes")
    assert not passed({"type": "regex", "forbid": ["miércoles"]}, "fue miércoles.\nRespuesta: martes")


def test_an_unknown_scope_is_a_bad_spec():
    for spec in ({"type": "contains", "all": ["a"], "scope": "lines"}, {"type": "constraints", "max_words": 3, "scope": "x"}, {"type": "regex", "pattern": "a", "scope": "y"}):
        v = check(spec, "a")
        assert not v["passed"] and "scope" in v["detail"]["error"]


# ------------------------------------------------------------------ per-field rules of the json checker
ROOM = {"type": "json", "expected": {"sala": "Magallanes"}}


@pytest.mark.parametrize("answer,rule,ok", [
    ("Magallanes", None, True),
    ("sala Magallanes", None, False),                           # the strict default
    ("sala Magallanes", "contains", True),
    ("la sala Magallanes, tercera planta", "contains", True),
    ("Sala Azul", "contains", False),
    ("sala Magallanes", "norm", True),
    ("Sala Magallanes", "norm", True),
    ("la sala de Magallanes", "norm", True),
    ("Magallanes", "norm", True),
    ("sala Azul", "norm", False),
    ("sala Magallanes, tercera planta", "norm", False),         # norm is an equality, not a containment
])
def test_json_field_rules_contains_and_norm(answer, rule, ok):
    spec = {**ROOM, **({"rules": {"sala": rule}} if rule else {})}
    assert passed(spec, '{"sala": "%s"}' % answer) is ok


@pytest.mark.parametrize("got,want", [
    ("c/ Mayor", "Mayor"), ("Calle Mayor", "mayor"), ("calle del Olmo", "Olmo"), ("c. Olmo", "Olmo"), ("Avda. de América", "América"),
    ("avenida de la Constitución", "Constitución"), ("provincia de Soria", "Soria"), ("Plaza Mayor", "Plaza Mayor"), ("Room 12", "12"), ("sala", "sala"),
])
def test_norm_drops_one_leading_generic_word(got, want):
    from galton_hoard.checkers.jsoncheck import values_equal
    assert values_equal(got, want, "norm")


def test_norm_takes_its_own_list_of_words():
    from galton_hoard.checkers.jsoncheck import values_equal
    assert not values_equal("pabellón 4", "4", "norm")
    assert values_equal("pabellón 4", "4", {"mode": "norm", "words": ["pabellón"]})


# ------------------------------------------------------------------ json
def test_json_inside_a_fence_and_extra_fields():
    assert passed({"type": "json", "expected": {"a": 1}}, '```json\n{"a": 1, "b": 2}\n```')


def test_json_wrong_value():
    assert not passed({"type": "json", "expected": {"a": 1}}, '{"a": 2}')


def test_json_schema_type_error_is_reported():
    v = check({"type": "json", "schema": {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}}, 'texto {"a": "x"}')
    assert not v["passed"] and "expected integer" in v["detail"]["schema_errors"][0]


def test_json_schema_array_rules():
    spec = {"type": "json", "schema": {"type": "array", "minItems": 2, "items": {"type": "string"}}}
    assert passed(spec, '["a","b"]')
    assert not passed(spec, '["a"]')
    assert not passed(spec, '["a", 3]')


def test_json_schema_enum_and_required():
    spec = {"type": "json", "schema": {"type": "object", "required": ["k"], "properties": {"k": {"enum": ["x", "y"]}}}}
    assert passed(spec, '{"k":"x"}')
    assert not passed(spec, '{"k":"z"}')
    assert not passed(spec, "{}")


def test_json_no_json_at_all():
    assert not passed({"type": "json", "expected": {"a": 1}}, "no hay nada")


# ------------------------------------------------------------------ tool_call
WEATHER = {"type": "tool_call", "expected": {"name": "get_weather", "arguments": {"city": "Madrid"}}}


def test_tool_call_from_structured_calls():
    assert passed(WEATHER, "", tool_calls=[{"name": "get_weather", "arguments": {"city": "Madrid"}}])


def test_tool_call_from_json_in_text():
    assert passed(WEATHER, '{"name":"get_weather","arguments":{"city":"Madrid"}}')


def test_tool_call_wrong_function_and_wrong_argument():
    assert not passed(WEATHER, "", tool_calls=[{"name": "other", "arguments": {}}])
    assert not passed(WEATHER, "", tool_calls=[{"name": "get_weather", "arguments": {"city": "Sevilla"}}])


def test_tool_call_none_expected():
    assert passed({"type": "tool_call", "none": True}, "Hola, no hace falta ninguna herramienta.")
    assert not passed({"type": "tool_call", "none": True}, "", tool_calls=[{"name": "x", "arguments": {}}])


# ------------------------------------------------------------------ constraints
@pytest.mark.parametrize("spec,text,ok", [
    ({"max_words": 5}, "uno dos tres", True),
    ({"min_words": 5}, "uno dos tres", False),
    ({"exact_sentences": 2}, "Hola. Adiós.", True),
    ({"language": "es"}, "El perro corre por el parque con la niña", True),
    ({"language": "es"}, "The dog runs through the park with the girl", False),
    ({"case": "upper"}, "TODO BIEN", True),
    ({"case": "upper"}, "Todo bien", False),
    ({"ends_with": "?"}, "¿Qué tal?", True),
    ({"ends_with": "?"}, "Qué tal.", False),
])
def test_constraints(spec, text, ok):
    assert passed({"type": "constraints", **spec}, text) is ok


def test_constraints_report_each_rule():
    v = check({"type": "constraints", "max_words": 2, "case": "upper"}, "hola que tal")
    assert len(v["detail"]["constraints"]) == 2 and not v["passed"]


# ------------------------------------------------------------------ needle and citations
def test_needle_ignores_digit_separators():
    assert passed({"type": "needle", "needle": "7391"}, "el código es 7.391")
    assert not passed({"type": "needle", "needle": "7391"}, "no sé")


def test_citations_right_wrong_and_extra():
    spec = {"type": "citations", "correct": ["p3"]}
    assert passed(spec, "Según [p3] es así")
    assert not passed(spec, "Según [p2] es así")
    v = check(spec, "[p3] y [p1]")
    assert not v["passed"] and 0 < v["score"] < 1 and v["detail"]["wrong"] == ["p1"]


# ------------------------------------------------------------------ python_tests
FN = {"type": "python_tests", "entry": "f", "tests": ["assert f(2) == 4", "assert f(3) == 9"]}


def test_python_tests_all_pass():
    v = check(FN, "```python\ndef f(x):\n    return x * x\n```")
    assert v["passed"] and v["detail"]["passed"] == 2


def test_python_tests_partial_credit():
    v = check(FN, "```python\ndef f(x):\n    return 4\n```")
    assert v["score"] == 0.5 and not v["passed"]


def test_python_tests_no_code():
    assert "no Python code" in check(FN, "sin código")["detail"]["reason"]


def test_python_tests_timeout():
    v = check({**FN, "timeout_s": 2}, "```python\ndef f(x):\n    while True: pass\n```")
    assert v["detail"].get("timed_out") and not v["passed"]


def test_python_tests_can_be_disabled():
    v = check(FN, "```python\ndef f(x):\n    return x * x\n```", CheckContext(allow_code=False))
    assert v.get("unavailable") and not v["passed"]


# ------------------------------------------------------------------ judge, family, none, combinators
def test_judge_without_a_judge_is_unavailable():
    assert check({"type": "judge", "rubric": "claro"}, "algo").get("unavailable")


def test_judge_scores_from_the_reply():
    ctx = CheckContext(judge=lambda req: {"score": 9, "reasons": "bien"})
    v = check({"type": "judge", "rubric": "claro"}, "algo", ctx)
    assert v["passed"] and v["score"] == 0.9


def test_family_without_a_hub_is_unavailable_not_failed():
    v = check({"type": "family", "app": "laplace", "tool": "calc"}, "2+2")
    assert v.get("unavailable")


def test_none_is_never_a_pass():
    v = check({"type": "none"}, "x")
    assert not v["passed"] and v["detail"]["unscored"]


def test_all_averages_and_requires_every_part():
    spec = {"type": "all", "checks": [{"type": "contains", "all": ["a"]}, {"type": "contains", "all": ["b"]}]}
    assert check(spec, "a")["score"] == 0.5 and not passed(spec, "a") and passed(spec, "ab")


def test_any_takes_the_best_part():
    spec = {"type": "any", "checks": [{"type": "contains", "all": ["a"]}, {"type": "contains", "all": ["b"]}]}
    assert passed(spec, "a") and not passed(spec, "z")


def test_empty_combinator_is_a_bad_spec():
    assert not passed({"type": "all", "checks": []}, "x")


def test_unknown_type_is_reported_never_a_pass():
    v = check({"type": "bogus"}, "x")
    assert not v["passed"] and "unknown checker type" in v["detail"]["error"]


def test_registry_is_consistent():
    assert set(NON_DETERMINISTIC) <= set(CHECKERS)
    assert set(checker_types()) == set(CHECKERS) | {"all", "any"}
    assert len(CHECKERS) == 15


def test_a_crashing_checker_does_not_raise(monkeypatch):
    monkeypatch.setitem(CHECKERS, "exact", lambda *_: 1 / 0)
    v = check({"type": "exact", "expected": "x"}, "x")
    assert not v["passed"]


# ------------------------------------------------------------------ realistic model output (review fixes)
def test_an_unfinished_think_block_is_reasoning_not_the_answer():
    from galton_hoard.checkers.textutil import split_reasoning
    visible, thought = split_reasoning("<think>pienso que 5, luego 7")
    assert visible == "" and "luego 7" in thought
    # a truncated reasoning that happens to end on the right number must not pass
    assert not passed({"type": "number", "expected": 7}, "<think>pienso que 5, luego 7")
    assert passed({"type": "number", "expected": 7}, "<think>pienso 5</think>La respuesta es 7")


@pytest.mark.parametrize("text,ok", [
    ("Respuesta: B. La opción A es incorrecta porque falta un dato.", True),
    ("Answer: (B)\n\nOption A is wrong because it ignores the cost.", True),
    ("La opción correcta es la B. La opción C no cumple.", True),
    ("Analicemos: la opción A falla y la opción C también. Por tanto, la respuesta es B.", True),
    ("Elijo la opción B.", True),
    ("Respuesta: C. La opción B parecía buena.", False),
])
def test_choice_prefers_the_answer_marker_over_options_mentioned_in_the_explanation(text, ok):
    assert passed({"type": "choice", "answer": "B"}, text) is ok


def test_python_blocks_skip_shell_and_json_fences_and_accept_info_strings():
    from galton_hoard.checkers.code import extract_code
    answer = "Instala:\n```bash\npip install nada\n```\nCódigo:\n```Python3\ndef f(x):\n    return x * 2\n```\n```json\n{\"a\": 1}\n```"
    assert extract_code(answer) == "def f(x):\n    return x * 2\n"
    assert extract_code('```py title="sol.py"\ndef g():\n    return 1\n```', "g") == "def g():\n    return 1\n"
    assert passed({"type": "python_tests", "entry": "f", "tests": ["assert f(2) == 4"]}, answer)


def test_python_tests_survive_a_temp_folder_that_cannot_be_removed(monkeypatch):
    """On Windows a killed child (or a process it started) can hold a file of the work folder for a moment: removing the folder
    fails with PermissionError, and that must not turn a measured answer into a crashed checker."""
    import os
    import shutil
    real_rmtree, real_unlink = shutil.rmtree, os.unlink
    left: list[str] = []

    def locked(path, *a, **kw):
        if "galton-code-" not in str(path):
            return real_rmtree(path, *a, **kw)
        left.append(str(path))
        error = PermissionError(13, "in use", str(path))
        if kw.get("onexc"):
            return kw["onexc"](os.unlink, str(path), error)
        if kw.get("onerror"):
            return kw["onerror"](os.unlink, str(path), (PermissionError, error, None))
        if not kw.get("ignore_errors"):
            raise error

    def unlink(path, *a, **kw):
        if "galton-code-" in str(path):
            raise PermissionError(13, "in use", str(path))
        return real_unlink(path, *a, **kw)

    monkeypatch.setattr(shutil, "rmtree", locked)
    monkeypatch.setattr(os, "unlink", unlink)
    try:
        result = check({"type": "python_tests", "tests": ["assert f(1) == 1"]}, "```python\ndef f(x):\n    return x\n```")
    finally:
        monkeypatch.undo()
        for path in set(left):
            real_rmtree(path, ignore_errors=True)
    assert result["passed"] and "error" not in result["detail"]
