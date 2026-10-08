"""The instruction-following translation benchmark (IFMTBench): rule checks, judge-backed constraints, combination, the data download and its checksum,
the stratified sample, and a whole run through the same path the UI and MCP use. Everything is invented and offline."""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest

import ifmt_data as data
from conftest import tool
from helpers import add_gguf, build_services, mock_client_factory, run_inline
from galton_hoard import ifmtbench
from galton_hoard.checkers import CheckContext, ModelOutput, run_checker
from galton_hoard.checkers import ifmt
from galton_hoard.errors import GaltonError
from galton_hoard.judging import has_judge


# ------------------------------------------------------------------ rule checks
class TestGlossary:
    TERMS = {"cat": ["gato"], "dog": ["perro"]}

    def test_every_target_term_present_passes(self):
        got = ifmt.validate_glossary("El gato y el perro.", json.dumps(self.TERMS))
        assert got["valid"] and got["matched"] == got["total"] == 2

    def test_a_missing_term_fails_and_says_which(self):
        got = ifmt.validate_glossary("El gato y el can.", json.dumps(self.TERMS))
        assert not got["valid"] and got["matched"] == 1 and "dog" in got["errors"][0]

    def test_a_dict_is_accepted_as_well_as_its_json(self):
        assert ifmt.validate_glossary("gato perro", self.TERMS)["valid"]

    def test_with_several_candidates_the_one_the_reference_uses_must_be_used(self):
        terms = {"bank": ["banco", "orilla"]}
        assert ifmt.validate_glossary("El banco abre.", terms, "El banco abre hoy.")["valid"]
        wrong = ifmt.validate_glossary("La orilla abre.", terms, "El banco abre hoy.")
        assert not wrong["valid"] and "Wrong term choice" in wrong["errors"][0]
        assert ifmt.validate_glossary("La orilla abre.", terms)["valid"], "without a reference any candidate counts"

    @pytest.mark.parametrize("terms", ["", None, "{}", "[1]", "{not json", {}])
    def test_no_usable_term_dictionary_fails_the_rule_so_the_judge_decides(self, terms):
        assert not ifmt.validate_glossary("gato", terms)["valid"]

    def test_an_empty_answer_fails(self):
        assert not ifmt.validate_glossary("", self.TERMS)["valid"]


class TestLayout:
    META = {"primary_delimiter": " | ", "source_chunks": ["a", "b", "c"]}

    def test_the_same_number_of_chunks_passes(self):
        assert ifmt.validate_layout("x | y | z", self.META, "a | b | c")["valid"]

    @pytest.mark.parametrize("answer", ["x | y", "x | y | z | w", "x y z"])
    def test_a_different_number_of_chunks_fails(self, answer):
        got = ifmt.validate_layout(answer, self.META, "a | b | c")
        assert not got["valid"] and "Chunk count mismatch" in got["errors"][0]

    def test_a_missing_delimiter_or_source_fails(self):
        assert not ifmt.validate_layout("x", {}, "a")["valid"]
        assert not ifmt.validate_layout("x", self.META, "")["valid"]
        assert not ifmt.validate_layout("", self.META, "a | b | c")["valid"]


class TestStructured:
    def check(self, fmt, origin, answer):
        return ifmt.validate_structured(origin, answer, fmt)

    def test_json_with_the_same_keys_and_types_passes_whatever_the_values(self):
        assert self.check("JSON", '{"a": "x", "b": [1, 2], "c": {"d": true}}', '{"a": "y", "b": [9, 9], "c": {"d": false}}')["valid"]

    @pytest.mark.parametrize("answer,why", [
        ('{"a": "x"}', "Missing keys"), ('{"a": "x", "b": 1, "z": 2}', "Extra keys"), ('{"a": 1, "b": 1}', "Type mismatch"),
        ('{"a": "x", "b": [1]}', "Array length"), ("not json", "Output JSON parse failed"), ('```json\n{"a": "x", "b": 1}\n```', "Output JSON parse failed")])
    def test_json_that_changes_the_structure_fails(self, answer, why):
        got = self.check("JSON", '{"a": "x", "b": 1}' if why != "Array length" else '{"a": "x", "b": [1, 2]}', answer)
        assert not got["valid"] and why in " ".join(got["errors"])

    def test_html_keeps_its_tags_in_order(self):
        origin = '<div class="a"><b>Hi</b> there</div>'
        assert self.check("HTML", origin, '<div class="a"><b>Hola</b> allí</div>')["valid"]
        assert self.check("HTML片段", origin, '<div><b>Hola</b> allí</div>')["valid"], "the benchmark compares tag names and order, not attribute values"
        assert "Tag count mismatch" in self.check("HTML", origin, '<div class="a"><b>Hola</b></div> <i></i>')["errors"]
        assert not self.check("html", origin, '<div class="a"><i>Hola</i> allí</div>')["valid"]
        assert not self.check("HTML", origin, "Hola allí")["valid"]

    def test_csv_keeps_rows_and_columns(self):
        origin = "a,b\n1,2\n3,4"
        assert self.check("CSV", origin, "x,y\n1,2\n3,4")["valid"]
        assert "Row count mismatch" in self.check("CSV", origin, "x,y\n1,2")["errors"]
        assert "Column count mismatch at row 2" in self.check("csv", origin, "x,y\n1,2,3\n3,4")["errors"]
        assert self.check("CSV", '"a, b",c\n1,2', '"x, y",z\n1,2')["valid"], "a quoted comma is not a column"

    def test_markdown_tables_keep_their_shape(self):
        origin = "| a | b |\n| - | - |\n| 1 | 2 |\n| 3 | 4 |"
        assert self.check("Markdown表格", origin, "| x | y |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |")["valid"]
        assert "Header column count mismatch" in self.check("Markdown", origin, "| x | y | z |\n| - | - | - |\n| 1 | 2 | 3 |\n| 3 | 4 | 5 |")["errors"]
        assert "Data row count mismatch" in self.check("markdown", origin, "| x | y |\n| - | - |\n| 1 | 2 |")["errors"]
        assert any("separator" in e for e in self.check("Markdown", origin, "| x | y |\n| 1 | 2 |\n| 3 | 4 |")["errors"])
        assert any("Insufficient" in e for e in self.check("Markdown", origin, "| x | y |")["errors"])

    def test_markdown_that_is_not_a_table_fails_even_for_the_reference(self):
        prose = "# Title\n\nSome text"
        assert not self.check("Markdown", prose, "# Título\n\nAlgo")["valid"], "upstream behaviour, kept on purpose and reported"

    def test_unknown_format_and_empty_text_fail(self):
        assert "Unknown data_format" in self.check("YAML", "a: 1", "a: 2")["errors"][0]
        assert not self.check("JSON", "", "{}")["valid"] and not self.check("JSON", "{}", "")["valid"]


class TestCodeAndTags:
    ASSETS = ["`npm test`", "<b>", "{name}", "https://example.org/x?y=1"]

    def test_every_asset_must_survive_unchanged(self):
        got = ifmt.validate_code_assets("Ejecuta `npm test`, <b>hola</b> {name} https://example.org/x?y=1", self.ASSETS)
        assert got["valid"] and got["matched"] == got["total"] == 4

    @pytest.mark.parametrize("answer,lost", [("Ejecuta `npm  test` <b> {name} https://example.org/x?y=1", "`npm test`"), ("`npm test` <B> {name} https://example.org/x?y=1", "<b>"),
                                            ("`npm test` <b> {nombre} https://example.org/x?y=1", "{name}"), ("`npm test` <b> {name} https://example.org/x", "https://example.org/x?y=1")])
    def test_an_altered_placeholder_tag_or_span_fails(self, answer, lost):
        got = ifmt.validate_code_assets(answer, self.ASSETS)
        assert not got["valid"] and got["errors"] == [f"Not matched: {lost}"]

    def test_nothing_to_preserve_passes_and_an_empty_answer_fails(self):
        assert ifmt.validate_code_assets("anything", [])["valid"]
        assert not ifmt.validate_code_assets("", self.ASSETS)["valid"]


# ------------------------------------------------------------------ the judge's replies
class TestJudgeReplies:
    @pytest.mark.parametrize("text,score", [("1", 1.0), ("0", 0.0), (" 1\n", 1.0), ("The answer is 0.", 0.0), ("<think>maybe 0</think>1", 1.0)])
    def test_glossary_reply(self, text, score):
        assert ifmt.parse_glossary_reply(text)["score"] == score

    @pytest.mark.parametrize("text", ["", "yes", "2", "maybe"])
    def test_a_glossary_reply_without_a_0_or_1_is_an_error(self, text):
        assert "error" in ifmt.parse_glossary_reply(text)

    def test_style_reply_reads_both_grades_and_nulls(self):
        got = ifmt.parse_style_reply('{"scores": {"style": 4, "background": null}}')
        assert got["style"] == 4.0 and got["background"] is None and got["score"] is None

    def test_style_reply_also_in_a_fence_or_with_chatter_or_lists(self):
        assert ifmt.parse_style_reply('```json\n{"scores": {"style": [3], "background": [5]}}\n```')["background"] == 5.0
        assert ifmt.parse_style_reply('Here you go: {"scores": {"style": 2, "background": 1}} done')["style"] == 2.0
        assert ifmt.parse_style_reply('{"style": 5, "background": null}')["style"] == 5.0, "a reply without the outer key is read too"
        assert ifmt.parse_style_reply("<think>hm</think>" + '{"scores": {"style": 1, "background": 0}}')["style"] == 1.0

    def test_style_reply_that_is_not_json_is_an_error(self):
        assert "error" in ifmt.parse_style_reply("I think it is good") and "error" in ifmt.parse_style_reply("")

    def test_the_shared_reader_dispatches_by_name(self):
        from galton_hoard.checkers.judge import parse_judge_reply
        assert parse_judge_reply("1", "ifmt_glossary")["glossary"] == 1
        assert parse_judge_reply('{"score": 8}')["score"] == 8.0, "the normal judge reply is read as before"


# ------------------------------------------------------------------ combining constraints
class TestCombination:
    def dims(self, **scores):
        return [{"class": c, "kind": "continuous" if c in ifmt.CONTINUOUS else "gate", "score": v} for c, v in scores.items()]

    def test_product_of_gates_times_mean_of_continuous(self):
        got = ifmt.compose(self.dims(glossary=1.0, structured=1.0, style=0.8, background=0.4))
        assert got == {"gate": 1.0, "continuous": 0.6, "final": 0.6}

    def test_one_failed_gate_zeroes_the_item(self):
        assert ifmt.compose(self.dims(glossary=1.0, structured=0.0, style=1.0))["final"] == 0.0

    def test_no_continuous_constraint_counts_as_one(self):
        assert ifmt.compose(self.dims(glossary=1.0, structured=1.0)) == {"gate": 1.0, "continuous": 1.0, "final": 1.0}

    def test_a_single_judged_constraint_scores_itself(self):
        assert ifmt.compose(self.dims(style=0.6))["final"] == 0.6


# ------------------------------------------------------------------ the ifmt checker
def judge_that(style=None, background=None, glossary=None, calls=None):
    def ask(request):
        if calls is not None:
            calls.append(request["parse"])
        assert request["messages"] and request["rubric"].startswith("ifmt:")
        if request["parse"] == "ifmt_glossary":
            return {"score": float(glossary), "glossary": glossary} if glossary is not None else {"error": "judge down"}
        return {"score": None, "style": style, "background": background}
    return ask


def run(spec, text, *, judge=None, reference="Referencia", prompt="Traduce."):
    return run_checker({"type": "ifmt", **spec}, ModelOutput(text=text), CheckContext(case={"reference": reference, "prompt_text": prompt}, judge=judge))


GLOSS = {"classes": ["glossary"], "term_dict": json.dumps({"cat": ["gato"]})}


class TestChecker:
    def test_rule_only_item_needs_no_judge(self):
        spec = {"classes": ["code"], "meta": {"extracted_assets": ["`x`"]}}
        assert not has_judge({"type": "ifmt", **spec})
        got = run(spec, "usa `x`")
        assert got["score"] == 1.0 and got["passed"] and not got.get("unavailable")
        assert run(spec, "usa x")["score"] == 0.0

    def test_judge_needed_for_style_background_and_glossary(self):
        for classes in (["style"], ["background"], ["glossary"], ["layout", "style"]):
            assert has_judge({"type": "ifmt", "classes": classes})
        assert not has_judge({"type": "ifmt", "classes": ["layout", "structured", "code"]})
        assert has_judge({"type": "all", "checks": [{"type": "ifmt", "classes": ["style"]}]})

    def test_glossary_rule_pass_never_calls_the_judge(self):
        calls: list[str] = []
        got = run(GLOSS, "El gato", judge=judge_that(glossary=0, calls=calls))
        assert got["score"] == 1.0 and calls == []
        assert got["detail"]["dimensions"][0]["method"] == "rule"

    def test_glossary_rule_failure_falls_back_to_the_judge_both_ways(self):
        calls: list[str] = []
        yes = run(GLOSS, "Los mininos", judge=judge_that(glossary=1, calls=calls))
        no = run(GLOSS, "Los felinos", judge=judge_that(glossary=0, calls=calls))
        assert yes["score"] == 1.0 and yes["passed"] and no["score"] == 0.0 and not no["passed"] and calls == ["ifmt_glossary"] * 2
        assert yes["detail"]["dimensions"][0]["method"] == "judge" and yes["detail"]["dimensions"][0]["rule_errors"]

    def test_style_grade_is_scaled_to_0_1_and_passes_from_a_marginal_pass(self):
        marginal = run({"classes": ["style"]}, "texto", judge=judge_that(style=3, background=None))
        strong = run({"classes": ["style"]}, "texto", judge=judge_that(style=5))
        weak = run({"classes": ["style"]}, "texto", judge=judge_that(style=2))
        assert (marginal["score"], marginal["passed"]) == (0.6, True) and strong["score"] == 1.0 and (weak["score"], weak["passed"]) == (0.4, False)

    def test_background_reads_its_own_grade(self):
        got = run({"classes": ["background"]}, "texto", judge=judge_that(style=5, background=2))
        assert got["score"] == 0.4

    def test_style_and_background_share_one_judge_call(self):
        calls: list[str] = []
        got = run({"classes": ["style", "background"]}, "texto", judge=judge_that(style=4, background=2, calls=calls))
        assert got["score"] == 0.6 and calls == ["ifmt_style"]

    def test_multi_constraint_item_is_gate_times_mean(self):
        spec = {"classes": ["glossary", "code", "style"], "term_dict": json.dumps({"cat": ["gato"]}), "meta": {"extracted_assets": ["`x`"]}}
        ok = run(spec, "gato `x`", judge=judge_that(style=4))
        broken = run(spec, "gato x", judge=judge_that(style=4))
        assert ok["score"] == 0.8 and ok["detail"]["multi"] and ok["detail"]["gate_score"] == 1.0 and ok["detail"]["continuous_avg"] == 0.8
        assert broken["score"] == 0.0 and not broken["passed"] and not broken.get("unavailable")

    def test_no_judge_leaves_the_item_unjudged_not_zero(self):
        got = run({"classes": ["style"]}, "texto", judge=None)
        assert got["unavailable"] and got["detail"]["unjudged"] == ["style"] and got["detail"]["dimensions"][0]["score"] is None
        got = run(GLOSS, "Los felinos", judge=None)
        assert got["unavailable"] and got["detail"]["unjudged"] == ["glossary"]

    def test_a_judge_that_fails_or_finds_the_constraint_not_requested_leaves_it_unjudged(self):
        assert run(GLOSS, "Los felinos", judge=judge_that(glossary=None))["unavailable"]
        got = run({"classes": ["background"]}, "texto", judge=judge_that(style=5, background=None))
        assert got["unavailable"] and "not requested" in got["detail"]["dimensions"][0]["reason"]

    def test_unjudged_part_of_a_multi_item_still_reports_the_rules(self):
        spec = {"classes": ["code", "style"], "meta": {"extracted_assets": ["`x`"]}}
        got = run(spec, "usa x", judge=None)
        by_class = {d["class"]: d for d in got["detail"]["dimensions"]}
        assert got["unavailable"] and by_class["code"]["score"] == 0.0 and by_class["style"]["score"] is None

    def test_empty_answer_scores_zero_without_asking_the_judge(self):
        calls: list[str] = []
        got = run({"classes": ["glossary", "style"], "term_dict": "{}"}, "   ", judge=judge_that(style=5, glossary=1, calls=calls))
        assert got["score"] == 0.0 and not got["passed"] and not got.get("unavailable") and calls == []

    def test_a_think_block_is_not_part_of_the_answer(self):
        spec = {"classes": ["code"], "meta": {"extracted_assets": ["`x`"]}}
        assert run(spec, "<think>use `x`</think>respuesta sin código")["score"] == 0.0
        assert run(spec, "<think>nada</think>usa `x`")["score"] == 1.0

    def test_whitespace_is_kept_for_layout_but_not_when_a_think_block_is_cut(self):
        spec = {"classes": ["layout"], "origin_text": "a ;; b ;; ", "meta": {"primary_delimiter": " ;; ", "source_chunks": ["a", "b", ""]}}
        assert run(spec, "x ;; y ;; ")["score"] == 1.0, "the trailing delimiter counts: the answer is not stripped"

    def test_self_judged_grades_are_marked(self):
        def me(request):
            return {"score": None, "style": 5, "background": None, "self_judged": True}
        got = run({"classes": ["style"]}, "texto", judge=me)
        assert got["detail"]["self_judged"] and got["detail"]["confidence"] == 0.5

    @pytest.mark.parametrize("spec", [{}, {"classes": []}, {"classes": ["poetry"]}, {"classes": ["style", "style"]}, {"classes": "style"}])
    def test_a_malformed_spec_is_a_bug_in_the_case_not_a_wrong_answer(self, spec):
        got = run(spec, "texto")
        assert not got["passed"] and "invalid checker" in got["detail"]["error"]

    def test_it_is_not_a_deterministic_checker(self):
        from galton_hoard.checkers import is_deterministic
        assert not is_deterministic({"type": "ifmt", "classes": ["code"]})


# ------------------------------------------------------------------ reading the data
class TestRows:
    def test_items_keep_what_scoring_needs(self, tmp_path):
        path = tmp_path / "single.jsonl"
        path.write_bytes(data.jsonl(data.single_rows(2)))
        items, errors = ifmtbench.read_items(path, "single")
        assert not errors and len(items) == 2 * 2 * 6
        glossary = next(i for i in items if i["families"] == ["glossary"])
        assert glossary["key"] in {f"{glossary['md5']}:中文", f"{glossary['md5']}:英语"} and glossary["instruction_lang"] in ("zh", "en")
        assert glossary["origin_language"] == "en" and glossary["target_language"] == "es" and glossary["term_dict"]

    def test_the_same_text_in_two_instruction_languages_is_two_items(self, tmp_path):
        path = tmp_path / "single.jsonl"
        path.write_bytes(data.jsonl(data.single_rows(1)))
        items, _ = ifmtbench.read_items(path, "single")
        assert len({i["md5"] for i in items}) == 6 and len({i["key"] for i in items}) == 12

    def test_unusable_rows_are_reported_not_guessed(self, tmp_path):
        good = data.single_rows(1)[0]
        lines = [json.dumps(good), "not json", json.dumps({**good, "class": ["机器翻译-不存在"]}), json.dumps({**good, "input": ""}), json.dumps([1]), "", json.dumps({k: v for k, v in good.items() if k != "md5"})]
        path = tmp_path / "x.jsonl"
        path.write_text("\n".join(lines), encoding="utf-8")
        items, errors = ifmtbench.read_items(path, "single")
        assert len(items) == 1 and [e["row"] for e in errors] == [2, 3, 4, 5, 7]
        assert "unknown constraint" in errors[1]["error"]

    def test_both_code_labels_and_every_label_map_to_a_constraint(self):
        assert {ifmtbench.FAMILY[data.TAGS_LABEL], ifmtbench.FAMILY[data.LABEL["code"]]} == {"code"}
        assert set(ifmtbench.FAMILY.values()) == set(ifmtbench.ORDER) == set(ifmt.CLASSES)


def all_items(tmp_path):
    out = []
    for key, rows in (("single", data.single_rows(8)), ("multi", data.multi_rows(8))):
        path = tmp_path / f"{key}.jsonl"
        path.write_bytes(data.jsonl(rows))
        out += ifmtbench.read_items(path, key)[0]
    return out


class TestSample:
    def test_quotas_per_single_type_and_evenly_over_the_multi_combinations(self, tmp_path):
        picked = ifmtbench.sample(all_items(tmp_path), per_type=3, multi=7, seed=1)
        counts: dict[str, int] = {}
        for item in picked:
            label = f"{item['group']}:{ifmtbench.stratum(item)}"
            counts[label] = counts.get(label, 0) + 1
        singles = {k: v for k, v in counts.items() if k.startswith("single")}
        multis = sorted(v for k, v in counts.items() if k.startswith("multi"))
        assert len(singles) == 6 and set(singles.values()) == {3}
        assert multis == [1, 1, 1, 2, 2], "7 over 5 combinations: the first two take the remainder"

    def test_the_same_seed_gives_the_same_sample_whatever_the_order_of_the_file(self, tmp_path):
        items = all_items(tmp_path)
        a = ifmtbench.sample(items, per_type=4, multi=10, seed=7)
        b = ifmtbench.sample(list(reversed(items)), per_type=4, multi=10, seed=7)
        assert [i["key"] for i in a] == [i["key"] for i in b]
        assert [i["key"] for i in a] != [i["key"] for i in ifmtbench.sample(items, per_type=4, multi=10, seed=8)]

    def test_different_source_texts_come_before_repeats_of_one_text(self, tmp_path):
        items = all_items(tmp_path)
        picked = [i for i in ifmtbench.sample(items, per_type=8, multi=0, seed=3) if i["families"] == ["glossary"]]
        assert len(picked) == 8 and len({i["md5"] for i in picked}) == 8, "8 distinct texts exist: no text is taken twice"
        again = [i for i in ifmtbench.sample(items, per_type=10, multi=0, seed=3) if i["families"] == ["glossary"]]
        assert len(again) == 10 and len({i["md5"] for i in again}) == 8

    def test_asking_for_more_than_exists_takes_everything(self, tmp_path):
        items = all_items(tmp_path)
        assert len([i for i in ifmtbench.sample(items, per_type=1000, multi=0, seed=1) if i["families"] == ["style"]]) == 16

    def test_any_prefix_is_balanced_across_types(self, tmp_path):
        picked = ifmtbench.sample(all_items(tmp_path), per_type=5, multi=0, seed=2)
        assert len({ifmtbench.stratum(i) for i in picked[:6]}) == 6


class TestCases:
    def items(self, tmp_path):
        return {ifmtbench.stratum(i) + ":" + i["instruction_lang"]: i for i in all_items(tmp_path)}

    def test_the_prompt_is_the_benchmarks_own_instruction_and_the_reference_is_kept(self, tmp_path):
        item = next(i for i in all_items(tmp_path) if i["families"] == ["glossary"])
        case = ifmtbench.build_case(item)
        assert case["prompt"] == {"text": item["input"]} and case["reference"] == item["output"] and case["weight"] == 1.0
        assert case["checker"]["type"] == "ifmt" and case["checker"]["classes"] == ["glossary"] and "ifmtbench" in case["tags"]
        assert case["max_tokens"] >= 512 and len(case["title"]) < 80

    def test_a_checker_carries_only_the_data_its_constraints_use(self, tmp_path):
        by = {tuple(i["families"]): ifmtbench.checker_spec(i) for i in all_items(tmp_path)}
        assert set(by[("layout",)]) == {"type", "classes", "item", "origin_text", "meta"} and set(by[("code",)]) == {"type", "classes", "item", "meta"}
        assert "term_dict" in by[("glossary",)] and "origin_text" not in by[("glossary",)]
        assert by[("structured",)]["data_format"] and "origin_text" in by[("glossary", "structured")] and "term_dict" in by[("glossary", "structured")]

    def test_titles_are_unique_for_every_item(self, tmp_path):
        titles = [ifmtbench.build_case(i)["title"] for i in all_items(tmp_path)]
        assert len(titles) == len(set(titles))

    def test_every_reference_passes_its_own_checker_except_the_markdown_that_is_not_a_table(self, tmp_path):
        failing = [i for i in all_items(tmp_path) if not ifmtbench.reference_passes(i)]
        assert failing and all(i["data_format"] == "Markdown" for i in failing)
        assert len(failing) < len(all_items(tmp_path)) / 5


# ------------------------------------------------------------------ the download and its checksum
@pytest.fixture
def source(monkeypatch):
    files = data.files()
    monkeypatch.setattr(ifmtbench, "PINS", data.pins(files))
    return files


def online(tmp_path, source, calls=None, **kw):
    return build_services(tmp_path, offline=False, client_factory=mock_client_factory(data.server(source, calls)), **kw)


class TestDownload:
    def test_downloads_verifies_and_keeps_the_file_in_the_data_folder(self, tmp_path, source):
        calls: list[str] = []
        svc = online(tmp_path, source, calls)
        got = ifmtbench.fetch_file(svc.config, "single", client_factory=svc.client_factory)
        assert got["downloaded"] and got["path"].read_bytes() == source["single"] and got["path"].parent == svc.config.external_dir / "ifmtbench"
        assert got["sha256"] == hashlib.sha256(source["single"]).hexdigest()
        assert calls and ifmtbench.COMMIT in calls[0] and calls[0].startswith("https://raw.githubusercontent.com/")

    def test_a_verified_copy_is_reused_without_the_network(self, tmp_path, source):
        calls: list[str] = []
        svc = online(tmp_path, source, calls)
        ifmtbench.fetch_file(svc.config, "single", client_factory=svc.client_factory)
        calls.clear()
        again = ifmtbench.fetch_file(svc.config, "single", client_factory=svc.client_factory)
        assert not again["downloaded"] and calls == []
        assert ifmtbench.fetch_file(svc.config, "single", client_factory=svc.client_factory, refresh=True)["downloaded"] and calls

    def test_a_download_that_does_not_match_the_pin_is_refused_and_not_kept(self, tmp_path, source):
        tampered = {**source, "single": source["single"].replace(b"gato", b"GATO")}
        svc = online(tmp_path, tampered)
        with pytest.raises(GaltonError) as raised:
            ifmtbench.fetch_file(svc.config, "single", client_factory=svc.client_factory)
        assert raised.value.key == "external_sha_mismatch" and "SHA-256" in raised.value.message
        assert not list((svc.config.external_dir / "ifmtbench").glob("*")), "nothing is stored"

    def test_a_cached_copy_that_was_altered_is_not_used(self, tmp_path, source):
        svc = online(tmp_path, source)
        path = ifmtbench.fetch_file(svc.config, "single", client_factory=svc.client_factory)["path"]
        path.write_bytes(path.read_bytes() + b'{"input": "injected"}\n')
        again = ifmtbench.fetch_file(svc.config, "single", client_factory=svc.client_factory)
        assert again["downloaded"] and path.read_bytes() == source["single"]
        offline = build_services(tmp_path / "other", offline=True)
        folder = offline.config.external_dir / "ifmtbench"
        folder.mkdir(parents=True)
        (folder / path.name).write_bytes(b"altered")
        with pytest.raises(GaltonError) as raised:
            ifmtbench.fetch_file(offline.config, "single", offline=True)
        assert raised.value.key == "external_offline" and not (folder / path.name).exists(), "an altered copy is removed, never read"

    def test_http_errors_and_a_dead_network_are_reported(self, tmp_path, source):
        svc = build_services(tmp_path, offline=False, client_factory=mock_client_factory(lambda request: httpx.Response(503)))
        with pytest.raises(GaltonError) as raised:
            ifmtbench.fetch_file(svc.config, "single", client_factory=svc.client_factory)
        assert raised.value.key == "external_download_failed" and "503" in raised.value.message

        def dead(request):
            raise httpx.ConnectError("no route")
        svc2 = build_services(tmp_path / "b", offline=False, client_factory=mock_client_factory(dead))
        with pytest.raises(GaltonError) as raised:
            ifmtbench.fetch_file(svc2.config, "single", client_factory=svc2.client_factory)
        assert raised.value.key == "external_download_failed" and "ConnectError" in raised.value.message

    def test_the_real_pins_are_the_sha256_of_the_published_files_and_name_a_commit(self):
        assert len(ifmtbench.COMMIT) == 40 and all(c in "0123456789abcdef" for c in ifmtbench.COMMIT)
        assert {k: p.sha256 for k, p in ifmtbench.PINS.items()} == {"single": "ffaab0947722711e5a7d0c47d6f7931868164ad9a068c253acc7ba909defc691",
                                                                      "multi": "c9fb393c49880e688e58515b215e1e586f9b72d143a404546f20de4a01a3e17f"}
        assert [p.rows for p in ifmtbench.PINS.values()] == [4506, 2838]


# ------------------------------------------------------------------ the tool
class TestImportTool:
    def test_imports_a_suite_with_credit_and_provenance(self, tmp_path, source):
        svc = online(tmp_path, source)
        out = tool(svc, "benchmark_import", per_type=3, multi=5, seed=11)
        suite = out["suite"]
        assert suite["name"] == "IFMTBench" and not suite["builtin"] and suite["category"] == "custom" and out["cases"] == 6 * 3 + 5
        assert "CC BY 4.0" in suite["description"] and "Tencent" in suite["description"] and ifmtbench.COMMIT[:7] in suite["description"]
        assert out["source"]["commit"] == ifmtbench.COMMIT and set(out["source"]["sha256"]) == {"single", "multi"} and out["source"]["data_licence"] == "CC-BY-4.0"
        assert "seed 11" in suite["notes"] and ifmtbench.MARKER in suite["notes"]
        assert out["by_type"]["single:glossary"] == 3 and sum(v for k, v in out["by_type"].items() if k.startswith("multi")) == 5
        assert out["needs_judge"] + out["rule_only"] == out["cases"] and out["rule_only"] == 9
        listed = tool(svc, "suites_list")
        assert any(s["id"] == suite["id"] and s["cases"] == out["cases"] for s in listed["suites"])
        assert tool(svc, "suite_get", suite=suite["id"], limit=3)["cases"][0]["checker_label"] == "ifmt"

    def test_warns_when_no_judge_is_chosen_and_stops_warning_when_one_is(self, tmp_path, source):
        svc = online(tmp_path, source)
        out = tool(svc, "benchmark_import", per_type=2, multi=0, name="Sin juez")
        assert out["notes"] and "judge.contestant" in out["notes"][0]
        judge = add_gguf(svc, "juez")
        svc.settings.set_many({"judge.contestant": judge["id"]})
        assert tool(svc, "benchmark_import", per_type=2, multi=0, name="Con juez")["notes"] == []

    def test_a_second_import_with_the_same_name_is_refused(self, tmp_path, source):
        svc = online(tmp_path, source)
        tool(svc, "benchmark_import", per_type=1, multi=0)
        with pytest.raises(GaltonError) as raised:
            tool(svc, "benchmark_import", per_type=1, multi=0)
        assert raised.value.key == "external_suite_exists"

    def test_the_same_arguments_give_the_same_cases_in_another_suite(self, tmp_path, source):
        svc = online(tmp_path, source)
        a = tool(svc, "benchmark_import", per_type=3, multi=5, seed=4, name="A")["suite"]["id"]
        b = tool(svc, "benchmark_import", per_type=3, multi=5, seed=4, name="B")["suite"]["id"]
        titles = lambda s: [c["title"] for c in svc.store.cases(s)]  # noqa: E731
        assert titles(a) == titles(b) and len(titles(a)) == 23

    def test_unsatisfiable_items_are_left_out_and_counted_unless_asked_for(self, tmp_path, source):
        svc = online(tmp_path, source)
        out = tool(svc, "benchmark_import", per_type=0, multi=1000, name="Limpio")
        assert out["left_out_unsatisfiable"] and all(k.startswith("multi:") and "structured" in k for k in out["left_out_unsatisfiable"])
        kept = tool(svc, "benchmark_import", per_type=0, multi=1000, name="Todo", keep_unsatisfiable=True)
        assert kept["left_out_unsatisfiable"] == {} and kept["cases"] == out["cases"] + sum(out["left_out_unsatisfiable"].values())

    def test_a_sample_of_nothing_a_bad_category_and_a_bad_download_are_errors(self, tmp_path, source):
        svc = online(tmp_path, source)
        with pytest.raises(GaltonError) as raised:
            tool(svc, "benchmark_import", per_type=0, multi=0)
        assert raised.value.key == "external_empty_sample"
        with pytest.raises(GaltonError):
            tool(svc, "benchmark_import", category="poetry")
        bad = online(tmp_path / "bad", {**source, "multi": b"{}\n"})
        with pytest.raises(GaltonError) as raised:
            tool(bad, "benchmark_import")
        assert raised.value.key == "external_sha_mismatch" and bad.store.find_suite("IFMTBench") is None, "no half-made suite"

    def test_offline_instances_only_use_a_verified_copy(self, tmp_path, source):
        svc = build_services(tmp_path)                                 # offline, as every test instance is
        with pytest.raises(GaltonError) as raised:
            tool(svc, "benchmark_import", per_type=1, multi=0)
        assert raised.value.key == "external_offline"
        folder = svc.config.external_dir / "ifmtbench"
        folder.mkdir(parents=True)
        (folder / "test_single_constraint.jsonl").write_bytes(source["single"])
        assert tool(svc, "benchmark_import", per_type=1, multi=0)["cases"] == 6

    def test_a_row_the_benchmark_cannot_use_is_reported_in_the_result(self, tmp_path, monkeypatch):
        files = data.files()
        files["single"] += b"this is not json\n"
        monkeypatch.setattr(ifmtbench, "PINS", data.pins(files))
        out = tool(online(tmp_path, files), "benchmark_import", per_type=2, multi=0)
        assert out["skipped_row_count"] == 1 and out["skipped_rows"][0]["file"] == "single"


# ------------------------------------------------------------------ a whole run
JUDGE_SAYS = {"glossary": "1", "style": '{"scores": {"style": 4, "background": 5}}'}


def judge_responder(req):
    prompt = req.messages[-1]["content"]
    return JUDGE_SAYS["glossary"] if "Glossary Compliance" in prompt else JUDGE_SAYS["style"]


def perfect_translator(rows):
    from galton_hoard.fakes import reference_responder
    return reference_responder(data.references(rows), accuracy=1.0)


@pytest.fixture
def bench(tmp_path, source, monkeypatch):
    svc = online(tmp_path, source)
    out = tool(svc, "benchmark_import", per_type=4, multi=5, seed=5)
    rows = [json.loads(line) for key in source for line in source[key].decode().splitlines()]
    return svc, out["suite"]["id"], rows


def results(svc, run, model):
    return {svc.store.case(r["case_id"])["title"]: r for r in svc.store.results(run_id=run["id"], contestant_id=model["id"])}


class TestRun:
    def test_rules_score_at_once_and_judged_items_wait_for_a_judge(self, bench):
        svc, suite, rows = bench
        model = add_gguf(svc, "traductor", responder=perfect_translator(rows))
        run = run_inline(svc, [suite], [model["id"]])
        assert run["state"] == "done"
        got = results(svc, run, model)
        waiting = next(r for t, r in got.items() if t.startswith("style "))
        assert waiting["judge_pending"] and waiting["detail"]["unjudged"] == ["style"] and waiting["detail"]["dimensions"][0]["score"] is None
        rule_only = [r for t, r in got.items() if t.split(" ")[0] in ("layout", "code", "structured")]
        assert len(rule_only) == 12 and all(r["score"] == 1.0 and r["passed"] and not r["judge_pending"] for r in rule_only)
        pending = [r for r in got.values() if r["judge_pending"]]
        assert pending and all(not r["passed"] for r in pending), "pending, never a silent zero"
        assert run["summary"]["pending_judge"] == len(pending)
        board = svc.board.leaderboard(suite=suite)["rows"][0]
        assert board["n"] == len(got) - len(pending), "pending items are not in the score"

    def test_the_judge_then_grades_what_waited_and_the_combination_is_applied(self, bench):
        svc, suite, rows = bench
        model = add_gguf(svc, "traductor", responder=perfect_translator(rows))
        run = run_inline(svc, [suite], [model["id"]])
        svc.settings.set_many({"judge.contestant": add_gguf(svc, "juez", responder=judge_responder)["id"]})
        outcome = svc.runner.judge_pending()
        assert outcome["graded"] and outcome["pending"] == 0
        got = results(svc, run, model)
        assert not any(r["judge_pending"] for r in got.values())
        assert all("unjudged" not in r["detail"] and "reason" not in r["detail"] for r in got.values()), "what waited for the judge no longer says so once graded"
        by_prefix = {}
        for title, r in got.items():
            by_prefix.setdefault(title.split(" ")[0], []).append(r["score"])
        assert set(by_prefix["style"]) == {0.8}, "style 4 of 5"
        assert set(by_prefix["background"]) == {1.0}, "background 5 of 5"
        assert set(by_prefix["glossary"]) == {1.0}
        assert set(by_prefix["glossary+style"]) == {0.8}, "gate 1 x style 0.8"
        assert set(by_prefix["glossary+style+background"]) == {0.9}, "gate 1 x mean(0.8, 1.0)"
        assert set(by_prefix["glossary+background"]) == {1.0}

    def test_a_model_that_ignores_the_instructions_fails_the_gates(self, bench):
        svc, suite, rows = bench
        svc.settings.set_many({"judge.contestant": add_gguf(svc, "juez", responder=lambda req: "0" if "Glossary Compliance" in req.messages[-1]["content"] else '{"scores": {"style": 1, "background": 1}}')["id"]})
        model = add_gguf(svc, "descuidado", responder=lambda req: "Una traducción libre sin nada de lo pedido.")
        run = run_inline(svc, [suite], [model["id"]])
        got = results(svc, run, model)
        assert all(r["score"] == 0.0 for t, r in got.items() if t.split(" ")[0] in ("glossary", "layout", "code", "structured"))
        assert set(r["score"] for t, r in got.items() if t.startswith("style ")) == {0.2}
        assert all(r["score"] == 0.0 for t, r in got.items() if "+" in t.split(" ")[0]), "a failed gate zeroes a multi-constraint item"

    def test_the_judge_receives_the_benchmarks_own_prompt_once_per_item(self, bench):
        svc, suite, rows = bench
        seen: list[str] = []

        def judge(req):
            seen.append(req.messages[-1]["content"])
            return judge_responder(req)
        svc.settings.set_many({"judge.contestant": add_gguf(svc, "juez", responder=judge)["id"]})
        model = add_gguf(svc, "traductor", responder=perfect_translator(rows))
        run_inline(svc, [suite], [model["id"]])
        svc.runner.judge_pending()
        assert all(p.startswith("\n# ROLE") and "<instruction>" in p and "<model_output>" in p for p in seen) and seen
        before = len(seen)
        run_inline(svc, [suite], [model["id"]])
        svc.runner.judge_pending()
        assert len(seen) == before, "grades are cached by what the judge saw"

    def test_per_constraint_pass_rates_show_where_a_model_fails(self, bench):
        svc, suite, rows = bench
        svc.settings.set_many({"judge.contestant": add_gguf(svc, "juez", responder=judge_responder)["id"]})
        good = perfect_translator(rows)

        def uneven(req):                                 # right everywhere except it drops the code spans
            reply = good(req)
            return reply.replace("`npm test`", "npm test") if isinstance(reply, str) else reply
        model = add_gguf(svc, "desigual", responder=uneven)
        run_inline(svc, [suite], [model["id"]])
        board = tool(svc, "leaderboard", suite=suite)
        constraints = board["rows"][0]["constraints"]
        assert constraints["code"]["pass_rate"] == 0.0 and constraints["code"]["n"] == 4
        assert constraints["layout"]["pass_rate"] == 1.0 and constraints["glossary"]["pass_rate"] == 1.0 and constraints["style"]["score"] == 0.8
        assert set(constraints) == {"glossary", "style", "background", "layout", "structured", "code"}
        assert all(c["unjudged"] == 0 for c in constraints.values())

    def test_without_a_judge_the_breakdown_counts_what_could_not_be_judged(self, bench):
        svc, suite, rows = bench
        model = add_gguf(svc, "traductor", responder=perfect_translator(rows))
        run_inline(svc, [suite], [model["id"]])
        constraints = svc.board.leaderboard(suite=suite)["rows"][0]["constraints"]
        assert constraints["layout"]["unjudged"] == 0 and constraints["layout"]["n"] == 4
        assert constraints["style"]["n"] == 0 and constraints["style"]["unjudged"] >= 4 and constraints["style"]["score"] is None

    def test_run_results_carry_the_constraint_detail_for_every_item(self, bench):
        svc, suite, rows = bench
        model = add_gguf(svc, "traductor", responder=perfect_translator(rows))
        run = run_inline(svc, [suite], [model["id"]])
        shown = tool(svc, "run_results", run=run["id"], limit=200)["results"]
        detail = next(r["detail"] for r in shown if r["title"].startswith("layout "))
        assert detail["dimensions"][0]["class"] == "layout" and detail["dimensions"][0]["passed"] is True and detail["final_score"] == 1.0

    def test_case_try_runs_one_item_through_the_same_checker(self, bench):
        svc, suite, rows = bench
        model = add_gguf(svc, "traductor", responder=perfect_translator(rows))
        case = next(c for c in svc.store.cases(suite) if c["title"].startswith("code "))
        out = tool(svc, "case_try", model=model["id"], case=case["id"])
        assert out["verdict"]["score"] == 1.0 and out["checker"] == "ifmt"
