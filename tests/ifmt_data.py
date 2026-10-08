"""Invented rows in the shape of the instruction-following translation benchmark, served from a fake download. No network, no real data."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx

from galton_hoard import ifmtbench
from galton_hoard.ifmtbench import FAMILY

LABEL = {"glossary": "机器翻译-术语表约束翻译", "style": "机器翻译-风格指令遵循", "background": "机器翻译-带上下文背景翻译", "layout": "机器翻译-布局保留翻译",
         "structured": "机器翻译-结构化数据翻译", "code": "机器翻译-内联代码保留翻译"}
assert all(FAMILY[v] == k for k, v in LABEL.items())
TAGS_LABEL = "机器翻译-代码标签保留翻译"
LANGS = ("中文", "英语")
TERMS = json.dumps({"cat": ["gato"], "dog": ["perro"]})
STRUCTURED = [
    ("JSON", '{"title": "Hello", "tags": ["a", "b"]}', '{"title": "Hola gato perro", "tags": ["a", "b"]}'),
    ("HTML", '<p class="x">Hello <b>world</b></p>', '<p class="x">Hola gato <b>perro</b></p>'),
    ("CSV", "name,city\nAna,Paris\nLuis,Rome", "nombre,ciudad\nAna,gato\nLuis,perro"),
    ("Markdown表格", "| a | b |\n| - | - |\n| 1 | 2 |", "| gato | perro |\n| - | - |\n| 1 | 2 |"),
    ("Markdown", "# Title\n\nSome text", "# Título\n\ngato perro"),     # not a table: the benchmark's own reader refuses even the reference
]


def md5(*parts: Any) -> str:
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()


def row(tag: str, i: int, classes: list[str], lang: str, **extra: Any) -> dict[str, Any]:
    base = {"input": f"#{tag}{i:03d}# Translate this text for case {i}.", "output": f"Traducción {tag} {i}.", "origin_language": "英语", "target_language": "西班牙语",
            "origin_text": f"Text {tag} {i}.", "class": classes, "md5": md5(tag, i), "instruction_lang": lang}
    return {**base, **extra}


def single_rows(per_family: int = 5) -> list[dict[str, Any]]:
    rows = []
    for i in range(per_family):
        for lang in LANGS:
            glossary = row("g", i, [LABEL["glossary"]], lang, input=f"#g{i:03d}# Translate: The cat and the dog sleep ({i}). Use cat = gato, dog = perro.",
                           output=f"El gato y el perro duermen ({i}).", origin_text=f"The cat and the dog sleep ({i}).", **({"term_dict": TERMS} if i % 4 != 3 else {}))
            rows += [
                glossary,
                row("s", i, [LABEL["style"]], lang, input=f"#s{i:03d}# Translate in a formal register: See you soon ({i})."),
                row("b", i, [LABEL["background"]], lang, input=f"#b{i:03d}# Background: a bank. Translate: The bank is closed ({i})."),
                row("l", i, [LABEL["layout"]], lang, input=f"#l{i:03d}# Keep every ' | ' delimiter. Translate: Item A | Item B | Item C ({i})", origin_text="Item A | Item B | Item C",
                    output="Elemento A | Elemento B | Elemento C", meta_data={"primary_delimiter": " | ", "source_chunks": ["Item A", "Item B", "Item C"]}),
            ]
            fmt, origin, out = STRUCTURED[i % len(STRUCTURED)]
            rows.append(row("t", i, [LABEL["structured"]], lang, input=f"#t{i:03d}# Keep the structure. Translate: {origin}", origin_text=origin, output=out, data_format=fmt))
            rows.append(row("c", i, [LABEL["code"] if i % 2 else TAGS_LABEL], lang, input=f"#c{i:03d}# Keep the code. Translate: Run `npm test` now ({i})", origin_text=f"Run `npm test` now ({i})",
                            output=f"Ejecuta `npm test` ahora ({i})", meta_data={"extracted_assets": ["`npm test`"]}))
    return rows


MULTI = [("glossary", "style"), ("glossary", "background"), ("glossary", "structured"), ("glossary", "style", "background"), ("glossary", "style", "structured")]


def multi_rows(per_combo: int = 4) -> list[dict[str, Any]]:
    rows = []
    for k, combo in enumerate(MULTI):
        for i in range(per_combo):
            extra: dict[str, Any] = {"term_dict": TERMS}
            origin, out = f"The cat and the dog ({k}{i}).", f"El gato y el perro ({k}{i})."
            if "structured" in combo:
                fmt, origin, out = STRUCTURED[(k + i) % len(STRUCTURED)]
                extra["data_format"] = fmt
            rows.append(row(f"m{k}", i, [LABEL[c] for c in reversed(combo)], LANGS[i % 2], input=f"#m{k}{i:03d}# Translate with cat = gato, dog = perro, {' and '.join(combo)}: {origin}",
                            origin_text=origin, output=out, **extra))
    return rows


def jsonl(rows: list[dict[str, Any]]) -> bytes:
    return ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")


def files() -> dict[str, bytes]:
    return {"single": jsonl(single_rows()), "multi": jsonl(multi_rows())}


def pins(data: dict[str, bytes]) -> dict[str, ifmtbench.Pin]:
    return {k: ifmtbench.Pin(ifmtbench.PINS[k].path, hashlib.sha256(v).hexdigest(), len(v), v.count(b"\n")) for k, v in data.items()}


def server(data: dict[str, bytes], calls: list[str] | None = None):
    """A mock handler that serves the fixture files at the benchmark's URLs."""
    by_path = {ifmtbench.PINS[k].path: v for k, v in data.items()}

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        for path, content in by_path.items():
            if str(request.url).endswith(path):
                return httpx.Response(200, content=content)
        return httpx.Response(404)

    return handler


def references(rows: list[dict[str, Any]]) -> dict[str, str]:
    """What a perfect translator says for each prompt: the prompt's first 200 characters -> the reference translation."""
    return {r["input"][:200]: r["output"] for r in rows}
