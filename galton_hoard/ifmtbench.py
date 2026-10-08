"""IFMTBench as a suite: an instruction-following translation benchmark whose data is downloaded, checked and sampled, never copied into this repository.

Source: github.com/Tencent-Hunyuan/Hy-MT2, folder ``IFMTBench`` (code Apache-2.0, data CC BY 4.0, Copyright (C) 2026 Tencent; see NOTICE).
The two data files are fetched from a URL that names an exact commit, kept under ``<data>/external/ifmtbench/`` and verified against the SHA-256
recorded here before a single row is read; a file that does not match is refused and removed, so a changed upstream can never slip into a ranking.
Rows are grouped by constraint (``glossary``, ``style``, ``background``, ``layout``, ``structured``, ``code``; several in the multi-constraint file)
and a seeded stratified sample becomes the cases of an ordinary user suite. Scoring is the ``ifmt`` checker.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from .checkers import CheckContext, ModelOutput, run_checker
from .checkers.ifmt import JUDGED
from .errors import GaltonError
from .hoard_link.atomic import write_bytes_atomic
from .messages import text as message
from .suites import normalise_case, validate_case

REPO = "Tencent-Hunyuan/Hy-MT2"
COMMIT = "ff1903ecaa724e10951a23c16817a2413c752b35"
URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
SOURCE_PAGE = "https://github.com/Tencent-Hunyuan/Hy-MT2/tree/main/IFMTBench"
CREDIT = "IFMTBench, Copyright (C) 2026 Tencent, data under CC BY 4.0 (github.com/Tencent-Hunyuan/Hy-MT2); code under Apache-2.0."
DEFAULT_NAME = "IFMTBench"
DEFAULT_PER_TYPE = 30
DEFAULT_MULTI = 30
DEFAULT_SEED = 2026
MARKER = "source:ifmtbench"                  # in the suite notes: this suite came from here


@dataclass(frozen=True)
class Pin:
    """One data file: where it is in the repository, how many bytes and rows it has and its SHA-256."""

    path: str
    sha256: str
    size: int
    rows: int


PINS: dict[str, Pin] = {
    "single": Pin("IFMTBench/data/test_single_constraint.jsonl", "ffaab0947722711e5a7d0c47d6f7931868164ad9a068c253acc7ba909defc691", 8_478_197, 4506),
    "multi": Pin("IFMTBench/data/test_multi_constraint.jsonl", "c9fb393c49880e688e58515b215e1e586f9b72d143a404546f20de4a01a3e17f", 14_066_944, 2838),
}

#: the benchmark's class labels (Chinese in the data) -> the constraint they score
FAMILY = {
    "机器翻译-术语表约束翻译": "glossary",
    "机器翻译-风格指令遵循": "style",
    "机器翻译-带上下文背景翻译": "background",
    "机器翻译-布局保留翻译": "layout",
    "机器翻译-结构化数据翻译": "structured",
    "机器翻译-内联代码保留翻译": "code",
    "机器翻译-代码标签保留翻译": "code",
}
#: language names in the data -> short codes (a name that is not here is shown as it is)
LANGUAGES = {
    "中文": "zh", "繁体中文": "zh-Hant", "Chinese": "zh", "英文": "en", "英语": "en", "日文": "ja", "日语": "ja", "Japanese": "ja", "韩文": "ko", "韩语": "ko",
    "法文": "fr", "法语": "fr", "德文": "de", "德语": "de", "西班牙文": "es", "西班牙语": "es", "俄文": "ru", "俄语": "ru", "阿拉伯语": "ar", "泰语": "th",
    "土耳其语": "tr", "捷克语": "cs", "藏语": "bo", "藏文": "bo", "古吉拉特文": "gu",
}
ORDER = ("glossary", "style", "background", "layout", "structured", "code")


def lang(name: Any) -> str:
    name = str(name or "").strip()
    name = name.split("：")[-1] if "：" in name else name        # one row says «该文字的语种是：中文»
    return LANGUAGES.get(name, name or "?")


# ------------------------------------------------------------------------------------------------- files
def folder(config: Any) -> Path:
    return Path(config.external_dir) / "ifmtbench"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_file(config: Any, key: str, *, client_factory: Optional[Callable[[], httpx.Client]] = None, refresh: bool = False, offline: bool = False) -> dict[str, Any]:
    """The path of the verified data file ``key`` (``single`` or ``multi``), downloading it when it is missing, altered or ``refresh`` is true.

    A download is checked against the pinned SHA-256 before it is written; a mismatch is refused and nothing is kept."""
    pin = PINS[key]
    target = folder(config) / Path(pin.path).name
    if target.is_file() and not refresh:
        if sha256_of(target) == pin.sha256:
            return {"path": target, "downloaded": False, "sha256": pin.sha256}
        target.unlink()                                               # an altered or half-written copy is never used
    if offline:
        raise GaltonError("unavailable", "external_offline", file=target.name)
    url = URL.format(repo=REPO, commit=COMMIT, path=pin.path)
    factory = client_factory or (lambda: httpx.Client(timeout=120.0, follow_redirects=True))
    try:
        with factory() as client:
            response = client.get(url)
    except httpx.HTTPError as exc:
        raise GaltonError("unavailable", "external_download_failed", url=url, detail=f"{type(exc).__name__}: {exc}") from exc
    if response.status_code != 200:
        raise GaltonError("unavailable", "external_download_failed", url=url, detail=f"HTTP {response.status_code}")
    data = response.content
    found = hashlib.sha256(data).hexdigest()
    if found != pin.sha256:
        raise GaltonError("invalid", "external_sha_mismatch", file=target.name, expected=pin.sha256[:16], found=found[:16])
    target.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(target, data)
    return {"path": target, "downloaded": True, "sha256": found}


# ------------------------------------------------------------------------------------------------- rows
def read_items(path: Path, group: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(items, errors)`` from a data file. An item keeps only what scoring and display need; a row that cannot be used is reported, never guessed."""
    items: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                items.append(_item(row, group))
            except (ValueError, KeyError, TypeError) as exc:
                errors.append({"row": number, "file": group, "error": f"{type(exc).__name__}: {exc}"[:200]})
    return items, errors


def _item(row: Any, group: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise TypeError("not an object")
    labels = row["class"] if isinstance(row["class"], list) else [row["class"]]
    families: list[str] = []
    for label in labels:
        family = FAMILY.get(label)
        if family is None:
            raise KeyError(f"unknown constraint class {label!r}")
        if family not in families:
            families.append(family)
    prompt, reference, md5 = row["input"], row["output"], row["md5"]
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(reference, str) or not isinstance(md5, str) or not md5:
        raise ValueError("input, output and md5 must be text")
    meta = row.get("meta_data") if isinstance(row.get("meta_data"), dict) else {}
    fmt = row.get("data_format") or meta.get("data_format") or ""
    instruction_lang = str(row.get("instruction_lang") or "")
    return {"key": f"{md5}:{instruction_lang}", "md5": md5, "group": group, "families": sorted(families, key=ORDER.index), "input": prompt, "output": reference,
            "origin_text": str(row.get("origin_text") or ""), "term_dict": row.get("term_dict") or "", "data_format": str(fmt), "meta": meta,
            "origin_language": lang(row.get("origin_language")), "target_language": lang(row.get("target_language")), "instruction_lang": lang(instruction_lang)}


def stratum(item: dict[str, Any]) -> str:
    return "+".join(item["families"])


# ------------------------------------------------------------------------------------------------- sampling
def sample(items: list[dict[str, Any]], *, per_type: int, multi: int, seed: int) -> list[dict[str, Any]]:
    """A stratified, seeded sample: ``per_type`` rows of every single-constraint type and ``multi`` rows spread evenly over the multi-constraint
    combinations (the first combinations take the remainder). Inside a stratum rows of different source texts come first (the same text appears in
    several instruction languages), then the rest. The result is interleaved across strata, so any prefix of it is balanced.
    It does not depend on the order of the file."""
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in items:
        strata.setdefault((item["group"], stratum(item)), []).append(item)
    singles = sorted(k for k in strata if k[0] == "single")
    multis = sorted(k for k in strata if k[0] == "multi")
    quota = {k: per_type for k in singles}
    for i, k in enumerate(multis):
        quota[k] = multi // len(multis) + (1 if i < multi % len(multis) else 0)
    picked: list[list[dict[str, Any]]] = []
    for k in [*singles, *multis]:
        rows = sorted(strata[k], key=lambda r: r["key"])
        random.Random(f"{seed}:{k[0]}:{k[1]}").shuffle(rows)
        seen: set[str] = set()
        first = [r for r in rows if not (r["md5"] in seen or seen.add(r["md5"]))]
        taken = {r["key"] for r in first}
        picked.append(sorted((first + [r for r in rows if r["key"] not in taken])[: quota[k]], key=lambda r: r["key"]))
    out: list[dict[str, Any]] = []
    for position in range(max((len(p) for p in picked), default=0)):
        out += [p[position] for p in picked if position < len(p)]
    return out


# ------------------------------------------------------------------------------------------------- cases
def answer_budget(reference: str) -> int:
    """Tokens a good answer may need: generous for scripts that take one token per character, bounded so a runaway answer is cut."""
    return max(512, min(4096, 2 * len(reference) + 256))


def checker_spec(item: dict[str, Any]) -> dict[str, Any]:
    """The ``ifmt`` checker of one item, carrying only the data its constraints need."""
    families = item["families"]
    spec: dict[str, Any] = {"type": "ifmt", "classes": families, "item": item["key"]}
    if "glossary" in families:
        spec["term_dict"] = item["term_dict"]
    if "structured" in families or "layout" in families:
        spec["origin_text"] = item["origin_text"]
    if "structured" in families and item["data_format"]:
        spec["data_format"] = item["data_format"]
    if item["meta"] and ("layout" in families or "code" in families):
        spec["meta"] = {k: v for k, v in item["meta"].items() if k in ("primary_delimiter", "source_chunks", "extracted_assets")}
    return spec


def _perfect_judge(request: dict[str, Any]) -> dict[str, Any]:
    """A judge that approves everything, to ask whether the rule checks alone can be satisfied."""
    return {"score": 1.0, "glossary": 1} if request.get("parse") == "ifmt_glossary" else {"score": None, "style": 5, "background": 5}


def reference_passes(item: dict[str, Any]) -> bool:
    """Does the benchmark's own reference translation satisfy the item's rule checks (with every judgement granted)? One that does not cannot be
    satisfied by any model: the Markdown items that are not tables, for instance, are refused by the benchmark's table reader."""
    result = run_checker(checker_spec(item), ModelOutput(text=item["output"]), CheckContext(case={"reference": item["output"], "prompt_text": item["input"]}, judge=_perfect_judge))
    return not result.get("unavailable") and result["score"] >= 1.0


def build_case(item: dict[str, Any]) -> dict[str, Any]:
    """The Galton case of one item: the benchmark's own instruction as the whole prompt, the reference translation as ``reference``, an ``ifmt`` checker."""
    families, spec = item["families"], checker_spec(item)
    pair = f"{item['origin_language']}→{item['target_language']}"
    case = normalise_case({
        "title": f"{'+'.join(families)} · {pair} · {item['md5'][:8]}/{item['instruction_lang']}",
        "prompt": {"text": item["input"]},
        "checker": spec,
        "weight": 1.0,
        "tags": ["ifmtbench", item["group"], *families, pair, f"instruction:{item['instruction_lang']}"],
        "reference": item["output"],
        "notes": f"{MARKER} {item['key']}",
        "max_tokens": answer_budget(item["output"]),
    })
    problems = validate_case(case)
    if problems:
        raise GaltonError("invalid", "case_invalid", problems="; ".join(problems), problem_items=list(problems))
    return case


def describe(per_type: int, multi: int, seed: int) -> str:
    return (f"Traducción con instrucciones: glosario, estilo, contexto, formato, datos estructurados y código o etiquetas. Muestra estratificada de IFMTBench "
            f"({per_type} por tipo con una restricción y {multi} con varias, semilla {seed}; datos fijados en el commit {COMMIT[:7]}). Puntuación de un caso: producto de las "
            f"restricciones de todo o nada por la media de las graduales. Estilo, contexto y el glosario que falla la regla los puntúa el modelo juez. "
            f"{CREDIT}")


# ------------------------------------------------------------------------------------------------- the import
def import_suite(svc: Any, *, name: str = "", per_type: int = DEFAULT_PER_TYPE, multi: int = DEFAULT_MULTI, seed: int = DEFAULT_SEED, category: str = "custom",
                 refresh: bool = False, keep_unsatisfiable: bool = False) -> dict[str, Any]:
    """Download (or reuse) and verify the data, take the sample and create the suite. Returns the suite and what it holds."""
    if per_type + multi <= 0:
        raise GaltonError("invalid", "external_empty_sample")
    suite_name = (name or DEFAULT_NAME).strip()
    if svc.store.find_suite(suite_name) is not None:
        raise GaltonError("conflict", "external_suite_exists", name=suite_name)
    offline = bool(svc.config.offline or svc.config.fake)
    files, wanted = {}, [k for k, n in (("single", per_type), ("multi", multi)) if n > 0]
    for key in wanted:
        files[key] = fetch_file(svc.config, key, client_factory=svc.client_factory, refresh=refresh, offline=offline)
    items: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for key in wanted:
        rows, bad = read_items(files[key]["path"], key)
        items += rows
        errors += bad
    unsatisfiable: dict[str, int] = {}
    if not keep_unsatisfiable:
        usable = []
        for item in items:
            if reference_passes(item):
                usable.append(item)
            else:
                label = f"{item['group']}:{stratum(item)}"
                unsatisfiable[label] = unsatisfiable.get(label, 0) + 1
        items = usable
    chosen = sample(items, per_type=per_type if "single" in wanted else 0, multi=multi if "multi" in wanted else 0, seed=seed)
    if not chosen:
        raise GaltonError("invalid", "external_no_rows")
    cases = [build_case(item) for item in chosen]
    sha = {k: v["sha256"] for k, v in files.items()}
    notes = f"{MARKER} commit {COMMIT} seed {seed} per_type {per_type} multi {multi} keep_unsatisfiable {keep_unsatisfiable} sha256 " + " ".join(f"{k}={v}" for k, v in sorted(sha.items()))
    with svc.db.transaction():
        suite = svc.store.create_suite(name=suite_name, description=describe(per_type, multi, seed), category=category, builtin=False, max_tokens=1024,
                                       position=100 + len(svc.store.suites()), notes=notes)
        for case in cases:
            svc.store.create_case(suite_id=suite["id"], position=svc.store.next_case_position(suite["id"]), source="import", max_tokens=case.get("max_tokens"),
                                  **{k: case[k] for k in ("title", "prompt", "checker", "weight", "tags", "tools", "images", "reference", "notes")})
    by_type: dict[str, int] = {}
    languages: dict[str, int] = {}
    for item in chosen:
        label = f"{item['group']}:{stratum(item)}"
        by_type[label] = by_type.get(label, 0) + 1
        languages[item["instruction_lang"]] = languages.get(item["instruction_lang"], 0) + 1
    judged = sum(1 for item in chosen if any(f in JUDGED for f in item["families"]))
    notes_out = [] if svc.settings.get("judge.contestant") else [message("ifmt_no_judge_note", n=judged)]
    return {"suite": svc.suite_card({**suite, "cases": len(cases)}), "cases": len(cases), "by_type": by_type, "instruction_languages": languages, "needs_judge": judged,
            "rule_only": len(cases) - judged, "source": {"repo": REPO, "commit": COMMIT, "url": SOURCE_PAGE, "sha256": sha, "downloaded": {k: v["downloaded"] for k, v in files.items()},
                                                          "data_licence": "CC-BY-4.0", "code_licence": "Apache-2.0", "credit": CREDIT},
            "left_out_unsatisfiable": unsatisfiable, "skipped_rows": errors[:20], "skipped_row_count": len(errors), "notes": notes_out}


__all__ = ["PINS", "COMMIT", "CREDIT", "MARKER", "import_suite", "sample", "build_case", "read_items", "fetch_file", "GATES", "CONTINUOUS"]
