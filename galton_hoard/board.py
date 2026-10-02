"""Reading the measurements: the leaderboard, the paired comparison of two models and the recommendation for a task described in words.

All of it is computed from stored results at request time (nothing is cached), so it always agrees with the database. Results only count for
the digest a model has *now*: when a model file changes under the same name its old results are stale and are shown as such, never mixed in.
"""

from __future__ import annotations

from typing import Any, Optional

from . import gguf_meta, placement, stats
from .errors import GaltonError
from .messages import text
from .routes import TASKS, Routes, names_for, vram_gb
from .suites import CATEGORIES
from .util import fold

FITS_16_GB = placement.FITS_16GB_MB / 1024  # GB usable on a 16 GB card once the driver and the display have taken their share

#: words that point at a task category, Spanish and English; the category with most hits wins (ties: the order below)
KEYWORDS: dict[str, tuple[str, ...]] = {
    "code": ("codigo", "programa", "programar", "funcion", "python", "javascript", "typescript", "bug", "script", "refactor", "sql", "regex", "code", "debug", "compilar", "algoritmo", "clase", "test unitario"),
    "math": ("matematica", "ecuacion", "integral", "derivada", "probabilidad", "algebra", "calculo", "math", "equation", "calculus", "demostracion", "estadistica", "porcentaje"),
    "tool_use": ("herramienta", "herramientas", "tool", "tools", "function calling", "llamar a una funcion", "agente", "agent", "api"),
    "extraction": ("extrae", "extraer", "extraccion", "extract", "factura", "invoice", "campos", "json", "parsear", "formulario", "datos estructurados", "cv"),
    "long_context": ("documento largo", "contexto largo", "long context", "libro", "muchas paginas", "transcripcion", "informe extenso", "long document", "contrato entero"),
    "rag": ("rag", "citas", "cita", "fuentes", "base de conocimiento", "retrieval", "cite", "citar", "segun los documentos", "pasajes"),
    "vision": ("imagen", "imagenes", "foto", "captura", "screenshot", "ocr", "grafico", "image", "picture", "diagrama", "mirar", "visual"),
    "summary": ("resumen", "resume", "resumir", "resumenes", "summarize", "summary", "tl;dr", "sintetiza", "sintesis"),
    "translation": ("traduce", "traducir", "traduccion", "translate", "translation", "idioma", "al ingles", "al espanol"),
    "writing_es": ("redacta", "redactar", "escribe", "escribir", "correo", "carta", "redaccion", "ortografia", "writing", "email", "articulo", "texto formal", "estilo"),
}


def guess_category(task: str) -> dict[str, Any]:
    """Which task category a free-text description points at. Falls back to ``general`` and says it did."""
    text = " " + fold(task) + " "
    hits: dict[str, list[str]] = {}
    for category, words in KEYWORDS.items():
        found = [w for w in words if f" {w}" in text or (len(w) > 4 and w in text)]
        if found:
            hits[category] = found
    if not hits:
        return {"category": "general", "guessed": False, "matched": [], "note": "No keyword matched: using the general category."}
    best = max(hits, key=lambda c: (len(hits[c]), -list(KEYWORDS).index(c)))
    return {"category": best, "guessed": True, "matched": hits[best], "alternatives": [c for c in hits if c != best]}


def suite_ids_for(store: Any, *, category: str = "", suite: str = "") -> tuple[list[str], str]:
    """The suites a category or a suite name stands for, plus a label. A category is a task (``code``, ``general``...) or a suite category."""
    if suite:
        found = store.find_suite(suite)
        if found is None:
            raise GaltonError("not_found", "no_suite", ref=suite)
        return [found["id"]], found["name"]
    if not category:
        return [s["id"] for s in store.suites()], "all"
    if category in TASKS:
        cats = TASKS[category]
    elif category in CATEGORIES:
        cats = (category,)
    else:
        raise GaltonError("invalid", "unknown_category", category=category, options=sorted({*TASKS, *CATEGORIES}))
    return [s["id"] for s in store.suites() if s["category"] in cats], category


def memory_gb(store: Any, c: dict[str, Any]) -> tuple[Optional[float], str]:
    """Memory a model needs: measured by a run, else reported by Ollama, else estimated from the file size. The second item says which."""
    measured = vram_gb(store, c)
    if measured is not None:
        return measured, "measured"
    if c.get("size_bytes") and c["kind"] == "gguf" or (c.get("size_bytes") and c.get("source") == "ollama"):
        return round((c["size_bytes"] * gguf_meta.FILE_FACTOR / (1024 ** 3)) + gguf_meta.HEADROOM_MB / 1024, 1), "estimated"
    return None, ""


class Board:
    def __init__(self, store: Any, settings: Any, routes: Routes, clock: Any):
        self.store, self.settings, self.routes, self.clock = store, settings, routes, clock

    # ------------------------------------------------------------------ leaderboard
    def _fresh(self, c: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in rows if not c["digest"] or r["digest"] == c["digest"]]

    def leaderboard(self, *, category: str = "", suite: str = "", days: int = 0, include_stale: bool = False, include_disabled: bool = False) -> dict[str, Any]:
        suite_ids, label = suite_ids_for(self.store, category=category, suite=suite)
        since = self.clock() - days * 86400 if days else 0.0
        rows_all = self.store.scoring_rows(suite_ids=suite_ids, since_ts=since) if suite_ids else []
        by_c: dict[str, list[dict[str, Any]]] = {}
        for r in rows_all:
            by_c.setdefault(r["contestant_id"], []).append(r)
        board, unmeasured = [], []
        last = self.store.last_measured()
        for c in self.store.contestants(include_missing=True):
            rows = by_c.get(c["id"], [])
            usable = c["enabled"] and not c["missing"]
            if not rows:
                if usable and not c["adhoc"]:
                    unmeasured.append({"contestant": c["id"], "name": c["name"], "kind": c["kind"]})
                continue
            if not usable and not include_disabled:
                continue
            fresh = self._fresh(c, rows)
            stale = not fresh
            counted = rows if (stale and include_stale) else fresh
            summary = stats.summarize(counted)
            mem, method = memory_gb(self.store, c)
            speed = stats.speed_summary(self.store.speed_rows(c["id"], since_ts=since, digest="" if (stale and include_stale) else c["digest"]))
            by_cat: dict[str, Any] = {}
            cut = stats.truncation(counted)
            warnings = []
            for cat in sorted({r["category"] for r in counted}):
                per = stats.per_case([r for r in counted if r["category"] == cat])
                weight = sum(p["weight"] for p in per.values())
                by_cat[cat] = {"score": round(sum(p["score"] * p["weight"] for p in per.values()) / weight, 3) if weight else None, "n": len(per),
                               "truncated": cut[cat]["truncated"], "results": cut[cat]["results"]}
                if cut[cat]["warn"]:
                    warnings.append(text("board_truncated", name=c["name"], n=cut[cat]["truncated"], total=cut[cat]["results"], category=cat, pct=round(100 * cut[cat]["share"])))
            board.append({
                "contestant": c["id"], "name": c["name"], "kind": c["kind"], "provider": c["provider"], "family": c["family"], "params_b": c["params_b"], "quant": c["quant"],
                "vision": c["vision"], "enabled": c["enabled"], "missing": c["missing"], "remote": c["remote"], "adhoc": c["adhoc"],
                "score": summary["score"], "ci": summary["ci"], "pass_rate": summary["pass_rate"], "pass_ci": summary["pass_ci"], "n": summary["n"], "n_results": summary["n_results"],
                "lower": summary["ci"][0] if summary["ci"] else None, "by_category": by_cat,
                "decode_tps": speed["decode_tps_median"], "decode_tps_p90": speed["decode_tps_p90"], "prompt_tps": speed["prompt_tps_median"], "ttft_ms": speed["ttft_ms_median"],
                "memory_gb": mem, "memory_method": method, "fits_16gb": (mem <= FITS_16_GB) if mem is not None else None,
                "stale": stale, "stale_n": len(rows) - len(fresh), "last_ts": (last.get(c["id"]) or {}).get("last_ts"),
                "truncated": sum(1 for r in counted if r.get("truncated")), "warnings": warnings,
            })
        board.sort(key=lambda r: (r["stale"] and not include_stale, -(r["lower"] if r["lower"] is not None else -1), -(r["decode_tps"] or 0), r["name"]))
        for i, row in enumerate(board, 1):
            row["rank"] = None if (row["stale"] and not include_stale) else i
        return {"scope": label, "category": category, "suite": suite, "suites": suite_ids, "rows": board, "unmeasured": unmeasured,
                "note": "Ranked by the lower bound of the 95 % interval of the score, so a model measured on few cases cannot win by luck. "
                        "Results of a model whose file changed are stale and are not ranked."}

    # ------------------------------------------------------------------ comparing two models
    def compare(self, a: str, b: str, *, category: str = "", suite: str = "", include_stale: bool = False, max_cases: int = 60) -> dict[str, Any]:
        ca, cb = self.store.find_contestant(a), self.store.find_contestant(b)
        for ref, found in ((a, ca), (b, cb)):
            if found is None:
                raise GaltonError("not_found", "no_model", ref=ref)
        if ca["id"] == cb["id"]:
            raise GaltonError("invalid", "compare_same_model")
        suite_ids, label = suite_ids_for(self.store, category=category, suite=suite)
        every = {c["id"]: (self.store.scoring_rows(contestant_ids=[c["id"]], suite_ids=suite_ids) if suite_ids else []) for c in (ca, cb)}
        rows = {c["id"]: every[c["id"]] if include_stale else self._fresh(c, every[c["id"]]) for c in (ca, cb)}
        result = stats.compare(rows[ca["id"]], rows[cb["id"]])
        titles = {k: self.store.find_case(k) for k in [p["case_id"] for p in result["per_case"]]}
        listing = sorted(result["per_case"], key=lambda p: -abs(p["diff"]))[:max_cases]
        result["per_case"] = [{**p, "title": (titles.get(p["case_id"]) or {}).get("title", p["case_id"])} for p in listing]
        stale = [c["name"] for c in (ca, cb) if every[c["id"]] and not self._fresh(c, every[c["id"]])]
        if stale and not include_stale:
            result["warnings"].append(text("board_stale_left_out", names=stale))
        if result["verdict"] == "no_data":
            who_a, who_b = len(rows[ca["id"]]), len(rows[cb["id"]])
            result["warnings"].append(text("board_unequal_results", a=ca["name"], who_a=who_a, b=cb["name"], who_b=who_b, label=label))
        sentence = {
            "better": f"{ca['name']} is better than {cb['name']}",
            "worse": f"{ca['name']} is worse than {cb['name']}",
            "no_clear_difference": f"No clear difference between {ca['name']} and {cb['name']}",
            "no_data": f"Not enough shared data to compare {ca['name']} and {cb['name']}",
        }[result["verdict"]]
        if result["diff"] is not None:
            sentence += f" (score difference {result['diff']:+.3f}, 95 % interval {result['diff_ci'][0]:+.3f} to {result['diff_ci'][1]:+.3f}, p = {result['p_value']}, {result['n']} shared cases)"
        speeds = {c["id"]: stats.speed_summary(self.store.speed_rows(c["id"], digest="" if include_stale else c["digest"])) for c in (ca, cb)}
        return {"scope": label, "a": {"id": ca["id"], "name": ca["name"], "summary": stats.summarize(rows[ca["id"]]), "speed": speeds[ca["id"]]},
                "b": {"id": cb["id"], "name": cb["name"], "summary": stats.summarize(rows[cb["id"]]), "speed": speeds[cb["id"]]}, "sentence": sentence + ".", **result}

    # ------------------------------------------------------------------ recommending
    def recommend(self, task: str, category: str = "") -> dict[str, Any]:
        guess = {"category": category, "guessed": False, "matched": []} if category else guess_category(task)
        chosen = guess["category"]
        if chosen not in TASKS:
            raise GaltonError("invalid", "unknown_task_category", category=chosen, options=list(TASKS))
        policy = self.settings.get("routes.policy")
        evaluation = self.routes.evaluate(chosen, policy, self.clock())
        ranked = evaluation["ranked"]
        out: dict[str, Any] = {"task": task, "category": chosen, "guess": guess, "excluded": evaluation["excluded"], "warnings": evaluation["warnings"]}
        if ranked:
            why, parts = self.routes.explain(chosen, ranked)
            item = lambda x: {"id": x["contestant"]["id"], "name": x["contestant"]["name"], "names": names_for(x["contestant"]), **self.routes._item(x)}  # noqa: E731
            out.update(enough_data=True, recommendation=item(ranked[0]), runner_up=item(ranked[1]) if len(ranked) > 1 else None, why=why, comparison=parts.get("comparison"))
            return out
        board = self.leaderboard(category=chosen)
        measured = [r for r in board["rows"] if r["rank"] is not None]
        out["enough_data"] = False
        if measured:
            top = measured[0]
            out.update(recommendation={"id": top["contestant"], "name": top["name"], "score": top["score"], "ci": top["ci"], "n": top["n"], "tok_s": top["decode_tps"],
                                       "vram_gb": top["memory_gb"], "provisional": True},
                       runner_up=None,
                       why=f"Provisional: {top['name']} is the best measured model for {chosen} ({top['score']:.2f} on {top['n']} cases) but the policy needs more evidence "
                           f"(see excluded) before it would be published.")
        else:
            available = [c["name"] for c in self.store.contestants(enabled=True, include_missing=False, include_adhoc=False)][:12]
            out.update(recommendation=None, runner_up=None, why=f"No model has been measured on {chosen} yet.", available_models=available,
                       next="Run the suites for this category on your models with run_start (or measure_new for the quick suite).")
        return out
