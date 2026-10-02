"""The routing table: which measured model is best for each kind of task on this computer, and publishing it for Hoard Link.

Policy (setting ``routes.policy``): for each task category the candidates are the enabled models with checked results on that category's
suites in the last N days **for the digest they have now**; remote endpoints are excluded unless allowed; too slow or too big models are
excluded when their speed or memory is known; the rest are ranked by the LOWER bound of the 95 % interval of their score (a conservative
estimate: a model measured on few cases cannot beat a well-measured one by luck), with decode speed as the tie-break. A category with fewer
checked cases than ``min_cases`` is never published.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from . import stats
from .hoard_link.atomic import write_json_atomic
from .messages import text
from .util import iso_local, usable_alias

SCHEMA = 1
#: task category -> categories of suites whose results count for it
TASKS: dict[str, tuple[str, ...]] = {
    "general": ("reasoning", "instruction", "writing_es"),
    "writing_es": ("writing_es",), "code": ("code",), "extraction": ("extraction",), "tool_use": ("tool_use",), "long_context": ("long_context",),
    "rag": ("rag",), "vision": ("vision",), "summary": ("summary",), "translation": ("translation",), "math": ("math",),
}
CAPABILITY = {"vision": "vision"}


def names_for(c: dict[str, Any]) -> list[str]:
    """Every name another app may see this model under: the Ollama tag, the server's model name, the aliases and the file stem."""
    out: list[str] = []
    for name in [c.get("ollama_ref"), c.get("model"), *c.get("aliases", []), c.get("name")]:
        name = str(name or "").strip()
        if name and " (llama.cpp)" not in name and usable_alias(name) and name.casefold() not in {n.casefold() for n in out}:
            out.append(name)
    return out


def vram_gb(store: Any, c: dict[str, Any]) -> Optional[float]:
    """Measured memory when a run recorded it; else what Ollama says the loaded model takes; else None."""
    for m in store.measurements(c["id"], limit=5):
        if m.get("vram_mb"):
            return round(m["vram_mb"] / 1024, 1)
    size = (c.get("meta") or {}).get("size_vram")
    return round(size / 1024 ** 3, 1) if size else None


class Routes:
    def __init__(self, store: Any, settings: Any, path: Callable[[], Path], *, clock: Callable[[], float], emit: Optional[Callable[[str, dict[str, Any]], None]] = None):
        self.store, self.settings, self.path, self.clock = store, settings, path, clock
        self.emit = emit or (lambda *_: None)

    # ------------------------------------------------------------------ candidates
    def evaluate(self, category: str, policy: dict[str, Any], now: float) -> dict[str, Any]:
        suite_cats = TASKS[category]
        suite_ids = [s["id"] for s in self.store.suites() if s["category"] in suite_cats]
        since = now - float(policy["days"]) * 86400
        candidates, excluded, warnings = [], [], []
        if not suite_ids:
            return {"category": category, "ranked": [], "excluded": [], "warnings": []}
        all_rows = self.store.scoring_rows(suite_ids=suite_ids, since_ts=since)
        by_contestant: dict[str, list[dict[str, Any]]] = {}
        for r in all_rows:
            by_contestant.setdefault(r["contestant_id"], []).append(r)
        for c in self.store.contestants(include_missing=True):
            def out(reason: str) -> None:
                excluded.append({"contestant": c["id"], "name": c["name"], "reason": reason, "why": text(f"excl_{reason}")})
            rows = by_contestant.get(c["id"], [])
            if c["missing"] or not c["enabled"]:
                if rows:
                    out("disabled")
                continue
            if not rows:
                continue
            if c["adhoc"]:
                out("adhoc")
                continue
            if c["remote"] and not (policy["include_remote"] and c["remote_ok"]):
                out("remote")
                continue
            fresh = [r for r in rows if r["digest"] == c["digest"]]
            if not fresh:
                out("stale")
                continue
            summary = stats.summarize(fresh)
            if summary["n"] < int(policy["min_cases"]):
                out("too_few")
                continue
            speed = stats.speed_by_device(self.store.speed_rows(c["id"], since_ts=since, digest=c["digest"]))
            tps = speed["decode_tps_median"]
            if tps is not None and tps < float(policy["min_tok_s"]):
                out("too_slow")
                continue
            vram = vram_gb(self.store, c)
            if vram is not None and vram > float(policy["max_vram_gb"]):
                out("too_big")
                continue
            cut = stats.truncation(fresh, key="contestant_id")[c["id"]]
            if cut["warn"]:
                warnings.append({"contestant": c["id"], "name": c["name"], "truncated": cut["truncated"], "results": cut["results"],
                                 "why": text("board_truncated", name=c["name"], n=cut["truncated"], total=cut["results"], category=category, pct=round(100 * cut["share"]))})
            candidates.append({"contestant": c, "summary": summary, "tok_s": tps, "cpu": speed["cpu"], "vram_gb": vram, "rows": fresh, "truncated": cut["truncated"]})
        # a speed measured on the CPU is not comparable with GPU speeds: it never counts in the speed term nor as the tie-break
        for x in candidates:
            x["speed"] = 0.0 if x["cpu"] else (x["tok_s"] or 0)
        fastest = max((x["speed"] for x in candidates), default=0) or 1.0
        wq, ws = float(policy["weight_quality"]), float(policy["weight_speed"])
        for x in candidates:
            x["rank"] = wq * x["summary"]["ci"][0] + ws * (x["speed"] / fastest)
        candidates.sort(key=lambda x: (-x["rank"], -x["speed"], x["contestant"]["name"]))
        return {"category": category, "ranked": candidates, "excluded": excluded, "warnings": warnings}

    @staticmethod
    def _item(x: dict[str, Any]) -> dict[str, Any]:
        s = x["summary"]
        return {"names": names_for(x["contestant"]), "score": round(s["score"], 2), "ci": [round(s["ci"][0], 2), round(s["ci"][1], 2)], "n": s["n"],
                "tok_s": round(x["tok_s"], 1) if x["tok_s"] is not None else None, **({"cpu": True} if x.get("cpu") else {}), "vram_gb": x["vram_gb"]}

    def explain(self, category: str, ranked: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        """The sentence that goes in the table, and its parts for the UI."""
        top = ranked[0]
        s = top["summary"]
        name = top["contestant"]["name"]
        text = f"{name} wins {category}: {s['score']:.2f} [{s['ci'][0]:.2f}-{s['ci'][1]:.2f}] on {s['n']} cases"
        if top["tok_s"] is not None:
            text += f", {top['tok_s']:.0f} tok/s" + (" (CPU)" if top.get("cpu") else "")
        detail: dict[str, Any] = {"winner": name, "runner_up": None, "comparison": None}
        if len(ranked) > 1:
            second = ranked[1]
            s2 = second["summary"]
            cmp = stats.compare(top["rows"], second["rows"])
            verdict = {"better": "the difference is significant", "worse": "the runner-up is significantly better on the shared cases", "no_clear_difference": "difference not significant",
                       "no_data": "no shared cases to compare"}[cmp["verdict"]]
            text += f"; runner-up {second['contestant']['name']} {s2['score']:.2f} [{s2['ci'][0]:.2f}-{s2['ci'][1]:.2f}]; {verdict}"
            detail.update(runner_up=second["contestant"]["name"], comparison={"verdict": cmp["verdict"], "p_value": cmp["p_value"], "n": cmp["n"]})
        return text + ".", detail

    # ------------------------------------------------------------------ the table
    def build(self) -> dict[str, Any]:
        policy = self.settings.get("routes.policy")
        now = self.clock()
        tasks: dict[str, Any] = {}
        detail: dict[str, Any] = {}
        for category in TASKS:
            result = self.evaluate(category, policy, now)
            ranked = result["ranked"]
            detail[category] = {"excluded": result["excluded"], "candidates": len(ranked), "warnings": result["warnings"]}
            if not ranked:
                continue
            explain, parts = self.explain(category, ranked)
            top_k = ranked[: int(policy["top_k"])]
            tasks[category] = {"capability": CAPABILITY.get(category, "llm"), "prefer": [self._item(x) for x in top_k], "explain": explain}
            detail[category].update(parts, ranked=[{"contestant": x["contestant"]["id"], "name": x["contestant"]["name"], "last_ts": max((r["ts"] for r in x["rows"]), default=None), "truncated": x["truncated"], **self._item(x)} for x in ranked])
        capabilities: dict[str, Any] = {}
        if "general" in tasks:
            capabilities["llm"] = {"prefer": tasks["general"]["prefer"]}
        if "vision" in tasks:
            capabilities["vision"] = {"prefer": tasks["vision"]["prefer"]}
        doc = {"schema": SCHEMA, "source": "galton", "updated_at": iso_local(now), "tasks": tasks, "capabilities": capabilities}
        return {"doc": doc, "detail": detail}

    @staticmethod
    def diff(current: Optional[dict[str, Any]], proposed: dict[str, Any]) -> dict[str, Any]:
        old_tasks = (current or {}).get("tasks", {})
        new_tasks = proposed.get("tasks", {})
        first = lambda t: ((t.get("prefer") or [{}])[0].get("names") or [None])[0]  # noqa: E731
        added = sorted(set(new_tasks) - set(old_tasks))
        removed = sorted(set(old_tasks) - set(new_tasks))
        changed, same = [], 0
        for task in sorted(set(new_tasks) & set(old_tasks)):
            if first(old_tasks[task]) != first(new_tasks[task]):
                changed.append({"task": task, "from": first(old_tasks[task]), "to": first(new_tasks[task])})
            elif old_tasks[task].get("prefer") != new_tasks[task].get("prefer"):
                changed.append({"task": task, "from": first(old_tasks[task]), "to": first(new_tasks[task]), "numbers_only": True})
            else:
                same += 1
        return {"added": added, "removed": removed, "changed": changed, "unchanged": same, "has_changes": bool(added or removed or changed)}

    def published(self) -> Optional[dict[str, Any]]:
        try:
            data = json.loads(self.path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def get(self) -> dict[str, Any]:
        built = self.build()
        current = self.published()
        return {"path": str(self.path()), "published": current, "proposed": built["doc"], "detail": built["detail"], "diff": self.diff(current, built["doc"]),
                "history": [{"id": p["id"], "ts": p["ts"], "diff": p["diff"], "note": p["note"]} for p in self.store.publications(10)]}

    def publish(self, note: str = "") -> dict[str, Any]:
        built = self.build()
        doc = built["doc"]
        if not doc["tasks"]:
            from .errors import GaltonError
            raise GaltonError("invalid", "nothing_to_publish")
        current = self.published()
        diff = self.diff(current, doc)
        path = self.path()
        write_json_atomic(path, doc)  # temp file + replace with retries: Hoard Link may be reading the table at that moment
        self.store.add_publication(path=str(path), doc=doc, diff=diff, note=note)
        self.emit("galton.routes.updated", {"path": str(path), "tasks": sorted(doc["tasks"]), "changed": [c["task"] for c in diff["changed"]], "added": diff["added"]})
        return {"path": str(path), "doc": doc, "diff": diff, "detail": built["detail"]}
