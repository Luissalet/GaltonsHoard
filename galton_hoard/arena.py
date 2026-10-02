"""Blind pairwise comparison: two anonymous answers to the same prompt, a vote, and ratings per category.

Pairs are built from answers already stored by runs, so the arena costs no model time. Open-ended cases (no automatic checker, or a judge)
come first because that is where a human opinion adds something; any other case can also be compared.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from . import stats
from .errors import GaltonError

VOTES = ("a", "b", "tie", "both_bad")


class Arena:
    def __init__(self, store: Any, settings: Any, rng: Optional[random.Random] = None):
        self.store, self.settings = store, settings
        self.rng = rng or random.Random()

    def _candidates(self, category: str) -> list[dict[str, Any]]:
        sql = ("SELECT r.case_id AS case_id, s.category AS category, c.checker AS checker, COUNT(DISTINCT r.contestant_id) AS models FROM results r "
               "JOIN cases c ON c.id = r.case_id JOIN suites s ON s.id = c.suite_id "
               "WHERE r.skipped = 0 AND r.error = '' AND r.output != '' AND r.run_id NOT IN (SELECT id FROM runs WHERE discarded = 1) ")
        params: list[Any] = []
        if category:
            sql += "AND s.category = ? "
            params.append(category)
        sql += "GROUP BY r.case_id HAVING models >= 2"
        return [dict(r) for r in self.store.db.query(sql, params)]

    @staticmethod
    def _open_ended(checker_json: str) -> bool:
        return '"type": "none"' in checker_json or '"type": "judge"' in checker_json or '"type":"judge"' in checker_json or '"type":"none"' in checker_json

    def next_pair(self, category: str = "") -> Optional[dict[str, Any]]:
        """A pair nobody has voted on yet, with the model names hidden, or ``None`` when every pair has been compared."""
        voted = self.store.voted_pairs()
        cands = self._candidates(category)
        self.rng.shuffle(cands)
        cands.sort(key=lambda c: not self._open_ended(c["checker"]))
        for cand in cands:
            latest: dict[str, dict[str, Any]] = {}
            for row in self.store.results(case_id=cand["case_id"], limit=500, live_only=True):
                if row["skipped"] or row["error"] or not row["output"].strip():
                    continue
                latest[row["contestant_id"]] = row  # results come oldest first: the last write wins
            ids = list(latest)
            self.rng.shuffle(ids)
            for i, x in enumerate(ids):
                for y in ids[i + 1:]:
                    if (cand["case_id"], x, y) in voted or latest[x]["output"].strip() == latest[y]["output"].strip():
                        continue
                    a, b = (x, y) if self.rng.random() < 0.5 else (y, x)
                    pid = self.store.add_arena_pair(case_id=cand["case_id"], category=cand["category"], a_result=latest[a]["id"], b_result=latest[b]["id"], a_contestant=a, b_contestant=b)
                    case = self.store.case(cand["case_id"])
                    prompt = case["prompt"]
                    text = prompt.get("text") or "\n".join(str(m.get("content", "")) for m in prompt.get("messages", []) if m.get("role") == "user")
                    if prompt.get("generate"):
                        text = f"[{case['title']}] (generated prompt)"
                    return {"pair": pid, "case": cand["case_id"], "title": case["title"], "category": cand["category"], "prompt": text,
                            "a": {"text": latest[a]["output"]}, "b": {"text": latest[b]["output"]}}
        return None

    def vote(self, pair_id: str, vote: str) -> dict[str, Any]:
        if vote not in VOTES:
            raise GaltonError("invalid", "arena_bad_vote", options=VOTES)
        pair = self.store.arena_pair(pair_id)
        if pair is None:
            raise GaltonError("not_found", "arena_no_pair", pair=pair_id)
        if pair["voted"]:
            raise GaltonError("conflict", "arena_already_voted")
        self.store.add_vote(pair_id=pair_id, case_id=pair["case_id"], category=pair["category"], a=pair["a_contestant"], b=pair["b_contestant"], vote=vote)
        self.store.mark_pair_voted(pair_id)
        names = {c: self.store.contestant(c)["name"] for c in (pair["a_contestant"], pair["b_contestant"])}
        return {"pair": pair_id, "vote": vote, "a": names[pair["a_contestant"]], "b": names[pair["b_contestant"]], "ratings": self.ratings(pair["category"])}

    def ratings(self, category: str = "") -> list[dict[str, Any]]:
        votes = [(v["a"], v["b"], v["vote"]) for v in self.store.votes(category)]
        rows = stats.arena_ratings(votes, min_votes=int(self.settings.get("arena.min_votes")))
        for r in rows:
            try:
                r["name"] = self.store.contestant(r["contestant"])["name"]
            except GaltonError:
                r["name"] = r["contestant"]
        return rows

    def categories_with_votes(self) -> list[str]:
        return sorted({v["category"] for v in self.store.votes() if v["category"]})
