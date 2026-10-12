"""A manual edit sends only the fields the user changed, so what an assistant changed in the meantime survives."""
from __future__ import annotations

import pytest

from conftest import tool
from helpers import add_gguf
from galton_hoard.errors import GaltonError
from galton_hoard.settings import ROUTES_POLICY_DEFAULT


def test_model_update_with_one_field_keeps_the_others(svc):
    a = add_gguf(svc, "alfa-q4")
    tool(svc, "model_update", model=a["id"], add_aliases=["alfa:latest"])          # what an assistant did meanwhile
    out = tool(svc, "model_update", model=a["id"], name="Alfa 4 bits")["model"]     # what the user saved: only the name
    assert out["name"] == "Alfa 4 bits" and "alfa:latest" in out["aliases"] and out["enabled"] is True


def test_case_update_with_one_field_keeps_prompt_checker_and_other_fields(svc):
    s = tool(svc, "suite_create", name="Mis pruebas", category="reasoning")["suite"]
    c = tool(svc, "case_add", suite=s["id"], title="Capital", prompt="Capital de Francia?", system="Se breve", expected="Paris", tags=["geo"], notes="n1")["case"]
    tool(svc, "case_update", case=c["id"], tags=["geo", "easy"])                    # the assistant tags it
    out = tool(svc, "case_update", case=c["id"], notes="n2")["case"]                # the user saved only the notes
    assert out["notes"] == "n2" and out["tags"] == ["geo", "easy"] and out["weight"] == 1.0
    full = tool(svc, "case_get", case=c["id"])["case"]
    assert full["prompt"]["text"] == "Capital de Francia?" and full["prompt"]["system"] == "Se breve" and full["checker"]["type"] == "contains"
    only_system = tool(svc, "case_update", case=c["id"], system="")["case"]         # clearing one field does not touch the prompt text
    assert tool(svc, "case_get", case=only_system["id"])["case"]["prompt"] == {"text": "Capital de Francia?"}


def test_suite_update_with_one_field_keeps_the_others(svc):
    s = tool(svc, "suite_create", name="Mis pruebas", category="reasoning", description="d", max_tokens=300)["suite"]
    out = tool(svc, "suite_update", suite=s["id"], description="otra")["suite"]
    assert out["name"] == "Mis pruebas" and out["category"] == "reasoning" and out["max_tokens"] == 300 and out["description"] == "otra"


def test_settings_set_merge_changes_only_the_given_policy_fields(svc):
    tool(svc, "settings_set", values={"routes.policy": {"days": 7, "min_cases": 3}})        # what the assistant set
    tool(svc, "settings_set", values={"routes.policy": {"top_k": 2}})
    assert svc.settings.get("routes.policy")["days"] == ROUTES_POLICY_DEFAULT["days"]      # without merge: the rest goes back to the defaults
    tool(svc, "settings_set", values={"routes.policy": {"days": 7, "min_cases": 3}})
    tool(svc, "settings_set", values={"routes.policy": {"top_k": 2}}, merge=True)
    policy = svc.settings.get("routes.policy")
    assert policy["days"] == 7 and policy["min_cases"] == 3 and policy["top_k"] == 2


def test_settings_set_merge_still_rejects_unknown_fields(svc):
    with pytest.raises(GaltonError):
        tool(svc, "settings_set", values={"routes.policy": {"colour": 1}}, merge=True)
