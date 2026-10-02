"""The web client's sources stay consistent with the backend: every translation key used exists in both languages, the message tables cover the
catalogue of galton_hoard/messages.py, and polling never waits for a visible tab for its first data."""

from __future__ import annotations

import re
from pathlib import Path

from galton_hoard.messages import ERRORS, NOTICES, TEXTS, fields_of

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "client" / "src"
ENTRY = re.compile(r'^\s*([a-z0-9_]+): \[(".*")\],?\s*$', re.M)
I18N = (SRC / "i18n.js").read_text(encoding="utf-8")
MSGS = (SRC / "msgs.js").read_text(encoding="utf-8")
ENTRIES = {**dict(ENTRY.findall(I18N)), **dict(ENTRY.findall(MSGS))}
SOURCES = [p for p in sorted(SRC.rglob("*")) if p.suffix in (".js", ".jsx") and p.name not in ("i18n.js", "msgs.js")]


def parts(raw: str) -> list[str]:
    return re.findall(r'"((?:[^"\\]|\\.)*)"', raw)


def placeholders(text: str) -> list[str]:
    return sorted(re.findall(r"\{([a-z_]+)\}", text))


def test_every_key_used_in_the_client_is_translated():
    missing, used = {}, set()
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        for m in [*re.finditer(r'\bt\(\s*"([a-z][a-z0-9_]*)"', text), *re.finditer(r'\bkey: "(nav_[a-z0-9_]+)"', text)]:
            used.add(m.group(1))
            if m.group(1) not in ENTRIES:
                missing.setdefault(m.group(1), path.name)
        for prefix in re.findall(r"\bt\(`([a-z0-9_]+)\$\{", text):
            assert any(k.startswith(prefix) for k in ENTRIES), prefix
    assert len(used) > 150
    assert not missing, missing


def test_both_languages_are_filled_and_placeholders_match():
    assert len(ENTRIES) > 400
    for key, raw in ENTRIES.items():
        pair = parts(raw)
        assert len(pair) == 2 and all(p.strip() for p in pair), key
        assert placeholders(pair[0]) == placeholders(pair[1]), key


def test_the_message_tables_cover_the_backend_catalogue_exactly():
    expected = {}
    for key, (message, hint) in ERRORS.items():
        expected[f"err_{key}"] = fields_of(message)
        if hint:
            expected[f"hint_{key}"] = fields_of(hint)
    for key, template in TEXTS.items():
        expected[f"msg_{key}"] = fields_of(template)
    for kind, (title, body) in NOTICES.items():
        expected[f"notice_title_{kind}"], expected[f"notice_body_{kind}"] = fields_of(title), fields_of(body)
    defined = set(dict(ENTRY.findall(MSGS)))
    assert defined == set(expected), (sorted(set(expected) - defined), sorted(defined - set(expected)))
    for key, names in expected.items():
        pair = parts(dict(ENTRY.findall(MSGS))[key])
        assert set(placeholders(pair[0])) == names == set(placeholders(pair[1])), key


def test_the_client_formats_messages_through_one_translator():
    for page in ("Ejecutar", "Modelos", "Clasificacion", "Pruebas", "Rutas", "Panel"):
        text = (SRC / "pages" / f"{page}.jsx").read_text(encoding="utf-8")
        assert "t.msg(" in text or "t.notice(" in text or "<Failure" in text, page
    ui = (SRC / "components" / "ui.jsx").read_text(encoding="utf-8")
    assert "t.error(error)" in ui and "export function Failure" in ui and "export function Problem" in ui
    api = (SRC / "api.js").read_text(encoding="utf-8")
    assert "error.key" in api and "problemItems" in api


def test_polling_ticks_when_the_page_becomes_visible():
    hooks = (SRC / "components" / "hooks.js").read_text(encoding="utf-8")
    assert 'addEventListener("visibilitychange"' in hooks and 'removeEventListener("visibilitychange"' in hooks
    body = hooks.split("useEffect(")[1]
    assert body.index("    run();") < body.index("setInterval"), "the first call comes before any timer"
    assert "const run = () => { if (!stopped) ref.current(); };" in hooks, "the first call does not look at document.hidden"
    assert "document.hidden" in hooks.split("const tick")[1].split("\n")[0] and "document.hidden" in hooks.split("const onVisible")[1].split("\n")[0]


def test_the_app_loads_the_dashboard_at_once_and_when_the_tab_returns():
    app = (SRC / "App.jsx").read_text(encoding="utf-8")
    effect = app.split("refreshDash();\n    api.health()")[0].rsplit("useEffect(", 1)[1]
    assert effect.strip() == "() => {", "refreshDash() is the first thing the effect does, whatever the visibility"
    assert 'document.addEventListener("visibilitychange", onVisible)' in app


def test_the_run_page_continues_an_interrupted_run_and_links_to_the_earlier_one():
    page = (SRC / "pages" / "Ejecutar.jsx").read_text(encoding="utf-8")
    assert 'api.call("run_resume"' in page and 'RESUMABLE.has(run.state) && !run.discarded' in page and "run.continues" in page and "key={param}" in page
    assert '"failed", "cancelled"' in (SRC / "meta.js").read_text(encoding="utf-8")
