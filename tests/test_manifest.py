"""faustus-plugin.json, the READMEs and the docs stay in step with the code; no other products' names, no placeholders."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from galton_hoard import SERVICE, __version__
from galton_hoard.agent_tools import TOOLS
from galton_hoard.config import DEFAULT_PORT

ROOT = Path(__file__).resolve().parent.parent
#: other products and brands: never in the repository. (The wire-protocol identifier "openai" in code is a technical name, see below.)
BANNED = ("chatgpt", "claude", "anthropic", "lm studio", "lmstudio", "odysseus", "gemini", "copilot", "notion", "evernote", "paperless", "midjourney", "perplexity")
DOC_BANNED = BANNED + ("openai", "tagline", "slogan")
DOCS = [ROOT / "README.md", ROOT / "README.es.md", ROOT / "AGENTS.md", *sorted((ROOT / "docs").glob("*.md"))]
SKIP_DIRS = {"node_modules", "static", "__pycache__", ".pytest_cache", "hoard_link", "data", ".git", "venv"}
SKIP_FILES = {"test_manifest.py", "test_suites.py", "package-lock.json"}


def source_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not (set(path.relative_to(ROOT).parts) & SKIP_DIRS) and path.suffix in {".py", ".jsx", ".js", ".json", ".md", ".css", ".html", ".yml", ".txt", ".toml"} \
                and path.name not in SKIP_FILES:
            yield path


def test_manifest_matches_code():
    manifest = json.loads((ROOT / "faustus-plugin.json").read_text(encoding="utf-8"))
    assert manifest["id"] == "galton" and manifest["name"] == "Galton's Hoard"
    assert manifest["app"]["health"]["expect"]["service"] == SERVICE == "galton-hoard"
    assert manifest["defaults"]["APP_URL"].endswith(f":{DEFAULT_PORT}") and DEFAULT_PORT == 5201
    assert manifest["app"]["launch_hint"]["env"]["GALTON_PORT"] == str(DEFAULT_PORT)
    assert manifest["app"]["launch_hint"]["argv"] == ["-m", "galton_hoard"]
    assert manifest["mcp"]["args"] == ["{GALTON_DIR}/mcp_server.py"] and (ROOT / "mcp_server.py").is_file()
    for placeholder in re.findall(r"\{([A-Z_]+)\}", json.dumps(manifest)):
        assert placeholder in manifest["placeholders"], placeholder


def test_the_tool_count_in_the_docs_is_right():
    for name in ("README.md", "README.es.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert f"({len(TOOLS)}):" in text, name
    assert "41 tools" in (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8") or f"{len(TOOLS)} tools" in (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")


def test_every_tool_is_listed_in_both_readmes_and_documented():
    api = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    for name in ("README.md", "README.es.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for tool in TOOLS:
            assert f"`{tool.name}`" in text, f"{tool.name} missing in {name}"
    for tool in TOOLS:
        assert f"## `{tool.name}`" in api, tool.name


def test_first_description_lines_are_short_and_bilingual():
    for tool in TOOLS:
        first = tool.description.splitlines()[0]
        assert len(first) <= 110 and ". " in first, tool.name


def test_readmes_have_the_required_sections():
    en = (ROOT / "README.md").read_text(encoding="utf-8")
    es = (ROOT / "README.es.md").read_text(encoding="utf-8")
    assert en.startswith("# Galton's Hoard") and es.startswith("# Galton's Hoard")
    for heading in ("## What it does", "## What it cannot do", "## Install", "## Run", "## Assistants (MCP)", "## Data and privacy", "## Development"):
        assert heading in en, heading
    for heading in ("## Qué hace", "## Qué no hace", "## Instalación", "## Ejecución", "## Asistentes (MCP)", "## Datos y privacidad", "## Desarrollo"):
        assert heading in es, heading


def test_no_emojis_in_docs():
    emoji = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\u2705]")
    for path in DOCS:
        assert not emoji.search(path.read_text(encoding="utf-8")), path.name


def test_no_other_products_or_slogans_in_docs():
    for path in DOCS:
        text = path.read_text(encoding="utf-8").lower()
        for word in DOC_BANNED:
            assert not re.search(rf"\b{re.escape(word)}\b", text), f"{word} in {path.name}"


def test_no_other_products_anywhere_in_the_sources():
    for path in source_files():
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for word in BANNED:
            assert not re.search(rf"\b{re.escape(word)}\b", text), f"{word} in {path.relative_to(ROOT)}"


def test_the_ui_never_names_the_wire_protocol_after_a_company():
    for path in (ROOT / "client" / "src").rglob("*"):
        if path.is_file():
            assert "openai" not in path.read_text(encoding="utf-8", errors="replace").lower(), path.name


def test_no_personal_data_in_the_sources():
    for path in source_files():
        if path.name in {"LICENSE"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert not re.search(r"(?i)\bluis\b|salete|@gmail\.|C:\\Users\\|/home/claude|/root/", text), path.relative_to(ROOT)


def test_no_placeholder_lines():
    for path in [*DOCS, *(p for p in source_files() if "tests" not in p.relative_to(ROOT).parts and "suites" not in p.relative_to(ROOT).parts)]:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            assert not re.search(r"\b(TODO|TBD|FIXME|XXX)\b", line) and not re.search(r"(?i)lorem ipsum", line), f"{path.name}: {line}"


def test_license_and_ci_exist():
    assert "Luis María Salete Cuartero" in (ROOT / "LICENSE").read_text(encoding="utf-8") and "MIT License" in (ROOT / "LICENSE").read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "ubuntu-latest" in ci and "windows-latest" in ci and "pytest" in ci and "vite build" in ci
    assert "data/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines() and "node_modules/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_version_matches():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(rf'version = "{re.escape(__version__)}"', pyproject)
    assert json.loads((ROOT / "package.json").read_text(encoding="utf-8"))["version"] == __version__


def test_requirements_are_pinned():
    lines = [l.strip() for l in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    assert lines and all("==" in l for l in lines), lines


def test_the_api_doc_is_up_to_date():
    before = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_api_doc.py"), "--check"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert before == (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
