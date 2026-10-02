"""Regenerate docs/API.md from the tool catalogue: python scripts/gen_api_doc.py  (--check: fail when the file is out of date)"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from galton_hoard.agent_tools import TOOLS  # noqa: E402


def main() -> int:
    lines = ["# Galton's Hoard — agent tools", "",
             "Every tool is served by the app at `GET /api/agent/tools` and `POST /api/agent/call` (Bearer token from `data/mcp-token`), "
             "by the stdio bridge `mcp_server.py`, and to the bundled UI through `POST /api/ui/call`. The argument tables are generated "
             "from the code (`python scripts/gen_api_doc.py`).", ""]
    for t in TOOLS:
        first, _, rest = t.description.partition("\n")
        lines += [f"## `{t.name}`", "", first]
        if rest.strip():
            lines += ["", rest.strip()]
        schema = t.input_model.model_json_schema()
        props, required = schema.get("properties", {}), set(schema.get("required", []))
        flags = ", ".join(k for k, v in t.annotations.items() if v)
        lines += ["", f"Annotations: {flags or 'none'}."]
        if props:
            lines += ["", "| Argument | Required | Description |", "|---|---|---|"]
            for name, spec in props.items():
                kind = spec.get("type") or "/".join(x.get("type", "") for x in spec.get("anyOf", []) if x.get("type"))
                if "enum" in spec:
                    kind = " \\| ".join(map(str, spec["enum"]))
                desc = (spec.get("description") or "").replace("|", "\\|").replace("\n", " ")
                lines.append(f"| `{name}` ({kind}) | {'yes' if name in required else 'no'} | {desc} |")
        lines.append("")
    lines += ["## REST routes for the UI", "", "- `GET /api/health`, `GET /api/status`",
              "- `GET /api/dashboard` — the routing table as it stands, the run in progress, the queue, notices, what was never measured or went stale, the GPUs "
              "and the scheduler.",
              "- `POST /api/dashboard/visit` — marks the notices seen.",
              "- `GET /api/runs/{id}/events?after=N` — the run (state, progress per model) and the results written after result id `N`, without the answer texts; "
              "the run page polls it while the run is live.",
              "- `GET /api/vision/{case}/{n}.png` — image `n` (from 1) of a case: a stored image or the one a vision case generates at run time; same-origin only.",
              "- `POST /api/ui/call` `{name, arguments}` — any tool above, uncapped (the UI is local).",
              "- `GET /manifest.webmanifest`, `GET /sw.js` — installable app.", ""]
    target = ROOT / "docs" / "API.md"
    text = "\n".join(lines)
    if "--check" in sys.argv:
        current = target.read_text(encoding="utf-8") if target.is_file() else ""
        if current != text:
            print("docs/API.md is out of date: run python scripts/gen_api_doc.py")
            return 1
        print(f"{len(TOOLS)} tools, docs/API.md is current")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(f"{len(TOOLS)} tools")
    return 0


if __name__ == "__main__":
    sys.exit(main())
