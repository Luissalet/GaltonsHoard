"""Stdio MCP bridge for Galton's Hoard.

It never opens the database: every tool call is proxied to the running app (`POST /api/agent/call`) with the Bearer token from
`<DATA_DIR>/mcp-token`. The tool list is fetched from `GET /api/agent/tools` (refreshed while the bridge runs), so the bridge and the app can
never disagree. When nothing answers, the bridge starts the app itself (`python -m galton_hoard`, detached, on the port of GALTON_URL) and waits
for it; GALTON_BRIDGE_AUTOSTART=0 turns that off. The bridge itself is the shared catalogue bridge of Hoard Link. The tools that wait for a run
(`wait_s` up to 600 s) or for the judge publish their waiting time in the catalogue; every other call may take up to 660 s, as it always could.
"""

from __future__ import annotations

import sys

from galton_hoard.hoard_link.bridge import CatalogBridge


def main() -> int:
    CatalogBridge(app="galton", service="galton-hoard", package="galton_hoard", default_port=5201, data_dir_env="GALTON_DATA_DIR",
                  title="Galton's Hoard", root=__file__, default_timeout=660.0).run_bridge()
    return 0


if __name__ == "__main__":
    sys.exit(main())
