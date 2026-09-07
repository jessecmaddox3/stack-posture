#!/usr/bin/env python3
"""Print a troubleshooting bundle. Pipe to pbcopy to put it on the clipboard.

    uv run python scripts/diagnostics.py | pbcopy
    uv run python scripts/diagnostics.py > ~/Desktop/posture-diagnostics.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from posture import cameras  # noqa: E402
from posture.config import Config  # noqa: E402
from posture.diagnostics import build_diagnostics, read_log_tail  # noqa: E402
from posture.store.db import connect, migrate  # noqa: E402


def main() -> int:
    config = Config.load()
    conn = connect(config.db_path)
    migrate(conn)
    try:
        print(build_diagnostics(conn, config, read_log_tail(config.log_path),
                                cameras.discover()))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
