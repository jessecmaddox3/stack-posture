#!/usr/bin/env python3
"""Build the source ZIP from the exact Git index being reviewed, never loose files."""
from __future__ import annotations

import hashlib
import subprocess
import zipfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "dist"
    output.mkdir(exist_ok=True)
    records = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root)
    with zipfile.ZipFile(output / "Stack.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for record in records.split(b"\0"):
            if not record:
                continue
            metadata, name = record.split(b"\t", 1)
            mode, oid, stage = metadata.decode().split()
            if stage != "0" or mode not in ("100644", "100755"):
                raise ValueError("Release index contains a conflict, symlink or submodule")
            path = name.decode()
            parts = Path(path).parts
            if any(part in (".env", "data", ".venv", "inbox", "captures") for part in parts):
                raise ValueError("Private-state path in release index")
            blob = subprocess.check_output(["git", "cat-file", "blob", oid], cwd=root)
            info = zipfile.ZipInfo("Stack/" + path, date_time=(2026, 9, 18, 0, 0, 0))
            info.create_system = 3
            info.external_attr = int(mode, 8) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, blob)
    for path in sorted(output.iterdir()):
        if path.suffix in (".zip", ".whl") or path.name.endswith(".tar.gz"):
            print(hashlib.sha256(path.read_bytes()).hexdigest(), path.name)


if __name__ == "__main__":
    main()
