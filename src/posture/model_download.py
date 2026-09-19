"""Verified, atomic download of Google's version-1 Full pose model."""
from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task?generation=1682642785209422"
)
MODEL_BYTES = 9398198
MODEL_SHA256 = "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1"


def verified(path: Path, size: int = MODEL_BYTES, sha256: str = MODEL_SHA256) -> bool:
    return (path.is_file() and path.stat().st_size == size
            and hashlib.sha256(path.read_bytes()).hexdigest() == sha256)


def fetch_model(path: Path, *, url: str = MODEL_URL, size: int = MODEL_BYTES,
                sha256: str = MODEL_SHA256, opener=urllib.request.urlopen) -> Path:
    if verified(path, size, sha256):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".stack-model-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output, opener(url, timeout=60) as response:
            total = 0
            while block := response.read(65536):
                total += len(block)
                if total > size:
                    raise ValueError("Model download exceeds expected size")
                output.write(block)
        if not verified(temporary, size, sha256):
            raise ValueError("Model verification failed; existing file was preserved")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main() -> None:
    import argparse
    from posture.config import Config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", type=Path)
    args = parser.parse_args()
    destination = args.destination or Config.load().model_path
    if args.destination is None and destination != Config().model_path:
        parser.error("Custom model path configured. Supply its path explicitly to replace it.")
    print(f"Verified model: {fetch_model(destination)}")


if __name__ == "__main__":
    main()
