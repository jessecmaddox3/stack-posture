"""One app or calibration process at a time, without killing another process."""
from contextlib import contextmanager
from pathlib import Path
import fcntl

from posture.config import DATA_DIR


@contextmanager
def instance_lock(path: Path | None = None):
    path = path or DATA_DIR / "stack.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Stack is already open. Choose Quit in its menu, then retry.") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
