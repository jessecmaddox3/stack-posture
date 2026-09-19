import hashlib
import io

import pytest

from posture.model_download import fetch_model


CONTENT = b"synthetic model, not weights"
DIGEST = hashlib.sha256(CONTENT).hexdigest()


def fetch(path, payload=CONTENT):
    return fetch_model(path, url="https://example.invalid/model", size=len(CONTENT),
                       sha256=DIGEST, opener=lambda *a, **k: io.BytesIO(payload))


def test_verified_cache_does_not_download(tmp_path):
    path = tmp_path / "model.task"
    path.write_bytes(CONTENT)
    fetch_model(path, size=len(CONTENT), sha256=DIGEST,
                opener=lambda *a, **k: pytest.fail("cache caused download"))
    assert path.read_bytes() == CONTENT


@pytest.mark.parametrize("payload", [b"short", b"x" * len(CONTENT)])
def test_invalid_download_preserves_existing_model(tmp_path, payload):
    path = tmp_path / "model.task"
    path.write_bytes(b"previous working model")
    with pytest.raises(ValueError):
        fetch(path, payload)
    assert path.read_bytes() == b"previous working model"
    assert list(tmp_path.iterdir()) == [path]


def test_corrupt_cache_is_atomically_replaced(tmp_path):
    path = tmp_path / "model.task"
    path.write_bytes(b"x" * len(CONTENT))
    fetch(path)
    assert path.read_bytes() == CONTENT
    assert list(tmp_path.iterdir()) == [path]


def test_network_failure_preserves_old_model(tmp_path):
    path = tmp_path / "model.task"
    path.write_bytes(b"old")
    def fail(*a, **k):
        raise OSError("offline")
    with pytest.raises(OSError):
        fetch_model(path, opener=fail)
    assert path.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [path]
