from dataclasses import replace

import pytest

from posture.config import Config
from posture.setup import camera_config_text, choose_cameras, save_setup
from posture.scoring import Baseline
from posture.store.db import connect, migrate
from posture.store.queries import active_baseline, save_baseline
from posture.types import PostureMetrics


def test_camera_edit_preserves_advanced_settings_and_comments():
    original = '# keep this\ngemini_enabled = false\napi_timeout_s = 45\n\n[cameras]\nfront = 0\n'
    updated = camera_config_text(original, {"side": 3})
    assert '# keep this\ngemini_enabled = false\napi_timeout_s = 45' in updated
    assert 'side = 3' in updated
    assert 'front = 0' not in updated


def test_inline_camera_table_fails_before_rewriting():
    with pytest.raises(ValueError, match="inline"):
        camera_config_text('cameras = { front = 0 }\n', {"side": 1})


def test_retry_reprobes_without_saving_images():
    answers = iter(["r", "0", "front"])
    probes, previews = [], []
    def discover():
        probes.append(1)
        return () if len(probes) == 1 else (0,)
    roles = choose_cameras(ask=lambda _: next(answers), discover=discover,
                           preview=previews.append, say=lambda _: None)
    assert roles == {"front": 0}
    assert len(probes) == 2
    assert previews == [0]


def test_cancel_camera_selection_makes_no_state():
    with pytest.raises(KeyboardInterrupt):
        choose_cameras(ask=lambda _: "q", discover=lambda: (0,),
                       preview=lambda _: pytest.fail("should not preview"), say=lambda _: None)


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_setup_save_failure_restores_old_config_and_baseline(tmp_path, monkeypatch, failure):
    config_path = tmp_path / "config.toml"
    config_path.write_text('gemini_enabled = false\n[cameras]\nfront = 0\n')
    before = config_path.read_bytes()
    config = replace(Config(), db_path=tmp_path / "history.db", camera_indices={"side": 1})
    conn = connect(config.db_path)
    migrate(conn)
    old = save_baseline(conn, PostureMetrics(cva_deg=60), "old")
    conn.close()
    def fail(*a, **k):
        raise failure("interrupted save")
    monkeypatch.setattr("posture.setup.save_baseline", fail)
    candidate = Baseline(0, "", "new", PostureMetrics(cva_deg=65))
    with pytest.raises(failure):
        save_setup(config, config_path, before.decode(), candidate)
    assert config_path.read_bytes() == before
    conn = connect(config.db_path)
    assert active_baseline(conn).id == old.id
    conn.close()


def test_successful_setup_keeps_opt_in_false(tmp_path):
    path = tmp_path / "config.toml"
    config = replace(Config(), db_path=tmp_path / "history.db", camera_indices={"side": 2})
    candidate = Baseline(0, "", "new", PostureMetrics(cva_deg=65), facing_sign=1)
    save_setup(config, path, f'db_path = "{config.db_path}"\n', candidate)
    restored = Config.load(path)
    assert restored.gemini_enabled is False
    assert restored.camera_indices == {"side": 2}
    conn = connect(config.db_path)
    assert active_baseline(conn).metrics.cva_deg == 65
    conn.close()
