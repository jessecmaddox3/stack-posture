import plistlib
from dataclasses import replace

import pytest

from posture.autostart import launchagent_bytes, require_successful_check
from posture.config import Config
from posture.instance import instance_lock
from posture.store.db import connect, migrate


def test_paths_with_xml_characters_roundtrip(tmp_path):
    project = tmp_path / "Stack & Friends <demo>"
    config = replace(Config(), log_path=tmp_path / "private & local" / "stack.log")
    data = plistlib.loads(launchagent_bytes(project, config))
    assert data["ProgramArguments"] == [str(project / "run.sh")]
    assert data["WorkingDirectory"] == str(project)
    assert data["RunAtLoad"] is True


def test_autostart_checks_configured_database_without_creating_it(tmp_path):
    config = replace(Config(), db_path=tmp_path / "custom.db")
    with pytest.raises(ValueError, match="Check Now"):
        require_successful_check(config)
    assert not config.db_path.exists()
    conn = connect(config.db_path)
    migrate(conn)
    conn.close()
    with pytest.raises(ValueError, match="Check Now"):
        require_successful_check(config)


def test_single_instance_lock_blocks_another_app_and_releases(tmp_path):
    path = tmp_path / "stack.lock"
    with instance_lock(path):
        with pytest.raises(RuntimeError, match="already open"):
            with instance_lock(path):
                pytest.fail("second process acquired lock")
    with instance_lock(path):
        pass
