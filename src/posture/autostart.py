"""Optional login launch. Called explicitly after a successful manual check."""
from __future__ import annotations

import os
import plistlib
import sqlite3
import subprocess
from pathlib import Path

from posture.config import Config
from posture.instance import instance_lock

LABEL = "com.stack-posture.stack"


def launchagent_bytes(project: Path, config: Config) -> bytes:
    return plistlib.dumps({
        "Label": LABEL,
        "ProgramArguments": [str(project / "run.sh")],
        "WorkingDirectory": str(project),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 60,
        "StandardOutPath": str(config.log_path.resolve().parent / "launch.log"),
        "StandardErrorPath": str(config.log_path.resolve().parent / "launch_error.log"),
    })


def require_successful_check(config: Config) -> None:
    problem = "Run Stack manually and use Check Now successfully before adding login launch."
    if not config.db_path.is_file():
        raise ValueError(problem)
    conn = sqlite3.connect(config.db_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        if not conn.execute("SELECT 1 FROM checks WHERE outcome = 'OK' LIMIT 1").fetchone():
            raise ValueError(problem)
    except sqlite3.Error as exc:
        raise ValueError(problem) from exc
    finally:
        conn.close()


def install(project: Path) -> None:
    project = project.resolve()
    config = Config.load()
    if not (project / ".venv/bin/python").is_file() or not config.model_path.is_file():
        raise ValueError("Finish Set Up Stack.command first.")
    require_successful_check(config)
    plist = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
    domain = f"gui/{os.getuid()}"
    # Do not install a duplicate alongside the foreground app or calibration.
    with instance_lock():
        plist.parent.mkdir(parents=True, exist_ok=True)
        config.log_path.parent.mkdir(parents=True, exist_ok=True)
        plist.write_bytes(launchagent_bytes(project, config))
        subprocess.run(["launchctl", "bootout", f"{domain}/{LABEL}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Release before bootstrap: the newly launched app must take this same lock.
    subprocess.run(["launchctl", "enable", f"{domain}/{LABEL}"], check=True)
    subprocess.run(["launchctl", "bootstrap", domain, str(plist)], check=True)
    print("Login launch registered. Stack opens Idle; choose Start Monitoring in its menu.\n"
          "A past successful check does not guarantee background camera permission.\n"
          "Verify Check Now after login. Keep this project folder in place.\n"
          "Remove login launch with scripts/uninstall_autostart.sh.")


if __name__ == "__main__":
    import sys
    try:
        install(Path(sys.argv[1]))
    except Exception as exc:
        print(f"Could not install login launch: {exc}", file=sys.stderr)
        raise SystemExit(1)
