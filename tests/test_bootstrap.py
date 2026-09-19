import shutil
import subprocess
from pathlib import Path


def command(path, body):
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


def scaffold(tmp_path, architecture):
    root = tmp_path / "Stack & Friends"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(Path(__file__).resolve().parents[1] / "scripts/bootstrap.sh", root / "scripts")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    command(binaries / "uname", f'if [ "$1" = -s ]; then echo Darwin; else echo {architecture}; fi\n')
    command(binaries / "sw_vers", 'echo 13.0\n')
    command(binaries / "uv", 'printf "%s\\n" "$@" > uv-invocation.txt\n')
    (root / ".venv/bin").mkdir(parents=True)
    command(root / ".venv/bin/python", 'printf "%s\\n" "$@" > python-invocation.txt\n')
    return root, {"HOME": str(tmp_path), "PATH": f"{binaries}:/usr/bin:/bin"}


def test_unsupported_mac_stops_before_dependencies(tmp_path):
    root, env = scaffold(tmp_path, "x86_64")
    result = subprocess.run(["/bin/bash", str(root / "scripts/bootstrap.sh")],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 1
    assert "Apple Silicon" in result.stdout
    assert not (root / "uv-invocation.txt").exists()


def test_setup_uses_locked_dependencies_from_its_own_folder(tmp_path):
    root, env = scaffold(tmp_path, "arm64")
    result = subprocess.run(["/bin/bash", str(root / "scripts/bootstrap.sh")],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (root / "uv-invocation.txt").read_text().splitlines() == [
        "sync", "--locked", "--python", "3.12"]
    assert (root / "python-invocation.txt").read_text().splitlines() == ["-m", "posture.setup"]
    assert not (tmp_path / "uv-invocation.txt").exists()
