"""Guided local camera selection and calibration. No keys, saved frames or uploads."""
from __future__ import annotations

import os
import re
import tempfile
import time
import tomllib
from dataclasses import replace
from pathlib import Path

from posture.config import CONFIG_PATH, Config
from posture.scoring import Baseline
from posture.store.db import connect, migrate
from posture.store.queries import save_baseline


def camera_config_text(original: str, roles: dict[str, int]) -> str:
    """Preserve settings and comments; edit only the conventional cameras table."""
    replace(Config(), camera_indices=roles).validate()
    parsed = tomllib.loads(original)
    if parsed.get("cameras") == roles:
        return original
    header = re.search(r"(?m)^\s*\[cameras\]\s*(?:#.*)?$", original)
    block = "[cameras]\n" + "".join(f"{role} = {index}\n" for role, index in roles.items())
    if header:
        following = re.search(r"(?m)^\s*\[", original[header.end():])
        end = header.end() + following.start() if following else len(original)
        updated = original[:header.start()] + block + "\n" + original[end:]
    elif "cameras" in parsed:
        raise ValueError("Keep or manually edit your inline cameras table before changing cameras.")
    else:
        prefix = original.rstrip() + "\n" if original.strip() else "gemini_enabled = false\n"
        updated = prefix + "\n" + block
    if tomllib.loads(updated).get("cameras") != roles:
        raise ValueError("Could not update cameras safely; settings were preserved")
    return updated


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".stack-settings-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(content)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def save_setup(config: Config, path: Path, original: str, baseline: Baseline) -> None:
    """Commit only a successful calibration; roll back settings on database failure."""
    updated = camera_config_text(original, config.camera_indices)
    existed = path.exists()
    conn = connect(config.db_path)
    try:
        migrate(conn)
        try:
            atomic_text(path, updated)
            save_baseline(conn, baseline.metrics, baseline.label, baseline.facing_sign)
        except BaseException:
            conn.rollback()
            if existed:
                atomic_text(path, original)
            else:
                path.unlink(missing_ok=True)
            raise
    finally:
        conn.close()


def fresh_discovery():
    from posture import cameras
    cameras.discover.cache_clear()
    return cameras.discover()


def preview_camera(index: int) -> None:
    import cv2
    from posture import cameras
    frame = cameras.capture(index)
    if frame is None:
        print("No frame. Check camera permission, close other camera apps, and retry.")
        return
    print(f"Camera {index}: press any key in the preview window to close it.")
    title = f"Stack camera {index} (not saved)"
    try:
        cv2.imshow(title, frame)
        while cv2.waitKey(100) < 0:
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()


def choose_cameras(*, ask=input, say=print, discover=fresh_discovery,
                   preview=preview_camera) -> dict[str, int]:
    while True:
        found = discover()
        if not found:
            say("No camera returned a frame. Allow Terminal/Python in System Settings > "
                "Privacy & Security > Camera, then retry.")
            if ask("[r] Retry, [q] Quit: ").strip().lower() == "r":
                continue
            raise KeyboardInterrupt
        say("Available camera numbers: " + ", ".join(map(str, found)))
        answer = ask("Camera number to preview (or q to quit): ").strip().lower()
        if answer == "q":
            raise KeyboardInterrupt
        if not answer.isdecimal() or int(answer) not in found:
            say("Choose one of the camera numbers above.")
            continue
        index = int(answer)
        preview(index)
        role = ask("Use this as [front], [side], or [retry]? ").strip().lower()
        if role not in ("front", "side"):
            continue
        roles = {role: index}
        others = [other for other in found if other != index]
        if others:
            extra = ask("Optional second camera number (Enter for just one): ").strip()
            if extra:
                if not extra.isdecimal() or int(extra) not in others:
                    say("The second camera must be a different available camera.")
                    continue
                preview(int(extra))
                if ask("Use this second view? [y/N]: ").strip().lower() == "y":
                    roles["side" if role == "front" else "front"] = int(extra)
        return roles


def calibrate_candidate(config: Config) -> Baseline | None:
    """Use an in-memory database until enough usable numeric samples exist."""
    from posture.baseline import run_calibration
    from posture.landmarks import PoseDetector
    conn = connect(Path(":memory:"))
    try:
        migrate(conn)
        with PoseDetector(config.model_path) as detector:
            return run_calibration(config, detector, conn, label="Desk baseline")
    finally:
        conn.close()


def main() -> int:
    from posture.instance import instance_lock
    from posture.model_download import fetch_model
    try:
        with instance_lock():
            config = Config.load()
            original = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else ""
            print("\nSet up Stack\n")
            print("This step uses your camera locally. Previews stay in memory. "
                  "Calibration saves numbers, never images. Gemini is not used during setup.")
            if input("Continue with camera setup? [y/N]: ").strip().lower() != "y":
                return 0
            if config.model_path == Config().model_path:
                print("Checking the local pose model (downloads 9.4 MB if needed)...")
                fetch_model(config.model_path)
            elif not config.model_path.is_file():
                raise ValueError("Your custom model is missing. Restore it or change model_path.")
            else:
                print("Keeping your custom model, which is not the verified bundled model pin.")
            if original and input("Keep existing camera choices? [Y/n]: ").lower() != "n":
                roles = config.camera_indices
            else:
                roles = choose_cameras()
            config = replace(config, camera_indices=roles)
            # Fail before calibration for an unusual config we cannot safely edit.
            camera_config_text(original, roles)
            print("Front view measures alignment. Side view also needs ear, shoulder and hip "
                  "visible to measure forward head position and lean.")
            while True:
                if input("Sit comfortably for your baseline. Ready for 10 seconds? [y/N]: ") \
                        .strip().lower() != "y":
                    return 0
                for count in (3, 2, 1):
                    print(f"{count}...")
                    time.sleep(1)
                baseline = calibrate_candidate(config)
                if baseline is not None and baseline.metrics.available():
                    break
                print("Too few usable measurements. Improve framing/lighting and retry.")
            save_setup(config, CONFIG_PATH, original, baseline)
            print("\nSaved your numeric baseline. Measured: "
                  + ", ".join(baseline.metrics.available()))
            if baseline.metrics.cva_deg is None:
                print("Forward head position is not measured with this baseline.")
            if baseline.facing_sign is None:
                print("Side-view facing direction was not clear; trunk lean is not measured.")
            print("\nNow double-click Open Stack.command. Keep its Terminal window open.\n"
                  "Click ▤ in the menu bar, choose Check Now, then Start Monitoring.\n"
                  "To recalibrate later, quit Stack and double-click Recalibrate Stack.command.")
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\nSetup canceled. Your previous settings and baseline were kept.")
        return 0
    except Exception as exc:
        print(f"\nSetup could not finish: {exc}\nFix the problem and run setup again.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
