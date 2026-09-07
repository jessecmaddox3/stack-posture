# Stack

> **TL;DR:** Stack is a local macOS menu bar app that measures desk posture
> against a baseline you set. Try its offline demo with no camera, account,
> Keychain, or network: `uv run python scripts/demo.py`.

[![Tests](https://github.com/jessecmaddox3/stack-posture/actions/workflows/tests.yml/badge.svg)](https://github.com/jessecmaddox3/stack-posture/actions/workflows/tests.yml)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Stack uses MediaPipe locally to measure posture from one or two cameras. Its
ratings are relative to a personal baseline, rather than a generic ideal. An
optional Gemini second opinion can add coaching text and compare agreement with
the local measurement. Gemini starts **off**, so a normal installation has no
account requirement and does not upload frames.

This is an early, local-use tool, not a medical device or clinical posture
assessment. Its measurements are heuristics for a stable desk setup. Do not use
it to diagnose, treat, or make medical decisions.

## Try it without hardware or an account

Stack needs macOS and Python 3.12. Install [uv](https://docs.astral.sh/uv/),
then clone the repository and run the deterministic demo:

```bash
git clone https://github.com/jessecmaddox3/stack-posture.git
cd stack-posture
uv sync --extra dev --locked
uv run python scripts/demo.py
```

The demo scores synthetic measurements only. It does not access a camera,
Keychain, filesystem state, Gemini, or the network.

## Install and run the app

1. Fetch the MediaPipe pose model. It is downloaded separately and is not in
   the repository.

   ```bash
   ./scripts/fetch_model.sh
   ```

2. Run camera discovery, then delete its full-frame previews after recording
   the camera indexes.

   ```bash
   uv run python scripts/identify_cameras.py
   ```

3. Create `~/.posture_monitor/config.toml` with the cameras you want to use:

   ```toml
   [cameras]
   front = 0
   # side = 1
   ```

4. Start Stack manually and grant macOS camera permission when prompted:

   ```bash
   ./run.sh
   ```

5. Use the menu-bar app to calibrate. Calibration stores numeric measurements,
   not normal monitoring frames. A side camera is needed to measure forward
   head posture and trunk lean.

Stack stores its baseline, history, logs, configuration, and on-demand
dashboard under `~/.posture_monitor/`. The package and data path deliberately
retain this established name for compatibility with existing installations.

## Privacy and optional Gemini

Normal monitoring and calibration process frames in memory and do not save
them. `identify_cameras.py` is the exception: it intentionally writes uncropped
camera previews to `~/.posture_monitor/camera_previews/` for manual camera
identification. Delete that directory after setup.

Gemini is disabled unless you explicitly set `gemini_enabled = true` in
`~/.posture_monitor/config.toml`. With it enabled, Stack crops and downscales a
person region in memory before sending it to Gemini. The crop removes much of
the room beside the person but can still include anything directly behind them.
Treat enabled Gemini as uploading a narrow, full-height desk image. The local
monitor, nudge, and dashboard work without Gemini.

To opt in, add these top-level keys before the `[cameras]` section:

```toml
gemini_enabled = true
comparison_sample_rate = 0.15
max_coaching_calls_per_day = 12

[cameras]
front = 0
```

Store an API key in macOS Keychain, or set `POSTURE_GEMINI_API_KEY` only in your
local environment. Do not commit keys, frames, database files, or configuration.
The settings cap coaching calls and the comparison sample rate. See the source
comments in `src/posture/config.py` before changing them.

## Limits

- Stack currently supports macOS only. It has not been tested on Windows or
  Linux, and its menu bar and Keychain integrations are macOS-specific.
- Camera compatibility, multi-camera behavior, popup placement, and focus
  behavior need real-hardware validation. The test suite uses synthetic frames
  and deliberately does not open a camera.
- Front-camera measurements cannot infer forward head posture or trunk lean.
  Those require a stable side view.
- Moving a camera changes measurements. Recalibrate after materially changing
  its position.
- Gemini wording and model availability can change. It does not drive local
  ratings, trend data, or nudges.

## Development and releases

```bash
uv sync --all-groups
uv run python -m pytest -q
uv run ruff check src tests scripts
uv build
```

The GitHub Actions workflow runs these checks on macOS and installs the built
wheel in a clean temporary environment before executing the offline demo.
`CONTRIBUTING.md` explains the synthetic-data policy and useful starter work.

The code is available under the [MIT License](LICENSE). See [NOTICE.md](NOTICE.md)
for dependency, model, dashboard asset, and optional Gemini notices. To report
a vulnerability privately, follow [SECURITY.md](SECURITY.md).
