# Contributing

> **TL;DR:** Keep changes focused, use synthetic data, and run the test suite
> and linter before opening a pull request.

## Development setup

Stack supports Apple Silicon macOS 13+ with Python 3.12. Install [uv](https://docs.astral.sh/uv/),
then run:

```bash
uv sync --extra dev --locked
uv run python -m pytest -q
uv run ruff check src tests scripts
```

Tests must not access a real camera, Keychain, Gemini account, or live network.
Use mocks and synthetic landmarks or frames instead. Do not add captured desk
photos, API keys, databases, generated dashboards, or configuration files.

The committed dashboard demo is generated only from `posture.demo`'s invented
measurements. Rebuild it with `uv run python -m posture.demo --dashboard docs/Stack-Demo.html`.
To explicitly test native inference, download the pinned model into a temporary
test-only directory and run `pytest tests/test_landmarks.py --native-model-path /path/to/test-model.task`.
These two opt-in tests use blank synthetic frames. They never discover a model
or photograph in the user's personal-state folder. Keep this integration separate
from the default camera-free suite, especially in headless environments.

## Pull requests

Explain the user-visible problem, the minimal fix, and the commands you ran.
Preserve third-party notices. Contributions are accepted under the MIT License
unless an asset has a separate, explicitly compatible license. AI assistance is
welcome when contributors verify the work and disclose material use in the PR.

Good first contributions include clearer hardware troubleshooting, synthetic
regression tests, accessibility improvements to the dashboard, and documented
camera compatibility reports.
