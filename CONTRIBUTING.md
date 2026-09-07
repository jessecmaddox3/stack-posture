# Contributing

> **TL;DR:** Keep changes focused, use synthetic data, and run the test suite
> and linter before opening a pull request.

## Development setup

Stack supports macOS with Python 3.12. Install [uv](https://docs.astral.sh/uv/),
then run:

```bash
uv sync --extra dev --locked
uv run python -m pytest -q
uv run ruff check src tests scripts
```

Tests must not access a real camera, Keychain, Gemini account, or live network.
Use mocks and synthetic landmarks or frames instead. Do not add captured desk
photos, API keys, databases, generated dashboards, or configuration files.

## Pull requests

Explain the user-visible problem, the minimal fix, and the commands you ran.
Preserve third-party notices. Contributions are accepted under the MIT License
unless an asset has a separate, explicitly compatible license. AI assistance is
welcome when contributors verify the work and disclose material use in the PR.

Good first contributions include clearer hardware troubleshooting, synthetic
regression tests, accessibility improvements to the dashboard, and documented
camera compatibility reports.
