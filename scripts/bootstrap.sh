#!/bin/bash
# Guided installation, invoked by Set Up Stack.command.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    echo "This version needs an Apple Silicon Mac (M1 or newer)."
    exit 1
fi
MAC_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
if [ "$MAC_MAJOR" -lt 13 ]; then
    echo "This version needs macOS 13 Ventura or newer."
    exit 1
fi
UV_BIN="$(command -v uv || true)"
for candidate in "$HOME/.local/bin/uv" /opt/homebrew/bin/uv /usr/local/bin/uv; do
    if [ -z "$UV_BIN" ] && [ -x "$candidate" ]; then UV_BIN="$candidate"; fi
done
if [ -z "$UV_BIN" ]; then
    echo "Stack needs uv to install its Python runtime and locked dependencies."
    echo "Install uv 0.11.28 from Astral into ~/.local/bin? Shell profiles stay unchanged."
    read -r -p '[y/N]: ' answer
    if [ "$answer" != y ] && [ "$answer" != Y ]; then
        echo "Canceled. Manual option: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    INSTALLER="$(mktemp -t stack-uv-installer)"
    trap 'rm -f "$INSTALLER"' EXIT
    curl --fail --location --proto '=https' --tlsv1.2 \
        'https://astral.sh/uv/0.11.28/install.sh' -o "$INSTALLER"
    env UV_NO_MODIFY_PATH=1 UV_INSTALL_DIR="$HOME/.local/bin" sh "$INSTALLER"
    UV_BIN="$HOME/.local/bin/uv"
fi
echo 'Installing locked dependencies and Python 3.12. First setup needs internet access.'
"$UV_BIN" sync --locked --python 3.12
exec .venv/bin/python -m posture.setup
