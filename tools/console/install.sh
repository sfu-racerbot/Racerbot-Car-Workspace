#!/usr/bin/env bash
# Install the terminal menu's Python dependencies once per car. See docs/console.md.
#
# Makes a virtual environment that can still see the system's ROS packages
# (--system-site-packages), installs the pinned Textual into it, and links
# `racerbot` into ~/.local/bin. Touches nothing outside your home folder and
# never uses sudo; it tells you if tmux or python3-venv is missing.
set -euo pipefail

WS="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${RACERBOT_CONSOLE_VENV:-$HOME/.local/share/racerbot/console-venv}"

if ! python3 -c 'import venv, ensurepip' 2>/dev/null; then
    echo "python3-venv is missing. Install it once with: sudo apt install python3-venv" >&2
    exit 1
fi
mkdir -p "$(dirname "$VENV")"
python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -r "$WS/src/racerbot_console/requirements.txt"

mkdir -p "$HOME/.local/bin"
ln -sf "$WS/tools/console/racerbot" "$HOME/.local/bin/racerbot"
chmod +x "$WS/tools/console/racerbot"

echo "Installed. Textual $("$VENV/bin/python" -c 'import textual; print(textual.__version__)')."
command -v tmux >/dev/null 2>&1 || echo "tmux is missing: sudo apt install tmux (the menu works without it, but stops if the tab closes)."
case ":$PATH:" in
    *":$HOME/.local/bin:"*) echo "Start it with: racerbot" ;;
    *) echo "Start it with: ~/.local/bin/racerbot (or open a new terminal so ~/.local/bin is on PATH)" ;;
esac
echo "Before the first use: colcon build --symlink-install --packages-select racerbot_console"
