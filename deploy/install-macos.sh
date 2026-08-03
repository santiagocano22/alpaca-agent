#!/usr/bin/env bash
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "Run with sudo" >&2; exit 1; }

APP_USER="santiagocano"
APP_GROUP="staff"
APP_DIR="/opt/alpaca-agent"
LABEL="com.santiagocano.alpaca-agent"
PLIST="/Library/LaunchDaemons/${LABEL}.plist"
PYTHON="/Users/santiagocano/.pyenv/versions/3.11.15/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

[[ -x "$PYTHON" ]] || { echo "Python 3.11 not found at $PYTHON" >&2; exit 1; }
[[ -f "$REPO_ROOT/.env" ]] || { echo "Missing $REPO_ROOT/.env" >&2; exit 1; }

mkdir -p "$APP_DIR" "$APP_DIR/logs"
rsync -a \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.env' \
  --exclude='research/data' \
  --exclude='*.db' \
  "$REPO_ROOT/" "$APP_DIR/"
chown -R "$APP_USER:$APP_GROUP" "$APP_DIR"

if [[ ! -d "$APP_DIR/.venv" ]]; then
  sudo -u "$APP_USER" "$PYTHON" -m venv "$APP_DIR/.venv"
fi
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"

if [[ ! -f "$APP_DIR/.env" ]]; then
  install -o "$APP_USER" -g "$APP_GROUP" -m 600 "$REPO_ROOT/.env" "$APP_DIR/.env"
fi

sudo -u "$APP_USER" sh -c "cd '$APP_DIR' && '$APP_DIR/.venv/bin/alembic' upgrade head"
sudo -u "$APP_USER" sh -c \
  "cd '$APP_DIR' && '$APP_DIR/.venv/bin/python' -m src.strategy.activate_cli \
   --strategy '$APP_DIR/strategies/etf_breakout_20_recommended.json'"

install -o root -g wheel -m 644 \
  "$APP_DIR/deploy/com.santiagocano.alpaca-agent.plist" "$PLIST"
plutil -lint "$PLIST"

launchctl bootout "system/$LABEL" >/dev/null 2>&1 || true
launchctl bootstrap system "$PLIST"
launchctl enable "system/$LABEL"
launchctl kickstart -kp "system/$LABEL"

echo "Installed and started $LABEL"
launchctl print "system/$LABEL" | sed -n '1,80p'
