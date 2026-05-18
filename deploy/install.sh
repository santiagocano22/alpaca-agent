#!/usr/bin/env bash
# deploy/install.sh — Install and configure the Alpaca trading bot on a fresh VM.
#
# Tested on Ubuntu 22.04 LTS.
#
# Usage:
#   sudo bash deploy/install.sh
#
# After running this script:
#   1. Copy your .env file to /etc/trading-agent/.env  (sudo-only readable)
#   2. sudo systemctl start trading-agent
#   3. sudo journalctl -u trading-agent -f   (to follow logs)

set -euo pipefail

APP_USER="trading-agent"
APP_DIR="/opt/trading-agent"
ENV_DIR="/etc/trading-agent"
PYTHON_MIN="3.11"
REPO_URL="${REPO_URL:-}"   # Set this to your git remote if you want auto-clone

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Root check ────────────────────────────────────────────────────────────────
[[ $EUID -eq 0 ]] || error "This script must be run as root (use sudo)."

# ── 1. System packages ────────────────────────────────────────────────────────
info "Updating system packages…"
apt-get update -q
apt-get install -y -q python3.11 python3.11-venv python3.11-dev git curl

python_ver=$(python3.11 --version 2>&1 | awk '{print $2}')
info "Python version: $python_ver"

# ── 2. Create system user ─────────────────────────────────────────────────────
if ! id "$APP_USER" &>/dev/null; then
    info "Creating system user $APP_USER…"
    useradd --system --shell /bin/false --home-dir "$APP_DIR" --create-home "$APP_USER"
else
    info "User $APP_USER already exists, skipping"
fi

# ── 3. Deploy application code ────────────────────────────────────────────────
if [[ -n "$REPO_URL" ]]; then
    info "Cloning repository from $REPO_URL…"
    if [[ -d "$APP_DIR/.git" ]]; then
        git -C "$APP_DIR" pull --ff-only
    else
        git clone "$REPO_URL" "$APP_DIR"
    fi
else
    # If running the script from inside the repo, copy it
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(dirname "$SCRIPT_DIR")"
    info "Copying code from $REPO_ROOT to $APP_DIR…"
    rsync -a --exclude='.git' --exclude='.venv' --exclude='*.db' \
        "$REPO_ROOT/" "$APP_DIR/"
fi
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# ── 4. Python virtual environment ─────────────────────────────────────────────
info "Setting up Python virtual environment…"
if [[ ! -d "$APP_DIR/.venv" ]]; then
    sudo -u "$APP_USER" python3.11 -m venv "$APP_DIR/.venv"
fi
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"
info "Dependencies installed"

# ── 5. Environment file ───────────────────────────────────────────────────────
mkdir -p "$ENV_DIR"
chmod 700 "$ENV_DIR"
if [[ ! -f "$ENV_DIR/.env" ]]; then
    cp "$APP_DIR/.env.example" "$ENV_DIR/.env"
    chmod 600 "$ENV_DIR/.env"
    chown root:root "$ENV_DIR/.env"
    warn "Copied .env.example to $ENV_DIR/.env — EDIT IT NOW with your real secrets!"
    warn "Especially: ALPACA_API_KEY, ALPACA_API_SECRET, ANTHROPIC_API_KEY,"
    warn "            TELEGRAM_BOT_TOKEN, TELEGRAM_AUTHORIZED_CHAT_ID"
else
    info ".env file already exists at $ENV_DIR/.env, skipping"
fi

# ── 6. Run Alembic migrations ─────────────────────────────────────────────────
info "Running database migrations…"
# Load env vars for migration (only non-secret defaults needed here)
export DATABASE_URL="sqlite+aiosqlite:////$APP_DIR/trading_bot.db"
export ALPACA_API_KEY="placeholder"
export ALPACA_API_SECRET="placeholder"
export ANTHROPIC_API_KEY="placeholder"
export TELEGRAM_BOT_TOKEN="placeholder"
export TELEGRAM_AUTHORIZED_CHAT_ID="0"

sudo -u "$APP_USER" \
    env DATABASE_URL="$DATABASE_URL" \
    ALPACA_API_KEY="$ALPACA_API_KEY" \
    ALPACA_API_SECRET="$ALPACA_API_SECRET" \
    ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
    TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN" \
    TELEGRAM_AUTHORIZED_CHAT_ID="$TELEGRAM_AUTHORIZED_CHAT_ID" \
    "$APP_DIR/.venv/bin/alembic" -c "$APP_DIR/alembic.ini" upgrade head
chown "$APP_USER:$APP_USER" "$APP_DIR/trading_bot.db" 2>/dev/null || true

# ── 7. Systemd service ────────────────────────────────────────────────────────
info "Installing systemd service…"
cp "$APP_DIR/deploy/trading-agent.service" /etc/systemd/system/trading-agent.service

# Point DATABASE_URL to the absolute path in the service file
sed -i "s|/opt/trading-agent/trading_bot.db|$APP_DIR/trading_bot.db|g" \
    /etc/systemd/system/trading-agent.service || true

systemctl daemon-reload
systemctl enable trading-agent
info "Service enabled (not yet started)"

# ── 8. Optional: SQLite backup timer ─────────────────────────────────────────
cat > /etc/systemd/system/trading-agent-backup.service << 'UNIT'
[Unit]
Description=Trading Bot SQLite Backup
After=trading-agent.service

[Service]
Type=oneshot
User=trading-agent
ExecStart=/bin/cp /opt/trading-agent/trading_bot.db \
    /opt/trading-agent/trading_bot.db.bak
UNIT

cat > /etc/systemd/system/trading-agent-backup.timer << 'TIMER'
[Unit]
Description=Run SQLite backup every 6 hours

[Timer]
OnBootSec=10min
OnUnitActiveSec=6h
Unit=trading-agent-backup.service

[Install]
WantedBy=timers.target
TIMER

systemctl daemon-reload
systemctl enable trading-agent-backup.timer
systemctl start trading-agent-backup.timer
info "SQLite backup timer enabled (every 6 hours)"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
info "Installation complete!"
echo ""
echo "  Next steps:"
echo "  1. Edit secrets:  sudo nano $ENV_DIR/.env"
echo "  2. Start bot:     sudo systemctl start trading-agent"
echo "  3. Follow logs:   sudo journalctl -u trading-agent -f"
echo "  4. Check status:  sudo systemctl status trading-agent"
echo ""
warn "DO NOT start the bot until you have configured $ENV_DIR/.env with real values."
