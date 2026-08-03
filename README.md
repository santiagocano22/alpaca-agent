# Alpaca Trading Bot

An automated trading bot that executes strategies defined in plain language, controlled entirely via Telegram. Claude parses your strategy once; a deterministic rules engine runs it in real-time without further LLM calls.

```
/strategy compra QQQ cuando RSI 14 < 30 en velas de 15min, stop-loss 2%, máx 10% del capital
```

## Features

- **Natural-language strategy** — describe your strategy in plain Spanish or English; Claude (Sonnet) converts it to validated rules once.
- **Deterministic execution** — RSI, EMA, SMA, MACD, Bollinger Bands, ATR, VWAP, breakout indicators; no LLM in the trading loop.
- **12-check risk manager** — position sizing, exposure limits, stop-loss, duplicate orders, buying power, clock drift.
- **Telegram control and diagnostics** — pause, resume, view positions, inspect streams/bars/rules, set risk overrides, ask questions, daily P&L summary.
- **Market-hours recovery** — recurring NYSE/NASDAQ calendar reconciliation, warmup after restarts, EOD close-all or hold, early closes and holiday detection.
- **Timeframe-correct live data** — Alpaca minute bars are aggregated into completed 5m, 15m or 1h strategy bars before evaluation.
- **Recoverable order lifecycle** — order intent is committed before submission and reconciled by client order ID after a restart.
- **Paper trading by default** — one env-var change + explicit confirmation flag to go live.
- **Persistent deployment** — macOS LaunchDaemon and Linux systemd service definitions with automatic restart.

---

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Get API keys](#get-api-keys)
3. [Local setup](#local-setup)
4. [Configure `.env`](#configure-env)
5. [Run locally](#run-locally)
6. [Test with a strategy](#test-with-a-strategy)
7. [Deploy permanently on macOS](#deploy-permanently-on-macos)
8. [Deploy on a VM (systemd)](#deploy-on-a-vm-systemd)
9. [Switch from paper to live trading](#switch-from-paper-to-live-trading)
10. [Telegram commands reference](#telegram-commands-reference)
11. [Architecture overview](#architecture-overview)
12. [Appendix: Docker (optional)](#appendix-docker-optional)

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 | 3.12+ not tested; 3.14 incompatible with pydantic-core wheel |
| Alpaca account | Paper tier | Free at [alpaca.markets](https://alpaca.markets) |
| Anthropic account | Any tier | At [console.anthropic.com](https://console.anthropic.com) |
| Telegram bot | — | Created via [@BotFather](https://t.me/BotFather) |
| Linux VM (deploy) | Ubuntu 22.04+ | 1 vCPU / 512 MB RAM minimum |

---

## Get API keys

### Alpaca (paper trading)

1. Sign up at <https://alpaca.markets> → choose **Paper Trading**.
2. Go to **Paper Trading → API Keys** → Generate new key.
3. Copy the **Key** (`ALPACA_API_KEY`) and **Secret** (`ALPACA_API_SECRET`).
4. Leave `ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2` (default).

### Anthropic

1. Sign up at <https://console.anthropic.com>.
2. Go to **API Keys** → **Create key**.
3. Copy the key (`ANTHROPIC_API_KEY`).
4. The bot uses `claude-sonnet-4-5` for strategy parsing and `claude-haiku-4-5` for daily summaries / `/ask`. Estimated cost: < $0.05/day under normal usage.

### Telegram

1. Open Telegram and message [@BotFather](https://t.me/BotFather).
2. Send `/newbot`, follow prompts → copy the **Bot Token** (`TELEGRAM_BOT_TOKEN`).
3. Find your **chat ID**: message [@userinfobot](https://t.me/userinfobot) or start your new bot and check the URL in the Telegram Web App. Copy the chat_id (`TELEGRAM_AUTHORIZED_CHAT_ID`).

---

## Local setup

```bash
# Clone
git clone <your-repo-url> trading-agent
cd trading-agent

# Create venv
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure secrets (see next section)
cp .env.example .env
nano .env   # or your favourite editor

# Run database migrations
alembic upgrade head
```

---

## Configure `.env`

Edit `.env` with your actual credentials:

```dotenv
ALPACA_API_KEY=PKXXXXXXXXXXXXXXXXXXXXX
ALPACA_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2

ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

TELEGRAM_BOT_TOKEN=1234567890:AAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TELEGRAM_AUTHORIZED_CHAT_ID=987654321   # your personal chat ID

LIVE_TRADING_CONFIRMED=false   # NEVER change to true without reading the safety section
DRY_RUN=false

EOD_CLOSE_MINUTES_BEFORE=5
WARMUP_MINUTES_BEFORE_OPEN=5
PENDING_STRATEGY_TTL_SECONDS=600
HEALTHCHECK_INTERVAL_MINUTES=15

DATABASE_URL=sqlite+aiosqlite:///./trading_bot.db
```

> **Security**: `.env` is in `.gitignore`. Never commit it. On a VM, store it at `/etc/trading-agent/.env` with mode `600` (root-readable only).

---

## Run locally

```bash
source .venv/bin/activate
python -m src.main
```

The bot will:
1. Run Alembic migrations.
2. Connect to Alpaca (paper).
3. Load the active strategy from DB (if any).
4. Start the Telegram bot (polling).
5. Start recurring market/calendar and order reconciliation.
6. Reconcile immediately: a mid-session start warms history and becomes active without waiting for the next day.

Stop with `Ctrl+C` — it will clean up streams and the scheduler gracefully.

---

## Test with a strategy

Once the bot is running, open Telegram and send commands to your bot:

### 1. Check status
```
/status
```
Expected: bot state IDLE/WARMUP/ACTIVE, equity, market state.

### 2. Load a strategy
```
/strategy compra QQQ cuando el RSI de 14 períodos esté por debajo de 30 en velas de 15 minutos. Vende cuando el RSI supere 70. Stop-loss del 2%, máximo 10% del capital por operación.
```
The bot calls Claude once (≈ 2–3 seconds) and replies with the parsed JSON:
```json
{
  "name": "RSI Oversold QQQ",
  "universe": ["QQQ"],
  "timeframe": "15Min",
  ...
}
```
Then: `/confirm` to activate, or `/cancel` to discard.

### 3. Check active strategy
```
/strategy
```

### 4. View positions (paper)
```
/positions
```

### 5. Pause/resume
```
/pause
/resume
```

### 6. Ask a question
```
/ask ¿cuándo fue la última vez que compraste QQQ y por qué?
```

### 7. Set a risk override
```
/setlimit max_position_pct 5
```

### 8. Close all positions
```
/closeall
/closeall confirm
```

### 9. Daily summary
Automatically sent 5 minutes after market close. You can also trigger it by observing logs.

### 10. Diagnose inactivity
```
/diagnostics
```
This shows the reconciled market state, stream health, current subscriptions,
age of the last bar, last rule evaluation, condition snapshot, last signal,
risk rejection and runtime error. It distinguishes “no signal” from “no data”
or “market state stuck”.

### 11. Backtest the active daily strategy
```
/backtest
/backtest 90
```
The command runs in the background and never accesses Alpaca's order endpoint.
It downloads enough warmup history, evaluates signals at each daily close and
simulates fills at the next session open. The Telegram report includes return,
maximum drawdown, fills, open positions, risk rejections and the pass rate of
every entry condition. The optional argument is calendar days (30 by default,
5–730 allowed).

For a reproducible command-line run:
```bash
python -m src.backtest_cli \
  --strategy strategies/etf_pullback_trend_filtered.json \
  --days 30 --initial-cash 100000 --slippage-bps 5 --send-telegram
```

The current implementation intentionally supports `1D` strategies only.
Assumptions are signal-at-close, next-open execution, fractional shares, no
taxes and configurable slippage (5 bps by default).

---

## Deploy permanently on macOS

The macOS installer uses the existing pyenv Python 3.11.15, copies the app to
`/opt/alpaca-agent`, migrates the database, activates the exact recommended
paper strategy and installs a system LaunchDaemon with `RunAtLoad` and
`KeepAlive` enabled.

From the repository root, run:

```bash
sudo ./deploy/enable-macos-host.sh
```

The host setup also disables computer sleep on AC power, enables wake-on-LAN
and enables restart after a power failure. Verify it with:

```bash
pmset -g custom
launchctl print system/com.santiagocano.alpaca-agent
tail -f /opt/alpaca-agent/logs/trading_bot.log
```

The deployed `.env` remains mode `600`, and the activation command refuses a
live Alpaca URL. FileVault remains enabled: after a total power loss, macOS may
require a person to unlock the encrypted disk before any LaunchDaemon can run.
Use a UPS if unattended recovery through outages is required.

---

## Deploy on a VM (systemd)

### 1. Provision a VM

Minimum: 1 vCPU, 512 MB RAM, Ubuntu 22.04. A `t3.micro` on AWS or equivalent works fine.

### 2. Run the install script

```bash
# As root on the VM:
git clone <your-repo-url> /tmp/trading-agent-src
cd /tmp/trading-agent-src
sudo bash deploy/install.sh
```

The script:
- Installs Python 3.11, git, and rsync
- Creates a `trading-agent` system user
- Copies the app to `/opt/trading-agent`
- Creates a Python virtualenv and installs dependencies
- Copies `.env.example` to `/etc/trading-agent/.env`
- Runs Alembic migrations
- Installs and enables the `trading-agent.service`
- Installs a SQLite backup timer (every 6 hours)

### 3. Configure secrets

```bash
sudo nano /etc/trading-agent/.env
```

Fill in all values (same as the local `.env` but with the production secrets).

### 4. Start the service

```bash
sudo systemctl start trading-agent
sudo systemctl status trading-agent
```

### 5. Follow logs

```bash
sudo journalctl -u trading-agent -f
```

### 6. Update the bot

```bash
# On the VM, as root:
cd /tmp/trading-agent-src
git pull
sudo rsync -a --exclude='.git' --exclude='.venv' --exclude='*.db' \
    . /opt/trading-agent/
sudo -u trading-agent /opt/trading-agent/.venv/bin/pip install -r /opt/trading-agent/requirements.txt
sudo -u trading-agent /opt/trading-agent/.venv/bin/alembic -c /opt/trading-agent/alembic.ini upgrade head
sudo systemctl restart trading-agent
```

### 7. Backup and restore database

```bash
# Manual backup
sudo cp /opt/trading-agent/trading_bot.db /opt/trading-agent/trading_bot.db.bak

# Restore
sudo systemctl stop trading-agent
sudo cp /opt/trading-agent/trading_bot.db.bak /opt/trading-agent/trading_bot.db
sudo systemctl start trading-agent
```

The automatic backup timer runs every 6 hours via `trading-agent-backup.timer`.

---

## Switch from paper to live trading

> ⚠️ **This section involves real money. Read carefully.**

### Prerequisites before going live

1. **Test extensively** on paper trading for at least 2–4 weeks.
2. Verify that every strategy behaves as expected.
3. Confirm that stop-losses fire correctly.
4. Review all risk limits in your strategy (`max_position_pct`, `stop_loss_pct`, etc.).

### Steps

1. **Get live Alpaca API keys**: Log in to Alpaca → switch to **Live Trading** → generate new keys.

2. **Update `/etc/trading-agent/.env`** (or local `.env`):
   ```dotenv
   ALPACA_API_KEY=your_live_api_key        # different from paper key
   ALPACA_API_SECRET=your_live_api_secret
   ALPACA_BASE_URL=https://api.alpaca.markets/v2
   LIVE_TRADING_CONFIRMED=true             # EXPLICIT OPT-IN
   ```

3. **The bot refuses to start if `LIVE_TRADING_CONFIRMED=false` and the URL points to live**. This is intentional.

4. **Set `DRY_RUN=true` first** for a final sanity check. In dry-run mode, orders are logged and Telegram-notified but not sent to Alpaca. Confirm the logic looks correct before setting `DRY_RUN=false`.

5. Start with a very small `max_position_pct` (e.g., 1%) until you have confidence.

6. Monitor closely for the first few days.

---

## Telegram commands reference

| Command | Description |
|---|---|
| `/start` | Welcome message and current state |
| `/status` | Bot state, market state, equity, P&L |
| `/positions` | Open positions with unrealized P&L |
| `/strategy` | Show active strategy |
| `/strategy <text>` | Parse and stage a new strategy (requires `/confirm`) |
| `/confirm` | Activate the staged strategy |
| `/cancel` | Discard the staged strategy |
| `/pause` | Stop sending new orders (positions remain open) |
| `/resume` | Resume sending orders |
| `/closeall` | Close all positions (asks for `/closeall confirm`) |
| `/closeall confirm` | Execute the close (or queue if market closed) |
| `/ask <question>` | Free-form question answered by Claude (Haiku) with context |
| `/setlimit <param> <value>` | Override risk limit for current session |
| `/diagnostics` | Calendar, streams, subscriptions, last bars/evaluations and rejection reason |
| `/backtest [days]` | Simulate the active daily strategy on historical data; defaults to 30 days |
| `/help` | List all commands |

**Valid `/setlimit` parameters:**
- `max_position_pct` — max % of equity per single position (0–100)
- `max_total_exposure_pct` — max total exposure across all positions (0–100)
- `stop_loss_pct` — effective stop-loss % (0–50)
- `max_concurrent_positions` — max number of open positions (1–50)

> Overrides are always *more restrictive* than the strategy default. A less-restrictive override is silently discarded and logged as a warning.

---

## Architecture overview

```
Telegram → strategy parser → validated Strategy
    │                         │
    │ /confirm                └─► hot-update stream subscriptions + warmup
    ▼
BotState / BotDeps ◄──── recurring market/calendar reconciler
    │
Alpaca 1m stream → timeframe aggregator → completed strategy bar
    │                                      │
    │                                      ▼
    └────────────────────────────── deterministic engine
                                           │
                                           ▼
                                  12-check risk manager
                                           │
                                           ▼
                         persist OrderAttempt → submit to Alpaca
                                           │
                                           ▼
                         trade stream / reconciler → Trade + status
```

### Key design decisions

- **LLM called only twice**: once to parse a new strategy (`claude-sonnet-4-5`), once for the daily summary / `/ask` (`claude-haiku-4-5`). Never in the trading loop.
- **12-check risk manager**: cheap checks first (paused flag, qty > 0) before expensive ones (account balance, positions).
- **Deterministic engine**: pure pandas indicator functions; no randomness or LLM.
- **Idempotent reconciliation**: market transitions, EOD liquidation and summaries are keyed by trading date so polling cannot execute them twice.
- **Observable inactivity**: `/diagnostics` exposes whether silence comes from closed market, stale streams, missing bars, unmet conditions or risk rejection.
- **systemd deployment**: non-root service user, `ProtectSystem=strict`, automatic restart on failure.

---

## Appendix: Docker (optional)

> Docker deployment is not part of v1 but is documented here for teams that prefer container-based deployments.

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "src.main"]
```

### `docker-compose.yml`

```yaml
version: "3.9"
services:
  trading-agent:
    build: .
    restart: on-failure
    env_file: .env
    volumes:
      - ./data:/app/data
    environment:
      DATABASE_URL: sqlite+aiosqlite:////app/data/trading_bot.db
```

### Usage

```bash
# Build and start
docker compose up -d

# Logs
docker compose logs -f trading-agent

# Update
docker compose build --no-cache
docker compose up -d
```

> Remember to mount a persistent volume for the SQLite database so it survives container restarts.
