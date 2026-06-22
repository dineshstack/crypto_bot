#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Crypto Bot — Ubuntu 24.04 LTS deployment script
#
# Usage:
#   1. scp -r crypto_bot/ user@your-vps:~/
#   2. ssh user@your-vps
#   3. cd ~/crypto_bot && bash deploy/install.sh
#
# What it does:
#   - Installs Python 3.12, pip, venv, system deps (libomp for XGBoost)
#   - Creates a Python virtual environment at ~/crypto_bot/venv
#   - Installs all pip dependencies
#   - Creates ml_models/ and logs/ directories
#   - Installs systemd service (auto-start, auto-restart on crash)
#   - Installs logrotate config (keeps bot.log under control)
#   - Prompts you to edit .env if it's still using placeholder values
# ─────────────────────────────────────────────────────────────────────────────

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_NAME="crypto-bot"
USER="$(whoami)"

echo "══════════════════════════════════════════"
echo "  Crypto Bot Deployment — Ubuntu 24.04"
echo "══════════════════════════════════════════"
echo "  App dir:  $APP_DIR"
echo "  User:     $USER"
echo ""

# ── 1. System packages ──────────────────────────────────────────────────────
echo "▸ Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    libomp-dev build-essential git curl

# ── 2. Python virtual environment ───────────────────────────────────────────
echo "▸ Setting up Python virtual environment..."
if [ ! -d "$APP_DIR/venv" ]; then
    python3.12 -m venv "$APP_DIR/venv"
    echo "  Created venv at $APP_DIR/venv"
else
    echo "  venv already exists, skipping creation"
fi

source "$APP_DIR/venv/bin/activate"
pip install --upgrade pip -q

# ── 3. Install Python dependencies ──────────────────────────────────────────
echo "▸ Installing Python packages (this takes 2-3 minutes)..."
pip install -r "$APP_DIR/requirements.txt" -q

echo "  ✓ Installed $(pip list 2>/dev/null | wc -l) packages"

# ── 4. Create directories ──────────────────────────────────────────────────
echo "▸ Creating directories..."
mkdir -p "$APP_DIR/ml_models"
mkdir -p "$APP_DIR/logs"

# ── 5. Check .env ──────────────────────────────────────────────────────────
echo "▸ Checking .env configuration..."
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo "  ⚠ Created .env from .env.example — EDIT IT NOW:"
    echo "    nano $APP_DIR/.env"
    echo ""
    echo "  You MUST set these before starting:"
    echo "    ANTHROPIC_API_KEY, BINANCE_API_KEY, BINANCE_SECRET"
    echo "    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID"
    echo "    SUPABASE_URL, SUPABASE_KEY"
    echo ""
    read -p "  Press Enter after editing .env (or Ctrl+C to abort)..."
else
    echo "  ✓ .env exists"
    # Warn if placeholder values detected
    if grep -q "sk-ant-\.\.\." "$APP_DIR/.env" 2>/dev/null; then
        echo "  ⚠ .env still has placeholder API keys — edit before starting!"
    fi
fi

# ── 6. Install systemd service ─────────────────────────────────────────────
echo "▸ Installing systemd service..."
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null << UNIT
[Unit]
Description=Claude-Powered BTC+ETH Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/python3 $APP_DIR/main.py
Restart=always
RestartSec=10
StartLimitIntervalSec=300
StartLimitBurst=5

# Environment
EnvironmentFile=$APP_DIR/.env

# Logging
StandardOutput=append:$APP_DIR/logs/bot.log
StandardError=append:$APP_DIR/logs/bot.log

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=$APP_DIR
PrivateTmp=true

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
echo "  ✓ Service installed: ${SERVICE_NAME}.service"

# ── 6b. API server systemd service ─────────────────────────────────────────
echo "▸ Installing API server service..."
sudo tee /etc/systemd/system/${SERVICE_NAME}-api.service > /dev/null << UNIT2
[Unit]
Description=Crypto Bot API Server (FastAPI)
After=network-online.target mysql.service
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/uvicorn api_server:app --host 0.0.0.0 --port 8100
Restart=always
RestartSec=10

EnvironmentFile=$APP_DIR/.env
StandardOutput=append:$APP_DIR/logs/api.log
StandardError=append:$APP_DIR/logs/api.log

NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=$APP_DIR
PrivateTmp=true

[Install]
WantedBy=multi-user.target
UNIT2

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}-api
echo "  ✓ API server installed: ${SERVICE_NAME}-api.service (port 8100)"

# ── 7. Install logrotate ────────────────────────────────────────────────────
echo "▸ Installing logrotate config..."
sudo tee /etc/logrotate.d/${SERVICE_NAME} > /dev/null << LOGROTATE
$APP_DIR/logs/bot.log
$APP_DIR/bot.log
{
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    size 50M
}
LOGROTATE

echo "  ✓ Logrotate: keeps 14 days, rotates at 50MB"

# ── 8. Summary ─────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo "  ✅ Deployment complete!"
echo "══════════════════════════════════════════"
echo ""
echo "  MySQL setup (run once):"
echo "    sudo mysql -e \"CREATE DATABASE IF NOT EXISTS crypto_bot;\""
echo "    sudo mysql -e \"CREATE USER IF NOT EXISTS 'crypto_bot'@'localhost' IDENTIFIED BY 'YOUR_PASSWORD';\""
echo "    sudo mysql -e \"GRANT ALL ON crypto_bot.* TO 'crypto_bot'@'localhost';\""
echo "    sudo mysql crypto_bot < $APP_DIR/mysql_schema.sql"
echo ""
echo "  Commands:"
echo "    sudo systemctl start crypto-bot       # Start the bot"
echo "    sudo systemctl start crypto-bot-api   # Start the API server"
echo "    sudo systemctl stop crypto-bot        # Stop the bot"
echo "    sudo systemctl status crypto-bot      # Check bot status"
echo "    sudo systemctl status crypto-bot-api  # Check API status"
echo "    journalctl -u crypto-bot -f           # Bot logs"
echo "    tail -f $APP_DIR/logs/api.log         # API logs"
echo ""
echo "  First run checklist:"
echo "    1. Create MySQL database (see above)"
echo "    2. Set MYSQL_PASSWORD and API_SECRET_KEY in .env"
echo "    3. Start with TESTNET=true first"
echo "    4. Start both services: sudo systemctl start crypto-bot crypto-bot-api"
echo "    5. Send /start in Telegram"
echo "    6. Switch to TESTNET=false when ready for live trading"
echo ""
echo "  ML model training (optional — run manually first time):"
echo "    source $APP_DIR/venv/bin/activate"
echo "    python3 -c \"import ml_signal, market_data as md; e=md.get_exchange(); ml_signal.train_model(e)\""
echo ""
