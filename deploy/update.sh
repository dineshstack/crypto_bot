#!/usr/bin/env bash
# Update the bot — pull latest code, reinstall deps, restart service
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE="crypto-bot"

echo "▸ Stopping bot..."
sudo systemctl stop $SERVICE 2>/dev/null || true

echo "▸ Updating Python dependencies..."
source "$APP_DIR/venv/bin/activate"
pip install -r "$APP_DIR/requirements.txt" -q

echo "▸ Restarting bot..."
sudo systemctl start $SERVICE

echo "▸ Checking status..."
sleep 2
if systemctl is-active --quiet $SERVICE; then
    echo "✅ Bot updated and running"
else
    echo "❌ Bot failed to start — check logs:"
    echo "   journalctl -u $SERVICE -n 20"
fi
