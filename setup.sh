#!/usr/bin/env bash
# Run this once on your VPS to set up the bot
set -e

echo "=== Claude Crypto Bot Setup ==="

# Python 3.11+ required
python3 --version

# Install dependencies
pip3 install -r requirements.txt

# Create .env from example if not present
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "✅ Created .env — EDIT IT NOW with your API keys before starting:"
    echo "   nano .env"
    echo ""
fi

echo ""
echo "=== Systemd service setup (optional, run as root) ==="
echo "sudo cp crypto_bot.service /etc/systemd/system/"
echo "sudo systemctl daemon-reload"
echo "sudo systemctl enable crypto_bot"
echo "sudo systemctl start crypto_bot"
echo ""
echo "=== Manual start ==="
echo "python3 main.py"
