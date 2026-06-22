#!/usr/bin/env bash
# Quick health check — run anytime to verify the bot is working
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE="crypto-bot"

echo "═══ Crypto Bot Health Check ═══"
echo ""

# 1. Service status
echo "▸ Service status:"
if systemctl is-active --quiet $SERVICE 2>/dev/null; then
    echo "  ✅ Running"
    uptime=$(systemctl show $SERVICE --property=ActiveEnterTimestamp | cut -d= -f2)
    echo "  Since: $uptime"
else
    echo "  ❌ Not running"
    echo "  Start with: sudo systemctl start $SERVICE"
fi
echo ""

# 2. Recent log activity
echo "▸ Last 5 log lines:"
if [ -f "$APP_DIR/logs/bot.log" ]; then
    tail -5 "$APP_DIR/logs/bot.log" | sed 's/^/  /'
elif [ -f "$APP_DIR/bot.log" ]; then
    tail -5 "$APP_DIR/bot.log" | sed 's/^/  /'
else
    echo "  No log file found"
fi
echo ""

# 3. Python venv
echo "▸ Python environment:"
if [ -f "$APP_DIR/venv/bin/python3" ]; then
    echo "  ✅ venv exists"
    PKGS=$("$APP_DIR/venv/bin/pip" list 2>/dev/null | wc -l)
    echo "  Packages: $PKGS"
else
    echo "  ❌ No venv — run deploy/install.sh"
fi
echo ""

# 4. .env check
echo "▸ Configuration:"
if [ -f "$APP_DIR/.env" ]; then
    echo "  ✅ .env exists"
    TESTNET=$(grep "^TESTNET=" "$APP_DIR/.env" | cut -d= -f2)
    echo "  Mode: ${TESTNET:-unknown} (TESTNET=$TESTNET)"
else
    echo "  ❌ No .env file"
fi
echo ""

# 5. ML model
echo "▸ ML model:"
if [ -f "$APP_DIR/ml_models/btc_ensemble_v2.joblib" ]; then
    SIZE=$(du -h "$APP_DIR/ml_models/btc_ensemble_v2.joblib" | cut -f1)
    MOD=$(stat -c %y "$APP_DIR/ml_models/btc_ensemble_v2.joblib" 2>/dev/null || stat -f %Sm "$APP_DIR/ml_models/btc_ensemble_v2.joblib" 2>/dev/null)
    echo "  ✅ Trained ($SIZE, modified: $MOD)"
else
    echo "  ⚠ Not trained yet — will train on first cycle"
fi

# 6. RL Q-table
if [ -f "$APP_DIR/ml_models/rl_q_table.json" ]; then
    STATES=$(python3 -c "import json; d=json.load(open('$APP_DIR/ml_models/rl_q_table.json')); print(len(d.get('q_table',{})))" 2>/dev/null || echo "?")
    echo "  RL: $STATES states explored"
fi
echo ""

# 7. Disk usage
echo "▸ Disk usage:"
du -sh "$APP_DIR" 2>/dev/null | sed 's/^/  Total: /'
du -sh "$APP_DIR/logs" 2>/dev/null | sed 's/^/  Logs:  /'
du -sh "$APP_DIR/ml_models" 2>/dev/null | sed 's/^/  Models:/'
echo ""

# 8. Memory usage
echo "▸ System resources:"
if command -v free &>/dev/null; then
    MEM=$(free -h | awk '/^Mem:/{print $3 "/" $2}')
    echo "  RAM: $MEM"
fi
if pgrep -f "python.*main.py" &>/dev/null; then
    PID=$(pgrep -f "python.*main.py" | head -1)
    RSS=$(ps -o rss= -p $PID 2>/dev/null | awk '{printf "%.0f", $1/1024}')
    echo "  Bot RSS: ${RSS}MB (PID $PID)"
fi
echo ""
echo "═══ Done ═══"
