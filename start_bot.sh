#!/bin/bash
# Start Telegram bot

cd "$(dirname "$0")"

# Check if already running
if pgrep -f "python bot.py" > /dev/null; then
    echo "Bot is already running (PID: $(pgrep -f 'python bot.py'))"
    exit 1
fi

# Start bot
echo "Starting bot..."
nohup .venv/bin/python bot.py > bot.log 2>&1 &
sleep 2

# Check status
if pgrep -f "python bot.py" > /dev/null; then
    echo "✓ Bot started successfully (PID: $(pgrep -f 'python bot.py'))"
    echo "Logs: tail -f bot.log"
else
    echo "✗ Failed to start bot"
    tail -20 bot.log
    exit 1
fi
