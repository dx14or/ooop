#!/bin/bash
# Stop Telegram bot

if pgrep -f "python bot.py" > /dev/null; then
    echo "Stopping bot (PID: $(pgrep -f 'python bot.py'))..."
    pkill -f "python bot.py"
    sleep 1

    if pgrep -f "python bot.py" > /dev/null; then
        echo "✗ Bot still running, force killing..."
        pkill -9 -f "python bot.py"
    else
        echo "✓ Bot stopped"
    fi
else
    echo "Bot is not running"
fi
