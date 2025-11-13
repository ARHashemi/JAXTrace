#!/bin/bash
# Monitor the integration test progress

LOG_FILE="logs/integration_test_threadeda.log"

echo "==================================================================="
echo "Integration Test Monitor"
echo "==================================================================="
echo ""

while true; do
    clear
    echo "==================================================================="
    echo "Integration Test Progress - $(date '+%H:%M:%S')"
    echo "==================================================================="
    echo ""

    # Show last 30 lines of log
    tail -30 "$LOG_FILE" 2>/dev/null || echo "Waiting for log file..."

    echo ""
    echo "==================================================================="
    echo "Press Ctrl+C to exit monitor"
    echo "==================================================================="

    sleep 5
done
