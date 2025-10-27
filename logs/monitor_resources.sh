#!/bin/bash
PID=$1
LOG_FILE="logs/resource_usage.log"

echo "=== Resource Monitoring Started at $(date) ===" > $LOG_FILE
echo "Monitoring PID: $PID" >> $LOG_FILE
echo "" >> $LOG_FILE

while kill -0 $PID 2>/dev/null; do
    echo "=== $(date +"%Y-%m-%d %H:%M:%S") ===" >> $LOG_FILE
    
    # CPU and RAM usage
    ps -p $PID -o pid,%cpu,%mem,rss,vsz,etime,cmd --no-headers >> $LOG_FILE 2>/dev/null
    
    # GPU usage (if nvidia-smi is available)
    if command -v nvidia-smi &> /dev/null; then
        echo "--- GPU Status ---" >> $LOG_FILE
        nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits >> $LOG_FILE 2>/dev/null
    fi
    
    echo "" >> $LOG_FILE
    sleep 10
done

echo "=== Process $PID completed at $(date) ===" >> $LOG_FILE
