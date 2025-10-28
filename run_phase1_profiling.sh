#!/bin/bash
#
# Phase 1 Profiling Script
# Runs example_workflow.py with resource monitoring (CPU, GPU, memory)
#

set -e

# Output directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/phase1_test_${TIMESTAMP}.log"
RESOURCE_LOG="$LOG_DIR/phase1_resources_${TIMESTAMP}.log"

echo "================================"
echo "Phase 1 Profiling Test"
echo "================================"
echo "Log file: $LOG_FILE"
echo "Resource log: $RESOURCE_LOG"
echo ""

# Start resource monitoring in background
echo "Starting resource monitoring..."
(
    echo "Timestamp,CPU%,MemoryMB,GPU%,GPUMemoryMB" > "$RESOURCE_LOG"
    while true; do
        TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

        # CPU and Memory
        CPU=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
        MEM=$(free -m | awk 'NR==2{printf "%.0f", $3}')

        # GPU usage (if nvidia-smi available)
        if command -v nvidia-smi &> /dev/null; then
            GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
            GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        else
            GPU="N/A"
            GPU_MEM="N/A"
        fi

        echo "$TIMESTAMP,$CPU,$MEM,$GPU,$GPU_MEM" >> "$RESOURCE_LOG"
        sleep 2
    done
) &
MONITOR_PID=$!

# Trap to kill monitoring on exit
trap "kill $MONITOR_PID 2>/dev/null || true" EXIT

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Running example_workflow.py..."
echo ""

# Run with time measurement
/usr/bin/time -v python example_workflow.py 2>&1 | tee "$LOG_FILE"

# Extract timing information
echo ""
echo "================================"
echo "Timing Summary"
echo "================================"
grep -E "(Total|Stage|Search|Interpolation|Cache)" "$LOG_FILE" || echo "No timing info found"

echo ""
echo "================================"
echo "Cache Statistics"
echo "================================"
grep -A 10 "Element Cache Statistics" "$LOG_FILE" || echo "No cache stats found"

echo ""
echo "================================"
echo "Resource Usage Summary"
echo "================================"
echo "Peak CPU: $(awk -F',' '{if(NR>1 && $2!="N/A") print $2}' "$RESOURCE_LOG" | sort -n | tail -1)%"
echo "Peak Memory: $(awk -F',' '{if(NR>1) print $3}' "$RESOURCE_LOG" | sort -n | tail -1) MB"
if command -v nvidia-smi &> /dev/null; then
    echo "Peak GPU: $(awk -F',' '{if(NR>1 && $4!="N/A") print $4}' "$RESOURCE_LOG" | sort -n | tail -1)%"
    echo "Peak GPU Memory: $(awk -F',' '{if(NR>1 && $5!="N/A") print $5}' "$RESOURCE_LOG" | sort -n | tail -1) MB"
fi

echo ""
echo "Full logs saved to:"
echo "  - $LOG_FILE"
echo "  - $RESOURCE_LOG"
echo ""
echo "Done!"
