#!/bin/bash
#
# Run reduced particle test with comprehensive resource monitoring
#

set -e

# Create logs directory
mkdir -p logs

echo "="*80"
echo "STARTING REDUCED PARTICLE TEST WITH MONITORING"
echo "="*80

# Import the reduced config
cat > logs/run_test.py << 'PYTHON_SCRIPT'
import sys
import os
sys.path.insert(0, os.getcwd())

# Import the reduced configuration
from config_reduced_particles import config_reduced

# Run the workflow with reduced config
from example_workflow import main
main(config=config_reduced)
PYTHON_SCRIPT

# Start monitoring in background
echo "Starting resource monitoring..."
(
    echo "timestamp,ram_used_gb,ram_avail_gb,gpu_mem_mb,gpu_util_pct" > logs/resource_monitor.csv
    while true; do
        timestamp=$(date +%s)

        # Get RAM usage
        ram_info=$(free -g | awk '/Mem:/ {print $3","$7}')

        # Get GPU usage
        gpu_info=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0,0")

        echo "$timestamp,$ram_info,$gpu_info" >> logs/resource_monitor.csv
        sleep 5
    done
) &
MONITOR_PID=$!
echo $MONITOR_PID > logs/monitor_pid.txt
echo "Monitor PID: $MONITOR_PID"

# Run the test
echo "Running reduced particle test..."
source .venv/bin/activate
python -u logs/run_test.py 2>&1 | tee logs/reduced_test.log

# Kill monitor
TEST_EXIT_CODE=${PIPESTATUS[0]}
kill $MONITOR_PID 2>/dev/null || true

# Create summary
echo ""
echo "="*80
echo "TEST COMPLETE"
echo "="*80
echo "Exit code: $TEST_EXIT_CODE"
echo "Logs saved to: logs/reduced_test.log"
echo "Resource data: logs/resource_monitor.csv"
echo "="*80

# Generate summary report
python3 << 'SUMMARY_SCRIPT'
import pandas as pd
import json

# Load resource data
try:
    df = pd.read_csv('logs/resource_monitor.csv')

    summary = {
        'ram_peak_gb': float(df['ram_used_gb'].max()),
        'ram_mean_gb': float(df['ram_used_gb'].mean()),
        'gpu_mem_peak_mb': int(df['gpu_mem_mb'].max()),
        'gpu_mem_mean_mb': float(df['gpu_mem_mb'].mean()),
        'gpu_util_peak_pct': int(df['gpu_util_pct'].max()),
        'gpu_util_mean_pct': float(df['gpu_util_pct'].mean()),
        'num_samples': len(df)
    }

    with open('logs/resource_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nRESOURCE USAGE SUMMARY:")
    print(f"  RAM Peak: {summary['ram_peak_gb']:.2f} GB")
    print(f"  RAM Mean: {summary['ram_mean_gb']:.2f} GB")
    print(f"  GPU Memory Peak: {summary['gpu_mem_peak_mb']} MB")
    print(f"  GPU Memory Mean: {summary['gpu_mem_mean_mb']:.1f} MB")
    print(f"  GPU Utilization Peak: {summary['gpu_util_peak_pct']}%")
    print(f"  GPU Utilization Mean: {summary['gpu_util_mean_pct']:.1f}%")
except Exception as e:
    print(f"Could not generate summary: {e}")
SUMMARY_SCRIPT

exit $TEST_EXIT_CODE
