#!/usr/bin/env bash
# Quick test: Run production script with only 2 velocity timesteps (like benchmark)
# This will help determine if the 40 timesteps is causing the 41.70 TiB error

echo "================================="
echo "Production Script - 2 Timesteps Test"
echo "================================="
echo ""
echo "Modifying VELOCITY_TIMESTEP_RANGE to (158, 159)..."

# Create a temporary modified version
cp production_tracking_fully_fused_timedep.py production_tracking_fully_fused_timedep_test2ts.py

# Replace the timestep range (line 88)
sed -i 's/VELOCITY_TIMESTEP_RANGE = (120, 159)/VELOCITY_TIMESTEP_RANGE = (158, 159)  # TEST: 2 timesteps like benchmark/' production_tracking_fully_fused_timedep_test2ts.py

echo "Running with 2 timesteps..."
echo ""

python3 production_tracking_fully_fused_timedep_test2ts.py 2>&1 | tee logs/production_test_2timesteps.log

echo ""
echo "================================="
echo "Test complete. Check logs/production_test_2timesteps.log"
echo ""
echo "If this WORKS → 40 timesteps is the problem"
echo "If this FAILS → something else is wrong"
echo "================================="
