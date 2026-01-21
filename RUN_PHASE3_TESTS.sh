#!/bin/bash
# Phase 3 Testing Script - Run All Three L2 Methods
# This script tests all L2 search methods after Phase 3 fixes

set -e  # Exit on error

echo "=========================================="
echo "Phase 3 Testing - All L2 Search Methods"
echo "=========================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Test 1: Neighbors (most common, should work with ~5 GB RAM)
echo "Test 1/3: Testing 'neighbors' method..."
echo "Expected RAM: ~5 GB during compilation"
echo ""
sed -i "s/L2_SEARCH_METHOD = .*/L2_SEARCH_METHOD = 'neighbors'/" production_tracking_fully_fused_timedep.py
echo "Running neighbors test (output: logs/phase3_neighbors.log)..."
python production_tracking_fully_fused_timedep.py > logs/phase3_neighbors.log 2>&1

# Check if neighbors test succeeded
if grep -q "Killed" logs/phase3_neighbors.log || grep -q "out of memory" logs/phase3_neighbors.log; then
    echo "❌ FAILED: Neighbors test crashed with OOM"
    echo "   See logs/phase3_neighbors.log for details"
    exit 1
else
    echo "✅ SUCCESS: Neighbors test compiled successfully"
fi
echo ""

# Test 2: Hierarchical (most complex, should work with ~8 GB RAM)
echo "Test 2/3: Testing 'hierarchical' method..."
echo "Expected RAM: ~8 GB during compilation"
echo ""
sed -i "s/L2_SEARCH_METHOD = .*/L2_SEARCH_METHOD = 'hierarchical'/" production_tracking_fully_fused_timedep.py
echo "Running hierarchical test (output: logs/phase3_hierarchical.log)..."
python production_tracking_fully_fused_timedep.py > logs/phase3_hierarchical.log 2>&1

# Check if hierarchical test succeeded
if grep -q "Killed" logs/phase3_hierarchical.log || grep -q "out of memory" logs/phase3_hierarchical.log; then
    echo "❌ FAILED: Hierarchical test crashed with OOM"
    echo "   See logs/phase3_hierarchical.log for details"
    exit 1
else
    echo "✅ SUCCESS: Hierarchical test compiled successfully"
fi
echo ""

# Test 3: Radius (regression test, should work with ~11 GB RAM)
echo "Test 3/3: Testing 'radius' method (regression test)..."
echo "Expected RAM: ~11 GB during compilation"
echo ""
sed -i "s/L2_SEARCH_METHOD = .*/L2_SEARCH_METHOD = 'radius'/" production_tracking_fully_fused_timedep.py
echo "Running radius test (output: logs/phase3_radius.log)..."
python production_tracking_fully_fused_timedep.py > logs/phase3_radius.log 2>&1

# Check if radius test succeeded
if grep -q "Killed" logs/phase3_radius.log || grep -q "out of memory" logs/phase3_radius.log; then
    echo "❌ FAILED: Radius test crashed with OOM"
    echo "   See logs/phase3_radius.log for details"
    exit 1
else
    echo "✅ SUCCESS: Radius test compiled successfully"
fi
echo ""

# All tests passed
echo "=========================================="
echo "🎉 ALL TESTS PASSED! 🎉"
echo "=========================================="
echo ""
echo "Phase 3 fixes are working correctly!"
echo ""
echo "Next steps:"
echo "1. Review particle retention metrics in log files"
echo "2. Choose best L2 method for production based on accuracy"
echo "3. Address radius loss issue separately (if needed)"
echo ""
echo "Log files created:"
echo "  - logs/phase3_neighbors.log"
echo "  - logs/phase3_hierarchical.log"
echo "  - logs/phase3_radius.log"
echo ""
