#!/bin/bash
#
# Run All JAXTrace Benchmarks
#
# This script runs both comprehensive benchmark suites and saves results to logs/
#
# Expected runtime: ~75 minutes total
#   - Point-in-tet benchmark: ~30 minutes
#   - L2 search benchmark: ~45 minutes
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "JAXTrace Comprehensive Benchmark Suite"
echo "================================================================================"
echo "Date: $(date)"
echo "Working directory: $PWD"
echo ""
echo "This will run:"
echo "  1. Point-in-tet methods benchmark (~30 min)"
echo "  2. L2 search methods benchmark (~45 min)"
echo ""
echo "Results will be saved to logs/"
echo "================================================================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Benchmark 1: Point-in-Tet Methods
echo "[1/2] Running Point-in-Tet Methods Benchmark..."
echo "-------------------------------------------------------"
echo "Testing 7 methods (current, skala, axis_aligned, pure_aa, skala_memory_opt, branchless_hybrid, inverse)"
echo "Distributions: Random Uniform (225K), Perturbed Centroids (3.3M)"
echo "Expected runtime: ~30 minutes"
echo ""

LOG_FILE_1="logs/benchmark_point_in_tet_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOG_FILE_1"
echo ""

python benchmark_point_in_tet_comprehensive.py 2>&1 | tee "$LOG_FILE_1"

RESULT_1=$?

if [ $RESULT_1 -eq 0 ]; then
    echo ""
    echo "✅ Point-in-Tet Benchmark Complete!"
    echo "   Log: $LOG_FILE_1"
    echo ""

    # Extract best method
    echo "Quick Results:"
    grep "Best Overall Method:" "$LOG_FILE_1" || echo "   (see log for details)"
    echo ""
else
    echo ""
    echo "❌ Point-in-Tet Benchmark Failed (exit code: $RESULT_1)"
    echo "   Check log: $LOG_FILE_1"
    exit $RESULT_1
fi

# Wait a moment before next benchmark
echo "Waiting 10 seconds before next benchmark..."
sleep 10

# Benchmark 2: L2 Search Methods
echo ""
echo "[2/2] Running L2 Search Methods Benchmark..."
echo "-------------------------------------------------------"
echo "Testing 6 configurations (radius=10, radius=30, incremental x2, neighbors, hierarchical)"
echo "RK4 tracking: 100 steps with inverse point-in-tet"
echo "Expected runtime: ~45 minutes"
echo ""

LOG_FILE_2="logs/benchmark_l2_search_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOG_FILE_2"
echo ""

python benchmark_l2_search_methods.py 2>&1 | tee "$LOG_FILE_2"

RESULT_2=$?

if [ $RESULT_2 -eq 0 ]; then
    echo ""
    echo "✅ L2 Search Benchmark Complete!"
    echo "   Log: $LOG_FILE_2"
    echo ""

    # Extract production config result
    echo "Quick Results:"
    grep "Production config achieves" "$LOG_FILE_2" || echo "   (see log for details)"
    echo ""
else
    echo ""
    echo "❌ L2 Search Benchmark Failed (exit code: $RESULT_2)"
    echo "   Check log: $LOG_FILE_2"
    exit $RESULT_2
fi

# Summary
echo ""
echo "================================================================================"
echo "ALL BENCHMARKS COMPLETE!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  - Point-in-Tet: $LOG_FILE_1"
echo "  - L2 Search:    $LOG_FILE_2"
echo ""
echo "Next steps:"
echo "  1. Review logs for detailed results"
echo "  2. Extract performance numbers for paper"
echo "  3. See BENCHMARK_GUIDE.md for interpretation"
echo ""
echo "Key metrics to look for:"
echo "  - Point-in-tet 'inverse' method speedup (expected: 4.36×)"
echo "  - L2 incremental speedup (expected: 1.8-2.8×)"
echo "  - Combined speedup (expected: 7.8-12×)"
echo ""
echo "================================================================================"
echo "Date: $(date)"
echo "================================================================================"
