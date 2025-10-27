#!/bin/bash
# Quick Reference Commands for Tomorrow's Session
# Date: October 9, 2025

echo "========================================="
echo "TEMPORAL BATCHING - TEST COMMANDS"
echo "========================================="
echo ""

# Navigate to project directory
cd /home/arhashemi/Workspace/welding/JAXTrace

# Activate virtual environment
source .venv/bin/activate

echo "Option 1: View Today's Run Log"
echo "------------------------------"
echo "cat logs/temporal_batching_run_20251009_094310.log | less"
echo ""

echo "Option 2: View Status Report"
echo "----------------------------"
echo "cat logs/RUN_STATUS_20251009.md | less"
echo ""

echo "Option 3: Try CPU-Only Mode (Should Work)"
echo "-----------------------------------------"
echo "# Edit example_workflow.py: 'device': 'cpu'"
echo "python example_workflow.py > logs/cpu_mode_run.log 2>&1"
echo ""

echo "Option 4: Test with Octree FEM (Proven Working)"
echo "-----------------------------------------------"
echo "# Edit example_workflow.py: 'use_temporal_batching': False"
echo "python example_workflow.py > logs/octree_run.log 2>&1"
echo ""

echo "Option 5: Test Temporal Batching with Synthetic Data"
echo "----------------------------------------------------"
echo "python test_temporal_batching.py"
echo ""

echo "Option 6: Check Current Configuration"
echo "-------------------------------------"
echo "python check_config.py"
echo ""

echo "Option 7: Monitor Long-Running Process"
echo "--------------------------------------"
echo "# Start in background:"
echo "nohup python example_workflow.py > logs/run_\$(date +%Y%m%d_%H%M%S).log 2>&1 &"
echo "# Check progress:"
echo "tail -f logs/run_*.log"
echo "# Check if still running:"
echo "ps aux | grep python | grep example_workflow"
echo ""

echo "========================================="
echo "CURRENT STATUS"
echo "========================================="
echo ""
echo "✅ Temporal batching: ENABLED (in example_workflow.py)"
echo "✅ Integration: COMPLETE"
echo "✅ Algorithm: CORRECT"
echo "⚠️  Issue: GPU memory exhaustion with large meshes"
echo ""
echo "Current Settings:"
echo "  - use_temporal_batching: True"
echo "  - temporal_window_size: 3"
echo "  - grid_resolution: 16"
echo "  - particles: 18,000 (30 × 40 × 15)"
echo "  - timesteps: 1,000"
echo ""
echo "Last Run Result:"
echo "  - Status: GPU OOM after loading 3 timesteps"
echo "  - Memory needed: 1.4 GB"
echo "  - GPU limit: 3.0 GB"
echo "  - Issue: Even window size=3 exceeds memory"
echo ""

echo "========================================="
echo "RECOMMENDED NEXT STEPS"
echo "========================================="
echo ""
echo "1. Review logs/RUN_STATUS_20251009.md for detailed analysis"
echo "2. Discuss which approach to take:"
echo "   A) CPU-only mode (slow but works)"
echo "   B) Implement streaming (best, needs 4-6 hours)"
echo "   C) Use octree FEM for this dataset (works now)"
echo "3. Test with small synthetic data (should work)"
echo ""

echo "Files for Review:"
echo "  - logs/temporal_batching_run_20251009_094310.log (full output)"
echo "  - logs/RUN_STATUS_20251009.md (analysis)"
echo "  - TEMPORAL_BATCHING_ISSUES.md (solutions)"
echo "  - example_workflow.py (line 1814: temporal_batching enabled)"
echo ""

echo "========================================="
