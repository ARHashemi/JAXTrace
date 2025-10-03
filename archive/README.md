# Archive Folder

This folder contains development artifacts, debug scripts, and documentation from the development process. These files are kept in the repository for reference but are not included in production releases or pull requests to main.

## 📁 Folder Structure

```
archive/
├── dev_docs/              # Development documentation
├── old_examples/          # Previous example workflows
├── debug_scripts/         # Debug and profiling scripts
├── output_archives/       # Old output directories
├── memory_reports/        # Memory profiling data
└── old_implementation/    # Previous implementation versions
```

## 📚 Development Documentation (`dev_docs/`)

Detailed guides and documentation created during development:

- **FEM and Octree Guides:**
  - `FEM_INTERPOLATION_GUIDE.md` - FEM interpolation explanation
  - `FEM_STATUS.md` - FEM implementation status
  - `FEM_WORKFLOW_EXPLAINED.md` - Detailed FEM workflow
  - `README_FEM.md` - FEM feature README
  - `README_OCTREE_FEM.md` - Octree FEM documentation
  - `OCTREE_OPTIMIZATION_SUMMARY.md` - Optimization details

- **Performance and Optimization:**
  - `GPU_OPTIMIZATION_GUIDE.md` - GPU optimization strategies
  - `PERFORMANCE_OPTIMIZATION_GUIDE.md` - General performance tuning
  - `QUICK_START_FAST.md` - Fast workflow guide

- **Configuration and Setup:**
  - `CONFIGURATION_GUIDE.md` - Comprehensive config reference
  - `QUICK_CONFIG_REFERENCE.md` - Quick config examples
  - `USER_CONFIGURATION_SUMMARY.md` - Config overview
  - `ENTRYPOINTS.md` - Entrypoint documentation
  - `REPO_SETUP_SUMMARY.md` - Repository setup details

- **Technical Details:**
  - `BOUNDARY_CONDITION_JIT_FIX.md` - JIT compilation fixes
  - `JIT_COMPILATION_FIX.md` - JIT troubleshooting
  - `WORKFLOW_UPDATE_SUMMARY.md` - Update summary

- **Images:**
  - `integration_accuracy.png` - Integration accuracy benchmark
  - `performance_benchmark.png` - Performance comparison
  - `seeding_strategies.png` - Particle seeding visualization

## 🔬 Old Examples (`old_examples/`)

Previous versions of example workflows showing evolution of features:

- `example_workflow_fast.py` - Fast workflow version
- `example_workflow_fem.py` - FEM interpolation example
- `example_workflow_memory_optimized.py` - Memory optimization focus
- `example_workflow_octree_fem.py` - Initial octree FEM
- `example_workflow_octree_fem_optimized.py` - Optimized octree FEM

These show the progression toward the final `example_workflow.py`.

## 🐛 Debug Scripts (`debug_scripts/`)

Scripts used for debugging and profiling during development:

- `diagnose_jit_issue.py` - JIT compilation diagnostics
- `test_gpu_memory.py` - GPU memory testing
- `test_memory_optimization.py` - Memory optimization tests
- `test_vtk_loading.py` - VTK data loading tests
- `fem_test_output.txt` - FEM test results
- `fem_workflow_output.txt` - FEM workflow output
- `octree_test.txt` - Octree test results

## 📊 Memory Reports (`memory_reports/`)

Memory profiling data from development:

- `memory_report_20250930_*.json` - Daily memory reports
- `memory_report_20251001_*.json` - Additional profiling
- `memory_tracking_detailed.json` - Detailed memory tracking
- `test_memory_tracking.json` - Memory test results

## 📤 Output Archives (`output_archives/`)

Previous run outputs:

- `output/` - Main output directory
- `output_jaxtrace/` - JAXTrace specific outputs
- `output_jaxtrace_enhanced/` - Enhanced output version
- `output_memory_optimized/` - Memory optimized outputs
- `output_minimal/` - Minimal example outputs

## 🏗️ Old Implementation (`old_implementation/`)

Previous implementation versions before major refactoring.

## ⚠️ Important Notes

1. **Not for Production:** These files are kept for reference only.

2. **Not in Pull Requests:** The archive folder is excluded from pull requests to main via proper git workflow.

3. **Reference Material:** Useful for:
   - Understanding development decisions
   - Debugging similar issues
   - Learning about optimization process
   - Historical context

4. **Maintenance:** Files here are not actively maintained and may be outdated.

## 🔍 How to Use

### View Archived Documentation
```bash
# Browse available docs
ls archive/dev_docs/

# Read specific guide
cat archive/dev_docs/README_OCTREE_FEM.md
```

### Use Old Example
```bash
# Copy to root if needed
cp archive/old_examples/example_workflow_fast.py .

# Run it
python example_workflow_fast.py
```

### Check Debug Scripts
```bash
# Run debug script
python archive/debug_scripts/diagnose_jit_issue.py
```

### Review Memory Reports
```bash
# View memory profiling
cat archive/memory_reports/memory_report_20251001_124656.json | jq .
```

## 📝 Adding to Archive

When archiving new development files:

```bash
# Documentation
mv my_dev_doc.md archive/dev_docs/

# Examples
mv old_example.py archive/old_examples/

# Debug scripts
mv debug_script.py archive/debug_scripts/

# Outputs
mv output_mytest/ archive/output_archives/

# Memory reports
mv memory_report_*.json archive/memory_reports/
```

## 🚫 Git Ignore

The archive folder is in `.gitignore` for development branches, but specific subdirectories (like `dev_docs/` and `debug_scripts/`) are tracked for reference on the development branch.

**In `.gitignore`:**
```
# Archive folder (excluded from main branch PR)
archive/
```

**For PR workflow:**
- Archive exists on `memory_optimization` branch
- Archive is excluded when merging to `main`
- Clean main branch has only production files
