# Testing Checklist - RAM Fix Phases

**Current Status**: ✅ Phase 1 Implemented | 📝 Phase 2 Ready

---

## Quick Testing Guide

### Phase 1 Tests (Run These First)

```bash
# Test 1: Radius (baseline - should work)
sed -i "s/L2_SEARCH_METHOD = .*/L2_SEARCH_METHOD = 'radius'/" production_tracking_fully_fused_timedep.py
python production_tracking_fully_fused_timedep.py > logs/phase1_radius.log 2>&1

# Test 2: Neighbors (critical test)
sed -i "s/L2_SEARCH_METHOD = .*/L2_SEARCH_METHOD = 'neighbors'/" production_tracking_fully_fused_timedep.py
python production_tracking_fully_fused_timedep.py > logs/phase1_neighbors.log 2>&1

# Test 3: Hierarchical (may need Phase 2)
sed -i "s/L2_SEARCH_METHOD = .*/L2_SEARCH_METHOD = 'hierarchical'/" production_tracking_fully_fused_timedep.py
python production_tracking_fully_fused_timedep.py > logs/phase1_hierarchical.log 2>&1
```

---

## Decision Tree

```
Phase 1 Test Results:
│
├─ Test 1 (radius) ✅ + Test 2 (neighbors) ✅ + Test 3 (hierarchical) ✅
│  └─> SUCCESS! All methods work. Phase 2 not needed. 🎉
│
├─ Test 1 (radius) ✅ + Test 2 (neighbors) ✅ + Test 3 (hierarchical) 🔴
│  └─> PARTIAL SUCCESS. Use radius/neighbors in production.
│     └─> Optional: Implement Phase 2 if you need hierarchical.
│
├─ Test 1 (radius) ✅ + Test 2 (neighbors) 🔴
│  └─> NEED PHASE 2. Report back for immediate implementation.
│
└─ Test 1 (radius) 🔴
   └─> UNEXPECTED! Check for other issues (GPU memory, system RAM, etc.)
```

---

## Expected RAM Usage

| Test | Phase 1 RAM | Phase 2 RAM | System Limit |
|------|-------------|-------------|--------------|
| Radius | 11 GB | 11 GB | ✅ Any system |
| Neighbors | 275 GB | 92 GB | ⚠️ Needs 512 GB |
| Hierarchical | 1.46 TB | 183 GB | 🔴 Needs 2 TB → ✅ 512 GB |

**Your system**: Check available RAM with `free -h`

---

## What to Look For

### Success Indicators:
```
Compiling RK4 step... (this is where RAM spike happens)
✅ Compilation complete (XX seconds)
✅ Running timestep 0...
✅ Particle retention: XX%
```

### Failure Indicators:
```
Compiling RK4 step...
🔴 Killed (OOM - out of memory)
```

OR in system logs:
```bash
sudo dmesg | tail -20
🔴 "Out of memory: Killed process ... (python)"
```

---

## Quick Check Commands

### Monitor RAM during compilation:
```bash
# In separate terminal while test runs:
watch -n 1 'free -h && ps aux | grep python | grep production | grep -v grep'
```

### Check if test crashed:
```bash
tail -50 logs/phase1_neighbors.log
# Look for "Killed" or "Out of memory"
```

### Check compilation success:
```bash
grep -i "compil" logs/phase1_neighbors.log
# Should see "Compilation complete" not "Killed"
```

---

## Reporting Results

When reporting results, include:

1. **Which test failed/succeeded**:
   - ✅ Radius: Worked
   - 🔴 Neighbors: Crashed during compilation
   - (not tested) Hierarchical: ...

2. **System info** (if crashed):
   ```bash
   free -h  # Total/available RAM
   nvidia-smi  # GPU memory
   ```

3. **Error message** (if crashed):
   ```bash
   tail -100 logs/phase1_neighbors.log
   sudo dmesg | tail -50 | grep -i "out of memory"
   ```

---

## Files Reference

- **Phase 1 Details**: [PHASE1_FIX_SUMMARY.md](PHASE1_FIX_SUMMARY.md)
- **Phase 2 Details**: [PHASE2_FIX_READY.md](PHASE2_FIX_READY.md)
- **Analysis**: [RAM_EXPLOSION_ANALYSIS.md](RAM_EXPLOSION_ANALYSIS.md)
- **Current File**: [morton_global_search.py:455-503](jaxtrace/gpu/search/morton_global_search.py#L455-L503)

---

## Ready to Test!

Start with **Test 1** (radius) to verify no regression, then proceed to **Test 2** (neighbors) which is the critical test. Report results and we'll proceed accordingly! 🚀
