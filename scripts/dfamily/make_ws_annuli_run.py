#!/usr/bin/env python3
"""Build the TARGETED D2 diagnostic run script for the workstation.

Differences from the full-domain run, and why:
  seed-source file   -- seeds only the two radial shells where the 21,429
                        interior losses occur (r 2.2-3.6 and 6.2-7.6 mm), so
                        the failure appears within tens of steps instead of
                        ~900 and every particle is where the defect lives.
  N_STEPS 600        -- enough for the loss to develop in-shell; the full run
                        needed 8000 only because particles had to advect in.
  LOG_INTERVAL 10    -- dense progress; the template already uses `python -u`
                        so the log is live rather than block-buffered.
  EXPORT_FREQ 10     -- per-step geometry is not needed, loss counts come from
                        the per-step log line and search_stats.csv.
"""
import re, sys
path, case, seeds = sys.argv[1:4]
s = open(path, encoding="utf-8").read()

def setvar(src, var, val):
    pat = re.compile(rf'^({var}=)("(?:[^"\\]|\\.)*"|\S*)(.*)$', re.M)
    out, n = pat.subn(lambda m: f"{m.group(1)}{val}{m.group(3)}", src, count=1)
    if not n:
        print(f"  WARN: {var}= not found", file=sys.stderr)
    return out

for k, v in [
    ("INPUT",              f'"{case}"'),
    # cylA keeps its meshes in post/0eule/ and its FEMUSS parts in post/1part/.
    # The D-family cases keep A2_<ts>.pvtu directly in post/, so both subdirs
    # must be cleared or the loader looks in post/0eule/ and fails.
    ("MESH_SUBDIR",        '""'),
    # keep diagnostic output out of post_pt/ so it cannot be confused with a
    # production tracking result for this case
    ("OUTPUT_CASE_SUBFOLDER", "post_pt_annuli"),
    ("FEMUSS_SUBDIR",      '""'),
    ("MESH_PATTERN",       '"A2_{timestep}.pvtu"'),
    ("VEL_START",          "34"),
    ("VEL_END",            "199"),
    ("N_STEPS",            "600"),
    ("DT",                 "0.0003629"),
    ("PIN_RPM",            "-600"),
    ("LEVELSET_MODE",      "zero_vel"),
    ("BOUNDARY_WALLS",     '"x_max=clamp"'),
    ("SEED_SOURCE",        "file"),
    ("SEED_FILE",          f'"{seeds}"'),
    ("EXPORT_ELEMENT_IDS", "1"),
    ("EXPORT_FREQ",        "10"),
    ("LOG_INTERVAL",       "10"),
]:
    s = setvar(s, k, v)

# SEED_FILE may not exist in the template; append it near SEED_SOURCE if so.
if "SEED_FILE=" not in s:
    s = re.sub(r"^(SEED_SOURCE=.*)$",
               lambda m: f'{m.group(1)}\nSEED_FILE="{seeds}"   # added: --seed-source file',
               s, count=1, flags=re.M)
    print("  SEED_FILE= was absent; inserted after SEED_SOURCE")

open(path, "w", encoding="utf-8").write(s)
print("  patched")
