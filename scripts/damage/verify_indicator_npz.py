"""Internal-consistency checks on a damage_models.npz indicator archive.

    python3 verify_indicator_npz.py <damage_models.npz>

Tests relationships that MUST hold if the kernel accumulated correctly, rather
than reprinting what the run log already said. Independent of the run itself, so
it catches a silently wrong accumulator that still produces plausible numbers.

⚠️ TOLERANCE NOTE. t_res is a running sum of dt over 8000 float32 additions, so
it is NOT an exact integer multiple of dt: repeated addition drifts by up to
~0.48 dt by step 8000 (reproduced exactly: 0.367 at n=6453 in the D2 archive).
Storage dtype is not the cause -- a perfect float32 n*dt sequence has zero drift.
The check is therefore on RELATIVE error (~5.7e-05 measured), not exact
divisibility. An earlier version of this script tested exact divisibility and
reported a false failure.
"""
import sys
import numpy as np
z = np.load(sys.argv[1] if len(sys.argv) > 1 else "/tmp/d2_ws.npz",
            allow_pickle=True)
d = z["damage"]; cols=[str(c) for c in z["columns"]]
ln, C, tres, w = (d[:, cols.index(k)] for k in ("ln_phi","C","t_res","w_bond"))
n_steps=int(z["n_steps"]); dt=float(z["dt"]); thr=float(z["residence_edot_threshold"])
T = n_steps*dt
print(f"  n={len(d)}  n_steps={n_steps} dt={dt} T={T:.4f}s thr={thr} 1/s")
ok=True

# 1. t_res is a TIME inside [0, T] and must be an exact multiple of dt
bad = (tres < -1e-12) | (tres > T + 1e-9)
print(f"  t_res in [0,T]                 : {'OK' if not bad.any() else f'FAIL {bad.sum()}'}")
ok &= not bad.any()
k = tres/dt
frac = np.abs(k - np.round(k))
rel = frac/np.maximum(k, 1.0)
print(f"  t_res ~ integer x dt (rel)     : max rel {rel.max():.2e} "
      f"(abs {frac.max():.3f} dt, float32 drift)  "
      f"{'OK' if rel.max() < 1e-3 else 'FAIL'}")
ok &= rel.max() < 1e-3

# 2. Both damage laws are non-negative accumulators (integrands >= 0)
print(f"  ln_phi >= 0                    : {'OK' if (ln>=-1e-9).all() else 'FAIL'}")
print(f"  C      >= 0                    : {'OK' if (C >=-1e-9).all() else 'FAIL'}")
ok &= (ln>=-1e-9).all() and (C>=-1e-9).all()

# 3. w_bond = -int(eta)dt with eta clamped to +-3  =>  |w| <= 3T
print(f"  |w_bond| <= 3T ({3*T:.3f})         : max {np.abs(w).max():.4f}  "
      f"{'OK' if np.abs(w).max() <= 3*T+1e-6 else 'FAIL'}")
ok &= np.abs(w).max() <= 3*T+1e-6

# 4. A particle that never entered the layer did little work: t_res==0 should
#    correlate with LOW ln_phi. Check the medians separate.
z0, z1 = ln[tres==0], ln[tres>0]
print(f"  median ln_phi: t_res==0 {np.median(z0):.4f} vs t_res>0 {np.median(z1):.4f}"
      f"  {'OK (separates)' if np.median(z1) > np.median(z0) else 'SUSPICIOUS'}")
ok &= np.median(z1) > np.median(z0)

# 5. Positions finite and inside a plausible workpiece
p = z["positions"]
print(f"  positions finite               : {'OK' if np.isfinite(p).all() else 'FAIL'}")
print(f"    x [{p[:,0].min()*1e3:.1f},{p[:,0].max()*1e3:.1f}] y [{p[:,1].min()*1e3:.1f},"
      f"{p[:,1].max()*1e3:.1f}] z [{p[:,2].min()*1e3:.1f},{p[:,2].max()*1e3:.1f}] mm")
ok &= np.isfinite(p).all()
frac_pos = 100*(w > 0).mean()
print(f"  w_bond > 0                     : {frac_pos:.1f} %  "
      f"(compression under the shoulder should dominate)")
print("\n" + ("ALL INTERNAL CHECKS PASS" if ok else "⚠️ SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
