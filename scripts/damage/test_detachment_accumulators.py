"""Standalone check of the two new accumulator columns, no mesh needed.

Replicates EXACTLY the kernel's update order:
    dmg_new = dmg + dt*inc          (inc[6] == 0)
    dmg_new = dmg_new.at[6].set(min(dmg[6], tau_k1))
then the freeze. Verifies:
  1. ln_J integrates div(u) and exp(ln_J) recovers det F
  2. tau_min is a true running MINIMUM, not polluted by the +inf start
  3. the freeze cannot latch a minimum from a skipped step
"""
import jax, jax.numpy as jnp, numpy as np

DT = 1e-4
N  = 50

def step(dmg, divu, tau, skipped):
    inc = jnp.stack([jnp.zeros(()), jnp.zeros(()), jnp.zeros(()),
                     jnp.zeros(()), jnp.zeros(()), divu, jnp.zeros(())])
    dmg_new = dmg + DT * inc
    dmg_new = dmg_new.at[6].set(jnp.minimum(dmg[6], tau))
    dmg_new = jnp.where(skipped, dmg, dmg_new)   # freeze, as in the kernel
    return dmg_new

# ---- 1 & 2: constant divergence, a dipping tau ----
dmg = jnp.zeros(7).at[6].set(jnp.inf)
DIVU = -3.0                       # 1/s, compressive
taus = np.linspace(1.0, 0.25, N)  # monotonically falling
for k in range(N):
    dmg = step(dmg, DIVU, float(taus[k]), False)
ln_J = float(dmg[5]); detF = np.exp(ln_J); tau_min = float(dmg[6])
exp_lnJ = DIVU * DT * N
print(f"1. ln_J      = {ln_J:+.8e}   expected {exp_lnJ:+.8e}")
# ⚠️ Tolerance is set by the PRODUCTION dtype (float32), not by exact
# arithmetic: 50 sequential adds accumulate ~5x float32 eps. A 1e-12
# tolerance here would be testing float64, which the run does not use.
assert abs(ln_J - exp_lnJ) < 1e-6 * abs(exp_lnJ) + 1e-12, "ln_J does not integrate div(u)"
print(f"   det F     = {detF:.8f}       expected {np.exp(exp_lnJ):.8f}")
print(f"2. tau_min   = {tau_min:.6f}        expected {taus.min():.6f}")
assert abs(tau_min - taus.min()) < 1e-6, "tau_min is not the running min"
assert np.isfinite(tau_min), "the +inf start leaked into the result"

# ---- 2b: tau RISING -- the min must stay at the FIRST value ----
dmg = jnp.zeros(7).at[6].set(jnp.inf)
rising = np.linspace(0.3, 0.9, N)
for k in range(N):
    dmg = step(dmg, 0.0, float(rising[k]), False)
print(f"2b. rising tau -> min = {float(dmg[6]):.6f}  expected {rising.min():.6f}")
assert abs(float(dmg[6]) - rising.min()) < 1e-6

# ---- 3: a skipped step carrying a LOW tau must NOT latch ----
dmg = jnp.zeros(7).at[6].set(jnp.inf)
dmg = step(dmg, 0.0, 0.8, False)    # legitimate
before = float(dmg[6])
dmg = step(dmg, -99.0, 0.01, True)  # skipped: tiny tau AND huge divergence
print(f"3. after skipped step: tau_min={float(dmg[6]):.6f} (was {before:.6f}), "
      f"ln_J={float(dmg[5]):+.3e}")
assert abs(float(dmg[6]) - before) < 1e-12, "a skipped step latched the minimum"
assert abs(float(dmg[5])) < 1e-12, "a skipped step contributed to ln_J"

# ---- 4: div(u)=0 gives det F == 1 exactly, the null result ----
dmg = jnp.zeros(7).at[6].set(jnp.inf)
for k in range(1000):
    dmg = step(dmg, 0.0, 0.5, False)
print(f"4. div(u)=0 over 1000 steps -> det F = {np.exp(float(dmg[5])):.17f}")
assert np.exp(float(dmg[5])) == 1.0, "det F != 1 for a divergence-free field"

print("\nALL PASS — the accumulator semantics are correct.")
