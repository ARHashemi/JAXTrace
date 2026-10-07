# Staged Validation Plan — FOM → grid → ROM → ROM-on-grid

**Companion to `IMPLEMENTATION_PLAN_void_damage.md`.**
Written 2026-09-23, against data inspected on the live mount.

Four stages, each isolating one error source. The point of the ladder is that when a
later stage disagrees with an earlier one, **you know which transformation caused it.**

---

## 0. What was found in the data (verified, not assumed)

Inspected `cylindrical_000.gid/post/cylindrical_119.pvtu` directly.

### Fields present

| Field | Components | Range | Note |
|---|---|---|---|
| `Displacement` | 3 | ±0.45 m/s | ✅ **is velocity** — §0.1 |
| `Pressure` | 1 | −7.40e7 … +6.44e7 Pa | **present — the gate is passed** |
| `Reactions` | 3 | ±144 | |
| `Temperature` | 1 | 21 … 451 °C | for Sellars–Tegart |
| `LEVEL` | 1 | −0.0032 … 0.0255 | level-set, <0 = inside tool |

Mesh: **180,461 points, 736,697 tetrahedra** (cell type 10). Matches the 180461 node
count in the ROM memory notes.

### ✅ 0.1 "Displacement" is the velocity field — CONFIRMED

Confirmed by the user 2026-09-23: the array named `Displacement` **is** velocity, in m/s.
No unit conversion, no division by $dt$. Use it directly as $\mathbf{u}$ when computing
$\nabla\mathbf{u}$. The loader already defaults to `field_name='Displacement'`
([`mesh_loader_timedep.py:20`](../JAXTrace/jaxtrace/gpu/mesh_loader_timedep.py#L20)).

### ⚠️ 0.2 Pressure sign — the assumption in the primer was WRONG

My primer said the stir zone is "uniformly +50 to +200 MPa compressive". **The data
says otherwise:**

- 59.7 % of nodes positive; median **+1.29 MPa**; percentiles (1/25/50/75/99) =
  −37 / −4.1 / +1.3 / +4.5 / +29.7 MPa
- Roughly symmetric about zero — this is a **deviatoric/incompressible-solve pressure**,
  not an absolute hydrostatic pressure including the forge load.

**This is good news for the method.** A uniformly compressive field would clamp
Rice–Tracey everywhere and give a flat, useless map. A signed field discriminates.

### ✅ 0.3 The pressure field shows the expected dipole

Angular median of Pressure in an 8 mm annulus around the tool axis (at origin):

| sector | P median |
|---|---|
| [−90°, −45°) | **−17.6 MPa** ← strongly negative |
| [−45°, 0°) | −8.3 MPa |
| [0°, +45°) | **+9.0 MPa** ← strongly positive |
| [+45°, +90°) | +7.4 MPa |
| [+90°, +180°) | +2.0 MPa |

A clean dipole across the tool. **This is the forging-ahead / release-behind asymmetry**
the whole filling mechanism rests on, visible directly in the raw FOM pressure.

> **Immediate consequence:** $\eta$ will be strongly positive on one side and negative on
> the other. Rice–Tracey's $\exp(3\eta/2)$ will separate them by a factor of
> $\exp(1.5 \times (9+17.6)/\sigma_{\text{eq}})$ — a large dynamic range. The signal
> should be visible.

### ✅ 0.3b Orientation and pressure sign — RESOLVED from the data (2026-09-23)

Both remaining checks are now settled empirically on case 000. **Record these in code.**

**Parameters:** `INLET_VELOCITY = +5.0e-3` m/s, `PIN_RPM = -400`, `DT = 3.75e-3`.

**Measured, not assumed:**

| Measurement | Result |
|---|---|
| Tangential velocity, ring $r\in[4.0,5.5]$ mm | **negative in every sector** → tool rotates **clockwise** ($\omega_z<0$), consistent with `PIN_RPM = -400` |
| Far-field median velocity ($L>8$ mm) | $(+0.00498, -0.00015, 0)$ m/s → **material flows $+x$**; the tool advances $-x$ in the workpiece frame |
| Domain | inlet $x_{\min}=-0.015$, outlet $x_{\max}=+0.030$, tool axis at origin |

**⇒ AS/RS labelling.** ⚠️ *A first attempt at this got it backwards; the definition is
relative to the* **tool travel direction**, *not to the material flow direction.*

In this simulation frame the workpiece flows $+x$ past a stationary tool, so in the
workpiece frame **the tool travels $-x$**. Advancing is the flank where the tool surface
velocity is *parallel* to travel, i.e. where $u_x < 0$.

Measured tool-surface $u_x$ at the flanks ($|x|<1$ mm, $r\in[4.0,5.0]$ mm):

| flank | $u_x$ median | parallel to travel ($-x$)? | |
|---|---|---|---|
| $+y$ | **+0.093** m/s | no | RETREATING |
| $-y$ | **−0.050** m/s | yes | **ADVANCING** |

$$\boxed{\ \text{ADVANCING} = -y\ (\theta\in(-180°,0°)), \qquad \text{RETREATING} = +y\ (\theta\in(0°,180°))\ }$$

**⇒ Pressure sign.** Comparing upstream (material approaching, forged) with downstream
(wake, released) in the same ring:

| region | $P$ median |
|---|---|
| upstream, $x<-2$ mm (forged) | **+1.70 MPa** |
| downstream, $x>+2$ mm (wake) | **−1.35 MPa** |

$P$ is **high where material is forged, low in the wake** — so `Pressure` is a compressive
pressure in the fluid convention. With solid-mechanics tension-positive convention:

$$\boxed{\ \sigma_m = -P\ }$$

giving $\sigma_m < 0$ (compressive) ahead of the tool and $\sigma_m > 0$ (tensile) in the
wake — exactly the condition for voids to open behind the tool.

> **The falsifiable prediction this buys:** damage should be **tensile in the wake**, and
> should peak in the wake on the **advancing ($-y$) side**.

### ✅ 0.3c Stage 1 acceptance test — PASSED (2026-09-23)

Ran `jaxtrace/damage/fields.py` on case 000 (`cylindrical_119.pvtu`, 180,461 nodes,
736,697 tets) with $\mu_{\text{eff}} = 10^6$ Pa·s. Triaxiality in a ring
$r\in[4,8]$ mm, $\dot\varepsilon>1$ s⁻¹:

| quadrant | $\eta$ median | frac $\eta>0$ | Rice–Tracey $\exp(1.5\eta)$ |
|---|---|---|---|
| **ADVANCING ($-y$), wake ($x>0$)** | **+0.080** | **0.943** | **1.127** (mean 1.312) |
| advancing, ahead | −0.002 | 0.409 | 0.997 |
| retreating ($+y$), wake | −0.053 | 0.009 | 0.924 |
| retreating, ahead | −0.004 | 0.267 | 0.993 |

**The tensile region is sharply localised to the advancing-side wake** — 94 % of nodes
there are in tension, versus 1 % in the retreating wake. That is the void-formation
signature the whole literature reports, recovered here from the FOM fields alone with
no fitting.

Note this was derived *independently* of the AS/RS labelling above and then found to
agree with it — two separate measurements (tool-surface velocity direction; triaxiality
sign) pointing at the same flank. That mutual consistency is the real result.

### 0.4 Steady state — final step only

Confirmed by the user: all these cases are steady, so **one snapshot per case is the
whole dataset.** Consistent with `our-pod-basis`: case 000 differs by 0.003 % between
ts118 and ts119.

Consequences, all simplifying:

- **No time interpolation is exercised** in the damage fields for now — but keep the
  `(n_timesteps, n_nodes)` layout with `n_timesteps == 1` (see §0.4c), not a
  steady-only `(n_nodes,)` array.
- **Memory is 2.2 MB** for three scalars at 180k nodes with `n_timesteps == 1`
  (vs 260 MB at 120 timesteps). Not a concern until transient fields arrive.
- Pathline integration is through a *frozen* field — a particle's damage is a pure line
  integral along a steady streamline. Exactly He et al.'s setting.

### 0.4b What "steady" does and does not exclude

An earlier draft said "a steady field cannot show a cavity opening and refilling", which
was too strong and needs stating precisely.

**Steady is a statement about the tool frame, not about material.** In the co-moving,
co-rotating frame a smooth cylindrical pin maps onto itself under rotation, so the
boundary conditions are time-independent and the FOM converges to $\partial(\cdot)/\partial t = 0$.
But for a **material point**:

$$\frac{D(\cdot)}{Dt} = \underbrace{\frac{\partial(\cdot)}{\partial t}}_{=\,0\ \text{(steady)}} \;+\; \underbrace{\mathbf{u}\cdot\nabla(\cdot)}_{\neq\,0}$$

The local derivative vanishes; the **advective** term does not. A particle travelling along
a streamline through a frozen field still experiences a *time-varying history* — forged as
it passes the leading side, released in the wake. Exactly the dipole measured in §0.3.

> **So a steady field does produce material displacement and refilling, and can produce
> voids.** Pathline integration captures this fully. This is precisely He et al.'s setting,
> and it is the setting of the classic steady FSW CFD literature (Seidel & Reynolds,
> Zhao, Almoussawi).

**What steady genuinely excludes** is the *per-revolution periodic* cavity:

| | Steady (smooth cylinder) | Transient (threaded / featured) |
|---|---|---|
| Symmetry | axisymmetric pin → tool-frame steady | threads break axisymmetry → $\partial_t \neq 0$ at $\omega$ |
| Void mechanism | **continuous stagnation / flow deficiency** | + cavity opens and partly refills each revolution |
| Literature | Morisada (X-ray stagnation), Zhao, Seidel | Ghate (per-rev, 2D), Dialami (tracers, rotating tool) |
| Captured by steady pathlines? | ✅ **fully** | ❌ needs a time-resolved field |

Both mechanisms put voids on the advancing side. The steady route is not a weaker
mechanism — it is the one the current cohort actually has.

⚠️ **Framing rule:** the steady damage field measures **stress-state and flow-history
susceptibility**. Do not describe it as capturing Ghate's per-revolution filling deficit —
that is a different (and additional) mechanism requiring §0.4c.

<sub>Note: a smooth cylindrical pin is the *more* defect-prone geometry, not less. Threads
exist to drive material downward and improve filling. He et al. report ~10× more porosity
with a threaded pin, but via increased deformation and heat input, not because smoothness
protects against voids.</sub>

### 0.4c Design requirement — keep the time-dependent path open

Current cases are steady, but **threaded tools with time-dependent fields are planned**.
The damage machinery must therefore use the *same* time-indexing contract as velocity, not
a steady-only shortcut.

The kernel already does this
([`rk4_fully_fused_timedep.py:637`](../JAXTrace/jaxtrace/gpu/tracking/rk4_fully_fused_timedep.py#L637)):

```python
n_timesteps = velocity_fields_gpu.shape[0]
vel_idx = time_idx % n_timesteps
velocity_field = velocity_fields_gpu[vel_idx]
```

> **Mirror this exactly for the damage scalars.** Store them as `(n_timesteps, n_nodes)`
> and index with the same `time_idx % n_timesteps`.
>
> **A steady case is then just `n_timesteps == 1`**: the modulo makes every step read
> index 0, with no branch, no special case, and no separate code path. Memory is
> 0.72 MB per scalar for a steady case and grows naturally when transient fields arrive.

This costs nothing now and avoids a rewrite later. **Do not write a `(n_nodes,)`
steady-only variant.**

Consequences for the loader: `load_velocity_sequence_from_pvtu` already takes
`field_name` ([`mesh_loader_timedep.py:20`](../JAXTrace/jaxtrace/gpu/mesh_loader_timedep.py#L20)),
so loading `Pressure` and `Temperature` sequences reuses the existing path — same
`timestep_range`, same topology check, same skip-forward logic.

⚠️ **When transient fields do arrive**, two further points need attention:
1. **Field cadence vs tracking `dt`.** The damage sample must come from the same
   time-interpolated state the velocity does; a mismatch aliases the damage history.
2. **The per-revolution cavity needs resolving in time** — several samples per tool
   revolution, not per weld-length. That is a sampling-rate requirement on the FOM export,
   not something the tracker can recover after the fact.

### 0.5 Two different data layouts

| Source | Timesteps with Pressure |
|---|---|
| `cylA.gid` (single case) | **only ts=159** (the last) |
| `cylindrical_0NN.gid` (20 cases) | **all 120** |

Since only the final step is needed, both work. `cylA` at ts=159; the cohort at ts=119.

---

## Field diagnostics on the FOM mesh (implementation plan phases 2–3, partial)

<sub>Written when this was called "Stage 1". It corresponds to parts of
`IMPLEMENTATION_PLAN_void_damage.md` phases 2 and 3, built ahead of phase 1.</sub>

**Purpose:** establish ground truth. Everything downstream is compared against this.
No projection, no ROM, no interpolation error beyond the FE basis itself.

### 1.1 Single case first: `cylA.gid`, ts=159

```
/home/arhashemi/fsw-gpu/flash/users/ali/data/cylA.gid/post/0eule/cylA_159.pvtu
```

Steps:

1. **Resolve §0.1** — is `Displacement` velocity? Check against DT and $\omega R$.
2. **Resolve §0.3 orientation** — label AS/RS from travel direction and $\omega$ sign.
3. Compute $\nabla u$ on the tet mesh. For linear tets the gradient is **constant per
   element** and exact — no finite differences needed:
   $\nabla u|_e = \sum_{i=0}^{3} u_i \otimes \nabla N_i$, where $\nabla N_i$ come from the
   inverse Jacobian of the tet. Then average element values to nodes (volume-weighted).
4. Build nodal $\dot\varepsilon_{\text{eff}}$, $\sigma_m = -P$ (⚠️ or $+P$ — settle the
   sign convention, §1.4), $\sigma_{\text{eq}}$, $\eta$, $\sigma_1/\sigma_{\text{eq}}$.
5. Integrate Rice–Tracey and Cockcroft–Latham along pathlines.
6. Deposit to grid, inspect.

### 1.2 ⚠️ 1.4 Settling the pressure sign convention

Two conventions collide:

- **Solid mechanics:** $\sigma_m = \tfrac13\text{tr}(\sigma)$, tension positive.
- **Fluid/FE codes:** often store $p$ with $\sigma = -p\,I + s$, so $\sigma_m = -p$.

**Do not guess.** Decide empirically using §0.3: the material **ahead of** the tool is
being *forged* (compressed) and **behind** it is being released (tensile). So:

$$\sigma_m < 0 \text{ ahead of the tool}, \qquad \sigma_m > 0 \text{ behind it}$$

Pick whichever sign of $P$ makes that true, and **record the decision in the code with
the justification**. This is a five-minute check that silently invalidates everything if
skipped.

### 1.3 Acceptance criteria for Stage 1

| Check | Expected |
|---|---|
| $\text{tr}(D) \approx 0$ | incompressibility — if violated, $\nabla u$ is wrong |
| $\dot\varepsilon_{\text{eff}}$ peaks at the tool interface | $O(1\text{–}100)$ s⁻¹ |
| $\sigma_{\text{eq}}$ in the shear-layer | $O(10\text{–}100)$ MPa — cross-check vs $3\mu\dot\varepsilon$ |
| Damage concentrates **advancing + trailing** | the literature signature |
| Rice–Tracey and Cockcroft–Latham agree spatially | independent laws, same field |

**If the AS/trailing signature does not appear, stop.** Debug before proceeding — a later
stage cannot fix a wrong Stage 1.

### 1.4 Then the 20-case cohort

`/home/arhashemi/fsw-gpu/scratch/shared/ROM/FOM/cylindrical_0NN.gid/post/cylindrical_119.pvtu`

Produces the **reference damage field per case** — the target for ROM in Stage 3–4, and
the baseline for the Q2 ranking test.

---

## ✅ Field diagnostics — COMPLETE 2026-09-25

**84 case-results across two machines.** Full numbers in `results/RESULTS_LOG.md`.

⚠️ These are **field diagnostics** — triaxiality evaluated pointwise in a
sampling window. They are not yet the pathline-integrated damage method, which
needs implementation-plan phase 1 (carrying state along a pathline) first.

| where | cases | mode | result |
|---|---|---|---|
| workstation | 21 × 3 rheologies | snapshot | 63 rows |
| LUMI ROM cohort | 22 | snapshot | 22 |
| LUMI PinShapes | 20 | 8-phase probe | 20 |
| LUMI PinShapes | 20 | full revolution (166 steps) | 20 |

### What these steps established

1. **The method works and is not noise.** Advancing-wake tension is recovered
   from FOM fields with no fitting, and it varies smoothly and monotonically
   with ω (Spearman ρ = −0.98 over the cohort).
2. **8 phases suffice for periodic tools.** The probe matches the 166-step
   revolution to within 0.021 (median 0.004), 19/20 flank agreement. Use
   `PHASE=probe` by default; `PHASE=full` only where the probe shows large spread.
3. **Tool geometry separates cleanly.** Threads → most advancing-tensile
   (median 1.202); tilt and flats → retreating (0.82, 0.75).
4. **The binary flank label is fragile; the continuous ratio is not.** Prefer
   the ratio as the modelled quantity (see O13).

### ⚠️ What they did NOT establish

- **No ground truth.** Nothing here has been compared against CT or macrographs
  (O4). Every number is an internally consistent prediction, not a validated one.
- **The ω crossover may be an artefact** of a fixed r/R window interacting with
  an ω-dependent shear-layer thickness (O11). Untested.
- **Rheology sets the magnitude**, and the Norton tables are borrowed for the
  cohort cases, which ship no `.mat` (O7).
- **No damage has been integrated along a pathline yet.** Everything so far is a
  snapshot statistic of the stress state; the history integral that distinguishes
  this method from a field plot needs implementation-plan phases 1, 3 and 4.

---

## How this work is organised

There are two loops, and they are not the same thing.

**The inner loop is `IMPLEMENTATION_PLAN_void_damage.md`, Phase 0 → 5.** That
document is the method: what to build, in what order, with what acceptance gates.
It is executed **end to end on one data configuration**, and it owns the phase
numbering. This document does not restate it.

**The outer loop is the four data configurations.** Once the implementation plan
has been completed, validated and written up on a configuration, the *same
finished method* is re-run on the next one:

```
   ┌─ run IMPLEMENTATION_PLAN Phase 0→5, end to end ─┐
   │                                                  │
   ▼                                                  │
  A. FOM mesh   → validate → report ──────────────────┤
   │                                                  │
   ▼  (only if A worked)                              │
  B. ROM mesh   → validate → report ──────────────────┤
   │                                                  │
   ▼                                                  │
  C. FOM grid   → validate → report ──────────────────┤
   │                                                  │
   ▼                                                  │
  D. ROM grid   → validate → report ──────────────────┘
```

Each pass produces its own report. A pass is a **go/no-go**: if the method fails
on configuration A there is nothing to carry forward, and B–D are not attempted.

### Why this order

| | configuration | what the pass tells you |
|---|---|---|
| **A** | FOM mesh | does the method work at all, on the cleanest data available |
| **B** | ROM mesh | how much ROM reconstruction error degrades it |
| **C** | FOM grid | how much grid projection degrades it |
| **D** | ROM grid | the production path, both error sources compounded |

B and C each change **one** thing relative to A, so a failure is attributable.
D changes both, and is only interpretable once B and C are known.

### Current position — configuration A, partway through the inner loop

Against `IMPLEMENTATION_PLAN_void_damage.md`:

| phase | what it is | status |
|---|---|---|
| **0** | prerequisites, sign/orientation checks | ✅ complete |
| **1** | carry a scalar on pathlines, <3 % throughput gate | 🟡 **kernel done + 5 gate tests; throughput gate unmeasured** |
| **2** | derived nodal fields (∇u → ε̇, σ_m, σ_eq, η) | ✅ `damage/fields.py` — built ahead of order |
| **3** | damage models (Rice–Tracey, Cockcroft–Latham) | 🟡 rheology done (`damage/rheology.py`); the ODEs are not integrated yet |
| **4** | deposition, validation, the honest test | ⬜ |
| **5** | if it works — closure branch, transient fields | ⬜ |

⚠️ **Phases 2 and 3 were partly built out of order**, as pointwise field
diagnostics — that is where the 62 case-results come from. Useful, and they
settled the sign conventions, orientation detection and rheology choice. But they
are **snapshot statistics**, not the pathline-integrated damage the plan
describes. Phase 1 — carrying state along a pathline — has not been done, and
without it phases 3–4 cannot be completed.

**So configuration A is not finished, and nothing moves to B until it is.**

---

## Outer loop — acceptance criteria per configuration

Applied to the **finished** method, once the implementation plan has been
completed on configuration A. Each compares against A.

These apply to the *finished* method. Each compares against configuration A.

### A. FOM mesh — the reference

No comparison; this pass defines the reference susceptibility field per case,
and its report is the one that decides whether B–D happen at all.

### B. ROM mesh — isolates ROM reconstruction error

⚠️ **The pressure-accuracy question lives here.** O15 showed the damage metric
is far more sensitive to pressure than to velocity: two runs of the same case
agreeing on ‖u‖ to ~5 % gave opposite void predictions because their pressure
fields differ (P median +1.3 vs +27.1 MPa, correlation 0.25 after centring).

Before building a pressure POD, use the O15 sensitivity measurement to set the
required accuracy. A pressure ROM that cannot hit it makes configuration B and D
pointless.

| metric | target |
|---|---|
| pressure reconstruction L2 | set from the O15 sensitivity measurement |
| η field L2 vs A | < 5 % |
| damage field L2 vs A | < 15 % |

### C. FOM grid — isolates projection error

**Compute gradients on the mesh, project the derived scalars.** Not: project raw
fields then differentiate on the grid. For linear tets ∇u is constant per element
and exact; differentiating an interpolated field stacks a second approximation on
the first, precisely where the field is steepest. Doing it mesh-side also makes
this configuration a clean isolation — it becomes purely "project A's scalars".

Order: cubic for most scalars, but watch η. It is a ratio, locally steep where
σ_eq is small, and cubic can overshoot — which Rice–Tracey then exponentiates.
Mask at source, clamp after; if overshoot persists use linear for η specifically.

| metric | target |
|---|---|
| relative L2 of η | < 5 % |
| relative L2 of final ln Φ | < 10 % |
| location of peak damage | same region |
| max\|η\| grid vs mesh | ≤ mesh × 1.1 (overshoot check) |

### D. ROM grid — the production path

Errors from B and C compound, and not independently — both distort the same
steep near-tool region.

| metric | target |
|---|---|
| damage field L2 vs A | < 25 % |
| **case ranking preserved** | **Spearman ρ > 0.9 — the one that matters** |

For classifying a safe zone in (v_adv, ω), an absolute error of 25 % is
irrelevant if the *ordering* of cases by susceptibility survives.
