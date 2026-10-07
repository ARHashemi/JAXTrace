# FSW Defect Physics — a primer from first principles

**Private study notes. Not for the meeting.**
Written for someone with physics + computation background, learning the mechanical-engineering framing.

Everything here starts from physics you already know (continuum mechanics, dimensional
analysis, ODE integration) and builds up to the engineering vocabulary, rather than
assuming the engineering terms.

> ## ⚠️ Stuck on a term? Do not read this document linearly.
>
> This primer is 1,789 lines organised for **reading through**. If a single
> unfamiliar word derails you, that structure works against you: "what is stress"
> lives at line 610, inside a 166-line section, which assumes two earlier sections.
> Three dependencies deep before you reach the thing you got stuck on.
>
> **→ Use `CARDS_one_concept_each.md` instead.** One concept per card, each
> self-contained in ~10 lines, no reading order. Jump to the term, read the card,
> go back to what you were doing. 16 cards cover everything the method uses:
> stress, strain rate, σ_m, deviator, σ_eq, η, advancing/retreating, pathline,
> why-not-a-grid, μ_eff, Norton, level-set, Reynolds, filling failure, the two
> damage models, FOM/ROM.
>
> Come back **here** when you want the derivation rather than the answer.

**How to read it.** §§1–6 are the physical picture: what FSW is, why the turbulence
and cavitation analogies fail quantitatively, and what actually forms a void.
§§6b–6d are the **machinery** — index notation, the material derivative, the yield
surface, and the Norton law your FOM actually used. §7 builds stress and strain from
tensors. §8 is the damage models, including He et al. (2008) read line by line from
the paper. §§9–10 map it onto the implementation and state what could still be wrong.
§10b is a complete symbol table with units; §11 a glossary.

If you want the shortest path to understanding the *code*: §6b.3 (why pathlines),
§6d (the rheology actually in use), §7.5 (the computation recipe), §9.

**Presenting this to the team?** Read **§9** (what was built and why, in order),
**§9b** (vocabulary of the implementation — the terms most likely to trip you up),
**§9c** (the questions they will ask, with defensible answers) and **§9d** (a
one-slide summary). §§1–6 are the physics you would need if someone challenges the
framing.

---

## 1. What FSW actually is, mechanically

A rotating tool — a **shoulder** (flat disc, pressed onto the plate surface) with a
**pin** (protruding stub, typically 3–6 mm) — is plunged into the joint line between
two plates and translated along it.

```
            ↓ axial force (~10-30 kN)
        ═════██═════   ← shoulder, rotating at ω, pressed on top surface
             ██
    ─────────██─────────  ← plate top surface
    ░░░░░░░░░██░░░░░░░░░
    ░░░░░░░░░██░░░░░░░░░  ← pin, stirring through the thickness
    ░░░░░░░░░░░░░░░░░░░░
    ────────────────────  ← backing plate (anvil)
              → traverse velocity v_adv
```

**Critical point: the metal never melts.** Friction and plastic work heat it to
roughly 0.7–0.9 of the melting temperature (~450–550 °C for aluminium). At that
temperature aluminium becomes *plastic* — it flows like an extremely stiff putty
under the stresses the tool imposes, then re-consolidates behind the tool.

This is why FSW is called a **solid-state** joining process. There is no weld pool,
no solidification, and no vapour phase anywhere in the process.

> **Why this matters for us:** every mechanism from fusion welding (porosity from
> dissolved gas coming out of solution on solidification, solidification cracking)
> is simply absent. So is every mechanism from liquid multiphase flow. The defect
> physics has to come from somewhere else.

---

## 2. Advancing vs retreating side — the one piece of jargon that matters

This is the single most important geometric fact in FSW, and it's pure kinematics.

The tool rotates with surface velocity $\mathbf{u}_{\text{tool}} = \boldsymbol{\omega} \times \mathbf{r}$
and simultaneously translates at $\mathbf{v}_{\text{adv}}$. On the two sides of the weld line,
these either **add** or **oppose**:

```
                    ADVANCING SIDE
            tool surface velocity  →→→
            traverse velocity      →→→
            net: FAST relative to workpiece
    ════════════════════════════════════════
                    ( tool )      → v_adv
    ════════════════════════════════════════
            tool surface velocity  ←←←
            traverse velocity      →→→
            net: SLOW (they cancel)
                    RETREATING SIDE
```

**Advancing side (AS):** $|\mathbf{u}_{\text{tool}} + \mathbf{v}_{\text{adv}}|$ is large.
**Retreating side (RS):** the two partially cancel.

Consequences that follow directly:

| | Advancing side | Retreating side |
|---|---|---|
| Relative velocity | high | low |
| Velocity gradient at tool interface | steep | shallow |
| Shear layer thickness | **thin** | thick |
| Flow turning required | **sharp** | gradual |
| Where defects appear | **here** | rarely |

There's nothing mysterious about the AS bias in defects — it's a direct consequence
of the velocity composition. It is *not* a pressure effect and *not* a vortex effect.

> For our ROM: $\omega_{\text{pin}}$ and $v_{\text{adv}}$ are exactly the two parameters
> that set this asymmetry. The ratio $\omega R / v_{\text{adv}}$ controls how strong it is.
> That's why the parameter space is the right space to classify in.

---

## 3. Why "viscosity" of 10⁶ Pa·s is not a typo

This trips up everyone coming from fluids. Water is $10^{-3}$ Pa·s. Honey is ~10.
Plasticised aluminium in FSW is measured at $10^5$–$5\times10^6$ Pa·s — **nine to ten
orders of magnitude more viscous than honey.**

### Where the number comes from

The metal isn't really a fluid. It's a solid deforming plastically. Engineers model
it *as if* it were a fluid by defining an **effective viscosity**:

$$\mu_{\text{eff}} \equiv \frac{\sigma_{\text{flow}}}{3\dot{\varepsilon}}$$

where $\sigma_{\text{flow}}$ is the material's **flow stress** (the stress at which it
yields and starts flowing plastically) and $\dot{\varepsilon}$ is the strain rate.

For hot aluminium: $\sigma_{\text{flow}} \sim 20\text{–}50$ MPa, and typical FSW strain
rates are $\dot\varepsilon \sim 1\text{–}100$ s⁻¹. So:

$$\mu_{\text{eff}} \sim \frac{30 \times 10^6}{3 \times 10} \sim 10^6 \ \text{Pa·s} \quad ✓$$

The number is a *restatement of the yield stress*, not a fluid property in the
Newtonian sense.

### The constitutive law

The standard model is **Sellars–Tegart / Zener–Hollomon**, which says flow stress
depends on strain rate and temperature through one combined variable:

$$Z = \dot{\varepsilon} \exp\!\left(\frac{Q}{RT}\right) \qquad \text{(Zener–Hollomon parameter)}$$

$$\sigma_{\text{flow}} = \frac{1}{\alpha}\sinh^{-1}\!\left[\left(\frac{Z}{A}\right)^{1/n}\right]$$

with $Q$ an activation energy (~150 kJ/mol for Al), and $\alpha, A, n$ fitted constants.

Physically: hotter → softer, faster deformation → harder (strain-rate hardening).

> ⚠️ **Sellars–Tegart is the textbook standard, but it is *not* what your FOM
> solved with.** The GiD `.mat` files specify a **Norton–Hoff power law** with
> temperature-tabulated coefficients, and that is what the damage pipeline uses by
> default. See **§6d** for the actual law, the real numbers from `C1.mat`, and why
> $\mu_{\text{eff}}$ is not a constant at all but falls roughly inversely with
> strain rate. Sellars–Tegart is kept as an independent cross-check (M5).
It's an Arrhenius-activated process, like any thermally activated rate process
you'd meet in solid-state physics.

**The key nonlinearity:** $\mu_{\text{eff}}$ depends on $\dot\varepsilon$, which depends on
$\nabla \mathbf{u}$, which depends on $\mu_{\text{eff}}$. Strongly coupled and strongly
nonlinear — this is *exactly* the nonlinearity that makes Barrier 1 (needing lots of
training data) real.

---

## 4. Reynolds number — and why it kills the turbulence analogy

$$\mathrm{Re} = \frac{\rho U L}{\mu} = \frac{\text{inertial forces}}{\text{viscous forces}}$$

You know this. The point is just how extreme the number is here.

| System | $\mu$ (Pa·s) | Re |
|---|---|---|
| Air over a wing | $2\times10^{-5}$ | $10^6$–$10^8$ |
| Water in a pipe | $10^{-3}$ | $10^4$–$10^6$ |
| Honey pouring | $10$ | $\sim 1$ |
| Glacier ice | $10^{13}$ | $10^{-15}$ |
| **FSW metal flow** | $10^5$–$10^6$ | $\mathbf{10^{-7}}$–$\mathbf{10^{-4}}$ |

At Re ≪ 1 the Navier–Stokes equations lose their inertial term entirely:

$$\underbrace{\rho\left(\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u}\cdot\nabla\mathbf{u}\right)}_{\text{negligible: } O(\mathrm{Re})} = -\nabla p + \nabla\cdot(\mu \nabla \mathbf{u})$$

$$\Rightarrow \quad \nabla p = \nabla\cdot(\mu\nabla\mathbf{u}), \qquad \nabla\cdot\mathbf{u}=0$$

This is **Stokes flow**. Properties that follow, all relevant to us:

1. **No turbulence.** Turbulence is an inertial instability — the $\mathbf{u}\cdot\nabla\mathbf{u}$
   term cascading energy to small scales. Remove inertia, remove turbulence. There is no
   Reynolds number at which this flow transitions; it is 7–12 orders below any transition.

2. **Instantaneous and quasi-reversible.** No memory, no inertial transients. The velocity
   field is slaved to the instantaneous boundary conditions. (Think of the classic
   Taylor–Couette reversibility demo with dye in glycerine.)

3. **Linear in the BCs, if $\mu$ were constant.** It isn't — $\mu(\dot\varepsilon, T)$ makes
   it nonlinear again. But the *inertial* nonlinearity is gone; what remains is
   constitutive.

4. **No vortex cores with low pressure.** A vortex core's pressure drop comes from
   centrifugal balance: $\partial p/\partial r = \rho u_\theta^2 / r$. That's an *inertial*
   effect, scaling with $\rho$. At Re → 0 it vanishes.

> **This is why "simulate turbulence and look for bubbles" cannot transfer.** There are
> no vortices in the inertial sense, no low-pressure cores, and no mechanism to make one.

---

## 5. Why cavitation is impossible here — the number

Cavitation happens when local pressure drops below the **vapour pressure** $p_v$ —
the pressure at which the liquid boils at that temperature.

```
  Water at 20°C:        p_v ≈ 2,300 Pa      ambient ≈ 100,000 Pa
                        → only need to drop pressure by ~98 kPa. Easy.
                        A ship propeller does it routinely.

  Aluminium at 750 K:   p_v ≈ 10⁻⁹ Pa       stir zone ≈ +10⁸ Pa (compressive)
                        → need to drop pressure by ~10⁸ Pa AND go to vacuum.
```

Aluminium at welding temperature is a *solid*. Its vapour pressure is essentially
zero — you'd need to reach a hard vacuum to boil it. Meanwhile the tool is pressing
down with tens of kN over a shoulder of a few cm², giving **+50 to +200 MPa compressive**.

$$\sigma_{\text{cav}} = \frac{p - p_v}{\tfrac{1}{2}\rho U^2} \approx \frac{10^8 - 10^{-9}}{\tfrac{1}{2}(2700)(1)^2} \approx \frac{10^8}{1350} \approx 7\times10^{4}$$

Cavitation inception needs $\sigma \lesssim 1$. We're at $\sim10^5$ on this normalisation,
and the *absolute* pressure gap is ~11 orders of magnitude from $p_v$.

**And the sign is wrong.** Compressive hydrostatic stress doesn't open voids — it
closes them. That's a real, useful effect (§8) and it's part of why FSW produces
dense, consolidated welds in the first place.

---

## 6. So what *does* make a void? The filling-failure picture

Here's the mechanism, built up physically.

### Step 1: a cavity must exist transiently

The pin is a solid cylinder moving through the plate at $v_{\text{adv}}$. It physically
displaces material. Behind it, there is — momentarily — a region that the pin has
vacated and that material has not yet filled.

Think of dragging your finger through thick mud: a trench opens behind the finger
and slowly collapses inward.

### Step 2: material must be delivered to fill it

The rotating pin drags a **shear layer** of plasticised metal around itself, from
the front, around (predominantly) the retreating side, to the back. This is the
delivery mechanism.

```
   top view:
                    ┌─── material path ───┐
                    ↓                     ↑
        front  →   (PIN)  →  wake/cavity region
                    ↑                     ↓
                    └── around RS ────────┘
```

### Step 3: the volume bookkeeping

Per tool revolution, the tool advances a distance

$$\Delta x = \frac{v_{\text{adv}}}{\omega/2\pi} = \text{"weld pitch"} \quad \text{[mm/rev]}$$

The cavity volume that opens per revolution scales as $V_{\text{cav}} \sim A_{\text{pin}} \cdot \Delta x$.

The volume the shear layer can deliver per revolution scales as
$V_{\text{del}} \sim \delta \cdot h \cdot (\omega R) \cdot \frac{2\pi}{\omega} = 2\pi \delta h R$,
where $\delta$ is shear layer thickness and $h$ the pin height.

$$\boxed{\ V_{\text{del}} < V_{\text{cav}} \ \Longrightarrow \ \text{void}\ }$$

**This is the whole mechanism.** It's a mass-continuity accounting problem, not a
nucleation problem. The "void" is simply metal that never arrived.

### Step 4: why it becomes a continuous tunnel

Each revolution leaves a small unfilled region at roughly the same position relative
to the tool. As the tool advances, these superimpose along the weld line:

```
   rev 1:  ○
   rev 2:   ○
   rev 3:    ○         →   ○○○○○○○○○  = continuous tunnel/wormhole
   rev 4:     ○
```

Ghate et al. (2020) make exactly this argument and validate it against CT scans.

### Why it lands on the advancing side

From §2: the AS shear layer is thinnest and the flow must turn most sharply there.
So $\delta$ is smallest on the AS → $V_{\text{del}}$ smallest → that's where the deficit shows up.

### The parameter trends this predicts

| Change | Effect on $V_{\text{del}}/V_{\text{cav}}$ | Defect risk |
|---|---|---|
| ↑ $v_{\text{adv}}$ | cavity opens faster | ↑ **worse** |
| ↑ $\omega$ | more delivery per unit length, hotter, softer | ↓ better (up to a limit) |
| ↑ temperature | lower $\sigma_{\text{flow}}$ → thicker shear layer | ↓ better |
| ↓ axial force | less compaction | ↑ worse |

These are exactly the trends reported experimentally — which is good evidence the
picture is right.

> **Direct relevance to your work:** $V_{\text{del}}/V_{\text{cav}}$ is a function of
> $(\omega, v_{\text{adv}})$. A "safe zone" in that plane is exactly the region where
> this ratio exceeds 1 with margin. That's Stage 1 of your objective, stated physically.

---

## 6b. Notation and continuum-mechanics machinery

§7 onwards uses index notation, tensors and the material derivative without
ceremony. This section establishes them, because three of them are the reason the
*whole method* is shaped the way it is.

### 6b.1 Index notation and the summation convention

A vector is $u_i$ ($i = 1,2,3$ meaning $x,y,z$). A rank-2 tensor is $A_{ij}$, a 3×3
array of numbers. The rule that makes the algebra compact:

> **Einstein summation.** A subscript repeated *twice in one term* is summed over
> $1,2,3$. A subscript appearing *once* is a free index and labels a component.

| expression | means | result is |
|---|---|---|
| $a_i b_i$ | $a_1b_1 + a_2b_2 + a_3b_3$ | a scalar (dot product) |
| $A_{ij}b_j$ | $\sum_j A_{ij}b_j$ for each $i$ | a vector (matrix–vector) |
| $A_{ii}$ | $A_{11}+A_{22}+A_{33}$ | a scalar (the **trace**) |
| $A_{ij}A_{ij}$ | sum of squares of all 9 entries | a scalar |
| $\delta_{ij}$ | 1 if $i=j$, else 0 | the identity ("Kronecker delta") |

So $\dot\varepsilon_{\text{eff}} = \sqrt{\tfrac23 D_{ij}D_{ij}}$ means: square all nine
entries of $D$, add them, times 2/3, square root. In NumPy that is exactly
`np.sqrt(2/3 * np.sum(D*D))`.

⚠️ **$A_{ij}A_{ij}$ is not $(A_{ii})^2$.** The first is the sum of squared entries
(a magnitude); the second is the squared trace. For a deviatoric tensor the trace is
zero while the magnitude is not — the distinction is the entire content of §7.3.

### 6b.2 Why a tensor and not just a matrix

Physically the difference is **how it transforms**. Rotate your coordinate axes by
$Q$; a rank-2 tensor's components change as $A' = QAQ^{\mathsf T}$. Quantities built
to be *invariant* under that rotation — the trace, the magnitude, the eigenvalues —
are the ones with physical meaning, because the material does not know how you
chose your axes.

This is why every scalar in this primer ($\dot\varepsilon_{\text{eff}}$, $\sigma_m$,
$\sigma_{\text{eq}}$, $\eta$, $\mu_L$) is an **invariant**. A criterion built from a
single component like $\sigma_{11}$ would give different answers in rotated frames
and be physically meaningless. In FSW, where material is continuously rotating, this
is not a pedantic point.

### 6b.3 The material derivative — why pathlines

You have $\mathbf{u}(\mathbf{x},t)$ stored at fixed mesh points (the **Eulerian**
description). But damage accumulates in a *material particle* (the **Lagrangian**
description). The bridge is the material derivative:

$$\boxed{\ \frac{D\phi}{Dt} = \underbrace{\frac{\partial \phi}{\partial t}}_{\text{local}} + \underbrace{u_j \frac{\partial \phi}{\partial x_j}}_{\text{advective}}\ }$$

- $\partial\phi/\partial t$ — the rate of change *at a fixed point in space*.
- $u_j\,\partial\phi/\partial x_j$ — change because the particle has **moved** to
  somewhere with a different $\phi$.

> **This is the mathematical reason the whole implementation is a particle tracker.**
>
> Damage obeys $D\Phi/Dt = (\text{source})$. In an Eulerian frame you would have to
> discretise the advective term $u_j\partial\Phi/\partial x_j$ — numerical diffusion,
> stability limits, and smearing of exactly the sharp gradients you care about.
>
> Follow the particle instead and the advective term **disappears by construction**:
> along a pathline $d\mathbf{x}/dt = \mathbf{u}$, so $D\Phi/Dt$ is literally
> $d\Phi/dt$, an ordinary ODE in $t$. That is why `dmg_new = dmg + dt * edot` is
> correct and complete — no gradient of $\Phi$ appears anywhere.

A **steady** field ($\partial/\partial t = 0$) still gives non-zero $D\phi/Dt$,
because the particle moves. This is why steady FOM fields can produce meaningful
accumulated damage — a point that caused confusion earlier in this project.

### 6b.4 Pathlines, streamlines, streaklines

For a **steady** field all three coincide. For time-dependent fields (threaded or
tilted pins, where the geometry is periodic at $\omega$) they differ, and only one is
correct here:

| curve | definition | use |
|---|---|---|
| **streamline** | tangent to $\mathbf{u}$ at one *instant* | snapshot visualisation |
| **pathline** | trajectory of one *particle* over time | **what damage integrates along** |
| **streakline** | locus of particles released from one point | dye-injection experiments |

⚠️ Integrating damage along a *streamline* of a time-dependent field would be wrong:
no particle follows it. The RK4 tracker produces pathlines, which is correct.

---

## 6c. Plasticity fundamentals — what "yield" means

§7 says the material flows when $\sigma_{\text{eq}}$ reaches the flow stress. Here is
where that comes from, since it is the least familiar piece coming from physics.

### 6c.1 Elastic vs plastic, and why FSW is essentially all plastic

| | elastic | plastic |
|---|---|---|
| reversible? | yes | **no** |
| strain magnitude in metals | $\lesssim 0.2\%$ | up to $\bar\varepsilon \sim 10$–$100$ in FSW |
| volume change | yes | **none** (isochoric) |
| depends on rate? | no | **yes** (and on $T$) |

FSW stir-zone strains are $\bar\varepsilon \sim 10$–$100$, i.e. **3–4 orders of
magnitude** larger than the elastic range. So the elastic part is negligible and the
material is modelled as a **rigid-viscoplastic** flow — which is exactly what licenses
treating it as a very viscous fluid (§3).

⚠️ This is also why **plastic incompressibility** ($D_{kk}=0$) is a law here and not
an approximation: plastic flow moves dislocations, which rearrange atoms without
changing their number density.

### 6c.2 The yield surface, and why only the deviator matters

Experimentally (Bridgman, 1940s): **hydrostatic pressure alone does not cause metals
to yield**, even at GPa levels. Squeeze a metal cube equally on all six faces and it
shrinks elastically but never flows.

So the yield condition cannot depend on $\sigma_m$ — only on the deviator $s_{ij}$.
The simplest rotation-invariant scalar built from $s_{ij}$ is its magnitude, giving
the **von Mises** criterion:

$$f(\sigma) = \sigma_{\text{eq}} - \sigma_{\text{flow}} = 0, \qquad \sigma_{\text{eq}} = \sqrt{\tfrac32 s_{ij}s_{ij}}$$

Geometrically: in the space of principal stresses $(\sigma_1,\sigma_2,\sigma_3)$, the
yield surface is a **cylinder** whose axis is the hydrostatic line
$\sigma_1=\sigma_2=\sigma_3$. Moving *along* the axis (changing $\sigma_m$) never
crosses the surface; moving *away* from it (changing the deviator) does.

$\sigma_{\text{eq}}$ is also written $\sqrt{3J_2}$, where
$J_2 = \tfrac12 s_{ij}s_{ij}$ is the **second invariant** of the deviator — this is why
von Mises plasticity is called "$J_2$ flow theory" in the literature.

> **The central tension of this whole project, in one paragraph.** Yielding is
> independent of $\sigma_m$ — but **void growth is not**. A void is a free surface, and
> pressure acting on it does work changing its volume. So $\sigma_m$ drops out of the
> flow rule but returns, exponentially, in the damage law (§8). That is why $\eta =
> \sigma_m/\sigma_{\text{eq}}$ is *the* variable: it measures how much volumetric
> driving there is *relative to* the shape-changing stress that is doing the deforming.

### 6c.3 Why damage models look the way they do

Three facts constrain any void-growth law, and together they force the structure:

1. **No deformation → no growth.** Voids need the matrix to flow, so the rate must
   carry a factor $\dot\varepsilon_{\text{eff}}$.
2. **Tension opens, compression closes.** The $\sigma_m$ dependence must change sign
   with $\sigma_m$, and experiment says the sensitivity is *exponential*, not linear.
3. **Growth is proportional to what exists.** A void of radius $R$ in a matrix
   straining at $\dot\varepsilon$ grows as $\dot R \propto R\dot\varepsilon$, so
   $\dot\Phi \propto \Phi$.

Any law obeying all three looks like
$\dot\Phi \sim \Phi\,\dot\varepsilon_{\text{eff}}\exp(c\,\eta)$ — which is Rice–Tracey,
and the skeleton of He's Eq. (1). The models differ in refinements, not in shape.

⚠️ **Consequence of (3): these are growth laws, not nucleation laws.** $\Phi=0$ is a
fixed point — no voids ever appear from nothing. Every such model needs a seeded
initial porosity $\Phi_0$, and the predicted result depends on a number nobody
measures directly. This is a real limitation, not a technicality (§10).

---

## 6d. The Norton law — what your FOM actually uses

⚠️ §3 introduces $\mu_{\text{eff}}$ and §7.4 gives Sellars–Tegart, but **neither is
what the FOM solved with.** The GiD `.mat` files specify a **Norton–Hoff power law**,
and `jaxtrace/damage/rheology.py` reads those exact tables. This is the default in
the damage pipeline, so it is worth understanding directly.

$$\boxed{\ \sigma_{\text{eq}} = \sqrt3\;\text{VISCO}(T)\,\cdot\,
\big(\sqrt3\,\dot\varepsilon_{\text{eff}}\big)^{\;\text{EXPVI}(T)}\ }$$

> ⚠️ **Note the two √3 factors — they are not decoration.** The naive form
> $\sigma_{\text{eq}} = \text{VISCO}\cdot\dot\varepsilon^{\,\text{EXPVI}}$ is **wrong by
> a factor of 1.816**, and §6.4a below explains exactly why. Read that before using
> any absolute stress number.

Both coefficients are **tabulated against temperature** and interpolated linearly
(held constant outside the range). From `C1.mat` (Al 6063, 54 entries, 25–675 °C):

| $T$ (°C) | VISCO (Pa·sᵐ) | EXPVI = $m$ | $\sigma_{\text{eq}}$ at $\dot\varepsilon=50\,$s⁻¹ | implied $\mu_{\text{eff}}$ |
|---|---|---|---|---|
| 25 | 1.131e8 | 0.02325 | 123.9 MPa | 0.83 MPa·s |
| 200 | 6.761e7 | 0.04135 | 79.5 MPa | 0.53 MPa·s |
| 400 | 3.065e7 | 0.07985 | 41.9 MPa | 0.28 MPa·s |
| 500 | 1.840e7 | 0.11090 | 28.4 MPa | 0.19 MPa·s |
| 600 | 1.007e7 | 0.15423 | 18.4 MPa | 0.12 MPa·s |
| 675 | 6.505e6 | 0.19741 | 14.1 MPa | 0.09 MPa·s |

### 6.4a Why there are two √3 factors — the shear-vs-von-Mises convention

**This is the single most important correction in the whole pipeline, and the
question you are most likely to be asked. Here is the whole thing in four steps.**

**Step 1 — what we did.** We computed $\sigma_{\text{eq}}$ from the `.mat` tables the
obvious way, $\text{VISCO}\cdot\dot\varepsilon^{\,\text{EXPVI}}$, and compared it
against the stress the **solver itself exported** ($\sqrt{3J_2}$, which it writes out
as a field). Over **581,488 cells** our value was consistently

$$\frac{\text{ours}}{\text{solver's}} = \mathbf{0.569}\qquad(\pm0.8\%)$$

A *constant* ratio is the signature of a **unit or convention mismatch**, not a bug:
a coding error would scatter.

**Step 2 — where the factor comes from.** "Equivalent stress" and "equivalent strain
rate" have **two conventions in common use**, and a power law written in one is wrong
in the other:

| | von Mises (what we use) | **shear** (what the tables use) |
|---|---|---|
| strain rate | $\dot\varepsilon_{vM}=\sqrt{\tfrac23\mathbf{D}':\mathbf{D}'}$ | $\gamma=\sqrt2\lVert\mathbf{e}\rVert$ |
| stress | $\sigma_{vM}=\sqrt{3J_2}$ | $\tau=\tfrac{\sqrt2}{2}\lVert\mathbf{s}\rVert$ |
| relation | — | $\gamma=\sqrt3\,\dot\varepsilon_{vM}$, $\ \sigma_{vM}=\sqrt3\,\tau$ |

The solver's own paper states the shear definitions explicitly — Venghaus, *Finite
Elements in Analysis & Design* **224** (2023) 103986, Publication 1, eq. (5) — and
`VISCO`/`EXPVI` are the coefficients of **that** law, i.e. $\tau(\gamma)$.

**Step 3 — convert, and both the input and the output pick up a √3.** Feeding the
table a von Mises rate means first converting the rate ($\times\sqrt3$ inside the
power), and the table returns a *shear* stress, which must be converted back
($\times\sqrt3$ outside):

$$\sigma_{\text{eq}}=\underbrace{\sqrt3}_{\tau\to\sigma_{vM}}\;\text{VISCO}(T)\;
\big(\underbrace{\sqrt3}_{\dot\varepsilon_{vM}\to\gamma}\dot\varepsilon\big)^{m}
\qquad\Rightarrow\qquad
\text{factor}=\sqrt3\,(\sqrt3)^{m}$$

At the typical $m\approx0.086$ this is $1.7321\times1.0484=\mathbf{1.8163}$.

**Step 4 — the check that makes it a derivation rather than a fit.**

| | ratio to the solver's own $\sqrt{3J_2}$ |
|---|---|
| tables used as-is | **0.5689** |
| with the conversion | **1.0346** |

Measured correction $1/0.5689 = \mathbf{1.8187}$ against predicted $\mathbf{1.8163}$
— **0.13 %**. Predicting the factor from the published definition and then matching it
to 0.13 % is what turns this from "a number that makes it fit" into a confirmed
derivation.

**So, can we use the same `.mat` file?** **Yes.** The tables are the right data for
the right material; they are simply written in the other convention, and the fix is
one constant factor applied once.

⚠️ **Two things this does NOT mean.**
1. It is **not** the "modified Norton" *friction* law (the tanh boundary condition) —
   different thing, confusingly similar name. This is the **material viscosity**.
2. All absolute $\eta$ and $\sigma_{\text{eq}}$ values reported before 2026-09-30 are
   low by 1.816×. **Ratios and rankings are unaffected**, which M5 confirmed
   independently ($\rho=+0.9995$).

---

### Reading the physics out of those two columns

**VISCO falls by 17× from 25 °C to 675 °C.** This is thermal softening, and it is the
dominant effect: hot material is far weaker, which is why the tool's frictional
heating is what makes the process possible at all.

**EXPVI ($m$, the rate-sensitivity exponent) rises from 0.023 to 0.197.** This matters
more than it looks:

- $m \to 0$ would be **perfectly plastic** — flow stress independent of how fast you
  deform.
- $m = 1$ would be **Newtonian viscous** — stress strictly proportional to rate.

So doubling the strain rate raises the flow stress by only

| | |
|---|---|
| at 25 °C | $2^{0.0232} = 1.016$ → **+1.6 %** |
| at 675 °C | $2^{0.1974} = 1.147$ → **+14.7 %** |

⚠️ **Cold aluminium is nearly rate-insensitive; hot aluminium is not.** Two
consequences for this project:

1. **$\sigma_{\text{eq}}$ is remarkably insensitive to errors in $\dot\varepsilon$.** A
   50 % error in strain rate moves the flow stress by ~1–7 %. This is *good news* for
   the ROM: velocity-gradient errors are strongly damped on the way to stress.
2. **But $\eta = \sigma_m/\sigma_{\text{eq}}$ inherits almost all its error from
   $\sigma_m$, i.e. from pressure.** The denominator is stiff; the numerator is not.
   This is the mechanism behind O15 — the metric is far more sensitive to pressure
   than to velocity — and it is a *prediction of the constitutive law*, not an
   accident of the data.

### Why $\mu_{\text{eff}}$ is not a material constant

Equating the Norton law with $\sigma_{\text{eq}} = 3\mu_{\text{eff}}\dot\varepsilon$:

$$\mu_{\text{eff}}(T,\dot\varepsilon) = \frac{\text{VISCO}(T)}{3}\,\dot\varepsilon^{\,m(T)-1}$$

Since $m \ll 1$, the exponent $m-1$ is close to $-1$ (it runs from $-0.977$ at 25 °C
to $-0.803$ at 675 °C), so $\mu_{\text{eff}}$ falls almost **inversely** with strain
rate — the material is strongly **shear-thinning**. Doubling $\dot\varepsilon$ roughly
halves the effective viscosity. The $10^5$–$10^6$ Pa·s
figures in §3 are therefore *local* values at particular $(T,\dot\varepsilon)$, not a
property of aluminium. Quoting a single number is a convenience, and the table above
is what the solver actually used.

### The three routes, and when they agree

| route | formula | needs | in code |
|---|---|---|---|
| **constant** | $\sigma_{\text{eq}} = 3\mu_0\dot\varepsilon$ | one number | `sigma_eq_constant` |
| **Norton** ✅ default | $\text{VISCO}(T)\dot\varepsilon^{\text{EXPVI}(T)}$ | the FOM's own `.mat` | `sigma_eq_norton` |
| **Sellars–Tegart** | $\tfrac{1}{\alpha}\sinh^{-1}[(Z/A)^{1/n}]$ | $T$ + literature constants | `sigma_eq_sellars_tegart` |

**Prefer Norton**: it is the law the FOM integrated, so using anything else
introduces an inconsistency between the velocity field and the stress computed from
it. Sellars–Tegart is a useful *cross-check* (M5) — if the damage ranking of cases is
the same under both, the conclusion is not an artefact of the rheology.

---

## 7. Stress and strain, from tensors up

§8 needs $\sigma_m$, $\sigma_{\text{eq}}$ and $\dot\varepsilon_{\text{eff}}$. This section defines
them properly and gives the computation recipe.

### 7.1 Strain rate: what the velocity field tells you

You have $\mathbf{u}(\mathbf{x})$ on a grid. Everything kinematic comes from its gradient:

$$L_{ij} = \frac{\partial u_i}{\partial x_j} \qquad \text{(velocity gradient tensor, 3×3, not symmetric)}$$

Split it into symmetric and antisymmetric parts:

$$L_{ij} = \underbrace{D_{ij}}_{\text{stretching}} + \underbrace{W_{ij}}_{\text{spin}}, \qquad
D_{ij} = \tfrac{1}{2}(L_{ij} + L_{ji}), \qquad W_{ij} = \tfrac{1}{2}(L_{ij} - L_{ji})$$

- $D_{ij}$ — **rate-of-deformation tensor** (also written $\dot\varepsilon_{ij}$). This is
  the part that actually deforms material: stretching, compressing, shearing.
- $W_{ij}$ — **spin tensor**. Pure rigid-body rotation. It rotates a material element
  without distorting it, so it does **no work and causes no damage**.

> **This distinction is the whole reason vorticity is useless here.** Vorticity is
> $\boldsymbol{\omega} = \nabla\times\mathbf{u}$, which is just the dual vector of $W_{ij}$ — the
> *rotation* part. In FSW the tool spins everything, so $W$ is large everywhere,
> including in perfectly sound welds. The damaging quantity is $D$, not $W$.

**Incompressibility.** Plastic deformation conserves volume, so $\text{tr}(D) = D_{kk} = \nabla\cdot\mathbf{u} = 0$.
Useful as a numerical check on your interpolated field — if $D_{kk}$ is far from zero,
your gradient computation or the projection has a problem.

### 7.2 The effective (equivalent) strain rate

$D_{ij}$ has 6 independent components. To get one scalar "how fast is this being
deformed", use the **von Mises equivalent strain rate**:

$$\boxed{\ \dot\varepsilon_{\text{eff}} = \sqrt{\tfrac{2}{3}\, D_{ij} D_{ij}}\ }$$

(summation implied; $D_{ij}D_{ij}$ is the sum of squares of all 9 entries).

The $\tfrac{2}{3}$ is a normalisation convention chosen so that in a **uniaxial** tension
test at rate $\dot\varepsilon$, you get $\dot\varepsilon_{\text{eff}} = \dot\varepsilon$ exactly.
Check it: uniaxial with incompressibility gives $D = \text{diag}(\dot\varepsilon, -\dot\varepsilon/2, -\dot\varepsilon/2)$, so
$D_{ij}D_{ij} = \dot\varepsilon^2(1 + \tfrac14 + \tfrac14) = \tfrac{3}{2}\dot\varepsilon^2$, and
$\sqrt{\tfrac23 \cdot \tfrac32 \dot\varepsilon^2} = \dot\varepsilon$ ✓

**Accumulated equivalent strain** — integrate along a pathline:

$$\bar\varepsilon(t) = \int_0^t \dot\varepsilon_{\text{eff}}\,dt'$$

This is a *history* variable, which is exactly why it wants to be carried by a tracked
particle rather than stored on a grid. Typical FSW stir-zone values are $\bar\varepsilon \sim 10$–$100$
(enormous — hundreds of percent strain).

### 7.3 Stress: the tensor and its two invariants

Stress $\sigma_{ij}$ is force per area on a surface with normal $j$, in direction $i$.
Symmetric, so 6 independent components. Split it the same way as strain:

$$\sigma_{ij} = \underbrace{\sigma_m \delta_{ij}}_{\text{hydrostatic / pressure}} + \underbrace{s_{ij}}_{\text{deviatoric}}$$

$$\sigma_m = \tfrac{1}{3}\sigma_{kk} = \tfrac{1}{3}(\sigma_{11}+\sigma_{22}+\sigma_{33}) = -p$$

- **$\sigma_m$ — mean (hydrostatic) stress.** Pure squeeze or pure pull. Changes *volume*,
  not shape. Sign convention: in solid mechanics tension is positive, so
  $\sigma_m = -p$ where $p$ is the usual fluid pressure. **Get the sign right — it
  determines whether voids grow or close.**
- **$s_{ij}$ — deviatoric stress.** Traceless ($s_{kk}=0$). Changes *shape*, not volume.
  This is what drives plastic flow.

The scalar measure of the deviatoric part is the **von Mises equivalent stress**:

$$\boxed{\ \sigma_{\text{eq}} = \sqrt{\tfrac{3}{2}\, s_{ij}s_{ij}}\ }$$

Same normalisation logic: in uniaxial tension, $\sigma_{\text{eq}} = |\sigma_{11}|$.

**Yield criterion.** The material flows plastically when $\sigma_{\text{eq}}$ reaches the
flow stress: $\sigma_{\text{eq}} = \sigma_{\text{flow}}(\bar\varepsilon, \dot\varepsilon, T)$.
Note that yielding depends *only* on the deviatoric part — you cannot make metal yield
by squeezing it uniformly from all sides, no matter how hard. This is the key fact
behind void closure.

### 7.4 Getting stress from your velocity field

Here's the part that matters practically. You have $\mathbf{u}$ and $p$; you need
$\sigma_{\text{eq}}$ and $\sigma_m$.

**The constitutive link.** For a viscoplastic material modelled as a generalised
Newtonian fluid:

$$s_{ij} = 2\mu_{\text{eff}}\, D_{ij}$$

Substituting into the definition of $\sigma_{\text{eq}}$:

$$\sigma_{\text{eq}} = \sqrt{\tfrac32 s_{ij}s_{ij}} = \sqrt{\tfrac32 \cdot 4\mu_{\text{eff}}^2 D_{ij}D_{ij}}
= 2\mu_{\text{eff}}\sqrt{\tfrac32 D_{ij}D_{ij}} = 3\mu_{\text{eff}}\,\dot\varepsilon_{\text{eff}}$$

$$\boxed{\ \sigma_{\text{eq}} = 3\,\mu_{\text{eff}}\,\dot\varepsilon_{\text{eff}}\ }$$

Which is just $\mu_{\text{eff}} \equiv \sigma_{\text{flow}}/3\dot\varepsilon$ from §3 rearranged —
the definition and this identity are the same statement. Consistent, and a good sign
the bookkeeping is right.

**Two routes to $\sigma_{\text{eq}}$, and they should agree:**

| Route | Formula | Needs |
|---|---|---|
| **A — kinematic** | $\sigma_{\text{eq}} = 3\mu_{\text{eff}}\dot\varepsilon_{\text{eff}}$ | $\mu_{\text{eff}}$ field from the FOM |
| **B — constitutive** | $\sigma_{\text{eq}} = \sigma_{\text{flow}}(Z)$ via Sellars–Tegart | $T$ field + material constants |

Route B needs only temperature and strain rate, both of which you have. If the FOM
exports $\mu_{\text{eff}}$, use route A and check against B — disagreement means the
constitutive parameters don't match what the FOM actually used.

**And $\sigma_m$ comes directly from pressure:**

$$\sigma_m = -p$$

⚠️ **Sign check, do this first.** In a region under the shoulder, verify $p > 0$
(compressive, several tens of MPa) so that $\sigma_m < 0$. If your FOM's pressure sign
convention is flipped, every triaxiality value flips and voids will appear to grow
exactly where they should close. This is the single easiest way to get the whole
analysis backwards.

### 7.5 Putting it together — the computation recipe

Per grid point (or per particle position, sampled from the grid):

```
1.  L = ∇u                                  # 3×3, from finite differences on the grid
2.  D = ½(L + Lᵀ)                           # rate of deformation
3.  check |tr(D)| ≪ ‖D‖                     # incompressibility sanity check
4.  ε̇_eff = sqrt(2/3 · Dᵢⱼ Dᵢⱼ)             # scalar strain rate
5.  σ_eq = 3 · μ_eff · ε̇_eff                # OR from Sellars-Tegart with T
6.  σ_m  = -p                               # CHECK SIGN
7.  η    = σ_m / max(σ_eq, ε)               # triaxiality; guard the denominator
```

**Numerical cautions:**

- **Step 1 is the delicate one.** $\nabla\mathbf{u}$ from a projected grid field amplifies
  interpolation error. Your mesh2grid cubic projection helps (C¹ continuity), but check
  gradients near the pin where the field is steepest.
- **Step 7 divides by $\sigma_{\text{eq}}$**, which → 0 far from the tool where nothing is
  deforming. There $\eta$ is meaningless and numerically explosive. Mask to the active
  zone — e.g. require $\dot\varepsilon_{\text{eff}} > 0.01\,\text{s}^{-1}$ — rather than
  letting it blow up.
- **$\eta$ is a ratio of two computed quantities**, so relative errors add. If pressure
  is the less well-reconstructed field in your ROM, $\eta$ inherits that noise.

### 7.6 The Lode parameter, computationally

From §8 you also want $\mu_L$. Practically, get the eigenvalues of $s_{ij}$
(symmetric 3×3, so real eigenvalues), sort $s_1 \ge s_2 \ge s_3$, then:

$$\mu_L = \frac{2s_2 - s_1 - s_3}{s_1 - s_3}$$

Equivalently, via the third invariant $J_3 = \det(s)$ and the **Lode angle**:

$$\cos(3\theta) = \frac{27 J_3}{2\sigma_{\text{eq}}^3}$$

The invariant form is usually preferable numerically — no eigenvalue solve, and it
vmaps cleanly. $J_2 = \tfrac12 s_{ij}s_{ij}$ and $J_3 = \det(s)$ are both cheap
closed-form expressions in the components.

---

## 8. The right borrowed theory: void closure in metal forming

If cavitation is the wrong analogue, what's the right one? **Bulk metal forming** —
rolling, forging, extrusion. Those processes also take solid metal to large plastic
strains at high temperature, and they have a well-developed theory of what happens
to voids.

### Stress triaxiality

Decompose the stress tensor into hydrostatic and deviatoric parts:

$$\sigma_{ij} = \underbrace{\sigma_m \delta_{ij}}_{\text{hydrostatic}} + \underbrace{s_{ij}}_{\text{deviatoric}}, \qquad \sigma_m = \tfrac{1}{3}\sigma_{kk}$$

Define the **von Mises equivalent stress** $\sigma_{\text{eq}} = \sqrt{\tfrac{3}{2}s_{ij}s_{ij}}$
and then:

$$\boxed{\ \eta = \frac{\sigma_m}{\sigma_{\text{eq}}} \quad \text{(stress triaxiality)}\ }$$

This single dimensionless number governs void behaviour:

| $\eta$ | State | Void behaviour |
|---|---|---|
| $\eta > 0$ | net tension | voids **grow** |
| $\eta \approx 0$ | pure shear | voids distort, don't grow much |
| $\eta < 0$ | net compression | voids **close** |

In FSW under the shoulder, $\eta$ is strongly negative — which is precisely why the
process consolidates material and can even heal pre-existing porosity.

**Voids appear where $\eta$ is locally least negative** (i.e. compression is lost) —
which happens in the wake, behind the pin, on the advancing side. Same place the
filling argument predicts. The two pictures agree.

### The Lode parameter

Triaxiality isn't quite enough. Two stress states can share $\eta$ but differ in the
*shape* of the deviatoric part (think: the three principal deviatoric stresses can be
arranged differently around the same mean). The **Lode parameter** captures this:

$$\mu_L = \frac{2\sigma_2 - \sigma_1 - \sigma_3}{\sigma_1 - \sigma_3} \in [-1, 1]$$

$\mu_L = -1$ is axisymmetric tension, $+1$ axisymmetric compression, $0$ pure shear.
It matters because shear-dominated states ($\eta \approx 0$) can still damage material
by *distorting* voids even without growing them — which classical models miss.

### Void evolution models — the ODEs in full

Three candidates, in increasing order of complexity. All are ODEs in a scalar carried
along a pathline, which is the form you want.

---

#### (a) Rice–Tracey — simplest, and the best starting point

The classic result for the growth rate of an isolated spherical void in a plastic matrix.
Rice & Tracey (1969) solved it and got an exponential dependence on triaxiality:

$$\boxed{\ \frac{\dot{R}}{R} = 0.283\,\dot{\varepsilon}_{\text{eff}}\, \exp\!\left(\frac{3\eta}{2}\right)\ }$$

where $R$ is void radius. In terms of volume fraction ($f \propto R^3$):

$$\frac{\dot f}{f} = 3\frac{\dot R}{R} = 0.849\,\dot\varepsilon_{\text{eff}} \exp\!\left(\tfrac{3}{2}\eta\right)$$

**Parameters needed: none.** The 0.283 is universal. This is why it's the right first
attempt — you can run it today with no calibration.

**Behaviour:** the $\exp(3\eta/2)$ factor is everything.

| $\eta$ | $\exp(3\eta/2)$ | Meaning |
|---|---|---|
| $+1$ | 4.48 | strong tension — rapid growth |
| $0$ | 1.00 | pure shear — slow growth |
| $-0.5$ | 0.47 | mild compression — suppressed |
| $-1$ | 0.22 | FSW-like — strongly suppressed |
| $-2$ | 0.05 | deep compression — essentially frozen |

⚠️ **Note the limitation:** at $\eta < 0$ this gives *slow growth*, never actual
**shrinkage**. Rice–Tracey was derived for tension and doesn't capture closure. For a
susceptibility *ranking* that's fine — regions with larger $\dot f$ are more at risk.
For absolute closure physics you need (c).

---

#### (b) Cockcroft–Latham — a damage indicator, not a physical void fraction

Widely used in forming because it's robust and cheap. Integrate the tensile part of
the maximum principal stress along the path:

$$\boxed{\ C = \int_0^{\bar\varepsilon} \frac{\langle\sigma_1\rangle}{\sigma_{\text{eq}}}\, d\bar\varepsilon\ }$$

where $\sigma_1$ is the largest principal stress and $\langle\cdot\rangle$ is the
Macaulay bracket: $\langle x\rangle = \max(x, 0)$. So compression contributes **nothing**
— the integrand switches off wherever the material is squeezed.

As an ODE along a pathline:

$$\frac{dC}{dt} = \frac{\max(\sigma_1, 0)}{\sigma_{\text{eq}}}\,\dot\varepsilon_{\text{eff}}$$

**Parameters:** one critical value $C_{\text{crit}}$, material-specific, from a tensile
or upset test. Typical range for aluminium alloys is $C_{\text{crit}} \sim 0.2$–$0.5$,
but it *must* be calibrated for your alloy to mean anything quantitatively.

**Getting $\sigma_1$:** largest eigenvalue of $\sigma_{ij} = s_{ij} + \sigma_m\delta_{ij}$,
or equivalently $\sigma_1 = \sigma_m + s_1$ where $s_1$ is the largest deviatoric
eigenvalue. Available from the Lode computation in §7.6.

**Why it suits FSW — and ⚠️ a correction to the obvious expectation.** The Macaulay
bracket encodes "compression doesn't damage", so the natural expectation is that the
shoulder's forging action makes $\sigma_1<0$ under the tool and $C$ stops accumulating
until the wake.

**Measured on our own nodal fields, that is NOT what happens.** $\sigma_1>0$ at
**~95 %** of active nodes inside the shoulder radius (4.9–5.9 % negative, checked
across eight PinShapes cases). The bracket almost never fires.

The reason is in the decomposition above: $\sigma_1=\sigma_m+\sigma_{\text{eq}}\hat s_1$,
and FSW flow is strongly **shear**-dominated, so the deviatoric eigenvalue carries far
more than the mean stress:

| term, median under the shoulder (A1) | value |
|---|---|
| $\hat s_1$ (deviatoric) | **+0.87** |
| $\eta=\sigma_m/\sigma_{\text{eq}}$ (volumetric) | **−0.05** |
| $\sigma_1/\sigma_{\text{eq}}$ | **+0.80** |

$\sigma_1$ therefore stays tensile even where the **mean** stress is compressive —
which it is at 55 % of those nodes.

⚠️ **What this means for the two laws.** They are *not* two versions of the same
indicator. Rice–Tracey reads the **volumetric** state through $\exp(3\eta/2)$ and *is*
suppressed by compression. Cockcroft–Latham here reads mainly the **deviatoric**
state and is not. That is a better reason to run both than "agreement is evidence":
they interrogate different parts of the stress tensor.

---

#### (c) GTN — full porous plasticity

The state-of-the-art form. Treat the material as porous plastic with $f$ as a genuine
state variable:

$$\boxed{\ \dot f = \dot f_{\text{growth}} + \dot f_{\text{nucleation}}\ }$$

$$\dot f_{\text{growth}} = (1-f)\,\dot\varepsilon^p_{kk}, \qquad
\dot f_{\text{nucleation}} = \frac{f_N}{s_N\sqrt{2\pi}}\exp\!\left[-\tfrac12\left(\frac{\bar\varepsilon - \varepsilon_N}{s_N}\right)^2\right]\dot{\bar\varepsilon}$$

**The growth term** is pure mass conservation: if the matrix is incompressible, any
volumetric plastic strain $\dot\varepsilon^p_{kk}$ must come from the voids opening or
closing. **This term carries the sign naturally** — under compression $\dot\varepsilon^p_{kk} < 0$
and $f$ genuinely *decreases*. That's the closure physics (a) lacks.

**The nucleation term** is a Gaussian in accumulated strain: new voids appear at
inclusions/particles, peaking at a characteristic strain $\varepsilon_N$.

**Parameters:**

| Symbol | Meaning | Typical (Al) |
|---|---|---|
| $f_0$ | initial void fraction | $10^{-4}$–$10^{-3}$ |
| $f_N$ | volume fraction of nucleating particles | 0.02–0.04 |
| $\varepsilon_N$ | mean nucleation strain | 0.1–0.3 |
| $s_N$ | std dev of nucleation strain | 0.1 |
| $q_1, q_2, q_3$ | Tvergaard fitting constants | 1.5, 1.0, $q_1^2$ |
| $f_c$ | critical $f$ for coalescence | 0.05–0.15 |
| $f_F$ | $f$ at final failure | 0.15–0.25 |

**Seven-plus parameters, all material-specific.** That's the cost. And note
$\dot\varepsilon^p_{kk}$ requires the *plastic volumetric* strain rate, which strictly
needs a porous-plasticity constitutive solve, not just the incompressible $D_{ij}$
you get from the velocity field. Approximating it is possible but is a modelling
decision, not a free lunch.

---

#### Which to use

| | Parameters | Captures closure | Effort |
|---|---|---|---|
| **Rice–Tracey** | none | no | trivial |
| **Cockcroft–Latham** | 1 | implicitly (switches off) | trivial |
| **Lee–Dawson** (He et al.) | ~10 + a 2nd ODE for $\kappa$ | **no — clamped at $\sigma_m\le0$** | moderate |
| **GTN** | 7+ | yes, properly | substantial |

Note that the published FSW precedent (He et al., Lee–Dawson) sits in the *no-closure*
column. Only GTN and the dedicated void-closure models (Saby, Zapara) actually let
$f$ decrease under compression.

**Recommendation for a first pass: run (a) and (b) together.** Both are a handful of
flops per particle per step, neither needs calibration to give a *relative* ranking,
and if they agree on where the risk concentrates, that's real evidence. Only reach for
GTN if you have calibration data and the relative ranking proves insufficient.

### He, Dawson & Boyce (2008) — read from the actual paper

> ✅ **Verified against the full text** (`Downloaded/FSW/he2008.pdf`), 2026-09-23.
> An earlier draft of this primer paraphrased their equation and got several details
> wrong. What follows is transcribed from the paper.

Their structure:

1. Solve the FSW flow **once**, steady-state, Eulerian FE, 3D → $\mathbf{u}$, $\theta$ (temperature),
   $\sigma$ on a fixed mesh (8000 20-node brick elements, 304L stainless).
2. Extract **streamlines** from that field.
3. Integrate the state-variable ODEs **along the streamlines** — their Eqs. (8)–(9):

$$\kappa(x_s) = \kappa_0 + \int_0^T \dot\kappa\,dt, \qquad
\Phi(x_s) = \Phi_0 + \int_0^T \dot\Phi\,dt$$

**The void growth equation — their Eq. (1)**, the Lee–Dawson model:

$$\boxed{\ \dot{\Phi} = \psi_0\,\bar{D}\,\Phi\,\frac{\exp\!\left(c_1\,\sigma_m/\kappa\right)}{1-\Phi}\ }$$

with the rate/temperature prefactor, their Eq. (2):

$$\psi_0 = \frac{f_0(\kappa_0/G)^m \exp\!\left[c_2 - Q/(R\theta_r)\right]}{\bar{D}_r}$$

Notation map to §7:

| Theirs | Meaning | Ours (§7) |
|---|---|---|
| $\Phi$ | void volume fraction (porosity) | $f$ |
| $\bar{D}$ | effective deviatoric deformation rate, $[\tfrac23\text{tr}(\mathbf{D}'\cdot\mathbf{D}')]^{1/2}$ | $\dot\varepsilon_{\text{eff}}$ |
| $\sigma_m$ | mean stress, $\text{tr}(\boldsymbol\sigma)/3$ | $\sigma_m$ ✓ |
| $\kappa$ | **strength / hardness** — a state variable | — (see below) |
| $\theta_r, \bar{D}_r$ | reference temperature and deformation rate | — |

**Three things I had wrong, worth internalising:**

⚠️ **1. The scaling stress is $\kappa$ (strength), not $\sigma_{\text{eq}}$.** Their §2 notes that
$\bar\sigma$ and $\sigma_Y$ are "two of the commonly used scaling factors"; they chose the
strength $\kappa$. So their driving ratio is $\sigma_m/\kappa$ — and every figure in the paper
plots $\sigma_m/\kappa$, not a triaxiality. It is a *sibling* of triaxiality, not the same number.

⚠️ **2. $\kappa$ is a second state variable with its own ODE** — their Eq. (5), a Voce-type
saturation law:

$$\dot\kappa = h_0\bar{D}\left(1 - \frac{\kappa}{\kappa^{\text{sat}}}\right)^{n_0}, \qquad
\kappa^{\text{sat}} = (C/\varphi)^{m_0},\quad \varphi = \theta\ln(D_0/\bar{D})$$

So it is a **coupled two-ODE system** integrated along each streamline, not one. $\kappa$
tracks strain hardening and dynamic recovery; the material hardens as it deforms,
which feeds back into the void growth rate.

⚠️ **3. Their model does not predict closure.** From §2, verbatim: *"Because the closure of
voids is energetically more difficult than opening them, when the mean stress is
hydrostatic compression ($\sigma_m \le 0$), the void growth rate is assumed to be zero, but
never negative."* (following Murakami & Ohno.) So under the shoulder, $\dot\Phi$ is **clamped
to zero** — porosity freezes rather than healing.

$$\dot\Phi = \begin{cases}
\psi_0\bar{D}\Phi\,\dfrac{\exp(c_1\sigma_m/\kappa)}{1-\Phi} & \sigma_m > 0 \\[2ex]
0 & \sigma_m \le 0
\end{cases}$$

This is a *modelling choice*, and it's the conservative one. If we want genuine closure
physics we have to go beyond this paper — which is what the Saby/Zapara void-closure
literature (§8, and the primer's §8 model menu) is for.

**Their parameters** (Table 3, 304L stainless): $\Phi_0 = 0.01\%$, $\theta_r = 373$ K,
$\bar{D}_r = 1.0$ s⁻¹, $c_1 = 1.5$, $c_2 = 96.0$. Note $c_1 = 1.5$ makes their exponential
$\exp(1.5\,\sigma_m/\kappa)$ numerically identical in form to Rice–Tracey's $\exp(3\eta/2)$ —
coincidence of calibration, but a useful sanity anchor.

---

### Eq. (1), term by term

Take the equation apart and ask of each factor: what is it, and why is it there?

$$\dot{\Phi} = \underbrace{\psi_0}_{\text{(A)}}\;\underbrace{\bar{D}}_{\text{(B)}}\;\underbrace{\Phi}_{\text{(C)}}\;\underbrace{\exp\!\left(c_1\sigma_m/\kappa\right)}_{\text{(D)}}\;\underbrace{\frac{1}{1-\Phi}}_{\text{(E)}}$$

---

**(C) $\Phi$ — the multiplicative factor. Why is growth proportional to what's already there?**

This is the most important structural feature, and it has a clean geometric reason.

A void of radius $R$ in a matrix straining at rate $\dot\varepsilon$ grows at $\dot R \propto R\dot\varepsilon$ —
the *surface* moves outward in proportion to the *existing* size, because the surrounding
material is being stretched uniformly. Volume $\propto R^3$, so $\dot\Phi \propto \Phi$.

Two consequences:

- **Growth is exponential in time**, not linear. Damage accelerates.
- **$\Phi = 0$ is a fixed point.** If there are no voids, none ever appear. This is why
  they must assume a nonzero initial porosity ($\Phi_0 = 0.01\%$, "uniformly distributed")
  — the model describes *growth*, not *nucleation*. It is a pure growth law.

> This is why I suggested integrating $\ln\Phi$ rather than $\Phi$ in §9: the equation is
> multiplicative, so $d(\ln\Phi)/dt$ removes the stiffness and keeps $\Phi>0$ automatically.

---

**(B) $\bar{D}$ — the deformation rate. Why does damage need deformation?**

$\bar{D} = [\tfrac23\text{tr}(\mathbf{D}'\cdot\mathbf{D}')]^{1/2}$ is exactly the
$\dot\varepsilon_{\text{eff}}$ of §7.2 (the prime denotes the deviatoric part of $\mathbf{D}$;
since plastic flow is isochoric, $\mathbf{D}' = \mathbf{D}$ anyway).

It sets the **clock**. Voids grow *per unit strain*, not per unit time — a stationary
stressed material does not accumulate damage in this model (no creep). Multiplying by
$\bar D$ converts "damage per unit strain" into "damage per unit time".

This is why it belongs on a **pathline**: a particle only accumulates damage while it is
being deformed, and only in proportion to how much deformation it experiences. Two
particles reaching the same place by different routes carry different $\Phi$.

---

**(D) $\exp(c_1\sigma_m/\kappa)$ — the driving force. Why exponential?**

$\sigma_m$ tries to pull the void open (if positive) or squeeze it shut (if negative).
$\kappa$ is how hard the material resists. The ratio is dimensionless: **"pull, in units
of the material's strength."**

The *exponential* form comes from the underlying plasticity solution. Rice & Tracey (1969)
solved the rigid-plastic field around a spherical void and found growth rate
$\propto\exp(3\sigma_m/2\sigma_Y)$. The physical reason: opening a void requires the
surrounding shell to yield, and the *volume* of material that must yield falls off
exponentially as the confining pressure rises. So the response to triaxiality is
exponential, not linear.

| $\sigma_m/\kappa$ | $\exp(1.5\times)$ | Regime |
|---|---|---|
| $+1$ | 4.48 | strong tension — rapid growth |
| $+0.5$ | 2.12 | mild tension |
| $0$ | 1.00 | pure shear — baseline |
| $-0.5$ | (0.47) | **clamped to 0** |
| $-1$ | (0.22) | **clamped to 0** |

Remember the clamp (§ above): He et al. set $\dot\Phi = 0$ for $\sigma_m \le 0$, so the
bracketed values never occur in their simulations.

**Why $\kappa$ and not $\sigma_{\text{eq}}$?** Because $\kappa$ is what the material can
*currently* resist, including its accumulated hardening history. $\sigma_{\text{eq}}$ is what
it *is currently experiencing*. For a question of the form "can this stress open a void
against the material's resistance?", the resistance is the right denominator. Both are
defensible; He et al. chose resistance.

---

**(E) $1/(1-\Phi)$ — the porosity correction. Why?**

If a fraction $\Phi$ of the volume is void, the remaining *solid* fraction is $1-\Phi$.
The same macroscopic deformation must be accommodated by less material, so the local
strain in the matrix is amplified by $1/(1-\Phi)$.

Consequences:

- At $\Phi \ll 1$ (FSW: $\Phi \sim 10^{-4}$) this factor is $\approx 1$ and **negligible**.
- As $\Phi \to 1$ it diverges — the **coalescence / runaway failure** limit.

So for our regime it's essentially inert. It matters for fracture prediction, not for
the small-porosity susceptibility mapping we want. Worth knowing you could drop it
without changing anything at $\Phi\sim10^{-4}$.

---

**(A) $\psi_0$ — the rate/temperature prefactor. What sets the absolute scale?**

$$\psi_0 = \frac{f_0(\kappa_0/G)^m \exp\!\left[c_2 - Q/(R\theta_r)\right]}{\bar{D}_r}$$

| Symbol | Meaning |
|---|---|
| $f_0$ | reference rate constant (Hart's model, $8.03\times10^{26}$ s⁻¹ for 304L) |
| $\kappa_0$ | initial strength (150 MPa) |
| $G$ | shear modulus (73 GPa) |
| $m$ | exponent (5.0) |
| $Q$ | activation energy (410 kJ/mol) |
| $R$ | gas constant |
| $\theta_r$ | reference temperature (373 K) |
| $\bar{D}_r$ | reference deformation rate (1.0 s⁻¹) |
| $c_2$ | fitting constant (96.0) |

Structurally this is an **Arrhenius prefactor**: $\exp[c_2 - Q/(R\theta_r)]$ is a thermally
activated rate, $(\kappa_0/G)^m$ is a dimensionless strength ratio, and dividing by
$\bar D_r$ makes the whole product dimensionless so that $\psi_0\bar{D}$ has units of s⁻¹.

⚠️ **Note $\theta_r$ is a *reference* temperature, fixed at 373 K — not the local
temperature.** So in their formulation $\psi_0$ is a **constant**, evaluated once. The
temperature dependence of the process enters through $\kappa$ (whose saturation
$\kappa^{\text{sat}}$ depends on $\theta$ via the Fisher factor) rather than directly
through $\psi_0$. Easy to misread as a local Arrhenius term; it isn't.

---

### The companion equation: $\kappa$, term by term

$$\dot\kappa = \underbrace{h_0}_{\text{hardening modulus}}\;\underbrace{\bar{D}}_{\text{clock}}\;\underbrace{\left(1 - \frac{\kappa}{\kappa^{\text{sat}}}\right)^{n_0}}_{\text{approach to saturation}}$$

A **Voce-type** law. Read it as: strength increases with deformation, but the increment
shrinks as $\kappa$ approaches a ceiling $\kappa^{\text{sat}}$, and stops entirely at it.

- **Physically:** dislocations accumulate (hardening) but also annihilate by dynamic
  recovery and recrystallisation (softening). $\kappa^{\text{sat}}$ is where the two balance.
- **Why FSW needs this:** strains are enormous ($\bar\varepsilon\sim10$–$100$). Without a
  saturation ceiling, any monotonic hardening law would predict absurd strengths. The paper
  says exactly this — saturation "assumes that the strength will not increase beyond a
  reasonable limit even if the strain might become very large."
- **Where temperature enters:** $\kappa^{\text{sat}} = (C/\varphi)^{m_0}$ with
  $\varphi = \theta\ln(D_0/\bar{D})$ — the Fisher factor. Hotter → lower saturation
  strength → softer material. This is the physically meaningful temperature coupling.

**Why the two ODEs are coupled:** $\kappa$ appears in the denominator of $\Phi$'s driving
term (D). So a particle that hardens more resists void growth more. You cannot integrate
$\Phi$ without simultaneously integrating $\kappa$.

---

### Summary table

| Factor | Name | Why present | Matters at $\Phi\sim10^{-4}$? |
|---|---|---|---|
| $\psi_0$ | Arrhenius prefactor | sets absolute rate scale | constant — fold into calibration |
| $\bar{D}$ | deformation rate | damage accrues per unit strain | **yes — the clock** |
| $\Phi$ | current porosity | growth $\propto$ existing void size | **yes — makes it exponential** |
| $\exp(c_1\sigma_m/\kappa)$ | driving force | tension opens, compression closes | **yes — the discriminator** |
| $1/(1-\Phi)$ | matrix correction | less solid carries the strain | no — $\approx 1$ |

**Their results, for calibration of expectations:** porosity goes from 0.01% initial to
~0.017% max with a smooth pin (a 70% increase), and ~0.105% with a threaded pin (~10×).
Highest porosity on the **advancing side**, on the **trailing** side of the pin — because
that's where $\sigma_m/\kappa > 0$ (tensile). Upstream of the pin the material is forged
(compressive) and nothing grows. They call the agreement with experiment *qualitative*.

---

**Why it still maps onto JAXTrace.** The method is a Lagrangian integration of state
variables along pathlines through a precomputed Eulerian field — exactly what your
tracking does. Instead of transporting position only, carry $(\Phi, \kappa)$ per particle
and integrate both alongside the position update.

Two honest differences from what I first wrote:

- It's **two coupled ODEs**, not one. Still pointwise, still vmaps — just two scalars.
- They needed $\kappa$ because their $\psi_0$ and scaling both depend on it. If you use
  a triaxiality-based law instead (Rice–Tracey, §8a), you avoid the second ODE but lose
  the strain-hardening coupling.

In 2008 they solved this on a 8000-element mesh with streamline integration in PETSc.
The throughput gap to $10^6$ GPU pathlines is real — but note their bottleneck was the
*coupled FE solve*, not the streamline count, so the gain is less dramatic than I
implied earlier.

---

## 8b. Did anyone build on He et al.? (citation check, 2026-09-23)

Searched the papers citing He 2008 and semantically similar work. **The honest finding:
almost nobody continued this specific line.**

**He 2008 has ~11 citations in 18 years (0.6/yr).** For comparison, Dialami 2020 has 124
(19/yr) and Nielsen 2009 has 113. Of those 11, most cite it in passing in reviews
(He 2014, Das 2024, Boukraa 2025); `Daw14` is the same group (Dawson & Boyce) applied to
foil markers rather than damage.

**No follow-up paper takes the Lee–Dawson-along-streamlines method and extends it.**

### What that means

Two readings, and both are probably partly true:

1. **Opportunity.** The approach was sound, was published in a good ASME journal, and was
   simply not picked up — possibly because in 2008 the FE machinery was heavy, and by the
   time cheap computing arrived the field had moved to CEL+VOF, which gives a picture of
   the void directly rather than a porosity scalar.
2. **Warning.** A method that nobody adopted in 18 years may have failed quietly. Their
   own agreement was only *qualitative*, and their porosity changes were small
   (0.01% → 0.017%). It is possible the signal is just too weak to be useful, and that
   people tried and did not publish the negative result.

**Practical implication for us:** Option 2 is closer to *unexplored* than to *established*.
That is good for novelty and bad for risk. The cheap version — Rice–Tracey plus
Cockcroft–Latham on an existing field, §8 — is the right way to find out which reading
is correct before investing.

### The work that actually did continue

Three lines are closer to our plan than He's own successors:

**Ansari, Agiwal, Franke, Zinn, Pfefferkorn & Rudraraju (2022, 2023)** — the most active
current group on FSW void prediction. Notably they **abandoned the damage-ODE route**:
they predict voids from temperature, plastic strain rate and material velocity fields
directly, validated across three alloys (6061-T6, 7075-T6, 5053-H18). This is the
flow-deficiency route (§6), not the porosity route. Worth reading as the contemporary
state of the art, and as evidence about which route practitioners find works.
`10.1115/1.4062270`

⚠️ **Fraser, Kiss, St-Georges & Drolet (2018)** — *closest prior art to Options 2–3, and
I should have found it earlier.* GPU-parallelised meshfree SPH (SPHriction-3D), proposes
**"a novel metric that automatically evaluates the presence and severity of defects in the
weld zone"**, validates against AA6061-T6, and then uses it to **find optimal advancing
speed and rpm by minimising defect volume**. That is structurally the same ambition as
Options 2–3: GPU throughput + a continuous defect metric + parameter optimisation.
**Read this before claiming novelty.** `10.3390/met8020101`

⚠️ **Cao, Fraser, Song, Drummond & Huang (2021), J. Comput. Phys.** — ML + reduced-order
computation of an FSW model. Same group as Fraser. Prior art for the ROM-surrogate
direction (Option 3). Need to check *what* was reduced and whether any defect field was
the ROM target. `10.1016/j.jcp.2021.110863`

**Cho, Kim & Kang (2008)** — independent description of the same strength-evolving-along-
streamlines machinery (Voce saturation), useful as a second explanation of the $\kappa$
ODE if the He paper's notation is dense. `10.4028/www.scientific.net/MSF.575-578.805`

### Revised novelty position

My earlier framing ("nobody has done continuous susceptibility fields at scale") was
**too strong**. Fraser 2018 did GPU + a defect metric + optimisation. What appears to
remain genuinely open:

- Fraser's defect metric is derived from **SPH particle deficiency** (a geometric/density
  measure), not from an integrated **constitutive damage law**. The physics content differs.
- Nobody appears to have combined **ROM over the parameter space** with a **pathline-
  integrated damage field** as the ROM target.
- Nobody has done the triaxiality/Lode-dependent **closure** branch in FSW at all.

That is a narrower and more defensible claim than the one in the deck. Worth reading
Fraser 2018 and Cao 2021 in full before the next presentation.

---

## 9. What was actually built, step by step — and why each step

⚠️ This section replaces an earlier sketch written *before* the work was done. It
now describes what exists, in the order it happened, with the reasoning. If you are
presenting the current state, this is the section to know.

### 9.0 The one-paragraph version

We take the FOM's velocity and pressure fields, compute the **stress triaxiality**
η = σ_m/σ_eq at every mesh node, and find that tensile (void-opening) conditions
are sharply localised to the **advancing-side wake** — the literature's void
signature, recovered with no fitting. We then built the machinery to *integrate*
damage along particle **pathlines** inside the existing GPU tracker, at a measured
cost of under 2 %. Along the way we established that the turbulence and cavitation
analogies do not transfer, and that the metric is far more sensitive to pressure
than to velocity.

---

### 9.1 Phase 0 — conventions, established by measurement not assumption

Two things had to be pinned down before any number meant anything.

**σ_m = −p (the sign of mean stress).** Solid mechanics takes tension as positive;
fluid solvers take compression as positive. Getting this backwards flips every
triaxiality value, so voids would appear to grow exactly where they close. We did
not assume it — we measured pressure upstream and downstream of the tool:

| region (case 000) | P median |
|---|---|
| upstream, x < −2 mm (material being forged) | **+1.70 MPa** |
| downstream, x > +2 mm (the wake) | **−1.35 MPa** |

Pressure is high where material is being forged, so P is compressive-positive, so
σ_m = −P. Confirmed by physics, not by convention.

**Which flank is "advancing".** ⚠️ *Advancing* and *retreating* are defined relative
to **tool travel**, not to material flow — a distinction we initially got backwards.
Material flows +x past a stationary tool, so the tool travels −x in the workpiece
frame, so the advancing flank is the one where the tool-surface velocity has
u_x < 0. Measured per case rather than assumed, because rotation sense differs
between cases:

| flank | tool-surface u_x | |
|---|---|---|
| +y | +0.093 m/s | retreating |
| −y | −0.050 m/s | **advancing** |

---

### 9.2 Stage 1 — the derived fields, and the headline result

`jaxtrace/damage/fields.py` turns (u, p, T) into the nodal scalars §7 defines:
ε̇_eff, σ_m, σ_eq, η, σ₁/σ_eq.

**The result, on 64 cases:** η is **sharply positive in the advancing-side wake and
negative in the retreating wake.** On case 000, 94 % of nodes are in tension in the
advancing wake versus 1 % in the retreating wake. This is the literature's void
signature, and we recovered it from the FOM fields with **no fitted parameters**.

⚠️ It was derived *independently* of the advancing/retreating labelling and then
found to agree with it — two separate measurements pointing at the same flank.

**One bug worth knowing about, because it would have been invisible.** The
shape-function gradients were computed as the *columns* of J⁻ᵀ instead of the
**rows of J⁻¹** (they are the same thing only for axis-aligned elements). The
incompressibility check ∇·u caught it: 0.54 where it should be ~0. Cross-checked
against VTK's own gradient filter (0.005) to confirm ours was the wrong one, not
VTK's. After the fix: 0.041. **A test that passes on symmetric inputs and fails on
skewed ones is now a permanent regression guard.**

---

### 9.3 The three quantitative "no"s — why we are not doing what turbulence does

Worth being able to state these crisply, because the natural question is *why not
just use the two-phase / turbulence machinery?*

| analogy | the number | verdict |
|---|---|---|
| **turbulence** | Re = ρUL/μ_eff ≈ **10⁻⁷–10⁻⁴** | Stokes creeping flow, **7–12 orders** below transition. There is no turbulence to model. |
| **cavitation** | σ_cav = (p−p_v)/(½ρU²) ≈ **7×10⁴** (needs ≲1) | And aluminium's vapour pressure at 750 K is ~10⁻⁹ Pa against a **+10⁸ Pa compressive** stir zone — **11 orders** the wrong way, *and* the wrong sign. |
| **vorticity as a defect indicator** | rigid rotation gives **D = 0** exactly | The tool spins everything, so vorticity is large even in perfect welds. Damage needs **D** (stretching), not **W** (spin). Proven analytically in the test suite. |

So the mechanism is **volumetric filling failure**, not cavitation: a transient
cavity behind the pin that the material flow fails to refill before the tool moves
on. Voids in FSW are a *bookkeeping* failure, not a nucleation event.

---

### 9.4 Phase 1 — carrying damage along a pathline, at ~1 % cost

**Why pathlines and not a grid?** Because damage is a **history integral**:
D = ∫ f dt *along the trajectory of a material particle*. A grid cell has no
history — it has whatever particles happened to pass through it. §6b.3 gives the
formal reason: following the particle makes the material derivative collapse to an
ordinary d/dt, so the advective term vanishes and no gradient of the damage
variable ever appears. Solving it on a grid instead would mean fighting numerical
diffusion, which smears exactly the sharp gradients we care about.

**What was added.** A build-time flag in the production RK4 kernel:

```python
damage_scalars_gpu = None        # -> the accumulator is COMPILED OUT entirely
edot_k1 = interpolate_scalar_single(pos, elem_k1, damage_field)
dmg_new = dmg + dt * edot_k1     # explicit Euler at the k1 stage
```

Design choices, each for a stated reason:

| choice | why |
|---|---|
| **Euler at k1**, not RK4 on the accumulator | the driver is itself an interpolated field, so 4th-order accuracy on top of interpolation error buys nothing while costing 4 extra samples/step |
| a **separate** scalar sampler | the velocity path stays byte-identical; the element search is *reused* (`elem_k1`), so the marginal cost is one 4-float gather already in cache |
| damage appended **last** in the return tuple | all existing 2-tuple call sites keep working unchanged |
| skip policies **mirrored** | damage must not accrue on a step whose position was discarded |
| **ln Φ**, not Φ, for Rice–Tracey | the ODE is multiplicative; log space cannot underflow and is stable to 1.2e-13 over 10,000 steps |

**⚠️ The mistake worth telling the team about.** The first implementation went into
`rk4_fully_fused_timedep.py` — a module the FSW runs **never execute**.
`run_tracking.py` gets its kernel from `create_rk4_comparison` in
`benchmark_femuss_comparison.py`, which holds its own inlined copy. The edit, the
tests and the first throughput number were all real and all described a kernel
nobody runs. **Lesson: benchmark through the production entry point, and check how
existing code calls a thing before assuming which thing it calls.**

**The throughput gate**, measured on the production kernel after the port:

| platform | slowdown | gate | accumulator live? |
|---|---|---|---|
| RTX 5090 (CUDA) | +0.15 … +0.51 % | <3 % ✅ | 99.8 % of particles |
| MI250X (ROCm, LUMI) | **+1.77 %** | <3 % ✅ | **100.0 %** |

⚠️ The "accumulator live" column exists because **a timing gate would pass
identically if the damage code were dead.** The gate now fails if nothing
accumulated.

---

### 9.5 The confound, and how it was resolved — the most subtle part

This is the part most likely to be challenged, so it is worth knowing in full.

**The observation.** Ranking the 20 PinShapes cases by their advancing/retreating
growth ratio separates them by tool family (threads > flutes > flats). Good news —
except the number of nodes the sampling window captured, `n_selected`, spanned
**2,375 to 114,260 — a 48× range at identical tool radius** — and correlated with
the answer at **ρ = +0.571**.

⚠️ **A statistic that depends on how much you sampled is not a result about the
physics.** So the family separation was provisional.

**The diagnosis.** We printed the raw radial velocity profile instead of trusting a
peak-finder, and found:

| r/R | nodes | \|u_θ\| |
|---|---|---|
| below 0.357 | **0** | *no material — this is the pin* |
| 0.397 | 124,564 | 0.100 |
| 0.476 | 12,984 | 0.031 |
| 0.674 | 339 | 0.007 |

Two facts fell out. **`r_tool` = 7 mm is the SHOULDER radius, not the pin** — the
pin surface sits at ~2.4 mm. And the window r/R ∈ [0.40, 0.70] had **72 % of its
nodes at r/R ≤ 0.45 and only 10 % beyond 0.50** — an effective thin annulus whose
position moves with each pin's geometry while the window itself does not. *That* is
the mechanism of the correlation.

**The fix.** Detect the pin surface per case (`r_pin`) and anchor the window to it,
running outward through the velocity decay. Result on 64 cases:

| population | ρ static | ρ pin-anchored |
|---|---|---|
| **PinShapes pooled** | **+0.409** | **−0.219** |
| A-Flats | +0.179 | **0.000** |
| D-Concavity | +0.400 | **0.000** |

**The confound is gone, and the family separation survives as geometry.** Also
answered: δ (shear-layer thickness) spans **0.53–2.80 mm** across cases, so the
cylindrical cohort's value does *not* transfer to featured pins.

---

### 9.6 The independent check nobody had done

⚠️ Every σ_eq so far came from the FOM's Norton tables, with no way to verify it —
and σ_eq is the **denominator** of η, so an error there biases every triaxiality
number.

Then we found the solver exports its **own deviatoric stress** as CellData
(6 components; it returns `None` from `GetPointData()`, which is why it was missed).
Verified it is the pure deviator: max|trace| = **2.98e-8 Pa**, machine zero. So
σ_eq = √(3J₂) is available with **no rheology assumption at all**.

| route | σ_eq vs the solver's |
|---|---|
| **Norton** (the case's own `.mat`) | **0.569×**, constant to ±0.8 % across 64 cases |
| **Sellars–Tegart** (literature constants, *no fitting*) | **0.871×** |

⚠️ **A generic literature model beats the case's own material file.** That is
backwards, and it localises the problem: not the solver's stress output, and not a
convention factor (ruled out — the Norton exponent m ≈ 0.09 damps *any* strain-rate
redefinition to 1.05×, nowhere near 1.76×). It points at **how the
`Plastic_viscosity` table is being interpreted**.

**What this means for the conclusions:** the adv/ret **ratio** divides the factor
out, so the family ranking and every conclusion built on it stand. **η magnitudes
carry a known ~1.76× correction** until this is resolved. That is the honest
statement.

---

### 9.7 Where it goes next

| stage | status |
|---|---|
| Phase 0 conventions, Stage 1 fields, 64-case survey | ✅ done |
| Phase 1 pathline accumulator + throughput gate | ✅ done, both platforms |
| M1/M2 δ and the window confound | ✅ resolved |
| M5 Norton vs Sellars–Tegart ranking | ✅ **PASS — ρ = +0.9995, 0/41 flank flips** |
| M4 pin geometry descriptors | ⚠️ **run** — `n_lobes`/`lobe_depth` on all 20; envelope fails on 9 (coarse STLs). Usable: 13 cases |
| **Phase 2: `eta` + `s1_ratio` nodal fields to the GPU** | ⚠️ **1 of 3 wired — THIS IS NEXT** |
| Phase 3: the damage ODEs (Rice–Tracey ln Φ, Cockcroft–Latham) | ⬜ blocked on Phase 2 |
| Phase 4: deposit to a grid, regress on (v_adv, ω) | ⬜ |

**The end goal**, in one sentence: a **continuous susceptibility field** over the
weld, rather than the binary defect/no-defect that most of the literature reports —
and because it is a field, it composes with the ROM to give an instantly-evaluable
process-window map.

---

## 9b. Vocabulary of the work itself

§11 glossaries the *physics*. These are the terms that come up when discussing the
**implementation**, which are the ones most likely to trip you up in a meeting.

### The data

| term | what it is |
|---|---|
| **FOM** | Full-Order Model — the original finite-element simulation. The ground truth here. |
| **ROM** | Reduced-Order Model — the fast surrogate built from FOM snapshots. Deferred to last in this plan. |
| **POD** | Proper Orthogonal Decomposition — the basis-extraction step (essentially PCA on simulation snapshots) that builds a ROM. |
| **PVTU / VTU** | Parallel VTK Unstructured grid. A `.pvtu` is a small XML index listing `.vtu` *pieces*, one per solver rank. |
| **PointData / CellData** | Fields stored at mesh **nodes** vs at element **centres**. ⚠️ `Stress` and `Strain` are CellData here; asking for them as PointData returns nothing. |
| **`Displacement`** | ⚠️ Misleadingly named — in these files it **is the velocity** in m/s, confirmed with the author. |
| **`LEVEL`** | The level-set field: < 0 means inside the tool. Used to find the tool axis and mask tool interior. |
| **VTKHDF** | The HDF5-based VTK format used for time-series output; one archive rather than thousands of files. |
| **snapshot vs phase average** | One timestep, vs an average over a full tool revolution (166 steps). ⚠️ They differ by a median 3.8 % but up to **+48 %** on sparsely-sampled cases. |

### The tracker

| term | what it is |
|---|---|
| **pathline** | The trajectory of one material particle over time. **What damage integrates along.** |
| **streamline** | Tangent to the velocity field at one instant. Equals a pathline only for steady fields. |
| **RK4** | 4th-order Runge–Kutta — the time integrator, 4 velocity samples per step. |
| **L0 / L1 / L2** | The three-tier element search. **L0**: is the particle still in its cached element? **L1**: hop to face-neighbours. **L2**: global search. Cheap → expensive. |
| **octree / Morton** | Spatial index for the global (L2) search. Morton ("Z-order") interleaves coordinate bits so nearby points get nearby integer keys. |
| **barycentric coordinates** | The four weights (summing to 1) expressing a point inside a tetrahedron. How fields are interpolated. |
| **initial assignment** | The dedicated routine that locates seed particles at the start. ⚠️ Not the same as the in-kernel L2 recovery — confusing the two produced a false "tracker bug" report. |
| **seeding** | Where particles start. ⚠️ Box seeding put 31 % of particles *outside* the mesh, because the FSW domain is not a box. |
| **deposition / union** | Mapping per-particle values back onto a grid. The **max** reduction gives "worst history any material passing here endured"; **count** gives coverage. |

### The GPU and the machines

| term | what it is |
|---|---|
| **JAX** | The array/autodiff library. Compiles Python to GPU kernels via XLA. |
| **JIT / XLA** | Just-In-Time compilation. First call compiles (slow), later calls are fast — so benchmarks must discard the first call. |
| **`vmap`** | Vectorising map: write the per-particle function once, run it across all particles in parallel. |
| **jaxpr** | JAX's intermediate representation. Inspectable — we used it to *prove* the damage code is absent when the flag is off. |
| **build-time flag** | A Python boolean closed over at trace time, so the disabled branch is not in the compiled graph at all. Zero runtime cost. |
| **CUDA / ROCm** | NVIDIA's and AMD's GPU stacks. The workstation is CUDA (RTX 5090); LUMI is ROCm (MI250X). |
| **MI250X / GCD** | LUMI's GPU. Each MI250X has two Graphics Compute Dies that appear as separate devices. |
| **SLURM / sbatch / squeue / sacct** | LUMI's job scheduler: submit, check the queue, check history. |
| **singularity** | The container runtime on LUMI; the JAX+ROCm environment ships as a `.sif` image. |
| **partition** | Which queue. `small` = CPU-only, `small-g` = with GPU. Our analysis scripts are NumPy/VTK, so they want `small`. |
| **`/projappl` vs `/scratch` vs `/flash`** | Code, bulk data, and fast scratch respectively. Convention here: scripts in `/projappl`, outputs in `/scratch`. |

### The statistics

| term | what it is |
|---|---|
| **Spearman ρ** | Rank correlation: does one quantity increase with another, regardless of the relationship's shape. Robust to outliers. |
| **confound** | A third variable driving both things you are comparing, so the apparent relationship is not causal. Here: sampled volume. |
| **Jensen's inequality** | mean(f(x)) ≠ f(mean(x)) for a curved f. ⚠️ I suspected it explained a discrepancy, **tested it, and it did not** — the real cause was mean vs median. |
| **median vs mean** | For a right-skewed quantity like exp(1.5η) the mean sits well above the median. Using the wrong one inflated a ratio from 1.45 to 1.95. |
| **prominence** | How far a spectral peak stands above the surrounding spectrum. A huge prominence on a *flat* spectrum means the spectrum is noise, not that the peak is real. |

---

## 9c. Questions the team will ask, and the honest answer

Ordered by how likely they are. Each answer is one you can defend with a number.

**"Why not just use a two-phase / VOF solver, like the bubble people?"**
Because there is no second phase. FSW is solid-state — no melting, no vapour. The
void is *empty space that failed to fill*, not a gas bubble. And the two numbers
that would justify borrowing that machinery both fail by orders of magnitude:
Re ≈ 10⁻⁷–10⁻⁴ (no turbulence) and the cavitation number is 10⁴–10⁵ times too
large *with the wrong sign* (§9.3).

**"Isn't high vorticity where defects form?"**
No, and this is provable rather than arguable. Vorticity is the *rotation* part of
the velocity gradient. A rigid rotation has vorticity but **D = 0 exactly** — no
deformation, no damage. The tool spins everything, so vorticity is large even in
sound welds. Damage needs the **stretching** part.

**"How do you know your triaxiality sign isn't backwards?"**
Measured, not assumed: pressure is **+1.70 MPa upstream** (material being forged)
and **−1.35 MPa downstream**, so P is compressive-positive and σ_m = −P (§9.1). And
the tensile region lands on the advancing side, which is where the literature puts
the defects — a second, independent confirmation.

**"Your sampling window looks arbitrary."**
It *was*, and we found the problem ourselves: the window correlated with the answer
at ρ = +0.571 because it was sized from the **shoulder** radius while the physics
sits at the **pin**, which moves between 0.86 and 3.47 mm across cases. Anchoring
the window to the measured pin surface removes the correlation (**+0.409 → −0.219**)
and the family separation survives (§9.5).

**"How much does the damage tracking cost?"**
**+0.15 to +1.77 %** depending on platform, measured A/B on the production kernel
with the accumulator verified live. Under the 3 % budget. The trick is reusing the
element search the velocity step already paid for.

**"Can you trust σ_eq? It comes from a fitted table."**
Only partly, and we can now say so quantitatively — *and* we tested what it costs.
The solver exports its own deviatoric stress, giving a rheology-free σ_eq. Norton
comes in at **0.569×** it, constant to ±0.8 % across 64 cases (§9.6).

Then we re-ran the whole survey under a **completely different rheology**
(Sellars–Tegart, literature constants, no fitting). Across 41 cases the rank
correlation of the adv/ret ratio between the two models is **ρ = +0.9995 with zero
tensile-flank flips**, and the family ranking (C-Threads > B-Flutes > A-Flats >
D-Concavity) is **identical**.

So: **the rheology sets the scale, the flow field sets the pattern.** Rankings and
ratios are trustworthy. Absolute magnitudes move by up to 34 % between models and
carry the ~1.76× question, which is a real caveat but not one that touches any
conclusion drawn from the ordering.

**"Has this been validated against real welds?"**
**No.** No CT scan or macrograph ground truth has been identified yet. Everything
so far is internal consistency: the predicted tensile region lands where the
literature reports defects, and the parameter trends have the expected sign. That
is a *necessary* check, not a sufficient one. ⚠️ Do not let this be overstated.

**"Why is this better than what's already published?"**
Existing work mostly reports **binary** defect/no-defect or a blob in one
cross-section. A per-pathline history integral, deposited to a grid, gives a
**continuous susceptibility field** — and because it is a field, it composes with
the ROM into an instantly-evaluable process-window map.

**"What could still be wrong?"** — §10 is the full list. The three to volunteer:
(1) no experimental validation; (2) the damage models are **growth** laws, so they
need an assumed initial porosity Φ₀ that nobody measures; (3) the metric is far
more sensitive to **pressure** than to velocity, so ROM pressure accuracy is the
binding constraint, not velocity accuracy.

---

## 9d. If you present one slide

> **What:** predict FSW voids by integrating a damage variable along particle
> pathlines through the FOM's own velocity and pressure fields.
>
> **Why it is not the turbulence/cavitation problem:** Re ≈ 10⁻⁷, cavitation number
> 10⁵ too large and the wrong sign, vorticity provably irrelevant. Voids here are a
> **filling failure**, not a nucleation event.
>
> **Key result so far:** tensile (void-opening) stress triaxiality is sharply
> localised to the **advancing-side wake** — 94 % vs 1 % of nodes in tension — with
> **no fitted parameters**, on 64 cases.
>
> **Engineering result:** damage accumulation along pathlines costs **< 2 %**
> throughput on the production GPU tracker.
>
> **Honest status:** no experimental validation yet; σ_eq carries a known ~1.76×
> scale question that affects magnitudes but not the ranking.

---

## 10. Honest caveats — what could go wrong

Worth knowing before committing:

1. **Calibration.** Every damage model has fitted constants ($f_c$, $f_F$, $q_1$, $q_2$
   for GTN; critical $C$ for Cockcroft–Latham). These are material-specific and usually
   fitted to tensile tests. Without calibration data for your alloy, $f$ is qualitative —
   a *relative* susceptibility ranking, not an absolute void fraction. That may be enough
   for classifying a safe zone, but it's a real limitation.

2. **Validation.** Without CT or macrographs on welds whose $(\omega, v)$ you also
   simulated, there's nothing to check against. This is the sharpest open question and
   it's why it's on the deck.

3. **Damage models assume tension.** Most were developed for ductile fracture under
   $\eta > 0$. FSW is mostly $\eta < 0$. The closure branch is less well established —
   this is exactly why the Lode-dependent models (Zapara, Chbihi) matter, and why a
   naive GTN might just predict $f \to 0$ everywhere.

4. **Steady-state assumption.** He et al. used a steady Eulerian field. The real process
   is periodic at the tool rotation frequency, and the cavity-filling argument is
   *inherently* per-revolution. A steady field may average out the very thing that
   causes the defect. Your time-dependent tracking could actually be an advantage here —
   worth thinking about whether to use the periodic field.

5. **Pressure field quality.** $\eta$ (or $\sigma_m/\kappa$) is a *ratio*, so errors in
   numerator and denominator both propagate. If the ROM reconstructs pressure less
   accurately than velocity (plausible — your notes show Pressure needed 4 modes vs
   Displacement's 3), the ratio inherits that noise.

6. **No published FSW precedent actually predicts closure.** He et al. clamp
   $\dot\Phi$ to zero under compression rather than letting porosity heal. If we want
   genuine closure physics we are going beyond the FSW literature into the forming
   literature — defensible, but it becomes a new claim needing its own validation,
   not a reproduction of established work.

---

## 10b. Complete symbol table

Every symbol used in this primer, with units and where it comes from. Units in
brackets; $[-]$ means dimensionless.

### Kinematics — computed from $\mathbf{u}$

| symbol | name | units | definition / source |
|---|---|---|---|
| $\mathbf{u}$, $u_i$ | velocity | m/s | FOM field, stored as `Displacement` in the PVTU |
| $\mathbf{x}$, $x_i$ | position | m | mesh node or particle position |
| $t$ | time | s | |
| $L_{ij}$ | velocity gradient | s⁻¹ | $\partial u_i/\partial x_j$; **not** symmetric |
| $D_{ij}$ | rate of deformation | s⁻¹ | $\tfrac12(L_{ij}+L_{ji})$ — deforms material |
| $W_{ij}$ | spin tensor | s⁻¹ | $\tfrac12(L_{ij}-L_{ji})$ — rotates only, **no damage** |
| $\boldsymbol{\omega}$ | vorticity | s⁻¹ | $\nabla\times\mathbf{u}$; dual of $W_{ij}$. Large everywhere in FSW ⇒ **useless as a defect indicator** |
| $\dot\varepsilon_{\text{eff}}$ | equivalent strain rate | s⁻¹ | $\sqrt{\tfrac23 D_{ij}D_{ij}}$. Median ≈ 30–60 in the stir zone |
| $\bar\varepsilon$ | accumulated strain | $[-]$ | $\int\dot\varepsilon_{\text{eff}}\,dt$ along a **pathline**. FSW: 10–100 |
| $D_{kk}$ | $\nabla\cdot\mathbf{u}$ | s⁻¹ | must be $\approx 0$ (plastic incompressibility) — **numerical check** |

### Stress

| symbol | name | units | definition / source |
|---|---|---|---|
| $\sigma_{ij}$ | Cauchy stress | Pa | symmetric, 6 independent components |
| $p$ | pressure | Pa | FOM field `Pressure`. Compression **positive** (fluid convention) |
| $\sigma_m$ | mean / hydrostatic stress | Pa | $\tfrac13\sigma_{kk} = -p$. **Tension positive.** ⚠️ sign verified empirically, §P0.2 |
| $s_{ij}$ | deviatoric stress | Pa | $\sigma_{ij}-\sigma_m\delta_{ij}$; traceless. Drives flow |
| $\sigma_{\text{eq}}$ | von Mises equivalent stress | Pa | $\sqrt{\tfrac32 s_{ij}s_{ij}} = \sqrt{3J_2}$. Median ≈ 100 MPa |
| $J_2$ | 2nd deviatoric invariant | Pa² | $\tfrac12 s_{ij}s_{ij}$ |
| $\sigma_1$ | max principal stress | Pa | largest eigenvalue of $\sigma_{ij}$ — for Cockcroft–Latham |
| $\sigma_{\text{flow}}$ | flow stress | Pa | yields when $\sigma_{\text{eq}} = \sigma_{\text{flow}}$ |

### The damage drivers

| symbol | name | units | definition |
|---|---|---|---|
| $\eta$ | **stress triaxiality** | $[-]$ | $\sigma_m/\sigma_{\text{eq}}$. **$\eta<0$ ⇒ voids close.** Clamped to ±3 in code |
| $\mu_L$ | Lode parameter | $[-]$ | deviatoric *shape*: $-1$ axisym. tension, $0$ pure shear, $+1$ axisym. compression |
| $\Phi$, $f$ | void volume fraction | $[-]$ | porosity. $\Phi_0 \approx 10^{-4}$ assumed seed |
| $C$ | Cockcroft–Latham damage | Pa | $\int\langle\sigma_1\rangle\,d\bar\varepsilon$ — dimensional, needs calibration |
| $\langle x\rangle$ | Macaulay bracket | — | $\max(x,0)$ — switches off the compressive branch |

### Rheology (Norton — what the FOM used)

| symbol | name | units | value / source |
|---|---|---|---|
| VISCO$(T)$ | Norton prefactor | Pa·sᵐ | tabulated in `<case>.mat`, 54 points. 1.13e8 → 6.5e6 over 25–675 °C |
| EXPVI$(T)$, $m$ | rate-sensitivity exponent | $[-]$ | 0.0233 → 0.197. $m\to0$ perfectly plastic, $m\to1$ Newtonian |
| $\mu_{\text{eff}}$ | effective viscosity | Pa·s | $\sigma_{\text{eq}}/(3\dot\varepsilon_{\text{eff}})$ — **not a constant**, shear-thinning |
| $T$ | temperature | °C | FOM field `Temperature`, 21–509 °C observed |
| $Z$ | Zener–Hollomon | s⁻¹ | $\dot\varepsilon\exp(Q/RT)$ — for Sellars–Tegart only |
| $Q$ | activation energy | J/mol | ~150 kJ/mol for Al |
| $R$ | gas constant | J/(mol·K) | 8.314 |

### Process parameters

| symbol | name | units | range in the cohort |
|---|---|---|---|
| $\omega$ | tool rotation rate | rpm | 400–800 (FOM cohort); 996–1000 (PinShapes). **Negative ⇒ clockwise** |
| $v_{\text{adv}}$ | traverse (welding) speed | mm/s | 5–10 (cohort); 10 = 600 mm/min (PinShapes, constant) |
| $R_{\text{tool}}$ | tool / shoulder radius | mm | 7.0 (PinShapes), 9.0 (cylA) |
| weld pitch | advance per revolution | mm/rev | $v_{\text{adv}}/(\omega/60)$ |
| $\delta$ | shear-layer thickness | mm | measured; edge at $r/R \approx 0.47$ |
| tilt | tool tilt angle | ° | 0 (A,B,C), 2 (D family) |

### Dimensionless groups

| symbol | definition | value here | meaning |
|---|---|---|---|
| $\mathrm{Re}$ | $\rho U L/\mu_{\text{eff}}$ | $10^{-7}$–$10^{-4}$ | **Stokes creeping flow.** 7–12 orders below turbulent transition ⇒ turbulence analogy dead |
| $\sigma_{\text{cav}}$ | $(p-p_v)/\tfrac12\rho U^2$ | $\sim 7\times10^{4}$ | inception needs $\lesssim 1$. The **absolute** gap to $p_v$ is ~11 orders ⇒ **cavitation impossible**, and the sign is wrong anyway (§5) |
| $\mathrm{Pe}$ | $UL/\alpha$ | 1 (traverse) – 43 (rotation) | advection vs conduction of heat; $\alpha_{\text{Al}}\approx6.5\times10^{-5}$ m²/s, $L=R_{\text{tool}}$ |

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **Advancing side (AS)** | Side where tool rotation and traverse velocities add. Defects appear here. |
| **Retreating side (RS)** | Side where they oppose. |
| **Stir zone / nugget** | Fully recrystallised region that passed through the shear layer. |
| **TMAZ** | Thermo-mechanically affected zone: deformed but not recrystallised. |
| **HAZ** | Heat-affected zone: thermal cycle only, no deformation. |
| **Shear layer** | Thin region around the tool where the velocity gradient is concentrated; the material-delivery conduit. |
| **Flow stress** $\sigma_{\text{flow}}$ | Stress at which the material yields and flows plastically. |
| **Effective viscosity** $\mu_{\text{eff}}$ | $\sigma_{\text{flow}}/3\dot\varepsilon$ — a fluid-mechanics restatement of plasticity. |
| **Zener–Hollomon** $Z$ | $\dot\varepsilon\exp(Q/RT)$ — combines strain rate and temperature into one variable. |
| **Velocity gradient** $L_{ij}$ | $\partial u_i/\partial x_j$. Splits into $D$ (deformation) + $W$ (spin). |
| **Rate of deformation** $D_{ij}$ | Symmetric part of $L$. The part that actually deforms and damages. |
| **Spin tensor** $W_{ij}$ | Antisymmetric part of $L$. Rigid rotation; does no work, causes no damage. Vorticity is its dual. |
| **Equivalent strain rate** $\dot\varepsilon_{\text{eff}}$ | $\sqrt{\tfrac23 D_{ij}D_{ij}}$. Scalar deformation rate. |
| **Accumulated strain** $\bar\varepsilon$ | $\int \dot\varepsilon_{\text{eff}}dt$ along a pathline. History variable. |
| **Deviatoric stress** $s_{ij}$ | Traceless part of $\sigma$. Drives plastic flow. |
| **Mean stress** $\sigma_m$ | $\tfrac13\sigma_{kk} = -p$. Changes volume, not shape. |
| **Equivalent stress** $\sigma_{\text{eq}}$ | $\sqrt{\tfrac32 s_{ij}s_{ij}}$. Compare against flow stress to test yielding. |
| **Stress triaxiality** $\eta$ | $\sigma_m/\sigma_{\text{eq}}$. Negative → voids close. |
| **Rice–Tracey** | Void growth law, $\dot R/R \propto \dot\varepsilon\exp(3\eta/2)$. No free parameters. |
| **Cockcroft–Latham** | Damage indicator $\int\langle\sigma_1\rangle/\sigma_{\text{eq}}\,d\bar\varepsilon$. One calibrated constant. |
| **Macaulay bracket** $\langle x\rangle$ | $\max(x,0)$. Switches off the compressive branch. |
| **Lode parameter** $\mu_L$ | Shape of the deviatoric stress state; distinguishes shear from axisymmetric. |
| **Void volume fraction** $f$ | Porosity fraction; the state variable in GTN. |
| **GTN** | Gurson–Tvergaard–Needleman porous-plasticity damage model. |
| **Weld pitch** | $v_{\text{adv}}/(\omega/2\pi)$, mm per revolution. |
| **Wormhole / tunnel defect** | Continuous subsurface void running along the weld. |
| **Flash** | Excess material extruded out from under the shoulder. Surface defect. |
| **Joint line remnant / kissing bond** | Original oxide layer not sufficiently disrupted. Different mechanism (tracer transport, not filling). |
| **CEL** | Coupled Eulerian–Lagrangian: FE method where material flows through a fixed mesh. |
| **VOF** | Volume of Fluid: tracks a material/void boundary by cell fill fraction. Here the "second phase" is void, not gas. |
| **Stop-action welding** | Abruptly halting the tool mid-weld and sectioning, to freeze the flow state. |
| **Einstein summation** | A subscript repeated twice in one term is summed 1..3. $A_{ii}$ = trace; $A_{ij}A_{ij}$ = sum of squared entries. |
| **Kronecker delta** $\delta_{ij}$ | 1 if $i=j$, else 0. The identity tensor. |
| **Invariant** | Scalar unchanged by rotating the coordinate axes (trace, magnitude, eigenvalues). Every physically meaningful scalar here is one. |
| **Material derivative** $D/Dt$ | $\partial/\partial t + u_j\partial/\partial x_j$. Rate of change following a particle. **Vanishes to $d/dt$ along a pathline** — the reason damage is tracked, not solved on a grid. |
| **Eulerian** | Description at fixed points in space (how the FOM stores fields). |
| **Lagrangian** | Description following material particles (how damage accumulates). |
| **Pathline** | Trajectory of one particle over time. **What damage integrates along.** |
| **Streamline** | Curve tangent to $\mathbf{u}$ at one instant. Equals a pathline only for steady fields. |
| **Isochoric** | Volume-preserving. Plastic flow is isochoric, hence $D_{kk}=0$. |
| **Rigid-viscoplastic** | Elasticity neglected entirely. Valid here because plastic strains are $10^3$–$10^4$× the elastic range. |
| **Yield surface** | Locus $\sigma_{\text{eq}}=\sigma_{\text{flow}}$ in principal-stress space. A **cylinder** about the hydrostatic axis — so pressure alone never yields metal (Bridgman). |
| **$J_2$ flow theory** | von Mises plasticity, named for the second deviatoric invariant $J_2=\tfrac12 s_{ij}s_{ij}$. |
| **Norton–Hoff law** | $\sigma_{\text{eq}} = \text{VISCO}(T)\,\dot\varepsilon^{\text{EXPVI}(T)}$. **The law the FOM actually integrated**; tables in each `.mat`. |
| **Rate sensitivity** $m$ | The Norton exponent. Small $m$ ⇒ flow stress nearly independent of strain rate. 0.023 (cold) to 0.197 (hot) for Al 6063. |
| **Shear-thinning** | Effective viscosity falls as strain rate rises. Here $\mu_{\text{eff}}\propto\dot\varepsilon^{m-1}$ with $m-1\approx-0.9$. |
| **Nucleation vs growth** | These models describe **growth only**; $\Phi=0$ is a fixed point, so an initial porosity $\Phi_0$ must be assumed. |

---

## 12. If you want to read three things

1. **Arbegast (2008)**, *Scripta Mater.* — four pages, no heavy maths, the flow-partitioning
   / volume-bookkeeping argument in its original form. Best single thing to read for intuition.
   `10.1016/j.scriptamat.2007.10.031`

2. **He, Dawson & Boyce (2008)**, *JEMT* — the streamline-integrated damage method.
   This is the one whose equations you'd actually implement.
   `10.1115/1.2840963`

3. **Saby, Bouchard & Bernacki (2015)**, *JMP* — review of void closure criteria. Gives you
   the menu of damage models and their trade-offs, without having to read each original.
   `10.1016/j.jmapro.2014.05.006`

All three are in the Zotero collection (*FSW › FSW Void & Wormhole Prediction*).
