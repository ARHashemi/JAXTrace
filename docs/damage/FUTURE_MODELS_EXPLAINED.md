# The two candidate models, explained from scratch

### Lee–Dawson and GTN — what they are, what every symbol means, and where each number comes from

*Written for myself. Assumes only what is in `PRIMER_fsw_physics_for_me.md` §7
(stress, strain rate, triaxiality). Everything about the He paper is transcribed
from the PDF (`Downloaded/FSW/he2008.pdf`), not recalled.*

---

## 0. First, the naming question

**Why "Lee–Dawson" and not "He"?**

Because He did not invent the model. From the paper itself, §2:

> *"In this study, a damage evolution model developed by **Lee and Dawson** …"*

and reference **[38]**:

> Lee, Y. S., and Dawson, P. R., 1993, *"Modeling Ductile Void Growth in Viscoplastic
> Materials: Parts I and II,"* **Mech. Mater., 15, pp. 21–52.**

So:

| | |
|---|---|
| **Lee & Dawson (1993)** | invented the void-growth model, for viscoplastic materials generally |
| **He, Dawson & Boyce (2008)** | **applied** it to FSW, by integrating it along streamlines of a steady flow solution |

Dawson is on both — he is the common author, which is why the FSW paper uses his
earlier model. When we say "the published FSW precedent", the precedent is **He's
method** (streamline integration) using **Lee & Dawson's model** (the equation).

⚠️ Saying "the He model" would be wrong; saying "the Lee–Dawson model applied by He
et al." is right.

---

## 1. What problem do these models solve that ours does not?

Recall what we run now:

| our law | what it tracks | what it misses |
|---|---|---|
| Rice–Tracey | $\ln(\Phi/\Phi_0)$ — a **growth factor** | no nucleation, no saturation, no closure, needs $\Phi_0$ |
| Cockcroft–Latham | $C$ — **normalised work** | not a physical quantity at all |

Both are **one-way**: the flow drives damage, damage never affects anything. Both
need an external constant we do not have ($\Phi_0$, or $C_\text{crit}$).

The two candidates each fix part of that:

- **Lee–Dawson** adds a **second state variable** (material strength) that evolves as
  the material works, so the void growth rate responds to how hardened the material
  has become. ⚠️ In its published form it is **two-way**: void growth changes the
  material's volume and feeds back into the flow solve (§2.6). Still no nucleation.
- **GTN** makes porosity a **genuine state variable inside the constitutive law**, so
  it has nucleation, growth *and* closure, and the material softens as it damages.
  Two-way. Much more expensive.

---

## 2. Lee–Dawson, term by term

### 2.1 The equation

From He et al. Eq. (1), transcribed:

$$\dot{\Phi} \;=\; \psi_0\,\bar{D}\,\Phi\,\frac{\exp\!\left(c_1\,\sigma_m/\kappa\right)}{1-\Phi}$$

Read it left to right:

| symbol | name | what it *is*, physically |
|---|---|---|
| $\dot\Phi$ | void growth rate | how fast porosity increases, per second |
| $\Phi$ | **void volume fraction** | the porosity itself: (volume of holes)/(total volume). $\Phi=0.0001$ means 0.01 % holes |
| $\psi_0$ | rate/temperature prefactor | a single number bundling material and temperature effects — §2.3 |
| $\bar D$ | **effective deformation rate** | how fast the material is being deformed. **This is our $\dot\varepsilon_\text{eff}$** — same definition, $\sqrt{\tfrac23\mathbf{D}':\mathbf{D}'}$ |
| $\sigma_m$ | **mean stress** | $\text{tr}(\boldsymbol\sigma)/3$. Positive = being pulled apart, negative = squeezed. **Same as ours** |
| $\kappa$ | **strength (hardness)** | how strong the material currently is. ⚠️ **A state variable with its own ODE** — §2.4 |
| $c_1$ | material constant | sets how strongly tension accelerates growth. He et al.: **1.5** |

### 2.2 Why each piece is there — the intuition

**$\Phi$ appears on the right.** Growth is *proportional to how much porosity already
exists*. This makes it **multiplicative**: a void that is already big grows faster.
⚠️ Consequence: **if $\Phi=0$ it stays 0 forever** — no void is ever born. Same
limitation as Rice–Tracey, and the reason $\Phi_0$ must be supplied.

**$\bar D$ multiplies everything.** No deformation, no damage. Material sitting still
accumulates nothing however stressed it is.

**$\exp(c_1\sigma_m/\kappa)$ — the driving term.** Tension ($\sigma_m>0$) multiplies
the rate; compression divides it. The ratio $\sigma_m/\kappa$ is *dimensionless*,
which is what lets a single $c_1$ work across temperatures.

⚠️ **This is NOT triaxiality.** Our $\eta=\sigma_m/\sigma_\text{eq}$ divides by the
*current equivalent stress*; Lee–Dawson divides by the *strength* $\kappa$. They are
siblings, not the same number. Every figure in the He paper plots $\sigma_m/\kappa$.

**$1/(1-\Phi)$ — the saturation term.** As $\Phi\to1$ (all holes) the rate blows up,
driving failure. This is **what Rice–Tracey lacks** and why our $\ln(\Phi/\Phi_0)$
runs to 5242 with no sense of "failed".

### 2.3 The prefactor $\psi_0$ — He et al. Eq. (2)

$$\psi_0=\frac{f_0\,(\kappa_0/G)^m\,\exp\!\left[c_2-Q/(R\theta_r)\right]}{\bar D_r}$$

| symbol | meaning | how you get it |
|---|---|---|
| $f_0$ | reference rate constant | fitted; He et al. report **8.03×10²⁶ s⁻¹** |
| $\kappa_0$ | initial strength | from a stress–strain test at the reference condition |
| $G$ | shear modulus | handbook value for the alloy (~26 GPa for Al) |
| $m$ | strain-rate sensitivity | **we already have this** — it is EXPVI in our `.mat` tables |
| $c_2$ | material constant | fitted; He et al.: **96.0** |
| $Q$ | activation energy | from hot-compression tests at several temperatures |
| $R$ | gas constant | 8.314 J/mol·K — universal |
| $\theta_r$ | reference temperature | chosen; He et al.: **373 K** |
| $\bar D_r$ | reference deformation rate | chosen; He et al.: **1.0 s⁻¹** |

**$\psi_0$ is a constant once the material is fixed** — it does not change along a
streamline. So in practice you compute it once and treat it as one number.

### 2.4 The second ODE — strength evolution, He et al. Eq. (5)

$$\dot\kappa=h_0\,\bar D\left(1-\frac{\kappa}{\kappa^\text{sat}}\right)^{n_0},
\qquad \kappa^\text{sat}=(C/\varphi)^{m_0},\qquad \varphi=\theta\ln(D_0/\bar D)$$

**What this says in words:** the material **hardens as it deforms** ($\dot\kappa>0$
while $\bar D>0$), but hardening **saturates** — as $\kappa$ approaches a ceiling
$\kappa^\text{sat}$ the bracket goes to zero and hardening stops. The ceiling itself
depends on temperature and deformation rate through $\varphi$: hotter or slower means
a lower ceiling, because recovery has time to act.

| symbol | meaning | how you get it |
|---|---|---|
| $h_0$ | hardening modulus | from the slope of a stress–strain curve |
| $n_0$ | hardening exponent | fitted to the same curve's shape |
| $\kappa^\text{sat}$ | saturation strength | computed, not fitted |
| $C, m_0, D_0$ | Zener–Hollomon-type constants | fitted across hot-compression tests at several $T$ and rate |
| $\theta$ | **temperature** | from the flow solution — we have this |

⚠️ **This is the real cost of Lee–Dawson.** It is a **coupled two-equation system**
integrated along every streamline:

$$\kappa(x_s)=\kappa_0+\int_0^T\dot\kappa\,dt,\qquad
\Phi(x_s)=\Phi_0+\int_0^T\dot\Phi\,dt$$

$\kappa$ feeds into $\dot\Phi$ through $\sigma_m/\kappa$, so you cannot compute one
without the other.

### 2.5 ⚠️ It cannot represent healing

From the paper, §2, verbatim:

> *"Because the closure of voids is energetically more difficult than opening them,
> when the mean stress is hydrostatic compression ($\sigma_m \le 0$), the void growth
> rate is assumed to be zero, but never negative."*

$$\dot\Phi=\begin{cases}\psi_0\bar D\Phi\dfrac{\exp(c_1\sigma_m/\kappa)}{1-\Phi} & \sigma_m>0\\[2ex] 0 & \sigma_m\le0\end{cases}$$

So porosity **freezes** under compression rather than closing. This is a deliberate
modelling choice (following Murakami & Ohno 1981), and a conservative one. It means
Lee–Dawson **cannot model the consolidation** that makes FSW work.

### 2.6 ⚠️ The piece I had missed: their flow is NOT incompressible

This matters more than it looks, and it is the thing to understand before assuming
Lee–Dawson drops into our pipeline.

**Their Eq. (10) — apparent density:**

$$\rho=\rho_0(1-\Phi)$$

If the solid matrix has true density $\rho_0$ and a fraction $\Phi$ of the volume is
holes, the *apparent* density of the porous material is lower. Obvious, but it is what
makes the next equation non-trivial.

**Their Eq. (11) — mass conservation, and the key line in the whole paper:**

$$\dot\rho+\rho\nabla\!\cdot\!\mathbf{u}=0
\qquad\Longleftrightarrow\qquad
\boxed{\ \operatorname{tr}(\mathbf{D})=\frac{\dot\Phi}{1-\Phi}\ }$$

Read the right-hand box carefully. $\operatorname{tr}(\mathbf{D})=\nabla\!\cdot\!\mathbf{u}$
is the **volumetric expansion rate**. The paper states, in §3.1:

> *"The material is assumed to be **microscopically** incompressible, and the volume
> change in the workpiece is due to **the growth of voids alone**."*

So:

| | |
|---|---|
| the **matrix** (the solid metal) | incompressible |
| the **material as a continuum** | **NOT** incompressible — it expands exactly as fast as its voids grow |

⚠️ **This is a two-way coupling, and I previously described Lee–Dawson as one-way.
That was wrong.** Void growth feeds back into the velocity field through the mass
balance. It is not a pure post-process in the original formulation.

### 2.7 The penalty formulation — how they impose that constraint

A velocity field cannot be solved freely and *also* be required to satisfy
$\operatorname{tr}(\mathbf{D})=\dot\Phi/(1-\Phi)$ — that is a constraint. The standard
trick is a **penalty method**: instead of enforcing the constraint exactly, add a term
that becomes enormously expensive when it is violated.

**Their Eq. (13) — equilibrium in weak (Galerkin) form:**

$$\int_V\boldsymbol\sigma'\!\cdot\!\nabla w\,dV-\int_S w\,\mathbf{t}\,dS
-\int_V w\,\mathbf{b}\,dV-\int_V p\,\nabla w\,dV=0$$

where $\boldsymbol\sigma'$ is deviatoric stress, $p$ pressure, $w$ a vector weighting
function, $\mathbf{t}$ traction, $\mathbf{b}$ body force. This is the ordinary
finite-element statement of $\nabla\!\cdot\!\boldsymbol\sigma+\mathbf{b}=0$ (their
Eq. 12), with inertia neglected.

**Their Eq. (14) — the penalty equation:**

$$\int_V\alpha\left\{p+\Lambda\left[\operatorname{tr}(\mathbf{D})-\frac{\dot\Phi}{1-\Phi}\right]\right\}dV=0$$

| symbol | meaning |
|---|---|
| $\alpha$ | scalar weighting function |
| $\Lambda$ | **the penalty parameter** |
| the bracket | **the constraint violation** — zero if mass is conserved |

**How it works.** $\Lambda$ is chosen very large. If the bracket is non-zero the term
dominates the equation, so the solver is forced to drive it towards zero. In the limit
$\Lambda\to\infty$ the constraint is satisfied exactly; in practice $\Lambda$ is
large but finite, trading a small violation for a well-conditioned system.

⚠️ Note what the bracket contains: **pressure is tied to void growth**. Setting
$\dot\Phi=0$ recovers the familiar incompressible penalty form
$p=-\Lambda\operatorname{tr}(\mathbf{D})$.

**Their Eq. (15) — the assembled system:**

$$\left[\mathbf{K}_u+\Lambda\mathbf{G}^{\!\top}\mathbf{M}_p^{-1}\mathbf{G}\right]\{\mathbf{u}\}
=\{\mathbf{F}\}+\mathbf{G}^{\!\top}\mathbf{M}_p^{-1}\{\mathbf{F}_\Phi\}$$

$\{\mathbf{u}\}$ is the nodal velocity vector; $\mathbf{K}_u,\mathbf{G},\mathbf{M}_p,\{\mathbf{F}\}$
are standard FE matrices of shape functions and their gradients. **The last term is
the void feedback**, their Eq. (16):

$$\{\mathbf{F}_\Phi\}=\Lambda\int_V\{\mathbf{N}_p\}\frac{\dot\Phi}{1-\Phi}\,dV$$

with $\{\mathbf{N}_p\}$ the pressure shape functions. **Porosity enters the
right-hand side of the momentum solve** — that is the coupling, explicitly.

### 2.8 Heat transfer — closing the loop

The three fields are solved together because each depends on the others.

**Energy balance, their Eq. (17):** $\rho\dot e+\nabla\!\cdot\!\mathbf{q}-\dot\Gamma=0$

with internal energy $e=C_p(\theta-\theta_r)$ (Eq. 18), Fourier's law
$\mathbf{q}=-k\nabla\theta$ (Eq. 19), and — importantly —

$$\dot\Gamma=\operatorname{tr}(\mathbf{D}\boldsymbol\sigma^{\!\top})=\bar\sigma\bar D
\qquad\text{(Eq. 20)}$$

**All the heat comes from viscous dissipation of the deformation itself.** Assembled
as $[\mathbf{H}]\{\boldsymbol\theta\}+\{\mathbf{B}\}=0$ (Eq. 21), with convective
boundaries $\mathbf{q}\!\cdot\!\mathbf{n}=h_c(\theta-\theta_\infty)$ (Eq. 22).

**Their constitutive closure, Eq. (7):**

$$\boldsymbol\sigma'=\frac{2\bar\sigma(1-\Phi)\mathbf{D}'}{3\bar D}$$

⚠️ Note the $(1-\Phi)$: **a porous material carries less deviatoric stress.** Damage
softens the material in the stress law too.

### 2.9 How the streamlines are actually extracted

This is the part the paper says least about, and the honest answer is that it does not
give a numerical recipe. What it does say, §2 and §3.3:

> *"The evolution of strength and voids is evaluated by integration **along the
> streamlines, which track the material through the Eulerian domain** [45,46]. **The
> velocity field determines streamlines.** For a point $x_s$ corresponding to time $T$
> on a streamline, the values of the state variables are given by"* — Eqs. (8), (9):

$$\kappa(x_s)=\kappa_0+\int_0^T\dot\kappa\,dt,\qquad
\Phi(x_s)=\Phi_0+\int_0^T\dot\Phi\,dt$$

And in §3.3, on the iteration:

> *"At each iteration, the velocities were first solved and the values were used to
> calculate the temperatures. Once these two solutions were converged, the state
> variables (strength and porosity) were evaluated **using a streamline technique**."*

**So the extraction method itself is deferred to refs [45, 46]** — Dawson (1984) and
Dawson (1987) — which we have not read. What we *can* state:

| | |
|---|---|
| the field is **steady**, so streamlines = pathlines = particle paths | |
| they integrate **forward in time** from a seed, exactly as we do | |
| $T$ is the **time along the streamline**, so each point carries its own history | |
| the whole thing sits **inside an iteration loop**, not after it | |

⚠️ **One numerical detail they do give, and it is worth copying:**

> *"When evaluating void growth, the mean stress ($\sigma_m$) at each nodal point is
> **averaged based on the values from all the elements that contain this nodal
> point**. This is intended to **avoid numerical oscillations** in void growth
> calculations."*

That is element-to-node averaging of $\sigma_m$ before sampling — the same operation
our `fields.py` performs with `elements_to_nodes`. Independent confirmation that the
step is necessary, not incidental.

**Solved in parallel with PETSc** [47], on 8000 20-node brick elements with 27-point
quadrature.

### 2.10 What He et al. actually did

1. Solve FSW flow **once**: steady-state, Eulerian FE, 3D → velocity, temperature,
   stress on a fixed mesh (8000 20-node bricks, 304L stainless).
2. Extract **streamlines** from that field.
3. Integrate the **two coupled ODEs** along each streamline.

⚠️ **Similar to what we do, but NOT identical — and §2.6–2.7 is why.** We integrate
along pathlines through a field that is computed **once and never revisited**. He et
al. integrate inside an **iteration loop**: velocities → temperatures → state
variables → back into the momentum equation through $\{\mathbf{F}_\Phi\}$, repeat
until converged.

Adopting Lee–Dawson therefore splits into two very different options:

| option | what it is | effort |
|---|---|---|
| **(a) one-way** | integrate $\kappa$ and $\Phi$ along our existing pathlines, ignore the feedback | **small** — one extra accumulator |
| **(b) as published** | couple $\dot\Phi$ back into the FOM's mass balance | **large** — requires changing the flow solver |

⚠️ **Option (a) is not the published model.** It would be "Lee–Dawson's rate equation,
post-processed" — defensible, but we should not claim to have reproduced He et al.'s
method. Their result depends on the coupling.

**Their parameters, Table 3 (304L stainless), read from the PDF:**

| parameter | value |
|---|---|
| Initial porosity $\Phi_0$ | **0.01 %** |
| Reference temperature $\theta_r$ | **373 K** |
| Reference deformation rate $\bar D_r$ | **1.0 s⁻¹** |
| $c_1$ | **1.5** |
| $c_2$ | **96.0** |
| $f_0$ | **8.03×10²⁶ s⁻¹** |

⚠️ These are **for 304L stainless steel**, not aluminium. They are not transferable.

**A useful coincidence:** $c_1 = 1.5$ makes their exponential $\exp(1.5\sigma_m/\kappa)$
identical in *form* to Rice–Tracey's $\exp(3\eta/2)$. Different denominator, same
shape — a sanity anchor, not the same model.

**And an independent confirmation of our own result** — from their §4, verbatim:

> *"the mean stress is mainly tensile ($\sigma_m/\kappa > 0$) [behind the pin], while
> on the leading side, it is essentially compressive … the metal upstream of the pin
> is forged (compressed) by the flow of material toward the pin, while the material
> downstream of the tool is separated from it."*

That is exactly the sign pattern we measured (compressive ahead, tensile in the wake,
9 of 10 cases).

---

## 3. GTN, term by term

**GTN = Gurson–Tvergaard–Needleman.** Gurson (1977) derived the original porous
yield surface; Tvergaard (1981) added fitting constants $q_1,q_2,q_3$ to match
numerical experiments; Needleman & Tvergaard (1984) added the coalescence treatment.
Three names, three contributions.

### 3.1 What makes GTN different in kind

Lee–Dawson and Rice–Tracey are **post-processors**: compute the flow, then integrate
a damage variable along it. The damage never affects the flow.

**GTN is a constitutive law.** Porosity $f$ enters the **yield criterion** itself, so
a damaged element is genuinely weaker and the surrounding material redistributes load
around it. That feedback is the whole point — and the reason it cannot be a
post-process.

### 3.2 The yield surface

$$\Phi_\text{GTN}=\left(\frac{\sigma_\text{eq}}{\sigma_y}\right)^2
+2q_1f^*\cosh\!\left(\frac{3q_2\sigma_m}{2\sigma_y}\right)-\left(1+q_3f^{*2}\right)=0$$

| symbol | meaning |
|---|---|
| $\sigma_\text{eq}$ | von Mises equivalent stress — **we have this** |
| $\sigma_y$ | yield strength of the *matrix* (the solid part, not the porous whole) |
| $\sigma_m$ | mean stress — **we have this** |
| $f^*$ | **effective** porosity (= $f$ until coalescence, then accelerated) |
| $q_1,q_2,q_3$ | Tvergaard constants, typically 1.5, 1.0, $q_1^2$ |

⚠️ Set $f=0$ and it reduces to $\sigma_\text{eq}=\sigma_y$, ordinary von Mises
plasticity. **GTN is von Mises plus a porosity term** — that is the cleanest way to
see it.

The $\cosh(\sigma_m/\sigma_y)$ term is what makes it pressure-sensitive: an ordinary
metal yields the same in tension and compression, a porous one does not.

### 3.3 Porosity evolution

$$\dot f=\dot f_\text{growth}+\dot f_\text{nucleation}$$

**Growth — pure mass conservation:**

$$\dot f_\text{growth}=(1-f)\,\dot\varepsilon^p_{kk}$$

If the solid matrix is incompressible, any change in total volume must be the voids
opening or closing. ⚠️ **This term carries the sign naturally**: under compression
$\dot\varepsilon^p_{kk}<0$ and $f$ genuinely **decreases**. **This is the closure
physics that both our laws and Lee–Dawson lack.**

**Nucleation — new voids appearing:**

$$\dot f_\text{nucleation}=\frac{f_N}{s_N\sqrt{2\pi}}
\exp\!\left[-\tfrac12\left(\frac{\bar\varepsilon-\varepsilon_N}{s_N}\right)^2\right]\dot{\bar\varepsilon}$$

A Gaussian in accumulated strain: voids are born at inclusions and second-phase
particles, most of them around a characteristic strain $\varepsilon_N$. **This is what
lets GTN start from $f=0$** — no $\Phi_0$ needed.

### 3.4 Coalescence

$$f^*=\begin{cases} f & f\le f_c\\[1ex]
f_c+\dfrac{1/q_1-f_c}{f_F-f_c}(f-f_c) & f>f_c\end{cases}$$

Below $f_c$ voids grow independently. Above it they start to merge, and the
*effective* porosity rises faster than the real one, accelerating to failure at $f_F$.

### 3.5 The parameters — and the honest cost

| symbol | meaning | typical (Al) | how to get it |
|---|---|---|---|
| $f_0$ | initial porosity | $10^{-4}$–$10^{-3}$ | metallography / density |
| $f_N$ | nucleating particle fraction | 0.02–0.04 | metallography |
| $\varepsilon_N$ | mean nucleation strain | 0.1–0.3 | **fitted** to notched-bar tests |
| $s_N$ | spread of nucleation strain | 0.1 | **fitted** |
| $q_1,q_2,q_3$ | Tvergaard constants | 1.5, 1.0, 2.25 | literature defaults |
| $f_c$ | coalescence onset | 0.05–0.15 | **fitted**, or from unit-cell simulation |
| $f_F$ | failure porosity | 0.15–0.25 | **fitted** |
| $\sigma_y(\bar\varepsilon,\dot{\bar\varepsilon},T)$ | matrix flow curve | — | hot-compression tests |

⚠️ **Seven-plus constants, several only obtainable by fitting to tests we do not
have.**

⚠️ **And a structural problem for us:** $\dot\varepsilon^p_{kk}$ is the *plastic
volumetric* strain rate. Our flow solution is **incompressible**, so
$\dot\varepsilon_{kk}=0$ **by construction** — the growth term would be identically
zero. Getting a non-zero value requires solving the porous-plasticity constitutive
law, which means **modifying the FOM, not post-processing it**.

---

## 3.6 ⚠️ "But our flow is incompressible — are these models even applicable?"

This is the right question to ask, and the answer is **yes, with one condition**.

### The numbers

He et al.'s own reported result is porosity going from **0.01 % to 0.017 %** across
the whole weld. Put that into their own Eq. (11):

$$\operatorname{tr}(\mathbf{D})=\frac{\dot\Phi}{1-\Phi}$$

| if that change happens over | implied $\operatorname{tr}(\mathbf{D})$ | as a fraction of our measured $\dot\varepsilon_\text{eff}\approx246\ \text{s}^{-1}$ |
|---|---|---|
| 0.1 s | 7.0×10⁻⁴ s⁻¹ | **2.8×10⁻⁶** |
| 1 s | 7.0×10⁻⁵ s⁻¹ | 2.8×10⁻⁷ |
| 10 s | 7.0×10⁻⁶ s⁻¹ | 2.8×10⁻⁸ |

**The volumetric expansion is six to eight orders of magnitude below the deviatoric
deformation rate.** Treating the flow as incompressible is an excellent approximation
*for computing the velocity field*.

### So what is the condition?

The compressibility is negligible **for the flow**, but it is not negligible **for the
damage**, because the damage *is* that expansion. The two statements are compatible:

| quantity | magnitude | is incompressibility OK? |
|---|---|---|
| velocity field | $\operatorname{tr}(\mathbf{D})/\dot\varepsilon\sim10^{-7}$ | ✅ yes, entirely |
| porosity itself | the whole signal | ❌ **it IS the neglected term** |

⚠️ **The condition is therefore: we may use the damage equations on an incompressible
field, but we must not then claim to compute an absolute porosity from
$\operatorname{tr}(\mathbf{D})$.** Our solver sets $\operatorname{tr}(\mathbf{D})=0$ by
construction, so any "porosity" read from the velocity divergence would be exactly
zero — which is why Rice–Tracey and Lee–Dawson are formulated as *rate laws driven by
stress*, not as volume bookkeeping.

### What this means concretely for each model

| model | compatible with an incompressible field? |
|---|---|
| **Rice–Tracey** (ours) | ✅ **Yes, by design.** Its rate depends only on $\dot\varepsilon_\text{eff}$ and $\eta$ — both available from an incompressible field. It never asks for $\operatorname{tr}(\mathbf{D})$. |
| **Cockcroft–Latham** (ours) | ✅ **Yes.** Needs only $\sigma_1$ and $\dot\varepsilon_\text{eff}$. |
| **Lee–Dawson, one-way** | ✅ **Yes.** $\dot\Phi$ depends on $\bar D$, $\sigma_m$, $\kappa$ — all available. We simply do not feed $\dot\Phi$ back. |
| **Lee–Dawson, as published** | ⚠️ **Not without changing the solver.** The feedback *is* the compressibility. |
| **GTN** | ❌ **No.** Its growth term is $(1-f)\dot\varepsilon^p_{kk}$, which our field sets to **exactly zero**. GTN cannot be post-processed onto an incompressible solution at all. |

✅ **The honest summary:** using Rice–Tracey, Cockcroft–Latham, or a one-way
Lee–Dawson on an incompressible field is **not** an inconsistency — those models were
built to be driven by stress and strain rate, not by volume change. **GTN is the one
that genuinely does not fit**, and that is a stronger reason to deprioritise it than
its parameter count.

---

## 3.7 Other indicator families — what else exists

A literature check (Consensus, 2026-10-06) looked for defect indicators that are
**not** void-growth models. ✅ **Everything it returned is already in our own
`FSW_void_prediction_literature_review.md`** — a useful negative result: our coverage
of this area is current. The families, and how each relates to what we have:

### (a) Contact-pressure differential — a single threshold

**Shi et al. (2022)**, *Int. J. Mech. Sci.*: the difference between maximum and
minimum tool–workpiece contact pressure predicts voids, with **Δp < 15 MPa → sound,
Δp > 15 MPa → void**.

✅ **The cheapest thing we could add.** One number per case, computed from the
pressure field we already load. ⚠️ Calibrated for one alloy/tool — treat the *form* as
transferable and the *value* as needing recalibration.

### (b) Machine learning on computed features

**Du et al. (2019)**, *npj Comput. Mater.*: 108 experimental datasets, three Al alloys.

| input features | accuracy |
|---|---|
| raw welding parameters | 83.3 % |
| features from an **analytical** model | 90–93.3 % |
| features from a **rigorous numerical** model | **96.6 %** |

**Temperature and maximum shear stress on the pin dominate.**

⚠️ **This is the result that should give us pause.** Their best accuracy comes from
simple *scalar* features — not from any void-growth integration. It raises a fair
question: would a handful of flow-derived scalars predict our 45 cases as well as a
damage integral does? **We cannot answer it, because we have no experimental labels.**
That is the same blocker as everywhere else.

### (c) Mass balance / flow partitioning

**Arbegast (2008)** — the conceptual ancestor: an "excess material function" and a
"forcing function" partitioning flow into the cavity behind the tool. **Qian et al.
(2013)** — an analytical model balancing material from ahead of the pin to the rear.

✅ **Closest in spirit to the actual physics** (filling failure), and cheap. ⚠️ Gives a
per-case scalar, not a spatial field.

### (d) Free-surface / volume-of-fluid

**Choudhary & Jain (2022)**, **Zhu et al. (2017)**, **Draper et al. (2025)** — CEL with
VOF, predicting tunnel/void/cavity/root defects and their morphology.

✅ **The only family that predicts a defect's actual shape.** ❌ Requires a different
solver — same class of work as GTN.

### (e) Tracer / particle tracking — what we already do

**Dialami, Cervera & Chiumenti (2020)** predict void, wormhole, flash, joint-line
remnant **and onion rings in one simulation**, by advecting tracers through the nodal
velocity field.

⚠️ **Worth knowing: this is our method, published, and it predicts more defect types
than we currently extract.** They get morphology from the tracer distribution itself
rather than from a damage variable. Our density-based void detection is a step in that
direction; their paper is the reference for doing it properly.

### What is genuinely missing from our toolkit

| indicator | have it? | cost to add |
|---|---|---|
| contact-pressure Δp threshold | ❌ | **low** — one scalar from fields we already load |
| flow-partitioning / mass balance | ❌ | low–moderate |
| ML on flow-derived features | ❌ | ⚠️ **blocked: needs experimental labels** |
| residence time / thermal history per particle | ❌ | **low** — one more accumulator |
| Lode-dependent void closure | ❌ | moderate — primer §7.6 already computes the Lode parameter |
| free-surface morphology | ❌ | high — different solver |

✅ **My reading: the contact-pressure criterion and a residence-time accumulator are
the two cheap additions.** Both are per-case scalars computable from what we already
have, and both are *independent* of the damage integral — so agreement between them
would be real corroboration, unlike the two damage laws which share inputs.

---

## 3.8 SPH for FSW defects — what exists, and why it is a different answer

**Yes, SPH for FSW exists and goes back twenty years.** A literature search
(Undermind, 2026-10-06) returned a clear body of work:

| paper | what it did |
|---|---|
| **Tartakovsky et al. (2006)** | the original: *"Modeling of Friction Stir Welding (FSW) process with Smooth Particle Hydrodynamics"* — 40 citations |
| **Bhojwani (2007)** | SPH modelling of the FSW process |
| **Fraser, St-Georges & Kiss (2016)** | *"A Mesh-Free Solid-Mechanics Approach for Simulating the Friction Stir-Welding Process"* — 29 citations, **PDF available** |
| **Ansari & Behnagh (2019)**, *MSMSE* | SPH for the **plunging phase** specifically |
| **Farahbakhsh, Barani Nia & Oterkus (2023)** | *"SPH Simulation of Pin Shape Influence on Material Flow in FSW"* — closest to our tool-geometry question |
| Timesli et al. (2011), Li et al. (2011), Patil et al. (2017) | further SPH FSW work, incl. friction stir spot welding |

⚠️ **But notice what the titles say — and do not say.** They are about *material
flow*, *deformation*, *thermal behaviour* and *pin shape influence*. **None of the
SPH titles claims defect or void prediction.** The papers that do claim it in this
list are CEL/FE, not SPH: Zhu et al. (2017), Das et al. (2021), Choudhary & Jain
(2022), Dialami et al. (2019, 2020).

### Why SPH is a fundamentally different answer to our problem

This is the important conceptual point, and it is worth being clear about.

| | our approach | SPH |
|---|---|---|
| what the particles are | **massless tracers** advected through a precomputed field | **the discretisation itself** — they carry mass and momentum |
| the flow | solved once, Eulerian, then post-processed | solved **by** the particles |
| a void is | a region no tracer reached — an *inference* | a region with no particles — a *direct consequence* of the solve |
| free surface | absent | **naturally represented** |
| cost | post-process, minutes | full solve, hours–days |

✅ **SPH does not need a damage model at all.** Because the particles *are* the
material and carry mass, a region they fail to fill is genuinely unfilled. The void
emerges from mass conservation rather than from an integrated indicator.

⚠️ **That is precisely the capability our method lacks**, and it is worth saying
plainly: our empty regions are an *inference* from tracer density, which is why they
are entangled with seeding, particle count and trapping (§O43–O54 in
`OPEN_QUESTIONS.md`). In SPH they would be a *result*.

### Where that leaves SPH on the shortlist

| | |
|---|---|
| ✅ **Conceptually the right tool** for a filling-failure defect | |
| ✅ **Established for FSW** — two decades of precedent | |
| ⚠️ **But a replacement for the FOM, not an addition to it** — the same class of work as GTN or CEL/VOF | |
| ⚠️ **And the published FSW SPH work is about flow, not defects** — using it for void prediction would be the novel step, not a known one | |

**Fair comparison with the other "big" options:**

| option | replaces the solver? | FSW defect precedent |
|---|---|---|
| Lee–Dawson (one-way) | ❌ no | ✅ He et al. 2008 |
| Lee–Dawson (as published) | ✅ yes | ✅ He et al. 2008 |
| GTN | ✅ yes | ⚠️ only a tensile specimen (Nielsen 2009) |
| **CEL + VOF** | ✅ yes | ✅ **strongest** — Zhu 2017, Das 2021, Choudhary 2022 |
| **SPH** | ✅ yes | ⚠️ flow yes, **defects not demonstrated** |

⚠️ **If the project ever moves to a solver that represents voids directly, the
evidence currently favours CEL+VOF over SPH** — not because SPH is worse in
principle, but because the defect-prediction precedent exists for one and not the
other.

---

## 3.9 Cross-domain survey — how voids are predicted from single-phase flow, generally

Deep literature search (Undermind, 2026-10-06), deliberately **not** restricted to
FSW: casting, additive manufacturing, extrusion, forging, injection moulding.
**244 papers ranked.** Full results:
`app.undermind.ai/projects/ad91d850-93a8-48f2-a729-f94d9279ccdf`

### The headline finding

> *A single-phase flow field can predict defect **susceptibility**, but it cannot by
> itself establish a **material-absence region**. The strongest approaches add a
> history variable, track material/interface transport, or explicitly introduce
> void/free-surface occupancy.*

✅ **This independently validates our framing.** We have been saying "susceptibility
ranking, not defect prediction" — that is exactly the distinction the literature
draws, and the reason is structural, not a shortcoming of our implementation.

### The five families, with their mathematical form

**(a) History variables integrated along streamlines** — what we do

- **He, Dawson & Boyce (2008)** — the FSW case, §2 above.
- **Solid-state bonding criteria**: D'Urso et al. (2011), **Buffa, Pellegrino &
  Fratini (2014)** *"Analytical bonding criteria for joint integrity prediction in
  FSW"*, Cooper & Allwood (2014), Wang et al. (2020, 2022). These integrate
  **pressure and stress over the interface history** rather than tracking porosity.

⚠️ **This family is new to us and directly relevant.** It asks "did this interface
bond?" instead of "did a void grow?" — arguably the better question for a *joining*
process. **Wang et al. (2022)** even derives an FSW **process window** from bonding
mechanics.

**(b) Tracer / transport — the closest match to a "refill" indicator**

- **Dialami et al. (2020)** — tracer-based prediction of multiple defect types.
- **Ghate et al. (2020)** — *"Ductile fracture based joint formation mechanism"*;
  predicts partially unfilled transient cavities, compared against **CT**.
- **Arbegast (2008)** — *"A flow-partitioned deformation zone model for defect
  formation during FSW"*, **309 citations**, the conceptual ancestor.

**(c) Explicit void-capable domains** — CEL / VOF

Represent material occupancy directly and predict defect *shape*. Zhu (2017),
Choudhary & Jain (2022). ⚠️ These **augment the single-phase formulation**; they do
not extract a void from it.

**(d) Cross-domain criteria — the genuinely new material**

| domain | criterion | reported threshold |
|---|---|---|
| **casting** | **Niyama**, $G/\sqrt{\dot T}$ — a local thermal-feeding surrogate | **≈2** for detectable microshrinkage, **≈1** for macroshrinkage (Carlson & Beckermann 2008, high-Ni steel) |
| casting | cold-shut / confluence from **time fields** | Feng et al. (2021) |
| casting | shrinkage porosity from solidification | Khalajzadeh & Beckermann (2020) |
| **LPBF / AM** | **lack-of-fusion** from melt-pool **overlap geometry** | Tang, Pistorius & Beuth (2017), **650 citations**; Mukherjee & DebRoy (2018) |
| **extrusion** | central bursting from **accumulated ductile damage** | McVeigh (2006) |

⚠️ **The Niyama criterion is worth understanding.** It is **pointwise and local** —
no integration along a path — and it is *the* industrial standard for shrinkage
porosity, with published numerical thresholds. The analogue for us would be a local
field combination flagging "this region cannot be fed", which is cheap to compute and
completely independent of our damage integral.

**(e) Machine learning** — Du et al. (2019), 96.6 % on compiled experiments (§3.7).

### ⚠️ Two explicit negative results

From the search summary, verbatim:

> *"The retrieved studies **do not establish a particle-density/continuity deficit
> metric**, nor do they use **LCS or FTLE** criteria as validated defect thresholds."*

**1. Nobody has published a particle-density deficit indicator.** That is what our
void-area measure is. ✅ Genuinely novel — ⚠️ and unvalidated, which is exactly the
state O43–O54 describe. The novelty cuts both ways: no precedent to borrow from, and
no precedent against which to sanity-check.

**2. Lagrangian coherent structures and FTLE have not been used as validated defect
criteria.** I had expected these to be the obvious transferable idea from fluid
mixing — they are not established here. An open opportunity, but a research
direction rather than something to adopt.

### What I would actually take from this

| idea | why | cost |
|---|---|---|
| **Solid-state bonding criterion** (Buffa 2014, Wang 2022) | asks the right question for a *joining* process; integrates pressure over interface history — our pathlines already carry pressure | **low–moderate** |
| **A Niyama-style local criterion** | pointwise, cheap, independent of the damage integral, with published thresholds in its own domain | **low** |
| **Flow-partitioning / mass balance** (Arbegast 2008) | the conceptual ancestor, 309 citations, directly about filling failure | low |

✅ **The bonding-criterion family is the most interesting thing this search found.**
It reframes the problem from "where do voids grow?" to "where does the material fail
to bond?" — which for FSW may be the more physical question, and it uses quantities
we already have.

---

## 3.10 The bonding criteria, properly — Piwnik–Plata and its relatives

This family deserves more than a mention: it is the most directly transferable thing
the survey found, and it has an explicit formula, a calibration procedure, and
cross-process validation.

### 3.10.1 The Piwnik–Plata criterion

**The idea.** For a *joining* process the question is not "did a void grow?" but
**"did the two surfaces bond?"** The criterion integrates the normalised interface
pressure over the time the surfaces are in contact:

$$\boxed{\;w=\int_0^{t_c}\frac{p}{\bar\sigma}\,dt\;}
\qquad\text{bond if}\quad w\ge w_\text{lim}$$

| symbol | meaning | do we have it? |
|---|---|---|
| $p$ | pressure on the interface | ✅ **yes** — our FOM pressure field |
| $\bar\sigma$ | effective (flow) stress | ✅ **yes** — $\sigma_\text{eq}$, §6.1 |
| $t_c$ | contact time | ✅ **yes** — time along the pathline |
| $w_\text{lim}$ | critical value | ❌ **must be calibrated** |

⚠️ **Note how close this is in form to Cockcroft–Latham.** Both integrate a
stress ratio along a path. The difference is *which* ratio: C–L uses
$\langle\sigma_1\rangle/\sigma_\text{eq}$ (tension-driven damage), Piwnik–Plata uses
$p/\bar\sigma$ (compression-driven bonding). **They are near-mirror images**, and we
already compute both ingredients.

### 3.10.2 How $w_\text{lim}$ is obtained — a real procedure

**D'Urso, Longo, Ceretti & Giardini (2011, 2012)** give a coupled
experimental–simulative method, and it is worth knowing because it is cheap:

1. **Flat-roll "sandwiches"** of two rectangular AA6082 specimens, at several heights
   → different compression ratios → different interface pressures.
2. Repeat at several **temperatures**.
3. Run **FEM** of the same rolling tests, with a user routine evaluating $w$.
4. Check **experimentally** whether each sandwich actually bonded.
5. $w_\text{lim}$ = the value separating bonded from unbonded, **as a function of
   temperature**.

✅ **This is a rolling test, not an FSW test** — far cheaper than producing and
sectioning welds. ⚠️ $w_\text{lim}$ depends on **material and temperature**, so it
must be done for our alloy.

### 3.10.3 Cross-process validation

**Fratini, Buffa, Valvo & Pellegrino (2016)** applied Piwnik–Plata to **FSW, linear
friction welding, porthole extrusion and roll bonding**, trained a neural network on
the field variables, and predicted bonding occurrence across all four.

✅ **The criterion transfers between solid-state processes** — strong evidence it is
capturing something general rather than being fitted to one geometry.

### 3.10.4 ⚠️ A serious caveat from the mechanism literature

**Wang, Gao, McDonnell & Feng (2020)**, *Extreme Mechanics Letters*, criticises the
usual assumption. Their finding, from the abstract:

> *"The commonly assumed **sintering-like diffusional-bonding hypothesis** is
> criticized in this work as **not the dominant mechanism**. … the thermomechanical
> history on the workpiece–workpiece interface traverses in the **creep-dominated
> regime** for the growth/shrinkage of interfacial cavities. The evolution of the
> bonding fraction relies **mainly on the creep strain rate** in the adjoining
> workpieces, **weakly on stress triaxiality**, and **negligibly on interfacial
> diffusion**."*

⚠️ **Read that last sentence carefully — it bears directly on our work.** If bonding
depends *mainly on creep strain rate* and only *weakly on stress triaxiality*, then
a triaxiality-driven law like Rice–Tracey may be reading a **second-order** variable
for this process. Our $\dot\varepsilon_\text{eff}$ would be the first-order one.

✅ This is testable with what we already have: correlate our family ranking against
$\int\dot\varepsilon\,dt$ alone versus $\int\dot\varepsilon\,e^{1.5\eta}dt$. If they
rank the cases identically, the exponential is not doing any work.

### 3.10.5 What makes a bond, experimentally

**Cooper & Allwood (2014)**, *JMPT*, 142 citations — a careful decoupled experiment
on solid bonding of aluminium. Their findings:

| factor | effect on weld strength |
|---|---|
| **interface strain** | a **minimum strain is required** for any bonding |
| **normal contact stress** | must exceed the **uniaxial yield stress** for a strong bond |
| temperature ↑ | **reduces** the minimum strain needed |
| shear ↑ | **reduces** the minimum strain needed |
| strain rate ↑ | little effect at low $T$; **reduces** strength at high $T$ |

✅ **Two thresholds we could evaluate directly:** a minimum accumulated interface
strain, and $p>\sigma_y$. Both from fields we already have.

---

## 3.11 The Niyama criterion — a local, pointwise alternative

Worth a section because it is **structurally unlike everything else we do**: no
integration along a path, no history variable, just a local combination of two fields.

### 3.11.1 The classical form

$$\boxed{\;\text{Ny}=\frac{G}{\sqrt{\dot T}}\;}$$

with $G$ the local thermal gradient and $\dot T$ the local cooling rate, evaluated at
the end of solidification. **Low Ny → shrinkage porosity**, because the thermal
conditions cannot feed liquid into the solidifying region.

**Carlson & Beckermann (2008)**, validated on high-Ni steel and Ni-based alloys:

| threshold | meaning |
|---|---|
| $\text{Ny}<2.0\ (^\circ\text{C·s})^{1/2}/\text{mm}$ | **micro**-shrinkage begins to appear |
| $\text{Ny}<1.0\ (^\circ\text{C·s})^{1/2}/\text{mm}$ | **macro**-shrinkage, visible on a radiograph |

### 3.11.2 The dimensionless form — the better idea

⚠️ The classical criterion's weakness is exactly ours: **the threshold is
alloy-specific and requires extensive experimentation.** **Guo et al. (2015)** and
**Kang et al. (2013)** use a *dimensionless* Niyama that

> *"avoids the need to find a threshold Niyama criterion below which shrinkage
> porosity forms — a criterion which can be determined only via extensive
> alloy-dependent experimentation"*

and predicts the **pore volume fraction directly**, validated against radiography and
metallography.

✅ **That is the pattern worth stealing.** Not the formula — our process has no
solidification — but the *idea* of non-dimensionalising a criterion so the threshold
becomes universal instead of per-alloy. It is the same problem our $\Phi_0$ and
$C_\text{crit}$ create.

⚠️ **What a Niyama analogue would be for FSW** is an open question. The casting
criterion compares a *driving* gradient against a *resisting* rate. The FSW analogue
might compare the rate at which the cavity opens (∝ travel speed) against the rate
material can be delivered (∝ rotation speed and flow stress) — which is close to what
**Arbegast's flow-partitioning** model does, and dimensionally natural.

✅ **This is developed further in `NEXT_STEPS_detachment_criterion.md`** (Part B): the
two candidate groups $\Pi_1=\omega r_\text{pin}/v_\text{adv}$ and a thermal-softening
$\Pi_2$, why a detachment rule can produce a *genuine* mass-deficit void rather than a
damage proxy, and how to falsify it.

---

## 4. Side by side

| | Rice–Tracey (ours) | Cockcroft–Latham (ours) | **Lee–Dawson** | **GTN** |
|---|---|---|---|---|
| what it tracks | $\ln(\Phi/\Phi_0)$ | $C$, work | $\Phi$ **and** $\kappa$ | $f$, inside the yield law |
| physical quantity? | yes (porosity) | no | yes | yes |
| nucleation | ❌ | ❌ | ❌ | ✅ |
| saturation / failure | ❌ | via $C_c$ | ✅ $1/(1-\Phi)$ | ✅ $f_c,f_F$ |
| **closure under compression** | ❌ | ❌ | ❌ **clamped to 0** | ✅ **genuine** |
| feedback to the flow | ❌ | ❌ | ✅ **as published** (❌ if post-processed) | ✅ |
| fitted constants | **0** | 1 ($C_c$) | ~10 | 7+ |
| extra state variables | 0 | 0 | **1** ($\kappa$) | $f$ in the solve |
| can post-process our field? | ✅ | ✅ | ⚠️ **only in a reduced, one-way form** | ❌ **needs the FOM** |
| FSW precedent | — | — | ✅ He et al. 2008 | Nielsen 2009 (tensile specimen only) |

---

## 5. What adopting each would actually take

### Lee–Dawson — a realistic next step

**Option (a), one-way — pipeline changes small.** We already integrate along
pathlines. We would:

1. add a second accumulator for $\kappa$ (one more float per particle);
2. sample **temperature** $\theta$ along the path — we already export it;
3. compute $\sigma_m/\kappa$ instead of $\sigma_m/\sigma_\text{eq}$;
4. apply the $\sigma_m\le0$ clamp;
5. average $\sigma_m$ element-to-node first — He et al. do this explicitly to avoid
   numerical oscillations, and our `fields.py` already does the same thing.

⚠️ **Option (b), as published — a different project.** Their Eq. (11) makes the
continuum compressible, $\operatorname{tr}(\mathbf{D})=\dot\Phi/(1-\Phi)$, enforced
by the penalty Eq. (14) with the void term $\{\mathbf{F}_\Phi\}$ on the momentum
right-hand side. Our FOM solves an **incompressible** flow, so reproducing their
method means changing the flow solver — the same class of work as GTN, not an
increment.

**The blocker is parameters, not code.** We would need, for *our* aluminium:
$h_0$, $n_0$, $C$, $m_0$, $D_0$, $Q$, $\kappa_0$, $c_1$, $c_2$, $f_0$, $\Phi_0$.
He's values are for 304L stainless.

⚠️ **Before adopting it, know what it does and does not buy.** It adds saturation and
a strength state variable. It still cannot nucleate, and still cannot close voids —
so it would *not* fix the limitation that matters most for FSW consolidation.

### GTN — a different project

Not a post-process. Porosity must live inside the constitutive solve, which means
changing the FOM, not JAXTrace. Plus 7+ fitted constants.

⚠️ Worth being honest that this is **a research project with a separate funding case**,
not an increment on what we have.

---

## 6. Ranked comparison — every option on the table

Ranked by **what I would actually do next**, not by sophistication. The ranking
weights four things: can it run on the field we already have; does it need data we do
not have; does it test something our current indicators cannot; and is there a
precedent to lean on.

### 6.1 The ranking

| # | option | what it adds | effort | needs calibration? | independent of our damage integral? | FSW precedent |
|---|---|---|---|---|---|---|
| **1** | **Piwnik–Plata bonding criterion** $w=\int p/\bar\sigma\,dt$ | asks the **right question** for a joining process; near-mirror of Cockcroft–Latham | **low** — one accumulator, ingredients already computed | ⚠️ $w_\text{lim}$, but by a **cheap rolling test** (D'Urso) | ⚠️ partly — shares $\sigma_\text{eq}$ | ✅ Buffa 2014, Fratini 2016 |
| **2** | **Strain-rate-only ranking** $\int\dot\varepsilon\,dt$ | ⚠️ **tests whether our exponential does any work** — Wang 2020 says bonding depends mainly on creep rate, weakly on triaxiality | **trivial** — we already store it | ❌ none | ✅ **yes** | ✅ implicit in Wang 2020 |
| **3** | **Contact-pressure differential** $\Delta p$ | a published numerical threshold | **low** | ⚠️ 15 MPa is alloy-specific | ✅ yes | ✅ Shi 2022 |
| **4** | **Residence time** in the shear layer | separates "how long" from "how hard", which our integral conflates | **low** — one accumulator | ❌ none | ✅ yes | ⚠️ indirect |
| **5** | **Cooper–Allwood thresholds** (min. strain; $p>\sigma_y$) | two **physically motivated** binary tests | **low** | ❌ $\sigma_y$ is known | ⚠️ partly | ✅ Cooper 2014 |
| **6** | **Flow partitioning / mass balance** | closest to the **actual** physics (filling failure) | low–moderate | ⚠️ some | ✅ yes | ✅ Arbegast 2008 (309 cit.) |
| **7** | **Lee–Dawson, one-way** | saturation + a strength state variable | moderate — 2nd ODE | ❌ **~10 constants for our alloy** | ❌ no — same family | ✅ He 2008 |
| **8** | **Lode-dependent void closure** | the **only** way to get genuine closure in a post-process | moderate — primer §7.6 has the Lode parameter | ⚠️ yes | ❌ no | ⚠️ forming, not FSW |
| **9** | **A Niyama-style dimensionless criterion** | ⚠️ **removes the per-alloy threshold problem** — the deepest idea here | moderate — the FSW analogue must be **derived** | ✅ that is the point | ✅ yes | ❌ none — novel |
| **10** | **ML on flow-derived features** | 96.6 % in the literature | low **once labels exist** | ❌ **blocked: no experimental labels** | ✅ yes | ✅ Du 2019 |
| **11** | **CEL + VOF** | defect **shape**, directly | **high** — replaces the solver | ⚠️ some | ✅ yes | ✅ **strongest**: Zhu, Das, Choudhary |
| **12** | **SPH** | voids emerge from mass conservation | **high** — replaces the solver | ⚠️ some | ✅ yes | ⚠️ flow only, **defects not shown** |
| **13** | **Lee–Dawson, as published** | the coupling | **high** — compressible solver | ❌ ~10 constants | ❌ no | ✅ He 2008 |
| **14** | **GTN** | nucleation + closure + feedback | **high** | ❌ **7+ constants** | ❌ no | ⚠️ tensile specimen only |

### 6.2 Reading the ranking

**Items 1–6 are all cheap and all do something our current pair cannot.** They are
ranked above the sophisticated models deliberately: each is a few hours of work on
data we already have, and several are **genuinely independent** of the damage
integral — which matters because our two damage laws share inputs, so their
agreement was never the corroboration it looked like (§O45).

⚠️ **Item 2 is the one I would do first**, and it is almost free. Wang et al. (2020)
report that bonding depends *mainly on creep strain rate* and only *weakly on stress
triaxiality*. If ranking the 45 cases by $\int\dot\varepsilon\,dt$ alone reproduces
our B-Flutes-highest result, then the $\exp(1.5\eta)$ factor is contributing nothing
and the whole damage apparatus is an expensive way to measure accumulated strain.
**That is a result either way**, and it costs one afternoon.

**Items 7–10 need something we do not have** — constants, or labels, or a derivation.

**Items 11–14 replace the solver.** Among them the evidence favours **CEL+VOF**: it
is the only one with a demonstrated FSW *defect*-prediction record.

### 6.3 ⚠️ The thing that outranks all of it

| | |
|---|---|
| **0** | **Obtain one experimental cross-section.** |

Every option above produces an unvalidated ranking. A single macrograph or CT scan
would do more than any model upgrade: it would convert the entire exercise from
"susceptibility ranking" to "validated or falsified". ⚠️ **Until then, more model
complexity buys precision, not accuracy** — which is exactly the lesson of He et
al.'s own 0.01 % → 0.017 % result, where the *locations* agreed with experiment and
the *values* depended entirely on fitted constants.

---

## 7. References

**The two models in detail**

1. **Lee, Y. S., and Dawson, P. R.** (1993). *Modeling Ductile Void Growth in
   Viscoplastic Materials: Parts I and II.* **Mech. Mater. 15**, 21–52. — the model.
2. **He, X., Dawson, P. R., and Boyce, D. E.** (2008). *ASME J. Eng. Mater. Technol.*
   **130**, 021006. — the FSW application. ✅ Read from PDF; all equations and Table 3
   values above are transcribed from it.
3. **Murakami, S., and Ohno, N.** (1981). — origin of the no-closure clamp (He ref. [43]).
4. **Gurson, A. L.** (1977). *ASME J. Eng. Mater. Technol.* **99**, 2–15.
5. **Tvergaard, V.** (1981); **Tvergaard, V., and Needleman, A.** (1984). — the TN in GTN.
6. **Nielsen, K. L.** (2009). *Effect of a shear modified Gurson model on damage
   development in a FSW tensile specimen.* — the only FSW-adjacent GTN application
   we have found, and it is on a tensile specimen, not the weld process.

**Bonding criteria (§3.10)**

7. **D'Urso, G., Longo, M., Ceretti, E., and Giardini, C.** (2011). *Coupled
   Simulative-Experimental Procedure for Studying the Solid State Bonding Phenomena.*
   **Key Eng. Mater.**, 181–188. — the $w_\text{lim}$ calibration procedure. Also
   D'Urso et al. (2012) for the alloy/temperature dependence.
8. **Buffa, G., Pellegrino, S., and Fratini, L.** (2014). *Analytical bonding criteria
   for joint integrity prediction in friction stir welding of aluminum alloys.*
   **JMPT**, 2102–2111.
9. **Fratini, L., Buffa, G., Valvo, E., and Pellegrino, S.** (2016). *Comparative
   analysis of bonding mechanism in solid state metal working processes.* — applies
   Piwnik–Plata across FSW, LFW, porthole extrusion and roll bonding.
10. **Wang, X., Gao, Y.-F., McDonnell, M., and Feng, Z.** (2020). *On the
    solid-state-bonding mechanism in friction stir welding.* **Extreme Mech. Lett.**
    100727. — ⚠️ bonding depends mainly on **creep strain rate**, weakly on
    triaxiality. Also Wang et al. (2022), *Materialia*, for the process window.
11. **Cooper, D. R., and Allwood, J.** (2014). *The influence of deformation conditions
    in solid-state aluminium welding processes on the resulting weld strength.*
    **JMPT**, 2576–2592, 142 citations. — minimum strain; $p>\sigma_y$.

**Casting / Niyama (§3.11)**

12. **Carlson, K., and Beckermann, C.** (2008). *Use of the Niyama Criterion to Predict
    Shrinkage-Related Leaks…* — $\text{Ny}_\text{micro}=2.0$, $\text{Ny}_\text{macro}=1.0$.
13. **Guo, J.-Z., Beckermann, C., Carlson, K.,** et al. (2015); **Kang, M.-D.,** et al.
    (2013), *Materials*, 1789–1802. — the **dimensionless** Niyama, which removes the
    per-alloy threshold.

**Other indicator families (§3.7, §3.9)**

14. **Arbegast, W.** (2008). *A flow-partitioned deformation zone model for defect
    formation during friction stir welding.* **Scripta Mater.**, 372–376, **309
    citations.**
15. **Shi, L.,** et al. (2022). **Int. J. Mech. Sci.** — the Δp < 15 MPa criterion.
16. **Du, Y., Mukherjee, T., and DebRoy, T.** (2019). *Conditions for void formation in
    friction stir welding from machine learning.* **npj Comput. Mater.** — 96.6 %.
17. **Dialami, N., Cervera, M., and Chiumenti, M.** (2020). **Eur. J. Mech. A/Solids**,
    128 citations. — tracer-based prediction of void, wormhole, flash, joint-line
    remnant and onion rings.
18. **Ghate, N.,** et al. (2020). **Int. J. Mech. Sci.** 105293. — per-revolution cavity
    filling, validated against CT.
19. **Tang, M., Pistorius, P., and Beuth, J.** (2017). *Prediction of lack-of-fusion
    porosity for powder bed fusion.* **Addit. Manuf.**, 39–48, **650 citations**;
    **Mukherjee & DebRoy** (2018), **JMP**, 235 citations.

**SPH (§3.8)**

20. **Tartakovsky, A.,** et al. (2006) — the original SPH FSW model; **Fraser,
    St-Georges & Kiss** (2016); **Ansari & Behnagh** (2019), **MSMSE**;
    **Farahbakhsh, Barani Nia & Oterkus** (2023).

**CEL / VOF (§3.9)**

21. **Zhu, Z.,** et al. (2017); **Das, D., Bag, S., and Pal, S.** (2021), **STWJ**,
    412–419; **Choudhary, A., and Jain, R.** (2022), **MAMS**, 2371–2384.

---

⚠️ **Status note on sources.** Only **He et al. (2008)** has been read in full, line by
line against the PDF — every equation and parameter attributed to it above is
transcribed, not recalled. Everything else is from **abstracts and search metadata**
(Consensus and Undermind, 2026-10-06); the Piwnik–Plata formula, the Niyama
thresholds and the Cooper–Allwood findings are quoted from abstracts, which state them
explicitly, but the surrounding derivations have not been checked.

⚠️ Items 1, 3, 4, 5 (Lee & Dawson 1993, Murakami & Ohno 1981, Gurson 1977, Tvergaard &
Needleman 1984) are standard textbook forms, cited but not read. **If a decision turns
on any of this, the papers to obtain are Lee & Dawson (1993)** — the model we would
implement — **and Wang et al. (2020)**, whose creep-rate claim would change how we
interpret our own results.
