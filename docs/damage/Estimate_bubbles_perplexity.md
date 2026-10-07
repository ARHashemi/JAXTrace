Yes, it is possible (and common) to *estimate* where bubbles will appear using only a single‑phase turbulent flow simulation plus vortex/pressure analysis, but that is only an approximate, model‑based prediction—not a “true” two‑phase simulation of bubble dynamics.  The rigorous approaches in the literature either (i) couple single‑phase turbulence to a cavitation/entrainment model, or (ii) directly simulate the interface in a two‑phase DNS/LES (VOF, level‑set, front‑tracking) and then analyze vortices and pressure fields. [courses.washington](https://courses.washington.edu/mengr537/Reading_Assignments/MultiphaseFlowHandbook_Crowe_Chapter12.pdf)

***

## Types of bubbles: two main cases

When people say “bubbles in a turbulent liquid”, they usually mean one of two things. [citeseerx.ist.psu](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=2e77f2097e52c6e90d22fa4224472efbf68ab602)

- **Cavitation bubbles in the bulk liquid**: vapor cavities created when the local static pressure drops below the liquid’s vapor pressure, typically inside vortex cores or regions of strong acceleration. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/ceat.202200465)
- **Air bubbles entrained from a free surface**: pockets of air pulled down from the air–water interface by strong free‑surface turbulence, gravity, and surface tension interactions (e.g. breaking waves, jets impacting pools). [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/136590/scale_separation_and_dependence_of_entrainment_bubblesize_distribution_in_freesurface_turbulence.pdf?sequence=2&isAllowed=y)

Both phenomena are tightly linked to turbulent vortical structures, but the physics and modeling strategies are slightly different. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/ceat.202200465)

***

## Why vortices are so important

A vortex is a region where the fluid mostly swirls around an axis, and the centrifugal effect requires a radial pressure gradient to balance it.  In a liquid vortex, the static pressure at the core is lower than in the surrounding fluid, and in strong vortices that core pressure can fall below the vapor pressure, triggering cavitation bubbles (“vortex cavitation”). [scribd](https://www.scribd.com/document/1005656811/Geometry-and-Operational-Physics-of-Vortex-Based-Nanobubble-Generators-A-Review-of-Design-Principles-Optimization-and-Performance-1)

Experiments and CFD of cavitating jets and valves show that cavitation bubbles form predominantly in vortex cores and shear layers, and are then transported downstream by the flow.  Similarly, in breaking‑wave and near‑wall vortical flows, bubbles are preferentially trapped and transported inside vortex cores and between counter‑rotating vortices. [www1.udel](http://www1.udel.edu/kirby/papers/ma-etal-cacr-12-08.pdf)

So: if your simulation can resolve vortices and pressure, those structures are very informative about where bubbles will *tend* to appear and accumulate. [www1.udel](http://www1.udel.edu/kirby/papers/ma-etal-cacr-12-08.pdf)

***

## Cavitation: single‑phase CFD plus bubble inception model

For cavitation in pure liquid (no free surface), a common “correct” approach in engineering is: [arxiv](https://arxiv.org/html/2409.02369v1)

1. **Solve single‑phase Navier–Stokes with turbulence (RANS, LES, or DNS)** for the liquid only, so you get the instantaneous velocity and pressure field. [arxiv](https://arxiv.org/html/2409.02369v1)
2. **Apply a cavitation model**: when local pressure drops below vapor pressure (or some cavitation‑number threshold), treat that region as containing vapor bubbles, typically via a transport equation for a vapor volume fraction (e.g. Zwart–Gerber–Belamri, Schnerr–Sauer) or a simplified equation of state. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/ceat.202200465)
3. **Couple turbulence and cavitation**: LES or DNS is often needed to correctly capture the intermittent low‑pressure regions where cavitation occurs, because time‑averaged RANS misses the extremes. [arxiv](https://arxiv.org/html/2409.02369v1)

For example, LES of cavitation in a control valve showed that cavitation bubbles form behind the narrowest cross‑section in shear‑layer vortices, and that resolving the instantaneous local pressure field is essential for predicting cavitation intensity. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/ceat.202200465)

This is exactly the scenario you described: they simulate *only* the liquid phase, but then use vortex/pressure/turbulence data plus a cavitation closure to infer where bubbles exist and how intense cavitation is.  That’s considered a standard, physically grounded approach—as long as the cavitation model is appropriate and the turbulence is adequately resolved. [courses.washington](https://courses.washington.edu/mengr537/Reading_Assignments/MultiphaseFlowHandbook_Crowe_Chapter12.pdf)

***

## Air entrainment at free surfaces: turbulence vs gravity and surface tension

For air bubbles entrained from a free surface (air–water interface), the literature emphasizes a competition between: [citeseerx.ist.psu](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=2e77f2097e52c6e90d22fa4224472efbf68ab602)

- **Disrupting effect of near‑surface turbulence**, which tends to wrinkle and tear the interface. [arxiv](https://arxiv.org/html/2506.10090v1)
- **Stabilizing effects of gravity and surface tension**, which try to flatten the interface and suppress entrainment of air below. [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/136590/scale_separation_and_dependence_of_entrainment_bubblesize_distribution_in_freesurface_turbulence.pdf?sequence=2&isAllowed=y)

Mechanistic models and DNS of strong free‑surface turbulence define regimes using dissipation rate \( \varepsilon \), Froude number (gravity vs inertia), Weber number (surface tension vs inertia), and Bond number (gravity vs surface tension). [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/155740/2022-01%20Vortex_Entrainment_JFM_new%5B95%5D_resaved.pdf?sequence=1&isAllowed=y)

One study builds an “entrainment regime map” in \( \varepsilon\)–\(r\) space (dissipation vs bubble size) and identifies a critical dissipation rate \( \varepsilon_{\text{cr}} \): above this value, appreciable air entrainment occurs; below it, the interface stays essentially bubble‑free.  Another set of two‑phase DNS of strong free‑surface turbulence shows that high Froude and Weber numbers lead to strongly deformed, wavy surfaces and numerous entrained air bubbles below the surface. [arxiv](https://arxiv.org/html/2506.10090v1)

Again, turbulence structures (vortices) appear near the free surface and help pull air downward, but the *decision* “does a bubble get entrained here?” is based on combined criteria involving turbulence intensity, gravity, and surface tension, not just vorticity alone. [arxiv](https://arxiv.org/html/2506.10090v1)

***

## Vortex‑based entrainment models (without full two‑phase CFD)

There are papers that do almost exactly what you noticed: they analyze vortex and velocity dynamics to infer entrainment volumes without explicitly tracking every bubble as a separate phase. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0301932215301920)

A recent study on a rectilinear vortex pair impinging on a free surface uses DNS to identify a phenomenological model linking entrained air volume to four key vortex parameters: circulation \( \Gamma \), effective radius \( a \), vertical rise velocity \( W \), and gravity \( g \).  They define a circulation‑flux Froude number [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/155740/2022-01%20Vortex_Entrainment_JFM_new%5B95%5D_resaved.pdf?sequence=1&isAllowed=y)

\[
\mathrm{Fr}_\Xi^2 = \frac{|\Gamma| W}{a^2 g}
\]

and show: if this parameter is below a critical value \( \mathrm{Fr}_{\Xi,\text{cr}}^2 \), no air is entrained; if it is above, the entrained volume increases roughly linearly with \( (\mathrm{Fr}_\Xi^2 - \mathrm{Fr}_{\Xi,\text{cr}}^2) \). [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/155740/2022-01%20Vortex_Entrainment_JFM_new%5B95%5D_resaved.pdf?sequence=1&isAllowed=y)

Other mechanistic models for free‑surface turbulent flows correlate bubble entrainment with local dissipation rate \( \varepsilon \) above a critical threshold and with turbulence parameters like Reynolds and Weber numbers.  In breaking‑wave simulations, high void fractions (many bubbles) consistently appear in surface rollers and regions between counter‑rotating vortices, and their entrainment is parameterized using turbulence statistics rather than explicitly resolving each bubble. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0301932215301920)

These models justify saying: “given the vortex structures and turbulence we computed in a single‑phase simulation, we can *predict* how much air will be entrained and where bubbles will be concentrated.” [www1.udel](http://www1.udel.edu/kirby/papers/ma-etal-cacr-12-08.pdf)

***

## Two‑phase DNS/LES: the high‑fidelity route

At the more rigorous end, two‑phase DNS or LES explicitly evolve both fluids and the interface: [nmri.go](https://www.nmri.go.jp/archives/turbulence/PDF/symposium/FY2003/Tryggvason.pdf)

- Methods like volume-of-fluid (VOF), level‑set, or front‑tracking represent the interface and apply continuity of normal and tangential stresses, including surface tension. [pubs.aip](https://pubs.aip.org/aip/pof/article/35/12/123320/2928886/The-mechanisms-of-jetting-vortex-sheet-and-vortex)
- Cavitation or air entrainment happens naturally when the pressure and turbulence deform the interface; bubbles appear as separate regions of gas or vapor. [pubs.aip](https://pubs.aip.org/aip/pof/article/35/12/123320/2928886/The-mechanisms-of-jetting-vortex-sheet-and-vortex)

Recent two‑phase DNS of strong free‑surface turbulence investigate how Froude, Weber, and Reynolds numbers affect surface deformation and energy exchange, and they explicitly resolve entrained bubbles and how their shapes and total surface area depend on surface tension.  Direct simulations of bubbles in vortical near‑wall flows show that bubbles drawn into vortex cores significantly alter the vorticity evolution and wall shear through mixing and vorticity cancellation. [nmri.go](https://www.nmri.go.jp/archives/turbulence/PDF/symposium/FY2003/Tryggvason.pdf)

This is the “gold standard”: you get full bubble shapes, breakup, coalescence, and detailed flow fields, but it is very expensive and usually limited to canonical configurations rather than real engineering geometries. [nmri.go](https://www.nmri.go.jp/archives/turbulence/PDF/symposium/FY2003/Tryggvason.pdf)

***

## So, is “just vortex/velocity analysis” correct?

Putting it all together:  

- **Qualitative / model‑based prediction**: Using single‑phase turbulence simulations, analyzing vortex structures and velocity/pressure fields, and then applying cavitation or entrainment criteria is widely accepted for predicting *where* bubbles are likely to appear and how intense cavitation or air entrainment will be. [courses.washington](https://courses.washington.edu/mengr537/Reading_Assignments/MultiphaseFlowHandbook_Crowe_Chapter12.pdf)
- **Incomplete if you ignore pressure and surface tension**: If someone only looks at “vorticity is large here, therefore bubbles will appear here” without considering whether the pressure actually drops below vapor pressure (cavitation) or whether turbulence is strong enough to overcome gravity and surface tension (air entrainment), that is not consistent with the mainstream literature. [citeseerx.ist.psu](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=2e77f2097e52c6e90d22fa4224472efbf68ab602)
- **Correct “single‑phase” practice**: The correct way is to (1) compute the velocity and pressure field with an adequate turbulence model (ideally LES/DNS in strongly cavitating or entraining flows), and (2) apply physically motivated thresholds or phenomenological models based on cavitation number, Froude/Weber/Bond numbers, dissipation rate \( \varepsilon \), vortex circulation/size, etc., to decide whether bubbles will form and in what volume or size range. [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/136590/scale_separation_and_dependence_of_entrainment_bubblesize_distribution_in_freesurface_turbulence.pdf?sequence=2&isAllowed=y)
- **When you truly need two‑phase**: If you care about detailed bubble shapes, collapse dynamics, jetting, shock emission, or precise bubble‑size distributions, you generally need a two‑phase model (VOF/level‑set/front‑tracking) or at least a detailed cavitation/void‑fraction transport model. [pubs.aip](https://pubs.aip.org/aip/pof/article/35/12/123320/2928886/The-mechanisms-of-jetting-vortex-sheet-and-vortex)

So your observation is half‑right: “just analyzing vortex and velocity dynamics” *can* be part of a correct approach, but only if it is tied to proper cavitation or entrainment criteria; on its own, vorticity is a necessary but not sufficient indicator for bubble formation. [courses.washington](https://courses.washington.edu/mengr537/Reading_Assignments/MultiphaseFlowHandbook_Crowe_Chapter12.pdf)

***

## From scratch: a simple mental model

Here is a “dummy‑level” narrative, aligned with the literature: [citeseerx.ist.psu](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=2e77f2097e52c6e90d22fa4224472efbf68ab602)

1. **Start with a turbulent flow.** Fluid moves chaotically, but it contains swirling eddies (vortices) of many sizes. [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/136590/scale_separation_and_dependence_of_entrainment_bubblesize_distribution_in_freesurface_turbulence.pdf?sequence=2&isAllowed=y)
2. **Vortices create low‑pressure cores.** Because the fluid spins, a pressure gradient develops; pressure is lowest at the center of the vortex. If that pressure is low enough, the liquid can locally “boil” into vapor (cavitation). [courses.washington](https://courses.washington.edu/mengr537/Reading_Assignments/MultiphaseFlowHandbook_Crowe_Chapter12.pdf)
3. **At a free surface, turbulence tries to tear the interface.** Turbulence pushes and pulls on the water surface; gravity and surface tension try to keep it smooth. When turbulence wins (high dissipation \( \varepsilon \), high Froude and Weber numbers), pieces of air get trapped and dragged downward as bubbles. [arxiv](https://arxiv.org/html/2506.10090v1)
4. **Bubbles prefer vortices.** Once bubbles exist, they tend to drift into vortex cores and between counter‑rotating vortices, where they get trapped and transported deeper or downstream. [www1.udel](http://www1.udel.edu/kirby/papers/ma-etal-cacr-12-08.pdf)
5. **To predict bubbles cheaply, you do this:**  
   - Compute the turbulent flow of the *liquid only* (single‑phase CFD). [arxiv](https://arxiv.org/html/2409.02369v1)
   - From that, read off where vortices are, how strong they are, and what the local pressure and dissipation look like. [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/155740/2022-01%20Vortex_Entrainment_JFM_new%5B95%5D_resaved.pdf?sequence=1&isAllowed=y)
   - Apply simple rules: “If pressure < vapor pressure here → cavitation bubble region”; “If turbulence intensity above threshold near free surface → air entrainment here; bubbles in some size range.” [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0301932215301920)
6. **For full bubble physics, you need more.** If you care about every bubble’s deformation, collapse, and jetting, you must actually simulate gas/vapor as a second phase and track the interface; low‑pressure vortices alone do not give you that detail. [nmri.go](https://www.nmri.go.jp/archives/turbulence/PDF/symposium/FY2003/Tryggvason.pdf)

***

## Practical takeaway for your simulations

If in papers you read they only look at vortices/velocity and then *qualitatively* state “this is where bubbles will appear,” they are implicitly relying on the well‑documented link between vortical structures, low pressure, and entrainment, but they are skipping an explicit cavitation/entrainment closure.  For engineering‑grade prediction, the literature supports either: [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/ceat.202200465)

- Single‑phase LES/DNS plus cavitation/entrainment models that turn vortex/pressure/turbulence information into bubble volume fractions and sizes. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0301932215301920)
- Full two‑phase DNS/LES when you need detailed bubble dynamics. [pubs.aip](https://pubs.aip.org/aip/pof/article/35/12/123320/2928886/The-mechanisms-of-jetting-vortex-sheet-and-vortex)

If you tell me more about the specific geometry (pipe, venturi, free surface, rotating machinery), I can point you to more targeted entrainment/cavitation models that match exactly your “vortex‑only” simulation style.



To check whether bubble/void formation *could* happen in your FSW‑type flows using theory, you want two layers of conditions:

1. **Can vortices strong enough to lower pressure or deform an interface exist?** → mainly Reynolds‑number and circulation‑based criteria involving velocity and viscosity.  
2. **Given such vortices, are they strong enough (relative to gravity, surface tension, vapor pressure) to actually create voids or entrain air?** → cavitation number, Froude, Weber, Morton, plus vortex‑specific Re/We/Fr combinations.  

Below I’ll go from real parameters (velocity, viscosity, etc.) to dimensionless forms, with explicit formulations from the literature.

***

## Step 1 – When do vortices form?

### Reynolds number and turbulence

For a flow with characteristic speed \(U\), length scale \(L\), density \(ρ\), and dynamic viscosity \(μ\), the **Reynolds number** is

\[
\mathrm{Re} = \frac{ρ U L}{μ}.
\]

Equivalently, using kinematic viscosity \(ν = μ/ρ\),

\[
\mathrm{Re} = \frac{U L}{ν}. 
\]

This measures the ratio of inertial to viscous forces; high Re means inertia dominates and vortices and turbulence are common. [diva-portal](https://www.diva-portal.org/smash/get/diva2:1154513/FULLTEXT01.pdf)

- At **low Re** (viscosity large, velocity small, or length small), flows are laminar and vortices are weak and diffusive.  
- At **moderate to high Re**, shear layers and boundary layers become unstable; coherent vortices, shear‑layer rollers, tip vortices, etc. appear. [brennen.caltech](http://brennen.caltech.edu/HTMMult/CavitatingFlows/vortexcavitation.pdf)

For your FSW case, the “fluid” is plasticized metal, but the same idea applies: compute an effective \(U\) and \(L\) (e.g. tool‑induced shear layer thickness, slip velocity) and the effective viscosity \(μ_{\text{eff}}\). If the resulting Re is \( \gg 1\), you *can* have vortices; if it is \(O(1)\) or less, significant vortices are unlikely. [diva-portal](https://www.diva-portal.org/smash/get/diva2:1154513/FULLTEXT01.pdf)

### Vorticity and circulation

Mathematically, vorticity is \( \boldsymbol{\omega} = \nabla \times \mathbf{u} \). A typical vortex strength measure is the **circulation**

\[
\Gamma = \oint_{\mathcal{C}} \mathbf{u} \cdot d\mathbf{s},
\]

around a closed loop surrounding the vortex. Large \(|\Gamma|\) at finite core radius \(a\) means strong swirl and large tangential velocities.  [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/155740/2022-01%20Vortex_Entrainment_JFM_new%5B95%5D_resaved.pdf?sequence=1&isAllowed=y)  

A convenient “vortex Reynolds number” is

\[
\mathrm{Re}_\Gamma = \frac{|\Gamma|}{ν},
\]

which measures how much swirl is present relative to viscous diffusion. Large \(\mathrm{Re}_\Gamma\) (\(\gg 1\)) means a concentrated, persistent vortex; small values mean the vortex is quickly diffused away. [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/155740/2022-01%20Vortex_Entrainment_JFM_new%5B95%5D_resaved.pdf?sequence=1&isAllowed=y)

So, *first check*: are your velocities and viscosities such that \( \mathrm{Re} \gg 1\) and \( \mathrm{Re}_\Gamma \gg 1\)? If not, you can almost rule out strong vortex‑driven void formation. [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/155740/2022-01%20Vortex_Entrainment_JFM_new%5B95%5D_resaved.pdf?sequence=1&isAllowed=y)

***

## Step 2 – Vortex core pressure and cavitation

Once you know vortices exist, the next question is: **does the vortex core pressure drop far enough to produce vapor voids (cavitation) or help suck air in?** This is where hydrodynamic cavitation theory comes in.  

### Cavitation number (real and dimensionless parameters)

The **cavitation number** \(σ\) compares how far the local static pressure is above vapor pressure to the dynamic pressure of the flow. A common definition for a flow with characteristic velocity \(U\) and static pressure \(p\) is: [dokumen](https://dokumen.pub/hydrodynamic-cavitation-devices-design-and-applications-9783527346431-9783527822874-9783527346455-9783527346448-3527346430.html)

\[
σ = \frac{p - p_v}{\frac{1}{2} ρ U^2},
\]

where  

- \(p\) is the local or reference static pressure (often upstream or ambient),  
- \(p_v\) is the vapor pressure of the liquid at operating temperature,  
- \(ρ\) is density, \(U\) is a characteristic local speed.  

**Cavitation inception** occurs when the minimum local pressure reaches \(p_v\); equivalently, when a suitable cavitation number drops below a critical value \(σ_i\) determined by the minimum pressure coefficient of the flow (including vortex core effects). [research.chalmers](https://research.chalmers.se/publication/504636/file/504636_Fulltext.pdf)

A simple minimum‑pressure criterion is: [research.chalmers](https://research.chalmers.se/publication/504636/file/504636_Fulltext.pdf)

\[
σ_i = - C_{p,\min},
\]

where \(C_{p,\min}\) is the minimum pressure coefficient of the flow (e.g. at a vortex core). Cavitation occurs when \(σ \le σ_i\). [research.chalmers](https://research.chalmers.se/publication/504636/file/504636_Fulltext.pdf)

### Vortex core pressure for Rankine/Lamb vortices

For a vortex of circulation \(\Gamma\) and viscous core radius \(a\), Arndt’s model (Rankine/Lamb vortex) yields a vortex core pressure drop: [brennen.caltech](http://brennen.caltech.edu/HTMMult/CavitatingFlows/vortexcavitation.pdf)

\[
p_{\min} - p_0 = C_{p,\min} \frac{ρ}{2} \left(\frac{\Gamma}{2\pi a}\right)^2,
\]

where  

- \(p_0\) is the ambient pressure,  
- \(C_{p,\min}\) is a negative constant depending on the vortex model (\(C_{p,\min} \approx -2.0\) for a Rankine vortex, \(\approx -1.74\) for a Lamb vortex). [dokumen](https://dokumen.pub/hydrodynamic-cavitation-devices-design-and-applications-9783527346431-9783527822874-9783527346455-9783527346448-3527346430.html)

Thus the **core pressure** is

\[
p_{\min} = p_0 + C_{p,\min} \frac{ρ}{2} \left(\frac{\Gamma}{2\pi a}\right)^2.
\]

Cavitation in the vortex core occurs when \(p_{\min} \le p_v\), i.e.

\[
p_0 + C_{p,\min} \frac{ρ}{2} \left(\frac{\Gamma}{2\pi a}\right)^2 \le p_v.
\]

Rearranging:

\[
\left(\frac{\Gamma}{2\pi a}\right)^2 \ge \frac{2 (p_0 - p_v)}{ρ |C_{p,\min}|}.
\]

So, in **real parameters**:

- Higher circulation \(|\Gamma|\) (stronger swirl)  
- Smaller core radius \(a\)  
- Lower ambient pressure \(p_0\)  
- Larger density \(ρ\)  

all favor cavitation; viscosity enters indirectly via how big \(a\) and \(\Gamma\) can be at a given Re. [brennen.caltech](http://brennen.caltech.edu/HTMMult/CavitatingFlows/vortexcavitation.pdf)

Dimensionless form: if you use \(U = U_\infty\) (freestream) and define \(σ\) as above, you can relate \(σ\) to \(\Gamma\) via a vortex pressure coefficient \(C_p(r)\). Cavitation inception then occurs when \(σ\) falls below the value implied by \(C_{p,\min}\). [diva-portal](https://www.diva-portal.org/smash/get/diva2:1154513/FULLTEXT01.pdf)

### Reynolds number effects on cavitation threshold

Experiments show that the **critical cavitation number** for vortex cavitation depends on Reynolds number; e.g. tip vortex cavitation inception cavitation number scales like \( \mathrm{Re}^{0.4}\) in some configurations. [witpress](https://www.witpress.com/Secure/elibrary/papers/NEVA93/NEVA93013FU.pdf)

So viscosity (through Re) affects:

- the boundary‑layer development and shear‑layer roll‑up that create the vortex,  
- the size and strength of the vortex core,  
- the cavitation threshold \(σ_i(\mathrm{Re})\). [apps.dtic](https://apps.dtic.mil/sti/tr/pdf/ADA211426.pdf)

For a first‑cut check in your FSW flow, you’d compute Re and a representative \(σ\) and compare to typical inception ranges; very large Re and small \(σ\) are needed for vortex cavitation. [witpress](https://www.witpress.com/Secure/elibrary/papers/NEVA93/NEVA93013FU.pdf)

***

## Step 3 – Free‑surface air entrainment: Froude, Weber, Reynolds, Morton

If there is an air–liquid interface (e.g. a free surface or gap where air can enter), **air entrainment** requires that turbulent stresses overcome gravity and surface tension. This is often expressed using **Froude**, **Weber**, **Reynolds**, and **Morton** numbers. [staff.civil.uq.edu](https://staff.civil.uq.edu.au/h.chanson/reprints/jhr_2013_3_223.pdf)

### Classical definitions

For a liquid with density \(ρ\), viscosity \(μ\), surface tension \(σ\), characteristic velocity \(U\), and length scale \(L\), common dimensionless numbers are: [data-ww3.ifremer](https://data-ww3.ifremer.fr/BIB/Brocchini_Peregrine_JFM2001.pdf)

- **Reynolds**: \( \displaystyle \mathrm{Re} = \frac{ρ U L}{μ} = \frac{U L}{ν}\).  
- **Froude**: \( \displaystyle \mathrm{Fr} = \frac{U}{\sqrt{g L}}\), comparing inertia to gravity.  
- **Weber**: \( \displaystyle \mathrm{We} = \frac{ρ U^2 L}{σ}\), comparing inertia to surface tension.  
- **Morton**: \( \displaystyle \mathrm{Mo} = \frac{g μ^4}{ρ σ^3}\), depends only on fluid properties and gravity. [staff.civil.uq.edu](https://staff.civil.uq.edu.au/h.chanson/reprints/jhr_2013_3_223.pdf)

Brocchini & Peregrine (JFM 2001) showed that regions of weak disturbance vs surface breakup and bubble entrainment can be mapped in the \((L,q)\) plane, where \(q\) is a turbulence velocity scale, using Fr and We: [data-ww3.ifremer](https://data-ww3.ifremer.fr/BIB/Brocchini_Peregrine_JFM2001.pdf)

- Low Fr and low We → surface nearly rigid, little/no entrainment.  
- High Fr and/or high We → strong deformation, breaking waves, bubble and droplet entrainment. [data-ww3.ifremer](https://data-ww3.ifremer.fr/BIB/Brocchini_Peregrine_JFM2001.pdf)

Chanson and others argue that fully dynamic similarity in aerated free‑surface flows requires matching Fr, Re, and Mo. [staff.civil.uq.edu](https://staff.civil.uq.edu.au/h.chanson/reprints/jhr_2013_3_223.pdf)

### Critical turbulence intensity and velocity thresholds

Experiments on free‑surface self‑aeration in open‑channel flows show: [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC6662671/)

- Bubble entrainment starts when **turbulent fluctuation velocity normal to the free surface** \(u'_{n}\) exceeds a critical value that allows turbulent stresses to overcome gravity and surface tension.  
- In many open‑channel experiments, entrainment begins at mean velocities of about **3–6 m/s** with Re on the order \(10^5\) and We on the order \(10^4\). [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC6662671/)

So in **real parameters**, you’d check:

1. Is \(u'_{n}\) (or some relevant near‑surface velocity scale) large enough that 

   \[
   ρ (u'_{n})^2 \gtrsim ρ g L \quad\text{and}\quad ρ (u'_{n})^2 \gtrsim \frac{σ}{L},
   \]

   which translates to Fr, We ≳ O(1)? [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC6662671/)

2. Is Re ≫ 1 so that turbulence can exist at all? [staff.civil.uq.edu](https://staff.civil.uq.edu.au/h.chanson/reprints/jhr_2013_3_223.pdf)

If both gravity and surface tension are weak compared to turbulent inertia (i.e. Fr and We are large), substantial air entrainment and bubble formation are expected. [dspace.mit](https://dspace.mit.edu/handle/1721.1/124246)

***

## Step 4 – Vortex‑specific entrainment criteria (Γ, a, W)

Recent work (MIT, JFM “Modelling entrainment volume due to surface‑parallel vortex…”) provides a very explicit connection between **vortex parameters and entrained air volume**.  This is particularly relevant if your flow has identifiable coherent vortices near an interface (tool shoulder or free surface). [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/155740/2022-01%20Vortex_Entrainment_JFM_new%5B95%5D_resaved.pdf?sequence=1&isAllowed=y)

### Circulation‑flux Froude number

Consider a vortex with:

- circulation \(\Gamma\),  
- effective radius \(a\),  
- vertical rise (or approach) velocity \(W\),  
- located at centroid depth \(z_c\) below the interface,  
- fluid density \(ρ_w\), viscosity \(ν_w\), surface tension \(σ\), gravity \(g\).  

The study defines a **circulation flux Froude number**: [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/155740/2022-01%20Vortex_Entrainment_JFM_new%5B95%5D_resaved.pdf?sequence=1&isAllowed=y)

\[
\mathrm{Fr}_\Xi^2 = \frac{|\Gamma| W}{a^2 g} = \frac{\Xi}{g},
\]

where \(\Xi = |\Gamma| W / a^2\) is a circulation flux density.  

They find:

- If \( \mathrm{Fr}_\Xi^2 < \mathrm{Fr}_{\Xi,\text{cr}}^2\), **no air is entrained**.  
- If \( \mathrm{Fr}_\Xi^2 > \mathrm{Fr}_{\Xi,\text{cr}}^2\), the **dimensionless entrained volume** \(V_o/V_\Gamma\) increases roughly linearly with \(\mathrm{Fr}_\Xi^2 - \mathrm{Fr}_{\Xi,\text{cr}}^2\). [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/155740/2022-01%20Vortex_Entrainment_JFM_new%5B95%5D_resaved.pdf?sequence=1&isAllowed=y)

Here \(V_\Gamma\) is a scaling volume based on \(a\) and \(\Gamma\).  

### Vortex Reynolds and Weber numbers

They also introduce vortex‑specific dimensionless numbers: [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/155740/2022-01%20Vortex_Entrainment_JFM_new%5B95%5D_resaved.pdf?sequence=1&isAllowed=y)

- **Vortex Reynolds number**

  \[
  \mathrm{Re}_\Gamma = \frac{|\Gamma|}{ν_w},
  \]

  measuring viscous effects.  

- **Vortex Weber number**

  \[
  \mathrm{We}_\Gamma = \frac{\Gamma^2}{a (σ/ρ_w)},
  \]

  measuring surface tension effects of the vortex.  

They show that:

- For \( 5 \lesssim \mathrm{Bo}_\Gamma \lesssim 50\) (a vortex Bond number, not spelled out here), the entrained volume depends linearly on \(\mathrm{We}_\Gamma\).  
- Entrained volume depends parabolically on \(\mathrm{Re}_\Gamma\) for \( \mathrm{Re}_\Gamma \lesssim 2580\); beyond that, viscosity has little effect on initial entrainment volume. [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/155740/2022-01%20Vortex_Entrainment_JFM_new%5B95%5D_resaved.pdf?sequence=1&isAllowed=y)

The overall parameterization for dimensionless entrained volume is: [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/155740/2022-01%20Vortex_Entrainment_JFM_new%5B95%5D_resaved.pdf?sequence=1&isAllowed=y)

\[
\frac{V_o}{V_\Gamma}
= f\left(
\mathrm{Fr}_\Xi^2 = \frac{|\Gamma| W}{a^2 g},\;
\mathrm{We}_\Gamma = \frac{\Gamma^2}{a (σ/ρ_w)},\;
\mathrm{Re}_\Gamma = \frac{|\Gamma|}{ν_w},\;
\frac{z_c}{a}
\right).
\]

In practice, \(\mathrm{Fr}_\Xi^2\) is the primary control parameter; \(\mathrm{We}_\Gamma\) and \(\mathrm{Re}_\Gamma\) modulate the result, and \(\mathrm{Re}_\Gamma\) is where viscosity enters explicitly. [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/155740/2022-01%20Vortex_Entrainment_JFM_new%5B95%5D_resaved.pdf?sequence=1&isAllowed=y)

This gives you a very concrete way to answer your question:

- **First**, compute \(\Gamma, a, W, ν_w\) and hence \(\mathrm{Fr}_\Xi^2\), \(\mathrm{Re}_\Gamma\), \(\mathrm{We}_\Gamma\) from your single‑phase simulation.  
- **Second**, compare \(\mathrm{Fr}_\Xi^2\) to the critical value \(\mathrm{Fr}_{\Xi,\text{cr}}^2\) reported in the paper to see if that vortex can entrain air. [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/155740/2022-01%20Vortex_Entrainment_JFM_new%5B95%5D_resaved.pdf?sequence=1&isAllowed=y)

***

## Step 5 – Putting it in a practical checklist

To check whether your FSW‑type flow can produce voids or air intrusion via vortices, a literature‑consistent workflow would be:

1. **Evaluate basic flow parameters** (real):  
   - Effective \(U\) (tool‑relative velocity and local shear).  
   - Length scales \(L\) (tool pin radius, layer thickness, vortex core radius).  
   - Fluid properties \(ρ, μ, σ, p_0, p_v\).  

2. **Compute dimensionless numbers**:  
   - \( \mathrm{Re} = ρ U L / μ\) – is the flow turbulent or at least strongly vortical? [diva-portal](https://www.diva-portal.org/smash/get/diva2:1154513/FULLTEXT01.pdf)
   - \( \mathrm{Fr} = U / \sqrt{gL}\) – can inertia overcome gravity at an interface? [data-ww3.ifremer](https://data-ww3.ifremer.fr/BIB/Brocchini_Peregrine_JFM2001.pdf)
   - \( \mathrm{We} = ρ U^2 L / σ\) – can inertia overcome surface tension at an interface? [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC6662671/)
   - \(σ = (p_0 - p_v) / (0.5 ρ U^2)\) – is cavitation possible for given vapor pressure? [dokumen](https://dokumen.pub/hydrodynamic-cavitation-devices-design-and-applications-9783527346431-9783527822874-9783527346455-9783527346448-3527346430.html)

3. **If coherent vortices are present**, extract vortex parameters:  
   - Circulation \(\Gamma\), core radius \(a\), approach velocity \(W\), viscous \(ν\).  
   - Compute \(\mathrm{Re}_\Gamma\), \(\mathrm{We}_\Gamma\), \(\mathrm{Fr}_\Xi^2\), and check against entrainment model conditions. [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/155740/2022-01%20Vortex_Entrainment_JFM_new%5B95%5D_resaved.pdf?sequence=1&isAllowed=y)

4. **Cavitation condition** (bulk voids):  
   - Use vortex core pressure formula to test whether \(p_{\min} \le p_v\):  

     \[
     p_{\min} = p_0 + C_{p,\min} \frac{ρ}{2} \left(\frac{\Gamma}{2\pi a}\right)^2.
     \]

     If this inequality holds, cavitation bubbles can form in that vortex core. [brennen.caltech](http://brennen.caltech.edu/HTMMult/CavitatingFlows/vortexcavitation.pdf)

5. **Air entrainment condition** (free surface / gap):  
   - At interfaces, check whether turbulent velocities and Fr/We are above thresholds where surface breakup and air entrainment occur (e.g. Fr ≳ O(1), We ≳ O(1)), and whether your \(U\) is in ranges where experiments see self‑aeration (3–6 m/s for water at typical scales). [dspace.mit](https://dspace.mit.edu/handle/1721.1/124246)
   - For vortex‑driven entrainment specifically, use \(\mathrm{Fr}_\Xi^2\) and \(\mathrm{Re}_\Gamma\), \(\mathrm{We}_\Gamma\) as in the MIT model. [dspace.mit](https://dspace.mit.edu/bitstream/handle/1721.1/155740/2022-01%20Vortex_Entrainment_JFM_new%5B95%5D_resaved.pdf?sequence=1&isAllowed=y)

If, after these checks, you find:

- \( \mathrm{Re} \gg 1\) and coherent vortices exist,  
- vortex core pressures can reach or go below \(p_v\) (low \(σ\)), or  
- near any air–metal interface, Fr and We (or \(\mathrm{Fr}_\Xi^2\)) are large enough that turbulence can defeat gravity and surface tension,  

then the literature says it is *theoretically* possible for vortex structures in your FSW flow to generate voids or air intrusion. [dokumen](https://dokumen.pub/hydrodynamic-cavitation-devices-design-and-applications-9783527346431-9783527822874-9783527346455-9783527346448-3527346430.html)

If you like, next we can go case‑by‑case for a particular FSW geometry: you give approximate \(U, L, μ_{\text{eff}}, ρ\), and we can walk through these inequalities numerically to see whether void formation is plausible in your parameter regime.
