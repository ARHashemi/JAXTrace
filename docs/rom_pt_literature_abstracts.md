# ROM PT · Raw literature archive (Consensus abstracts)

**Date:** 2026-07-31
**Purpose:** Full abstracts + citation counts + URLs for every paper
surfaced across the two Consensus search passes (26 queries total,
of which 19 were on-topic and are archived here). Kept as a
"gold-copy" archive so the doc set remains self-contained if
Consensus goes down or the free-tier search budget is exhausted.

**Companion docs**:
- [`rom_pt_roadmap_REVIEW2_evaluation.md`](rom_pt_roadmap_REVIEW2_evaluation.md) — first evaluation + priority table
- [`rom_pt_implementation_dossier.md`](rom_pt_implementation_dossier.md) — implementation blueprints with citations
- This file — raw abstracts, ordered by topic

Papers are numbered to match the reference tables in the dossier
(1–14 from first pass, 34–96 from second pass). Query text is
recorded above each block so a future reader can rerun / expand.

---

## First pass (2026-07-30, evaluation of REVIEW2)

### Query · "Xie Iliescu Lagrangian inner product POD reduced order model FTLE"

**[1] Xie, Nolan, Ross, Mou, Iliescu (2020) · Lagrangian Reduced Order Modeling of Finite Time Lyapunov Exponents** · 4 citations
<https://consensus.app/papers/details/f52431fea1f35c768a6adac63b482fb1/?utm_source=claude_desktop>
> There are two main strategies for improving the projection-based reduced order model (ROM) accuracy: (i) improving the ROM, i.e., adding new terms to the standard ROM; and (ii) improving the ROM basis, i.e., constructing ROM bases that yield more accurate ROMs. In this paper, we use the latter. We propose new Lagrangian inner products that we use together with Eulerian and Lagrangian data to construct new Lagrangian ROMs. We show that the new Lagrangian ROMs are orders of magnitude more accurate than the standard Eulerian ROMs, i.e., ROMs that use standard Eulerian inner product and data to construct the ROM basis. Specifically, for the quasi-geostrophic equations, we show that the new Lagrangian ROMs are more accurate than [...]

**[2] Xie et al. (2018) · arXiv preprint of [1]** · 10 citations
<https://consensus.app/papers/details/7a2aeb5a0cfa5cc494df041cfd3a8636/?utm_source=claude_desktop>
Same abstract as [1] — full quote in dossier §3.

**[3] Parish, Rizzi, Blonigan (2022) · On the impact of dimensionally-consistent and physics-based inner products for POD-Galerkin and least-squares model reduction of compressible flows** · 25 citations · ArXiv
<https://consensus.app/papers/details/9fdc346d0b4d500f8636b37839cf2b58/?utm_source=claude_desktop>
> Model reduction of the compressible Euler equations based on proper orthogonal decomposition (POD) and Galerkin orthogonality or least-squares residual minimization requires the selection of inner product spaces in which to perform projections and measure norms. The most popular choice is the vector-valued L2({\Omega}) inner product space. This choice, however, yields dimensionally-inconsistent reduced-order model (ROM) formulations which often lack robustness. In this work, we try to address this weakness by studying a set of dimensionally-consistent inner products with application to the compressible Euler equations. First, we demonstrate that non-dimensional inner products have a positive impact on both POD and Galerkin/least-squares ROMs. Second, we further demonstrate that physics-based inner products based on entropy principles result in drastically more accurate and robust ROM formulations than those based on non-dimensional L2({\Omega}) inner products. As test cases, we consider the following well-known problems: the one-dimensional Sod shock tube, the two-dimensional Kelvin-Helmholtz instability and two-dimensional homogeneous isotropic turbulence.

### Query · "divergence-free POD reduced order model incompressible Navier-Stokes velocity basis"

**[4] Akhtar, Nayfeh, Ribbens (2009) · On the stability and extension of reduced-order Galerkin models in incompressible flows** · 224 citations · Theoretical and Computational Fluid Dynamics
<https://consensus.app/papers/details/933ae415d39650048d32ea52c27aea12/?utm_source=claude_desktop>
> Proper orthogonal decomposition (POD) has been used to develop a reduced-order model of the hydrodynamic forces acting on a circular cylinder. Direct numerical simulations of the incompressible Navier-Stokes equations have been performed using a parallel computational fluid dynamics (CFD) code to simulate the flow past a circular cylinder. Snapshots of the velocity and pressure fields are used to calculate the divergence-free velocity and pressure modes, respectively. We use the dominant of these velocity POD modes (a small number of eigenfunctions or modes) in a Galerkin procedure to project the Navier-Stokes equations onto a low-dimensional space, thereby reducing the distributed-parameter problem into a finite-dimensional nonlinear dynamical system in time.

**[5] Stabile & Rozza (2017) · Finite volume POD-Galerkin stabilised reduced order methods for the parametrised incompressible Navier-Stokes equations** · 219 citations · Computers & Fluids
<https://consensus.app/papers/details/ead7a2245d1a501795770d835ad3ed3a/?utm_source=claude_desktop>
> In this work a stabilised and reduced Galerkin projection of the incompressible unsteady Navier-Stokes equations for moderate Reynolds number is presented. The full-order model, on which the Galerkin projection is applied, is based on a finite volumes approximation. The reduced basis spaces are constructed with a POD approach. Two different pressure stabilisation strategies are proposed and compared: the former one is based on the supremizer enrichment of the velocity space, and the latter one is based on a pressure Poisson equation approach.

**[6] Novo & Rubino (2020) · Error analysis of POD stabilized methods for incompressible flows** · 36 citations · ArXiv
<https://consensus.app/papers/details/def9702c3df45a6c9277be5bb5334e1e/?utm_source=claude_desktop>
> Proper orthogonal decomposition (POD) stabilized methods for the Navier-Stokes equations are considered and analyzed. We consider two cases, the case in which the snapshots are based on a non inf-sup stable method and the case in which the snapshots are based on an inf-sup stable method. For both cases we construct approximations to the velocity and the pressure. For the first case, we analyze a method in which the snapshots are based on a stabilized scheme with equal order polynomials for the velocity and the pressure with Local Projection Stabilization (LPS) for the gradient of the velocity and the pressure. For the POD method we add the same kind of LPS stabilization for the gradient of the velocity and the pressure than the direct method, together with grad-div stabilization. In the second case, the snapshots are based on an inf-sup stable Galerkin method with grad-div stabilization and for the POD model we apply also grad-div stabilization. **In this case, since the snapshots are discretely divergence-free, the pressure can be removed from the formulation of the POD approximation to the velocity.** To approximate the pressure, needed in many engineering applications, we use a supremizer pressure recovery method. Error bounds with constants independent on inverse powers of the viscosity parameter are proved for both methods. Numerical experiments show the accuracy and performance of the schemes.

**[7] Gräßle, Hinze, Ulbrich (2019) · Model order reduction for space-adaptive simulations of Navier-Stokes** · PAMM
<https://consensus.app/papers/details/06db025349d25ee89613a2a744f9f09d/?utm_source=claude_desktop>
> We consider model order reduction with proper orthogonal decomposition (POD) for simulations of incompressible, laminar flow governed by the Navier-Stokes equations. [...] In order to guarantee stability of the POD surrogate model, we either use a projection onto a common divergence-free space or utilize a supremizer enrichment of the reduced velocity space. In a numerical experiment, we compare the accuracy of the resulting reduced solution with the standard approach using a static, uniform snapshot discretization.

**[8] Star, Sanderse, Stabile, Rozza, Degroote (2020) · Reduced order models for the incompressible Navier-Stokes equations on collocated grids using a 'discretize-then-project' approach** · 9 citations · Int. J. Numer. Methods Fluids
<https://consensus.app/papers/details/2afcea692ec7596ebba5f821ef31604d/?utm_source=claude_desktop>
> A novel reduced order model (ROM) for incompressible flows is developed by performing a Galerkin projection based on a fully (space and time) discrete full order model (FOM) formulation. This 'discretize-then-project' approach requires no pressure stabilization technique (even though the pressure term is present in the ROM) nor a boundary control technique (to impose the boundary conditions at the ROM level). [...] Two variants of the time discretization method, the inconsistent and consistent flux method, have been investigated. **The latter leads to divergence-free velocity fields, also on the ROM level**, whereas the velocity fields are only approximately divergence-free in the former method. For both methods, accurate results have been obtained for test cases with different types of boundary conditions: a lid-driven cavity and an open-cavity (with an inlet and outlet). **The ROM obtained with the consistent flux method, having divergence-free velocity fields, is slightly more accurate but also slightly more expensive to solve compared to the inconsistent flux method.**

**[9] Lee et al. (2020) · On the Importance of Numerical Error in Constructing POD-based Reduced-Order Models of Nonlinear Fluid Flows**
<https://consensus.app/papers/details/6ae4c72d22f251cca054e5e19ea5ae11/?utm_source=claude_desktop>
> The proper orthogonal decomposition (POD) provides a useful method for studying nonlinear fluid dynamics; however, the construction and use of the POD basis modes for Reduced Order Modeling (ROM) introduces several sources of error which can jeopardize the fidelity of the resulting ROM simulations. Given the chaotic nature of turbulent fluid flows, an understanding of these sources of error and their influence on the simulated dynamics is important to the successful implementation of the POD method. In particular, application of the divergence theorem to the pressure term in the Galerkin formulation of the Navier-Stokes equations presents a clear requirement for when the pressure term can be neglected in constructing the ROM. In the present paper, sources of error are identified which call into question if a POD basis for an incompressible flow is divergence-free and if the pressure term can thus be correctly neglected.

### Query · "POD proper orthogonal decomposition energy truncation weighted inner product low velocity region under-resolved"

**[10] Christensen, Brøns, Sørensen (1999) · Evaluation of POD-Based Decomposition Techniques Applied to Parameter-Dependent Nonturbulent Flows** · 121 citations · SIAM J. Sci. Comput.
<https://consensus.app/papers/details/113e721cea5d5808a3f03c6228e403f8/?utm_source=claude_desktop>
> The proper orthogonal decomposition (POD) method as a systematic technique to analyze parameter-dependent problems may be inappropriate as a practical tool for generating low-dimensional models. **We propose a weighted POD (w-POD) as an alternative to give higher priority to low-energetic or important modes by simply weighting.** A predefined POD (p-POD) is suggested, where modes are selected not only on the basis of energy but also on some a priori knowledge of the system. The techniques are tested on a flow problem undergoing steady or unsteady transition [...]. It is shown that modes extracted locally may generally not contain information regarding the global dynamics [...]. It is demonstrated that by weighting and predefining base vectors it is possible to improve the POD technique's capability to generate low-dimensional models.

**[11] Dellacasagrande, Guardone, Simoni (2021) · Identification of coexisting dynamics in boundary layer flows through POD with weighting matrices** · Meccanica
<https://consensus.app/papers/details/d91633c6ea3c5c19947e9e8f45fe24cb/?utm_source=claude_desktop>
> A different version of the classic proper orthogonal decomposition (POD) procedure introducing spatial and temporal weighting matrices is proposed. Furthermore, a newly defined non-Euclidean (NE) inner product that retain similarities with the POD is introduced in the paper. **The aim is to emphasize fluctuation events localized in spatio-temporal regions with low kinetic energy magnitude, which are not highlighted by the classic POD.** [...] Modes obtained by the non-Euclidean POD (NE-POD) procedure (where weighted projections are considered) are shown to better extract low energy events sparse in time and space with respect to modes extracted by other variants.

**[12] Olesen, Hodžić, Andersen, Sørensen, Velte (2022) · Dissipation-optimized Proper Orthogonal Decomposition** · 12 citations · Physics of Fluids
<https://consensus.app/papers/details/04148b54bb475ca4a76689a5135d0bc1/?utm_source=claude_desktop>
> We present a formalism for dissipation-optimized decomposition of the strain rate tensor (SRT) of turbulent flow data using Proper Orthogonal Decomposition (POD). [...] The lowest dissipation-optimized POD (d-POD) modes are compared to the lowest conventional turbulent kinetic energy (TKE) optimized POD (e-POD) modes. **The lowest d-POD modes show a richer small-scale structure, along with traces of the large-scale structure characteristic of e-POD modes**, indicating that the former capture structures across a wider range of spatial scales. Profiles of both TKE and dissipation are reconstructed using both decompositions [...]. Both TKE and dissipation are reconstructed more efficiently in the dissipation-rich near-wall region using d-POD modes, and in the TKE-rich bulk using e-POD modes.

### Query · "friction stir welding particle tracking reduced order model mixing shear"

**[13] Cao et al. (2021) · Machine learning and reduced order computation of a friction stir welding model** · 4 citations · J. Comput. Phys.
<https://consensus.app/papers/details/23e2a9dc5e165c12b1dbeb4c0f7d3fb0/?utm_source=claude_desktop>
> The friction stir welding process can be modeled using a system of heat transfer and Navier-Stokes equations with a shear dependent viscosity. Finding numerical solutions of this system of nonlinear partial differential equations over a set of parameter space, however, is extremely time-consuming. Therefore, it is desirable to find a computationally efficient method that can be used to obtain an approximation of the solution with acceptable accuracy. In this paper, we present a reduced basis method for solving the parametrized coupled system of heat and Navier-Stokes equations using a proper orthogonal decomposition (POD). In addition, we apply a machine learning algorithm based on an artificial neural network (ANN) to learn (approximately) the relationship between relevant parameters and the POD coefficients.

### Query · "time-dependent POD basis snapshot temporal ROM Navier-Stokes accuracy"

**[14] García-Archilla, Novo, Rubino (2022) · POD-ROMs for incompressible flows including snapshots of the temporal derivative of the full order solution** · 11 citations · SIAM J. Numer. Anal.
<https://consensus.app/papers/details/c9d5fcdb253d5e449950094e5defb2bb/?utm_source=claude_desktop>
> In this paper we study the influence of including snapshots that approach the velocity time derivative in the numerical approximation of the incompressible Navier-Stokes equations by means of proper orthogonal decomposition (POD) methods. Our set of snapshots includes the velocity approximation at the initial time from a full order mixed finite element method (FOM) together with approximations to the time derivative at different times. [...] For the POD method we study the differences between projecting onto L^2 and H^1. In both cases pointwise in time error bounds can be proved. **Including grad-div stabilization both in the FOM and POD methods error bounds with constants independent on inverse powers of the viscosity can be obtained.**

**[15] García-Archilla, Novo, Rubino (2023) · POD-ROMs for incompressible flows including snapshots of the temporal derivative of the full order solution: Error bounds for the pressure** · 4 citations · J. Numerical Math.
<https://consensus.app/papers/details/6c4e950b974853f9b6f84210849f5aa9/?utm_source=claude_desktop>
> Reduced order methods (ROMs) for the incompressible Navier-Stokes equations, based on proper orthogonal decomposition (POD), are studied that include snapshots which approach the temporal derivative of the velocity from a full order mixed finite element method (FOM). [...] The present paper studies two different procedures to compute approximations to the pressure and proves error bounds for the pressure that are independent of inverse powers of the viscosity parameter.

**[16] García-Archilla, Novo, Rubino (2024) · POD-ROM methods: from a finite set of snapshots to continuous-in-time approximations** · 4 citations · ArXiv
<https://consensus.app/papers/details/9ef2af95b09b5a95bc263fa192e10dfa/?utm_source=claude_desktop>
> This paper studies discretization of time-dependent partial differential equations (PDEs) by proper orthogonal decomposition reduced order models (POD-ROMs). Most of the analysis in the literature has been performed on fully-discrete methods using first order methods in time, typically the implicit Euler time integrator. Our aim is to show which kind of error bounds can be obtained using any time integrator, both in the full order model (FOM), applied to compute the snapshots, and in the POD-ROM method. [...] Optimal pointwise-in-time error bounds are proved for the L^2(\Omega) norm of the error for a semilinear reaction-diffusion model problem. [...] Our detailed analysis allows to show that, in some situations, **a small number of snapshots in a given time interval might be sufficient to accurately approximate the solution in the full interval.**

### Query · "Lagrangian coherent structures FTLE friction stir mixing flow shear layer"

**[17] Kun Li et al. (2022) · Computation of Lagrangian Coherent Structures from Experimental Fluid Trajectory Measurements in a Mechanically Agitated Vessel** · 25 citations · Chemical Engineering Science
<https://consensus.app/papers/details/29f22ac5f77855d986d8ece317dd8efd/?utm_source=claude_desktop>
> In mechanically agitated vessels, bulk flow circulation which plays a leading role in macroscale mixing is controlled by hidden Lagrangian coherent structures (LCSs). We use a numerical finite-time Lyapunov exponent (FTLE) approach, for the first time, to resolve such LCSs. **Experimental 3D Lagrangian trajectories obtained from a unique positron emission particle tracking (PEPT) technique** are used to drive the FTLE model. By computing forward and backward FTLE fields and extracting repelling and attracting FTLE ridges in various azimuthal planes of the flow, a highly complex flow topology is unravelled which varies significantly with azimuthal position. We demonstrate how LCSs organise and quantify the chaotic behaviour of fluid particle paths that underpin mixing through the exchange of fluid between zones of different kinematics.

**[18] Bashiri et al. (2016) · Investigation of turbulent fluid flows in stirred tanks using a non-intrusive particle tracking technique** · 36 citations · Chemical Engineering Science
<https://consensus.app/papers/details/f6d548bfa3ea579cad6a25120b72eaa6/?utm_source=claude_desktop>
> Fully turbulent fluid flows in a laboratory-scale stirred tank (ST) equipped with a radial flow impeller (Rushton turbine; RT) or an axial flow impeller (pitched blade turbine; PBT) were analyzed using the radioactive particle tracking (RPT) technique. The present study covered the Eulerian and Lagrangian descriptions of fluid motions. [...] **Lagrangian mixing measurements showed that particle trajectories can be used to generate Poincaré maps**, which in turn can be used as a tool to visualize the 3D flow structure inside mixing systems. **Two mixing indices, one based on the concept of stochastic independence and the other on the statistical concept of memory loss in mixing processes**, were used to measure mixing times using RPT results.

**[19] Shadden, Lekien, Marsden (2005) · Definition and properties of Lagrangian coherent structures from finite-time Lyapunov exponents in two-dimensional aperiodic flows** · 1439 citations · Physica D
<https://consensus.app/papers/details/f3e7724c147b537ab134203df30ecf9e/?utm_source=claude_desktop>
Canonical LCS/FTLE reference. Full abstract in the first evaluation doc.

### Query · "FTLE particle tracking accuracy integration scheme Runge-Kutta interpolation error velocity field"

**[20] Pokrajac, Lazic (2002) · An efficient algorithm for high accuracy particle tracking in finite elements** · 26 citations · Advances in Water Resources
<https://consensus.app/papers/details/a5ef30b6fa025145a95a68d5b8de485c/?utm_source=claude_desktop>
> We propose an algorithm for particle tracking based on Cheng's method [Int. J. Numer. Meth. 39 (1996) 1111-1136]. Velocities in a flow field are known at a series of points and interpolated between them by finite element local functions. Tracking is performed in local coordinates, element by element, using any standard ODE solution method. The exit from an element is found using the polynomials to interpolate between the tracking points. The algorithm was tested and compared to Pollock's and Cheng's method in a series of numerical experiments, in which the Euler, Runge-Kutta 2, Runge-Kutta 5(4) and Runge-Kutta 6(4) ODE solution methods were combined with first-, second-, third- and fifth-order exit polynomials. [...] **The numerical experiments confirmed that the accuracy of the exit polynomial has to be consistent with the ODE solution method. Quadratic interpolation of velocities on a coarser mesh often gives more accurate path lines and requires less computational time than linear interpolation. Pollock's method for particle tracking is viable only if input data are rather inaccurate and path lines nearly straight. Cheng's method is appropriate for moderately accurate input data, while the proposed algorithm with Runge-Kutta 5(4) or Runge-Kutta 6(4) method and fifth-order exit polynomial has excellent accuracy.** Computational time is about 10 times longer than for Cheng's method while the accuracy is increased by several orders of magnitude.

**[21] Beznosov et al. (2025) · High order interpolation of magnetic fields with vector potential reconstruction for particle simulations** · 1 citation · Comput. Phys. Commun.
<https://consensus.app/papers/details/770bc529ee665f828855fad69f0c61a3/?utm_source=claude_desktop>
> We propose a method for interpolating divergence-free continuous magnetic fields via vector potential reconstruction using Hermite interpolation, which ensures high-order continuity for applications requiring adaptive, high-order ordinary differential equation (ODE) integrators, such as the Dormand-Prince method. The method provides C(m) continuity and achieves high-order accuracy, making it particularly suited for particle trajectory integration and Poincaré section analysis under optimal integration order and timestep adjustments. Through numerical experiments, we demonstrate that the Hermite interpolation method preserves volume and continuity, which are critical for conserving toroidal canonical momentum and magnetic moment in guiding center simulations, especially over long-term trajectory integration. Furthermore, **we analyze the impact of insufficient derivative continuity on Runge-Kutta schemes and show how it degrades accuracy at low error tolerances, introducing discontinuity-induced truncation errors.**

**[22] Rössler et al. (2018) · Trajectory errors of different numerical integration schemes diagnosed with the MPTRAC advection module driven by ECMWF operational analyses** · 23 citations · Geoscientific Model Development
<https://consensus.app/papers/details/9da7d93f3adc53009325a5c891c76f6d/?utm_source=claude_desktop>
> We analyzed global truncation errors of six explicit integration schemes of the Runge-Kutta family, which we implemented in the Massive-Parallel Trajectory Calculations (MPTRAC) advection module. The simulations were driven by wind fields from operational analysis and forecasts of the European Centre for Medium-Range Weather Forecasts (ECMWF) at T1279L137 spatial resolution and 3 h temporal sampling. [...] In total, more than 5000 different transport simulations were performed [...]. **We found that the truncation errors of the six numerical schemes fall into three distinct groups, which mostly depend on the numerical order of the scheme. Schemes of the same order differ little in accuracy, but some methods need less computational time [...] we recommend the third-order Runge-Kutta method with a time step of 170 s or the midpoint scheme with a time step of 100 s for efficient simulations of up to 10 days.**

**[23] Coppola et al. (2001) · Nonlinear particle tracking for high-order elements** · 45 citations · J. Comput. Phys.
<https://consensus.app/papers/details/e4c41a132aaa5219997d90d0f9ae35b4/?utm_source=claude_desktop>
> The problem of calculating particle trajectories on unstructured meshes using a high-order polynomial approximation of the velocity field is addressed. The calculation of the particle trajectory is based on a Runge-Kutta integration in time. A convenient way of implementing high-order approximations is to employ an auxiliary mapping that transforms a finite element into a topologically equivalent parent element within a normalized parametric space. This presents two possible choices of space in which to perform the time integration of the particle position: the physical space or the parametric space. We present algorithms for implementing both particle tracking strategies using high-order elements and discuss their merits. The main drawback of both methods is their reliance on nonlinear procedures to calculate the particle trajectory. A novel alternative hybrid approach that advances a particle in both the physical and the parametric space without requiring nonlinear iterations is proposed.

**[24] Yeung & Pope (1988) · An algorithm for tracking fluid particles in numerical simulations of homogeneous turbulence** · 353 citations · J. Comput. Phys.
<https://consensus.app/papers/details/7692f4fdedfe5db1b0442df2be5584ce/?utm_source=claude_desktop>
> Lagrangian statistical quantities are of fundamental physical importance in our understanding of turbulence [...]. A particle-tracking algorithm is developed to extract accurate Lagrangian statistics from numerically calculated velocity fields. Lagrangian time-series are obtained from the method of direct numerical simulation, which supplies the Eulerian velocity field on a three-dimensional grid network. The accuracy of the Lagrangian time series depends primarily on the accuracy of the interpolation scheme used to calculate fluid-particle velocities. Interpolation schemes based on Taylor series and on cubic splines have been implemented and tested. Errors in computed particle displacements are quantified for simple, frozen velocity fields. [...] **It is demonstrated that with adequate spatial resolution, accurate estimates of Lagrangian statistics [...] can be obtained either with a third-order Taylor series interpolation scheme or with a cubic spline scheme. Cubic splines give higher interpolation accuracy, but they are more difficult to implement in codes that rely on secondary storage.**

### Query · "POD reduced order model stirred tank rotor Rushton turbine mixing particle tracking"

**[25] Mikhaylov et al. (2023) · Three-dimensional characterisation of macro-instabilities in a turbulent stirred tank flow and reconstruction from sparse measurements using machine learning methods** · 5 citations · Chemical Engineering Research and Design
<https://consensus.app/papers/details/ef509961cc0a5608999805d1a476f09f/?utm_source=claude_desktop>
> We apply Proper Orthogonal Decomposition (POD) to characterise rotating, three dimensional, large scale coherent structures inside an unbaffled stirred tank agitated by a Rushton turbine at turbulent flow conditions (Re = 30000). **The four leading POD modes come in pairs**, with frequencies 0.6 and 0.2 times the impeller rotational frequency (in an inertial reference frame). Investigation of the spatial structure suggest that the two pairs correspond to precessing Macro-instabilities that rotate in a direction opposite to that of the impeller. Four Machine Learning methods are employed to reconstruct the dominant pair from sparse velocity measurements. The pair was reconstructed well by all algorithms using data from 1, 2, and 6 sensors. [...] **A reduced order model consisting of the mean and the first two modes reconstructs well the largest structures of the flow but, as expected, does not reproduce the finer features.**

**[26] Mikhaylov et al. (2021) · Reconstruction of large-scale flow structures in a stirred tank from limited sensor data** · 18 citations · AIChE Journal
<https://consensus.app/papers/details/c7337d91c01b56509c904cacb993f229/?utm_source=claude_desktop>
> We combine reduced order modeling and system identification to reconstruct the temporal evolution of large-scale vortical structures behind the blades of a Rushton impeller. We performed direct numerical simulations at Reynolds number 600 and employed proper orthogonal decomposition (POD) to extract the dominant modes and their temporal coefficients. **We then applied the identification algorithm, N4SID, to construct an estimator that captures the relation between the velocity signals at sensor points (input) and the POD coefficients (output). We show that the first pair of modes can be very well reconstructed using the velocity time signal from even a single sensor point.**

**[27] Arosemena, Battista, Solari (2023) · Proper orthogonal decomposition modal analysis in a baffled stirred tank: a base tool for the study of structures** · 2 citations · Flow
<https://consensus.app/papers/details/560762d290bd53899bd0352200d8f89d/?utm_source=claude_desktop>
> Proper orthogonal decomposition (POD) is applied to three-dimensional (3-D) velocity fields collected from large-eddy simulations (LES) of a baffled stirred tank. In the LES, the tank operates with a Rushton-type impeller under turbulent conditions [...] and the working fluid exhibits either Newtonian or shear-thinning rheology. **It is found that: (i) regardless of the working fluid rheology, it seems feasible to decompose the 3-D field into its mean, most energetic periodic and fluctuating components using POD, allowing, for instance, reduced-order modelling of the energetic periodic motions for mixing enhancement purposes, and (ii) vortical structures related to turbulence are mostly tubular.**

**[28] Jiang et al. (2023) · Reduced-order modeling of solid-liquid mixing in a stirred tank using data-driven singular value decomposition** · 18 citations · Chemical Engineering Research and Design
<https://consensus.app/papers/details/f58e418b94c05c99a99f91daa61bcf4f/?utm_source=claude_desktop>
> Stirred tanks are widely used across the (bio)chemical and process industries for solid-liquid mixing. Predicting solid suspension behavior under varying agitation speeds is critical for process control and optimization. However, inherent turbulence and multiphase interactions challenges the simulation in terms of accuracy and speed. In this work, a reduced-order model (ROM) to simulate solid-liquid flows in a stirred tank was developed, which uses singular value decomposition (SVD) to learn the flow patterns from computational fluid dynamics (CFD) simulations. **The results show that the use of the ROM can result in a reduction of computation time of up to three orders of magnitude with reasonable accuracy.**

**[29] Peace et al. (2025) · Study of the dispersed and flooded aeration regimes in two-phase gas-liquid stirred tanks using Positron Emission Particle Tracking** · 3 citations · Chemical Engineering Research and Design
<https://consensus.app/papers/details/f3539021842b5618b761b89b88cc464c/?utm_source=claude_desktop>
> This investigation employs Positron Emission Particle Tracking (PEPT) to provide insights into the hydrodynamics and mixing behaviour of two-phase gas-liquid stirred tank systems. Using a T = 0.2 m diameter tank agitated by a standard Rushton turbine (D = C = T/3), industrially relevant superficial gas velocities (U SG = 0-1.67 cm s -1) are investigated across the dispersed and flooded aeration regimes in a viscous Newtonian fluid. [...] **Additionally, a Lagrangian-based method is applied for the first time to assess the mixing performance of aerated systems**, demonstrating that an increase in gas flow rate profoundly influences the mixing dynamics, regardless of the aeration regime.

**[30] Qian, Fang, Deng, Han (2023) · Extraction of Lagrangian Coherent Structures in the framework of the Lagrangian-Eulerian Stabilized Collocation Method (LESCM)** · 14 citations · Computer Methods in Applied Mechanics and Engineering
<https://consensus.app/papers/details/29f44035f49f5b15a1bdca79a5157e25/?utm_source=claude_desktop>
> In this study, we propose a novel and accurate numerical technique based on the Lagrangian-Eulerian Stabilized Collocation Method (LESCM) for computing the Finite Time Lyapunov Exponents (FTLEs), which is essential for extracting LCSs in viscous incompressible flows. [...] **The errors in FTLEs caused by the Particle Shifting Technique (PST) are completely avoided due to the Eulerian characteristic of LESCM as the deformation gradient is calculated based on fixed Eulerian nodes rather than fluid particles with unphysical shifting.** Consequently, the novel technique based on LESCM surpasses the accuracy of pure Lagrangian particle methods and provides an accurate way of detecting complex LCSs in flow fields. Additionally, By harnessing the remarkable efficiency of LESCM, MATLAB can now handle up to 16 million particles with ease, eliminating the need for parallel computation techniques.

**[31] Lagares et al. (2023) · A GPU-Accelerated Particle Advection Methodology for 3D Lagrangian Coherent Structures in High-Speed Turbulent Boundary Layers** · 7 citations · Energies
<https://consensus.app/papers/details/e7dcf10b8c3c505b87d599d3a28c282f/?utm_source=claude_desktop>
> In this work, we introduce a scalable and efficient GPU-accelerated methodology for volumetric particle advection and finite-time Lyapunov exponent (FTLE) calculation, focusing on the analysis of Lagrangian coherent structures (LCS) in large-scale direct numerical simulation (DNS) datasets [...]. Our novel owning-cell locator method enables efficient constant-time cell search, and the algorithm draws inspiration from classical search algorithms and modern multi-level approaches in numerical linear algebra. **The proposed method is implemented for both multi-core CPUs and Nvidia GPUs, demonstrating strong scaling up to 32,768 CPU cores and up to 62 Nvidia V100 GPUs.**

**[32] Raben et al. (2013) · Computation of finite-time Lyapunov exponents from time-resolved particle image velocimetry data** · 39 citations · Experiments in Fluids
<https://consensus.app/papers/details/3eeb3a7021d15e29a6b6d7ecccff317a/?utm_source=claude_desktop>
> This work presents two new methods for computing finite-time Lyapunov exponents (FTLEs) from noisy spatiotemporally resolved experimentally measured image data of the type used for particle image velocimetry (PIV) or particle tracking velocimetry (PTV). [...] **Comparisons of the traditional velocity field integration (VFI) method for computing FTLE fields with these new methods show that FMC produces significantly more accurate estimates of the FTLE field for both synthetic data and experimental data** especially in cases where the particle number density is low. [...] When comparing the ability to match the true separatrix of a flow, FMC is shown to be a far superior method. **The separatrix from FMC has an 80 % overlap with the true solution as compared to approximately 25 % for the PFM and only 1 % for the VFI method.**

---

## Second pass (2026-07-31)

### Query · "POD reduced order model Lagrangian particle tracking mixing quality accuracy CFD-DEM Eulerian Lagrangian"

**[38] Duan et al. (2023) · Feasibility Analysis of a POD-Based Reduced Order Model with Application in Eulerian-Lagrangian Simulations** · 17 citations · Ind. & Eng. Chem. Res.
<https://consensus.app/papers/details/4f51abe86082549d8cae1f4fafb4b401/?utm_source=claude_desktop>
> Computational fluid dynamics coupled with the discrete element method (CFD-DEM) is widely employed for simulating multiphase flows involving particles, but the heavy computational cost is a major concern. Reduced order models (ROMs) based on proper orthogonal decomposition (POD) offer new potential to greatly reduce the computational cost. This study aims to investigate the feasibility of POD-based ROMs for Eulerian-Lagrangian simulations, considering the few studies in this field. **For feasibility analysis, whether the dominant POD modes are essentially similar between the training and testing data sets is a crucial condition. If this condition is not perfectly met, an inconsistent training problem easily takes place, resulting in a so-called consistency error. This error could deteriorate the predictability of a POD-based ROM.** Based on a theoretical analysis of consistency error, the most accurate solution that POD-based ROMs can produce is obtained. Furthermore, not only a new common POD-mode number between the training and testing data sets but also a novel predictability ratio are proposed for general feasibility analysis. The simulations of fluidized and spouted beds are taken as examples to show the application of the feasibility analysis. **It is found that the POD-based ROM shows poor and excellent predictability for Lagrangian and Eulerian variables, respectively.** Thus, it is suggested to map Lagrangian variables in DEM simulations on fixed Eulerian meshes to improve the predictability of the solid particle behavior in the POD-based ROM.

**[39] Shnapp et al. (2026) · Lagrangian Proper Orthogonal Decomposition**
<https://consensus.app/papers/details/f13a00a5fc8056fcbda4d7fa83fe6ef9/?utm_source=claude_desktop>
> We introduce a modal representation for Lagrangian trajectories in turbulence, termed Lagrangian Proper Orthogonal Decomposition (LPOD). An ensemble of particle trajectories is used to construct velocity time series, which are normalized independently for each trajectory to isolate fluctuations. Principal Component Analysis is then applied to the resulting dataset, with temporal instances defining the feature space. The method is tested on trajectories from both direct numerical simulations of homogeneous isotropic turbulence and three-dimensional particle-tracking experiments, showing that the leading modes exhibit similar structures and energy distributions in both cases. Truncated reconstructions are obtained by combining modes and coefficients, rescaling the fluctuations, and integrating in time. **For trajectories of the order of the integral time scale, single-particle dispersion and curvature statistics are accurately reproduced using a limited number of modes (c.a. 10), whereas capturing the tails of acceleration distributions requires a larger set (c.a. 30-60). Longer trajectories require progressively more modes for accurate reconstruction.**

**[40] Bhattacharyya et al. (2020) · An Energy Closure Criterion for Model Reduction of a Kicked Euler-Bernoulli Beam** · 5 citations · J. Vib. Acoust.
<https://consensus.app/papers/details/7b0fb54d90cc5518802798ecad7f59e9/?utm_source=claude_desktop>
> Reduced order models (ROMs) can be simulated with lower computational cost while being more amenable to theoretical analysis. Here, we examine the performance of the proper orthogonal decomposition (POD), a data-driven model reduction technique. We show that the accuracy of ROMs obtained using POD depends on the type of data used and, more crucially, on the criterion used to select the number of proper orthogonal modes (POMs) used for the model. [...] **We show that conventional variance-based mode selection leads to inaccurate models for sufficiently impulsive loading and that this poor performance is explained by the energy imbalance on the reduced subspace. Specifically, the subspace of POMs capturing a fixed amount (say, 99.9%) of the total variance underestimates the energy input and dissipated in the ROM, yielding inaccurate reduced-order simulations.** This problem becomes more acute as the loading becomes more spatio-temporally localized (more impulsive). Thus, energy closure analysis provides an improved method for generating ROMs with energetics that properly reflect that of the full system.

**[41] Brindise & Vlachos (2017) · Proper orthogonal decomposition truncation method for data denoising and order reduction** · 48 citations · Experiments in Fluids
<https://consensus.app/papers/details/6caf7607c7505528b8ac5783e42161f2/?utm_source=claude_desktop>
> Proper orthogonal decomposition (POD) is used widely in experimental fluid dynamics for reducing noise in a measured flow field. The efficacy of POD is governed by the selection of modes used for the velocity field reconstruction. Currently, the determination of which or how many modes to keep is a user-defined subjective choice, where an arbitrary amount of energy to retain in the reconstruction, such as 99% cumulative energy, is chosen. **Here, we present a novel, fully autonomous, and objective mode-selection method, which we term the entropy-line fit (ELF) method. The ELF method computes the Shannon entropy of the spatial discrete cosine transform of the eigenmodes, and using a two-line fit of the entropy mode spectrum, distinguishes between the modes carrying meaningful signal and those containing noise.**

### Query · "Lagrangian inner product POD basis matrix formulation numerical implementation weighted SVD"

**[42] Haibo Li (2023) · Generalizing the SVD of a matrix under non-standard inner product and its applications to linear ill-posed problems** · ArXiv
<https://consensus.app/papers/details/6374e2606167549d979c5840b83e020e/?utm_source=claude_desktop>
> The singular value decomposition (SVD) of a matrix is a powerful tool for many matrix computation problems. In this paper, we consider generalizing the standard SVD to analyze and compute the regularized solution of linear ill-posed problems that arise from discretizing the first kind Fredholm integral equations. For the commonly used quadrature method for discretization, a regularizer of the form ‖x‖²_M := xᵀMx should be exploited, where M is symmetric positive definite. To handle this regularizer, we give the weighted SVD (WSVD) of a matrix under the M-inner product. **Several important applications of WSVD, such as low-rank approximation and solving the least squares problems with minimum ‖·‖_M-norm, are studied. We propose the weighted Golub-Kahan bidiagonalization (WGKB) to compute several dominant WSVD components and a corresponding weighted LSQR algorithm to iteratively solve the least squares problem.**

**[43] Fareed, Singler, Zhang, Shen (2018) · A note on incremental POD algorithms for continuous time data** · 17 citations · Applied Numerical Mathematics
<https://consensus.app/papers/details/57666dd393205fffbd52138dd39b6485/?utm_source=claude_desktop>
> [...] we developed an incremental approach to compute the proper orthogonal decomposition (POD) of PDE simulation data. Specifically, we developed an incremental algorithm for the SVD with respect to a weighted inner product for the discrete time POD computations. For continuous time data, we used an approximate approach to arrive at a discrete time POD problem and then applied the incremental SVD algorithm.

**[44] Fareed et al. (2018) · Error analysis of an incremental POD algorithm for PDE simulation data** · 20 citations · J. Comput. Appl. Math.
<https://consensus.app/papers/details/d9458a08ac2350889141b903ff76e26b/?utm_source=claude_desktop>
> [...] We show the algorithm produces the exact SVD of an approximate data matrix, and the operator norm error between the approximate and exact data matrices is bounded above by the computed error bound. This error bound also allows us to bound the error in the incrementally computed singular values and singular vectors.

**[45] Alla & Kutz (2016) · Randomized model order reduction** · 33 citations · Advances in Computational Mathematics
<https://consensus.app/papers/details/595604a77a615f5782ce33b597e32425/?utm_source=claude_desktop>
> The singular value decomposition (SVD) has a crucial role in model order reduction. [...] The aim of this work is to provide an efficient computation of low-rank POD and/or DMD modes via randomized matrix decompositions. This is possible due to the randomized singular value decomposition (rSVD) which is a fast and accurate alternative of the SVD.

### Query · "divergence-free reduced basis Helmholtz-Hodge projection POD snapshots implementation velocity"

**[46] Xigui Li et al. (2026) · Project and Generate: Divergence-Free Neural Operators for Incompressible Flows** · 2 citations · ArXiv
<https://consensus.app/papers/details/df607d464f1e5e02ada8789ca8867544/?utm_source=claude_desktop>
> Learning-based models for fluid dynamics often operate in unconstrained function spaces, leading to physically inadmissible, unstable simulations. While penalty-based methods offer soft regularization, they provide no structural guarantees, resulting in spurious divergence and long-term collapse. In this work, we introduce a unified framework that enforces the incompressible continuity equation as a hard, intrinsic constraint for both deterministic and generative modeling. **First, to project deterministic models onto the divergence-free subspace, we integrate a differentiable spectral Leray projection grounded in the Helmholtz-Hodge decomposition, which restricts the regression hypothesis space to physically admissible velocity fields.** [...] Experiments on 2D Navier-Stokes equations demonstrate exact incompressibility up to discretization error and substantially improved stability and physical consistency.

**[47] Lanyu Li et al. (2024) · Error propagation of direct pressure gradient integration and a Helmholtz-Hodge decomposition-based pressure field reconstruction method for image velocimetry** · 9 citations · Experiments in Fluids
<https://consensus.app/papers/details/6de17f9b1cef587d90e4a2b9dba4d473/?utm_source=claude_desktop>
> [...] We propose to use a novel HHD-based pressure field reconstruction strategy that offers the following advantages or features: (i) effective processing of noisy scattered or structured image velocimetry data on a complex domain; (ii) **using radial basis functions (RBFs) with divergence/curl-free kernels to provide divergence-free correction to the velocity fields for incompressible flows and curl-free correction for pressure gradients**; and (iii) enforcing divergence/curl-free constraints without using Lagrangian multipliers.

**[48] Kaneko et al. (2022) · Augmented reduced order models for turbulence** · 4 citations
<https://consensus.app/papers/details/fecbf799e1445d268328912856003c45/?utm_source=claude_desktop>
> The authors introduce an augmented-basis method (ABM) to stabilize reduced-order models (ROMs) of turbulent incompressible flows. The method begins with standard basis functions derived from proper orthogonal decomposition (POD) of snapshot sets taken from a full-order model. These are then augmented with divergence-free projections of a subset of the nonlinear interaction terms that constitute a significant fraction of the time-derivative of the solution. **The augmenting bases, which are rich in localized high wavenumber content, are better able to dissipate turbulent kinetic energy than the standard POD bases. Several examples illustrate that the ABM significantly out-performs L 2-, H 1- and Leray-stabilized POD ROM approaches.**

### Query · "POD basis grid resampling versus unstructured mesh finite element snapshot interpolation quality"

**[49] Gräßle, Hinze (2017) · POD reduced-order modeling for evolution equations utilizing arbitrary finite element discretizations** · 49 citations · Advances in Computational Mathematics
<https://consensus.app/papers/details/458a3beef72f5df494a60af9c8969a46/?utm_source=claude_desktop>
> The main focus of the present work is the inclusion of spatial adaptivity for the snapshot computation in the offline phase of model order reduction utilizing proper orthogonal decomposition (POD-MOR) for nonlinear parabolic evolution problems. We consider snapshots which live in different finite element spaces, which means in a fully discrete setting that the snapshots are vectors of different length. From a numerical point of view, this leads to the problem that the usual POD procedure which utilizes a singular value decomposition of the snapshot matrix, cannot be carried out. In order to overcome this problem, we here construct the POD model/basis using the eigensystem of the correlation matrix (snapshot Gramian), which is motivated from a continuous perspective and is set up explicitly, e.g., **without the necessity of interpolating snapshots into a common finite element space.**

**[50] Ullmann, Lang, Rotkvic (2016) · POD-Galerkin reduced-order modeling with adaptive finite element snapshots** · 68 citations · J. Comput. Phys.
<https://consensus.app/papers/details/6f27cb7a4911532ba1942a43bb8b3190/?utm_source=claude_desktop>
> We consider model order reduction by proper orthogonal decomposition (POD) for parametrized partial differential equations, where the underlying snapshots are computed with adaptive finite elements. [...] We propose a method to create a POD-Galerkin model without interpolating the snapshots onto their common finite element mesh.

**[51] Nakamura, Yano (2024) · Application of proper orthogonal decomposition to flow fields around various geometries and reduced-order modeling** · 21 citations · CMAME
<https://consensus.app/papers/details/33c89c61ddd55b6c928769396627ab59/?utm_source=claude_desktop>
> This study is focused on a reduced-order model (ROM) based on proper orthogonal decomposition (POD) for unsteady flow around a stationary object, which allows prediction with different object geometry as a parameter. The conventional POD method is applicable only to data with the same computational grid for all snapshots. **This study proposed a novel POD methodology that performs on flow snapshots, including some time-series data of flow fields around objects of different shapes and numerically computed by different computational grids.** [...] The mean squared error between the flow fields obtained via the ROM and the directly solved Navier-Stokes equations was under 10⁻⁷ when the reconstructed flow and the flow included in the snapshot had the same frequency as that of Kármán vorticities behind the objects.

**[52] Gooijer, Havinga, Boogaard (2021) · Evaluation of POD based surrogate models of fields resulting from nonlinear FEM simulations** · 25 citations · Advanced Modeling and Simulation in Engineering Sciences
<https://consensus.app/papers/details/5118da687c555d5188fcf14afb9ef0ce/?utm_source=claude_desktop>
> POD-based surrogate models with Radial Basis Function interpolation are used to model high-dimensional FE data fields. The effect of (pre)processing methods on the accuracy of the result field is systematically investigated. [...] Special attention is given to data fields consisting of several physical meanings, e.g. displacement, strain and stress. A distinction is made between the errors due to truncation and due to interpolation of the data. **It is found that scaling the data per physical part substantially increases the accuracy of the surrogate model.**

### Query · "adaptive mesh refinement Cartesian grid derived unstructured tet element size mapping octree"

**[34] Fidkowski (2020) · Metric-based, goal-oriented mesh adaptation using machine learning** · 39 citations · J. Comput. Phys.
<https://consensus.app/papers/details/c030727c1f70512c9b7a7509757c1791/?utm_source=claude_desktop>
> This paper presents a machine-learning approach for determining the optimal anisotropy in a computational mesh, in the context of an output-based adaptive solution procedure. Artificial neural networks are used to predict the desired element aspect ratio from readily accessible features of the primal and adjoint solutions. Whereas the sizing of the element is still based on an adjoint-weighted residual error estimate, the network augments this information with element stretching magnitude and direction.

**[35] Balan et al. (2021) · A review and comparison of error estimators for anisotropic mesh adaptation for flow simulations** · 22 citations · Computers & Fluids
<https://consensus.app/papers/details/d4b2a597c55a50089c493338e4c8595a/?utm_source=claude_desktop>
> The current work aims to review the various error estimators and the corresponding metric fields available for anisotropic mesh adaptation, and compare their mesh convergence behavior for various flow problems. [...] All the metric fields considered in this work are implemented in NASA's open-source grid mechanics package, refine, and FUN3D-SFE.

**[36] Frey (2005) · Anisotropic mesh adaptation for CFD computations** · 386 citations · CMAME
<https://consensus.app/papers/details/d0f5f415917d5e51b1232f9243ee48f9/?utm_source=claude_desktop>
> Unstructured mesh adaptation is now widely used in numerical simulations to improve the accuracy of the solutions as well as to capture the behavior of physical phenomena. In this paper, we propose a general purpose error estimate based on the interpolation error that produces an anisotropic metric map used to govern the mesh element creation.

**[37] Prouvost, Popinet, Basilisk (2024) · A metric-based adaptive mesh refinement criterion under constrain for solving elliptic problems on quad/octree grids** · 3 citations · J. Comput. Phys.
<https://consensus.app/papers/details/2b2628ea29215001a4c9ca3c966df25d/?utm_source=claude_desktop>
> We show that in general, when solving elliptic equations such as the Poisson-Helmholtz equation, the minimization of the interpolation error often used as local refinement criteria does not always guarantee the minimization of the total numerical error. Numerical and theoretical arguments are given to unveil the critical role of **the mesh compression - the size aspect ratio between the finest cell size and the mean cell size of an adapted mesh - to determine whether the estimated error is purely local meaning that the interpolation error is a good enough error model for the total error or if other, non-local, sources of error need to be accounted for.**

### Query · "JAX GPU proper orthogonal decomposition SVD Galerkin XLA differentiable fluid simulation"

**[53] Bezgin, Buhendwa, Adams (2024) · JAX-Fluids 2.0: Towards HPC for Differentiable CFD of Compressible Two-phase Flows** · 43 citations · Comput. Phys. Commun.
<https://consensus.app/papers/details/1d81628aa705532887e26df45a353e37/?utm_source=claude_desktop>
> JAX-Fluids is a Python-based fully-differentiable CFD solver designed for compressible single- and two-phase flows. In this work, the first version is extended to incorporate high-performance computing (HPC) capabilities. **We introduce a parallelization strategy utilizing JAX primitive operations that scales efficiently on GPU (up to 512 NVIDIA A100 graphics cards) and TPU (up to 1024 TPU v3 cores) HPC systems.** We further demonstrate the stable parallel computation of automatic differentiation gradients across extended integration trajectories.

### Query · "JAX automatic differentiation particle tracking Lagrangian machine learning velocity field neural"

**[54] Du et al. (2025) · JAX-MPM: a learning-augmented differentiable meshfree framework for GPU-accelerated Lagrangian simulation and geophysical inverse modeling** · 4 citations · Engineering with Computers
<https://consensus.app/papers/details/e57ecc89754f5a959b002610fb4b11cb/?utm_source=claude_desktop>
> Differentiable programming has emerged as a powerful paradigm in scientific computing, enabling automatic differentiation through simulation pipelines and naturally supporting both forward and inverse modeling. We present JAX-MPM, a general-purpose differentiable meshfree solver based on the material point method (MPM) and implemented in the modern JAX ecosystem. **Results show that a high-resolution 3D granular cylinder collapse with 2.7 million particles completes 1000 time steps in approximately 22 s (single precision) and 98 s (double precision) on a single GPU.** Beyond high-fidelity forward modeling, we demonstrate the framework's inverse modeling capabilities through tasks such as velocity field reconstruction and the estimation of spatially varying friction from sparse data.

**[55] Pradhan et al. (2025) · JAX-LaB: A High-Performance, Differentiable Lattice Boltzmann Library for Modeling Multiphase Fluid Dynamics in Geosciences and Engineering** · JAMES
<https://consensus.app/papers/details/589f34263d8251919c13070eb6f7d66b/?utm_source=claude_desktop>
> We introduce JAX-LaB, a differentiable, Python-based Lattice Boltzmann simulation library designed for modeling multiphase and multiphysics fluid dynamics problems in hydrologic, geologic, and engineered porous media settings. The library is designed as an extension to XLB, and it is built on the JAX framework. The library delivers a performant, hardware-agnostic implementation that seamlessly integrates with machine learning libraries and scales efficiently across CPUs, multi-GPU setups, and distributed environments.

**[56] Fan et al. (2025) · Diff-FlowFSI: A GPU-Optimized Differentiable CFD Platform for High-Fidelity Turbulence and FSI Simulations** · 14 citations
<https://consensus.app/papers/details/734406c742205e3fa822b79f128dacc0/?utm_source=claude_desktop>
> In this work, we introduce Diff-FlowFSI, a GPU-accelerated, fully differentiable CFD platform designed for high-fidelity turbulence and FSI simulations. Implemented in JAX, Diff-FlowFSI features a vectorized finite volume solver combined with the immersed boundary method to handle complex geometries and fluid-structure coupling.

**[57] Zhang (2026) · A differentiable, shock-capturing neural solver for compressible flow simulation** · 1 citation · Physics of Fluids
<https://consensus.app/papers/details/fe7aff8de88454fab5635c4bd395f9f1/?utm_source=claude_desktop>
> We present JAX-Shock: a fully-differentiable, GPU-accelerated, high-order shock-capturing solver for efficient simulation of the compressible Navier-Stokes equations. Built entirely in JAX, the framework leverages automatic differentiation to enable gradient-based optimization, parameter inference, and end-to-end training of deep learning-augmented models.

### Query · "GPU particle tracking Runge-Kutta CUDA parallel Lagrangian large scale computational"

**[58] Yang et al. (2021) · Accelerating the Lagrangian particle tracking of residence time distributions and source water mixing towards large scales** · 17 citations · Comput. Geosci.
<https://consensus.app/papers/details/08558f631d8f5ca78cb42b674b34793b/?utm_source=claude_desktop>
> In this study, we accelerate the Lagrangian particle tracking program EcoSLIM, using a combination of distributed (e.g. MPI) and multi-core accelerator (CUDA) approaches for large-scale and long-term simulations. **Of these combinations, the OpenMP-CUDA parallelism performed the best moving from single-GPU to multi-GPU. The multi-GPU shows strong scalability which becomes increasingly efficient with more particles.**

**[59] Wang et al. (2021) · An GPU-accelerated particle tracking method for Eulerian-Lagrangian simulations using hardware ray tracing cores** · 27 citations · Comput. Phys. Commun.
<https://consensus.app/papers/details/3fab5de2510751019235f94930dd8e8b/?utm_source=claude_desktop>
> To address the high computational cost of particle tracking for realistic Eulerian-Lagrangian simulations, a novel efficient and robust particle tracking method (RT method) for unstructured meshes is presented. **The method, for the first time, leverages both hardware ray tracing (RT) cores and GPU parallel computing technology to accelerate Eulerian-Lagrangian simulations. The method includes a hardware-accelerated hosting cell locator using bounding volume hierarchy tree (BVH).** [...] Benchmark results indicate that our RT method leads to a roughly 1.8-2.0× performance improvement compared to the reference NS method for large-scale simulations.

**[60] Schmalfuss et al. (2026) · SCALE-TRACK: Asynchronous Euler-Lagrange particle tracking on heterogeneous computing architecture** · ArXiv
<https://consensus.app/papers/details/f32db0a2c7d05277a12e67b5eccf18d7/?utm_source=claude_desktop>
> We present SCALE-TRACK, a scalable two-way coupled EL particle tracking algorithm, designed to exploit heterogeneous exascale computing environments. With asynchronous coupling, cache-friendly data structures, and chunk-based partitioning, we address key limitations of existing EL implementations. **On a local workstation, we simulated 1.4 billion particles in a test case featuring a single graphics processing unit (GPU). Scaling runs on an HPC cluster show excellent strong and weak scaling, with up to 256 billion particles being tracked on up to 256 GPUs.**

**[61] Suriano et al. (2026) · The PLUTO code on GPUs: Offloading Lagrangian Particle methods** · 2 citations · Astron. Comput.
<https://consensus.app/papers/details/eaa789ddc9445bd4b7076d6f60d544e4/?utm_source=claude_desktop>
> We present a GPU-compatible C++ re-design of the Lagrangian Particles (LP) module of the PLUTO code, that by means of the programming model OpenACC and the Message Passing Interface library, is capable of targeting both single commercial GPUs as well as multi-node (pre-)exascale computing facilities. **The code has been benchmarked up to 28672 parallel CPUs cores and 1024 parallel GPUs demonstrating ~(80-90)% weak scaling parallel efficiency and good strong scaling capabilities. Our results demonstrated a speedup of 6 times when solving that same benchmark test with 128 full GPU nodes against the same amount of full high-end CPU nodes.**

**[62] Baldan et al. (2023) · Efficient Lagrangian particle tracking algorithms for distributed-memory architectures** · 10 citations · Computers & Fluids
<https://consensus.app/papers/details/a4c6c9181bc859ee8d83b18074603c59/?utm_source=claude_desktop>
> This paper focuses on the solution of the dispersed phase of Eulerian-Lagrangian one-way coupled particle laden flows. An efficient two-constraint domain partitioning for 2D and 3D unstructured hybrid meshes is proposed and implemented in distributed memory architectures. **In addition, an innovative parallel ray-tracing location algorithm is presented. A global identifier is assigned to each particle resulting in a significant reduction of the overall communication among processes.**

**[63] Zhao, Chen (2022) · Leveraging ray tracing cores for particle-based simulations on GPUs** · 37 citations · IJNME
<https://consensus.app/papers/details/5e496117fd9855998b97b79bafd711d4/?utm_source=claude_desktop>
> This article presents a novel approach to accelerate particle-based simulations by leveraging ray tracing (RT) cores in addition to CUDA cores on RTX GPUs. **The neighbor search problem is first numerically converted into a general ray tracing problem so that it can be possible to utilize the hardware acceleration of RT cores.** [...] It demonstrates that the RT-based simulations are 10%-60% faster than the cell-based ones, depending on the simulated problems and GPU specs.

### Query · "adaptive Runge-Kutta Dormand-Prince high order embedded step size control shear flow accuracy"

**[64] Ranocha, Dalcin, Parsani, Ketcheson (2021) · Optimized Runge-Kutta Methods with Automatic Step Size Control for Compressible Computational Fluid Dynamics** · 67 citations · Communications on Applied Mathematics and Computation
<https://consensus.app/papers/details/08afc433407d598b9706403fdf1cdda9/?utm_source=claude_desktop>
> We develop error-control based time integration algorithms for compressible fluid dynamics (CFD) applications and show that they are efficient and robust in both the accuracy-limited and stability-limited regime. Focusing on discontinuous spectral element semidiscretizations, we design new controllers for existing methods and for some new embedded Runge-Kutta pairs. **The optimized methods give improved performance and naturally adopt a step size close to the maximum stable CFL number at loose tolerances, while additionally providing control of the temporal error at tighter tolerances.**

**[65] Vermeire (2023) · Embedded paired explicit Runge-Kutta schemes** · 10 citations · J. Comput. Phys.
<https://consensus.app/papers/details/7c80b4bff3b758a894c2cb37b9dcd993/?utm_source=claude_desktop>
> Paired Explicit Runge-Kutta (P-ERK) schemes use different numbers of active stages based on local stiffness criteria, significantly reducing computational cost relative to classical explicit Runge-Kutta schemes. [...] It is demonstrated that adaptive time stepping using embedded P-ERK schemes yields excellent agreement with reference data, **while being up to seven times less computationally expensive than classical embedded pairs for all cases.**

### Query · "volume-preserving symplectic integrator incompressible flow particle tracking divergence free trajectory"

**[66] Tapley (2019) · Computing cost-effective particle trajectories in numerically calculated incompressible fluids using geometric methods** · 2 citations · arXiv: Computational Physics
<https://consensus.app/papers/details/a0f0ffede1dd54ed820f77a163eda37a/?utm_source=claude_desktop>
> We present an novel algorithm for tracking massless solid particles in a divergence-free velocity field that is only available at discrete points in space and time such as those arising from a direct numerical simulation of Navier-Stokes. **The algorithm creates a divergence-free approximation to the numerical field using matrix valued radial basis functions, which is integrated in time using a volume-preserving map.** The resulting method is able to calculate accurate trajectories in a helical vortex using much larger step-sizes and a far lower number of interpolation points which results in a more efficient algorithm compared to a conventional scheme.

**[67] Quispel (1995) · Volume-preserving integrators** · 62 citations · Physics Letters A
<https://consensus.app/papers/details/bf64f5faf7e15234bea91a0afbdb190f/?utm_source=claude_desktop>
> We obtain a novel family of general n-dimensional volume-preserving integrators which can be used to numerically integrate divergence free vector fields.

**[68] Kato, Zenitani (2021) · Volume-preserving particle integrator based on exact flow of velocity for nonrelativistic particle-in-cell simulations** · 1 citation
<https://consensus.app/papers/details/08ab3190648f5aa8b6ac0642177d7083/?utm_source=claude_desktop>
Volume-preserving PIC integrator based on exact flow of velocity.

**[69] Wang et al. (2019) · Volume-preserving exponential integrators and their applications** · 9 citations · J. Comput. Phys.
<https://consensus.app/papers/details/c95220f85d875db6ae214c72e7efe080/?utm_source=claude_desktop>
> This paper studies the volume-preserving property of exponential integrators for different vector fields. For exponential integrators, we first derive a necessary and sufficient condition of volume preservation. Then based on this condition, volume-preserving exponential integrators are discussed in detail for four kinds of vector fields.

**[70] He et al. (2015) · Volume-preserving algorithms for charged particle dynamics** · 124 citations · J. Comput. Phys.
<https://consensus.app/papers/details/199ca8e0ec8450f1af393ab7a8cc14fa/?utm_source=claude_desktop>
> The paper reports the development of volume-preserving algorithms using the splitting technique for charged particle motion under the Lorentz force. [...] **This new class of numerical methods, which includes the Boris algorithm as a special case, conserves phase space volume, and globally bounds the numerical errors of energy, momentum, and other adiabatic invariants up to the order of the method over a very long simulation time.**

**[71] Qin et al. (2013) · Why is Boris algorithm so good** · 288 citations · Physics of Plasmas
<https://consensus.app/papers/details/a9fae08d45ab5c2d88caae3879bdcc25/?utm_source=claude_desktop>
> Due to its excellent long term accuracy, the Boris algorithm is the de facto standard for advancing a charged particle. Despite its popularity, up to now there has been no convincing explanation why the Boris algorithm has this advantageous feature. In this paper, we provide an answer to this question. **We show that the Boris algorithm conserves phase space volume, even though it is not symplectic. The global bound on energy error typically associated with symplectic algorithms still holds for the Boris algorithm, making it an effective algorithm for the multi-scale dynamics of plasmas.**

**[72] Gorges et al. (2022) · Reducing volume and shape errors in front tracking by divergence-preserving velocity interpolation and parabolic fit vertex positioning** · 9 citations · J. Comput. Phys.
<https://consensus.app/papers/details/ccd6df2c728f56e7aacdfc98cd8719e4/?utm_source=claude_desktop>
> [...] Errors in preserving the divergence of the velocity field when interpolating the velocity from the fluid mesh to the vertices of the triangles of the front are a primary reason for volume conservation errors when advecting the front. **The proposed interpolation method preserves the discrete divergence of the fluid velocity by construction** [...]. The presented interpolation method conserves the volume and shape up to an order of magnitude better than the conventionally used interpolation methods.

### Query · "friction stir welding material flow tracer particle mixing quantification experimental validation"

**[73] Dialami, Chiumenti, Cervera, Agelet de Saracibar (2015) · Material flow visualization in Friction Stir Welding via particle tracing** · 56 citations · Int. J. Material Forming
<https://consensus.app/papers/details/63f6dceafc305f34a682e161da78a163/?utm_source=claude_desktop>
> This work deals with the modeling of the material flow in Friction Stir Welding (FSW) processes using particle tracing method. **For the computation of particle trajectories, three accurate and computationally efficient integration methods are implemented within a FE model for FSW process: the Backward Euler with Sub-stepping (BES), the 4-th order Runge-Kutta (RK4) and the Back and Forth Error Compensation and Correction (BFECC) methods.**

**[74] Dialami et al. (2020) · Defect formation and material flow in Friction Stir Welding** · 122 citations · Eur. J. Mech. A/Solids
<https://consensus.app/papers/details/e927c4fb7b8c5da79fab5769d19b406b/?utm_source=claude_desktop>
> This work addresses the issue of the simulation and prediction of defect formation through the analysis of the material mixing during Friction Stir Welding (FSW). [...] **The model is capable of predicting defects such as void, wormhole, flash and joint line remnant, as well as the formation of "onion rings" in a single simulation.**

**[75] Kumar, Singh, Pandey (2018) · Material flow visualization and determination of strain rate during friction stir welding** · 75 citations · J. Mater. Process. Technol.
<https://consensus.app/papers/details/6c4d671a22bd5dfc9f8b6c2381fe1c11/?utm_source=claude_desktop>
> Particle image velocimetry (PIV) technique was adopted to understand material flow and measure strain rate around the tool pin during friction stir welding (FSW). **The maximum velocity was noted to be 60% (close to the pin surface) of the pin peripheral velocity, and strain-rate was found to be 20 s⁻¹ (0.6 mm away from the pin periphery) at FSW parameters of 170 rpm and 50 mm min⁻¹. The strain rate was found to increase from 8 s⁻¹ to 44 s⁻¹ with increase in rotational speed from 75 rpm to 425 rpm.**

**[76] Ambrosio, Dessein, Bourrat (2023) · Material flow in friction stir welding: A review** · 102 citations · J. Mater. Process. Technol.
<https://consensus.app/papers/details/5f4d21c9977451ab88b0beafad911392/?utm_source=claude_desktop>
> The solid-state welding technology known as friction stir welding (FSW) is slowly replacing melting-based welding technologies. [...] critical issues such as the role of probe and shoulder, the mechanisms governing the process, i.e. stirring and/or extrusion, and the contact state between the tool and workpiece, i.e., sliding and/or sticking, are challenging to study. Here, experimental attempts carried out by the FSW community since its patent to clarify the material flow are critically reviewed.

**[77] Stubblefield et al. (2023) · A computational and experimental approach to understanding material flow behavior during additive friction stir deposition (AFSD)** · 33 citations · Computational Particle Mechanics
<https://consensus.app/papers/details/02d1c29cee355d329fd76d3a868b9082/?utm_source=claude_desktop>
> In this study, a combined computational and experimental particle tracking investigation was performed for a solid-state additive manufacturing and repair process, Additive Friction Stir Deposition (AFSD). Specifically, smoothed particle hydrodynamics (SPH) simulations of AFSD were conducted in-order to elucidate deposition mechanics. **The particle tracking of the SPH AFSD simulations was validated using experimental depositions of two feedstock varieties, including anodized AA6061-T6 feedstock to track external particles and AA6061-T6 copper wire core feedstock to track internal particles.**

**[78] Chen et al. (2021) · Study on in-situ material flow behaviour during friction stir welding via a novel material tracing technology** · 55 citations · J. Mater. Process. Technol.
<https://consensus.app/papers/details/d41cdb311f515241917d22334a59f9a1/?utm_source=claude_desktop>
> A novel material tracing technology was proposed using ER2319 aluminium alloy welding wire as the tracer material to study the material flow behaviour at different positions of friction stir welded 6061-T6 aluminium alloy joint. **The advancing side (AS) and the retreating side (RS) of the SZ are dominated by the shear action and the extrusion action, respectively. The extrusion force and friction force are the driving forces for material flow during friction stir welding.**

### Query · "friction stir mixing index Poincare map onion rings quantitative validation experimental tracer"

**[79] Krishnan (2002) · On the formation of onion rings in friction stir welds** · 485 citations · Materials Science and Engineering A
<https://consensus.app/papers/details/07fc2b574ef05297aeb3809cc39270a6/?utm_source=claude_desktop>
> Onion rings are the most prominent features of most friction stir welds. The origin and the effect of these on properties are not clearly understood. **In this paper, an attempt has been made to explain the formation of onion rings. The formation of onion ring is found to be a geometric effect due to the fact that cylindrical sheets of material are extruded during each rotation of the tool and the cutting through the section of the material produces an apparent 'Onion Rings'. [...] The spacing of the markings has been found to be equal to the forward motion of the tool in one rotation.**

**[80] Shuo Li et al. (2021) · POD-based identification approach for powder mixing mechanism in Eulerian-Lagrangian simulations** · 19 citations · Advanced Powder Technology
<https://consensus.app/papers/details/d2ab90ba44da50c9b0a54bba3ae4e070/?utm_source=claude_desktop>
> The discrete element method (DEM) coupled with computational fluid dynamics (CFD) is employed to simulate powder mixing. [...] **The results show that the mixing mechanism is dominated by convection in the early stage and by diffusion in the late stage. Besides, a novel mixing identification technique is established by giving the relation between POD modes and mixing mechanisms, namely, clumped and random spatial distributions of the POD modes appear in convective and diffusive mixing, respectively.**

**[81] Gai Zhang et al. (2024) · New metrics for measuring 2D uniformity in stirring system based on reconstruction of the particle trajectory** · 1 citation · Chemical Engineering Research and Design
<https://consensus.app/papers/details/b07a133490a25157b9939440d534e28a/?utm_source=claude_desktop>
> This study proposes an assessment method that combines a method for positioning using dual cameras and a point pattern density fluctuation (PD) method based on disordered hyperuniformity. [...] **A relationship model between λ and mixing time was established, with a Pearson correlation coefficient (r) of −0.9867, indicating a strong negative linear correlation.**

### Query · "POD ROM two-stage surrogate parametric flow field prediction operating parameters mean velocity"

**[90] Min et al. (2024) · Flow fields prediction for data-driven model of parallel twin cylinders based on POD-RBFNN and POD-BPNN surrogate models** · 48 citations · Annals of Nuclear Energy
<https://consensus.app/papers/details/bca3d282463956fda8e4855eaad0f3f1/?utm_source=claude_desktop>
> The POD-RBFNN surrogate model uses the Radial Basis Function Neural Network (RBFNN) to train the POD mode coefficients obtained from the POD algorithm, while the POD-BPNN surrogate model uses the Backpropagation Neural Network (BPNN) for the same purpose. **It is found that both the POD-RBFNN and POD-BPNN surrogate models proposed in this paper not only significantly improve efficiency but also maintain a high level of accuracy. However, the training time of the POD-RBFNN surrogate model is significantly shorter than that of the POD-BPNN surrogate model. Additionally, the POD-RBFNN surrogate model exhibits smaller Root Mean Square Errors (RMSE) and Mean Absolute Error (MAE).**

**[85] Ding et al. (2024) · Data-driven surrogate modeling and optimization of supercritical jet into supersonic crossflow** · 8 citations · Chinese Journal of Aeronautics
<https://consensus.app/papers/details/73eba141709b5be4ab4e0f582092e552/?utm_source=claude_desktop>
> This study introduces parametric Reduced-Order Models (ROMs) based on Convolutional AutoEncoders (CAE), Fully Connected AutoEncoders (FCAE), and Proper Orthogonal Decomposition (POD) to fast emulate spatial distributions of physical variables for a supercritical jet into a supersonic crossflow under different operating conditions. **Results indicate that CAE-based ROMs exhibit superior prediction accuracy while FCAE-based ROMs show inferior predictive accuracy but minimal uncertainty. [...] POD-based ROMs underperform in regions of strong nonlinear flow dynamics, coupled with higher overall prediction uncertainties.** Both AE- and POD-based ROMs achieve online predictions approximately 9 orders of magnitude faster than conventional simulations.

**[91] Yang et al. (2020) · POD-based surrogate modeling of transitional flows using an adaptive sampling in Gaussian process** · 14 citations · Int. J. Heat Fluid Flow
<https://consensus.app/papers/details/879ec1ea771c5fce836bf4801b4f835f/?utm_source=claude_desktop>
> A surrogate model, based on proper orthogonal decomposition (POD) with the adaptive sampling method, was proposed to predict the transitional flow past rough flat plates. Gaussian process regression was used to map the input parameters to the POD expansion coefficients. **The variance and gradient of Gaussian process were taken as the criteria for the adaptive sampling.**

### Query · "non-intrusive reduced order model radial basis interpolation Gaussian process POD coefficients"

**[87] Hesthaven, Ubbiali (2018) · Non-intrusive reduced order modeling of nonlinear problems using neural networks** · 628 citations · J. Comput. Phys.
<https://consensus.app/papers/details/f62e99b5db3a55c4bba87b5acb99621c/?utm_source=claude_desktop>
> We develop a non-intrusive reduced basis (RB) method for parametrized steady-state partial differential equations (PDEs). The method extracts a reduced basis from a collection of high-fidelity solutions via a proper orthogonal decomposition (POD) and employs artificial neural networks (ANNs), particularly multi-layer perceptrons (MLPs), to accurately approximate the coefficients of the reduced model.

**[88] Guo, Hesthaven (2018) · Reduced order modeling for nonlinear structural analysis using Gaussian process regression** · 259 citations · CMAME
<https://consensus.app/papers/details/9b897d6fab6f569f8f189a496ec4db24/?utm_source=claude_desktop>
> A non-intrusive reduced basis (RB) method is proposed for parametrized nonlinear structural analysis undergoing large deformations and with elasto-plastic constitutive relations. In this method, a reduced basis is constructed from a set of full-order snapshots by the proper orthogonal decomposition (POD), and the Gaussian process regression (GPR) is used to approximate the projection coefficients.

**[89] Xiao et al. (2015) · Non-intrusive reduced-order modelling of the Navier-Stokes equations based on RBF interpolation** · 183 citations · IJNMF
<https://consensus.app/papers/details/d0ba90b79ba95157b8b06f90fea9554a/?utm_source=claude_desktop>
> We present a new non-intrusive model reduction method for the Navier-Stokes equations. The method replaces the traditional approach of projecting the equations onto the reduced space with a radial basis function (RBF) multi-dimensional interpolation.

### Query · "POD autoencoder neural network incompressible flow prediction accuracy compared linear reduced order"

**[82] Ahmed et al. (2021) · Nonlinear proper orthogonal decomposition for convection-dominated flows** · 48 citations · ArXiv
<https://consensus.app/papers/details/cb438c3ef5555bada27b542c14f343ab/?utm_source=claude_desktop>
> In this letter, we put forth a nonlinear proper orthogonal decomposition (POD) framework, which is an end-to-end Galerkin-free model combining autoencoders with long short-term memory networks for dynamics. By eliminating the projection error due to the truncation of Galerkin models, a key enabler of the proposed nonintrusive approach is the kinematic construction of a nonlinear mapping between the full-rank expansion of the POD coefficients and the latent space where the dynamics evolve.

**[83] Grimberg, Farhat, Youkilis (2020) · On the stability of projection-based model order reduction for convection-dominated laminar and turbulent flows** · 108 citations · J. Comput. Phys.
<https://consensus.app/papers/details/ea46855961585636aa5ff41659d37398/?utm_source=claude_desktop>
> **This paper argues in an orderly manner that the real culprit behind most if not all reported numerical instabilities of PROMs for turbulence and convection-dominated turbulent flow problems is the Galerkin framework that has been used for constructing the PROMs. The paper also shows that alternatively, a Petrov-Galerkin framework can be used to construct numerically stable and accurate PROMs for convection-dominated laminar as well as turbulent flow problems, without resorting to additional closure models or tailoring of the subspace of approximation.**

**[84] Fu et al. (2023) · A non-linear non-intrusive reduced order model of fluid flow by auto-encoder and self-attention deep learning methods** · 57 citations · IJNME
<https://consensus.app/papers/details/ec888655fe48533d8b04a9392ec7451d/?utm_source=claude_desktop>
> This paper presents a new nonlinear non-intrusive reduced-order model (NL-NIROM) that outperforms traditional proper orthogonal decomposition (POD)-based reduced order model (ROM). This improvement is achieved through the use of auto-encoder (AE) and self-attention based deep learning methods.

**[86] Zhu et al. (2024) · Compressed neural networks for reduced order modeling** · 4 citations · Physics of Fluids
<https://consensus.app/papers/details/8ec3e2f156fd54709761bfdd1590a236/?utm_source=claude_desktop>
> The present work aims at compressing the autoencoder model via two distinctively different approaches, i.e., pruning and singular value decomposition (SVD). [...] **It is shown that pruning and SVD reduce the size of the autoencoder network to 6% and 3% for the two simple laminar cases (or 18% and 13%, 20%, and 10% for the two complex turbulent channel flow cases), respectively, with approximately the same order of accuracy.**

### Query · "POD ROM parametric surrogate friction stir welding heat flow neural network Gaussian process" (partial duplicate — mostly already captured; new hit only)

**[13-second-copy] Cao et al. (2021)** — already in first-pass block above.

### Query · "POD reduced order model Lagrangian particle tracking mixing quality accuracy CFD-DEM Eulerian Lagrangian" · additional hits

**[93] Fang et al. (2025) · Proper Orthogonal Decomposition-based Model-Order Reduction for Smoothed Particle Hydrodynamics Simulation** · 2 citations
<https://consensus.app/papers/details/2595d5fd121757fd90dfd9ea1055f299/?utm_source=claude_desktop>
> In this paper, we present a projection-based model-order reduction (MOR) technique for smoothed particle hydrodynamics (SPH) simulations, which is a mesh-free approach within the Lagrangian framework. **To illustrate the effectiveness of this approach, we consider the friction stir spot welding problem, which involves the coupling of flow equations and heat equation. Our findings reveal that, with the same degrees of freedom, POD-MOR significantly reduces computational error compared to the uniform reduction of particle numbers in SPH simulations.**

**[94] S. Li et al. (2022) · Development of a reduced-order model for large-scale Eulerian-Lagrangian simulations** · 41 citations · Advanced Powder Technology
<https://consensus.app/papers/details/4bc7651b71d45da8b8924b7292516254/?utm_source=claude_desktop>
> We propose a nonintrusive reduced-order model for Eulerian-Lagrangian simulations (ROM-EL) to efficiently reproduce gas-solid flow in fluidized beds. In the model, a Lanczos based proper orthogonal decomposition (LPOD) is newly employed to efficiently generate a set of POD bases. [...] **The macroscopic properties, such as the particle distribution, bed height, pressure drop, and distribution of bubble size, are shown to agree well in the CFD-DEM model and ROM-EL. Further, our proposed ROM-EL reduces the computational cost by several orders of magnitude compared with the CFD-DEM simulation.**

**[95] Razavi et al. (2026) · Nonintrusive Model Order Reduction Theory in a Fixed Size Domain with Particle in Fluid Flow Application** · Science Discovery Energy
<https://consensus.app/papers/details/7aaeff23edfc55778435b1bb3d4ea71f/?utm_source=claude_desktop>
> This work introduces a new nonintrusive Reduced Order Modeling (ROM) strategy that integrates Proper Orthogonal Decomposition (POD) with a previously developed nonintrusive model-order reduction framework to achieve substantial computational acceleration while preserving the fidelity of CFD-DEM simulations. [...] **Predictions for a new operating condition show excellent agreement between the ROM-ROM model, nonintrusive full-space predictions, and the full CFD-DEM solution. Performance analysis demonstrates that the ROM-ROM approach is approximately 3×10⁵ times faster than the full CFD-DEM simulation and about 40 times faster than the nonintrusive full-space method.**

**[96] Hijazi et al. (2019) · Data-Driven POD-Galerkin Reduced Order Model for Turbulent Flows** · 207 citations · J. Comput. Phys.
<https://consensus.app/papers/details/d0702962f96d5e3bb9c2767fefb10921/?utm_source=claude_desktop>
> In this work we present a Reduced Order Model which is specifically designed to deal with turbulent flows in a finite volume setting. The method used to build the reduced order model is based on the idea of merging/combining projection-based techniques with data-driven reduction strategies. In particular, the work presents a mixed strategy that exploits a data-driven reduction method to approximate the eddy viscosity solution manifold and a classical POD-Galerkin projection approach for the velocity and the pressure fields, respectively. **The newly proposed reduced order model has been validated on benchmark test cases in both steady and unsteady settings with Reynolds up to Re=O(10⁵).**

### Query · "Boris pusher volume preserving charged particle vector field advection interpolation accuracy"

Papers [70], [71] already archived above.

Additional relevant:

**[97] Higuera, Cary (2017) · Structure-Preserving Second-Order Integration Of Relativistic Charged Particle Trajectories In Electromagnetic Fields** · 91 citations · IEEE ICOPS
<https://consensus.app/papers/details/73ead0d2f7b1533a805c611d890b9476/?utm_source=claude_desktop>
> Time-centered, hence second-order, methods for integrating the relativistic momentum of charged particles in an electromagnetic field are derived. [...] **The Boris method and the current method are volume-preserving, while the method of Vay and the current method preserve the E×B velocity. Thus, of these second-order relativistic momentum integrations, only the integrator introduced here both preserves volume and gives the correct E×B velocity.**

### Query · "POD autoencoder neural network incompressible flow prediction accuracy compared linear reduced order"

**[98] Nakamura, Yano (2026) · Improvement of Reduced-Order Model for Two-Dimensional Cylinder Flow Based on Global Proper Orthogonal Decomposition in terms of Robustness and Computational Speed** · J. Fluids Eng.
<https://consensus.app/papers/details/022499c90b40566e8cd7a5083ccfbd77/?utm_source=claude_desktop>
> In this study, we propose a ROM framework that achieves fast and robust flow prediction even when the dataset contains a large number of flow conditions. The proposed approach employs a novel two-step order-reduction strategy based on POD. In the second reduction step, flow conditions that are most relevant to the target prediction are selectively retained, thereby reducing the computational cost without sacrificing accuracy. [...] **Furthermore, the proposed ROM reduces the computational cost by approximately 50% compared with a conventional POD-based ROM constructed using flow data at 27 different Reynolds numbers.**

**[99] Sato et al. (2025) · Parametric reduced-order modelling and mode sensitivity of actuated cylinder flow from a matrix manifold perspective** · 5 citations · J. Fluid Mech.
<https://consensus.app/papers/details/b6db7ae099b7572b9db8cbc86d0d7d17/?utm_source=claude_desktop>
> We present a framework for parametric proper orthogonal decomposition (POD)-Galerkin reduced-order modelling (ROM) of fluid flows that accommodates variations in flow parameters and control inputs. [...] The sensitivity analysis, by defining distance between POD modes for different parameters, is applied to the flow around a rotating cylinder with varying Reynolds numbers and rotation rates. **The sensitivity of the subspace spanned by POD modes to parameter changes is represented by a tangent vector on the Grassmann manifold.** [...] The reconstruction error of the ROM is intimately linked to the subspace-estimation error, which is in turn closely related to subspace sensitivity.

---

*Archive complete. 19 Consensus queries, ~99 papers surfaced, ~65 archived here as directly relevant. Rerun the same queries after Aug 1 (budget reset) to catch any papers indexed since.*
