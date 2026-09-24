<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I want to ask another question about manifold analysis. I'm thinking to verify the shape of my manifold of data. I tried different approaches as described in the attached pdf. But now, I'm thinking to check how curved/bumpy is the manifold. So, can I calculate the highper vectors that connect two data points and compute their angles or use cross or dot product and estimate the curvature of shape of the manifold. Search the publications and evaluate and refine this idea. Is it well known method?

Yes — this is a well-known idea, with a specific name and precise mathematics behind it, though it needs some refinement for your low-sample-count setting. Let me lay it out.

***

## Your Idea, Formalized

You are proposing: take pairs (or triples) of data points on the manifold, form the secant vectors connecting them, and measure how the *direction* of these secant vectors changes — either via the angle between consecutive secants or via cross/dot products — to estimate how much the manifold bends. This is exactly the discrete analogue of curvature, and it is indeed a well-established family of techniques.

***

## The Established Methods That Match Your Idea

### 1. Menger Curvature (Triangle-Based, Closest to Your Description)

For any three points $\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3$ on the manifold, the **Menger curvature** is defined as the reciprocal of the radius of the circle passing through them:

$$
\kappa(\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3) = \frac{4 \cdot \text{Area}(\triangle \mathbf{p}_1\mathbf{p}_2\mathbf{p}_3)}{\|\mathbf{p}_1 - \mathbf{p}_2\|\,\|\mathbf{p}_2 - \mathbf{p}_3\|\,\|\mathbf{p}_3 - \mathbf{p}_1\|}
$$

The triangle area is computed exactly via the cross product of the two secant (edge) vectors — this is the precise formalization of your "cross product between two vectors connecting data points" idea. If three points are nearly collinear (flat, locally linear manifold), the area is near zero and $\kappa \approx 0$; if they bend sharply, $\kappa$ is large. This is a genuinely classical discrete curvature estimator, used extensively in point-cloud geometry and manifold learning.[^1][^2]

### 2. Angle-Deficit / Turning-Angle Curvature (Closest to Your "Angle Between Vectors" Idea)

For a sequence of points along a path (e.g., ordered by one parameter, like increasing $v_{adv}$), the discrete curvature at point $i$ is the **turning angle** between consecutive secant vectors:

$$
\theta_i = \angle(\mathbf{p}_{i+1} - \mathbf{p}_i,\ \mathbf{p}_i - \mathbf{p}_{i-1})
$$

computed via $\cos\theta_i = \dfrac{(\mathbf{p}_{i+1}-\mathbf{p}_i)\cdot(\mathbf{p}_i-\mathbf{p}_{i-1})}{\|\mathbf{p}_{i+1}-\mathbf{p}_i\|\,\|\mathbf{p}_i-\mathbf{p}_{i-1}\|}$. This is the discrete curve curvature and is the 1D special case of the general point-cloud curvature problem. It requires your points to be ordered along a path — which you effectively have via the MDS axis-1 ordering by $v_{adv}$ or $\omega_{pin}$ that you found in your report. [^2][^3]

### 3. Local PCA Curvature (The Standard Method for Unordered Point Clouds — More Robust)

This is the modern, more rigorous version of your idea, and it is the standard technique in the point-cloud/manifold-learning literature. For each point $\mathbf{p}_i$, take its local neighborhood (k-nearest neighbors), fit a local tangent plane via PCA, and measure the curvature as the ratio of the **smallest eigenvalue** to the **sum of all eigenvalues** of the local covariance matrix:[^4][^1]

$$
\sigma_i = \frac{\lambda_{d+1}}{\lambda_1 + \lambda_2 + \cdots + \lambda_D}
$$

where $d$ is the assumed intrinsic dimension and $D$ is the ambient dimension. If the manifold is locally flat, the points lie exactly in the $d$-dimensional tangent plane and $\lambda_{d+1} \approx 0$; the larger $\sigma_i$, the more the local neighborhood departs from a flat tangent plane — i.e., the more curved.[^1]

This approach is more robust than the raw triangle/angle method because it uses many neighbors rather than just 2–3 points, which matters critically at your sample size (17–20 points).

### 4. Weingarten Map / Principal Curvature Estimation (More Rigorous, Requires More Data)

For manifolds with a well-estimated normal direction, the **Weingarten map** relates changes in the normal vector to changes in tangential position — this recovers principal curvatures and directions rigorously, but requires substantially more local samples than you have (typically 20+ neighbors per point for reliable estimates).[^5]

***

## Critical Evaluation for Your Specific Case

### The core problem: sample count

All these curvature estimators need **local neighborhoods** to be statistically meaningful. Menger curvature from 3 arbitrary points is a single noisy sample of curvature, not an average — with only 17–20 total data points in a 2D-3D parameter space, most triples of points are far apart on the manifold, and the resulting curvature estimate reflects the **coarse global bending** of the parameter-to-response map, not the fine local shape.

This is not necessarily bad — it is actually **exactly what you want** for your problem, because your goal is not fine-scale surface roughness but the question "is the parameter-to-field map globally nonlinear/curved enough that linear POD needs many modes?" A small number of Menger-curvature triangles, chosen deliberately (e.g., triples along increasing $v_{adv}$ at fixed $\omega_{pin}$, or vice versa), gives you a **direct geometric confirmation** of what your active-subspace and intrinsic-dimension analysis already suggested.

### How this complements (not replaces) what you already did

Your PDF report already established via linear PCA + intrinsic dimension estimators (TwoNN, correlation dimension) that the manifold has intrinsic dimension ≈ 2.5–3.3 while needing 9–11 linear modes for 90% energy — the classic signature of a curved manifold. The angle/cross-product curvature idea is a **complementary, more interpretable and more local diagnostic**: instead of a single global dimension number, it gives you a **curvature value per triple of configurations**, letting you map out *where* in parameter space the manifold bends most sharply. This is valuable because it can directly guide your next-sample placement — exactly the recommendation from the previous discussion (add samples in the high-$|\omega|$/low-$v_{adv}$ corner). If the curvature there is measurably highest, that is independent geometric confirmation the sampling priority is correct.

### Concrete recommended implementation

1. **Order your snapshots along each parameter axis** (fix $\omega_{pin}$, vary $v_{adv}$, and vice versa) — this gives you natural 1D paths for the turning-angle curvature.
2. **Compute Menger curvature for all consecutive triples** along each path — this directly measures how much the field snapshot trajectory bends as you vary one parameter.
3. **Cross-check against local PCA curvature** using k=4–6 nearest neighbors (in the full high-dimensional snapshot space) for each configuration — more robust to the sparse, irregular sampling in 2D parameter space.
4. **Compare curvature magnitude between density and particle manifolds** — your report already shows particles have higher intrinsic dimension (3.3 vs 2.5); the curvature estimator should show particles have measurably higher curvature values too, providing convergent evidence.

### One caution

With only 17–20 points, do not over-interpret a single curvature number as a precise geometric quantity — treat it as a **relative, comparative diagnostic** (is region A more curved than region B? Is the particle manifold more curved than the density manifold?) rather than an absolute curvature value with error bars. This mirrors the caution your own report already applied to TwoNN at small sample counts.[^6]

***

## Bottom Line

Your idea is a legitimate, well-established technique — it has several named variants in the literature (Menger curvature, discrete turning-angle curvature, local-PCA curvature, Weingarten map estimation).  The cross-product/angle approach specifically corresponds to Menger curvature and discrete turning-angle curvature. Given your sample size, the **local-PCA curvature estimator** is the most statistically robust choice, while **Menger curvature along ordered 1D parameter paths** is the most interpretable and easiest to compute with your existing 17–20 snapshots — and it directly complements (not replaces) the intrinsic-dimension analysis you already completed.[^2][^5][^1]
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://arxiv.org/abs/2106.09972

[^2]: https://arxiv.org/pdf/2506.06779.pdf

[^3]: https://arxiv.org/html/2502.03750v1

[^4]: https://arxiv.org/pdf/2510.15141v1.pdf

[^5]: https://ar5iv.labs.arxiv.org/html/1905.10725

[^6]: rom_feasibility_report.pdf

[^7]: https://www.emergentmind.com/topics/local-point-cloud-curvature

[^8]: https://www.aimspress.com/aimspress-data/math/2024/11/PDF/math-09-11-1557.pdf

[^9]: https://aimspress.com/article/doi/10.3934/math.20241557?viewType=HTML

[^10]: https://geometry.cs.ucl.ac.uk/projects/2018/pcpnet/paper_docs/GuerreroEtAl_PCPNet_EG18.pdf

[^11]: https://www.isprs.org/proceedings/XXXVII/congress/5_pdf/83.pdf

[^12]: https://pages.uoregon.edu/micahw/papers/ache_warren_manifold_learningMarch21.pdf

[^13]: https://nlpr.ia.ac.cn/2008papers/gjhy/gh129.pdf

[^14]: https://arxiv.org/pdf/2010.09419v1.pdf

[^15]: https://brickisland.net/DDGSpring2016/wp-content/uploads/2016/05/NoisyCurvatureWriteup.pdf

[^16]: https://ouci.dntb.gov.ua/en/works/9jLxMRq4/

