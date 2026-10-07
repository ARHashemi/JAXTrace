# Table T7 · Float-precision ablation — FinalFuelPinPoly1560

Per-tet correctness (`correct_rate_strict`) of each MALMO variant on the FinalFuelPinPoly1560 mesh across the {float32, float64} × {1e-4, 1e-6, 1e-8} PIT-tolerance design.  Sampling: in-mesh barycentric, $N_p = 100\,000$ queries.  Rates ≥ 99.99 % (fewer than 1 miss in 10 000) are rendered as 100.00 %.

| variant | precision | $\varepsilon = 10^{-4}$ | $\varepsilon = 10^{-6}$ | $\varepsilon = 10^{-8}$ |
|---|---|---:|---:|---:|
| MALMO$^\mathrm{A}$ | fp32 | 100.00 % | 100.00 % | 100.00 % |
| MALMO$^\mathrm{A}$ | fp64 | 100.00 % | 100.00 % | 100.00 % |
| MALMO$^\mathrm{C}$ | fp32 | 100.00 % | 100.00 % | 100.00 % |
| MALMO$^\mathrm{C}$ | fp64 | 100.00 % | 100.00 % | 100.00 % |
| MALMO$^\mathrm{V}$ | fp32 | 100.00 % | 100.00 % | 100.00 % |
| MALMO$^\mathrm{V}$ | fp64 | 100.00 % | 100.00 % | 100.00 % |
