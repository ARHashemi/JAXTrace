# RTX 5090 Benchmark Report

Generated: 2026-06-15T15:22:25

This document organizes new benchmark runs from the RTX 5090
workstation in the same structure as the paper's Section 6
(Numerical Validation) and Section 7 (Application). Use these
tables to update `sec6_validation.tex` and `sec7_application.tex`.

## Manifest

```json
{
  "started": "2026-06-15T13:54:48+00:00",
  "hostname": "fsw-gpu",
  "jaxtrace_root": "/flash/shared/jax/JAXTrace",
  "jaxtrace_commit": "22696ce937c6a0952ac2f84852c60f10272c921d",
  "jaxtrace_branch": "main",
  "mesh_base": "/flash/users/ali/data/cylA.gid/post",
  "mesh_pattern": "cylA_{timestep.pvtu}",
  "vel_range": [159, 159],
  "sec6_n_particles": 10000,
  "sec7_n_steps": 2684,
  "sec7_dt": 0.0025,
  "gpu": "NVIDIA GeForce RTX 5090, 32607 MiB, 595.71.05",
  "jax_version": "0.10.0",
  "python": "Python 3.14.4"
}

```

---

## Section 6 — Numerical Validation

### 6.1  Data-structure statistics per registration strategy

Replaces the "Data-structure statistics" sub-block of Table 1 in
`sec6_validation.tex`. Morton-linear is omitted because it is a
flat element array (no cells); see `VALIDATION_UPDATE.md §1`.

| Registration | Active cells | elems/cell (mean) | cells/elem (mean) | max elems/cell |
|---|---|---|---|---|
| Vertex-multi octree | 666,162 | 18.3 | 4.0 | — |
| Parent-cube octree | 517,309 | 5.9 | 1.0 | 8 |
| AABB-overlap octree | 620,760 | 30.8 | 6.3 | 162 |

### 6.2.1  Found rate F(%) under perturbation

Replaces `tab:found_rate`. The MALMO\textsuperscript{AABB} row,
previously "---" at σ ≥ 0.5, is now populated with actual numbers.

| Method | 0.0x | 0.1x | 0.2x | 0.5x | 0.7x | 1.0x |
|---|---|---|---|---|---|---|
| Morton-linear w=5 | 31.98% | 29.93% | 27.01% | 24.60% | 23.63% | 23.46% |
| Morton-linear w=21 | 59.12% | 52.46% | 50.36% | 49.00% | 47.75% | 47.45% |
| MALMO_V 1x1x1 | 50.17% | 49.93% | 49.93% | 49.69% | 49.22% | 49.10% |
| MALMO_V 3x3x3 | 100.00% | 99.98% | 99.70% | 98.51% | 97.71% | 96.61% |
| MALMO_V 5x5x5 | 100.00% | 99.98% | 99.70% | 98.51% | 97.71% | 96.61% |
| MALMO_C 3x3x3 | 100.00% | 99.98% | 99.70% | 98.51% | 97.71% | 96.61% |
| MALMO_AABB 3x3x3 | 100.00% | 99.98% | 99.70% | 98.51% | 97.71% | 96.61% |

### 6.2.2  Domain-interior search failures N_fail

Replaces `tab:search_failures`. Row order matches the paper.

| Method | 0.0x | 0.1x | 0.2x | 0.5x | 0.7x | 1.0x |
|---|---|---|---|---|---|---|
| Morton-linear w=5 | 6802 | 7005 | 7269 | 7391 | 7408 | 7315 |
| Morton-linear w=21 | 4088 | 4752 | 4934 | 4951 | 4996 | 4916 |
| MALMO_V 1x1x1 | 4983 | 5005 | 4977 | 4882 | 4849 | 4751 |
| MALMO_V 3x3x3 | 0 | 0 | 0 | 0 | 0 | 0 |
| MALMO_V 5x5x5 | 0 | 0 | 0 | 0 | 0 | 0 |
| MALMO_C 3x3x3 | 0 | 0 | 0 | 0 | 0 | 0 |
| MALMO_AABB 3x3x3 | 0 | 0 | 0 | 0 | 0 | 0 |

### 6.3  Intra-element found rate

Replaces `tab:intra_found`.

```
      centroid: generated 10,000 particles
        random: generated 10,000 particles
     near_face: generated 10,000 particles
     near_edge: generated 10,000 particles
   near_vertex: generated 10,000 particles

  --- radius r=2 ---
        centroid: found=3,152/10,000, correct_elem=3,152, wrong_elem=0, NOT_FOUND=6,848, time=2.386 ± 0.007 (2.375--2.397)s
          random: found=2,465/10,000, correct_elem=2,465, wrong_elem=0, NOT_FOUND=7,535, time=2.445 ± 0.136 (2.383--2.778)s
       near_face: found=2,308/10,000, correct_elem=2,308, wrong_elem=0, NOT_FOUND=7,692, time=2.387 ± 0.011 (2.369--2.399)s
       near_edge: found=1,928/10,000, correct_elem=1,928, wrong_elem=0, NOT_FOUND=8,072, time=2.390 ± 0.009 (2.381--2.407)s
     near_vertex: found=1,121/10,000, correct_elem=1,121, wrong_elem=0, NOT_FOUND=8,879, time=2.386 ± 0.008 (2.377--2.398)s

  --- radius r=10 ---
        centroid: found=5,934/10,000, correct_elem=5,934, wrong_elem=0, NOT_FOUND=4,066, time=2.390 ± 0.009 (2.374--2.403)s
          random: found=4,974/10,000, correct_elem=4,974, wrong_elem=0, NOT_FOUND=5,026, time=2.392 ± 0.008 (2.382--2.405)s
       near_face: found=4,950/10,000, correct_elem=4,950, wrong_elem=0, NOT_FOUND=5,050, time=2.395 ± 0.005 (2.387--2.403)s
       near_edge: found=4,898/10,000, correct_elem=4,898, wrong_elem=0, NOT_FOUND=5,102, time=2.379 ± 0.002 (2.375--2.382)s
     near_vertex: found=4,544/10,000, correct_elem=4,542, wrong_elem=2, NOT_FOUND=5,456, time=2.395 ± 0.003 (2.388--2.397)s

  --- 1x1x1 ---
        centroid: found=4,942/10,000, correct_elem=4,942, wrong_elem=0, NOT_FOUND=5,058, time=2.082 ± 0.177 (1.998--2.515)s
          random: found=4,988/10,000, correct_elem=4,988, wrong_elem=0, NOT_FOUND=5,012, time=1.999 ± 0.008 (1.987--2.013)s
       near_face: found=4,977/10,000, correct_elem=4,977, wrong_elem=0, NOT_FOUND=5,023, time=2.003 ± 0.006 (1.997--2.012)s
       near_edge: found=4,991/10,000, correct_elem=4,991, wrong_elem=0, NOT_FOUND=5,009, time=2.174 ± 0.328 (2.007--2.960)s
     near_vertex: found=4,912/10,000, correct_elem=4,910, wrong_elem=2, NOT_FOUND=5,088, time=2.164 ± 0.322 (1.995--2.939)s

  --- 3x3x3 ---
        centroid: found=10,000/10,000, correct_elem=10,000, wrong_elem=0, NOT_FOUND=0, time=2.190 ± 0.168 (2.114--2.601)s
          random: found=10,000/10,000, correct_elem=10,000, wrong_elem=0, NOT_FOUND=0, time=2.110 ± 0.003 (2.107--2.115)s
       near_face: found=10,000/10,000, correct_elem=10,000, wrong_elem=0, NOT_FOUND=0, time=2.107 ± 0.008 (2.096--2.125)s
       near_edge: found=10,000/10,000, correct_elem=10,000, wrong_elem=0, NOT_FOUND=0, time=2.120 ± 0.012 (2.105--2.145)s
     near_vertex: found=10,000/10,000, correct_elem=9,998, wrong_elem=2, NOT_FOUND=0, time=2.114 ± 0.005 (2.104--2.120)s

  --- 5x5x5 ---
        centroid: found=10,000/10,000, correct_elem=10,000, wrong_elem=0, NOT_FOUND=0, time=2.518 ± 0.004 (2.513--2.524)s
          random: found=10,000/10,000, correct_elem=10,000, wrong_elem=0, NOT_FOUND=0, time=2.591 ± 0.166 (2.505--2.996)s
       near_face: found=10,000/10,000, correct_elem=10,000, wrong_elem=0, NOT_FOUND=0, time=2.524 ± 0.012 (2.513--2.548)s
       near_edge: found=10,000/10,000, correct_elem=10,000, wrong_elem=0, NOT_FOUND=0, time=2.525 ± 0.006 (2.514--2.535)s
     near_vertex: found=10,000/10,000, correct_elem=9,998, wrong_elem=2, NOT_FOUND=0, time=2.523 ± 0.009 (2.515--2.543)s

  --- 3x3x3^PC ---
        centroid: found=10,000/10,000, correct_elem=10,000, wrong_elem=0, NOT_FOUND=0, time=1.829 ± 0.007 (1.823--1.846)s
          random: found=10,000/10,000, correct_elem=10,000, wrong_elem=0, NOT_FOUND=0, time=1.824 ± 0.004 (1.818--1.832)s
       near_face: found=10,000/10,000, correct_elem=10,000, wrong_elem=0, NOT_FOUND=0, time=1.829 ± 0.002 (1.826--1.833)s
       near_edge: found=10,000/10,000, correct_elem=10,000, wrong_elem=0, NOT_FOUND=0, time=1.826 ± 0.009 (1.817--1.845)s
     near_vertex: found=10,000/10,000, correct_elem=9,998, wrong_elem=2, NOT_FOUND=0, time=1.976 ± 0.325 (1.836--2.773)s

  --- 3x3x3^AABB ---
        centroid: found=10,000/10,000, correct_elem=10,000, wrong_elem=0, NOT_FOUND=0, time=2.299 ± 0.010 (2.282--2.315)s
          random: found=10,000/10,000, correct_elem=10,000, wrong_elem=0, NOT_FOUND=0, time=2.309 ± 0.007 (2.299--2.322)s
       near_face: found=10,000/10,000, correct_elem=10,000, wrong_elem=0, NOT_FOUND=0, time=2.315 ± 0.009 (2.305--2.336)s
       near_edge: found=10,000/10,000, correct_elem=10,000, wrong_elem=0, NOT_FOUND=0, time=2.309 ± 0.003 (2.305--2.313)s
     near_vertex: found=10,000/10,000, correct_elem=9,998, wrong_elem=2, NOT_FOUND=0, time=2.317 ± 0.005 (2.307--2.323)s
```

### 6.4.1  Computational performance — N_p = 10,000

Replaces `tab:timing`. Wall-time min–max over 7 timed runs after
3 warm-ups.

| Method | 0.0x |
|---|---|
| Morton-linear w=5 | 2.337 ± 0.009 (2.320--2.349) 2.340 ± 0.004 (2.334--2.345) 2.379 ± 0.088 (2.328--2.593) 2.326 ± 0.005 (2.318--2.333) 2.331 ± 0.009 (2.321--2.350) 2.328 ± 0.005 (2.317--2.335) |
| Morton-linear w=21 | 2.333 ± 0.009 (2.317--2.343) 2.335 ± 0.009 (2.326--2.350) 2.335 ± 0.004 (2.330--2.343) 2.327 ± 0.009 (2.312--2.339) 2.319 ± 0.010 (2.307--2.329) 2.326 ± 0.008 (2.307--2.334) |
| MALMO_V 1x1x1 | 1.979 ± 0.123 (1.926--2.279) 1.920 ± 0.006 (1.911--1.928) 1.927 ± 0.007 (1.916--1.937) 1.925 ± 0.004 (1.918--1.931) 1.924 ± 0.007 (1.916--1.940) 1.929 ± 0.010 (1.916--1.949) |
| MALMO_V 3x3x3 | 2.035 ± 0.004 (2.030--2.041) 2.036 ± 0.008 (2.023--2.050) 2.044 ± 0.011 (2.035--2.069) 2.307 ± 0.400 (2.052--2.991) 2.178 ± 0.327 (2.033--2.979) 2.039 ± 0.010 (2.022--2.053) |
| MALMO_V 5x5x5 | 2.416 ± 0.008 (2.406--2.433) 2.586 ± 0.328 (2.439--3.389) 2.470 ± 0.012 (2.458--2.494) 2.511 ± 0.009 (2.501--2.530) 2.527 ± 0.009 (2.514--2.540) 2.514 ± 0.010 (2.498--2.527) |
| MALMO_C 3x3x3 | 1.900 ± 0.327 (1.757--2.700) 1.770 ± 0.011 (1.755--1.791) 1.764 ± 0.006 (1.755--1.770) 1.752 ± 0.004 (1.746--1.756) 1.758 ± 0.005 (1.749--1.764) 1.758 ± 0.008 (1.748--1.771) |
| MALMO_AABB 3x3x3 | 2.367 ± 0.271 (2.252--3.030) 2.246 ± 0.005 (2.235--2.254) 2.263 ± 0.010 (2.248--2.280) 2.276 ± 0.008 (2.265--2.293) 2.270 ± 0.009 (2.256--2.283) 2.277 ± 0.005 (2.270--2.285) |

### 6.4.1bis  Performance summary (Queries/s, PIT/s, relative)

| Method | Queries/s | PIT tests/s | Mean time (s) | Relative |
|---|---|---|---|---|
| Morton-linear w=5 | 4,279 | n/a | 2.3370 | 1.18x |
| Morton-linear w=21 | 4,286 | n/a | 2.3330 | 1.18x |
| MALMO_V 1x1x1 | 5,053 | 116963 | 1.9791 | 1.00x |
| MALMO_V 3x3x3 | 4,915 | 921392 | 2.0346 | 1.03x |
| MALMO_V 5x5x5 | 4,139 | 3945880 | 2.4159 | 1.22x |
| MALMO_C 3x3x3 | 5,264 | 400767 | 1.8999 | 0.96x |
| MALMO_AABB 3x3x3 | 4,225 | 1004664 | 2.3669 | 1.20x |

### 6.4.1ter  Mean PIT tests per query

| Method | 0.0x | 0.1x | 0.2x | 0.5x | 0.7x | 1.0x |
|---|---|---|---|---|---|---|
| Morton-linear w=5 | n/a | n/a | n/a | n/a | n/a | n/a |
| Morton-linear w=21 | n/a | n/a | n/a | n/a | n/a | n/a |
| MALMO_V 1x1x1 | 23.1 | 23.1 | 23.1 | 23.1 | 23.3 | 23.2 |
| MALMO_V 3x3x3 | 187.5 | 186.2 | 188.1 | 198.3 | 202.8 | 206.4 |
| MALMO_V 5x5x5 | 953.3 | 949.5 | 959.5 | 1015.8 | 1039.2 | 1068.8 |
| MALMO_C 3x3x3 | 76.1 | 75.9 | 75.9 | 77.7 | 78.1 | 78.3 |
| MALMO_AABB 3x3x3 | 237.8 | 236.5 | 239.6 | 252.9 | 258.5 | 263.7 |

### 6.4.3  Throughput scaling

Replaces `tab:scalability`. Reported for MALMO\textsuperscript{C}
$3\times3\times3$.

```
  Warmup runs: 3, Timing runs: 7

  N_p=   1,000: found=1,000/1,000, time=1.849 ± 0.010 (1.840--1.872)s, 541 queries/s, 1849.3 us/query, mean_PIT=76.1
  N_p=   2,000: found=2,000/2,000, time=1.848 ± 0.010 (1.836--1.867)s, 1,083 queries/s, 923.8 us/query, mean_PIT=76.1
  N_p=   5,000: found=5,000/5,000, time=1.827 ± 0.006 (1.815--1.835)s, 2,737 queries/s, 365.4 us/query, mean_PIT=75.9
  N_p=  10,000: found=10,000/10,000, time=1.832 ± 0.004 (1.827--1.837)s, 5,457 queries/s, 183.2 us/query, mean_PIT=76.0
  N_p=  20,000: found=20,000/20,000, time=1.834 ± 0.011 (1.824--1.860)s, 10,904 queries/s, 91.7 us/query, mean_PIT=76.1
  N_p=  50,000: found=50,000/50,000, time=1.858 ± 0.009 (1.848--1.876)s, 26,905 queries/s, 37.2 us/query, mean_PIT=75.9
  N_p= 100,000: found=100,000/100,000, time=1.890 ± 0.009 (1.879--1.907)s, 52,904 queries/s, 18.9 us/query, mean_PIT=75.9
  N_p= 200,000: found=200,000/200,000, time=1.976 ± 0.003 (1.971--1.980)s, 101,223 queries/s, 9.9 us/query, mean_PIT=76.0
  N_p= 500,000: found=500,000/500,000, time=2.221 ± 0.008 (2.214--2.238)s, 225,147 queries/s, 4.4 us/query, mean_PIT=75.9

       N_p    Time (s)     Queries/s    us/query     PIT tests/s   Mean PIT     Found
  --------  ----------  ------------  ----------  --------------  ---------  --------
     1,000      1.8493           541      1849.3          41,133       76.1    100.0%
     2,000      1.8475         1,083       923.8          82,403       76.1    100.0%
     5,000      1.8269         2,737       365.4         207,613       75.9    100.0%
    10,000      1.8324         5,457       183.2         414,870       76.0    100.0%
    20,000      1.8343        10,904        91.7         829,424       76.1    100.0%
    50,000      1.8584        26,905        37.2       2,041,572       75.9    100.0%
   100,000      1.8902        52,904        18.9       4,015,298       75.9    100.0%
   200,000      1.9758       101,223         9.9       7,691,523       76.0    100.0%
   500,000      2.2208       225,147         4.4      17,098,389       75.9    100.0%

Benchmark complete!
```

### 6.4.4  Resolving level distribution

Replaces `tab:level_dist`.

```

  Particles: 10,000, Found: 10,000, Not found: 0

   Level     Count    Fraction  Cumulative
  ------  --------  ----------  ----------
       8         7       0.07%       0.07%
       9         4       0.04%       0.11%
      10        15       0.15%       0.26%
      11        39       0.39%       0.65%
      12       141       1.41%       2.06%
      13     1,292      12.92%      14.98%
      14     8,502      85.02%     100.00%

  Mean resolving level: 13.82
  Median resolving level: 14
  Mode resolving level: 14
```

### 6.4.5  Build cost and amortisation

Replaces the build-cost paragraph and Appendix `sec:app_build_cost`.
Per `VALIDATION_UPDATE.md §3`, verify whether JAX JIT warm-up is
included; cold vs. amortised should be separated.

```
Stage                                       Time (s)    % of build
------------------------------------------------------------------------------------------
mesh_io_pvtu_load                              2.591         0.22%
mesh_node_deduplication                        3.218         0.27%
aa_metadata_precompute                        20.787         1.75%
inverse_matrices_precompute                   17.212         1.45%
octree_extract_vertex_multi                  188.780        15.88%
octree_upload_vertex_multi                     0.095         0.01%
octree_extract_parent_cube                    78.357         6.59%
octree_upload_parent_cube                      0.068         0.01%
morton_structure_build_upload                 85.860         7.22%
------------------------------------------------------------------------------------------
TOTAL build (preprocessing + structures)    1189.067       100.00%

  Mesh size: 3,050,196 elements, 571,533 nodes
  Vertex-multi octree:  666,162 cells, 18.3 elem/cell, 4.00 cells/elem
  Parent-cube octree:   517,309 cells, 5.9 elem/cell (max 8)
  Octree GPU footprint: 81.4 MB total, 6.0 MB hot working set
  Inverse-matrix table: 292.8 MB

  Amortisation (queries needed to pay back build, using 0.0x perturb):
  Method               queries/s   t_query (s)   breakeven (queries)
  --------------------------------------------------------------------------------------
  radius r=2               4,279     2.337e-04             5,087,913
  radius r=10              4,286     2.333e-04             5,096,671
  1x1x1                    5,053     1.979e-04             6,008,090
  3x3x3                    4,915     2.035e-04             5,844,160
  5x5x5                    4,139     2.416e-04             4,921,930
  3x3x3^PC                 5,264     1.900e-04             6,258,672
  3x3x3^AABB               4,225     2.367e-04             5,023,667
```

---

## Section 7 — Application

### 7.x  End-to-end tracking on cylA mesh

Replaces the headline-numbers table in `sec7_application.tex`. The
full per-step CSV and deviation maps are under
`sec7_femuss_comparison/` for plotting.

| Metric | Value |
|---|---|
| Tracking wall time (s) | — |
| Throughput (p·step/s) | — |
| GPU memory | — |
| Mean deviation (m) | — |
| Max deviation (m) | — |

---

## Notes on the previous version

- VALIDATION_UPDATE.md §1: Morton-linear "cells" entries removed.
- §2: "(max 8)" annotation on MALMO\textsuperscript{C} dropped — it
  was a stray copy of the elems-per-cell maximum.
- §3: Build times verified (see §6.4.5 above). Watch for JIT
  warm-up inclusion in those numbers.
- §3: MALMO\textsuperscript{AABB} now evaluated at all σ levels;
  the table no longer has "---" entries at σ ≥ 0.5.
