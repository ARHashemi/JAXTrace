# Section 6 (Validation) — re-run report

## Structural findings (block paper finalisation)

### F3: Cell count mislabelled in Table 6.1

* **symptom**: The paper attributes 517,309 cells to MALMO^C (parent-cube), but that number is actually the Morton-linear structure's cell count. The true MALMO^C parent-cube octree has 518,858 cells.
* **root cause**: Copy-paste error at paper draft time; the two structures were treated as one row in Table 6.1.
* **fix in code**: No code change required. The harness already prints the correct labels for both structures.
* **paper-side action**: Update Table 6.1 (tab:mesh_properties) MALMO^C row: active cells 518,858 (max occupancy 24). Add a separate row or footnote for the Morton-linear structure count if it is used elsewhere in the text.

## Canonical protocol used for this re-run

* **n_particles**: `10000`
* **batch_size**: `50000`
* **warmup_runs**: `3`
* **timing_runs**: `7`
* **perturbations**: `[0.0, 0.1, 0.2, 0.5, 0.7, 1.0]`
* **position_types**: `['centroid', 'random', 'near_face', 'near_edge', 'near_vertex']`
* **scalability_sizes**: `[1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]`
* **queries_per_second**: `n_p / mean(times)  [canonical rule; single aggregation across ALL tables]`
* **failure_decomposition_counts**: `raw (level, offset) neighbour-hit histogram — sum is NOT a batch size; paper caption must state this OR the analyser must be patched to per-particle classification`

## Table-by-table diff (paper vs re-run)

### tab:found_rate (Found rate under perturbation) — PASS

* **Morton-linear w=5** — OK
    * paper:    `[31.98, 29.93, 27.01, 24.6, 23.63, 23.46]`
    * measured: `[31.98, 29.93, 27.01, 24.6, 23.63, 23.46]`
* **Morton-linear w=21** — OK
    * paper:    `[59.12, 52.46, 50.36, 49.0, 47.75, 47.45]`
    * measured: `[59.12, 52.46, 50.36, 49.0, 47.75, 47.45]`
* **MALMO_V 1x1x1** — OK
    * paper:    `[50.17, 49.93, 49.93, 49.69, 49.22, 49.1]`
    * measured: `[50.17, 49.93, 49.93, 49.69, 49.22, 49.1]`
* **MALMO_V 3x3x3** — OK
    * paper:    `[100.0, 99.98, 99.7, 98.51, 97.71, 96.61]`
    * measured: `[100.0, 99.98, 99.7, 98.51, 97.71, 96.61]`
* **MALMO_V 5x5x5** — OK
    * paper:    `[100.0, 99.98, 99.7, 98.51, 97.71, 96.61]`
    * measured: `[100.0, 99.98, 99.7, 98.51, 97.71, 96.61]`
* **MALMO_C 3x3x3** — OK
    * paper:    `[100.0, 99.98, 99.7, 98.51, 97.71, 96.61]`
    * measured: `[100.0, 99.98, 99.7, 98.51, 97.71, 96.61]`
* **MALMO_AABB 3x3x3** — OK
    * paper:    `[100.0, 99.98, 99.7, 98.51, 97.71, 96.61]`
    * measured: `[100.0, 99.98, 99.7, 98.51, 97.71, 96.61]`

### tab:search_failures (in-bbox misses) — PASS

* **Morton-linear w=5** — OK
    * paper:    `[6802, 7005, 7269, 7391, 7408, 7315]`
    * measured: `[6802, 7005, 7269, 7391, 7408, 7315]`
* **Morton-linear w=21** — OK
    * paper:    `[4088, 4752, 4934, 4951, 4996, 4916]`
    * measured: `[4088, 4752, 4934, 4951, 4996, 4916]`
* **MALMO_V 1x1x1** — OK
    * paper:    `[4983, 5005, 4977, 4882, 4849, 4751]`
    * measured: `[4983, 5005, 4977, 4882, 4849, 4751]`
* **MALMO_V 3x3x3** — OK
    * paper:    `[0, 0, 0, 0, 0, 0]`
    * measured: `[0, 0, 0, 0, 0, 0]`
* **MALMO_V 5x5x5** — OK
    * paper:    `[0, 0, 0, 0, 0, 0]`
    * measured: `[0, 0, 0, 0, 0, 0]`
* **MALMO_C 3x3x3** — OK
    * paper:    `[0, 0, 0, 0, 0, 0]`
    * measured: `[0, 0, 0, 0, 0, 0]`
* **MALMO_AABB 3x3x3** — OK
    * paper:    `[0, 0, 0, 0, 0, 0]`
    * measured: `[0, 0, 0, 0, 0, 0]`

### tab:intra_found (Intra-element found rate) — PASS

* **MALMO_V 1x1x1** — OK
    * paper:    `[49.42, 49.88, 49.77, 49.91, 49.1]`
    * measured: `[49.42, 49.88, 49.77, 49.91, 49.12]`
* **MALMO_V 3x3x3** — OK
    * paper:    `[100.0, 100.0, 100.0, 100.0, 99.98]`
    * measured: `[100.0, 100.0, 100.0, 100.0, 100.0]`
* **MALMO_V 5x5x5** — OK
    * paper:    `[100.0, 100.0, 100.0, 100.0, 99.98]`
    * measured: `[100.0, 100.0, 100.0, 100.0, 100.0]`
* **MALMO_C 3x3x3** — OK
    * paper:    `[100.0, 100.0, 100.0, 100.0, 99.98]`
    * measured: `[100.0, 100.0, 100.0, 100.0, 100.0]`
* **MALMO_AABB 3x3x3** — OK
    * paper:    `[100.0, 100.0, 100.0, 100.0, 99.98]`
    * measured: `[100.0, 100.0, 100.0, 100.0, 100.0]`

### tab:timing (Computational performance @ sigma=0) — **CHANGED**

* **MALMO_C 3x3x3** — CHANGED
    * paper:    `{'time_min': 1.746, 'time_max': 1.9, 'mean_pit': 76.1, 'qps': 5264}`
    * measured: `{'time_min': 1.84, 'time_max': 1.868, 'mean_pit': 76.3, 'qps': 5407}`
* **MALMO_V 1x1x1** — CHANGED
    * paper:    `{'time_min': 1.911, 'time_max': 1.979, 'mean_pit': 23.1, 'qps': 5053}`
    * measured: `{'time_min': 1.978, 'time_max': 2.0, 'mean_pit': 23.1, 'qps': 5041}`
* **MALMO_V 3x3x3** — OK
    * paper:    `{'time_min': 2.023, 'time_max': 2.041, 'mean_pit': 187.5, 'qps': 4915}`
    * measured: `{'time_min': 2.06, 'time_max': 2.092, 'mean_pit': 187.5, 'qps': 4828}`
* **Morton-linear w=21** — OK
    * paper:    `{'time_min': 2.307, 'time_max': 2.335, 'mean_pit': None, 'qps': 4286}`
    * measured: `{'time_min': 2.351, 'time_max': 2.371, 'mean_pit': None, 'qps': 4232}`
* **Morton-linear w=5** — OK
    * paper:    `{'time_min': 2.32, 'time_max': 2.349, 'mean_pit': None, 'qps': 4279}`
    * measured: `{'time_min': 2.373, 'time_max': 2.387, 'mean_pit': None, 'qps': 4199}`
* **MALMO_AABB 3x3x3** — CHANGED
    * paper:    `{'time_min': 2.235, 'time_max': 2.367, 'mean_pit': 237.8, 'qps': 4225}`
    * measured: `{'time_min': 2.259, 'time_max': 2.275, 'mean_pit': 237.8, 'qps': 4406}`
* **MALMO_V 5x5x5** — OK
    * paper:    `{'time_min': 2.406, 'time_max': 2.433, 'mean_pit': 953.3, 'qps': 4139}`
    * measured: `{'time_min': 2.423, 'time_max': 2.447, 'mean_pit': 953.3, 'qps': 4112}`

### tab:scalability (throughput vs batch size) — **CHANGED**

* **1000** — OK
    * paper:    `{'time': 1.849, 'qps': 541, 'us_per_q': 1849, 'pit_s': 41133}`
    * measured: `{'n_p': 1000, 'time_mean': 1.858, 'qps': 538, 'us_per_q': 1858.0, 'pit_s': 40974, 'mean_pit': 76.1, 'found_pct': 100.0}`
* **2000** — OK
    * paper:    `{'time': 1.848, 'qps': 1083, 'us_per_q': 924, 'pit_s': 82403}`
    * measured: `{'n_p': 2000, 'time_mean': 1.8596, 'qps': 1076, 'us_per_q': 929.8, 'pit_s': 82026, 'mean_pit': 76.3, 'found_pct': 100.0}`
* **5000** — OK
    * paper:    `{'time': 1.827, 'qps': 2737, 'us_per_q': 365, 'pit_s': 207613}`
    * measured: `{'n_p': 5000, 'time_mean': 1.834, 'qps': 2726, 'us_per_q': 366.8, 'pit_s': 207503, 'mean_pit': 76.1, 'found_pct': 100.0}`
* **10000** — OK
    * paper:    `{'time': 1.832, 'qps': 5457, 'us_per_q': 183, 'pit_s': 414870}`
    * measured: `{'n_p': 10000, 'time_mean': 1.8548, 'qps': 5391, 'us_per_q': 185.5, 'pit_s': 411269, 'mean_pit': 76.3, 'found_pct': 100.0}`
* **20000** — OK
    * paper:    `{'time': 1.834, 'qps': 10904, 'us_per_q': 92, 'pit_s': 829424}`
    * measured: `{'n_p': 20000, 'time_mean': 1.85, 'qps': 10811, 'us_per_q': 92.5, 'pit_s': 824545, 'mean_pit': 76.3, 'found_pct': 100.0}`
* **50000** — OK
    * paper:    `{'time': 1.858, 'qps': 26905, 'us_per_q': 37, 'pit_s': 2041572}`
    * measured: `{'n_p': 50000, 'time_mean': 1.8656, 'qps': 26801, 'us_per_q': 37.3, 'pit_s': 2038749, 'mean_pit': 76.1, 'found_pct': 100.0}`
* **100000** — OK
    * paper:    `{'time': 1.89, 'qps': 52904, 'us_per_q': 19, 'pit_s': 4015298}`
    * measured: `{'n_p': 100000, 'time_mean': 1.9269, 'qps': 51897, 'us_per_q': 19.3, 'pit_s': 3949027, 'mean_pit': 76.1, 'found_pct': 100.0}`
* **200000** — OK
    * paper:    `{'time': 1.976, 'qps': 101223, 'us_per_q': 10, 'pit_s': 7691523}`
    * measured: `{'n_p': 200000, 'time_mean': 2.071, 'qps': 96571, 'us_per_q': 10.4, 'pit_s': 7356952, 'mean_pit': 76.2, 'found_pct': 100.0}`
* **500000** — CHANGED
    * paper:    `{'time': 2.221, 'qps': 225147, 'us_per_q': 4, 'pit_s': 17098389}`
    * measured: `{'n_p': 500000, 'time_mean': 2.4865, 'qps': 201083, 'us_per_q': 5.0, 'pit_s': 15310178, 'mean_pit': 76.1, 'found_pct': 100.0}`

### tab:level_dist (Resolving level histogram) — PASS

> Log uses absolute octree depth (8..14); paper uses relative depth (1..7 with finest = level 7). This diff maps 14↔7, 13↔6, 12↔5, 11↔4, and aggregates log levels 8..10 into paper 'level ≤ 3'. See LEVEL_LOG_TO_PAPER for the full mapping.

* **7** — OK
    * paper:    `{'count': 8502, 'frac': 85.0}`
    * measured: `{'count': 8502, 'frac': 85.02}`
* **6** — OK
    * paper:    `{'count': 1292, 'frac': 12.9}`
    * measured: `{'count': 1292, 'frac': 12.92}`
* **5** — OK
    * paper:    `{'count': 141, 'frac': 1.4}`
    * measured: `{'count': 141, 'frac': 1.41}`
* **4** — OK
    * paper:    `{'count': 39, 'frac': 0.4}`
    * measured: `{'count': 39, 'frac': 0.39}`
* **3** — OK
    * paper:    `{'count': 26, 'frac': 0.3}`
    * measured: `{'count': 26, 'frac': 0.26}`

### tab:failure_decomposition (1x1x1 misses) — PASS

> The paper table's counts (7463 / 9995 / 2542) sum to 20,000, which is NOT a batch size — it is the total number of (level, offset) neighbour hits across all failed particles. The raw harness histogram is the same statistic, so the measured 'sum_of_offset_hits' should match the paper's total (20,000) at the same N_p and sigma. If a per-particle decomposition is desired, the caption needs to be updated OR the failure analyser patched to classify each particle by its smallest recovering offset.

* **face_adjacent** — OK
    * paper:    `{'count': 7463, 'share': 37.3}`
    * measured: `{'raw_count': 7463, 'raw_pct': 37.3}`
* **edge_adjacent** — OK
    * paper:    `{'count': 9995, 'share': 50.0}`
    * measured: `{'raw_count': 9995, 'raw_pct': 50.0}`
* **corner_adjacent** — OK
    * paper:    `{'count': 2542, 'share': 12.7}`
    * measured: `{'raw_count': 2542, 'raw_pct': 12.7}`

### tab:bytes (memory traffic per query) — PASS

* **Morton-linear w=5** — OK
    * paper:    `62177`
    * measured: `62177`
* **Morton-linear w=21** — OK
    * paper:    `62177`
    * measured: `62177`
* **MALMO_AABB 3x3x3** — OK
    * paper:    `45724`
    * measured: `45724`
* **MALMO_V 1x1x1** — OK
    * paper:    `40233`
    * measured: `40233`
* **MALMO_V 3x3x3** — OK
    * paper:    `40229`
    * measured: `40229`
* **MALMO_V 5x5x5** — OK
    * paper:    `40229`
    * measured: `40229`
* **MALMO_C 3x3x3** — OK
    * paper:    `32693`
    * measured: `32706`
