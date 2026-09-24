# Figure rebuild notes for R2-10 + R1-d

Reviewer 2 explicitly asked for a "major revision" of Figures 1, 2, 3, 5.
Reviewer 1 flagged pixelated PDF images (R1-d).  The eight paper figures
fall into three tool buckets:

| Fig | Source tool  | Author of the rebuild | Reviewer-flagged? |
|----:|--------------|-----------------------|:-----------------:|
| 1   | Matplotlib + Inkscape | Ali (Inkscape)        | R2-10             |
| 2   | Matplotlib + Inkscape | Ali (Inkscape)        | R2-10             |
| 3   | ParaView              | Ali (ParaView)        | R2-10             |
| 4   | Diagram (Inkscape)    | Ali (Inkscape)        | —                 |
| 5   | ParaView              | Ali (ParaView)        | R2-10             |
| 6   | Matplotlib            | keep as-is (data unavailable) | —          |
| 7   | Matplotlib            | keep as-is (data unavailable) | —          |
| 8   | ParaView              | Ali (ParaView)        | R1-d              |

The shared style module `scripts/figstyle/figstyle.py` is applied to
every new matplotlib figure (Figs A, B, C) so they match each other.

## Shared style rules (apply to every figure)

- **No title inside the figure** — the LaTeX caption is the title.
- **Font**: sans-serif; axis labels 9–10 pt; tick labels 8–9 pt; legend
  9 pt.
- **No orange text or lines** (Reviewer 2 flagged orange on Fig 1).
- **Colour**: use the Wong 2011 colour-blind-safe palette in
  `figstyle.py` (`palette["blue"]`, `["vermillion"]`, `["green"]`, …).
  For MALMO variants use the fixed mapping in
  `figstyle.framework_colour`.
- **Legend**: never covers any data element.  For matplotlib scripts
  `place_legend(ax)` picks the emptiest corner automatically; for
  ParaView / Inkscape figures you place it manually in an empty
  region.
- **DPI**: 300+ for print.  Every matplotlib fig is exported at
  `savefig.dpi = 320` by default via `figstyle.setup()`.
- **Prefer vector**: PDF for matplotlib + Inkscape; PNG only when
  the source is ParaView (with dpi ≥ 300).

---

## Fig 1 — `fig:morton` (sec4) — Inkscape

Reviewer said "many small indices and labels that are difficult to
read (and orange font is also difficult to read)".

Edit checklist for the Inkscape `.svg` sources:
- [ ] Every text element ≥ 8 pt at final print size.
- [ ] Remove all orange fill/stroke; replace with dark grey (#333333)
      or Wong blue (#0072B2).
- [ ] Enlarge subscripts / superscripts on cell indices to at least
      6 pt.
- [ ] Remove any embedded figure title; the LaTeX caption handles it.
- [ ] Re-export at 300+ dpi PDF.
- [ ] Confirm text is embedded (Inkscape → "Convert text to path" is
      NOT recommended for CMAME; use "PDF/X-4" if the class allows).

## Fig 2 — `fig:malmo` (sec4) — Inkscape

Same edit checklist as Fig 1 above, plus:
- [ ] Colour of dotted lines: use dark grey (#666666), not orange.
- [ ] For the 3×3×3 neighbourhood highlight, use a Wong-blue
      translucent fill (`#0072B2` at α=0.2) with a solid outline.

## Fig 3 — `fig:tracking_overview` (sec7) — ParaView

Reviewer said "very small axes and color bars and the contour lots
is difficult to see (why so small and disorganized, with different
camera angles?)".

Rebuild checklist:
- [ ] **Unify camera angle across all 5 snapshots** — pick one
      informative angle (e.g. -y axis view showing the workpiece
      top surface + weld path) and reuse it verbatim for steps
      1, 350, 900, 1300, 2600.  In ParaView, save the current
      camera position (View → Camera → Save current) once and
      Apply it to every timestep before the screenshot.
- [ ] Colour bar:
    - Position: fixed to the right of the render window at all
      timesteps (do not let ParaView auto-place it).
    - Font size: axis label ≥ 12 pt, tick label ≥ 10 pt
      (ParaView defaults are much too small for print).
    - Width: 4% of the render window height; length: 60% of the
      render window height.
    - Colormap: perceptually-uniform (viridis or cividis).  Legacy
      "cool-warm" is OK if the scalar is signed and centred on 0;
      otherwise avoid.
- [ ] Axes labels: enlarge to ≥ 12 pt (View → Axes / Corner axes).
- [ ] Background: solid white (Advanced → Background) so the print
      looks clean.
- [ ] Panel layout: arrange the 5 snapshots as a **single-row** LaTeX
      subfigure so they read left-to-right in time-order.  Alternative:
      a 2 (top row 1, 350) × 3 (bottom row 900, 1300, 2600) grid.
- [ ] Screenshot at 1600×1200 px minimum; File → Save Screenshot
      with "Transparent background" OFF and "Override colour palette"
      set to "PrintBackground".

Reference ParaView state file (starting point):
`scripts/figstyle/paraview_states/fig3_tracking_state.md` (this doc)
— once you dial in a good camera in ParaView, save it as
`fig3_tracking_state.pvsm` and check it in.

## Fig 4 — `fig:rk4_pipeline` (sec7) — Inkscape (not flagged)

The RK4 pipeline diagram is not on Reviewer 2's flagged list.  Keep as-is
unless you notice font / colour inconsistencies with the other figures
during the R2-10 pass.  If updated, apply the same shared-style rules
(no orange, sans-serif ≥ 8 pt, no title).

## Fig 5 — `fig:spatial_deviation` (sec7) — ParaView

Reviewer said "dense and the error color scale is difficult to read".

Rebuild checklist:
- [ ] Colormap: perceptually-uniform (**viridis** for
      "small is good", **cividis** for accessible-print).  Do NOT
      use the ParaView default rainbow.
- [ ] Colour bar scale: **log** (Colour Map Editor → Use log scale
      when mapping data to colours).  The error field spans multiple
      orders of magnitude and log spreads it visually.
- [ ] Colour bar font size: label ≥ 12 pt, ticks ≥ 10 pt.  Width:
      4% of render window height.
- [ ] Camera: same angle as Fig 3 for reader continuity.
- [ ] Background: solid white.
- [ ] Screenshot ≥ 1600×1200 at 300+ dpi.

## Fig 6 — `fig:error_histogram` (sec7) — matplotlib (keep as-is)

Data not available for regeneration.  Existing `fig_error_histogram_sec7.pdf`
is already vector so R1-d does not apply.  Optionally: expand the
caption per R1-c to enumerate what the bins represent (error norm
$\|\bm{x}_{\text{JT}}-\bm{x}_{\text{FEMUSS}}\|_2$, particle count on
$y$, etc.).

## Fig 7 — `fig:trajectory_comparison` (sec7) — matplotlib (keep as-is)

Same: data unavailable, PDF is vector, R1-d does not apply.  Optional
caption polish for R1-c.

## Fig 8 — `fig:yz_crosssection` (sec7) — ParaView (R1-d)

Reviewer 1 flagged pixelation of the raster images in sec7.

Rebuild checklist:
- [ ] Re-export each of the two YZ-clipped snapshots at ≥ 300 dpi.
- [ ] Enforce identical camera + clip parameters between the FEMUSS
      and JAXTrace panels so the side-by-side comparison is
      unambiguous.
- [ ] Colour bar: fixed position, font size ≥ 10 pt, viridis or
      cividis colormap; label the units on the bar.
- [ ] Add a small "FEMUSS" / "JAXTrace" text label in the top-left
      corner of each panel (10 pt sans-serif, dark grey), or handle
      the labelling in the LaTeX subfigure captions.

---

## New figures added (all matplotlib, done)

- **Fig A — `fig_batch_scaling_FSW_paper.pdf`** (+ Kim counterpart)
  Kernel-only throughput vs batch size for the R2-7 response.
  Log-log; 3 MALMO variants; no title.  See
  `scripts/scone_bench/build_batch_scaling_plot.py`.

- **Fig B — `fig_mesh_cohort_sec6.pdf`**
  Horizontal bar chart of the 10-mesh cohort tet counts, coloured by
  structural class (octree-aligned / general-unstructured / strongly-
  irregular), annotated with fill_ratio.  See
  `scripts/figstyle/fig_mesh_cohort_sec6.py`.

- **Fig C — `fig_warmup_speedup_sec6.pdf`**
  10×3 heatmap of the cold-vs-warm GPU speedup from the R2-7
  T1–T5 re-sweep.  Log-normalised colour scale; per-cell speedup
  ratio drawn as text.  See
  `scripts/figstyle/fig_warmup_speedup_sec6.py`.

All three use `figstyle.setup()` for a consistent look.
