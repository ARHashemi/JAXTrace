import plotly.graph_objects as go
import numpy as np
import json

def morton2d(ix, iy, bits=4):
    code = 0
    for i in range(bits):
        code |= ((ix >> i) & 1) << (2 * i)
        code |= ((iy >> i) & 1) << (2 * i + 1)
    return code

# ── Build mesh (same geometry as all other figures) ──────────────────────────
triangles = []
for row in range(4):
    for col in range(4):
        if row >= 2 and col < 2:
            continue
        x0, y0 = col, row
        triangles.append([(x0, y0), (x0+1, y0), (x0, y0+1)])
        triangles.append([(x0+1, y0), (x0+1, y0+1), (x0, y0+1)])

for row in range(2):
    for col in range(2):
        for sr in range(2):
            for sc in range(2):
                x0 = col + sc * 0.5
                y0 = (row + 2) + sr * 0.5
                triangles.append([(x0, y0), (x0+0.5, y0), (x0, y0+0.5)])
                triangles.append([(x0+0.5, y0), (x0+0.5, y0+0.5), (x0, y0+0.5)])

is_refined = [
    (np.mean([v[0] for v in t]) < 2) and (np.mean([v[1] for v in t]) >= 2)
    for t in triangles
]
centroids = np.array([np.mean(t, axis=0) for t in triangles])

# ── Morton encoding (single-level: centroid-only registration) ────────────────
N = 16
cx_idx = np.clip((centroids[:, 0] / 4 * N).astype(int), 0, N - 1)
cy_idx = np.clip((centroids[:, 1] / 4 * N).astype(int), 0, N - 1)
morton_codes = np.array([morton2d(cx_idx[i], cy_idx[i]) for i in range(len(centroids))])
order = np.argsort(morton_codes)
sorted_cents = centroids[order]
sorted_codes = morton_codes[order]

# ── Print-ready colours ───────────────────────────────────────────────────────
COARSE_FILL   = "rgba(210,228,248,0.70)"   # light blue for coarse triangles
REFINED_FILL  = "rgba(255,235,180,0.70)"   # light amber for refined triangles
EDGE_COL      = "rgba(80,120,180,0.85)"    # blue-grey edge
CURVE_COL     = "#c0392b"                  # red Morton curve (readable on white)
CENTROID_COL  = "#c0392b"
CODE_COL      = "#7f0000"                  # dark red for code labels
REFBOX_COL    = "#b03060"                  # plum for refined-region box

fig = go.Figure()

# ── Draw mesh triangles ───────────────────────────────────────────────────────
for i, tri in enumerate(triangles):
    xs = [v[0] for v in tri] + [tri[0][0]]
    ys = [v[1] for v in tri] + [tri[0][1]]
    fill = REFINED_FILL if is_refined[i] else COARSE_FILL
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", fill="toself",
        fillcolor=fill, line=dict(color=EDGE_COL, width=0.9),
        showlegend=False, hoverinfo="skip"
    ))

# ── Morton curve through sorted centroids ────────────────────────────────────
# Curve line
fig.add_trace(go.Scatter(
    x=sorted_cents[:, 0], y=sorted_cents[:, 1],
    mode="lines",
    line=dict(color=CURVE_COL, width=2.0, dash="dot"),
    name="Morton curve (Z-order)",
    hoverinfo="skip"
))
# Centroid dots
fig.add_trace(go.Scatter(
    x=sorted_cents[:, 0], y=sorted_cents[:, 1],
    mode="markers",
    marker=dict(symbol="circle", size=7, color=CURVE_COL,
                line=dict(color="white", width=1.0)),
    name="Element centroid",
    hoverinfo="skip"
))

# ── Morton code labels (sparse, every ~8 centroids) ──────────────────────────
label_step = max(1, len(sorted_cents) // 10)
for k in range(0, len(sorted_cents), label_step):
    fig.add_annotation(
        x=sorted_cents[k, 0], y=sorted_cents[k, 1],
        text=str(sorted_codes[k]),
        showarrow=False, font=dict(size=9, color=CODE_COL),
        xshift=10, yshift=10
    )

# ── Refined region boundary ───────────────────────────────────────────────────
fig.add_shape(type="rect", x0=0, y0=2, x1=2, y1=4,
              line=dict(color=REFBOX_COL, width=2.0, dash="dash"))
fig.add_annotation(
    x=1.0, y=4.22,
    text=r"Refined region (level $\ell+1$, $\delta=0.5$)",
    showarrow=False, font=dict(size=10, color=REFBOX_COL)
)

# ── Morton discontinuity callout ──────────────────────────────────────────────
# Annotate a pair of spatially adjacent centroids near the refinement boundary
# that are far apart in the Morton ordering — the core failure of the method
boundary_sorted_idxs = [k for k in range(len(sorted_cents))
                        if abs(sorted_cents[k, 0] - 1.9) < 0.5
                        and abs(sorted_cents[k, 1] - 2.1) < 0.5]
if len(boundary_sorted_idxs) >= 2:
    i1, i2 = boundary_sorted_idxs[0], boundary_sorted_idxs[-1]
    fig.add_annotation(
        x=sorted_cents[i1, 0], y=sorted_cents[i1, 1],
        ax=sorted_cents[i2, 0], ay=sorted_cents[i2, 1],
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=2, arrowwidth=2.0, arrowcolor="#2ecc71",
        showarrow=True, text=""
    )
    fig.add_annotation(
        x=(sorted_cents[i1, 0] + sorted_cents[i2, 0]) / 2,
        y=(sorted_cents[i1, 1] + sorted_cents[i2, 1]) / 2 + 0.22,
        text=f"Morton gap<br>codes {sorted_codes[i1]}→{sorted_codes[i2]}",
        showarrow=False,
        font=dict(size=9, color="#1a7a40"),
        bgcolor="rgba(240,255,240,0.85)",
        bordercolor="#2ecc71", borderwidth=1
    )

# ── Legend entries for coarse / refined ──────────────────────────────────────
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers",
    marker=dict(size=12, symbol="square", color=COARSE_FILL,
                line=dict(color=EDGE_COL, width=1)),
    name="Coarse element (level $\\ell$)"
))
fig.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers",
    marker=dict(size=12, symbol="square", color=REFINED_FILL,
                line=dict(color=EDGE_COL, width=1)),
    name="Fine element (level $\\ell+1$)"
))

# ── Panel label (a) ───────────────────────────────────────────────────────────
fig.add_annotation(
    x=0.01, y=0.99, xref="paper", yref="paper",
    text="<b>(a)</b>", showarrow=False,
    font=dict(size=14, color="black"),
    xanchor="left", yanchor="top"
)

# ── Layout — white/print-ready ────────────────────────────────────────────────
fig.update_layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
    xaxis=dict(
        title_text="<i>x</i>",
        range=[-0.2, 4.4], scaleanchor="y",
        title_font=dict(size=12), tickfont=dict(size=10),
        showgrid=True, gridcolor="rgba(200,200,200,0.5)",
        zeroline=False, linecolor="black", mirror=True
    ),
    yaxis=dict(
        title_text="<i>y</i>",
        range=[-0.3, 4.55],
        title_font=dict(size=12), tickfont=dict(size=10),
        showgrid=True, gridcolor="rgba(200,200,200,0.5)",
        zeroline=False, linecolor="black", mirror=True
    ),
    showlegend=True,
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="center", x=0.5,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(180,180,180,0.7)", borderwidth=1
    ),
    margin=dict(t=80, b=50, l=55, r=30),
    width=900, height=820,
)

fname = "fig_morton_curve_sec3.png"
fig.write_image(fname, scale=2)
with open(fname + ".meta.json", "w") as f:
    json.dump({
        "caption": (
            "Fig. 3.1 — Morton (Z-order) curve threading element centroids "
            "on a two-level adaptive mesh (2-D schematic). "
            "Each centroid is registered in exactly one cell (single-cell registration). "
            "Red dotted curve: Z-order traversal; numbered labels: Morton codes. "
            "Coarse elements (blue) use cell size delta=1; "
            "fine elements in the refined region (amber, dashed box) use delta=0.5. "
            "The green arrow highlights a Morton discontinuity near the refinement boundary: "
            "two spatially adjacent centroids on either side of the coarse-fine interface "
            "can differ by many positions in the 1D Morton ordering, "
            "so a radius-R band search may miss the correct element entirely."
        ),
        "description": (
            "Print-ready 2D schematic for Section 3.4 (Taxonomy). "
            "White background. Morton curve in red. "
            "Coarse elements light blue, fine elements light amber."
        )
    }, f)

print("✅ fig_morton_curve_sec3.png saved")