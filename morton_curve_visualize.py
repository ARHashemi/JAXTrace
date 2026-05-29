
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import json

# ── helpers ──────────────────────────────────────────────────────────────────
def morton2d(ix, iy, bits=4):
    """Interleave bits of ix and iy to get a 2D Morton code."""
    code = 0
    for i in range(bits):
        code |= ((ix >> i) & 1) << (2 * i)
        code |= ((iy >> i) & 1) << (2 * i + 1)
    return code

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE A – Morton curve through triangle centroids on a 2-D refined mesh
# ─────────────────────────────────────────────────────────────────────────────

# Build a simple 2-D triangulated mesh (uniform + one refined region)
# Base grid 4×4 cells; each cell split into 2 triangles
# Then refine the top-left 2×2 block into 4 sub-cells each
np.random.seed(42)

triangles = []   # list of (v0,v1,v2) in (x,y)
# Coarse cells (skip top-left 2×2)
for row in range(4):
    for col in range(4):
        if row >= 2 and col < 2:   # will be refined
            continue
        x0, y0 = col, row
        # Two triangles per cell
        triangles.append([(x0, y0), (x0+1, y0), (x0, y0+1)])
        triangles.append([(x0+1, y0), (x0+1, y0+1), (x0, y0+1)])

# Refined region (rows 2-3, cols 0-1) → 0.5-spaced grid
for row in range(2):
    for col in range(2):
        for sr in range(2):
            for sc in range(2):
                x0 = col + sc * 0.5
                y0 = (row + 2) + sr * 0.5
                triangles.append([(x0, y0), (x0+0.5, y0), (x0, y0+0.5)])
                triangles.append([(x0+0.5, y0), (x0+0.5, y0+0.5), (x0, y0+0.5)])

# Compute centroids
centroids = np.array([np.mean(t, axis=0) for t in triangles])

# Assign Morton codes based on a normalized grid position
# Map centroids to integer grid indices (use 4-bit = 16 cells per axis)
N = 16
cx_idx = np.clip((centroids[:, 0] / 4 * N).astype(int), 0, N - 1)
cy_idx = np.clip((centroids[:, 1] / 4 * N).astype(int), 0, N - 1)
morton_codes = np.array([morton2d(cx_idx[i], cy_idx[i]) for i in range(len(centroids))])

# Sort centroids by Morton code
order = np.argsort(morton_codes)
sorted_cents = centroids[order]
sorted_codes = morton_codes[order]

# ── determine colour per triangle (coarse vs refined) ──────────────────────
is_refined = [
    (np.mean([v[0] for v in t]) < 2) and (np.mean([v[1] for v in t]) >= 2)
    for t in triangles
]

fig_a = go.Figure()

# Draw triangles
for i, tri in enumerate(triangles):
    xs = [v[0] for v in tri] + [tri[0][0]]
    ys = [v[1] for v in tri] + [tri[0][1]]
    fill = "#1a3a5c" if is_refined[i] else "#0d2035"
    line_col = "#4a90d9"
    fig_a.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", fill="toself",
        fillcolor=fill, line=dict(color=line_col, width=1.2),
        showlegend=False, hoverinfo="skip"
    ))

# Draw Morton curve through sorted centroids
fig_a.add_trace(go.Scatter(
    x=sorted_cents[:, 0], y=sorted_cents[:, 1],
    mode="lines+markers",
    line=dict(color="#f0a500", width=2.5, dash="dot"),
    marker=dict(symbol="circle", size=9, color="#f0a500",
                line=dict(color="white", width=1.2)),
    name="Morton curve",
    hovertemplate="Code: %{customdata}<br>(%{x:.2f}, %{y:.2f})<extra></extra>",
    customdata=sorted_codes
))

# Annotate a few Morton codes along the curve
label_step = max(1, len(sorted_cents) // 10)
for k in range(0, len(sorted_cents), label_step):
    fig_a.add_annotation(
        x=sorted_cents[k, 0], y=sorted_cents[k, 1],
        text=f"<b>{sorted_codes[k]}</b>",
        showarrow=False, font=dict(size=14, color="#f0a500"),
        xshift=12, yshift=12
    )

# Mark refined region boundary
fig_a.add_shape(type="rect", x0=0, y0=2, x1=2, y1=4,
                line=dict(color="#e05c5c", width=3, dash="dash"))
fig_a.add_annotation(x=1, y=4.2, text="<b>Refined region</b>", showarrow=False,
                     font=dict(size=16, color="#e05c5c"))

fig_a.update_layout(
    title=dict(text="Morton Curve through Triangle Centroids<br>"
               "<span style='font-size:18px;font-weight:normal;'>"
               "Z-order (Morton) traversal on 2-D mesh with local refinement</span>",
               font=dict(size=24)),
    xaxis=dict(title_text="x", range=[-0.15, 4.3], scaleanchor="y",
               title_font=dict(size=18), tickfont=dict(size=14)),
    yaxis=dict(title_text="y", range=[-0.3, 4.5],
               title_font=dict(size=18), tickfont=dict(size=14)),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5,
                font=dict(size=16)),
    width=1400,
    height=1050,
)

fig_a.update_layout(
    title=None,
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Arial", size=18, color="black"),
    margin=dict(l=18, r=18, t=18, b=18, pad=0),
    showlegend=True,
    legend=dict(
        x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        orientation="v",
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor="rgba(0,0,0,0.25)",
        borderwidth=1,
        font=dict(size=15, color="black")
    )
)

fig_a.update_xaxes(
    title_text="x",
    range=[-0.02, 4.02],
    showgrid=False,
    zeroline=False,
    showline=True,
    linewidth=1.6,
    linecolor="black",
    mirror=True,
    ticks="outside",
    tickfont=dict(size=15, color="black"),
    title_font=dict(size=18, color="black"),
    scaleanchor="y",
    constrain="domain"
)

fig_a.update_yaxes(
    title_text="y",
    range=[-0.02, 4.02],
    showgrid=False,
    zeroline=False,
    showline=True,
    linewidth=1.6,
    linecolor="black",
    mirror=True,
    ticks="outside",
    tickfont=dict(size=15, color="black"),
    title_font=dict(size=18, color="black"),
    constrain="domain"
)

# Write with 300 DPI
fig_a.write_image("morton_curve_mesh_hires.pdf", scale=2)

with open("morton_curve_mesh_hires.png.meta.json", "w") as f:
    json.dump({
        "caption": "Fig A – Morton (Z-order) curve threading triangle centroids on a 2-D mesh with a refined region (red dashed box). Yellow dots are sorted centroids; orange labels show Morton codes.",
        "description": "2D triangulated mesh with coarse and refined regions. The Morton curve traverses all triangle centroids in Z-order, illustrated by a dotted yellow line."
    }, f)

print("Fig A (high-res) saved.")
print(f"Triangles: {len(triangles)}, sorted centroids: {len(sorted_cents)}")
print(f"Figure size: 1400×1050 pixels at scale=2 (effective 2800×2100 pixels, ~300 DPI at 9.3×7 inches)")