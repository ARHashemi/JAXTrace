
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import json

def morton2d(ix, iy, bits=4):
    code = 0
    for i in range(bits):
        code |= ((ix >> i) & 1) << (2 * i)
        code |= ((iy >> i) & 1) << (2 * i + 1)
    return code

# ── Rebuild the same mesh/centroids as Fig A ─────────────────────────────────
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

centroids = np.array([np.mean(t, axis=0) for t in triangles])
N = 16
cx_idx = np.clip((centroids[:, 0] / 4 * N).astype(int), 0, N - 1)
cy_idx = np.clip((centroids[:, 1] / 4 * N).astype(int), 0, N - 1)
morton_codes = np.array([morton2d(cx_idx[i], cy_idx[i]) for i in range(len(centroids))])
order = np.argsort(morton_codes)
sorted_cents = centroids[order]
sorted_codes = morton_codes[order]
orig_to_sorted = {orig: pos for pos, orig in enumerate(order)}

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B – Position search: encode query → nearest centroid → radius search
# ─────────────────────────────────────────────────────────────────────────────
# Query points (4 representative positions)
queries = {
    "k₁": np.array([0.7, 0.5]),
    "k₂": np.array([2.5, 1.2]),
    "k₃": np.array([0.4, 3.1]),   # inside refined region
    "k₄": np.array([3.2, 3.6]),
}
radius = 4   # Morton-curve neighbor radius

fig_b = go.Figure()

# Draw mesh background (faint)
is_refined = [
    (np.mean([v[0] for v in t]) < 2) and (np.mean([v[1] for v in t]) >= 2)
    for t in triangles
]
for i, tri in enumerate(triangles):
    xs = [v[0] for v in tri] + [tri[0][0]]
    ys = [v[1] for v in tri] + [tri[0][1]]
    fill = "#1a3a5c" if is_refined[i] else "#0d2035"#"rgba(26,58,92,0.45)" if is_refined[i] else "rgba(13,32,53,0.45)"
    fig_b.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", fill="toself",
        fillcolor=fill, line=dict(color="rgba(74,144,217,0.4)", width=1.0),
        showlegend=False, hoverinfo="skip"
    ))

# Draw Morton curve (faint background)
fig_b.add_trace(go.Scatter(
    x=sorted_cents[:, 0], y=sorted_cents[:, 1],
    mode="lines", line=dict(color="rgba(240,165,0,0.3)", width=2.0, dash="dot"),
    showlegend=False, hoverinfo="skip"
))

# For each query point, find nearest centroid by Morton code and highlight radius
colors = ["#e05c5c", "#5ce05c", "#5cc8e0", "#cc5ce0"]
used_legend = set()

for (qname, qpos), qcol in zip(queries.items(), colors):
    # Encode query to Morton
    qix = int(np.clip(qpos[0] / 4 * N, 0, N - 1))
    qiy = int(np.clip(qpos[1] / 4 * N, 0, N - 1))
    q_code = morton2d(qix, qiy)

    # Find nearest centroid by Morton code distance
    dists = np.abs(sorted_codes - q_code)
    nearest_sorted_idx = int(np.argmin(dists))

    # Radius neighbors on Morton curve
    lo = max(0, nearest_sorted_idx - radius)
    hi = min(len(sorted_cents) - 1, nearest_sorted_idx + radius)
    neighbor_range = list(range(lo, hi + 1))

    # Highlight neighbor centroids on the curve
    nb_x = sorted_cents[neighbor_range, 0]
    nb_y = sorted_cents[neighbor_range, 1]
    lbl_nb = f"Radius={radius} neighbors" if "radius" not in used_legend else None
    used_legend.add("radius")
    fig_b.add_trace(go.Scatter(
        x=nb_x, y=nb_y, mode="markers",
        marker=dict(symbol="circle-open", size=20, color=qcol,
                    line=dict(width=3, color=qcol)),
        name=f"{qname} search window" if lbl_nb else None,
        showlegend=(lbl_nb is not None),
        hovertemplate=f"{qname} neighbor<br>(%{{x:.2f}}, %{{y:.2f}})<extra></extra>"
    ))

    # Highlight nearest centroid
    nc = sorted_cents[nearest_sorted_idx]
    fig_b.add_trace(go.Scatter(
        x=[nc[0]], y=[nc[1]], mode="markers",
        marker=dict(symbol="star", size=20, color=qcol, line=dict(width=2, color="white")),
        name=f"{qname} nearest", showlegend=True,
        hovertemplate=f"{qname} nearest<br>Morton code: {sorted_codes[nearest_sorted_idx]}<extra></extra>"
    ))

    # Draw query point
    fig_b.add_trace(go.Scatter(
        x=[qpos[0]], y=[qpos[1]], mode="markers+text",
        marker=dict(symbol="x", size=18, color=qcol,
                    line=dict(width=3, color=qcol)),
        text=[f"<b>{qname}</b>"], textposition="top right",
        textfont=dict(color=qcol, size=16),
        showlegend=False,
        hovertemplate=f"{qname} query<br>({qpos[0]:.2f},{qpos[1]:.2f})<br>Morton: {q_code}<extra></extra>"
    ))

    # Arrow: query → nearest centroid
    fig_b.add_annotation(
        x=nc[0], y=nc[1], ax=qpos[0], ay=qpos[1],
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=3, arrowwidth=2.5, arrowcolor=qcol, arrowsize=1.0,
        showarrow=True
    )

# Morton code labels at sorted centroids (sparse)
for k in range(0, len(sorted_cents), 8):
    fig_b.add_annotation(
        x=sorted_cents[k, 0], y=sorted_cents[k, 1],
        text=f"<b>{sorted_codes[k]}</b>",
        showarrow=False, font=dict(size=12, color="rgba(240,165,0,0.7)"),
        xshift=10, yshift=10
    )

# Refined region boundary
fig_b.add_shape(type="rect", x0=0, y0=2, x1=2, y1=4,
                line=dict(color="#e05c5c", width=3, dash="dash"))
fig_b.add_annotation(x=1, y=4.2, text="<b>Refined region</b>",
                     showarrow=False, font=dict(size=16, color="#e05c5c"))

fig_b.update_layout(
    title=dict(text="L2 Search: Query Encoding → Morton Nearest → Radius Check<br>"
               "<span style='font-size:18px;font-weight:normal;'>"
               "k₁–k₄ query positions encoded to Morton; ★ = nearest centroid; "
               "open circles = radius window for point-in-tet</span>",
               font=dict(size=24)),
    xaxis=dict(title_text="x", range=[-0.15, 4.3], scaleanchor="y",
               title_font=dict(size=18), tickfont=dict(size=14)),
    yaxis=dict(title_text="y", range=[-0.3, 4.5],
               title_font=dict(size=18), tickfont=dict(size=14)),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5,
                font=dict(size=14)),
    width=1400,
    height=1050,
)

fig_b.update_layout(
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

fig_b.update_xaxes(
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

fig_b.update_yaxes(
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
fig_b.write_image("morton_search_hires.svg", scale=2)

with open("morton_search_hires.png.meta.json", "w") as f:
    json.dump({
        "caption": "Fig B – L2 position search: each query point (k₁–k₄) is Morton-encoded, matched to the nearest centroid (★), then point-in-tet is checked over a ±radius window on the Morton curve (open circles).",
        "description": "2D mesh showing four query positions. Each query is mapped to the nearest centroid by Morton-code proximity. Open circles show the radius window of candidates checked for point-in-tet."
    }, f)

print("Fig B (high-res) saved.")
print(f"Figure size: 1400×1050 pixels at scale=2 (effective 2800×2100 pixels, ~300 DPI at 9.3×7 inches)")