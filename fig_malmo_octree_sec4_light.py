import plotly.graph_objects as go
import numpy as np

def morton2d(ix, iy, bits=4):
    code = 0
    for i in range(bits):
        code |= ((ix >> i) & 1) << (2 * i)
        code |= ((iy >> i) & 1) << (2 * i + 1)
    return code

# ── Mesh ──────────────────────────────────────────────────────────────────────
triangles = []
for row in range(4):
    for col in range(4):
        if row >= 2 and col < 2:
            continue
        x0, y0 = col, row
        triangles.append([(x0,y0),(x0+1,y0),(x0,y0+1)])
        triangles.append([(x0+1,y0),(x0+1,y0+1),(x0,y0+1)])
for row in range(2):
    for col in range(2):
        for sr in range(2):
            for sc in range(2):
                x0 = col + sc*0.5; y0 = (row+2) + sr*0.5
                triangles.append([(x0,y0),(x0+0.5,y0),(x0,y0+0.5)])
                triangles.append([(x0+0.5,y0),(x0+0.5,y0+0.5),(x0,y0+0.5)])

is_refined = [(np.mean([v[0] for v in t]) < 2) and (np.mean([v[1] for v in t]) >= 2)
              for t in triangles]

# ── Two-level cells ───────────────────────────────────────────────────────────
cells_L, cells_L1 = [], []
for row in range(4):
    for col in range(4):
        if row >= 2 and col < 2:
            continue
        cells_L.append({"x0":col,"y0":row,"size":1.0,"cx":col+0.5,"cy":row+0.5,"level":"L"})
for row in range(4):
    for col in range(4):
        x0=col*0.5; y0=2+row*0.5
        cells_L1.append({"x0":x0,"y0":y0,"size":0.5,"cx":x0+0.25,"cy":y0+0.25,"level":"L+1"})

N_L, N_L1 = 8, 16
for c in cells_L:
    ix = int(np.clip(c["cx"]/4*N_L,  0, N_L -1))
    iy = int(np.clip(c["cy"]/4*N_L,  0, N_L -1))
    c["morton"] = morton2d(ix, iy); c["gix"]=ix; c["giy"]=iy
for c in cells_L1:
    ix = int(np.clip(c["cx"]/4*N_L1, 0, N_L1-1))
    iy = int(np.clip(c["cy"]/4*N_L1, 0, N_L1-1))
    c["morton"] = morton2d(ix, iy); c["gix"]=ix; c["giy"]=iy

all_sorted = sorted(cells_L + cells_L1,
                    key=lambda c: (c["morton"], 0 if c["level"]=="L" else 1))
sorted_cx  = [c["cx"]    for c in all_sorted]
sorted_cy  = [c["cy"]    for c in all_sorted]
sorted_lvl = [c["level"] for c in all_sorted]

# ── Colour palette (journal light style) ─────────────────────────────────────
# Mesh: very faint warm-grey tint; edges dark grey
COARSE_MESH_FILL = "rgba(215,225,235,0.55)"   # pale blue-grey for coarse region
FINE_MESH_FILL   = "rgba(215,230,215,0.55)"   # pale green-grey for refined region
MESH_EDGE        = "rgba(90,110,130,0.50)"    # mid slate, thin

# Cell outlines
CELL_L_EDGE   = "#1a6fa8"   # strong blue  — level ℓ
CELL_L_FILL   = "rgba(26,111,168,0.10)"
CELL_L1_EDGE  = "#b07800"   # dark amber   — level ℓ+1
CELL_L1_FILL  = "rgba(176,120,0,0.10)"

# Morton curve
MORTON_LINE   = "rgba(140,140,140,0.60)"
MORTON_DOT_L  = "#1a6fa8"
MORTON_DOT_L1 = "#b07800"

# Annotations
RED_DASH      = "#c0392b"   # refined-region boundary + "no cell" label
LABEL_L       = "#1a6fa8"
LABEL_L1      = "#8a5e00"

# ── Figure ────────────────────────────────────────────────────────────────────
fig = go.Figure()

# Mesh triangles — faint tinted fill + dark-grey edges
for i, tri in enumerate(triangles):
    xs=[v[0] for v in tri]+[tri[0][0]]
    ys=[v[1] for v in tri]+[tri[0][1]]
    fill = FINE_MESH_FILL if is_refined[i] else COARSE_MESH_FILL
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself",
                             fillcolor=fill,
                             line=dict(color=MESH_EDGE, width=0.6),
                             showlegend=False, hoverinfo="skip"))

# Coarse cells (level ℓ) — blue
lc_shown = False
for c in cells_L:
    x0,y0,s = c["x0"],c["y0"],c["size"]
    xs=[x0,x0+s,x0+s,x0,x0]; ys=[y0,y0,y0+s,y0+s,y0]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself",
                             fillcolor=CELL_L_FILL,
                             line=dict(color=CELL_L_EDGE, width=1.6),
                             name="level ℓ cell" if not lc_shown else None,
                             showlegend=not lc_shown, hoverinfo="skip"))
    lc_shown = True

# Fine cells (level ℓ+1) — amber
lf_shown = False
for c in cells_L1:
    x0,y0,s = c["x0"],c["y0"],c["size"]
    xs=[x0,x0+s,x0+s,x0,x0]; ys=[y0,y0,y0+s,y0+s,y0]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself",
                             fillcolor=CELL_L1_FILL,
                             line=dict(color=CELL_L1_EDGE, width=1.6),
                             name="level ℓ+1 cell" if not lf_shown else None,
                             showlegend=not lf_shown, hoverinfo="skip"))
    lf_shown = True

# Morton traversal curve
marker_colors = [MORTON_DOT_L if lv=="L" else MORTON_DOT_L1 for lv in sorted_lvl]
fig.add_trace(go.Scatter(x=sorted_cx, y=sorted_cy, mode="lines+markers",
                         line=dict(color=MORTON_LINE, width=1.4, dash="dot"),
                         marker=dict(symbol="circle", size=6, color=marker_colors,
                                     line=dict(color="white", width=0.8)),
                         name="sorted array (Morton, level)", hoverinfo="skip"))

# ── Sparse cell-index labels ──────────────────────────────────────────────────
# Coarse: 3×2 block in lower-right (clearly coarse region)
for col, row in [(2,0),(3,0),(2,1),(3,1),(2,2),(3,2)]:
    hits = [c for c in cells_L if c["x0"]==col and c["y0"]==row]
    if hits:
        c = hits[0]
        fig.add_annotation(x=c["cx"], y=c["cy"],
                           text=f"({c['gix']},{c['giy']},ℓ)",
                           showarrow=False,
                           font=dict(size=8, color=LABEL_L, family="Arial"))

# Fine: 2×2 block at bottom-left of refined block
for fx0, fy0 in [(0.0,2.0),(0.5,2.0),(0.0,2.5),(0.5,2.5)]:
    hits = [c for c in cells_L1 if abs(c["x0"]-fx0)<0.01 and abs(c["y0"]-fy0)<0.01]
    if hits:
        c = hits[0]
        fig.add_annotation(x=c["cx"], y=c["cy"],
                           text=f"({c['gix']},{c['giy']},ℓ+1)",
                           showarrow=False,
                           font=dict(size=7, color=LABEL_L1, family="Arial"))

# Refined-region boundary
fig.add_shape(type="rect", x0=0, y0=2, x1=2, y1=4,
              line=dict(color=RED_DASH, width=1.4, dash="dash"))

# "no level-ℓ cell" label
fig.add_annotation(x=1.0, y=3.0, text="no level-ℓ cell",
                   showarrow=False,
                   font=dict(size=10, color=RED_DASH, family="Arial"),
                   bgcolor="rgba(255,255,255,0.85)",
                   bordercolor=RED_DASH, borderwidth=1)

# ── Layout ────────────────────────────────────────────────────────────────────
fig.update_layout(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Arial", size=11, color="black"),
    margin=dict(l=50, r=12, t=12, b=45),
    showlegend=True,
    legend=dict(x=0.57, y=0.04, xanchor="left", yanchor="bottom",
                orientation="v",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="rgba(0,0,0,0.20)", borderwidth=1,
                font=dict(size=10, color="black")),
    width=500, height=500,
)
fig.update_xaxes(title_text="x", range=[-0.05,4.05],
                 showgrid=False, zeroline=False,
                 showline=True, linewidth=1.3, linecolor="black", mirror=True,
                 ticks="outside", tickfont=dict(size=10), title_font=dict(size=11),
                 scaleanchor="y", constrain="domain")
fig.update_yaxes(title_text="y", range=[-0.05,4.05],
                 showgrid=False, zeroline=False,
                 showline=True, linewidth=1.3, linecolor="black", mirror=True,
                 ticks="outside", tickfont=dict(size=10), title_font=dict(size=11),
                 constrain="domain")

img_bytes = fig.to_image(format="svg", scale=2)
with open("fig_malmo_octree_sec4_light.svg", "wb") as f:
    f.write(img_bytes)
print("Fig A saved: fig_malmo_octree_sec4_light.pdf")
