import plotly.graph_objects as go
import numpy as np

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
        cells_L.append({"x0":col, "y0":row, "size":1.0, "cx":col+0.5, "cy":row+0.5})
for row in range(4):
    for col in range(4):
        x0=col*0.5; y0=2+row*0.5
        cells_L1.append({"x0":x0, "y0":y0, "size":0.5, "cx":x0+0.25, "cy":y0+0.25})

def cell_exists_L(col, row):
    return any(c for c in cells_L if c["x0"]==col and c["y0"]==row)

def cell_exists_L1(x0, y0):
    return any(c for c in cells_L1 if abs(c["x0"]-x0)<0.01 and abs(c["y0"]-y0)<0.01)

# ── Query point: inside refined region, near coarse boundary ──────────────────
# This is the interesting case: ℓ+1 window crosses the boundary (some absent),
# and the ℓ window has the centre cell refined away (absent).
qpos = (1.85, 2.15)

# ── Level ℓ+1 neighbourhood (3×3 on fine grid, h=0.5) ────────────────────────
fc_col = int(np.floor(qpos[0] / 0.5))   # = 3
fc_row = int(np.floor((qpos[1]-2.0)/0.5))  # = 0
fc_x0 = fc_col * 0.5   # = 1.5
fc_y0 = 2.0 + fc_row * 0.5  # = 2.0

nb_L1 = []
for dc in range(-1, 2):
    for dr in range(-1, 2):
        nx0 = fc_x0 + dc*0.5
        ny0 = fc_y0 + dr*0.5
        exists = cell_exists_L1(nx0, ny0)
        nb_L1.append({"x0":nx0, "y0":ny0, "size":0.5,
                      "cx":nx0+0.25, "cy":ny0+0.25,
                      "exists":exists, "dc":dc, "dr":dr,
                      "center":(dc==0 and dr==0)})

# ── Level ℓ neighbourhood (3×3 on coarse grid, h=1.0) ────────────────────────
cc = int(np.floor(qpos[0] / 1.0))  # = 1
cr = int(np.floor(qpos[1] / 1.0))  # = 2

nb_L = []
for dc in range(-1, 2):
    for dr in range(-1, 2):
        nc_col, nc_row = cc+dc, cr+dr
        in_domain = (0 <= nc_col < 4) and (0 <= nc_row < 4)
        exists = in_domain and cell_exists_L(nc_col, nc_row)
        nb_L.append({"x0":nc_col, "y0":nc_row, "size":1.0,
                     "cx":nc_col+0.5, "cy":nc_row+0.5,
                     "exists":exists, "in_domain":in_domain,
                     "dc":dc, "dr":dr,
                     "center":(dc==0 and dr==0)})

# ── Helper: index label string ────────────────────────────────────────────────
def idx_label(base_i, base_j, level_str, dc, dr):
    def fmt(base, d):
        if d == 0:   return base
        if d == 1:   return f"{base}+1"
        if d == -1:  return f"{base}−1"   # −1
        if d > 1:    return f"{base}+{d}"
        return f"{base}{d}"
    i_str = fmt(base_i, dc)
    j_str = fmt(base_j, dr)
    return f"({i_str},{j_str},{level_str})"

# ── Figure ────────────────────────────────────────────────────────────────────
fig = go.Figure()

# Mesh triangles — plain fill, thin edge, NO highlighted element tinting
for i, tri in enumerate(triangles):
    xs=[v[0] for v in tri]+[tri[0][0]]
    ys=[v[1] for v in tri]+[tri[0][1]]
    fill = "rgba(26,58,92,0.85)" if is_refined[i] else "rgba(13,32,53,0.85)"
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself",
                             fillcolor=fill,
                             line=dict(color="rgba(74,144,217,0.18)", width=0.5),
                             showlegend=False, hoverinfo="skip"))

# Background cell grid outlines (very faint)
for c in cells_L:
    x0,y0,s = c["x0"],c["y0"],c["size"]
    fig.add_trace(go.Scatter(x=[x0,x0+s,x0+s,x0,x0], y=[y0,y0,y0+s,y0+s,y0],
                             mode="lines",
                             line=dict(color="rgba(50,180,255,0.18)", width=0.7),
                             showlegend=False, hoverinfo="skip"))
for c in cells_L1:
    x0,y0,s = c["x0"],c["y0"],c["size"]
    fig.add_trace(go.Scatter(x=[x0,x0+s,x0+s,x0,x0], y=[y0,y0,y0+s,y0+s,y0],
                             mode="lines",
                             line=dict(color="rgba(240,200,50,0.18)", width=0.7),
                             showlegend=False, hoverinfo="skip"))

# ─────────────────────────────────────────────────────────────────────────────
# LEVEL ℓ+1 neighbourhood
# Four categories drawn in z-order: absent first (behind), present on top
# ─────────────────────────────────────────────────────────────────────────────
leg = {"L1_present": True, "L1_absent": True, "L_present": True, "L_absent": True}

# ℓ+1 absent slots (fine cells that would be neighbours but don't exist)
for nb in nb_L1:
    if nb["exists"]:
        continue
    x0,y0,s = nb["x0"],nb["y0"],nb["size"]
    xs=[x0,x0+s,x0+s,x0,x0]; ys=[y0,y0,y0+s,y0+s,y0]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself",
                             fillcolor="rgba(240,200,50,0.00)",
                             line=dict(color="#c8a000", width=1.4, dash="dot"),
                             name="level ℓ+1, absent" if leg["L1_absent"] else None,
                             showlegend=leg["L1_absent"], hoverinfo="skip"))
    leg["L1_absent"] = False

# ℓ+1 present slots
for nb in nb_L1:
    if not nb["exists"]:
        continue
    x0,y0,s = nb["x0"],nb["y0"],nb["size"]
    xs=[x0,x0+s,x0+s,x0,x0]; ys=[y0,y0,y0+s,y0+s,y0]
    lw = 2.8 if nb["center"] else 2.0
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself",
                             fillcolor="rgba(240,200,50,0.22)",
                             line=dict(color="#f0c832", width=lw),
                             name="level ℓ+1, present" if leg["L1_present"] else None,
                             showlegend=leg["L1_present"], hoverinfo="skip"))
    leg["L1_present"] = False

# ── ℓ+1 bounding box
bb1_x0 = fc_x0 - 0.5; bb1_x1 = fc_x0 + 1.0
bb1_y0 = fc_y0 - 0.5; bb1_y1 = fc_y0 + 1.0
fig.add_shape(type="rect", x0=bb1_x0, y0=bb1_y0, x1=bb1_x1, y1=bb1_y1,
              line=dict(color="#f0c832", width=2.0, dash="dash"),
              layer="above")
fig.add_annotation(x=(bb1_x0+bb1_x1)/2, y=bb1_y1+0.08,
                   text="3×3 window (ℓ+1)", showarrow=False,
                   font=dict(size=9, color="#917400", family="Arial"),
                   bgcolor="rgba(255,255,255,0.82)", borderpad=2)

# ── ℓ+1 cell-index labels (all 9 slots, inside or outside the window)
for nb in nb_L1:
    label = idx_label("i′", "j′", "ℓ+1", nb["dc"], nb["dr"])
    col = "#f0c832" if nb["exists"] else "#b09020"
    fig.add_annotation(x=nb["cx"], y=nb["cy"], text=label,
                       showarrow=False,
                       font=dict(size=8, color=col, family="Arial"),
                       bgcolor="rgba(255,255,255,0.0)")

# ─────────────────────────────────────────────────────────────────────────────
# LEVEL ℓ neighbourhood
# ─────────────────────────────────────────────────────────────────────────────

# ℓ absent slots (refined away or out of domain)
for nb in nb_L:
    if nb["exists"] or not nb["in_domain"]:
        continue
    x0,y0 = nb["x0"],nb["y0"]
    xs=[x0,x0+1,x0+1,x0,x0]; ys=[y0,y0,y0+1,y0+1,y0]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself",
                             fillcolor="rgba(224,92,92,0.00)",
                             line=dict(color="#e05c5c", width=1.4, dash="dot"),
                             name="level ℓ, absent (refined away)" if leg["L_absent"] else None,
                             showlegend=leg["L_absent"], hoverinfo="skip"))
    leg["L_absent"] = False

# ℓ present slots
for nb in nb_L:
    if not nb["exists"]:
        continue
    x0,y0,s = nb["x0"],nb["y0"],nb["size"]
    xs=[x0,x0+s,x0+s,x0,x0]; ys=[y0,y0,y0+s,y0+s,y0]
    lw = 2.8 if nb["center"] else 2.0
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself",
                             fillcolor="rgba(50,180,255,0.16)",
                             line=dict(color="#32b4ff", width=lw),
                             name="level ℓ, present" if leg["L_present"] else None,
                             showlegend=leg["L_present"], hoverinfo="skip"))
    leg["L_present"] = False

# ── ℓ bounding box
bbL_x0 = cc-1; bbL_x1 = cc+2
bbL_y0 = cr-1; bbL_y1 = cr+2
fig.add_shape(type="rect", x0=bbL_x0, y0=bbL_y0, x1=bbL_x1, y1=bbL_y1,
              line=dict(color="#32b4ff", width=2.0, dash="dash"),
              layer="above")
fig.add_annotation(x=(bbL_x0+bbL_x1)/2, y=bbL_y1+0.08,
                   text="3×3 window (ℓ)", showarrow=False,
                   font=dict(size=9, color="#005f7a", family="Arial"),
                   bgcolor="rgba(255,255,255,0.82)", borderpad=2)

# ── ℓ cell-index labels for existing cells only (to avoid cluttering absent slots)
for nb in nb_L:
    if not nb["in_domain"]:
        continue
    label = idx_label("i", "j", "ℓ", nb["dc"], nb["dr"])
    col = "#32b4ff" if nb["exists"] else "#e05c5c"
    fig.add_annotation(x=nb["cx"], y=nb["cy"], text=label,
                       showarrow=False,
                       font=dict(size=8, color=col, family="Arial"),
                       bgcolor="rgba(255,255,255,0.0)")

# Refined-region boundary
fig.add_shape(type="rect", x0=0, y0=2, x1=2, y1=4,
              line=dict(color="#e05c5c", width=1.4, dash="dash"),
              layer="above")

# Query point
fig.add_trace(go.Scatter(x=[qpos[0]], y=[qpos[1]],
                         mode="markers+text",
                         marker=dict(symbol="x", size=14, color="#e05c5c",
                                     line=dict(width=3.0, color="#e05c5c")),
                         text=["q"], textposition="top right",
                         textfont=dict(size=11, color="#e05c5c", family="Arial"),
                         name="query point q", showlegend=True,
                         hoverinfo="skip"))

# ── Layout ────────────────────────────────────────────────────────────────────
fig.update_layout(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Arial", size=11, color="black"),
    margin=dict(l=50, r=12, t=22, b=45),
    showlegend=True,
    legend=dict(x=0.53, y=0.98, xanchor="left", yanchor="top",
                orientation="v",
                bgcolor="rgba(255,255,255,0.90)",
                bordercolor="rgba(0,0,0,0.20)", borderwidth=1,
                font=dict(size=9, color="black"),
                tracegroupgap=2),
    width=500, height=500,
)
fig.update_xaxes(title_text="x", range=[-0.05, 4.05],
                 showgrid=False, zeroline=False,
                 showline=True, linewidth=1.3, linecolor="black", mirror=True,
                 ticks="outside", tickfont=dict(size=10), title_font=dict(size=11),
                 scaleanchor="y", constrain="domain")
fig.update_yaxes(title_text="y", range=[-0.05, 4.05],
                 showgrid=False, zeroline=False,
                 showline=True, linewidth=1.3, linecolor="black", mirror=True,
                 ticks="outside", tickfont=dict(size=10), title_font=dict(size=11),
                 constrain="domain")

img_bytes = fig.to_image(format="svg", scale=2)
with open("fig_malmo_3x3_sec4.svg", "wb") as f:
    f.write(img_bytes)
print("Fig B saved: fig_malmo_3x3_sec4.pdf")
