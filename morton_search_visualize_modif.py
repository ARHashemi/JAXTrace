import plotly.graph_objects as go
import numpy as np
import json

# ── SHARED style constants (CMAME print) ─────────────────────────────────────
AX_FONT   = 11    # axis title pt
TICK_FONT = 10    # tick label pt
LEG_FONT  = 10    # legend pt
ANN_SM    = 9     # small annotation (code labels)
ANN_MD    = 10    # medium annotation
ANN_LG    = 11    # large annotation (region labels)
GRID_COL  = "rgba(200,200,200,0.45)"
LEG_BG    = "rgba(255,255,255,0.92)"
LEG_BD    = "rgba(180,180,180,0.80)"
ANN_BG    = "rgba(255,255,255,0.92)"

def morton2d(ix, iy, bits=4):
    code = 0
    for i in range(bits):
        code |= ((ix >> i) & 1) << (2 * i)
        code |= ((iy >> i) & 1) << (2 * i + 1)
    return code

# ── Shared mesh build (used by all 4 figures) ─────────────────────────────────
def build_mesh():
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
                    x0=col+sc*0.5; y0=(row+2)+sr*0.5
                    triangles.append([(x0,y0),(x0+0.5,y0),(x0,y0+0.5)])
                    triangles.append([(x0+0.5,y0),(x0+0.5,y0+0.5),(x0,y0+0.5)])
    is_refined=[(np.mean([v[0] for v in t])<2)and(np.mean([v[1] for v in t])>=2)
                for t in triangles]
    return triangles, is_refined

triangles, is_refined = build_mesh()

print("Mesh built:", len(triangles), "triangles")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — morton_curve_visualize.py  (file:168)
# STYLE ONLY: white bg, no title, CMAME font sizes
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)

centroids = np.array([np.mean(t, axis=0) for t in triangles])
N = 16
cx_idx = np.clip((centroids[:,0]/4*N).astype(int), 0, N-1)
cy_idx = np.clip((centroids[:,1]/4*N).astype(int), 0, N-1)
morton_codes = np.array([morton2d(cx_idx[i], cy_idx[i]) for i in range(len(centroids))])
order = np.argsort(morton_codes)
sorted_cents = centroids[order]
sorted_codes = morton_codes[order]

fig_a = go.Figure()

# Draw triangles — ORIGINAL fills and line colours unchanged
for i, tri in enumerate(triangles):
    xs=[v[0] for v in tri]+[tri[0][0]]; ys=[v[1] for v in tri]+[tri[0][1]]
    fill="#1a3a5c" if is_refined[i] else "#0d2035"
    fig_a.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself",
        fillcolor=fill, line=dict(color="#4a90d9", width=1.2),
        showlegend=False, hoverinfo="skip"))

# Morton curve — ORIGINAL colour unchanged
fig_a.add_trace(go.Scatter(
    x=sorted_cents[:,0], y=sorted_cents[:,1],
    mode="lines+markers",
    line=dict(color="#f0a500", width=2.5, dash="dot"),
    marker=dict(symbol="circle", size=9, color="#f0a500",
                line=dict(color="white", width=1.2)),
    name="Morton curve",
    hovertemplate="Code: %{customdata}(%{x:.2f}, %{y:.2f})",
    customdata=sorted_codes
))

# Code labels — ORIGINAL colour, reduced size for print
label_step = max(1, len(sorted_cents)//10)
for k in range(0, len(sorted_cents), label_step):
    fig_a.add_annotation(
        x=sorted_cents[k,0], y=sorted_cents[k,1],
        text=f"{sorted_codes[k]}",
        showarrow=False, font=dict(size=ANN_SM, color="#f0a500"),  # size 14→9
        xshift=12, yshift=12)

# Refined region — ORIGINAL colour unchanged
fig_a.add_shape(type="rect", x0=0, y0=2, x1=2, y1=4,
                line=dict(color="#e05c5c", width=3, dash="dash"))
fig_a.add_annotation(x=1, y=4.2, text="Refined region", showarrow=False,
    font=dict(size=ANN_LG, color="#e05c5c"))  # size 16→11

fig_a.update_layout(
    paper_bgcolor="white", plot_bgcolor="white",               # ← STYLE CHANGE
    # title removed — goes in LaTeX \caption{}                 # ← STYLE CHANGE
    xaxis=dict(title_text="x", range=[-0.15,4.3], scaleanchor="y",
               title_font=dict(size=AX_FONT), tickfont=dict(size=TICK_FONT),  # 18→11, 14→10
               showgrid=True, gridcolor=GRID_COL,
               linecolor="black", mirror=True, zeroline=False),
    yaxis=dict(title_text="y", range=[-0.3,4.5],
               title_font=dict(size=AX_FONT), tickfont=dict(size=TICK_FONT),
               showgrid=True, gridcolor=GRID_COL,
               linecolor="black", mirror=True, zeroline=False),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5,
                font=dict(size=LEG_FONT),                      # size 16→10
                bgcolor=LEG_BG, bordercolor=LEG_BD, borderwidth=1),
    width=1400, height=1050,
    margin=dict(t=55, b=60, l=70, r=40),                      # t 165→55 (no title)
)

fig_a.write_image("morton_curve_mesh_hires.png", scale=2)
with open("morton_curve_mesh_hires.png.meta.json","w") as f:
    json.dump({"caption":"Fig. A — Morton (Z-order) curve through triangle centroids on a 2-D mesh with local refinement (dashed red box). Orange dots: sorted centroids; labels: Morton codes.",
               "description":"CMAME print-ready. White background. Content identical to original."}, f)
print("✅ Fig 1 done")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — morton_search_visualize-2.py  (file:169)
# STYLE ONLY: white bg, no title, CMAME font sizes
# ══════════════════════════════════════════════════════════════════════════════
orig_to_sorted = {orig: pos for pos, orig in enumerate(order)}

queries = {
    "k\u2081": np.array([0.7, 0.5]),
    "k\u2082": np.array([2.5, 1.2]),
    "k\u2083": np.array([0.4, 3.1]),
    "k\u2084": np.array([3.2, 3.6]),
}
radius = 4

fig_b = go.Figure()

# Mesh background — ORIGINAL fills unchanged
for i, tri in enumerate(triangles):
    xs=[v[0] for v in tri]+[tri[0][0]]; ys=[v[1] for v in tri]+[tri[0][1]]
    fill="#1a3a5c" if is_refined[i] else "#0d2035"
    fig_b.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself",
        fillcolor=fill, line=dict(color="rgba(74,144,217,0.4)", width=1.0),
        showlegend=False, hoverinfo="skip"))

# Faint Morton curve — ORIGINAL
fig_b.add_trace(go.Scatter(x=sorted_cents[:,0], y=sorted_cents[:,1],
    mode="lines", line=dict(color="rgba(240,165,0,0.3)", width=2.0, dash="dot"),
    showlegend=False, hoverinfo="skip"))

# ORIGINAL colours for query points
colors = ["#e05c5c", "#5ce05c", "#5cc8e0", "#cc5ce0"]
used_legend = set()

for (qname, qpos), qcol in zip(queries.items(), colors):
    qix=int(np.clip(qpos[0]/4*N,0,N-1)); qiy=int(np.clip(qpos[1]/4*N,0,N-1))
    q_code=morton2d(qix,qiy)
    dists=np.abs(sorted_codes-q_code)
    nearest_sorted_idx=int(np.argmin(dists))
    lo=max(0,nearest_sorted_idx-radius); hi=min(len(sorted_cents)-1,nearest_sorted_idx+radius)
    neighbor_range=list(range(lo,hi+1))
    nb_x=sorted_cents[neighbor_range,0]; nb_y=sorted_cents[neighbor_range,1]
    lbl_nb=f"Radius={radius} neighbors" if "radius" not in used_legend else None
    used_legend.add("radius")

    fig_b.add_trace(go.Scatter(x=nb_x, y=nb_y, mode="markers",
        marker=dict(symbol="circle-open", size=20, color=qcol,
                    line=dict(width=3, color=qcol)),
        name=f"{qname} search window" if lbl_nb else None,
        showlegend=(lbl_nb is not None),
        hovertemplate=f"{qname} neighbor(%{{x:.2f}}, %{{y:.2f}})"))

    nc=sorted_cents[nearest_sorted_idx]
    fig_b.add_trace(go.Scatter(x=[nc[0]], y=[nc[1]], mode="markers",
        marker=dict(symbol="star", size=20, color=qcol, line=dict(width=2, color="white")),
        name=f"{qname} nearest", showlegend=True,
        hovertemplate=f"{qname} nearest Morton:{sorted_codes[nearest_sorted_idx]}"))

    fig_b.add_trace(go.Scatter(x=[qpos[0]], y=[qpos[1]], mode="markers+text",
        marker=dict(symbol="x", size=18, color=qcol, line=dict(width=3, color=qcol)),
        text=[f"{qname}"], textposition="top right",
        textfont=dict(color=qcol, size=12),                    # size 16→12
        showlegend=False,
        hovertemplate=f"{qname} ({qpos[0]:.2f},{qpos[1]:.2f}) Morton:{q_code}"))

    fig_b.add_annotation(x=nc[0], y=nc[1], ax=qpos[0], ay=qpos[1],
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=3, arrowwidth=2.5, arrowcolor=qcol, arrowsize=1.0, showarrow=True)

# Morton code labels — ORIGINAL colour, reduced size
for k in range(0, len(sorted_cents), 8):
    fig_b.add_annotation(x=sorted_cents[k,0], y=sorted_cents[k,1],
        text=f"{sorted_codes[k]}", showarrow=False,
        font=dict(size=ANN_SM, color="rgba(240,165,0,0.7)"),  # size 12→9
        xshift=10, yshift=10)

fig_b.add_shape(type="rect", x0=0, y0=2, x1=2, y1=4,
                line=dict(color="#e05c5c", width=3, dash="dash"))
fig_b.add_annotation(x=1, y=4.2, text="Refined region", showarrow=False,
    font=dict(size=ANN_LG, color="#e05c5c"))  # size 16→11

fig_b.update_layout(
    paper_bgcolor="white", plot_bgcolor="white",               # ← STYLE CHANGE
    # title removed                                            # ← STYLE CHANGE
    xaxis=dict(title_text="x", range=[-0.15,4.3], scaleanchor="y",
               title_font=dict(size=AX_FONT), tickfont=dict(size=TICK_FONT),
               showgrid=True, gridcolor=GRID_COL,
               linecolor="black", mirror=True, zeroline=False),
    yaxis=dict(title_text="y", range=[-0.3,4.5],
               title_font=dict(size=AX_FONT), tickfont=dict(size=TICK_FONT),
               showgrid=True, gridcolor=GRID_COL,
               linecolor="black", mirror=True, zeroline=False),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5,
                font=dict(size=LEG_FONT),                      # size 14→10
                bgcolor=LEG_BG, bordercolor=LEG_BD, borderwidth=1),
    width=1400, height=1050,
    margin=dict(t=75, b=60, l=70, r=40),                      # t 165→75
)

fig_b.write_image("morton_search_hires.png", scale=2)
with open("morton_search_hires.png.meta.json","w") as f:
    json.dump({"caption":"Fig. B — L2 search: k1-k4 query positions encoded to Morton codes; stars = nearest centroid by code distance; open circles = radius-R window of PIT candidates.",
               "description":"CMAME print-ready. White background. Content identical to original."}, f)
print("✅ Fig 2 done")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — mesh_aligned_octree_morton_visualize-3.py  (file:170)
# STYLE ONLY: white bg, no title, CMAME font sizes, fix dark annotation bgcolors
# ══════════════════════════════════════════════════════════════════════════════
def morton2d_bits5(ix, iy, bits=5):
    code = 0
    for i in range(bits):
        code |= ((ix >> i) & 1) << (2 * i)
        code |= ((iy >> i) & 1) << (2 * i + 1)
    return code

cells_L, cells_L1 = [], []
for row in range(4):
    for col in range(4):
        if row >= 2 and col < 2:
            continue
        elems=[i for i,t in enumerate(triangles)
               if not is_refined[i]
               and col<=np.mean([v[0] for v in t])<col+1
               and row<=np.mean([v[1] for v in t])<row+1]
        cells_L.append({"x0":col,"y0":row,"size":1.0,"cx":col+0.5,"cy":row+0.5,
                        "elems":elems,"level":"L"})
for row in range(4):
    for col in range(4):
        x0=col*0.5; y0=2+row*0.5
        elems=[i for i,t in enumerate(triangles)
               if is_refined[i]
               and x0<=np.mean([v[0] for v in t])<x0+0.5
               and y0<=np.mean([v[1] for v in t])<y0+0.5]
        cells_L1.append({"x0":x0,"y0":y0,"size":0.5,"cx":x0+0.25,"cy":y0+0.25,
                         "elems":elems,"level":"L+1"})

N_L=8; N_L1=16
for c in cells_L:
    ix=int(np.clip(c["cx"]/4*N_L,0,N_L-1)); iy=int(np.clip(c["cy"]/4*N_L,0,N_L-1))
    c["morton"]=morton2d_bits5(ix,iy,bits=4); c["gix"]=ix; c["giy"]=iy
for c in cells_L1:
    ix=int(np.clip(c["cx"]/4*N_L1,0,N_L1-1)); iy=int(np.clip(c["cy"]/4*N_L1,0,N_L1-1))
    c["morton"]=morton2d_bits5(ix,iy,bits=4); c["gix"]=ix; c["giy"]=iy

all_cells=cells_L+cells_L1
all_sorted=sorted(all_cells, key=lambda c:(c["morton"],0 if c["level"]=="L" else 1))
sorted_cx=[c["cx"] for c in all_sorted]; sorted_cy=[c["cy"] for c in all_sorted]
sorted_mc=[c["morton"] for c in all_sorted]; sorted_lvl=[c["level"] for c in all_sorted]

fig_c = go.Figure()

# Triangle background — ORIGINAL fills unchanged
for i, tri in enumerate(triangles):
    xs=[v[0] for v in tri]+[tri[0][0]]; ys=[v[1] for v in tri]+[tri[0][1]]
    fill="#1a3a5c" if is_refined[i] else "#0d2035"
    fig_c.add_trace(go.Scatter(x=xs,y=ys,mode="lines",fill="toself",
        fillcolor=fill, line=dict(color="rgba(74,144,217,0.3)",width=0.7),
        showlegend=False, hoverinfo="skip"))

# Coarse cells — ORIGINAL colours
lc_shown=False
for c in cells_L:
    x0,y0,s=c["x0"],c["y0"],c["size"]
    xs=[x0,x0+s,x0+s,x0,x0]; ys=[y0,y0,y0+s,y0+s,y0]
    fig_c.add_trace(go.Scatter(x=xs,y=ys,mode="lines",fill="toself",
        fillcolor="rgba(50,180,255,0.10)", line=dict(color="#32b4ff",width=2.0),
        name="Level L (coarse, size=1)" if not lc_shown else None,
        showlegend=not lc_shown, hoverinfo="skip"))
    lc_shown=True

# Fine cells — ORIGINAL colours
lf_shown=False
for c in cells_L1:
    x0,y0,s=c["x0"],c["y0"],c["size"]
    xs=[x0,x0+s,x0+s,x0,x0]; ys=[y0,y0,y0+s,y0+s,y0]
    fig_c.add_trace(go.Scatter(x=xs,y=ys,mode="lines",fill="toself",
        fillcolor="rgba(255,200,50,0.15)", line=dict(color="#f0c832",width=2.0),
        name="Level L+1 (fine, size=0.5)" if not lf_shown else None,
        showlegend=not lf_shown, hoverinfo="skip"))
    lf_shown=True

# Morton curve — ORIGINAL colours per level
marker_colors=["#38bdf8" if lv=="L" else "#f0a500" for lv in sorted_lvl]
fig_c.add_trace(go.Scatter(x=sorted_cx,y=sorted_cy,mode="lines+markers",
    line=dict(color="rgba(200,200,200,0.5)",width=2.0,dash="dot"),
    marker=dict(symbol="circle",size=9,color=marker_colors,
                line=dict(color="white",width=1.2)),
    name="Single sorted array (morton, level)",
    hoverinfo="skip"))

# Code labels — ORIGINAL colours, reduced size
for k in range(0,len(all_sorted),5):
    c=all_sorted[k]
    col="#38bdf8" if c["level"]=="L" else "#f0a500"
    fig_c.add_annotation(x=c["cx"],y=c["cy"],
        text=f"{c['morton']}{c['level']}",
        showarrow=False, font=dict(size=ANN_SM,color=col),   # size 11→9
        xshift=12, yshift=12)

# "No Level-L cell" annotation — bgcolor fix: dark→light
fig_c.add_annotation(x=1.0,y=3.0,
    text="No Level-L cell<br>(region refined to L+1)",
    showarrow=False, font=dict(size=ANN_MD,color="#e05c5c"),  # size 13→10
    bgcolor=ANN_BG,                                            # dark→white
    bordercolor="#e05c5c", borderwidth=1, align="center")

# Multi-cell registration — ORIGINAL
boundary_tris=[i for i,t in enumerate(triangles)
               if abs(np.mean([v[0] for v in t])-2.0)<0.45
               and abs(np.mean([v[1] for v in t])-2.35)<0.45]
for idx in boundary_tris[:2]:
    bt=triangles[idx]
    bxs=[v[0] for v in bt]+[bt[0][0]]; bys=[v[1] for v in bt]+[bt[0][1]]
    fig_c.add_trace(go.Scatter(x=bxs,y=bys,mode="lines",fill="toself",
        fillcolor="rgba(224,92,92,0.55)", line=dict(color="#e05c5c",width=2.5),
        name="Multi-level registered element" if idx==boundary_tris[0] else None,
        showlegend=(idx==boundary_tris[0]), hoverinfo="skip"))

# Refined region box — ORIGINAL
fig_c.add_shape(type="rect",x0=0,y0=2,x1=2,y1=4,
                line=dict(color="#e05c5c",width=2,dash="dash"))
fig_c.add_annotation(x=1,y=4.27, text="Refined region (L+1)",
    showarrow=False, font=dict(size=ANN_LG,color="#e05c5c"))  # size 15→11

fig_c.update_layout(
    paper_bgcolor="white", plot_bgcolor="white",               # ← STYLE CHANGE
    # title removed                                            # ← STYLE CHANGE
    xaxis=dict(title_text="x", range=[-0.2,4.4], scaleanchor="y",
               title_font=dict(size=AX_FONT), tickfont=dict(size=TICK_FONT),
               showgrid=True, gridcolor=GRID_COL,              # dark grid→light
               linecolor="black", mirror=True, zeroline=False),
    yaxis=dict(title_text="y", range=[-0.4,4.7],
               title_font=dict(size=AX_FONT), tickfont=dict(size=TICK_FONT),
               showgrid=True, gridcolor=GRID_COL,
               linecolor="black", mirror=True, zeroline=False),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5,
                font=dict(size=LEG_FONT),                      # size 14→10
                bgcolor=LEG_BG,                                # dark→white
                bordercolor=LEG_BD, borderwidth=1),            # dark bd→light
    width=1400, height=1120,
    margin=dict(t=65, b=70, l=80, r=60),                      # t 165→65, b 145→70
)

img_bytes=fig_c.to_image(format="png", scale=2)
with open("new_l2_multilevel_octree.png","wb") as f_img:
    f_img.write(img_bytes)
with open("new_l2_multilevel_octree.png.meta.json","w") as f:
    json.dump({"caption":"Fig. A (MALMO) — Two-level octree: coarse Level-L cells (blue) exist only outside the refined region; Level-L+1 fine cells (yellow) cover the refined area. All cells in a single array sorted by (morton_code, level) composite key. Red elements = multi-level registration at boundary.",
               "description":"CMAME print-ready. White background. Content identical to original."}, f)
print("✅ Fig 3 done")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — mesh_aligned_3x3x3_search_visualize-4.py  (file:171)
# STYLE ONLY: white bg, no title, CMAME font sizes, fix dark annotation bgcolors
# ══════════════════════════════════════════════════════════════════════════════

# cells_L, cells_L1 already built above (same logic as original)
# Add integer grid indices for 3x3 window logic
for c in cells_L:
    c["ix"]=c["x0"]; c["iy"]=c["y0"]
for c in cells_L1:
    c["ix"]=int(round(c["x0"]/0.5)); c["iy"]=int(round((c["y0"]-2)/0.5))

fig_d = go.Figure()

# ── Draw mesh background — ORIGINAL fills/edges ──────────────────────────────
for i, tri in enumerate(triangles):
    xs=[v[0] for v in tri]+[tri[0][0]]; ys=[v[1] for v in tri]+[tri[0][1]]
    fill="#1a3a5c" if is_refined[i] else "#0d2035"
    fig_d.add_trace(go.Scatter(x=xs,y=ys,mode="lines",fill="toself",
        fillcolor=fill, line=dict(color="rgba(74,144,217,0.25)",width=0.7),
        showlegend=False, hoverinfo="skip"))

# ── Draw all octree cells (faint) — ORIGINAL colours ─────────────────────────
for c in cells_L:
    x0,y0,s=c["x0"],c["y0"],c["size"]
    xs=[x0,x0+s,x0+s,x0,x0]; ys=[y0,y0,y0+s,y0+s,y0]
    fig_d.add_trace(go.Scatter(x=xs,y=ys,mode="lines",
        line=dict(color="rgba(50,180,255,0.25)",width=1.0),
        showlegend=False, hoverinfo="skip"))
for c in cells_L1:
    x0,y0,s=c["x0"],c["y0"],c["size"]
    xs=[x0,x0+s,x0+s,x0,x0]; ys=[y0,y0,y0+s,y0+s,y0]
    fig_d.add_trace(go.Scatter(x=xs,y=ys,mode="lines",
        line=dict(color="rgba(240,200,50,0.25)",width=1.0),
        showlegend=False, hoverinfo="skip"))

# ── Per-level 3×3 search — two query scenarios (one coarse, one fine) ─────────
query_scenarios = [
    {"qpos": np.array([2.7, 0.7]),  "level":"L",   "cells": cells_L,  "col":"#e05c5c", "label":"q (level L)"},
    {"qpos": np.array([0.6, 2.8]),  "level":"L+1", "cells": cells_L1, "col":"#5cc8e0", "label":"q (level L+1)"},
]

ctr_shown={"L":False,"L+1":False}
win_shown={"L":False,"L+1":False}
cand_shown={"L":False,"L+1":False}

for sc in query_scenarios:
    qpos=sc["qpos"]; qcol=sc["col"]; qlab=sc["label"]
    cell_list=sc["cells"]; lev=sc["level"]

    # find center cell
    center_cell=min(cell_list,
        key=lambda c:(c["cx"]-qpos[0])**2+(c["cy"]-qpos[1])**2)
    ci,cj=center_cell["ix"],center_cell["iy"]

    # 3×3 window
    window=[(c,c["ix"]-ci,c["iy"]-cj) for c in cell_list
            if abs(c["ix"]-ci)<=1 and abs(c["iy"]-cj)<=1]

    # Draw window cells — ORIGINAL colour scheme
    for c,di,dj in window:
        x0,y0,s=c["x0"],c["y0"],c["size"]
        xs=[x0,x0+s,x0+s,x0,x0]; ys=[y0,y0,y0+s,y0+s,y0]
        is_ctr=(di==0 and dj==0)
        fc="rgba(224,92,92,0.30)" if is_ctr else "rgba(92,200,224,0.20)"
        lc=qcol; lw=2.8 if is_ctr else 2.0
        leg_name=None
        if is_ctr and not ctr_shown[lev]:
            leg_name=f"Center cell ({lev})"; ctr_shown[lev]=True
        elif not is_ctr and not win_shown[lev]:
            leg_name=f"3×3 window ({lev})"; win_shown[lev]=True
        fig_d.add_trace(go.Scatter(x=xs,y=ys,mode="lines",fill="toself",
            fillcolor=fc, line=dict(color=lc,width=lw),
            name=leg_name, showlegend=(leg_name is not None), hoverinfo="skip"))
        if not is_ctr:
            fig_d.add_annotation(x=x0+s/2,y=y0+s/2,
                text=f"({di:+d},{dj:+d})",
                showarrow=False, font=dict(size=ANN_SM,color=qcol))

    # Draw candidate elements — ORIGINAL green
    for c,di,dj in window:
        for eidx in c["elems"]:
            tri=triangles[eidx]
            xs=[v[0] for v in tri]+[tri[0][0]]; ys=[v[1] for v in tri]+[tri[0][1]]
            fig_d.add_trace(go.Scatter(x=xs,y=ys,mode="lines",fill="toself",
                fillcolor="rgba(80,200,120,0.45)",
                line=dict(color="#50c878",width=1.5),
                name=f"PIT candidates ({lev})" if not cand_shown[lev] else None,
                showlegend=not cand_shown[lev], hoverinfo="skip"))
            cand_shown[lev]=True

    # Draw query position — ORIGINAL style
    fig_d.add_trace(go.Scatter(x=[qpos[0]],y=[qpos[1]],mode="markers+text",
        marker=dict(symbol="x-thin",size=16,color=qcol,line=dict(width=3,color=qcol)),
        text=[f"<b>{qlab}</b>"], textposition="top right",
        textfont=dict(color=qcol,size=11),                     # size: keep as-is (was 14 original)
        name=qlab, showlegend=True, hoverinfo="skip"))

    # Arrow query → center cell centroid
    fig_d.add_annotation(x=center_cell["cx"],y=center_cell["cy"],
        ax=qpos[0],ay=qpos[1],
        xref="x",yref="y",axref="x",ayref="y",
        arrowhead=2,arrowwidth=2.0,arrowcolor=qcol,arrowsize=1.0,showarrow=True)

# Multi-cell registration — ORIGINAL
mcr_shown=False
for idx in boundary_tris[:2]:
    bt=triangles[idx]
    bxs=[v[0] for v in bt]+[bt[0][0]]; bys=[v[1] for v in bt]+[bt[0][1]]
    fig_d.add_trace(go.Scatter(x=bxs,y=bys,mode="lines",fill="toself",
        fillcolor="rgba(255,165,0,0.55)",
        line=dict(color="#ffa500",width=2.5),
        name="Multi-cell registered" if not mcr_shown else None,
        showlegend=not mcr_shown, hoverinfo="skip"))
    mcr_shown=True

# Refined region box — ORIGINAL
fig_d.add_shape(type="rect",x0=0,y0=2,x1=2,y1=4,
                line=dict(color="#e05c5c",width=2,dash="dash"))
fig_d.add_annotation(x=1,y=4.32,text="Refined region (L+1)",
    showarrow=False, font=dict(size=ANN_LG,color="#e05c5c"))

fig_d.update_layout(
    paper_bgcolor="white", plot_bgcolor="white",               # ← STYLE CHANGE
    # title removed                                            # ← STYLE CHANGE
    xaxis=dict(title_text="x",range=[-0.2,4.4],scaleanchor="y",
               title_font=dict(size=AX_FONT),tickfont=dict(size=TICK_FONT),
               showgrid=True,gridcolor=GRID_COL,               # dark grid→light
               linecolor="black",mirror=True,zeroline=False),
    yaxis=dict(title_text="y",range=[-0.4,4.7],
               title_font=dict(size=AX_FONT),tickfont=dict(size=TICK_FONT),
               showgrid=True,gridcolor=GRID_COL,
               linecolor="black",mirror=True,zeroline=False),
    showlegend=True,
    legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="center",x=0.5,
                font=dict(size=LEG_FONT),                      # size 14→10
                bgcolor=LEG_BG,                                # dark→white
                bordercolor=LEG_BD,borderwidth=1),
    width=1400,height=1120,
    margin=dict(t=75,b=70,l=80,r=60),                         # t 165→75, b 145→70
)

img_bytes=fig_d.to_image(format="png",scale=2)
with open("mesh_aligned_3x3x3_search_hires.png","wb") as f_img:
    f_img.write(img_bytes)
with open("mesh_aligned_3x3x3_search_hires.png.meta.json","w") as f:
    json.dump({"caption":"Fig. B (MALMO) — Per-level 3×3 neighbourhood search. Two query positions (q level-L in red, q level-L+1 in cyan) each map to a center cell; the 8 surrounding cells are checked; green = PIT candidate elements. Orange = multi-cell registered element at the refinement boundary.",
               "description":"CMAME print-ready. White background. Content identical to original."}, f)
print("✅ Fig 4 done")