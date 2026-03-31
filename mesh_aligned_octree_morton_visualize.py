
import plotly.graph_objects as go
import numpy as np
import json

def morton2d(ix, iy, bits=5):
    code = 0
    for i in range(bits):
        code |= ((ix >> i) & 1) << (2 * i)
        code |= ((iy >> i) & 1) << (2 * i + 1)
    return code

# ─── Mesh: same as before ────────────────────────────────────────────────────
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

is_refined = [(np.mean([v[0] for v in t])<2) and (np.mean([v[1] for v in t])>=2) for t in triangles]

# ─── Two-level octree cells ──────────────────────────────────────────────────
# Level L (coarse, size=1): entire domain 4×4
# Level L+1 (fine, size=0.5): only in x:[0,2], y:[2,4]

cells_L  = []   # coarse level cells (exist where NOT refined)
cells_L1 = []   # fine level cells (only in refined region)

for row in range(4):
    for col in range(4):
        if row >= 2 and col < 2:
            # This region is refined → NO coarse cell here
            continue
        elems = [i for i,t in enumerate(triangles)
                 if not is_refined[i] and
                 col <= np.mean([v[0] for v in t]) < col+1 and
                 row <= np.mean([v[1] for v in t]) < row+1]
        cells_L.append({"x0":col,"y0":row,"size":1.0,"cx":col+0.5,"cy":row+0.5,
                         "elems":elems,"level":"L"})

for row in range(4):
    for col in range(4):
        x0=col*0.5; y0=2+row*0.5
        elems=[i for i,t in enumerate(triangles)
               if is_refined[i] and
               x0 <= np.mean([v[0] for v in t]) < x0+0.5 and
               y0 <= np.mean([v[1] for v in t]) < y0+0.5]
        cells_L1.append({"x0":x0,"y0":y0,"size":0.5,"cx":x0+0.25,"cy":y0+0.25,
                          "elems":elems,"level":"L+1"})

all_cells = cells_L + cells_L1

# Morton codes per level (each level uses its own grid resolution)
N_L  = 8   # coarse grid resolution
N_L1 = 16  # fine grid resolution

for c in cells_L:
    ix = int(np.clip(c["cx"]/4*N_L,  0, N_L -1))
    iy = int(np.clip(c["cy"]/4*N_L,  0, N_L -1))
    c["morton"] = morton2d(ix, iy, bits=4)
    c["gix"]=ix; c["giy"]=iy

for c in cells_L1:
    ix = int(np.clip(c["cx"]/4*N_L1, 0, N_L1-1))
    iy = int(np.clip(c["cy"]/4*N_L1, 0, N_L1-1))
    c["morton"] = morton2d(ix, iy, bits=4)
    c["gix"]=ix; c["giy"]=iy

# SINGLE sorted array: all cells sorted by (morton, level) composite key
# This matches the GPU data structure: one cell_morton_codes + cell_levels array
all_cells = cells_L + cells_L1
all_sorted = sorted(all_cells, key=lambda c: (c["morton"], 0 if c["level"]=="L" else 1))
sorted_cx = [c["cx"] for c in all_sorted]
sorted_cy = [c["cy"] for c in all_sorted]
sorted_mc = [c["morton"] for c in all_sorted]
sorted_lvl = [c["level"] for c in all_sorted]

# ═══════════════════════════════════════════════════════════════════════════════
# FIG A – Two-level octree + per-level Morton curves
# ═══════════════════════════════════════════════════════════════════════════════
fig_a = go.Figure()

# Triangle background
for i, tri in enumerate(triangles):
    xs=[v[0] for v in tri]+[tri[0][0]]; ys=[v[1] for v in tri]+[tri[0][1]]
    fill="#1a3a5c" if is_refined[i] else "#0d2035"#"rgba(26,58,92,0.4)" if is_refined[i] else "rgba(13,32,53,0.4)"
    fig_a.add_trace(go.Scatter(x=xs,y=ys,mode="lines",fill="toself",
        fillcolor=fill,line=dict(color="rgba(74,144,217,0.3)",width=0.7),
        showlegend=False,hoverinfo="skip"))

# Coarse cells (level L)
lc_shown=False
for c in cells_L:
    x0,y0,s=c["x0"],c["y0"],c["size"]
    xs=[x0,x0+s,x0+s,x0,x0]; ys=[y0,y0,y0+s,y0+s,y0]
    fig_a.add_trace(go.Scatter(x=xs,y=ys,mode="lines",fill="toself",
        fillcolor="rgba(50,180,255,0.10)",line=dict(color="#32b4ff",width=2.0),
        name="Level L (coarse, size=1)" if not lc_shown else None,
        showlegend=not lc_shown,hovertemplate=f"Level L cell ({c['cx']:.1f},{c['cy']:.1f})<extra></extra>"))
    lc_shown=True

# Fine cells (level L+1)
lf_shown=False
for c in cells_L1:
    x0,y0,s=c["x0"],c["y0"],c["size"]
    xs=[x0,x0+s,x0+s,x0,x0]; ys=[y0,y0,y0+s,y0+s,y0]
    fig_a.add_trace(go.Scatter(x=xs,y=ys,mode="lines",fill="toself",
        fillcolor="rgba(255,200,50,0.15)",line=dict(color="#f0c832",width=2.0),
        name="Level L+1 (fine, size=0.5)" if not lf_shown else None,
        showlegend=not lf_shown,hovertemplate=f"Level L+1 cell ({c['cx']:.2f},{c['cy']:.2f})<extra></extra>"))
    lf_shown=True

# Single Morton curve through ALL cells (sorted by (morton, level) composite key)
# Color each marker by its level to show the interleaving
marker_colors = ["#38bdf8" if lv=="L" else "#f0a500" for lv in sorted_lvl]
hover_labels = [f"Morton {m} (Level {lv})<br>({x:.2f},{y:.2f})"
                for m,lv,x,y in zip(sorted_mc, sorted_lvl, sorted_cx, sorted_cy)]
fig_a.add_trace(go.Scatter(x=sorted_cx,y=sorted_cy,mode="lines+markers",
    line=dict(color="rgba(200,200,200,0.5)",width=2.0,dash="dot"),
    marker=dict(symbol="circle",size=9,color=marker_colors,
                line=dict(color="white",width=1.2)),
    name="Single sorted array (morton, level)",
    hovertemplate="%{customdata}<extra></extra>",
    customdata=hover_labels))

# Annotate a few Morton codes (mixed levels)
for k in range(0,len(all_sorted),5):
    c=all_sorted[k]
    col="#38bdf8" if c["level"]=="L" else "#f0a500"
    fig_a.add_annotation(x=c["cx"],y=c["cy"],
        text=f"<b>{c['morton']}</b><sub>{c['level']}</sub>",
        showarrow=False,font=dict(size=11,color=col),xshift=12,yshift=12)

# "NO CELL HERE" label in the gap where coarse cells are absent
fig_a.add_annotation(x=1.0,y=3.0,
    text="<b>No Level-L cell</b><br>(region refined to L+1)",
    showarrow=False,font=dict(size=13,color="#e05c5c"),
    bgcolor="rgba(15,23,42,0.75)",bordercolor="#e05c5c",borderwidth=1,
    align="center")

# Multi-cell registration: element near x=2, y=2 boundary assigned to cells at both levels
boundary_tris = [i for i,t in enumerate(triangles)
    if abs(np.mean([v[0] for v in t])-2.0)<0.45 and
       abs(np.mean([v[1] for v in t])-2.35)<0.45]
for idx in boundary_tris[:2]:
    bt=triangles[idx]
    bxs=[v[0] for v in bt]+[bt[0][0]]; bys=[v[1] for v in bt]+[bt[0][1]]
    fig_a.add_trace(go.Scatter(x=bxs,y=bys,mode="lines",fill="toself",
        fillcolor="rgba(224,92,92,0.55)",line=dict(color="#e05c5c",width=2.5),
        name="Multi-level registered element" if idx==boundary_tris[0] else None,
        showlegend=(idx==boundary_tris[0]),
        hovertemplate="Registered at both L and L+1<extra></extra>"))

# Refined region box
fig_a.add_shape(type="rect",x0=0,y0=2,x1=2,y1=4,
    line=dict(color="#e05c5c",width=2,dash="dash"))
fig_a.add_annotation(x=1,y=4.27,text="<b>Refined region (L+1)</b>",
    showarrow=False,font=dict(size=15,color="#e05c5c"))

fig_a.update_layout(
    title=dict(text="Two-Level Octree & Single Sorted (Morton, Level) Array",
        font=dict(size=22),x=0.5,xanchor="center"),
    xaxis=dict(title_text="x",range=[-0.2,4.4],scaleanchor="y",
               title_font=dict(size=18),tickfont=dict(size=14),
               showgrid=True,gridcolor="rgba(255,255,255,0.07)"),
    yaxis=dict(title_text="y",range=[-0.4,4.7],
               title_font=dict(size=18),tickfont=dict(size=14),
               showgrid=True,gridcolor="rgba(255,255,255,0.07)"),
    showlegend=True,
    legend=dict(orientation="h",yanchor="top",y=-0.08,xanchor="center",x=0.5,
                font=dict(size=14),bgcolor="rgba(15,23,42,0.85)",
                bordercolor="rgba(100,150,200,0.3)",borderwidth=1),
    width=1400,height=1120,
    # paper_bgcolor="#0f172a",plot_bgcolor="#0d1f33",
    margin=dict(t=165,b=145,l=80,r=60),
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

img_bytes = fig_a.to_image(format="svg", scale=2)
with open("new_l2_multilevel_octree.svg", "wb") as f_img:
    f_img.write(img_bytes)
with open("new_l2_multilevel_octree.png.meta.json","w") as f:
    json.dump({"caption":
        "Fig A – Two-level octree: coarse Level-L cells (blue) exist only outside the refined region; "
        "Level-L+1 fine cells (yellow) cover the refined area. All cells stored in a single array "
        "sorted by (morton_code, level) composite key. Red elements = multi-level registration at boundary.",
        "description":"Two-level octree overlaid on mesh. Coarse cells are absent where the mesh is refined. "
        "Single sorted array with (morton, level) composite key enables O(log n) binary search."}, f)
print("Fig A saved.")