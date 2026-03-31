
import plotly.graph_objects as go
import numpy as np
import json

def morton2d(ix, iy, bits=4):
    code = 0
    for i in range(bits):
        code |= ((ix >> i) & 1) << (2 * i)
        code |= ((iy >> i) & 1) << (2 * i + 1)
    return code

# ─── Mesh rebuild ────────────────────────────────────────────────────────────
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
is_refined=[(np.mean([v[0] for v in t])<2)and(np.mean([v[1] for v in t])>=2) for t in triangles]

# ─── Two-level cells (same as Fig A) ─────────────────────────────────────────
cells_L = []
for row in range(4):
    for col in range(4):
        if row >= 2 and col < 2: continue
        elems=[i for i,t in enumerate(triangles)
               if not is_refined[i] and
               col<=np.mean([v[0] for v in t])<col+1 and
               row<=np.mean([v[1] for v in t])<row+1]
        cells_L.append({"x0":col,"y0":row,"size":1.0,"cx":col+0.5,"cy":row+0.5,"elems":elems})

cells_L1 = []
for row in range(4):
    for col in range(4):
        x0=col*0.5; y0=2+row*0.5
        elems=[i for i,t in enumerate(triangles)
               if is_refined[i] and
               x0<=np.mean([v[0] for v in t])<x0+0.5 and
               y0<=np.mean([v[1] for v in t])<y0+0.5]
        cells_L1.append({"x0":x0,"y0":y0,"size":0.5,"cx":x0+0.25,"cy":y0+0.25,"elems":elems})

# ─── Query scenario: particle inside refined region, near coarse boundary
# Search order: FINEST → COARSEST (level L+1 first, then L)
# Level L+1 search (1st): i'=floor(1.85/0.5)=3, j'=floor(2.15/0.5)=4 → 3×3 finds fine cells
# Level L search (2nd): i=floor(1.85/1.0)=1, j=floor(2.15/1.0)=2 → NO cell (refined away) → -1
#   but neighbors at (2,2), (2,1) etc. DO exist → searched

qpos = np.array([1.85, 2.15])  # inside refined region, near coarse boundary

# ── Level L: compute grid cell for query (cell may not exist at this level)
cc_L = int(np.floor(qpos[0] / 1.0))  # coarse cell col
cr_L = int(np.floor(qpos[1] / 1.0))  # coarse cell row
neighbors_L_exist   = []   # cells that actually exist at Level L
neighbors_L_missing = []   # cells that WOULD be neighbors but don't exist (refined away)

for dc in range(-1,2):
    for dr in range(-1,2):
        nc, nr = cc_L+dc, cr_L+dr
        if not (0<=nc<4 and 0<=nr<4):
            continue  # out of domain
        cell_exists = any(c for c in cells_L if c["x0"]==nc and c["y0"]==nr)
        if cell_exists:
            neighbors_L_exist.append({"x0":nc,"y0":nr,"size":1.0,"cx":nc+0.5,"cy":nr+0.5,
                                       "center":(dc==0 and dr==0),"dc":dc,"dr":dr})
        else:
            # Cell doesn't exist at Level L (refined away)
            neighbors_L_missing.append({"x0":nc,"y0":nr,"size":1.0,"cx":nc+0.5,"cy":nr+0.5,
                                         "dc":dc,"dr":dr,"is_center":(dc==0 and dr==0)})

# ── Level L+1: re-index query on fine grid; 3×3 covers 3×0.5 = 1.5 coarse width
# i' = floor(qpos[0]/0.5), j' = floor((qpos[1]-2.0)/0.5) → fine grid origin at y=2
# Fine cells have x0 = col*0.5, y0 = 2 + row*0.5
fc_col = int(np.floor(qpos[0] / 0.5))  # = 3 for qpos[0]=1.85
fc_row = int(np.floor((qpos[1] - 2.0) / 0.5))  # = 0 for qpos[1]=2.15
fc_x = fc_col * 0.5  # = 1.5
fc_y = 2.0 + fc_row * 0.5  # = 2.0
fine_center = next((c for c in cells_L1 if abs(c["x0"]-fc_x)<0.01 and abs(c["y0"]-fc_y)<0.01), None)
fs = 0.5

# 3×3 at Level L+1 around fine_center
neighbors_L1 = []
for dc in range(-1,2):
    for dr in range(-1,2):
        nx0 = fc_x + dc*0.5; ny0 = fc_y + dr*0.5
        cell_exists = any(c for c in cells_L1 if abs(c["x0"]-nx0)<0.01 and abs(c["y0"]-ny0)<0.01)
        if cell_exists:
            neighbors_L1.append({"x0":nx0,"y0":ny0,"size":0.5,"cx":nx0+0.25,"cy":ny0+0.25,
                                  "center":(dc==0 and dr==0),"dc":dc,"dr":dr})

# ═══════════════════════════════════════════════════════════════════════════════
# FIG B – Multi-level 3×3 search at coarse/refined transition
# ═══════════════════════════════════════════════════════════════════════════════
fig_b = go.Figure()

# Triangle background
for i, tri in enumerate(triangles):
    xs=[v[0] for v in tri]+[tri[0][0]]; ys=[v[1] for v in tri]+[tri[0][1]]
    fill="#1a3a5c" if is_refined[i] else "#0d2035"#"rgba(26,58,92,0.35)" if is_refined[i] else "rgba(13,32,53,0.35)"
    fig_b.add_trace(go.Scatter(x=xs,y=ys,mode="lines",fill="toself",
        fillcolor=fill,line=dict(color="rgba(74,144,217,0.25)",width=0.7),
        showlegend=False,hoverinfo="skip"))

# All cells (faint outlines)
for c in cells_L:
    x0,y0,s=c["x0"],c["y0"],c["size"]
    fig_b.add_trace(go.Scatter(x=[x0,x0+s,x0+s,x0,x0],y=[y0,y0,y0+s,y0+s,y0],mode="lines",
        line=dict(color="rgba(50,180,255,0.25)",width=1.0),showlegend=False,hoverinfo="skip"))
for c in cells_L1:
    x0,y0,s=c["x0"],c["y0"],c["size"]
    fig_b.add_trace(go.Scatter(x=[x0,x0+s,x0+s,x0,x0],y=[y0,y0,y0+s,y0+s,y0],mode="lines",
        line=dict(color="rgba(240,200,50,0.25)",width=1.0),showlegend=False,hoverinfo="skip"))

# ── STEP 1: LEVEL L+1 search (finest first) ─────────────────────────────────
lL1_ctr=True; lL1_nb=True
for nc in neighbors_L1:
    x0,y0,s=nc["x0"],nc["y0"],nc["size"]
    if nc["center"]:
        fc="rgba(240,200,50,0.40)"; bc="#f0c832"; lw=3.5
        lname="Level-L+1 query cell (i′,j′)" if lL1_ctr else None; lL1_ctr=False
    else:
        fc="rgba(240,165,0,0.18)"; bc="#f0a500"; lw=2.5
        lname="Level-L+1 neighbor (exists)" if lL1_nb else None; lL1_nb=False
    fig_b.add_trace(go.Scatter(x=[x0,x0+s,x0+s,x0,x0],y=[y0,y0,y0+s,y0+s,y0],mode="lines",fill="toself",
        fillcolor=fc,line=dict(color=bc,width=lw),
        name=lname,showlegend=(lname is not None),
        hovertemplate=f"Level-L+1 cell ({nc['cx']:.2f},{nc['cy']:.2f})<extra></extra>"))
    dc,dr=nc["dc"],nc["dr"]
    sc_="+" if dc>0 else (""if dc==0 else "−"); sr_="+" if dr>0 else (""if dr==0 else "−")
    label="<b>i′,j′</b>" if (dc==0 and dr==0) else f"i′{sc_}{abs(dc) if dc else ''},j′{sr_}{abs(dr) if dr else ''}"
    fig_b.add_annotation(x=nc["cx"],y=nc["cy"],text=label,showarrow=False,
        font=dict(size=11,color="rgba(255,230,150,0.95)"))

# Elements checked at Level L+1
checked_L1=set()
for nc in neighbors_L1:
    for i,t in enumerate(triangles):
        tcx=np.mean([v[0] for v in t]); tcy=np.mean([v[1] for v in t])
        if nc["x0"]<=tcx<nc["x0"]+nc["size"] and nc["y0"]<=tcy<nc["y0"]+nc["size"]:
            checked_L1.add(i)
for i in checked_L1:
    tri=triangles[i]
    xs=[v[0] for v in tri]+[tri[0][0]]; ys=[v[1] for v in tri]+[tri[0][1]]
    fig_b.add_trace(go.Scatter(x=xs,y=ys,mode="lines",fill="toself",
        fillcolor="rgba(240,200,50,0.22)",line=dict(color="#f0c832",width=1.5),
        showlegend=False,hoverinfo="skip"))

# Level L+1 bounding box
x_L1_min=fine_center["x0"]-0.5; x_L1_max=fine_center["x0"]+1.0
y_L1_min=fine_center["y0"]-0.5; y_L1_max=fine_center["y0"]+1.0
fig_b.add_shape(type="rect",x0=x_L1_min,y0=y_L1_min,x1=x_L1_max,y1=y_L1_max,
    line=dict(color="#f0c832",width=2.5,dash="dash"))
fig_b.add_annotation(x=(x_L1_min+x_L1_max)/2,y=y_L1_max+0.12,
    text="<b>Step 1: Level-L+1 3×3 window</b><br><span style='font-size:12px'>(finest first; 3×0.5 = 1.5 coarse width)</span>",
    showarrow=False,font=dict(size=14,color="#f0c832"),align="center")

# ── STEP 2: LEVEL L search (coarser, only if not found at L+1) ─────────────
# Missing neighbors (refined-away): hatched appearance
miss_shown=False
for nc in neighbors_L_missing:
    x0,y0,s=nc["x0"],nc["y0"],1.0
    fig_b.add_trace(go.Scatter(x=[x0,x0+s,x0+s,x0,x0],y=[y0,y0,y0+s,y0+s,y0],mode="lines",fill="toself",
        fillcolor="rgba(224,92,92,0.10)",line=dict(color="#e05c5c",width=2,dash="dot"),
        name="Level-L neighbor → -1 (refined, skipped)" if not miss_shown else None,
        showlegend=not miss_shown,
        hovertemplate="No Level-L cell here<br>find_cell returns -1 → skipped<extra></extra>"))
    miss_shown=True
    label_text = "<b>i,j → ✕ -1</b>" if nc.get("is_center") else "✕ -1"
    fig_b.add_annotation(x=nc["cx"],y=nc["cy"],text=label_text,showarrow=False,
        font=dict(size=18,color="#e05c5c"))

# Existing Level-L neighbors (note: center cell (1,2) is refined away — only neighbors exist)
lL_nb=True
for nc in neighbors_L_exist:
    x0,y0,s=nc["x0"],nc["y0"],nc["size"]
    fc="rgba(50,180,255,0.15)"; bc="#32b4ff"; lw=2.5
    lname="Level-L neighbor (exists)" if lL_nb else None; lL_nb=False
    fig_b.add_trace(go.Scatter(x=[x0,x0+s,x0+s,x0,x0],y=[y0,y0,y0+s,y0+s,y0],mode="lines",fill="toself",
        fillcolor=fc,line=dict(color=bc,width=lw),
        name=lname,showlegend=(lname is not None),
        hovertemplate=f"Level-L cell ({nc['cx']:.1f},{nc['cy']:.1f})<extra></extra>"))
    dc,dr=nc["dc"],nc["dr"]
    sc_="+" if dc>0 else (""if dc==0 else "−"); sr_="+" if dr>0 else (""if dr==0 else "−")
    label=f"<b>i,j</b>" if (dc==0 and dr==0) else f"i{sc_}{abs(dc) if dc else ''},j{sr_}{abs(dr) if dr else ''}"
    fig_b.add_annotation(x=nc["cx"],y=nc["cy"],text=label,showarrow=False,
        font=dict(size=13,color="rgba(200,230,255,0.9)"))

# Elements checked at Level L
checked_L = set()
for nc in neighbors_L_exist:
    for i,t in enumerate(triangles):
        tcx=np.mean([v[0] for v in t]); tcy=np.mean([v[1] for v in t])
        if nc["x0"]<=tcx<nc["x0"]+nc["size"] and nc["y0"]<=tcy<nc["y0"]+nc["size"]:
            checked_L.add(i)
for i in checked_L:
    tri=triangles[i]
    xs=[v[0] for v in tri]+[tri[0][0]]; ys=[v[1] for v in tri]+[tri[0][1]]
    fig_b.add_trace(go.Scatter(x=xs,y=ys,mode="lines",fill="toself",
        fillcolor="rgba(56,189,248,0.20)",line=dict(color="#38bdf8",width=1.5),
        showlegend=False,hoverinfo="skip"))

# Level L bounding box
x_L_min=cc_L-1; x_L_max=cc_L+2; y_L_min=cr_L-1; y_L_max=cr_L+2
fig_b.add_shape(type="rect",x0=x_L_min,y0=y_L_min,x1=x_L_max,y1=y_L_max,
    line=dict(color="#38bdf8",width=2.5,dash="dash"))
fig_b.add_annotation(x=(x_L_min+x_L_max)/2,y=y_L_max+0.12,
    text="<b>Step 2: Level-L 3×3 window</b><br><span style='font-size:12px'>(coarser fallback; only if not found at L+1)</span>",
    showarrow=False,font=dict(size=14,color="#38bdf8"),align="center")

# Query point
fig_b.add_trace(go.Scatter(x=[qpos[0]],y=[qpos[1]],mode="markers+text",
    marker=dict(symbol="x",size=22,color="#e05c5c",line=dict(width=3,color="#e05c5c")),
    text=["<b>q</b>"],textposition="top right",textfont=dict(color="#e05c5c",size=18),
    name="Query position",showlegend=True,
    hovertemplate=f"Query ({qpos[0]:.2f},{qpos[1]:.2f})<extra></extra>"))

# Refined region boundary
fig_b.add_shape(type="rect",x0=0,y0=2,x1=2,y1=4,
    line=dict(color="#e05c5c",width=2,dash="dash"))
fig_b.add_annotation(x=1,y=4.27,text="<b>Refined region (L+1)</b>",
    showarrow=False,font=dict(size=15,color="#e05c5c"))

# Key insight callout
fig_b.add_annotation(x=3.5,y=0.5,
    text="<b>Key insight:</b><br>Search finest→coarsest<br>(L+1 first, then L).<br>Each level recomputes<br>grid indices at its own<br>cell size and searches<br>3×3 independently.<br>Center cell at L may<br>not exist (→ -1, skip).",
    showarrow=False,font=dict(size=13,color="#cbd5e1"),align="left",
    bgcolor="rgba(15,23,42,0.88)",bordercolor="rgba(100,150,200,0.4)",borderwidth=1)

fig_b.update_layout(
    title=dict(text="Multi-Level 3x3 Search (Finest to Coarsest)",
        font=dict(size=21),x=0.5,xanchor="center"),
    xaxis=dict(title_text="x",range=[-0.2,4.6],scaleanchor="y",
               title_font=dict(size=18),tickfont=dict(size=14),
               showgrid=True,gridcolor="rgba(255,255,255,0.07)"),
    yaxis=dict(title_text="y",range=[-0.4,4.75],
               title_font=dict(size=18),tickfont=dict(size=14),
               showgrid=True,gridcolor="rgba(255,255,255,0.07)"),
    showlegend=True,
    legend=dict(orientation="h",yanchor="top",y=-0.08,xanchor="center",x=0.5,
                font=dict(size=13),bgcolor="rgba(15,23,42,0.85)",
                bordercolor="rgba(100,150,200,0.3)",borderwidth=1),
    # width=1400,height=1150,
    # paper_bgcolor="#0f172a",plot_bgcolor="#0d1f33",
    margin=dict(t=170,b=150,l=80,r=60),
)

fig_b.update_layout(
    title=None,
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Arial", size=16, color="black"),
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
    tickfont=dict(size=14, color="black"),
    title_font=dict(size=16, color="black"),
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
    tickfont=dict(size=14, color="black"),
    title_font=dict(size=16, color="black"),
    constrain="domain"
)

img_bytes = fig_b.to_image(format="svg", scale=2)
with open("new_l2_multilevel_3x3.svg", "wb") as f_img:
    f_img.write(img_bytes)
with open("new_l2_multilevel_3x3.png.meta.json","w") as f:
    json.dump({"caption":
        "Fig B – Multi-level 3×3 search (finest→coarsest) at a refinement boundary. "
        "Step 1: Level-L+1 (yellow, finest) searches 3×3 on fine grid. "
        "Step 2: Level-L (blue, coarser) searches 3×3 but skips missing cells (red ✕ = -1). "
        "Stops as soon as containing element is found.",
        "description":"Multi-level 3x3 search visualization. Search proceeds finest→coarsest. "
        "Level-L+1 searches fine grid first; Level-L falls back to coarse grid, skipping refined-away cells."}, f)
print("Fig B saved.")
print(f"Checked at Level L: {len(checked_L)} elements, Level L+1: {len(checked_L1)} elements")