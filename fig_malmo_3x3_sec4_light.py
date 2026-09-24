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

# ── Query point: inside COARSE region, near the refined boundary ──────────────
# q in coarse region → motivates searching ℓ+1 first (most absent here),
# then ℓ (all present). Clearly shows the finest-to-coarsest pass order.
qpos = (2.35, 2.60)

# ── Level ℓ+1 neighbourhood (3×3 on fine grid, h=0.5) ────────────────────────
# Map q onto the fine grid even though q is in the coarse region.
# Most fine slots won't exist here → absent category is well populated.
fc_col = int(np.floor(qpos[0] / 0.5))
fc_row = int(np.floor((qpos[1]-2.0) / 0.5))
fc_x0  = fc_col * 0.5
fc_y0  = 2.0 + fc_row * 0.5

nb_L1 = []
for dc in range(-1, 2):
    for dr in range(-1, 2):
        nx0 = fc_x0 + dc*0.5; ny0 = fc_y0 + dr*0.5
        exists = cell_exists_L1(nx0, ny0)
        nb_L1.append({"x0":nx0,"y0":ny0,"size":0.5,
                      "cx":nx0+0.25,"cy":ny0+0.25,
                      "exists":exists,"dc":dc,"dr":dr,
                      "center":(dc==0 and dr==0)})

# ── Level ℓ neighbourhood (3×3 on coarse grid, h=1.0) ────────────────────────
cc = int(np.floor(qpos[0] / 1.0))
cr = int(np.floor(qpos[1] / 1.0))

nb_L = []
for dc in range(-1, 2):
    for dr in range(-1, 2):
        nc_col, nc_row = cc+dc, cr+dr
        in_domain = (0 <= nc_col < 4) and (0 <= nc_row < 4)
        exists = in_domain and cell_exists_L(nc_col, nc_row)
        nb_L.append({"x0":nc_col,"y0":nc_row,"size":1.0,
                     "cx":nc_col+0.5,"cy":nc_row+0.5,
                     "exists":exists,"in_domain":in_domain,
                     "dc":dc,"dr":dr,"center":(dc==0 and dr==0)})

# ── Index label helper ────────────────────────────────────────────────────────
def idx_label(base_i, base_j, level_str, dc, dr):
    def fmt(base, d):
        if d == 0:  return base
        if d == 1:  return f"{base}+1"
        if d ==-1:  return f"{base}−1"
        if d > 1:   return f"{base}+{d}"
        return f"{base}{d}"
    return f"({fmt(base_i,dc)},{fmt(base_j,dr)},{level_str})"

# ── Colour palette (journal light style) ─────────────────────────────────────
COARSE_MESH_FILL = "rgba(215,225,235,0.55)"
FINE_MESH_FILL   = "rgba(215,230,215,0.55)"
MESH_EDGE        = "rgba(90,110,130,0.45)"

# Level ℓ+1 (amber tones)
L1_PRESENT_EDGE  = "#b07800"
L1_PRESENT_FILL  = "rgba(176,120,0,0.13)"
L1_ABSENT_EDGE   = "#b07800"   # same hue, dotted

# Level ℓ (blue tones)
L_PRESENT_EDGE   = "#1a6fa8"
L_PRESENT_FILL   = "rgba(26,111,168,0.13)"
L_ABSENT_EDGE    = "#c0392b"   # red dotted — refined away

# Labels
LABEL_L1_PRES    = "#7a5200"
LABEL_L1_ABS     = "#9a7200"
LABEL_L_PRES     = "#1a6fa8"
LABEL_L_ABS      = "#c0392b"

RED_DASH         = "#c0392b"   # refined-region boundary
QUERY_COL        = "#222222"   # dark neutral query marker

# ── Figure ────────────────────────────────────────────────────────────────────
fig = go.Figure()

# Mesh triangles
for i, tri in enumerate(triangles):
    xs=[v[0] for v in tri]+[tri[0][0]]
    ys=[v[1] for v in tri]+[tri[0][1]]
    fill = FINE_MESH_FILL if is_refined[i] else COARSE_MESH_FILL
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself",
                             fillcolor=fill,
                             line=dict(color=MESH_EDGE, width=0.6),
                             showlegend=False, hoverinfo="skip"))

# Background cell grid outlines (very faint)
for c in cells_L:
    x0,y0,s = c["x0"],c["y0"],c["size"]
    fig.add_trace(go.Scatter(x=[x0,x0+s,x0+s,x0,x0],y=[y0,y0,y0+s,y0+s,y0],
                             mode="lines",
                             line=dict(color="rgba(26,111,168,0.18)",width=0.7),
                             showlegend=False, hoverinfo="skip"))
for c in cells_L1:
    x0,y0,s = c["x0"],c["y0"],c["size"]
    fig.add_trace(go.Scatter(x=[x0,x0+s,x0+s,x0,x0],y=[y0,y0,y0+s,y0+s,y0],
                             mode="lines",
                             line=dict(color="rgba(176,120,0,0.18)",width=0.7),
                             showlegend=False, hoverinfo="skip"))

# ── Legend flags ──────────────────────────────────────────────────────────────
leg = {"L1_present":True,"L1_absent":True,"L_present":True,"L_absent":True}

# ── Level ℓ+1: absent first (behind), present on top ─────────────────────────
for nb in nb_L1:
    if nb["exists"]: continue
    x0,y0,s = nb["x0"],nb["y0"],nb["size"]
    xs=[x0,x0+s,x0+s,x0,x0]; ys=[y0,y0,y0+s,y0+s,y0]
    fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines",fill="toself",
                             fillcolor="rgba(0,0,0,0)",
                             line=dict(color=L1_ABSENT_EDGE,width=1.3,dash="dot"),
                             name="level ℓ+1, absent" if leg["L1_absent"] else None,
                             showlegend=leg["L1_absent"],hoverinfo="skip"))
    leg["L1_absent"] = False

for nb in nb_L1:
    if not nb["exists"]: continue
    x0,y0,s = nb["x0"],nb["y0"],nb["size"]
    xs=[x0,x0+s,x0+s,x0,x0]; ys=[y0,y0,y0+s,y0+s,y0]
    lw = 2.6 if nb["center"] else 1.9
    fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines",fill="toself",
                             fillcolor=L1_PRESENT_FILL,
                             line=dict(color=L1_PRESENT_EDGE,width=lw),
                             name="level ℓ+1, present" if leg["L1_present"] else None,
                             showlegend=leg["L1_present"],hoverinfo="skip"))
    leg["L1_present"] = False

# ℓ+1 bounding box
bb1_x0=fc_x0-0.5; bb1_x1=fc_x0+1.0
bb1_y0=fc_y0-0.5; bb1_y1=fc_y0+1.0
fig.add_shape(type="rect",x0=bb1_x0,y0=bb1_y0,x1=bb1_x1,y1=bb1_y1,
              line=dict(color=L1_PRESENT_EDGE,width=1.8,dash="dash"),layer="above")
fig.add_annotation(x=(bb1_x0+bb1_x1)/2,y=bb1_y1+0.09,
                   text="3×3 window (ℓ+1)",showarrow=False,
                   font=dict(size=9,color=LABEL_L1_PRES,family="Arial"),
                   bgcolor="rgba(255,255,255,0.85)",borderpad=2)

# ℓ+1 index labels — all 9 slots
for nb in nb_L1:
    label = idx_label("i′","j′","ℓ+1",nb["dc"],nb["dr"])
    col   = LABEL_L1_PRES if nb["exists"] else LABEL_L1_ABS
    fig.add_annotation(x=nb["cx"],y=nb["cy"],text=label,
                       showarrow=False,
                       font=dict(size=8,color=col,family="Arial"))

# ── Level ℓ: absent first ─────────────────────────────────────────────────────
for nb in nb_L:
    if nb["exists"] or not nb["in_domain"]: continue
    x0,y0=nb["x0"],nb["y0"]
    xs=[x0,x0+1,x0+1,x0,x0]; ys=[y0,y0,y0+1,y0+1,y0]
    fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines",fill="toself",
                             fillcolor="rgba(0,0,0,0)",
                             line=dict(color=L_ABSENT_EDGE,width=1.3,dash="dot"),
                             name="level ℓ, absent (refined away)" if leg["L_absent"] else None,
                             showlegend=leg["L_absent"],hoverinfo="skip"))
    leg["L_absent"] = False

for nb in nb_L:
    if not nb["exists"]: continue
    x0,y0,s=nb["x0"],nb["y0"],nb["size"]
    xs=[x0,x0+s,x0+s,x0,x0]; ys=[y0,y0,y0+s,y0+s,y0]
    lw = 2.6 if nb["center"] else 1.9
    fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines",fill="toself",
                             fillcolor=L_PRESENT_FILL,
                             line=dict(color=L_PRESENT_EDGE,width=lw),
                             name="level ℓ, present" if leg["L_present"] else None,
                             showlegend=leg["L_present"],hoverinfo="skip"))
    leg["L_present"] = False

# ℓ bounding box
bbL_x0=cc-1; bbL_x1=cc+2
bbL_y0=cr-1; bbL_y1=cr+2
fig.add_shape(type="rect",x0=bbL_x0,y0=bbL_y0,x1=bbL_x1,y1=bbL_y1,
              line=dict(color=L_PRESENT_EDGE,width=1.8,dash="dash"),layer="above")
fig.add_annotation(x=(bbL_x0+bbL_x1)/2,y=bbL_y1+0.09,
                   text="3×3 window (ℓ)",showarrow=False,
                   font=dict(size=9,color=L_PRESENT_EDGE,family="Arial"),
                   bgcolor="rgba(255,255,255,0.85)",borderpad=2)

# ℓ index labels — all in-domain slots
for nb in nb_L:
    if not nb["in_domain"]: continue
    label = idx_label("i","j","ℓ",nb["dc"],nb["dr"])
    col   = LABEL_L_PRES if nb["exists"] else LABEL_L_ABS
    fig.add_annotation(x=nb["cx"],y=nb["cy"],text=label,
                       showarrow=False,
                       font=dict(size=8,color=col,family="Arial"))

# Refined-region boundary
fig.add_shape(type="rect",x0=0,y0=2,x1=2,y1=4,
              line=dict(color=RED_DASH,width=1.4,dash="dash"),layer="above")

# Query point
fig.add_trace(go.Scatter(x=[qpos[0]],y=[qpos[1]],
                         mode="markers+text",
                         marker=dict(symbol="x",size=14,color=QUERY_COL,
                                     line=dict(width=2.8,color=QUERY_COL)),
                         text=["q"],textposition="top right",
                         textfont=dict(size=11,color=QUERY_COL,family="Arial"),
                         name="query point q",showlegend=True,hoverinfo="skip"))

# ── Layout ────────────────────────────────────────────────────────────────────
fig.update_layout(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Arial",size=11,color="black"),
    margin=dict(l=50,r=12,t=22,b=45),
    showlegend=True,
    legend=dict(x=0.01,y=0.99,xanchor="left",yanchor="top",
                orientation="v",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="rgba(0,0,0,0.20)",borderwidth=1,
                font=dict(size=9,color="black"),
                tracegroupgap=2),
    width=500,height=500,
)
fig.update_xaxes(title_text="x",range=[-0.05,4.05],
                 showgrid=False,zeroline=False,
                 showline=True,linewidth=1.3,linecolor="black",mirror=True,
                 ticks="outside",tickfont=dict(size=10),title_font=dict(size=11),
                 scaleanchor="y",constrain="domain")
fig.update_yaxes(title_text="y",range=[-0.05,4.05],
                 showgrid=False,zeroline=False,
                 showline=True,linewidth=1.3,linecolor="black",mirror=True,
                 ticks="outside",tickfont=dict(size=10),title_font=dict(size=11),
                 constrain="domain")

img_bytes = fig.to_image(format="svg",scale=2)
with open("fig_malmo_3x3_sec4_light.svg","wb") as f:
    f.write(img_bytes)
print("Fig B saved: fig_malmo_3x3_sec4_light.pdf")
