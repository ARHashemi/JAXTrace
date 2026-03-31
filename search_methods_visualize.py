from pathlib import Path
import textwrap, json, py_compile, os, runpy

out = Path('output')
out.mkdir(exist_ok=True)

common = '''import json
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageChops


def morton2d(ix, iy, bits=4):
    code = 0
    for i in range(bits):
        code |= ((ix >> i) & 1) << (2 * i)
        code |= ((iy >> i) & 1) << (2 * i + 1)
    return code


def build_triangles():
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
    return triangles


def is_refined_triangle(tri):
    cx = np.mean([v[0] for v in tri])
    cy = np.mean([v[1] for v in tri])
    return (cx < 2.0) and (cy >= 2.0)


def triangle_centroids(triangles):
    return np.array([np.mean(np.array(t), axis=0) for t in triangles])


def build_cells(triangles):
    is_refined = [is_refined_triangle(t) for t in triangles]
    cents = triangle_centroids(triangles)
    cells_L, cells_L1 = [], []
    for row in range(4):
        for col in range(4):
            if row >= 2 and col < 2:
                continue
            elems = [i for i, c in enumerate(cents)
                     if (not is_refined[i]) and (col <= c[0] < col+1) and (row <= c[1] < row+1)]
            cells_L.append(dict(x0=col, y0=row, size=1.0, cx=col+0.5, cy=row+0.5, elems=elems, level='L'))
    for row in range(4):
        for col in range(4):
            x0 = col * 0.5
            y0 = 2 + row * 0.5
            elems = [i for i, c in enumerate(cents)
                     if is_refined[i] and (x0 <= c[0] < x0+0.5) and (y0 <= c[1] < y0+0.5)]
            cells_L1.append(dict(x0=x0, y0=y0, size=0.5, cx=x0+0.25, cy=y0+0.25, elems=elems, level='L+1'))
    return cells_L, cells_L1, is_refined, cents


def add_mesh_background(fig, triangles, is_refined):
    for i, tri in enumerate(triangles):
        xs = [v[0] for v in tri] + [tri[0][0]]
        ys = [v[1] for v in tri] + [tri[0][1]]
        fill = 'rgba(206, 227, 248, 0.75)' if is_refined[i] else 'rgba(228, 235, 242, 0.95)'
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines', fill='toself',
            fillcolor=fill,
            line=dict(color='rgba(60, 70, 80, 0.35)', width=0.9),
            showlegend=False, hoverinfo='skip'
        ))


def add_refined_box(fig):
    fig.add_shape(type='rect', x0=0, y0=2, x1=2, y1=4,
                  line=dict(color='rgb(180, 50, 47)', width=2.2, dash='dash'))
    fig.add_annotation(x=1.0, y=3.87, text='Refined region',
                       showarrow=False, font=dict(size=15, color='rgb(160, 35, 35)'),
                       bgcolor='rgba(255,255,255,0.85)', bordercolor='rgba(160,35,35,0.35)', borderwidth=1)


def apply_pub_style(fig, x_range=(0,4), y_range=(0,4), legend_xy=(0.02,0.98)):
    fig.update_layout(
        title=None,
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=1280,
        height=1080,
        margin=dict(l=18, r=18, t=18, b=18, pad=0),
        font=dict(family='Arial, Helvetica, sans-serif', size=17, color='black'),
        showlegend=True,
        legend=dict(
            x=legend_xy[0], y=legend_xy[1], xanchor='left', yanchor='top', orientation='v',
            bgcolor='rgba(255,255,255,0.92)', bordercolor='rgba(0,0,0,0.22)', borderwidth=1,
            font=dict(size=14, color='black'), itemsizing='constant'
        )
    )
    fig.update_xaxes(
        title_text='x', range=[x_range[0]-0.02, x_range[1]+0.02],
        showgrid=False, zeroline=False, showline=True, mirror=True,
        linecolor='black', linewidth=1.5, ticks='outside', ticklen=6,
        tickfont=dict(size=14), title_font=dict(size=18), scaleanchor='y', constrain='domain'
    )
    fig.update_yaxes(
        title_text='y', range=[y_range[0]-0.02, y_range[1]+0.02],
        showgrid=False, zeroline=False, showline=True, mirror=True,
        linecolor='black', linewidth=1.5, ticks='outside', ticklen=6,
        tickfont=dict(size=14), title_font=dict(size=18), constrain='domain'
    )


def trim_png(path):
    img = Image.open(path).convert('RGB')
    bg = Image.new('RGB', img.size, 'white')
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        img.crop(bbox).save(path)


def save_fig(fig, png_name, caption, description):
    fig.write_image(png_name, scale=3)
    trim_png(png_name)
    with open(png_name + '.meta.json', 'w') as f:
        json.dump({'caption': caption, 'description': description}, f, indent=2)
'''

files = {}
files['morton_curve_visualize_pub.py'] = common + '''
triangles = build_triangles()
is_refined = [is_refined_triangle(t) for t in triangles]
centroids = triangle_centroids(triangles)
N = 16
cx_idx = np.clip((centroids[:, 0] / 4 * N).astype(int), 0, N - 1)
cy_idx = np.clip((centroids[:, 1] / 4 * N).astype(int), 0, N - 1)
morton_codes = np.array([morton2d(cx_idx[i], cy_idx[i]) for i in range(len(centroids))])
order = np.argsort(morton_codes)
sorted_cents = centroids[order]
sorted_codes = morton_codes[order]

fig = go.Figure()
add_mesh_background(fig, triangles, is_refined)
fig.add_trace(go.Scatter(
    x=sorted_cents[:,0], y=sorted_cents[:,1], mode='lines+markers',
    line=dict(color='rgb(214, 138, 0)', width=2.4, dash='dot'),
    marker=dict(symbol='circle', size=8.5, color='rgb(214, 138, 0)', line=dict(color='white', width=1.0)),
    name='Morton-ordered centroids', hovertemplate='Morton %{customdata}<extra></extra>', customdata=sorted_codes
))
for k in range(0, len(sorted_cents), max(1, len(sorted_cents)//12)):
    fig.add_annotation(
        x=float(sorted_cents[k,0]), y=float(sorted_cents[k,1]), text=str(int(sorted_codes[k])),
        showarrow=False, xshift=10, yshift=10,
        font=dict(size=12, color='rgb(140, 82, 0)'),
        bgcolor='rgba(255,255,255,0.80)'
    )
add_refined_box(fig)
apply_pub_style(fig, legend_xy=(0.02, 0.98))
save_fig(fig, 'morton_curve_mesh_pub.png',
         'Morton curve through triangle centroids on a two-level refined mesh.',
         'Publication-style 2D schematic of the centroid-based Morton ordering on a locally refined triangular mesh.')
'''

files['morton_search_visualize_pub.py'] = common + '''
triangles = build_triangles()
is_refined = [is_refined_triangle(t) for t in triangles]
centroids = triangle_centroids(triangles)
N = 16
cx_idx = np.clip((centroids[:, 0] / 4 * N).astype(int), 0, N - 1)
cy_idx = np.clip((centroids[:, 1] / 4 * N).astype(int), 0, N - 1)
morton_codes = np.array([morton2d(cx_idx[i], cy_idx[i]) for i in range(len(centroids))])
order = np.argsort(morton_codes)
sorted_cents = centroids[order]
sorted_codes = morton_codes[order]
queries = {'k₁': np.array([0.7, 0.5]), 'k₂': np.array([2.5, 1.2]), 'k₃': np.array([0.4, 3.1]), 'k₄': np.array([3.2, 3.6])}
radius = 4
colors = ['rgb(198,56,44)', 'rgb(44,139,74)', 'rgb(28,132,181)', 'rgb(129,70,165)']

fig = go.Figure()
add_mesh_background(fig, triangles, is_refined)
fig.add_trace(go.Scatter(
    x=sorted_cents[:,0], y=sorted_cents[:,1], mode='lines',
    line=dict(color='rgba(214, 138, 0, 0.35)', width=1.8, dash='dot'),
    name='Morton curve', showlegend=True, hoverinfo='skip'
))
legend_once_window = False
legend_once_nearest = False
for (qname, qpos), qcol in zip(queries.items(), colors):
    qix = int(np.clip(qpos[0] / 4 * N, 0, N - 1))
    qiy = int(np.clip(qpos[1] / 4 * N, 0, N - 1))
    q_code = morton2d(qix, qiy)
    nearest_idx = int(np.argmin(np.abs(sorted_codes - q_code)))
    lo = max(0, nearest_idx - radius)
    hi = min(len(sorted_cents)-1, nearest_idx + radius)
    nb = sorted_cents[lo:hi+1]
    fig.add_trace(go.Scatter(
        x=nb[:,0], y=nb[:,1], mode='markers',
        marker=dict(symbol='circle-open', size=17, color=qcol, line=dict(color=qcol, width=2.4)),
        name='Morton radius window' if not legend_once_window else None,
        showlegend=not legend_once_window, hoverinfo='skip'
    ))
    legend_once_window = True
    nc = sorted_cents[nearest_idx]
    fig.add_trace(go.Scatter(
        x=[nc[0]], y=[nc[1]], mode='markers',
        marker=dict(symbol='star', size=18, color=qcol, line=dict(color='black', width=0.9)),
        name='Nearest centroid' if not legend_once_nearest else None,
        showlegend=not legend_once_nearest,
        hovertemplate=f'{qname}: nearest centroid, Morton {int(sorted_codes[nearest_idx])}<extra></extra>'
    ))
    legend_once_nearest = True
    fig.add_trace(go.Scatter(
        x=[qpos[0]], y=[qpos[1]], mode='markers+text',
        marker=dict(symbol='x', size=16, color=qcol, line=dict(color=qcol, width=2.5)),
        text=[qname], textposition='top right', textfont=dict(size=16, color=qcol),
        name='Query point' if qname == 'k₁' else None, showlegend=(qname == 'k₁'),
        hovertemplate=f'{qname}: query, Morton {q_code}<extra></extra>'
    ))
    fig.add_annotation(x=float(nc[0]), y=float(nc[1]), ax=float(qpos[0]), ay=float(qpos[1]),
                       xref='x', yref='y', axref='x', ayref='y', showarrow=True,
                       arrowhead=3, arrowsize=1.0, arrowwidth=2.0, arrowcolor=qcol)
for k in range(0, len(sorted_cents), 10):
    fig.add_annotation(x=float(sorted_cents[k,0]), y=float(sorted_cents[k,1]), text=str(int(sorted_codes[k])),
                       showarrow=False, xshift=8, yshift=8,
                       font=dict(size=11, color='rgb(150,95,0)'), bgcolor='rgba(255,255,255,0.78)')
add_refined_box(fig)
apply_pub_style(fig, legend_xy=(0.66, 0.98))
save_fig(fig, 'morton_search_pub.png',
         'Morton-band search: query encoding, nearest centroid selection, and radius-based candidate window.',
         'Publication-style 2D schematic showing four queries, nearest-centroid matching by Morton code, and the radius window searched on the Morton curve.')
'''

files['mesh_aligned_octree_morton_visualize_pub.py'] = common + '''
triangles = build_triangles()
cells_L, cells_L1, is_refined, cents = build_cells(triangles)
N_L, N_L1 = 8, 16
for c in cells_L:
    ix = int(np.clip(c['cx'] / 4 * N_L, 0, N_L - 1))
    iy = int(np.clip(c['cy'] / 4 * N_L, 0, N_L - 1))
    c['morton'] = morton2d(ix, iy, bits=4)
for c in cells_L1:
    ix = int(np.clip(c['cx'] / 4 * N_L1, 0, N_L1 - 1))
    iy = int(np.clip(c['cy'] / 4 * N_L1, 0, N_L1 - 1))
    c['morton'] = morton2d(ix, iy, bits=4)
all_sorted = sorted(cells_L + cells_L1, key=lambda c: (c['morton'], 0 if c['level'] == 'L' else 1))

fig = go.Figure()
add_mesh_background(fig, triangles, is_refined)
shown_L = False
for c in cells_L:
    x0, y0, s = c['x0'], c['y0'], c['size']
    fig.add_trace(go.Scatter(x=[x0, x0+s, x0+s, x0, x0], y=[y0, y0, y0+s, y0+s, y0],
                             mode='lines', fill='toself', fillcolor='rgba(75, 153, 218, 0.08)',
                             line=dict(color='rgb(56, 121, 179)', width=2.0),
                             name='Level L cells' if not shown_L else None, showlegend=not shown_L, hoverinfo='skip'))
    shown_L = True
shown_L1 = False
for c in cells_L1:
    x0, y0, s = c['x0'], c['y0'], c['size']
    fig.add_trace(go.Scatter(x=[x0, x0+s, x0+s, x0, x0], y=[y0, y0, y0+s, y0+s, y0],
                             mode='lines', fill='toself', fillcolor='rgba(229, 184, 65, 0.10)',
                             line=dict(color='rgb(184, 136, 19)', width=1.8),
                             name='Level L+1 cells' if not shown_L1 else None, showlegend=not shown_L1, hoverinfo='skip'))
    shown_L1 = True
fig.add_trace(go.Scatter(
    x=[c['cx'] for c in all_sorted], y=[c['cy'] for c in all_sorted], mode='lines+markers',
    line=dict(color='rgba(100,100,100,0.55)', width=2.0, dash='dot'),
    marker=dict(size=8.5, color=['rgb(56, 121, 179)' if c['level'] == 'L' else 'rgb(214, 138, 0)' for c in all_sorted],
                line=dict(color='white', width=0.9)),
    name='Sorted (Morton, level) cell centroids', hoverinfo='skip'
))
for k in range(0, len(all_sorted), 6):
    c = all_sorted[k]
    fig.add_annotation(x=c['cx'], y=c['cy'], text=f"{int(c['morton'])},{c['level']}", showarrow=False,
                       xshift=10, yshift=10, font=dict(size=11, color='black'), bgcolor='rgba(255,255,255,0.82)')
fig.add_annotation(x=1.0, y=3.02, text='No level-L cell<br>(refined to L+1)', showarrow=False,
                   font=dict(size=13, color='rgb(150,35,35)'), align='center',
                   bgcolor='rgba(255,255,255,0.88)', bordercolor='rgba(150,35,35,0.35)', borderwidth=1)
boundary_ids = [i for i, t in enumerate(triangles)
                if abs(np.mean([v[0] for v in t]) - 2.0) < 0.45 and abs(np.mean([v[1] for v in t]) - 2.35) < 0.45]
for j, idx in enumerate(boundary_ids[:2]):
    tri = triangles[idx]
    xs = [v[0] for v in tri] + [tri[0][0]]
    ys = [v[1] for v in tri] + [tri[0][1]]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', fill='toself', fillcolor='rgba(200,70,60,0.45)',
                             line=dict(color='rgb(170,45,38)', width=2.2),
                             name='Boundary element with multi-cell registration' if j == 0 else None,
                             showlegend=(j == 0), hoverinfo='skip'))
add_refined_box(fig)
apply_pub_style(fig, legend_xy=(0.02, 0.98))
save_fig(fig, 'new_l2_multilevel_octree_pub.png',
         'Matched two-level octree and composite (Morton, level) ordering of leaf-cell centroids.',
         'Publication-style 2D schematic of the MALMO leaf cells, composite Morton-level ordering, and boundary element shown for multi-cell registration near a cell boundary.')
'''

files['mesh_aligned_3x3x3_search_visualize_pub.py'] = common + '''
triangles = build_triangles()
cells_L, cells_L1, is_refined, cents = build_cells(triangles)
q = np.array([1.55, 2.55])
center = None
for c in cells_L1:
    if c['x0'] <= q[0] < c['x0'] + c['size'] and c['y0'] <= q[1] < c['y0'] + c['size']:
        center = c
        break
assert center is not None
neighbors = [c for c in cells_L1 if abs((c['x0'] - center['x0']) / 0.5) <= 1 and abs((c['y0'] - center['y0']) / 0.5) <= 1]
neighbor_elem_ids = sorted({e for c in neighbors for e in c['elems']})

fig = go.Figure()
add_mesh_background(fig, triangles, is_refined)
shown_neighbor = False
for c in neighbors:
    x0, y0, s = c['x0'], c['y0'], c['size']
    fig.add_trace(go.Scatter(x=[x0, x0+s, x0+s, x0, x0], y=[y0, y0, y0+s, y0+s, y0],
                             mode='lines', fill='toself', fillcolor='rgba(244, 170, 74, 0.08)',
                             line=dict(color='rgb(212, 127, 26)', width=2.1),
                             name='3×3 neighborhood' if not shown_neighbor else None, showlegend=not shown_neighbor, hoverinfo='skip'))
    shown_neighbor = True
x0, y0, s = center['x0'], center['y0'], center['size']
fig.add_trace(go.Scatter(x=[x0, x0+s, x0+s, x0, x0], y=[y0, y0, y0+s, y0+s, y0],
                         mode='lines', fill='toself', fillcolor='rgba(84, 150, 229, 0.24)',
                         line=dict(color='rgb(41, 98, 174)', width=3.0), name='Query cell', showlegend=True, hoverinfo='skip'))
for idx in neighbor_elem_ids:
    tri = triangles[idx]
    xs = [v[0] for v in tri] + [tri[0][0]]
    ys = [v[1] for v in tri] + [tri[0][1]]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', fill='toself', fillcolor='rgba(85, 153, 95, 0.34)',
                             line=dict(color='rgba(63, 117, 71, 0.85)', width=1.2),
                             name='Candidate elements' if idx == neighbor_elem_ids[0] else None,
                             showlegend=(idx == neighbor_elem_ids[0]), hoverinfo='skip'))
fig.add_trace(go.Scatter(x=[q[0]], y=[q[1]], mode='markers+text',
                         marker=dict(symbol='x', size=18, color='rgb(30,30,30)', line=dict(color='rgb(30,30,30)', width=2.4)),
                         text=['q'], textposition='top right', textfont=dict(size=16, color='black'),
                         name='Query point', showlegend=True, hoverinfo='skip'))
fig.add_annotation(x=center['cx'], y=center['cy'], text='(i,j)', showarrow=False,
                   font=dict(size=14, color='rgb(25,70,130)'), bgcolor='rgba(255,255,255,0.82)')
fig.add_annotation(x=1.55, y=3.72, text='2D view of the 3×3×3 search', showarrow=False,
                   font=dict(size=14, color='rgb(120,70,10)'), bgcolor='rgba(255,255,255,0.86)')
add_refined_box(fig)
apply_pub_style(fig, legend_xy=(0.63, 0.98))
save_fig(fig, 'mesh_aligned_3x3x3_search_pub.png',
         'Mesh-aligned neighborhood search with the query cell and its 3×3 neighborhood in the refined region.',
         'Publication-style 2D schematic representing the 3×3×3 MALMO neighborhood search, with the query cell highlighted and candidate elements shaded inside the searched cells.')
'''

for name, content in files.items():
    path = out / name
    path.write_text(textwrap.dedent(content), encoding='utf-8')
    py_compile.compile(str(path), doraise=True)

cwd = os.getcwd()
os.chdir(out)
for name in files:
    runpy.run_path(name, run_name='__main__')
os.chdir(cwd)

manifest = {
    'scripts': list(files.keys()),
    'images': [
        'morton_curve_mesh_pub.png',
        'morton_search_pub.png',
        'new_l2_multilevel_octree_pub.png',
        'mesh_aligned_3x3x3_search_pub.png'
    ]
}
(out / 'publication_ready_fig_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))