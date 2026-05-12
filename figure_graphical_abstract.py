"""
Graphical abstract for "A Data-Driven Test of Local Recoverability for Materials Design"
Vector PDF output. Editable, typo-free, built from matplotlib primitives.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, Circle, Rectangle
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D  # noqa

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig = plt.figure(figsize=(13.8, 4.4))
gs = GridSpec(1, 4, figure=fig, width_ratios=[1.0, 1.0, 1.15, 1.0],
              wspace=0.18, left=0.045, right=0.965, top=0.88, bottom=0.10)

# =============================================================================
# Panel 1: Control space
# =============================================================================
ax1 = fig.add_subplot(gs[0, 0])
np.random.seed(7)
N = 320
x1 = np.random.uniform(-1, 1, N)
x2 = np.random.uniform(-1, 1, N)
color_field = 0.6 * x1 + 0.4 * x2 + 0.3 * np.sin(2 * x1)
ax1.scatter(x1, x2, c=color_field, cmap="Spectral_r",
            s=22, edgecolor="white", linewidth=0.3, zorder=3)

ax1.set_xlim(-1.05, 1.05)
ax1.set_ylim(-1.05, 1.05)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$", rotation=0, labelpad=10)
ax1.set_aspect("equal")
for spine in ax1.spines.values():
    spine.set_linewidth(0.8)

ax1.text(0.5, -0.16,
         "Process controls $x_i$\nsampled across a 2D\ndesign space",
         transform=ax1.transAxes, ha="center", va="top",
         fontsize=8.0, style="italic", color="#444")

# =============================================================================
# Panel 2: Microstructure data  (wider spread, ellipses to the right)
# =============================================================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_aspect("equal")
ax2.axis("off")


def make_ch_like(seed, size=80):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((size, size))
    f = gaussian_filter(f, sigma=3.5)
    return (f > 0).astype(float)


def make_voronoi_like(seed, size=80, n_seeds=18):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, size, (n_seeds, 2))
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    d2 = (grid_x[..., None] - pts[:, 0])**2 + (grid_y[..., None] - pts[:, 1])**2
    labels = np.argmin(d2, axis=2)
    shade = rng.uniform(0.2, 1.0, n_seeds)
    return shade[labels]


def make_fft_pattern(seed, size=80):
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    r = np.sqrt(xx**2 + yy**2)
    base = np.exp(-12 * r**2)
    base += 0.4 * np.exp(-30 * (r - 0.35)**2) * np.cos(
        6 * np.arctan2(yy, xx) + rng.uniform(0, 2 * np.pi)
    )
    return base


# Wider spread: each card reveals ~55% of the previous; stack fills more width
stack_generators = [
    (make_voronoi_like, 11, "gray"),
    (make_ch_like,      17, "gray"),
    (make_fft_pattern,  51, "magma"),
]
n_cards = len(stack_generators)
card_w, card_h = 0.30, 0.30
step_x = 0.17
step_y = 0.025
base_x, base_y = 0.03, 0.33

for i, (gen, seed, cm) in enumerate(stack_generators):
    bx = base_x + i * step_x
    by = base_y + i * step_y
    border = Rectangle((bx - 0.012, by - 0.012),
                       card_w + 0.024, card_h + 0.024,
                       facecolor="white", edgecolor="#444", linewidth=0.8,
                       transform=ax2.transAxes, zorder=2 * i + 1)
    ax2.add_patch(border)
    inner = ax2.inset_axes([bx, by, card_w, card_h])
    inner.imshow(gen(seed), cmap=cm, interpolation="nearest", aspect="equal")
    inner.set_xticks([])
    inner.set_yticks([])
    for sp in inner.spines.values():
        sp.set_visible(False)
    inner.set_zorder(2 * i + 2)

# Ellipsis dots immediately to the right of the topmost card
top_card_right = base_x + (n_cards - 1) * step_x + card_w + 0.018
top_card_y_center = base_y + (n_cards - 1) * step_y + card_h / 2
for dx in [0.025, 0.060, 0.095]:
    ax2.plot(top_card_right + dx, top_card_y_center,
             marker="o", color="#444", markersize=3.5,
             transform=ax2.transAxes, zorder=20, clip_on=False)

ax2.text(0.5, 0.10,
         "Distinct controls may yield\nvisibly different fields but\nsimilar descriptor values",
         transform=ax2.transAxes, ha="center", va="top",
         fontsize=8.0, style="italic", color="#444")

# =============================================================================
# Panel 3: Descriptor space (main cloud + clearly visible blue cluster)
# =============================================================================
ax3 = fig.add_subplot(gs[0, 2], projection="3d")

np.random.seed(11)
M = 600
u = np.random.uniform(-1, 1, M)
v = np.random.uniform(-1, 1, M)

# --- Main sheet: smooth color gradient blue (low u) -> red (high u)
width_at_v = 1.0 - 0.35 * (v + 1) / 2.0
X_main = u * width_at_v + 0.04 * np.random.randn(M)
Y_main = v + 0.04 * np.random.randn(M)
Z_main = 0.10 * u * v + 0.04 * np.random.randn(M)

ax3.scatter(X_main, Y_main, Z_main, c=u, cmap="Spectral_r",
            s=14, alpha=0.85, edgecolor="white", linewidth=0.2,
            vmin=-1, vmax=1)

# --- Blue cluster: low-u (blue) points placed where the main scatter is red.
#     Rendered identically to the main points so it reads as part of the same
#     population whose descriptor coords happen to overlap a different control
#     region.
Mf = 140
uf = np.random.uniform(-0.95, -0.55, Mf)              # strongly blue
Xf = 0.40 + 0.10 * np.random.randn(Mf)                 # scooted left a bit
Yf = -0.45 + 0.12 * np.random.randn(Mf)
Zf = 0.30 + 0.04 * np.random.randn(Mf)
ax3.scatter(Xf, Yf, Zf, c=uf, cmap="Spectral_r",
            s=14, alpha=0.85, edgecolor="white", linewidth=0.2,
            vmin=-1, vmax=1)

ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_zticks([])
ax3.grid(False)
ax3.view_init(elev=22, azim=-55)
ax3.set_box_aspect((1.2, 1.2, 0.8))
for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
    axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    axis.line.set_color((1.0, 1.0, 1.0, 0.0))

# Annotations: cluster sits in the upper portion of the panel, so its
# caption goes up; the well-separated caption moves to the lower region.
ax3.text2D(0.02, 0.78, "Blue cluster: \n similar descriptors from\ndifferent controls",
           transform=ax3.transAxes, fontsize=8.0, color="#1A4A78",
           bbox=dict(boxstyle="round,pad=0.3", fc="white",
                     ec="#1A4A78", lw=0.8, alpha=0.9))
ax3.text2D(0.50, 0.02, "Locally well-\nseparated region",
           transform=ax3.transAxes, fontsize=8.0, color="#1A4A78",
           bbox=dict(boxstyle="round,pad=0.3", fc="white",
                     ec="#1A4A78", lw=0.8, alpha=0.9))

# =============================================================================
# Panel 4: Local recoverability score heatmap (jet)
# =============================================================================
ax4 = fig.add_subplot(gs[0, 3])

nx, ny = 80, 80
xg = np.linspace(-1, 1, nx)
yg = np.linspace(-1, 1, ny)
Xg, Yg = np.meshgrid(xg, yg)

e_field = 0.85 * np.ones_like(Xg)
e_field -= 0.7 * np.exp(-((Xg - 0.55)**2 + (Yg - 0.55)**2) / 0.10)
e_field -= 0.35 * np.exp(-((Xg + 0.05)**2 / 0.5 + (Yg + 0.05)**2 / 0.04))
e_field -= 0.45 * np.exp(-((Xg - 0.45)**2 + (Yg + 0.55)**2) / 0.15)
e_field = gaussian_filter(e_field, sigma=1.2)
e_field = np.clip(e_field, 0.05, 1.0)

im = ax4.imshow(e_field, extent=[-1, 1, -1, 1], origin="lower",
                cmap="jet", aspect="equal", vmin=0, vmax=1)
ax4.contour(Xg, Yg, e_field, levels=[0.4, 0.65],
            colors="black", linewidths=0.6, alpha=0.7)

ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_xlabel("$x_1$")
ax4.set_ylabel("$x_2$", rotation=0, labelpad=10)
for spine in ax4.spines.values():
    spine.set_linewidth(0.8)

ax4.annotate("high\nscore", xy=(-0.55, -0.55), ha="center", va="center",
             fontsize=8.5, color="#0B3D6B",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, ec="none"))
ax4.annotate("ambiguous", xy=(-0.25, 0.0), ha="center", va="center",
             fontsize=8.5, color="#3A2A0A",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, ec="none"))
ax4.annotate("non-recoverable", xy=(0.55, 0.55), ha="center", va="center",
             fontsize=8.0, color="#7A1A1A",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.88, ec="none"))
ax4.annotate("poor local\nrecovery", xy=(0.45, -0.55), ha="center", va="center",
             fontsize=8.0, color="#7A1A1A",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.88, ec="none"))

cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04, shrink=0.85)
cbar.ax.tick_params(labelsize=7)
cbar.set_label("$e_i$", rotation=0, labelpad=8, fontsize=10)
cbar.outline.set_linewidth(0.6)

ax4.text(0.5, -0.16,
         "High $e_i$: local control variation\nis recoverable from nearby\ndescriptor neighborhoods",
         transform=ax4.transAxes, ha="center", va="top",
         fontsize=8.0, style="italic", color="#444")

# =============================================================================
# Finalize: panel positions, aligned titles, arrows, k-NN inset
# =============================================================================
fig.canvas.draw()
panel_axes = [ax1, ax2, ax3, ax4]
boxes = [a.get_position() for a in panel_axes]

# --- Aligned panel titles via fig.text at a shared y figure-coord
title_y = 0.93   # uniform y for all four titles
title_texts = [
    "Control space  $(x)$",
    "Microstructure data",
    "Descriptor space  $(y)$",
    "Local recoverability score  $e_i$",
]
for text, box in zip(title_texts, boxes):
    cx = 0.5 * (box.x0 + box.x1)
    fig.text(cx, title_y, text, ha="center", va="bottom",
             fontweight="bold", fontsize=12)


def draw_chunky_arrow(x0, x1, y, fig,
                      shaft_h=0.045, head_h=0.085, head_len=0.011,
                      facecolor="#B8D4F0", edgecolor="#7AA8D8"):
    body_x1 = x1 - head_len
    verts = [
        (x0,      y - shaft_h / 2),
        (body_x1, y - shaft_h / 2),
        (body_x1, y - head_h / 2),
        (x1,      y),
        (body_x1, y + head_h / 2),
        (body_x1, y + shaft_h / 2),
        (x0,      y + shaft_h / 2),
    ]
    poly = mpatches.Polygon(verts, closed=True,
                            facecolor=facecolor, edgecolor=edgecolor,
                            linewidth=0.8, transform=fig.transFigure,
                            zorder=10)
    fig.patches.append(poly)


arrow_y = 0.52
reference_arrow_length = min(
    boxes[i + 1].x0 - boxes[i].x1 - 0.008 for i in range(2)
)
for i in range(3):
    x_s = boxes[i].x1 + 0.004
    x_e = boxes[i + 1].x0 - 0.004
    if i == 2:
        x_e = min(x_e, x_s + reference_arrow_length)
    if x_e > x_s:
        draw_chunky_arrow(x_s, x_e, arrow_y, fig)

# Square local k-NN inset, tucked into upper-right corner of panel 3
fig_w, fig_h = fig.get_figwidth(), fig.get_figheight()
panel3_box = boxes[2]
inset_w_fig = 0.078
inset_h_fig = inset_w_fig * (fig_w / fig_h)
inset_x = panel3_box.x1 - inset_w_fig - 0.005
inset_y = panel3_box.y1 - inset_h_fig - 0.04
ax_inset = fig.add_axes([inset_x, inset_y, inset_w_fig, inset_h_fig])
ax_inset.set_xlim(-1, 1)
ax_inset.set_ylim(-1, 1)
ax_inset.set_aspect("equal", adjustable="box")
ax_inset.set_xticks([])
ax_inset.set_yticks([])
rng = np.random.default_rng(3)
pts = rng.normal(0, 0.5, (60, 2))
ax_inset.scatter(pts[:, 0], pts[:, 1], s=6, c="#888", alpha=0.6)
anchor = np.array([0.1, 0.05])
dists = np.linalg.norm(pts - anchor, axis=1)
knn_idx = np.argsort(dists)[:12]
ax_inset.scatter(pts[knn_idx, 0], pts[knn_idx, 1], s=14,
                 c="#D4564B", edgecolor="white", linewidth=0.5)
ax_inset.scatter(*anchor, s=50, marker="*", c="#1A1A1A", zorder=5)
disk = Circle(anchor, 0.45, fill=False, edgecolor="#1A1A1A", lw=1.0, ls="--")
ax_inset.add_patch(disk)
ax_inset.set_title("local $k$-NN", fontsize=7.5, pad=2)
for spine in ax_inset.spines.values():
    spine.set_linewidth(0.6)

fig.savefig("graphical_abstract.pdf", bbox_inches="tight")
fig.savefig("graphical_abstract.png", bbox_inches="tight", dpi=200)
print("Saved")