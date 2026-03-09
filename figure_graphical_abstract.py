import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.path import Path
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']
plt.rcParams['mathtext.fontset'] = 'dejavusans'

fig = plt.figure(figsize=(14, 7.5), dpi=250, facecolor='#fafaf7')

# ---- HELPERS ----
def generate_micro(size=80, seed=42, sigma=3.5, threshold=0.0):
    np.random.seed(seed)
    noise = np.random.randn(size, size)
    smooth = gaussian_filter(noise, sigma=sigma)
    return (smooth > threshold).astype(float)

def generate_score_map(size=80, base_score=0.8, noise_level=0.03, center_dip=False):
    np.random.seed(42)
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    distSq = X**2 + Y**2
    
    if center_dip:
        # Non-identifiable: base score is low, but we want a "blue" spot in the middle
        # Viridis 0.15 is purple. Viridis 0.4 is blue/teal.
        score = base_score - 0.35 * np.exp(-10 * distSq)
    else:
        # Identifiable: high score with some slight coherent variation + noise
        score = np.full((size, size), base_score)
        score += 0.04 * np.cos(3 * X) * np.sin(3 * Y)
    
    score += np.random.randn(size, size) * noise_level
    return np.clip(score, 0, 1)

def viridis_rgb(val):
    """Return RGBA tuple for a 0-1 value on viridis."""
    cmap = plt.colormaps['viridis']
    return cmap(val)

def draw_3d_box_scored(ax, x, y, w, h, d, score, ec='#999', noisy=False, center_dip=False):
    """Draw a 3D box with front face colored by viridis score."""
    dx, dy = d * 0.4, d * 0.3
    
    if noisy:
        s_map = generate_score_map(size=80, base_score=score, center_dip=center_dip)
        ax.imshow(s_map, extent=[x, x+w, y, y+h], origin='lower', 
                  cmap='viridis', zorder=2, interpolation='bilinear', vmin=0, vmax=1)
        # Outline for front
        rect = plt.Rectangle((x, y), w, h, fill=False, ec=ec, lw=0.5, zorder=3)
        ax.add_patch(rect)
        display_score = np.mean(s_map)
    else:
        face_color = viridis_rgb(score)
        front = plt.Polygon([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], closed=True,
                            fc=face_color, ec=ec, lw=0.5, zorder=2)
        ax.add_patch(front)
        display_score = score
        
    # Darken for top/side
    top_color = viridis_rgb(min(1, display_score + 0.08))
    side_color = viridis_rgb(max(0, display_score - 0.08))
    
    top = plt.Polygon([[x,y+h],[x+dx,y+h+dy],[x+w+dx,y+h+dy],[x+w,y+h]], closed=True,
                      fc=top_color, ec=ec, lw=0.5, zorder=2)
    right = plt.Polygon([[x+w,y],[x+w+dx,y+dy],[x+w+dx,y+h+dy],[x+w,y+h]], closed=True,
                        fc=side_color, ec=ec, lw=0.5, zorder=2)
    ax.add_patch(top); ax.add_patch(right)

def draw_3d_manifold(ax, cx, cy, w, h, d, fill_color, edge_color, pinch=0.5):
    """Draw a 3D extruded manifold shape with depth."""
    hw, hh = w/2, h/2
    neck = pinch * hw
    dx, dy = d * 0.35, d * 0.25
    
    # Generate the 2D profile points (right side, top to bottom)
    n = 60
    right_profile = []
    for i in range(n+1):
        t = i / n
        if t <= 0.5:
            width = hw - (hw - neck) * (2*t)**2
        else:
            width = neck + (hw - neck) * (2*(t-0.5))**2
        yy = cy + hh - 2*hh*t
        right_profile.append((cx + width, yy))
    
    # Left profile (mirror)
    left_profile = []
    for i in range(n+1):
        t = i / n
        if t <= 0.5:
            width = hw - (hw - neck) * (2*t)**2
        else:
            width = neck + (hw - neck) * (2*(t-0.5))**2
        yy = cy - hh + 2*hh*t
        left_profile.append((cx - width, yy))
    
    # Back face (offset by dx, dy) - draw first
    back_verts = [(px + dx, py + dy) for (px, py) in right_profile]
    back_verts += [(px + dx, py + dy) for (px, py) in left_profile]
    back_verts.append(back_verts[0])
    back_codes = [Path.MOVETO] + [Path.LINETO] * (len(back_verts) - 2) + [Path.CLOSEPOLY]
    
    # Slightly darker fill for back
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(fill_color)
    back_color = (max(0, r - 0.06), max(0, g - 0.06), max(0, b - 0.06))
    
    ax.add_patch(mpatches.PathPatch(Path(back_verts, back_codes),
                 fc=back_color, ec=edge_color, lw=0.6, alpha=0.5, zorder=0))
    
    # Top connecting face (between front-top and back-top edges)
    # Connect the top curves of front and back
    top_verts = []
    top_codes = []
    # Front top edge (right profile, first ~15 points where y is near top)
    front_top = right_profile[:8]
    back_top = [(px + dx, py + dy) for (px, py) in front_top]
    
    top_face = front_top + list(reversed(back_top))
    top_face.append(top_face[0])
    top_codes_l = [Path.MOVETO] + [Path.LINETO] * (len(top_face) - 2) + [Path.CLOSEPOLY]
    ax.add_patch(mpatches.PathPatch(Path(top_face, top_codes_l),
                 fc=back_color, ec=edge_color, lw=0.4, alpha=0.4, zorder=0.5))
    
    # Right side connecting strips (between front right profile and back right profile)
    for i in range(len(right_profile)-1):
        strip = [
            right_profile[i],
            right_profile[i+1],
            (right_profile[i+1][0] + dx, right_profile[i+1][1] + dy),
            (right_profile[i][0] + dx, right_profile[i][1] + dy),
            right_profile[i]
        ]
        strip_codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        
        # Color the side strips by viridis based on y-position (distance from center)
        mid_y = (strip[0][1] + strip[1][1]) / 2
        t_norm = abs(mid_y - cy) / hh  # 0 at center, 1 at edges
        side_alpha = 0.3
        side_c = viridis_rgb(t_norm * 0.4 + 0.1)
        
        ax.add_patch(mpatches.PathPatch(Path(strip, strip_codes),
                     fc=side_c, ec='none', lw=0, alpha=side_alpha, zorder=0.5))
    
    # Front face (on top)
    front_verts = right_profile + left_profile
    front_verts.append(front_verts[0])
    front_codes = [Path.MOVETO] + [Path.LINETO] * (len(front_verts) - 2) + [Path.CLOSEPOLY]

    # --- NEW: Top Flat Face (Parallelogram) ---
    # Connect Top-Left-Front -> Top-Right-Front -> Top-Right-Back -> Top-Left-Back
    # left_profile[-1] is Top-Left Front
    # right_profile[0] is Top-Right Front
    tl_f = left_profile[-1]
    tr_f = right_profile[0]
    tr_b = (tr_f[0] + dx, tr_f[1] + dy)
    tl_b = (tl_f[0] + dx, tl_f[1] + dy)
    
    top_flat_verts = [tl_f, tr_f, tr_b, tl_b, tl_f]
    top_flat_codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    
    # Use same color as top-connecting face (darker fill)
    ax.add_patch(mpatches.PathPatch(Path(top_flat_verts, top_flat_codes),
                 fc=back_color, ec=edge_color, lw=0.6, alpha=0.6, zorder=0.6))

    ax.add_patch(mpatches.PathPatch(Path(front_verts, front_codes),
                 fc=fill_color, ec=edge_color, lw=1.2, zorder=1))

# ============================================================
# PANEL 1: IDENTIFIABLE (top)
# ============================================================
ax1 = fig.add_axes([0.01, 0.51, 0.49, 0.46])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 5.2)
ax1.set_aspect('equal')
ax1.axis('off')

bg1 = FancyBboxPatch((0.05, 0.05), 9.9, 4.85, boxstyle="round,pad=0.12",
                      fc='white', ec='#d8d8d0', lw=0.7, zorder=0)
ax1.add_patch(bg1)
ax1.text(5, 5.15, 'I D E N T I F I A B L E', fontsize=10, fontweight='bold',
         color='#1a9850', ha='center', va='top', zorder=5)

# Control box - HIGH score (yellow-green)
draw_3d_box_scored(ax1, 0.4, 0.7, 1.6, 2.3, 0.5, score=0.88, noisy=True)
ax1.text(1.2, 0.4, r'$x_1$', fontsize=9, color='#333', ha='center', style='italic')
ax1.text(0.12, 1.85, r'$x_2$', fontsize=9, color='#333', ha='center', style='italic', rotation=90)
ax1.text(1.35, 3.35, 'Control Space', fontsize=7.5, fontweight='bold', color='#555', ha='center')

# Score label on the box
ax1.text(1.2, 1.85, r'$e \approx 1$', fontsize=12, color='#333', ha='center', va='center',
         fontweight='bold', zorder=6)

# Control points
ax1.add_patch(Circle((0.9, 2.4), 0.1, fc='#1a9850', ec='white', lw=1.3, zorder=5))
ax1.add_patch(Circle((1.6, 1.2), 0.1, fc='#2e7d32', ec='white', lw=1.3, zorder=5))

# f label and arrows
ax1.text(3.05, 2.9, r'$f$', fontsize=12, color='#333', ha='center', style='italic')
ax1.annotate('', xy=(3.8, 2.3), xytext=(2.35, 2.4),
            arrowprops=dict(arrowstyle='->', color='#1a9850', lw=1.3))
ax1.annotate('', xy=(3.8, 1.4), xytext=(2.35, 1.2),
            arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=1.3))

# 3D Descriptor manifold
draw_3d_manifold(ax1, 5.0, 1.95, 1.8, 2.8, 0.55, '#f1f8e9', '#66bb6a', pinch=0.6)
ax1.text(5.0, 3.6, 'Descriptor Space', fontsize=7.5, fontweight='bold', color='#555', ha='center')

# Well-separated descriptor points
ax1.add_patch(Circle((4.8, 2.5), 0.1, fc='#1a9850', ec='white', lw=1.3, zorder=5))
ax1.add_patch(Circle((5.2, 1.3), 0.1, fc='#2e7d32', ec='white', lw=1.3, zorder=5))
ax1.add_patch(Circle((4.8, 2.5), 0.38, fc='none', ec='#1a9850', lw=0.7, ls='--', zorder=4, alpha=0.5))
ax1.add_patch(Circle((5.2, 1.3), 0.38, fc='none', ec='#2e7d32', lw=0.7, ls='--', zorder=4, alpha=0.5))

# Dashed lines to micros
ax1.plot([5.65, 6.55], [2.5, 2.55], '--', color='#1a9850', lw=0.5, alpha=0.4, zorder=3)
ax1.plot([5.65, 6.55], [1.3, 1.2], '--', color='#2e7d32', lw=0.5, alpha=0.4, zorder=3)

# Microstructure insets
micro1 = generate_micro(80, seed=42, sigma=3.5)
micro2 = generate_micro(80, seed=77, sigma=4.5)

ax_m1 = fig.add_axes([0.33, 0.75, 0.05, 0.09])
ax_m1.imshow(micro1, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
ax_m1.set_xticks([]); ax_m1.set_yticks([])
for s in ax_m1.spines.values(): s.set_edgecolor('#1a9850'); s.set_linewidth(1.8)

ax_m2 = fig.add_axes([0.33, 0.58, 0.05, 0.09])
ax_m2.imshow(micro2, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
ax_m2.set_xticks([]); ax_m2.set_yticks([])
for s in ax_m2.spines.values(): s.set_edgecolor('#2e7d32'); s.set_linewidth(1.8)

# Check label
ax1.text(8.2, 0.35, '✓  Distinct outcomes → stable inverse', fontsize=7.5, color='#1a9850',
         ha='center', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.25', fc='#e8f5e9', ec='#a5d6a7', lw=0.5))

# ============================================================
# PANEL 2: NON-IDENTIFIABLE (bottom)
# ============================================================
ax2 = fig.add_axes([0.01, 0.02, 0.49, 0.46])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 5.2)
ax2.set_aspect('equal')
ax2.axis('off')

bg2 = FancyBboxPatch((0.05, 0.05), 9.9, 4.85, boxstyle="round,pad=0.12",
                      fc='white', ec='#d8d8d0', lw=0.7, zorder=0)
ax2.add_patch(bg2)
ax2.text(5, 5.15, 'N O N - I D E N T I F I A B L E', fontsize=10, fontweight='bold',
         color='#c0392b', ha='center', va='top', zorder=5)

# Control box - LOW score (purple/dark)
draw_3d_box_scored(ax2, 0.4, 0.7, 1.6, 2.3, 0.5, score=0.12, noisy=True, center_dip=True)
ax2.text(1.2, 0.4, r'$x_1$', fontsize=9, color='#ccc', ha='center', style='italic')
ax2.text(0.12, 1.85, r'$x_2$', fontsize=9, color='#ccc', ha='center', style='italic', rotation=90)
ax2.text(1.35, 3.35, 'Control Space', fontsize=7.5, fontweight='bold', color='#555', ha='center')

# Score label on the box
ax2.text(1.2, 1.85, r'$e \approx 0$', fontsize=12, color='white', ha='center', va='center',
         fontweight='bold', zorder=6)

# Control points
ax2.add_patch(Circle((0.8, 2.6), 0.1, fc='#c0392b', ec='white', lw=1.3, zorder=5))
ax2.add_patch(Circle((1.7, 1.0), 0.1, fc='#e74c3c', ec='white', lw=1.3, zorder=5))

ax2.text(3.05, 2.9, r'$f$', fontsize=12, color='#333', ha='center', style='italic')
ax2.annotate('', xy=(3.95, 2.0), xytext=(2.35, 2.6),
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.3))
ax2.annotate('', xy=(3.95, 1.85), xytext=(2.35, 1.0),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.3))

# 3D Descriptor manifold (more pinched)
draw_3d_manifold(ax2, 5.0, 1.95, 1.8, 2.8, 0.55, '#fce4ec', '#e57373', pinch=0.35)
ax2.text(5.0, 3.6, 'Descriptor Space', fontsize=7.5, fontweight='bold', color='#555', ha='center')

# Overlapping points
ax2.add_patch(Circle((4.92, 1.98), 0.1, fc='#c0392b', ec='white', lw=1.3, zorder=5))
ax2.add_patch(Circle((5.02, 1.88), 0.1, fc='#e74c3c', ec='white', lw=1.3, zorder=5))
ax2.add_patch(Circle((4.97, 1.93), 0.5, fc=(0.75, 0.22, 0.17, 0.06), ec='#c0392b',
                      lw=0.7, ls='--', zorder=4, alpha=0.5))
ax2.text(4.97, 1.25, 'overlap', fontsize=6.5, color='#c0392b', ha='center', fontweight='bold')

# Similar microstructures
micro3 = generate_micro(80, seed=13, sigma=3.3, threshold=0.02)
micro4 = generate_micro(80, seed=13, sigma=3.3, threshold=-0.02)

ax2.plot([5.6, 6.55], [2.0, 2.55], '--', color='#c0392b', lw=0.5, alpha=0.4, zorder=3)
ax2.plot([5.6, 6.55], [1.9, 1.2], '--', color='#e74c3c', lw=0.5, alpha=0.4, zorder=3)

ax_m3 = fig.add_axes([0.33, 0.26, 0.05, 0.09])
ax_m3.imshow(micro3, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
ax_m3.set_xticks([]); ax_m3.set_yticks([])
for s in ax_m3.spines.values(): s.set_edgecolor('#c0392b'); s.set_linewidth(1.8)

ax_m4 = fig.add_axes([0.33, 0.09, 0.05, 0.09])
ax_m4.imshow(micro4, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
ax_m4.set_xticks([]); ax_m4.set_yticks([])
for s in ax_m4.spines.values(): s.set_edgecolor('#e74c3c'); s.set_linewidth(1.8)

# ≈ between micros
ax2.text(7.5, 1.9, '≈', fontsize=16, color='#c0392b', ha='center', fontweight='bold', zorder=10)

# X label
ax2.text(8.2, 0.35, '✗  Overlapping outcomes → ambiguous inverse', fontsize=7.5, color='#c0392b',
         ha='center', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.25', fc='#ffebee', ec='#ef9a9a', lw=0.5))

# ============================================================
# RIGHT: Score + Method panel
# ============================================================
ax3 = fig.add_axes([0.51, 0.02, 0.48, 0.95])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 22)
ax3.set_aspect('auto')
ax3.axis('off')

bg3 = FancyBboxPatch((0.15, 0.15), 9.7, 21.3, boxstyle="round,pad=0.12",
                      fc='white', ec='#d8d8d0', lw=0.7, zorder=0)
ax3.add_patch(bg3)

# Title
ax3.text(5, 21.0, 'Identifiability Score  $e$', fontsize=14, fontweight='bold', color='#222',
         ha='center', family='serif')

# Colorbar (large, prominent)
ax_cb = fig.add_axes([0.71, 0.32, 0.03, 0.46])
cbar_data = np.linspace(0, 1, 256).reshape(-1, 1)
ax_cb.imshow(cbar_data[::-1], aspect='auto', cmap='viridis', extent=[0, 1, 0, 1])
ax_cb.set_xticks([])
ax_cb.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_cb.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax_cb.yaxis.tick_right()
ax_cb.tick_params(axis='y', length=3, pad=3)

# Annotations on colorbar
ax3.text(2.8, 19.2, 'navigable', fontsize=8, color='#1a9850', fontweight='bold', ha='right')
ax3.text(2.8, 7.8, 'ambiguous', fontsize=8, color='#c0392b', fontweight='bold', ha='right')

# Divider
ax3.plot([0.8, 9.2], [6.8, 6.8], '-', color='#e0e0d8', lw=0.8)

# Method section
ax3.text(5, 6.0, 'LOO Explained Variation', fontsize=11, fontweight='bold', color='#333',
         ha='center', family='serif')
ax3.text(5, 5.2, 'For each point in control space:', fontsize=8.5, color='#555', ha='center')

steps = ['Build outcome neighborhood', 'Fit local ridge inverse', 'Score via LOO residuals']
for idx, text in enumerate(steps):
    yy = 4.3 - idx * 0.9
    ax3.add_patch(Circle((2.2, yy), 0.3, fc=(54/255, 92/255, 141/255, 0.12), ec='none', zorder=3))
    ax3.text(2.2, yy, str(idx+1), fontsize=8.5, color='#365c8d', ha='center', va='center', fontweight='bold')
    ax3.text(2.9, yy, text, fontsize=8.5, color='#444', ha='left', va='center')

# Arrow to formula
ax3.annotate('', xy=(5, 1.65), xytext=(5, 2.15),
            arrowprops=dict(arrowstyle='->', color='#365c8d', lw=1.2))

# Formula box
formula_box = FancyBboxPatch((1.5, 0.7), 7.0, 0.85, boxstyle="round,pad=0.12",
                              fc='#f0f4f8', ec='#365c8d', lw=0.8, zorder=2)
ax3.add_patch(formula_box)
ax3.text(5, 1.2, r'$e = 1 - \|R\|_F^2 \,/\, \|X\|_F^2$', fontsize=11, color='#333',
         ha='center', va='center', family='serif')

# Subtle flow arrows from left panels to right panel
for yy in [0.72, 0.26]:
    fig.patches.append(FancyArrowPatch(
        (0.545, yy), (0.575, yy), transform=fig.transFigure,
        arrowstyle='->', color='#ccc', lw=0.8, mutation_scale=8, zorder=0))

plt.savefig('graphical_abstract_v4.png', dpi=250, bbox_inches='tight',
            facecolor='#fafaf7', edgecolor='none')
plt.savefig('graphical_abstract_v4.pdf', bbox_inches='tight',
            facecolor='#fafaf7', edgecolor='none')
print("Done!")