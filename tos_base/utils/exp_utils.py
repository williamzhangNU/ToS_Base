from typing import List, Dict, Tuple, Optional
import random, io, os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import imageio

from ..core.room import Room
from ..core.object import Agent
from ..actions.actions import ActionSequence
from ..managers.exploration_manager import ExplorationManager
from ..managers.spatial_solver import SpatialSolver
from .room_utils import RoomPlotter


def _full_domain_size(grid_size: int) -> int:
    g = int(grid_size)
    return (2 * g + 1) * (2 * g + 1)


def _gaussian_kernel(sigma: float = 0.5) -> np.ndarray:
    s = max(1, int(np.ceil(2.2 * sigma)))
    ax = np.arange(-s, s + 1)
    xx, yy = np.meshgrid(ax, ax)
    ker = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    ker /= ker.sum() if ker.sum() > 0 else 1.0
    return ker.astype(np.float32)


def _accumulate_gaussians(grid: np.ndarray, positions: List[Tuple[int, int]], grid_size: int, sigma: float = 1.0) -> np.ndarray:
    ker = _gaussian_kernel(sigma); ks = ker.shape[0]; r = ks // 2; g = int(grid_size)
    for (x, y) in positions:
        cx, cy = int(x) + g, int(y) + g
        x0, x1 = max(0, cx - r), min(grid.shape[1], cx + r + 1)
        y0, y1 = max(0, cy - r), min(grid.shape[0], cy + r + 1)
        kx0, kx1 = (0 if cx - r >= 0 else r - cx), (ks if cx + r + 1 <= grid.shape[1] else r + (grid.shape[1] - cx))
        ky0, ky1 = (0 if cy - r >= 0 else r - cy), (ks if cy + r + 1 <= grid.shape[0] else r + (grid.shape[0] - cy))
        grid[y0:y1, x0:x1] += ker[ky0:ky1, kx0:kx1]
    return grid


def _accumulate_gaussians_continuous(positions: List[Tuple[int, int]], XX: np.ndarray, YY: np.ndarray, sigma: float) -> np.ndarray:
    heat = np.zeros_like(XX, dtype=np.float32)
    if sigma <= 0: sigma = 1e-6
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for (x, y) in positions:
        dx, dy = (XX - float(x)), (YY - float(y))
        heat += np.exp(-(dx * dx + dy * dy) * inv2s2).astype(np.float32)
    return heat


_DOPAMINE_HEX = [
    '#FF6B6B', '#FFD166', '#06D6A0', '#118AB2', '#9B5DE5', '#F15BB5',
    '#00F5D4', '#F4A261', '#3A86FF', '#8338EC', '#FF006E', '#8AC926',
    '#2EC4B6', '#FFBE0B', '#EF476F', '#4ECDC4'
]

def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def _rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = [int(max(0, min(1, c)) * 255) for c in rgb]
    return '#%02x%02x%02x' % (r, g, b)

def _pick_icon(name: str, files: List[str]) -> Optional[str]:
    if not files: return None
    lname = name.lower().replace(' ', '_')
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0].lower()
        if base == lname: return f
    if 'door' in lname:
        for f in files:
            base = os.path.splitext(os.path.basename(f))[0].lower()
            if base == 'door': return f
    return random.choice(files)

def _trim_transparent(icon: np.ndarray, thr: int = 10) -> np.ndarray:
    if icon.ndim == 3 and icon.shape[2] == 4:
        a = icon[..., 3] > thr
        if a.any():
            ys, xs = np.where(a)
            return icon[ys.min():ys.max()+1, xs.min():xs.max()+1]
    return icon

def _read_icon(path: Optional[str]) -> Optional[np.ndarray]:
    return _trim_transparent(imageio.v2.imread(path))

def _dominant_rgb(icon: np.ndarray) -> Tuple[float, float, float]:
    arr = icon.astype(np.float32) / 255.0
    if arr.ndim < 3: arr = np.dstack([arr, arr, arr, np.ones_like(arr)])
    if arr.shape[2] == 3:
        alpha = np.ones(arr.shape[:2], dtype=np.float32); rgb = arr[..., :3]
    else:
        alpha = arr[..., 3]; rgb = arr[..., :3]
    mask = alpha > 0.1
    if not mask.any(): return tuple(np.clip(rgb.mean(axis=(0,1)), 0, 1))
    rgb = rgb[mask]
    q = np.clip((rgb * 255).astype(np.uint8) >> 3, 0, 31)
    uq, counts = np.unique(q, axis=0, return_counts=True)
    deq = (uq.astype(np.float32) + 0.5) / 32.0
    lum = (0.2126 * deq[:,0] + 0.7152 * deq[:,1] + 0.0722 * deq[:,2])
    mid = (lum >= 0.20) & (lum <= 0.90)
    dom = deq[np.argmax(counts * mid if mid.any() else counts)]
    L = (0.2126*dom[0] + 0.7152*dom[1] + 0.0722*dom[2])
    if   L < 0.18: dom = dom * 0.6 + 0.4
    elif L > 0.92: dom = dom * 0.8
    return tuple(np.clip(dom, 0, 1))

def _add_icon(ax, icon: np.ndarray, x: float, y: float, zoom: float = 0.12):
    if icon is None: return
    ax.add_artist(AnnotationBbox(OffsetImage(icon, zoom=zoom), (x, y), frameon=False, zorder=50))  # on top

def _zoom_from_icon_dim(ax, dim: int, s_points2: float = 70.0) -> float:
    dpi = float(ax.figure.get_dpi())
    area_px = float(s_points2) * (dpi / 72.0) ** 2
    diam_px = float(np.sqrt(max(1e-6, 4.0 * area_px / np.pi)))
    return max(0.02, min(0.6, 0.95 * diam_px / float(max(1, int(dim)))))

def _assign_identity_colors(names: List[str]) -> Dict[str, str]:
    colors, pal = {}, list(_DOPAMINE_HEX); random.shuffle(pal)
    for i, n in enumerate(sorted(names)):
        if i < len(pal): colors[n] = pal[i]
        else:
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(i / max(1, len(names)), 0.65, 0.95)
            colors[n] = '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))
    return colors


# ---------- NEW: HDR thresholds and contour drawer ----------
def _hdr_thresholds(pdf: np.ndarray, masses=(0.5, 0.8)) -> List[float]:
    """Return iso-density thresholds for given cumulative masses."""
    flat = pdf.ravel()
    if flat.sum() <= 0: return [1.0 for _ in masses]
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    total = csum[-1]
    thrs = []
    for m in masses:
        idx = np.searchsorted(csum, m * total, side='left')
        idx = int(np.clip(idx, 0, len(flat)-1))
        thrs.append(float(flat[order[idx]]))
    return thrs

def _draw_hdr_contours(ax, pdf: np.ndarray, extent: Tuple[float,float,float,float],
                       color: str, is_single: bool=False, center: Optional[Tuple[float,float]]=None):
    """Draw HDR 50% solid and 80% dashed contours. For singletons, ensure visibility."""
    if pdf.sum() <= 0: return
    lv50, lv80 = _hdr_thresholds(pdf, (0.5, 0.8))
    # Draw contours in data coords using same extent/origin as imshow
    cs50 = ax.contour(pdf, levels=[lv50], colors=[color], linewidths=1.8, linestyles='-', origin='lower', extent=extent, zorder=35)
    cs80 = ax.contour(pdf, levels=[lv80], colors=[color], linewidths=1.4, linestyles='--', origin='lower', extent=extent, zorder=35)

    # Fallback for degenerate singletons: tiny smooth rings so lines remain visible
    def _empty(qcs): 
        return (not getattr(qcs, 'allsegs', None)) or (len(qcs.allsegs[0]) == 0)
    if is_single and ( _empty(cs50) or _empty(cs80) ):
        if center is None: return
        x0, y0 = center
        # minimal radii (visible but compact)
        r50, r80 = 0.25, 0.45
        e50 = patches.Circle((x0, y0), r50, fill=False, edgecolor=color, linestyle='-', linewidth=1.8, zorder=36)
        e80 = patches.Circle((x0, y0), r80, fill=False, edgecolor=color, linestyle='--', linewidth=1.4, zorder=36)
        ax.add_patch(e50); ax.add_patch(e80)


class ReplayHelper:
    """Replay utilities for heatmaps and trajectory GIFs."""

    def __init__(self, room: 'Room', agent: 'Agent', grid_size: Optional[int] = None):
        self.room = room.copy(); self.agent = agent.copy()
        g = (max(self.room.mask.shape) if getattr(self.room, 'mask', None) is not None else 10)
        self.grid_size = int(g if grid_size is None else grid_size)

    def plot_observation_heatmaps(self, action_results: List, max_positions: int = 500, sigma: float = 0.8,
                                  out_dir: Optional[str] = None, fps: int = 2, axes: bool = True,
                                  use_icons: bool = False, icons_dir: Optional[str] = None,
                                  use_icon_colors: bool = False, bg: float = 0.96, small_bg: float = 0.08) -> List[np.ndarray]:
        """
        Main changes:
        - Replace covariance ellipses with HDR iso-contours (50% solid, 80% dashed).
        - For singletons, guarantee contour visibility with tiny fallback rings.
        - Shrink singletons' colored area via tighter sigma + smooth radial cap (<1 grid).
        """
        mgr = ExplorationManager(self.room, self.agent)
        names = [o.name for o in self.room.all_objects] + ['initial_pos']
        solver = SpatialSolver(names, grid_size=self.grid_size)
        solver.set_initial_position('initial_pos', (0, 0))

        full_size = _full_domain_size(self.grid_size)
        obj_names = [o.name for o in self.room.all_objects if o.name != 'initial_pos']
        _color_by_name = _assign_identity_colors(obj_names)

        idir = icons_dir or os.path.join(os.path.dirname(__file__), 'icons')
        icon_files = sorted(glob.glob(os.path.join(idir, '*.png')))
        agent_icon = _read_icon(os.path.join(idir, 'agent.png')) if icon_files else None
        _icon_by_name: Dict[str, Optional[np.ndarray]] = {}
        _icon_dim_by_name: Dict[str, int] = {}
        if use_icons and icon_files:
            for n in obj_names:
                ic = _read_icon(_pick_icon(n, icon_files)); _icon_by_name[n] = ic
                if ic is not None: _icon_dim_by_name[n] = max(int(ic.shape[0]), int(ic.shape[1]))
        if use_icon_colors and icon_files:
            for n in obj_names:
                ic = _icon_by_name.get(n) if use_icons else _read_icon(_pick_icon(n, icon_files))
                if ic is not None: _color_by_name[n] = _rgb_to_hex(_dominant_rgb(ic))

        def _render(domains: Dict[str, set], step_title: str, bounds: Tuple[float, float, float, float]) -> np.ndarray:
            g = int(self.grid_size); H = W = 2 * g + 1
            fig = plt.figure(figsize=(7.0, 7.6), dpi=130); fig.patch.set_facecolor((bg, bg, bg))
            LEGEND_ICON_S, STRIP_ICON_S = 46.0, 36.0

            # ---- MAIN PLOT KNOBS ----
            SIGMA_SINGLE = 0.08           # tighter singletons
            T_MULTI, T_SINGLE = 0.30, 0.93
            SHARP_MULTI, SHARP_SINGLE = 0.55, 2.20
            COUNT_REF, COUNT_GAMMA = 6.0, 0.85
            W_MIN, W_MAX = 0.90, 2.60
            R_SINGLE_MAX, R_EDGE = 0.80, 0.08  # <1 grid; soft radial cap only for singletons

            n_legend = len(obj_names) if (use_icons and len(obj_names) > 0) else 0
            rows = int(np.ceil(n_legend / 8.0)) if n_legend else 0
            leg_h = 0.05 * rows; top = 0.93 - leg_h if rows else 0.96
            ax = fig.add_axes([0.08, 0.26, 0.84, max(0.2, top - 0.26)])
            ax.set_title(step_title)
            if axes: ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.set_aspect('equal'); ax.grid(axes, linestyle=':', linewidth=0.4, alpha=0.35)

            x_min, x_max, y_min, y_max = bounds
            ax.set_xlim(x_min - 0.5, x_max + 0.5); ax.set_ylim(y_min - 0.5, y_max + 0.5)
            xs = np.linspace(-g, g, W); ys = np.linspace(-g, g, H)
            XX, YY = np.meshgrid(xs, ys)
            extent = (-g, g, -g, g)

            canvas = np.zeros((H, W, 3), dtype=np.float32)
            alpha_map = np.zeros((H, W), dtype=np.float32)

            # collect HDR to draw after image
            hdr_queue: List[Tuple[np.ndarray, str, bool, Optional[Tuple[float,float]]]] = []

            for obj in self.room.all_objects:
                name = obj.name
                if name == 'initial_pos': continue
                dom = list(domains.get(name, set()))
                if not dom or len(dom) >= full_size: continue
                if len(dom) > max_positions: dom = random.sample(dom, max_positions)
                n = max(1, len(dom))

                sigma_local = (SIGMA_SINGLE if n == 1 else sigma)
                heat_raw = _accumulate_gaussians_continuous(dom, XX, YY, sigma_local)
                s = float(heat_raw.sum())
                if s <= 0: continue
                heat_prob = heat_raw / s

                # normalized shape -> threshold -> sharpen
                h = heat_prob / (float(heat_prob.max()) + 1e-12)
                t = T_SINGLE if n == 1 else T_MULTI
                h = np.clip((h - t) / (1.0 - t), 0.0, 1.0)
                h = np.power(h, (SHARP_SINGLE if n == 1 else SHARP_MULTI))

                # count-weighted brightness
                w = (COUNT_REF / n) ** COUNT_GAMMA
                if n == 1: w *= 1.35
                w = float(np.clip(w, W_MIN, W_MAX))
                boost = np.clip(h * w, 0.0, 1.0)

                # extra shrink ONLY for singleton colored area: smooth radial cap (<1 grid)
                if n == 1 and len(dom) == 1:
                    (cx, cy) = dom[0]
                    rr = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
                    # smooth step: ~1 inside, decays across edge
                    cap = 1.0 / (1.0 + np.exp((rr - R_SINGLE_MAX) / max(1e-6, R_EDGE)))
                    boost *= cap

                rgb = np.array(_hex_to_rgb(_color_by_name.get(name, '#E69F00')), dtype=np.float32)
                comp = boost[..., None] * rgb
                canvas = np.maximum(canvas, comp)
                alpha_map = np.maximum(alpha_map, boost)

                # Queue HDR contours (draw later, after imshow)
                ctr_center = dom[0] if (n == 1 and len(dom) == 1) else None
                hdr_queue.append((heat_prob, _color_by_name.get(name, '#E69F00'), (n == 1), ctr_center))

                # draw single center (icon/dot)
                if n == 1:
                    (x, y) = dom[0]
                    if use_icons and _icon_by_name.get(name) is not None:
                        dim = _icon_dim_by_name.get(name, max(_icon_by_name[name].shape[0], _icon_by_name[name].shape[1]))
                        _add_icon(ax, _icon_by_name[name], x, y, zoom=_zoom_from_icon_dim(ax, dim, s_points2=56.0))
                    else:
                        ax.scatter([x], [y], s=70, c=_color_by_name.get(name, '#E69F00'),
                                   marker='o', edgecolors='k', linewidths=0.5, zorder=45)

            # paint heat
            alpha = np.clip(alpha_map, 0, 1)
            final = (1.0 - alpha)[..., None] * np.array([bg, bg, bg], dtype=np.float32) + alpha[..., None] * canvas
            ax.imshow(final, origin='lower', extent=extent, interpolation='bicubic', zorder=0)

            # draw HDR contours (50% solid, 80% dashed)
            for pdf, color, is_single, ctr in hdr_queue:
                _draw_hdr_contours(ax, pdf, extent, color, is_single=is_single, center=ctr)

            if not axes: ax.set_xticks([]); ax.set_yticks([])

            # initial agent
            if use_icons and agent_icon is not None:
                dim0 = max(int(agent_icon.shape[0]), int(agent_icon.shape[1]))
                _add_icon(ax, agent_icon, 0, 0, zoom=_zoom_from_icon_dim(ax, dim0, s_points2=72.0))
            else:
                ax.scatter([0], [0], s=90, c='#000000', marker='*', linewidths=0.6, zorder=48)

            # legend
            if use_icons and icon_files:
                rows = int(np.ceil(len(obj_names) / 8.0)) or 1
                cols = int(np.ceil(len(obj_names) / rows))
                y0 = (0.93 - 0.05 * rows) + 0.01
                lax = fig.add_axes([0.08, y0, 0.84, 0.05 * rows]); lax.axis('off')
                for i, n in enumerate(obj_names):
                    row, col = divmod(i, cols)
                    x = 0.02 + col * (0.96 / max(1, cols)); y = 1.0 - (row + 0.5) / rows
                    color = _color_by_name.get(n, '#E69F00')
                    lax.add_patch(patches.Rectangle((x, y - 0.035), 0.03, 0.07, transform=lax.transAxes,
                                                    color=color, ec='k', lw=0.3))
                    ic = _icon_by_name.get(n)
                    if ic is not None:
                        dim = _icon_dim_by_name.get(n, max(ic.shape[0], ic.shape[1]))
                        ab = AnnotationBbox(OffsetImage(ic, zoom=_zoom_from_icon_dim(lax, dim, s_points2=LEGEND_ICON_S)),
                                            (x + 0.05, y), frameon=False, xycoords='axes fraction')
                        lax.add_artist(ab)
            else:
                handles = [patches.Patch(color=_color_by_name.get(o.name, '#E69F00'), label=o.name)
                           for o in self.room.all_objects if o.name != 'initial_pos']
                by_label = {h.get_label(): h for h in handles}
                ax.legend(handles=list(by_label.values()), loc='upper right', fontsize=8, framealpha=0.85)

            # bottom mini-strips (also use HDR contours)
            names_local = sorted({o.name for o in self.room.all_objects} | {n for n in domains.keys()})
            names_local = [n for n in names_local if n != 'initial_pos']; nobj = len(names_local)
            if nobj > 0:
                left0, width, gap, height, bottom = 0.08, 0.84 / max(nobj, 1), 0.01, 0.16, 0.05
                width = min(width, 0.15)
                for i, name in enumerate(names_local):
                    lx = 0.08 + i * (width + gap)
                    if lx + width > 0.92: break
                    ax_s = fig.add_axes([lx, bottom, width, height]); ax_s.set_xticks([]); ax_s.set_yticks([])
                    ax_s.imshow(np.full((H, W), small_bg), origin='lower', extent=extent, cmap='gray', vmin=0, vmax=1)
                    ax_s.set_xlim(x_min - 0.5, x_max + 0.5); ax_s.set_ylim(y_min - 0.5, y_max + 0.5)

                    dom = list(domains.get(name, set()))
                    if not dom or len(dom) >= full_size:
                        ic = _icon_by_name.get(name) if (use_icons and icon_files) else None
                        if ic is not None:
                            dim = _icon_dim_by_name.get(name, max(ic.shape[0], ic.shape[1]))
                            ab = AnnotationBbox(OffsetImage(ic, zoom=_zoom_from_icon_dim(ax_s, dim, s_points2=STRIP_ICON_S)),
                                                (0.5, 1.08), frameon=False, xycoords='axes fraction')
                            ab.set_clip_on(False); ax_s.add_artist(ab)
                        else:
                            ax_s.set_title(name, fontsize=7, color=_color_by_name.get(name, '#444444'), pad=1)
                        cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
                        circ = patches.Ellipse((cx, cy), 0.08*(x_max-x_min), 0.08*(y_max-y_min),
                                               fill=False, edgecolor='#BBBBBB', linestyle='--', linewidth=0.8, zorder=10)
                        ax_s.add_patch(circ); continue

                    if len(dom) > max_positions: dom = random.sample(dom, max_positions)
                    sigma_local = (SIGMA_SINGLE if len(dom) == 1 else sigma)
                    heat = _accumulate_gaussians(np.zeros((H, W), dtype=np.float32), dom, self.grid_size, sigma=sigma_local)
                    ssum = float(heat.sum())
                    if ssum <= 0: 
                        continue
                    prob = heat / ssum  # normalize for HDR
                    vmax = (prob.max() + 1e-9)
                    h = np.clip(prob / vmax, 0, 1)
                    t = T_SINGLE if len(dom) == 1 else T_MULTI
                    h = np.clip((h - t) / (1.0 - t), 0, 1)
                    rgb = np.array(_hex_to_rgb(_color_by_name.get(name, '#E69F00')), dtype=np.float32)
                    bg_rgb = np.array([small_bg, small_bg, small_bg], dtype=np.float32)
                    h = np.power(h, 0.55)
                    final_small = bg_rgb + (rgb - bg_rgb) * np.clip(h * 1.2, 0, 1)[..., None]
                    ax_s.imshow(final_small, origin='lower', extent=extent, interpolation='bicubic', zorder=0)

                    # mini HDR contours + singleton center
                    ctr_center = dom[0] if len(dom) == 1 else None
                    _draw_hdr_contours(ax_s, prob, extent, _color_by_name.get(name, '#E69F00'),
                                       is_single=(len(dom) == 1), center=ctr_center)
                    if len(dom) == 1:
                        (x, y) = dom[0]
                        ax_s.scatter([x], [y], s=40, c=_color_by_name.get(name, '#E69F00'),
                                     marker='o', edgecolors='k', linewidths=0.4, zorder=45)
                    ic = _icon_by_name.get(name) if (use_icons and icon_files) else None
                    if ic is not None:
                        dim = _icon_dim_by_name.get(name, max(ic.shape[0], ic.shape[1]))
                        ab = AnnotationBbox(OffsetImage(ic, zoom=_zoom_from_icon_dim(ax_s, dim, s_points2=STRIP_ICON_S)),
                                            (0.5, 1.08), frameon=False, xycoords='axes fraction')
                        ab.set_clip_on(False); ax_s.add_artist(ab)
                    else:
                        ax_s.set_title(name, fontsize=7, color=_color_by_name.get(name, '#E69F00'), pad=1)

            buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=fig.dpi); plt.close(fig); buf.seek(0)
            return imageio.v2.imread(buf)

        # collect observe snapshots
        snapshots: List[Dict[str, set]] = []
        for res in action_results:
            if res.action_type in ('observe', 'observe_approx'):
                triples = res.data.get('relation_triples', []) if hasattr(res, 'data') else []
                if triples: solver.add_observation(triples)
                snapshots.append(solver.get_possible_positions())

        # shared bounds
        g = int(self.grid_size); full_size = _full_domain_size(self.grid_size)
        if snapshots:
            coords = []
            for d in snapshots:
                for n, dom in d.items():
                    if n == 'initial_pos' or (not dom) or len(dom) >= full_size: continue
                    coords.extend(list(dom))
            if coords:
                xs_all = [c[0] for c in coords]; ys_all = [c[1] for c in coords]
                x_min, x_max = min(xs_all) - 1, max(xs_all) + 1
                y_min, y_max = min(ys_all) - 1, max(ys_all) + 1
            else:
                x_min, x_max, y_min, y_max = -g, g, -g, g
        else:
            x_min, x_max, y_min, y_max = -g, g, -g, g
        bounds = (x_min, x_max, y_min, y_max)

        # render frames
        frames: List[np.ndarray] = []
        for i, doms in enumerate(snapshots):
            frames.append(_render(doms, step_title=f"Observe @ step {i}", bounds=bounds))

        # save (optional)
        if out_dir and frames:
            for i, img in enumerate(frames):
                imageio.v2.imwrite(f"{out_dir.rstrip('/')}/heatmap_{i:03d}.png", img)
            imageio.mimsave(f"{out_dir.rstrip('/')}/heatmaps.gif", frames, duration=(1.0 / max(1, int(fps))))
        return frames

    def animate_agent_trajectory(self, action_results: List, out_path: Optional[str] = None, fps: int = 2) -> str:
        mgr = ExplorationManager(self.room, self.agent); frames: List[np.ndarray] = []
        for res in action_results:
            act = ActionSequence._parse_single_action(res.action_command) if res.action_command else None
            if act is not None: _ = mgr.execute_success_action(act)
            frame = RoomPlotter.plot_to_image(mgr.exploration_room, mgr.agent, observe=(res.action_type in ('observe', 'observe_approx')), dpi=120)
            frames.append(frame)
        out_file = out_path or 'trajectory.gif'
        imageio.mimsave(out_file, frames, duration=(1.0 / max(1, int(fps))))
        return out_file

    @staticmethod
    def flatten_turns(turns: List) -> List:
        seq = [];  [seq.extend(list(getattr(t, 'actions', []) or [])) for t in turns];  return seq


if __name__ == "__main__":
    from ..managers.agent_proxy import OracleAgentProxy, InquisitorAgentProxy, AnalystAgentProxy
    from ..utils.room_utils import RoomGenerator
    from ..core.constant import ObjectInfo
    candidate_objects = [
        ObjectInfo(name='basket', has_orientation=True),
        ObjectInfo(name='cabinet', has_orientation=True),
        ObjectInfo(name='chair', has_orientation=True),
        ObjectInfo(name='laptop', has_orientation=True),
        ObjectInfo(name='keyboard', has_orientation=True),
        ObjectInfo(name='office-chair', has_orientation=True),
        ObjectInfo(name='printer', has_orientation=True),
        ObjectInfo(name='backpack', has_orientation=True),
        ObjectInfo(name='table-lamp', has_orientation=True),
        ObjectInfo(name='table', has_orientation=True),
        ObjectInfo(name='tv', has_orientation=True),
        ObjectInfo(name='bookshelf', has_orientation=True),
        ObjectInfo(name='floor-lamp', has_orientation=True),
    ]
    # room, agent = RoomGenerator.generate_room(
    #     room_size=[15, 15], n_objects=4, np_random=np.random.default_rng(2),
    #     level=1, main=6,
    #     candidate_objects=[ObjectInfo(name='basket', has_orientation=True),
    #                        ObjectInfo(name='chair', has_orientation=True),
    #                        ObjectInfo(name='kettle', has_orientation=True),
    #                        ObjectInfo(name='table', has_orientation=True)]
    # )
    room, agent = RoomGenerator.generate_room(
        room_size=[20, 20], n_objects=8, np_random=np.random.default_rng(0),
        level=1, main=8, candidate_objects=candidate_objects
    )
    print(room, agent)
    # proxy = AnalystAgentProxy(room, agent, delegate='observer_analyst', observer_delegate='strategist')
    proxy = InquisitorAgentProxy(room, agent)
    proxy.run()
    action_results = ReplayHelper.flatten_turns(proxy.turns)
    replay = ReplayHelper(room, agent)
    replay.plot_observation_heatmaps(action_results, out_dir='heatmaps', fps=1, axes=False, use_icons=True, use_icon_colors=True)
    # replay.animate_agent_trajectory(action_results, out_path='trajectory.gif', fps=1)