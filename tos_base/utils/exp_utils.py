from typing import List, Dict, Tuple, Optional
import random
import io
import time
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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
    s = max(1, int(3 * sigma))
    ax = np.arange(-s, s + 1)
    xx, yy = np.meshgrid(ax, ax)
    ker = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    ker /= ker.sum() if ker.sum() > 0 else 1.0
    return ker


def _accumulate_gaussians(grid: np.ndarray, positions: List[Tuple[int, int]], grid_size: int, sigma: float = 1.0) -> np.ndarray:
    ker = _gaussian_kernel(sigma)
    ks = ker.shape[0]
    r = ks // 2
    g = int(grid_size)
    for (x, y) in positions:
        cx, cy = int(x) + g, int(y) + g
        x0, x1 = max(0, cx - r), min(grid.shape[1], cx + r + 1)
        y0, y1 = max(0, cy - r), min(grid.shape[0], cy + r + 1)
        kx0, kx1 = (0 if cx - r >= 0 else r - cx), (ks if cx + r + 1 <= grid.shape[1] else r + (grid.shape[1] - cx))
        ky0, ky1 = (0 if cy - r >= 0 else r - cy), (ks if cy + r + 1 <= grid.shape[0] else r + (grid.shape[0] - cy))
        grid[y0:y1, x0:x1] += ker[ky0:ky1, kx0:kx1]
    return grid


def _choose_colormaps(names: List[str]) -> Dict[str, str]:
    palettes = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys", "YlOrBr", "YlGn", "PuRd", "BuPu", "GnBu", "BuGn"]
    cmap = {}
    for i, n in enumerate(names):
        cmap[n] = palettes[i % len(palettes)]
    return cmap


# Color helpers: bright "dopamine" identity hues
_BASE_HEX = [
    '#FF6B6B', '#F7B801', '#06D6A0', '#118AB2', '#9B5DE5', '#F15BB5', '#00F5D4', '#FFD166',
    '#EF476F', '#4ECDC4', '#C7F464', '#FF8C42', '#3A86FF', '#8338EC', '#FB5607', '#FF006E',
    '#2EC4B6', '#FFBE0B', '#8AC926', '#1982C4'
]


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = [int(max(0, min(1, c)) * 255) for c in rgb]
    return '#%02x%02x%02x' % (r, g, b)


def _icon_files(icons_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(icons_dir, '*.png')))
    return files


def _pick_icon(name: str, files: List[str]) -> Optional[str]:
    if not files:
        return None
    lname = name.lower().replace(' ', '_')
    # exact match first
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0].lower()
        if base == lname:
            return f
    # gate -> door.png if available
    if 'gate' in lname:
        for f in files:
            base = os.path.splitext(os.path.basename(f))[0].lower()
            if base == 'door':
                return f
    # otherwise random pick
    return random.choice(files)


def _read_icon(path: Optional[str]) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    try:
        return imageio.v2.imread(path)
    except Exception:
        return None


def _dominant_rgb(icon: np.ndarray) -> Tuple[float, float, float]:
    arr = icon.astype(np.float32) / 255.0
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[..., 3]
        m = alpha > 0.1
        if m.any():
            rgb = arr[..., :3][m]
            return tuple(np.clip(rgb.mean(axis=0), 0, 1))
        return tuple(arr[..., :3].mean(axis=(0, 1)))
    return tuple(np.clip(arr[..., :3].mean(axis=(0, 1)), 0, 1))


def _add_icon(ax, icon: np.ndarray, x: float, y: float, zoom: float = 0.12):
    if icon is None:
        return
    ab = AnnotationBbox(OffsetImage(icon, zoom=zoom), (x, y), frameon=False, zorder=10)
    ax.add_artist(ab)


def _add_icon_frac(ax, icon: np.ndarray, x: float, y: float, zoom: float = 0.06):
    if icon is None:
        return
    ab = AnnotationBbox(OffsetImage(icon, zoom=zoom), (x, y), frameon=False, xycoords='axes fraction', zorder=10)
    ab.set_clip_on(False)
    ax.add_artist(ab)


def _icon_zoom_for_scatter(ax, icon: Optional[np.ndarray], s_points2: float) -> float:
    if icon is None:
        return 0.06
    dpi = float(ax.figure.get_dpi())
    area_px = float(s_points2) * (dpi / 72.0) ** 2
    diam_px = float(np.sqrt(max(1e-6, 4.0 * area_px / np.pi)))
    h, w = icon.shape[0], icon.shape[1]
    return max(0.02, min(0.6, 0.95 * diam_px / float(max(h, w))))


def _zoom_from_icon_dim(ax, dim: int, s_points2: float = 70.0) -> float:
    dpi = float(ax.figure.get_dpi())
    area_px = float(s_points2) * (dpi / 72.0) ** 2
    diam_px = float(np.sqrt(max(1e-6, 4.0 * area_px / np.pi)))
    return max(0.02, min(0.6, 0.95 * diam_px / float(max(1, int(dim)))))


def _credible_thresholds(p: np.ndarray, coverages=(0.8, 0.5)) -> List[float]:
    ps = np.sort(p.ravel())[::-1]
    if ps.size == 0:
        return [0.0 for _ in coverages]
    csum = np.cumsum(ps)
    s = csum[-1] if csum[-1] > 0 else 1.0
    csum /= s
    levels = []
    for cov in coverages:
        idx = np.searchsorted(csum, cov)
        idx = min(max(idx, 0), len(ps) - 1)
        levels.append(ps[idx])
    return levels


def _assign_identity_colors(names: List[str]) -> Dict[str, str]:
    """Assign a distinct, stable color to each name."""
    colors = {}
    for i, n in enumerate(sorted(names)):
        if i < len(_BASE_HEX):
            colors[n] = _BASE_HEX[i]
        else:
            # fallback: evenly spread hues
            hsv = (i / max(1, len(names)), 0.6, 0.9)
            import colorsys
            rgb = colorsys.hsv_to_rgb(*hsv)
            colors[n] = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    return colors


class ReplayHelper:
    """Simple replay utilities for heatmaps and trajectory GIFs."""

    def __init__(self, room: Room, agent: Agent, grid_size: Optional[int] = None):
        self.room = room.copy()
        self.agent = agent.copy()
        g = (max(self.room.mask.shape) if getattr(self.room, 'mask', None) is not None else 10)
        self.grid_size = int(g if grid_size is None else grid_size)

    # ---- Heatmaps from Observe steps ----
    def plot_observation_heatmaps(self, action_results: List, max_positions: int = 500, sigma: float = 1,
                                  out_dir: Optional[str] = None, fps: int = 2, axes: bool = True,
                                  use_icons: bool = False, icons_dir: Optional[str] = None,
                                  use_icon_colors: bool = False, bg: float = 0.96, small_bg: float = 0.08) -> List[np.ndarray]:
        """Execute actions and, at each observe, draw multi-object heatmaps of possible positions.

        - Only plot objects whose domain is initialized (domain size < full grid).
        - For very large domains, randomly sample up to max_positions.
        - Each object's domain becomes a smooth heat via local Gaussians (sigma).
        - Save a GIF at given fps if out_dir is provided. Toggle axes/grid via axes.
        - Icons: replace legend labels, singletons and small titles; optional icon-derived colors.
        - Backgrounds: main uses bg; small multiples use small_bg.
        """
        mgr = ExplorationManager(self.room, self.agent)
        names = [o.name for o in self.room.all_objects] + ['initial_pos']
        solver = SpatialSolver(names, grid_size=self.grid_size)
        solver.set_initial_position('initial_pos', (0, 0))

        frames: List[np.ndarray] = []
        full_size = _full_domain_size(self.grid_size)
        # unique, stable colors for each object name
        obj_names = [o.name for o in self.room.all_objects if o.name != 'initial_pos']
        _color_by_name = _assign_identity_colors(obj_names)

        # icons + optional icon colors
        idir = icons_dir or os.path.join(os.path.dirname(__file__), 'icons')
        icon_files = _icon_files(idir)
        agent_icon = _read_icon(os.path.join(idir, 'agent.png')) if icon_files else None
        _icon_by_name: Dict[str, Optional[np.ndarray]] = {}
        _icon_dim_by_name: Dict[str, int] = {}
        if use_icons and icon_files:
            for n in obj_names:
                _icon_by_name[n] = _read_icon(_pick_icon(n, icon_files))
                if _icon_by_name[n] is not None:
                    h, w = _icon_by_name[n].shape[0], _icon_by_name[n].shape[1]
                    _icon_dim_by_name[n] = max(int(h), int(w))
        if use_icon_colors and icon_files:
            for n in obj_names:
                ic = _icon_by_name.get(n) if use_icons else _read_icon(_pick_icon(n, icon_files))
                if ic is not None:
                    _color_by_name[n] = _rgb_to_hex(_dominant_rgb(ic))

        def _render(domains: Dict[str, set], step_title: str, bounds: Tuple[float, float, float, float]) -> np.ndarray:
            g = int(self.grid_size)
            H = W = 2 * g + 1
            fig = plt.figure(figsize=(7.0, 7.6))
            fig.patch.set_facecolor((bg, bg, bg))
            # dynamic legend space to avoid overlap
            n_legend = len(obj_names) if (use_icons and len(obj_names) > 0) else 0
            rows = int(np.ceil(n_legend / 8.0)) if n_legend else 0
            leg_h = 0.05 * rows
            top = 0.93 - leg_h if rows else 0.96
            ax = fig.add_axes([0.08, 0.26, 0.84, max(0.2, top - 0.26)])
            ax.set_title(step_title)
            if axes:
                ax.set_xlabel('x')
                ax.set_ylabel('y')
            ax.set_aspect('equal')
            ax.grid(axes, linestyle=':', linewidth=0.4, alpha=0.35)

            # coordinate mesh for contours (grid coordinates)
            x_min, x_max, y_min, y_max = bounds
            xs = np.linspace(-g, g, W)
            ys = np.linspace(-g, g, H)
            XX, YY = np.meshgrid(xs, ys)
            ax.set_xlim(x_min - 0.5, x_max + 0.5)
            ax.set_ylim(y_min - 0.5, y_max + 0.5)
            
            # helper: smooth covariance ellipses (50% solid, 80% dashed)
            def _draw_cov_ellipses(_ax, _heat, _color):
                s = float(_heat.sum());
                if s <= 0: return
                mx = float((_heat * XX).sum() / s); my = float((_heat * YY).sum() / s)
                dx, dy = (XX - mx), (YY - my)
                cxx = float((_heat * (dx * dx)).sum() / s)
                cyy = float((_heat * (dy * dy)).sum() / s)
                cxy = float((_heat * (dx * dy)).sum() / s)
                cov = np.array([[cxx, cxy], [cxy, cyy]]) + 1e-6 * np.eye(2)
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
                theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
                def _ellipse(q, ls, lw):
                    w = 2.0 * np.sqrt(max(q * vals[0], 1e-9))
                    h = 2.0 * np.sqrt(max(q * vals[1], 1e-9))
                    e = patches.Ellipse((mx, my), w, h, angle=theta, fill=False, edgecolor=_color, linestyle=ls, linewidth=lw)
                    _ax.add_patch(e)
                _ellipse(1.386, '-', 1.25)   # ~50% chi2_2
                _ellipse(3.219, '--', 1.0)   # ~80% chi2_2

            # main: stronger heat visibility (ICLR style): color-lighten + boosted alpha
            canvas = np.zeros((H, W, 3), dtype=np.float32)
            alpha_map = np.zeros((H, W), dtype=np.float32)
            for obj in self.room.all_objects:
                name = obj.name
                if name == 'initial_pos':
                    continue
                dom = list(domains.get(name, set()))
                if not dom or len(dom) >= full_size:
                    continue
                if len(dom) > max_positions:
                    dom = random.sample(dom, max_positions)
                heat = _accumulate_gaussians(np.zeros((H, W), dtype=np.float32), dom, self.grid_size, sigma=sigma)
                vmax = (heat.max() + 1e-9)
                h = np.clip(heat / vmax, 0, 1)
                boost = np.clip(np.power(h, 0.5) * 1.35, 0, 1)
                rgb = np.array(_hex_to_rgb(_color_by_name.get(name, '#E69F00')), dtype=np.float32)
                comp = boost[..., None] * rgb
                canvas = np.maximum(canvas, comp)  # lighten
                alpha_map = np.maximum(alpha_map, boost)
                # smooth ellipse outlines instead of polygonal contours
                color = _color_by_name.get(name, '#E69F00')
                _draw_cov_ellipses(ax, heat, color)

                if len(dom) == 1:
                    (x, y) = dom[0]
                    if use_icons and _icon_by_name.get(name) is not None:
                        dim = _icon_dim_by_name.get(name)
                        zoom = _zoom_from_icon_dim(ax, dim, s_points2=70.0) if dim is not None else _icon_zoom_for_scatter(ax, _icon_by_name[name], s_points2=70.0)
                        _add_icon(ax, _icon_by_name[name], x, y, zoom=zoom)
                    else:
                        ax.scatter([x], [y], s=70, c=color, marker='o', edgecolors='k', linewidths=0.5, zorder=6)
            # blend onto bg using per-pixel alpha from intensity
            alpha = np.clip(alpha_map, 0, 1)
            final = (1.0 - alpha)[..., None] * np.array([bg, bg, bg], dtype=np.float32) + alpha[..., None] * canvas
            ax.imshow(final, origin='lower', extent=(-g, g, -g, g), interpolation='bilinear')
            if not axes:
                ax.set_xticks([]); ax.set_yticks([])

            # initial agent position marker/icon
            init_x, init_y = 0, 0
            if use_icons and agent_icon is not None:
                h0, w0 = agent_icon.shape[0], agent_icon.shape[1]
                zoom = _zoom_from_icon_dim(ax, max(int(h0), int(w0)), s_points2=90.0)
                _add_icon(ax, agent_icon, init_x, init_y, zoom=zoom)
            else:
                ax.scatter([init_x], [init_y], s=90, c='#000000', marker='*', linewidths=0.6, zorder=8)

            # legend (color patch left + icon right), top rows, wrap when long
            if use_icons and icon_files:
                rows = int(np.ceil(len(obj_names) / 8.0)) or 1
                cols = int(np.ceil(len(obj_names) / rows))
                y0 = top + 0.01
                lax = fig.add_axes([0.08, y0, 0.84, 0.05 * rows]); lax.axis('off')
                for i, n in enumerate(obj_names):
                    row, col = divmod(i, cols)
                    x = 0.02 + col * (0.96 / max(1, cols))
                    y = 1.0 - (row + 0.5) / rows
                    color = _color_by_name.get(n, '#E69F00')
                    rect = patches.Rectangle((x, y - 0.035), 0.03, 0.07, transform=lax.transAxes, color=color, ec='k', lw=0.3)
                    lax.add_patch(rect)
                    ic = _icon_by_name.get(n)
                    if ic is not None:
                        ab = AnnotationBbox(OffsetImage(ic, zoom=0.06), (x + 0.05, y), frameon=False, xycoords='axes fraction')
                        lax.add_artist(ab)
            else:
                handles = [patches.Patch(color=_color_by_name.get(o.name, '#E69F00'), label=o.name) for o in self.room.all_objects if o.name != 'initial_pos']
                by_label = {h.get_label(): h for h in handles}
                ax.legend(handles=list(by_label.values()), loc='upper right', fontsize=8, framealpha=0.85)

            # bottom row of small multiples (use identity colors; include all domain keys e.g., door)
            names = sorted({o.name for o in self.room.all_objects} | {n for n in domains.keys()})
            names = [n for n in names if n != 'initial_pos']
            n = len(names)
            if n > 0:
                left0, width, gap, height, bottom = 0.08, 0.84 / max(n, 1), 0.01, 0.16, 0.05
                width = min(width, 0.15)
                for i, name in enumerate(names):
                    lx = 0.08 + i * (width + gap)
                    if lx + width > 0.92:
                        break
                    ax_s = fig.add_axes([lx, bottom, width, height])
                    ax_s.set_xticks([]); ax_s.set_yticks([])
                    # deep, non-black small plot background
                    ax_s.imshow(np.full((H, W), small_bg), origin='lower', extent=(-g, g, -g, g), cmap='gray', vmin=0, vmax=1)
                    ax_s.set_xlim(x_min - 0.5, x_max + 0.5)
                    ax_s.set_ylim(y_min - 0.5, y_max + 0.5)
                    dom = list(domains.get(name, set()))
                    if not dom or len(dom) >= full_size:
                        if use_icons and icon_files and _icon_by_name.get(name) is not None:
                            ab = AnnotationBbox(OffsetImage(_icon_by_name[name], zoom=0.055), (0.5, 1.08), frameon=False, xycoords='axes fraction')
                            ab.set_clip_on(False)
                            ax_s.add_artist(ab)
                        else:
                            ax_s.set_title(name, fontsize=7, color=_color_by_name.get(name, '#444444'), pad=1)
                        # subtle placeholder circle for not-initialized (centered in view)
                        cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
                        w, h = 0.08 * (x_max - x_min), 0.08 * (y_max - y_min)
                        circ = patches.Ellipse((cx, cy), max(w, 1e-3), max(h, 1e-3), fill=False, edgecolor='#BBBBBB', linestyle='--', linewidth=0.8)
                        ax_s.add_patch(circ)
                        continue
                    if len(dom) > max_positions:
                        dom = random.sample(dom, max_positions)
                    heat = _accumulate_gaussians(np.zeros((H, W), dtype=np.float32), dom, self.grid_size, sigma=sigma)
                    vmax = (heat.max() + 1e-9)
                    h = np.clip(heat / vmax, 0, 1)
                    rgb = np.array(_hex_to_rgb(_color_by_name.get(name, '#E69F00')), dtype=np.float32)
                    bg_rgb = np.array([small_bg, small_bg, small_bg], dtype=np.float32)
                    # emphasize heat on dark bg with gamma and saturation
                    h = np.power(h, 0.55)
                    final_small = bg_rgb + (rgb - bg_rgb) * np.clip(h * 1.2, 0, 1)[..., None]
                    ax_s.imshow(final_small, origin='lower', extent=(-g, g, -g, g), interpolation='bilinear')
                    ax_s.set_xlim(x_min - 0.5, x_max + 0.5)
                    ax_s.set_ylim(y_min - 0.5, y_max + 0.5)
                    color = _color_by_name.get(name, '#E69F00')
                    _draw_cov_ellipses(ax_s, heat, color)
                    if len(dom) == 1:
                        (x, y) = dom[0]
                        ax_s.scatter([x], [y], s=40, c=color, marker='o', edgecolors='k', linewidths=0.4, zorder=6)
                    if use_icons and icon_files and _icon_by_name.get(name) is not None:
                        ab = AnnotationBbox(OffsetImage(_icon_by_name[name], zoom=0.055), (0.5, 1.08), frameon=False, xycoords='axes fraction')
                        ab.set_clip_on(False)
                        ax_s.add_artist(ab)
                    else:
                        ax_s.set_title(name, fontsize=7, color=color, pad=1)
                    # no agent mark in small plots

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=130)  # constant canvas -> consistent frame shape
            plt.close(fig)
            buf.seek(0)
            return imageio.v2.imread(buf)

        # 1) collect snapshots per observe
        snapshots: List[Dict[str, set]] = []
        step_idx = 0
        for res in action_results:
            # We assume results reflect sequential execution; update manager state with any Observe results
            # Note: We only need relation_triples to update solver
            if res.action_type in ('observe', 'observe_approx'):
                triples = res.data.get('relation_triples', []) if hasattr(res, 'data') else []
                if triples:
                    solver.add_observation(triples)
                domains = solver.get_possible_positions()
                snapshots.append(domains)
            step_idx += 1

        # 2) compute global bounds with minimal whitespace (only initialized domains)
        g = int(self.grid_size)
        if snapshots:
            coords = []
            for d in snapshots:
                for n, dom in d.items():
                    if n == 'initial_pos':
                        continue
                    if not dom or len(dom) >= full_size:
                        continue
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

        # 3) render frames using shared bounds
        frames = []
        for i, doms in enumerate(snapshots):
            frame = _render(doms, step_title=f"Observe @ step {i}", bounds=bounds)
            frames.append(frame)

        if out_dir is not None and frames:
            for i, img in enumerate(frames):
                imageio.v2.imwrite(f"{out_dir.rstrip('/')}/heatmap_{i:03d}.png", img)
            # also export GIF for the sequence
            imageio.mimsave(f"{out_dir.rstrip('/')}/heatmaps.gif", frames, duration=(1.0 / max(1, int(fps))))
        return frames

    # ---- Agent trajectory to GIF ----
    def animate_agent_trajectory(self, action_results: List, out_path: Optional[str] = None, fps: int = 2) -> str:
        """Export a GIF showing the agent pose after each ActionResult using RoomPlotter."""
        # We will simulate by re-executing commands via ExplorationManager only for state updates where needed.
        mgr = ExplorationManager(self.room, self.agent)
        frames: List[np.ndarray] = []
        for i, res in enumerate(action_results):
            # Rebuild action from the logged command string when needed to update state
            cmd = res.action_command
            act = ActionSequence._parse_single_action(cmd) if cmd else None
            if act is not None:
                _ = mgr.execute_success_action(act)
            observe = (res.action_type in ('observe', 'observe_approx'))
            frame = RoomPlotter.plot_to_image(mgr.exploration_room, mgr.agent, observe=observe, dpi=120)
            frames.append(frame)
        out_file = out_path or 'trajectory.gif'
        imageio.mimsave(out_file, frames, duration=(1.0 / max(1, int(fps))))
        return out_file

    @staticmethod
    def flatten_turns(turns: List) -> List:
        """Flatten AgentProxy turns to a flat list of ActionResult."""
        seq = []
        for t in turns:
            seq.extend(list(getattr(t, 'actions', []) or []))
        return seq


if __name__ == "__main__":
    from ..managers.agent_proxy import OracleAgentProxy, InquisitorAgentProxy, AnalystAgentProxy
    from ..utils.room_utils import RoomGenerator
    from ..core.constant import ObjectInfo
    room, agent = RoomGenerator.generate_room(
        room_size=[15, 15],
        n_objects=4,
        np_random=np.random.default_rng(2),
        level=1,
        main=6,
        candidate_objects=[ObjectInfo(name='basket', has_orientation=True), ObjectInfo(name='chair', has_orientation=True), ObjectInfo(name='kettle', has_orientation=True), ObjectInfo(name='table', has_orientation=True)]
    )
    print(room)
    print(agent)
    # proxy = InquisitorAgentProxy(room, agent)
    proxy = AnalystAgentProxy(room, agent, delegate='observer_analyst', observer_delegate='strategist')
    proxy.run()
    action_results = ReplayHelper.flatten_turns(proxy.turns)
    # print(action_results)
    replay = ReplayHelper(room, agent)
    # replay.animate_agent_trajectory(action_results, out_path='trajectory.gif')
    replay.plot_observation_heatmaps(action_results, out_dir='heatmaps', fps=1, axes=False, use_icons=True, use_icon_colors=True)