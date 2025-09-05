from typing import List, Dict, Tuple, Optional
import random
import io
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import patches
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


# Color helpers: unique, reproducible identity hues
_BASE_HEX = [
    '#0072B2', '#E69F00', '#009E73', '#CC79A7', '#F0E442', '#D55E00', '#8C564B', '#7F7F7F',
    '#56B4E9', '#9467BD', '#2CA02C', '#FF7F0E', '#1F77B4', '#17BECF', '#BCBD22', '#AEC7E8',
    '#98DF8A', '#FFBB78', '#C49C94', '#F7B6D2'
]


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


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
    def plot_observation_heatmaps(self, action_results: List, max_positions: int = 500, sigma: float = 0.5,
                                  out_dir: Optional[str] = None, show: bool = False) -> List[np.ndarray]:
        """Execute actions and, at each observe, draw multi-object heatmaps of possible positions.

        - Only plot objects whose domain is initialized (domain size < full grid).
        - For very large domains, randomly sample up to max_positions.
        - Each object's domain becomes a smooth heat via local Gaussians (sigma).
        """
        mgr = ExplorationManager(self.room, self.agent)
        names = [o.name for o in self.room.all_objects] + ['initial_pos']
        solver = SpatialSolver(names, grid_size=self.grid_size)
        solver.set_initial_position('initial_pos', (0, 0))

        frames: List[np.ndarray] = []
        full_size = _full_domain_size(self.grid_size)
        # unique, stable colors for each object name
        _color_by_name = _assign_identity_colors([o.name for o in self.room.all_objects if o.name != 'initial_pos'])

        def _render(domains: Dict[str, set], step_title: str, bounds: Tuple[float, float, float, float]) -> np.ndarray:
            g = int(self.grid_size)
            H = W = 2 * g + 1
            fig = plt.figure(figsize=(7.0, 7.6))
            ax = fig.add_axes([0.08, 0.26, 0.84, 0.7])  # main
            ax.set_title(step_title)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
            ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.35)

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

            # main view: low-alpha colored heat + smooth linework (Scheme B)
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
                ssum = float(heat.sum())
                if ssum > 0:
                    heat /= ssum  # probability mass = 1 for each object
                # colored overlay with unified brightness and limited alpha
                vmax = (heat.max() + 1e-9)
                rgba = np.zeros((H, W, 4), dtype=np.float32)
                rgb = _hex_to_rgb(_color_by_name.get(name, '#E69F00'))
                rgba[..., :3] = rgb
                rgba[..., 3] = np.clip(heat / vmax, 0, 1) * 0.6
                ax.imshow(rgba, origin='lower', extent=(-g, g, -g, g), interpolation='bilinear')
                # smooth ellipse outlines instead of polygonal contours
                color = _color_by_name.get(name, '#E69F00')
                _draw_cov_ellipses(ax, heat, color)

                if len(dom) == 1:
                    (x, y) = dom[0]
                    ax.scatter([x], [y], s=70, c=color, marker='o', edgecolors='k', linewidths=0.5, zorder=5)  # circle

            # legend
            handles = [patches.Patch(color=_color_by_name.get(o.name, '#E69F00'), label=o.name) for o in self.room.all_objects if o.name != 'initial_pos']
            by_label = {h.get_label(): h for h in handles}
            ax.legend(handles=list(by_label.values()), loc='upper right', fontsize=8, framealpha=0.85)

            # bottom row of small multiples (consistent background even if uninitialized)
            names = [o.name for o in self.room.all_objects if o.name != 'initial_pos']
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
                    ax_s.imshow(np.full((H, W), 0.96), origin='lower', extent=(-g, g, -g, g), cmap='gray', vmin=0, vmax=1)
                    ax_s.set_xlim(x_min - 0.5, x_max + 0.5)
                    ax_s.set_ylim(y_min - 0.5, y_max + 0.5)
                    dom = list(domains.get(name, set()))
                    if not dom or len(dom) >= full_size:
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
                    ssum = float(heat.sum())
                    if ssum > 0:
                        heat /= ssum
                    # Option A: perceptually-uniform colormap (viridis)
                    ax_s.imshow(heat / (heat.max() + 1e-9), origin='lower', extent=(-g, g, -g, g), cmap='viridis', interpolation='bilinear')
                    ax_s.set_xlim(x_min - 0.5, x_max + 0.5)
                    ax_s.set_ylim(y_min - 0.5, y_max + 0.5)
                    color = _color_by_name.get(name, '#E69F00')
                    _draw_cov_ellipses(ax_s, heat, color)
                    ax_s.set_title(name, fontsize=7, color=color, pad=1)

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
            imageio.mimsave(f"{out_dir.rstrip('/')}/heatmaps.gif", frames, duration=0.8)
        if show:
            for img in frames:
                plt.figure(figsize=(5, 5))
                plt.imshow(img)
                plt.axis('off')
                plt.show()
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
    from ..managers.agent_proxy import OracleAgentProxy, InquisitorAgentProxy
    from ..utils.room_utils import RoomGenerator
    room, agent = RoomGenerator.generate_room(
        room_size=[20, 20],
        n_objects=9,
        np_random=np.random.default_rng(2),
        level=0,
        main=15
    )
    print(room)
    print(agent)
    proxy = InquisitorAgentProxy(room, agent)
    proxy.run()
    action_results = ReplayHelper.flatten_turns(proxy.turns)
    # print(action_results)
    replay = ReplayHelper(room, agent)
    replay.animate_agent_trajectory(action_results, out_path='trajectory.gif')
    replay.plot_observation_heatmaps(action_results, out_dir='heatmaps', show=False)