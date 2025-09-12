from typing import List, Dict, Tuple, Optional
import random, io, os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import imageio

from ...core.room import Room
from ...core.object import Agent
from ...actions.actions import ActionSequence
from ...managers.exploration_manager import ExplorationManager
from ...managers.spatial_solver import SpatialSolver
from ..room_utils import RoomPlotter


def _trim_transparent(icon: np.ndarray, thr: int = 10) -> np.ndarray:
    if icon.ndim == 3 and icon.shape[2] == 4:
        a = icon[..., 3] > thr
        if a.any():
            ys, xs = np.where(a)
            return icon[ys.min():ys.max()+1, xs.min():xs.max()+1]
    return icon


def _load_icon(name: str, files: List[str]) -> Optional[np.ndarray]:
    if not files: return None
    lname = name.lower().replace(' ', '_')
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0].lower()
        if base == lname:
            return _trim_transparent(imageio.v2.imread(f))
    if 'door' in lname:
        for f in files:
            base = os.path.splitext(os.path.basename(f))[0].lower()
            if base == 'door':
                return _trim_transparent(imageio.v2.imread(f))
    return _trim_transparent(imageio.v2.imread(random.choice(files)))


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


class ReplayHelper:
    """Replay utilities for observation heatmaps and trajectories."""

    def __init__(self, room: 'Room', agent: 'Agent', grid_size: Optional[int] = None):
        self.room = room.copy()
        self.agent = agent.copy()
        g = (max(self.room.mask.shape) if getattr(self.room, 'mask', None) is not None else 10)
        self.grid_size = int(g if grid_size is None else grid_size)

    def _build_colors_and_icons(
        self,
        obj_names: List[str],
        use_icons: bool,
        icons_dir: Optional[str],
    ) -> Tuple[Dict[str, str], Dict[str, Optional[np.ndarray]], Optional[np.ndarray]]:
        color_by_name = {n: '#E69F00' for n in obj_names}
        icon_by_name: Dict[str, Optional[np.ndarray]] = {}
        agent_icon = None
        if use_icons:
            idir = icons_dir or os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'icons'))
            files = sorted(glob.glob(os.path.join(idir, '*.png')))
            if files:
                agent_path = os.path.join(idir, 'agent.png')
                agent_icon = _trim_transparent(imageio.v2.imread(agent_path)) if os.path.exists(agent_path) else None
                for n in obj_names:
                    icon_by_name[n] = _load_icon(n, files)
        return color_by_name, icon_by_name, agent_icon

    def plot_observation_heatmaps(
        self,
        action_results: List,
        out_dir: Optional[str] = None,
        fps: int = 2,
        use_icons: bool = False,
        icons_dir: Optional[str] = None,
    ) -> List[np.ndarray]:
        mgr = ExplorationManager(self.room, self.agent)
        _ = mgr
        snapshots = self._collect_snapshots(action_results)
        obj_names = [o.name for o in self.room.all_objects if o.name != 'initial_pos']
        color_by_name, icon_by_name, agent_icon = self._build_colors_and_icons(obj_names, use_icons, icons_dir)
        frames = [
            RoomPlotter.plot_to_image(mgr.exploration_room, mgr.agent, observe=False, dpi=120)
            for _ in snapshots
        ]
        if out_dir and frames:
            os.makedirs(out_dir, exist_ok=True)
            for i, img in enumerate(frames):
                imageio.v2.imwrite(f"{out_dir.rstrip('/')}/heatmap_{i:03d}.png", img)
            imageio.mimsave(f"{out_dir.rstrip('/')}/heatmaps.gif", frames, duration=(1.0 / max(1, int(fps))))
        return frames

    def _collect_snapshots(self, action_results: List) -> List[Dict[str, set]]:
        names = [o.name for o in self.room.all_objects] + ['initial_pos']
        solver = SpatialSolver(names, grid_size=self.grid_size)
        solver.set_initial_position('initial_pos', (0, 0))
        snapshots: List[Dict[str, set]] = []
        for res in action_results:
            at = getattr(res, 'action_type', None)
            if at in ('observe', 'query'):
                triples = res.data.get('relation_triples', []) if hasattr(res, 'data') else []
                if triples:
                    solver.add_observation(triples)
                    snapshots.append(solver.get_possible_positions())
        return snapshots

    def animate_agent_trajectory(self, action_results: List, out_path: Optional[str] = None, fps: int = 2) -> str:
        mgr = ExplorationManager(self.room, self.agent); frames: List[np.ndarray] = []
        for res in action_results:
            act = ActionSequence._parse_single_action(res.action_command) if res.action_command else None
            if act is not None: _ = mgr.execute_success_action(act)
            frame = RoomPlotter.plot_to_image(mgr.exploration_room, mgr.agent, observe=(res.action_type in ('observe',)), dpi=120)
            frames.append(frame)
        out_file = out_path or 'trajectory.gif'
        imageio.mimsave(out_file, frames, duration=(1.0 / max(1, int(fps))))
        return out_file


