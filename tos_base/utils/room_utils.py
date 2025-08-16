import numpy as np
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt

from ..core.room import Room
from ..core.constant import CANDIDATE_OBJECTS, ObjectInfo
from ..core.object import Object, Agent, Gate
from .generate_room_layout import generate_room_layout


class RoomGenerator:
    """Room generator.
    1. Coordinates:
        - World coordinates are (x, y). Mask indexing is [y, x]. That is, columns map to x and rows map to y.
        - In mask indexing, column increasing means x increasing; row increasing means y increasing.
        - Neighbor convention on masks: "up" is y+1 and "down" is y-1; "right" is x+1 and "left" is x-1.
        - For visualization, we render with origin at bottom-left to keep y increasing upwards in world space.
    2. Room id:
        - Room id is an integer in [1, 99].
        - 0 for wall, 1 for main room, 100 for north-south door, 101 for east-west door.
    """
    @staticmethod
    def _default_mask(room_size: tuple[int, int]) -> np.ndarray:
        w, h = int(room_size[0]), int(room_size[1])
        mask = np.ones((h + 1, w + 1), dtype=np.int8)
        mask[[0, -1], :] = -1
        mask[:, [0, -1]] = -1
        return mask

    @staticmethod
    def _get_valid_positions(mask: np.ndarray, room_id: int | None = None) -> List[Tuple[int, int]]:
        """Get valid positions in the mask."""
        if room_id is not None:
            valid = np.argwhere(mask.T == int(room_id))
        else:
            valid = np.argwhere((mask.T >= 1) & (mask.T < 100))
        return valid
    
    @staticmethod
    def _gen_gates_from_mask(msk: np.ndarray) -> List[Gate]:
        gates: List[Gate] = []
        h, w = msk.shape
        cnt = 0
        # vertical doors (100): look up and down for room ids
        ys, xs = np.where(msk == 100)
        for y, x in zip(ys.tolist(), xs.tolist()):
            up, down = int(msk[y + 1, x]), int(msk[y - 1, x])
            if 1 <= up < 100 and 1 <= down < 100 and up != down:
                g = Gate(
                    name=f"door_{cnt}",
                    pos=np.array([x, y], dtype=int),
                    ori=np.array([0, 1], dtype=int),
                    room_id=[int(up), int(down)],
                    ori_by_room={int(up): np.array([0, 1], dtype=int), int(down): np.array([0, -1], dtype=int)},
                )
                gates.append(g); cnt += 1
        # horizontal doors (101): look left and right for room ids
        ys, xs = np.where(msk == 101)
        for y, x in zip(ys.tolist(), xs.tolist()):
            left, right = int(msk[y, x - 1]), int(msk[y, x + 1])
            if 1 <= left < 100 and 1 <= right < 100 and left != right:
                g = Gate(
                    name=f"door_{cnt}",
                    pos=np.array([x, y], dtype=int),
                    ori=np.array([1, 0], dtype=int),
                    room_id=[int(left), int(right)],
                    ori_by_room={int(left): np.array([-1, 0], dtype=int), int(right): np.array([1, 0], dtype=int)},
                )
                gates.append(g); cnt += 1
        return gates

    @staticmethod
    def generate_room(
        room_size: Tuple[int, int],
        n_objects: int,
        np_random: np.random.Generator,
        room_name: str = 'room',
        candidate_objects: List[ObjectInfo] = CANDIDATE_OBJECTS,
        level: int = 0,
        main: Optional[int] = None,
    ) -> Tuple[Room, Agent]:
        """Generate a multi-room layout, gates, objects, and agent.
        - Mask is generated via generate_room_layout; gates derived from mask.
        - Agent is sampled from main room (room id = 1).
        """
        n = int(max(room_size[0], room_size[1]))
        mask = generate_room_layout(n=n, level=int(level), main=main, np_random=np_random)

        gates = RoomGenerator._gen_gates_from_mask(mask)

        # sample objects and agent; ensure agent in main room (id=1)
        objects = RoomGenerator._gen_objects(
            n=n_objects,
            random_generator=np_random,
            room_size=list(room_size),
            perspective_taking=True,
            candidate_list=candidate_objects,
            mask=mask,
        )

        # build room
        room = Room(objects=objects, name=room_name, mask=mask.copy(), gates=gates)

        # assign agent room id
        agent_pos = room.get_random_point(np_random, room_id=1)
        while any(np.allclose(agent_pos, obj.pos) for obj in objects):
            agent_pos = room.get_random_point(np_random, room_id=1)
        agent = Agent(name='agent', pos=agent_pos)
        agent.room_id = 1
        agent.init_room_id = agent.room_id
        return room, agent
    
    @staticmethod
    def _gen_objects(
        n: int,
        random_generator: np.random.Generator,
        room_size: list[int] = [5, 5],
        perspective_taking: bool = False,
        candidate_list: list[ObjectInfo] = CANDIDATE_OBJECTS,
        mask: Optional[np.ndarray] = None,
    ) -> List[Object]:
        """Sample objects (names, orientations) and positions from mask."""
        if mask is None:
            mask = RoomGenerator._default_mask((room_size[0], room_size[1]))
        valid_positions = RoomGenerator._get_valid_positions(mask)
        assert len(valid_positions) >= n
        random_generator.shuffle(valid_positions)
        obj_positions = valid_positions[:n]

        indices = random_generator.choice(len(candidate_list), n, replace=False)
        selected_object_info = [candidate_list[i] for i in indices]
        orientations = random_generator.integers(0, 4, n)
        ori_vectors = {0: [0, 1], 1: [1, 0], 2: [0, -1], 3: [-1, 0]}

        objects = []
        for obj_info, pos, ori_idx in zip(selected_object_info, obj_positions, orientations):
            ori = np.array(ori_vectors[int(ori_idx)]) if obj_info.has_orientation and perspective_taking else np.array([0, 1])
            objects.append(Object(name=obj_info.name, pos=np.array(pos, dtype=int), ori=ori, has_orientation=obj_info.has_orientation))
        return objects


class RoomPlotter:
    @staticmethod
    def plot(room: Room, agent: Agent | None, mode: str = 'text', save_path: str | None = None):
        has_mask = getattr(room, 'mask', None) is not None
        if mode == 'text':
            # helpers
            arrow = lambda v: { (0,1): '↑', (1,0): '→', (0,-1): '↓', (-1,0): '←' }.get(tuple(v), '•')
            col = lambda s,c: f"\033[{c}m{s}\033[0m"
            BLUE, RED, GRN = 34, 31, 32
            # grid from mask if available
            if has_mask:
                h, w = room.mask.shape
                min_x, max_x, min_y, max_y = 0, w - 1, 0, h - 1
                grid = [[' '] * w for _ in range(h)]
                # draw background: rooms '.', walls '#', doors '+'
                for y in range(h):
                    for x in range(w):
                        v = int(room.mask[y, x])
                        if v == -1:
                            ch = ' '
                        elif v == 0:
                            ch = '#'
                        elif v in (100, 101):
                            ch = '+'
                        else:
                            ch = '·'
                        grid[h - 1 - y][x] = ch
                # room labels at centers
                rids = sorted(int(r) for r in np.unique(room.mask) if 1 <= int(r) < 100)
                for rid in rids:
                    ys, xs = np.where(room.mask == rid)
                    if len(xs) == 0:
                        continue
                    cy, cx = int(np.mean(ys)), int(np.mean(xs))
                    grid[h - 1 - cy][cx] = str(rid % 10)
            else:
                # bounds/grid from objects if no mask
                min_x, max_x, min_y, max_y = room.get_boundary()
                min_x, max_x, min_y, max_y = int(min_x)-1, int(max_x)+1, int(min_y)-1, int(max_y)+1
                w, h = max_x - min_x + 1, max_y - min_y + 1
                grid = [['·'] * w for _ in range(h)]
            # objects + gates with orientation
            for obj in room.all_objects:
                x, y = int(obj.pos[0]) - min_x, max_y - int(obj.pos[1])
                if 0 <= y < h and 0 <= x < w:
                    ch = 'D' if obj in getattr(room, 'gates', []) else (arrow(obj.ori) if getattr(obj, 'has_orientation', True) else '●')
                    grid[y][x] = col(ch, RED if obj in getattr(room, 'gates', []) else BLUE)
            # agent (current + init)
            if agent is not None:
                ax, ay = int(agent.pos[0]) - min_x, max_y - int(agent.pos[1])
                if 0 <= ay < h and 0 <= ax < w:
                    grid[ay][ax] = col(arrow(agent.ori), GRN)
                iax, iay = int(agent.init_pos[0]) - min_x, max_y - int(agent.init_pos[1])
                if 0 <= iay < h and 0 <= iax < w and not (iax == ax and iay == ay):
                    grid[iay][iax] = col('×', GRN)
            print(f"--- {room.name} ---")
            for y in range(h):
                print(' '.join(grid[y]))
            if agent is not None:
                def name_ori(v):
                    m = {(0,1): 'N', (1,0): 'E', (0,-1): 'S', (-1,0): 'W'}
                    return m.get(tuple(v), '?')
                print(f"A:{tuple(agent.pos)} {name_ori(agent.ori)}  a(init):{tuple(agent.init_pos)} {name_ori(getattr(agent,'init_ori',agent.ori))}")
            return
        elif mode == 'img':

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_facecolor('white')
            if has_mask:
                h, w = room.mask.shape
                min_x, max_x, min_y, max_y = 0, w - 1, 0, h - 1
                # build label map for background coloring
                label = np.zeros_like(room.mask, dtype=int)
                label[room.mask == 0] = 1
                label[(room.mask == 100) | (room.mask == 101)] = 2
                rids = sorted(int(r) for r in np.unique(room.mask) if 1 <= int(r) < 100)
                for i, rid in enumerate(rids, start=3):
                    label[room.mask == rid] = i
                colors = ['#111111', '#888888', '#ffcc33', '#4e79a7', '#59a14f', '#f28e2b', '#e15759', '#76b7b2', '#edc949', '#af7aa1']
                # repeat colors if many rooms
                while len(colors) <= label.max():
                    colors += colors[3:]
                from matplotlib.colors import ListedColormap
                ax.imshow(label, origin='lower', cmap=ListedColormap(colors[:label.max()+1]),
                          extent=(min_x-0.5, max_x+0.5, min_y-0.5, max_y+0.5), interpolation='nearest', zorder=0, alpha=0.18)
                # room id labels
                for rid in rids:
                    ys, xs = np.where(room.mask == rid)
                    if len(xs) == 0:
                        continue
                    cx, cy = float(np.mean(xs)), float(np.mean(ys))
                    ax.text(cx, cy, str(rid), color='white', ha='center', va='center', fontsize=8, zorder=1, alpha=0.85)
            else:
                min_x, max_x, min_y, max_y = room.get_boundary()
            # grid
            ax.set_xlim(min_x-0.5, max_x+0.5); ax.set_ylim(min_y-0.5, max_y+0.5)
            ax.set_xticks(np.arange(int(min_x), int(max_x)+1)); ax.set_yticks(np.arange(int(min_y), int(max_y)+1))
            ax.grid(True, color='#bdbdbd', linewidth=0.2)
            for s in ax.spines.values(): s.set_visible(False)
            # draw objects/gates + orientation (distinct markers/colors; legend, no text labels)
            palette = plt.get_cmap('tab10').colors
            markers = ['o','s','D','P','X','h','H','*','p','d']
            name_to_idx, seen_labels = {}, set()
            offs = [(0.12,0.18), (0.18,-0.18), (-0.18,0.18), (-0.18,-0.18)]
            for obj in room.all_objects:
                x, y = float(obj.pos[0]), float(obj.pos[1])
                if isinstance(obj, Gate):
                    color, marker, label = 'crimson', 'D', ('gate' if 'gate' not in seen_labels else None)
                    seen_labels.add('gate')
                else:
                    i = name_to_idx.setdefault(obj.name, len(name_to_idx))
                    color, marker = palette[i % len(palette)], markers[i % len(markers)]
                    label = obj.name if obj.name not in seen_labels else None
                    seen_labels.add(obj.name)
                ax.scatter(x, y, c=[color], marker=marker, s=64, edgecolors='white', linewidths=0.7, zorder=3, label=label)
                # coords label
                # orientation arrow
                if getattr(obj, 'has_orientation', True):
                    dx, dy = float(obj.ori[0])*0.4, float(obj.ori[1])*0.4
                    ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=0.5, color='grey', width=0.005)
            if agent is not None:
                ax.scatter(agent.pos[0], agent.pos[1], c='green', marker='s', s=70, edgecolors='white', linewidths=0.7, label='agent', zorder=5)
                ax.scatter(agent.init_pos[0], agent.init_pos[1], c='green', marker='x', s=60, label='agent_init', zorder=4)
                dx, dy = float(agent.ori[0])*0.4, float(agent.ori[1])*0.4
                idx, idy = float(agent.init_ori[0])*0.4, float(agent.init_ori[1])*0.4
                ax.quiver(agent.pos[0], agent.pos[1], dx, dy, angles='xy', scale_units='xy', scale=0.5, color='grey', width=0.005)
                ax.quiver(agent.init_pos[0], agent.init_pos[1], idx, idy, angles='xy', scale_units='xy', scale=0.5, color='grey', width=0.005)
            h,l = ax.get_legend_handles_labels(); 
            if l:
                d = dict(zip(l,h)); ax.legend(d.values(), d.keys(), loc='upper right', frameon=False, fontsize=9)
            ax.set_aspect('equal'); ax.set_title(room.name)
            if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(); return
        else:
            raise ValueError('mode must be text or img')


def get_topdown_info(room: Room, agent: Agent) -> str:
    mapping = {(0, 1): "north", (0, -1): "south", (1, 0): "east", (-1, 0): "west"}
    lines = [f"Agent at ({agent.pos[0]}, {agent.pos[1]}) facing {mapping[tuple(agent.ori)]}"]
    for obj in room.all_objects:
        if isinstance(obj, Gate):
            gate_ori_info = f'on east-west wall' if mapping[tuple(obj.ori)] in ('north', 'south') else f'on north-south wall'
            obj_info = f"Gate: {obj.name} at ({obj.pos[0]}, {obj.pos[1]}) {gate_ori_info}"
        else:
            obj_info = f"Object: {obj.name} at ({obj.pos[0]}, {obj.pos[1]}) facing {mapping[tuple(obj.ori)]}"
        lines.append(obj_info)
    return "\n".join(lines)


def get_room_description(room: Room, agent: Agent, with_topdown: bool = False) -> str:
    room_type = "multiple rooms connected by doors" if room.gates else "a room"
    desc = f"Imagine {room_type}. You face north.\nObjects: {', '.join([o.name for o in room.all_objects])}"
    if with_topdown:
        desc += "\n" + get_topdown_info(room, agent)
    return desc



if __name__ == '__main__':
    room, agent = RoomGenerator.generate_room(
        room_size=[20, 20],
        main=5,
        n_objects=3,
        level=3,
        np_random=np.random.default_rng(42),
    )
    print(room)
    print(agent)

    RoomPlotter.plot(room, agent, mode='img', save_path='room.png')
    RoomPlotter.plot(room, agent, mode='text')

    # print(get_room_description(room, agent, with_topdown=True))