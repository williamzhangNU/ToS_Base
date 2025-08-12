import numpy as np
from typing import Optional, Tuple, List

from ..core.room import Room
from ..core.constant import CANDIDATE_OBJECTS, ObjectInfo
from ..core.object import Object, Agent, Gate


class RoomGenerator:
    @staticmethod
    def _default_mask(room_size: tuple[int, int]) -> np.ndarray:
        w, h = int(room_size[0]), int(room_size[1])
        mask = np.ones((h + 1, w + 1), dtype=np.int8)
        mask[[0, -1], :] = -1
        mask[:, [0, -1]] = -1
        return mask

    @staticmethod
    def _mask_to_world_coords(mask: np.ndarray) -> List[Tuple[int, int]]:
        """Convert mask coordinates to world coordinates.
        World coordinate (x, y) is (y, x) in mask coordinates.
        Origin is top-left for both room and mask, and x+ is col+, y+ is row+.
        """
        valid = []
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x] not in (0, -1):
                    valid.append((x, y))
        return valid

    @staticmethod
    def generate_room(
        room_size: Tuple[int, int],
        n_objects: int,
        generation_type: str,
        np_random: np.random.Generator,
        room_name: str = 'room',
        candidate_objects: List[ObjectInfo] = CANDIDATE_OBJECTS,
        gates: List[Gate] = [],
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[Room, Agent]:
        """Generate a room and an agent.
        Place objects and agent on valid mask cells. If no mask, build from (w,h).
        """
        if mask is None:
            mask = RoomGenerator._default_mask(room_size)
        # delegate object and agent sampling to _gen_objects
        objects, agent = RoomGenerator._gen_objects(
            n=n_objects,
            random_generator=np_random,
            room_size=list(room_size),
            perspective_taking=(generation_type == 'pov'),
            candidate_list=candidate_objects,
            mask=mask,
        )

        # build room
        room = Room(objects=objects, name=room_name, mask=mask.copy(), gates=gates)

        # assign agent room id after room is built (prefer object_map/mask via room API)
        info = room.get_cell_info(int(agent.pos[0]), int(agent.pos[1]))
        agent.room_id = info.get('room_id')
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
    ) -> Tuple[list[Object], Agent]:
        """Sample objects (names, orientations) and positions from mask; sample agent position too."""
        if mask is None:
            mask = RoomGenerator._default_mask((room_size[0], room_size[1]))
        valid_positions = RoomGenerator._mask_to_world_coords(mask)
        assert len(valid_positions) >= n + 1
        random_generator.shuffle(valid_positions)
        obj_positions = valid_positions[:n]
        agent_pos = valid_positions[n]

        indices = random_generator.choice(len(candidate_list), n, replace=False)
        selected_object_info = [candidate_list[i] for i in indices]
        orientations = random_generator.integers(0, 4, n)
        ori_vectors = {0: [0, 1], 1: [1, 0], 2: [0, -1], 3: [-1, 0]}

        objects = []
        for obj_info, pos, ori_idx in zip(selected_object_info, obj_positions, orientations):
            ori = np.array(ori_vectors[int(ori_idx)]) if obj_info.has_orientation and perspective_taking else np.array([0, 1])
            objects.append(Object(name=obj_info.name, pos=np.array(pos, dtype=int), ori=ori, has_orientation=obj_info.has_orientation))
        agent = Agent(name='agent', pos=np.array(agent_pos, dtype=int))
        return objects, agent


class RoomPlotter:
    @staticmethod
    def plot(room: Room, agent: Agent | None, mode: str = 'text', save_path: str | None = None):
        if mode == 'text':
            # helpers
            arrow = lambda v: { (0,1): '↑', (1,0): '→', (0,-1): '↓', (-1,0): '←' }.get(tuple(v), '•')
            col = lambda s,c: f"\033[{c}m{s}\033[0m"
            BLUE, RED, GRN = 34, 31, 32
            # bounds/grid
            min_x, max_x, min_y, max_y = room.get_boundary()
            min_x, max_x, min_y, max_y = int(min_x)-1, int(max_x)+1, int(min_y)-1, int(max_y)+1
            w, h = max_x - min_x + 1, max_y - min_y + 1
            grid = [['·'] * w for _ in range(h)]
            # objects + gates with orientation
            for obj in room.all_objects:
                x, y = int(obj.pos[0]) - min_x, max_y - int(obj.pos[1])
                if 0 <= y < h and 0 <= x < w:
                    ch = arrow(obj.ori) if getattr(obj, 'has_orientation', True) else '●'
                    grid[y][x] = col(ch, RED if obj in room.gates else BLUE)
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
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print('matplotlib required')
                return
            fig, ax = plt.subplots(figsize=(6, 6))
            min_x, max_x, min_y, max_y = room.get_boundary()
            # grid
            ax.set_xlim(min_x-0.5, max_x+0.5); ax.set_ylim(min_y-0.5, max_y+0.5)
            ax.set_xticks(np.arange(int(min_x), int(max_x)+1)); ax.set_yticks(np.arange(int(min_y), int(max_y)+1))
            ax.grid(True, linestyle='-', color='#ddd', linewidth=0.8)
            # draw objects/gates + orientation
            for obj in room.all_objects:
                x, y = float(obj.pos[0]), float(obj.pos[1])
                color = 'red' if obj in room.gates else 'blue'
                ax.scatter(x, y, c=color, s=60, edgecolors='k', zorder=3)
                ax.annotate(obj.name, (x+0.12, y+0.12), fontsize=9, color=color)
                if getattr(obj, 'has_orientation', True):
                    ax.quiver(x, y, float(obj.ori[0]), float(obj.ori[1]), angles='xy', scale_units='xy', scale=3, color=color, zorder=4, width=0.006)
            if agent is not None:
                ax.scatter(agent.pos[0], agent.pos[1], c='green', marker='s', s=70, edgecolors='k', label='agent', zorder=5)
                ax.quiver(float(agent.pos[0]), float(agent.pos[1]), float(agent.ori[0]), float(agent.ori[1]), angles='xy', scale_units='xy', scale=3, color='green', zorder=6, width=0.008)
                ax.scatter(agent.init_pos[0], agent.init_pos[1], c='green', marker='x', s=60, label='agent_init', zorder=4)
                ax.quiver(float(agent.init_pos[0]), float(agent.init_pos[1]), float(agent.init_ori[0]), float(agent.init_ori[1]), angles='xy', scale_units='xy', scale=3, color='green', zorder=6, width=0.008)
                h,l = ax.get_legend_handles_labels(); 
                if l:
                    d = dict(zip(l,h)); ax.legend(d.values(), d.keys(), loc='upper right')
            ax.set_aspect('equal'); ax.set_title(room.name)
            if save_path: plt.savefig(save_path, bbox_inches='tight')
            plt.close(); return
        else:
            raise ValueError('mode must be text or img')


def get_topdown_info(room: Room, agent: Agent) -> str:
    mapping = {(0, 1): "north", (0, -1): "south", (1, 0): "east", (-1, 0): "west"}
    lines = [f"Agent at ({agent.pos[0]}, {agent.pos[1]}) facing {mapping[tuple(agent.ori)]}"]
    for obj in room.objects:
        lines.append(f"{obj.name} at ({obj.pos[0]}, {obj.pos[1]}) facing {mapping[tuple(obj.ori)]}")
    return "\n".join(lines)


def get_room_description(room: Room, agent: Agent, with_topdown: bool = False) -> str:
    desc = f"Imagine a room. You face north.\nObjects: {', '.join([o.name for o in room.objects])}"
    if with_topdown:
        desc += "\n" + get_topdown_info(room, agent)
    return desc



if __name__ == '__main__':
    room, agent = RoomGenerator.generate_room(
        room_size=[10, 10],
        n_objects=3,
        generation_type='rand',
        np_random=np.random.default_rng(42),
    )
    print(room)
    print(room.object_map)
    print(agent)

    RoomPlotter.plot(room, agent, mode='img', save_path='room.png')
    RoomPlotter.plot(room, agent, mode='text')