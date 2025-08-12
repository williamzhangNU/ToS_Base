import numpy as np
from typing import List, Dict, Any
import copy

from .object import Object, Gate


class BaseRoom:
    """Minimal room without mask/gates input. Holds only name and objects."""

    def __init__(self, objects: List[Object], name: str = 'room'):
        self.name = name
        self.mask = None
        self._init_objects(objects, [])

    def _init_objects(self, objects: List[Object], gates: List[Object] | None = None):
        self.objects = copy.deepcopy(objects)
        self.gates = copy.deepcopy(gates or [])
        self.object_map = None
        self.all_objects = self.objects + self.gates
        names = [obj.name for obj in self.all_objects]
        assert len(names) == len(set(names)), "All object names must be unique"

    def add_object(self, obj: Object):
        self._init_objects(self.objects + [obj], self.gates)

    def remove_object(self, obj_name: str):
        self._init_objects([o for o in self.objects if o.name != obj_name], self.gates)

    def get_object_by_name(self, name: str) -> Object:
        for obj in self.all_objects:
            if obj.name == name:
                return obj
        raise ValueError(f"Object '{name}' not found in room")

    def has_object(self, name: str) -> bool:
        return any(obj.name == name for obj in self.all_objects)

    def get_boundary(self):
        positions = np.array([obj.pos for obj in self.all_objects])
        min_x, min_y = np.min(positions, axis=0)
        max_x, max_y = np.max(positions, axis=0)
        min_x_bound = min_x - min(max_x - min_x, 1)
        max_x_bound = max_x + min(max_x - min_x, 1)
        min_y_bound = min_y - min(max_y - min_y, 1)
        max_y_bound = max_y + min(max_y - min_y, 1)
        return min_x_bound, max_x_bound, min_y_bound, max_y_bound

    def get_random_point(self, rng: np.random.Generator, room_id: int | None = None) -> np.ndarray:
        """Random coordinate around object bbox; room_id ignored (no mask)."""
        if not self.all_objects:
            return np.array([0, 0], dtype=int)
        positions = np.array([obj.pos for obj in self.all_objects])
        min_x, min_y = np.min(positions, axis=0)
        max_x, max_y = np.max(positions, axis=0)
        low_x, high_x = int(np.floor(min_x)) - 1, int(np.ceil(max_x)) + 1
        low_y, high_y = int(np.floor(min_y)) - 1, int(np.ceil(max_y)) + 1
        return np.array([rng.integers(low_x, high_x + 1), rng.integers(low_y, high_y + 1)], dtype=int)

    def get_objects_orientation(self):
        ori_mapping = {(0, 1): "north", (0, -1): "south", (1, 0): "east", (-1, 0): "west"}
        desc = "Orientation of objects in the room are: \n"
        for obj in self.objects:
            desc += f"{obj.name} facing {ori_mapping[tuple(obj.ori)]}\n"
        return desc

    def get_cell_info(self, x: int, y: int) -> Dict[str, Any]:
        info: Dict[str, Any] = {"room_id": None, "object_name": None, "gate_name": None}
        for g in self.gates:
            if int(g.pos[0]) == int(x) and int(g.pos[1]) == int(y):
                info["gate_name"] = g.name
                return info
        for o in self.objects:
            if int(o.pos[0]) == int(x) and int(o.pos[1]) == int(y):
                info["object_name"] = o.name
                return info
        return info

    def copy(self) -> 'BaseRoom':
        return self.from_dict(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'objects': [obj.to_dict() for obj in self.objects],
            'all_objects': [obj.to_dict() for obj in self.all_objects],
            'gates': [obj.to_dict() for obj in self.gates],
            'gates_by_room': {},
            'rooms_by_gate': {},
            'adjacent_rooms_by_room': {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseRoom':
        return cls(objects=[Object.from_dict(o) for o in data.get('objects', [])], name=data.get('name', 'room'))

    def __repr__(self):
        objects_details = [f"{obj.name}@{tuple(obj.pos)}:{tuple(obj.ori)}" for obj in self.objects]
        return f"BaseRoom(name={self.name}, objects=[{', '.join(objects_details)}])"


class Room(BaseRoom):
    """Full room with required mask and optional gates; inherits BaseRoom."""

    def __init__(self, objects: List[Object], mask: np.ndarray, name: str = 'room', gates: List[Gate] | None = None):
        assert mask is not None, "mask must not be None"
        self.name = name
        self.mask = mask.copy()
        self._init_objects(objects, gates or [])
        self._init_map()
        self._build_membership_from_mask()

    def _init_map(self):
        self.object_map = np.zeros_like(self.mask, dtype=int)
        self.name_to_idx = {o.name: i + 1 for i, o in enumerate(self.all_objects)}
        self.idx_to_name = {i + 1: o.name for i, o in enumerate(self.all_objects)}
        for o in self.all_objects:
            x, y = int(o.pos[0]), int(o.pos[1])
            self.object_map[y, x] = self.name_to_idx[o.name]

    def get_boundary(self):
        """Bounds from mask shape: x in [0,w-1], y in [0,h-1]."""
        h, w = self.mask.shape
        return 0, w - 1, 0, h - 1

    def get_random_point(self, rng: np.random.Generator, room_id: int | None = None) -> np.ndarray:
        """Random valid mask coordinate; filter by room_id if provided."""
        valid = np.argwhere(self.mask == int(room_id)) if room_id is not None else np.argwhere(self.mask != -1)
        if valid.size == 0:
            return np.array([0, 0], dtype=int)
        y, x = valid[rng.integers(0, len(valid))]
        return np.array([int(x), int(y)], dtype=int)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            'name': self.name,
            'objects': [obj.to_dict() for obj in self.objects],
            'gates': [obj.to_dict() for obj in self.gates],
            'all_objects': [obj.to_dict() for obj in self.all_objects],
            'gates_by_room': getattr(self, 'gates_by_room', {}),
            'rooms_by_gate': getattr(self, 'rooms_by_gate', {}),
            'adjacent_rooms_by_room': getattr(self, 'adjacent_rooms_by_room', {}),
        }
        if self.mask is not None:
            data['mask'] = self.mask.tolist()
        if getattr(self, 'object_map', None) is not None:
            data['object_map'] = self.object_map.tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Room':
        mask = np.array(data['mask'], dtype=np.int8)
        return cls(
            objects=[Object.from_dict(obj_data) for obj_data in data['objects']],
            name=data['name'],
            mask=mask,
            gates=[Gate.from_dict(obj_data) for obj_data in data.get('gates', [])]
        )

    def _build_membership_from_mask(self) -> None:
        # Assign room_id for non-gate objects from mask
        for obj in self.objects:
            x, y = int(obj.pos[0]), int(obj.pos[1])
            rid = int(self.mask[y, x])
            assert rid > 0, f"Object {obj.name} is not in a room"
            obj.room_id = rid

        # Build gate mappings and adjacency
        gates_by_room: dict[int, list[str]] = {}
        rooms_by_gate: dict[str, list[int]] = {}
        adjacent_rooms_by_room: dict[int, list[int]] = {}
        for g in self.gates:
            assert isinstance(g.room_id, list) and len(g.room_id) == 2, f"Gate {g.name} must have room_id list set in generator"
            a, b = int(g.room_id[0]), int(g.room_id[1])
            rooms_by_gate[g.name] = [a, b]
            gates_by_room.setdefault(a, []).append(g.name)
            gates_by_room.setdefault(b, []).append(g.name)
            adjacent_rooms_by_room.setdefault(a, []).append(b)
            adjacent_rooms_by_room.setdefault(b, []).append(a)

        # Deduplicate adjacency lists
        for k in list(adjacent_rooms_by_room.keys()):
            adjacent_rooms_by_room[k] = sorted(list(set(adjacent_rooms_by_room[k])))
        self.gates_by_room, self.rooms_by_gate, self.adjacent_rooms_by_room = gates_by_room, rooms_by_gate, adjacent_rooms_by_room

    def get_cell_info(self, x: int, y: int) -> Dict[str, Any]:
        info: Dict[str, Any] = {"room_id": None, "object_name": None, "gate_name": None}
        if self.mask is not None:
            if 0 <= y < self.mask.shape[0] and 0 <= x < self.mask.shape[1]:
                rid = int(self.mask[y, x])
                info["room_id"] = rid if rid > 0 else None
        if getattr(self, 'object_map', None) is not None and self.object_map is not None:
            if 0 <= y < self.object_map.shape[0] and 0 <= x < self.object_map.shape[1]:
                idx = int(self.object_map[y, x])
                if idx > 0:
                    name = self.idx_to_name.get(idx)
                    if name is not None:
                        if any(g.name == name for g in self.gates):
                            info["gate_name"] = name
                        else:
                            info["object_name"] = name
        return info

    def __repr__(self):
        objects_details = [f"{obj.name}@{tuple(obj.pos)}:{tuple(obj.ori)}" for obj in self.objects]
        return f"Room(name={self.name}, objects=[{', '.join(objects_details)}])"



if __name__ == '__main__':
    from ..utils.room_utils import RoomGenerator
    room, agent = RoomGenerator.generate_room(
        room_size=[10, 10],
        n_objects=3,
        generation_type='rand',
        np_random=np.random.default_rng(42),
    )
    print(room)
    print(agent)
    print(room.get_random_point(np.random.default_rng(42)))
    print(room.object_map)
    print(room.get_boundary())
    print(room.get_cell_info(2, 2))
        # print(room.gates_by_room)
        # print(room.rooms_by_gate)
        # print(room.adjacent_rooms_by_room)




