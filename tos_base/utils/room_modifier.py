from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.room import Room
from ..core.object import Object


@dataclass
class ChangedObject:
    """Represents a reported/ground-truth change for one object."""

    name: str
    pos: bool = False
    ori: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name}
        if self.pos:
            out["pos"] = True
        if self.ori:
            out["ori"] = True
        return out

    def merge(self, other: "ChangedObject") -> None:
        if self.name != other.name:
            raise ValueError(f"Cannot merge changes for different objects: {self.name} vs {other.name}")
        self.pos = bool(self.pos or other.pos)
        self.ori = bool(self.ori or other.ori)

    @classmethod
    def parse(cls, text: str) -> "ChangedObject":
        """Parse strings like:
        - "apple: position"
        - "chair orientation"
        - "table moved"
        """
        s = str(text).strip().strip('"\'')

        pos_kws = ("position", "location", "moved", "pos")
        ori_kws = ("orientation", "rotation", "rotated", "ori", "facing")

        if ":" in s:
            name, rest = s.split(":", 1)
        else:
            parts = s.rsplit(" ", 1)
            if len(parts) != 2:
                raise ValueError(f"Cannot parse: {s}")
            name, rest = parts[0], parts[1]

        rest_low = str(rest).lower()
        is_pos = any(k in rest_low for k in pos_kws)
        is_ori = any(k in rest_low for k in ori_kws)
        if not (is_pos or is_ori):
            raise ValueError(f"Unknown change type: {rest}")

        return cls(name=str(name).strip().strip('"\'').replace("_", " "), pos=is_pos, ori=is_ori)


class RoomModifier:
    def modify(self, room: Room) -> Tuple[Room, List[ChangedObject]]:
        raise NotImplementedError


class ObjectModifier(RoomModifier):
    """Modify a room by moving OR rotating objects (never both per object)."""

    _ROTATIONS = (np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0]))

    def __init__(self, seed: int, n_changes: int = 1, modification_type: Optional[str] = None):
        self.rng = np.random.default_rng(seed)
        self.n_changes = int(max(0, n_changes))
        self.modification_type = (modification_type or "").strip().lower() or None  # move|rotate|None

    def modify(self, room: Room) -> Tuple[Room, List[ChangedObject]]:
        r = room.copy()
        candidates = [o for o in r.objects if o.name != "agent"]
        if self.modification_type == "rotate":
            candidates = [o for o in candidates if getattr(o, "has_orientation", False)]
        if not candidates or self.n_changes <= 0:
            return r, []

        n = min(self.n_changes, len(candidates))
        targets = list(self.rng.choice(candidates, size=n, replace=False))

        changes: List[ChangedObject] = []
        for obj in targets:
            kind = self._choose_kind(obj)
            if kind == "rotate":
                if self._rotate(obj):
                    changes.append(ChangedObject(name=obj.name, ori=True))
                elif self._move(r, obj):
                    changes.append(ChangedObject(name=obj.name, pos=True))
            else:  # move
                if self._move(r, obj):
                    changes.append(ChangedObject(name=obj.name, pos=True))
                elif self._rotate(obj):
                    changes.append(ChangedObject(name=obj.name, ori=True))

        # Rebuild to refresh object_map / membership derived from positions.
        return (Room.from_dict(r.to_dict()), changes)

    def _choose_kind(self, obj: Object) -> str:
        if self.modification_type in ("move", "rotate"):
            if self.modification_type == "rotate" and not getattr(obj, "has_orientation", False):
                return "move"
            return self.modification_type
        # random, but avoid rotate for non-orientable objects
        return "rotate" if (getattr(obj, "has_orientation", False) and bool(self.rng.integers(0, 2))) else "move"

    def _move(self, room: Room, obj: Object) -> bool:
        if getattr(room, "mask", None) is None:
            return False
        mask = room.mask
        valid = np.argwhere((mask >= 1) & (mask < 100))
        if valid.size == 0:
            return False

        cur = (int(obj.pos[0]), int(obj.pos[1]))
        occupied = {(int(o.pos[0]), int(o.pos[1])) for o in room.all_objects if o.name != obj.name}
        candidates = [tuple(map(int, p)) for p in valid if tuple(map(int, p)) not in occupied and tuple(map(int, p)) != cur]
        if not candidates:
            return False

        new_pos = candidates[int(self.rng.integers(0, len(candidates)))]
        obj.pos = np.array(new_pos, dtype=int)
        return True

    def _rotate(self, obj: Object) -> bool:
        if not getattr(obj, "has_orientation", False):
            return False
        cur = tuple(int(x) for x in getattr(obj, "ori", np.array([0, 1])))
        opts = [r for r in self._ROTATIONS if tuple(int(x) for x in r) != cur]
        if not opts:
            return False
        obj.ori = opts[int(self.rng.integers(0, len(opts)))]
        return True


__all__ = ["ChangedObject", "RoomModifier", "ObjectModifier"]


if __name__ == "__main__":
    from .eval_utilities import create_and_plot_room
    from .room_utils import RoomPlotter

    seed = 0
    room, agent, _ = create_and_plot_room(seed=seed, plot=True)
    modified_room, changes = ObjectModifier(seed=seed, n_changes=3).modify(room)
    assert all((c.pos ^ c.ori) for c in changes), "Each object must change position OR orientation (not both)."
    print("Changes:", [c.to_dict() for c in changes])
    RoomPlotter.plot(modified_room, agent, mode="img", save_path=f"room_{seed}_modified.png")

