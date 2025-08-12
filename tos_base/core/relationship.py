from enum import Enum, auto
from typing import Union, Optional
import numpy as np
from dataclasses import dataclass


class Dir(Enum):
    """
    Represents directional relationships in 2D space.
    
    Can be mapped to specific interpretations (ego/allocentric) based on perspective.
    """
    SAME = auto()
    FORWARD = auto()  # Vertical positive
    BACKWARD = auto()  # Vertical negative  
    RIGHT = auto()     # Horizontal positive
    LEFT = auto()      # Horizontal negative
    UNKNOWN = auto()
    
    @classmethod
    def from_delta(cls, delta: float) -> 'Dir':
        """Create direction from a numerical delta."""
        if abs(delta) < 1e-6:
            return cls.SAME
        return cls.FORWARD if delta > 0 else cls.BACKWARD
    
    @classmethod
    def from_horizontal_delta(cls, delta: float) -> 'Dir':
        """Create horizontal direction from delta."""
        if abs(delta) < 1e-6:
            return cls.SAME
        return cls.RIGHT if delta > 0 else cls.LEFT


@dataclass(frozen=True)
class DirPair:
    """Pair of directions representing horizontal and vertical relationships."""
    horiz: Dir  # Horizontal relation
    vert: Dir   # Vertical relation
    
    def __post_init__(self):
        """Validate direction values."""
        h_valid = self.horiz in (Dir.SAME, Dir.RIGHT, Dir.LEFT, Dir.UNKNOWN)
        v_valid = self.vert in (Dir.SAME, Dir.FORWARD, Dir.BACKWARD, Dir.UNKNOWN)
        
        if not h_valid:
            raise ValueError(f"Invalid horizontal direction: {self.horiz}")
        if not v_valid:
            raise ValueError(f"Invalid vertical direction: {self.vert}")
    
    def __getitem__(self, i) -> Dir:
        if i == 0:
            return self.horiz
        elif i == 1:
            return self.vert
        raise IndexError(f"Index {i} out of range")


@dataclass(frozen=True)
class DirectionRel:
    pair: DirPair

    # Perspective mappings
    EGO_LABELS = {
        Dir.SAME: 'same',
        Dir.FORWARD: 'front',
        Dir.BACKWARD: 'back',
        Dir.RIGHT: 'right',
        Dir.LEFT: 'left',
        Dir.UNKNOWN: 'unknown',
    }
    ALLO_LABELS = {
        Dir.SAME: 'same',
        Dir.FORWARD: 'north', 
        Dir.BACKWARD: 'south',
        Dir.RIGHT: 'east',
        Dir.LEFT: 'west',
        Dir.UNKNOWN: 'unknown',
    }
    TRANSFORMS = {
        (0, 1): {dir_: dir_ for dir_ in Dir},
        (1, 0): {Dir.FORWARD: Dir.LEFT, Dir.BACKWARD: Dir.RIGHT, Dir.RIGHT: Dir.FORWARD, Dir.LEFT: Dir.BACKWARD, Dir.SAME: Dir.SAME, Dir.UNKNOWN: Dir.UNKNOWN},
        (0, -1): {Dir.FORWARD: Dir.BACKWARD, Dir.BACKWARD: Dir.FORWARD, Dir.RIGHT: Dir.LEFT, Dir.LEFT: Dir.RIGHT, Dir.SAME: Dir.SAME, Dir.UNKNOWN: Dir.UNKNOWN},
        (-1, 0): {Dir.FORWARD: Dir.RIGHT, Dir.BACKWARD: Dir.LEFT, Dir.RIGHT: Dir.BACKWARD, Dir.LEFT: Dir.FORWARD, Dir.SAME: Dir.SAME, Dir.UNKNOWN: Dir.UNKNOWN},
    }

    def to_string(self, perspective: str = 'ego') -> str:
        return self.pair_to_string(self.pair, perspective)

    @classmethod
    def pair_to_string(cls, direction: Union[Dir, DirPair], perspective: str = 'ego') -> str:
        assert perspective in ('ego', 'allo'), f"Invalid perspective: {perspective}"
        labels = cls.EGO_LABELS if perspective == 'ego' else cls.ALLO_LABELS
        if isinstance(direction, Dir):
            return labels[direction]
        h_str = labels[direction.horiz]
        v_str = labels[direction.vert]
        if direction.horiz == Dir.UNKNOWN or direction.vert == Dir.UNKNOWN:
            return 'unknown'
        if direction.horiz == Dir.SAME and direction.vert == Dir.SAME:
            return 'same'
        if direction.horiz == Dir.SAME:
            return v_str
        if direction.vert == Dir.SAME:
            return h_str
        return f"{h_str}-{v_str}"

    @classmethod
    def transform(cls, dir_pair: DirPair, orientation: tuple) -> DirPair:
        ori_tuple = tuple(map(int, orientation))
        if ori_tuple not in cls.TRANSFORMS:
            raise ValueError(f"Invalid orientation: {ori_tuple}")
        t = cls.TRANSFORMS[ori_tuple]
        h = t[dir_pair.horiz]
        v = t[dir_pair.vert]
        if (h in (Dir.FORWARD, Dir.BACKWARD)) or (v in (Dir.RIGHT, Dir.LEFT)):
            return DirPair(v, h)
        return DirPair(h, v)

    @classmethod
    def get_direction(cls, pos1: tuple, pos2: tuple, orientation: tuple = None) -> 'DirectionRel':
        p1 = np.array(pos1)
        p2 = np.array(pos2)
        diff = p1 - p2
        h_dir = Dir.from_horizontal_delta(diff[0])
        v_dir = Dir.from_delta(diff[1])
        pair = DirPair(h_dir, v_dir)
        pair = cls.transform(pair, orientation) if orientation is not None else pair
        return DirectionRel(pair)

    @classmethod
    def get_relative_orientation(cls, obj_ori: tuple, anchor_ori: tuple) -> DirPair:
        base_dir = cls.get_direction(obj_ori, (0, 0)).pair
        return cls.transform(base_dir, anchor_ori)


@dataclass(frozen=True)
class DegreeRel:
    value: float
    def to_string(self, perspective: str = 'ego') -> str:
        if perspective == 'ego':
            direction = "right" if self.value > 0 else "left"
            return f"{abs(self.value):.1f} degrees to the {direction}"
        else:
            # allo, default facing north
            direction = "east" if self.value > 0 else "west"
            return f"{abs(self.value):.1f} degrees to the {direction}"
    
    @classmethod
    def get_degree(cls, pos1: tuple, pos2: tuple, anchor_ori: tuple = (0, 1)) -> 'DegreeRel':
        """Signed angle deg from anchor_ori to (pos1-pos2), clockwise/right positive.
        By default, anchor_ori is (0, 1) facing north.
        """
        p1 = np.array(pos1, dtype=float)
        p2 = np.array(pos2, dtype=float)
        v = p1 - p2
        if np.linalg.norm(v) < 1e-12:
            return DegreeRel(0.0)
        v = v / np.linalg.norm(v)
        r = np.array(anchor_ori, dtype=float)
        r = r / (np.linalg.norm(r) or 1.0)
        dot = float(np.clip(np.dot(r, v), -1.0, 1.0))
        cross_z = float(r[0] * v[1] - r[1] * v[0])
        ang_ccw = np.degrees(np.arctan2(cross_z, dot))
        return DegreeRel(float(-ang_ccw))


@dataclass(frozen=True)
class DistanceRel:
    value: float
    def to_string(self, perspective: str = 'ego') -> str:
        if perspective == 'ego':
            return f"{self.value:.2f} away"
        return f"{self.value:.2f} away"
    
    @classmethod
    def get_distance(cls, pos1: tuple, pos2: tuple) -> 'DistanceRel':
        p1 = np.array(pos1, dtype=float)
        p2 = np.array(pos2, dtype=float)
        return DistanceRel(float(np.linalg.norm(p1 - p2)))


@dataclass(frozen=True)
class TotalRelationship:
    dir: DirectionRel
    deg: Optional[DegreeRel] = None
    dist: Optional[DistanceRel] = None

    def to_string(self, perspective: str = 'ego') -> str:
        if self.deg is None or self.dist is None:
            return self.dir.to_string(perspective)
        return f"({self.dir.to_string(perspective)}, {self.deg.to_string(perspective)}, {self.dist.to_string(perspective)})"

    @classmethod
    def relationship(
        cls,
        pos1: tuple,
        pos2: tuple,
        anchor_ori: Optional[tuple] = None,
        full: bool = False,
    ) -> 'TotalRelationship':
        dir_rel = DirectionRel.get_direction(pos1, pos2, anchor_ori)
        if not full:
            return dir_rel
        deg_v = DegreeRel.get_degree(pos1, pos2, (anchor_ori if anchor_ori is not None else (0, 1)))
        dist_v = DistanceRel.get_distance(pos1, pos2)
        return cls(dir=dir_rel, deg=deg_v, dist=dist_v)

    # ---- Unified interface helpers ----
    @classmethod
    def get_direction(
        cls,
        pos1: tuple,
        pos2: tuple,
        anchor_ori: Optional[tuple] = None,
    ) -> DirectionRel:
        """Get directional relation between two positions.
        
        Optionally provide anchor_ori to transform according to the anchor orientation.
        """
        return DirectionRel.get_direction(pos1, pos2, anchor_ori)

    @classmethod
    def get_orientation(
        cls,
        obj_ori: tuple,
        anchor_ori: tuple,
    ) -> tuple[DirPair, str]:
        """Get relative orientation of an object with respect to an anchor orientation.

        Returns DirPair and orientation string among {'forward','backward','right','left'}.
        """
        dir_pair = DirectionRel.get_relative_orientation(tuple(obj_ori), tuple(anchor_ori))
        mapping = {
            DirPair(Dir.SAME, Dir.FORWARD): 'forward',
            DirPair(Dir.SAME, Dir.BACKWARD): 'backward',
            DirPair(Dir.RIGHT, Dir.SAME): 'right',
            DirPair(Dir.LEFT, Dir.SAME): 'left',
        }
        return dir_pair, mapping[dir_pair]

    @classmethod
    def get_degree(cls, pos1: tuple, pos2: tuple, anchor_ori: tuple = (0, 1)) -> DegreeRel:
        return DegreeRel.get_degree(pos1, pos2, anchor_ori)

    @classmethod
    def get_distance(cls, pos1: tuple, pos2: tuple) -> DistanceRel:
        return DistanceRel.get_distance(pos1, pos2)
