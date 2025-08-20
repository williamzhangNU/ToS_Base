from enum import Enum, auto
from typing import Union, Optional, ClassVar, Tuple
import numpy as np
from dataclasses import dataclass


"""Relationship primitives for spatial reasoning."""


class Dir(Enum):
    """2D directions."""
    SAME = auto()
    FORWARD = auto()   # +y
    BACKWARD = auto()  # -y
    RIGHT = auto()     # +x
    LEFT = auto()      # -x
    UNKNOWN = auto()

    @classmethod
    def from_vertical_delta(cls, d: float) -> 'Dir':
        return cls.SAME if abs(d) < 1e-6 else (cls.FORWARD if d > 0 else cls.BACKWARD)

    @classmethod
    def from_horizontal_delta(cls, d: float) -> 'Dir':
        return cls.SAME if abs(d) < 1e-6 else (cls.RIGHT if d > 0 else cls.LEFT)


@dataclass()
class DirPair:
    """(horiz, vert) pair."""
    horiz: Dir
    vert: Dir

    def __post_init__(self):
        if self.horiz not in (Dir.SAME, Dir.RIGHT, Dir.LEFT, Dir.UNKNOWN):
            raise ValueError(f"Invalid horizontal direction: {self.horiz}")
        if self.vert not in (Dir.SAME, Dir.FORWARD, Dir.BACKWARD, Dir.UNKNOWN):
            raise ValueError(f"Invalid vertical direction: {self.vert}")

    def __getitem__(self, i) -> Dir:
        if i == 0: return self.horiz
        if i == 1: return self.vert
        raise IndexError(i)


@dataclass()
class DirectionRel:
    """Direction (DirPair) + bearing degree."""
    pair: DirPair
    degree: float = 0.0  # +: right/east, -: left/west, 0: front/north

    # Field of view
    FIELD_OF_VIEW: ClassVar[float] = 90.0

    # Perspective labels
    EGO: ClassVar[dict[Dir, str]] = {Dir.SAME: 'same', Dir.FORWARD: 'front', Dir.BACKWARD: 'back', Dir.RIGHT: 'right', Dir.LEFT: 'left', Dir.UNKNOWN: 'unknown'}
    ALLO: ClassVar[dict[Dir, str]] = {Dir.SAME: 'same', Dir.FORWARD: 'north', Dir.BACKWARD: 'south', Dir.RIGHT: 'east', Dir.LEFT: 'west', Dir.UNKNOWN: 'unknown'}
    
    # Transformations
    TRANSFORMS: ClassVar[dict[tuple[int, int], dict[Dir, Dir]]] = {
        (0, 1): {d: d for d in Dir},
        (1, 0): {Dir.FORWARD: Dir.LEFT, Dir.BACKWARD: Dir.RIGHT, Dir.RIGHT: Dir.FORWARD, Dir.LEFT: Dir.BACKWARD, Dir.SAME: Dir.SAME, Dir.UNKNOWN: Dir.UNKNOWN},
        (0, -1): {Dir.FORWARD: Dir.BACKWARD, Dir.BACKWARD: Dir.FORWARD, Dir.RIGHT: Dir.LEFT, Dir.LEFT: Dir.RIGHT, Dir.SAME: Dir.SAME, Dir.UNKNOWN: Dir.UNKNOWN},
        (-1, 0): {Dir.FORWARD: Dir.RIGHT, Dir.BACKWARD: Dir.LEFT, Dir.RIGHT: Dir.BACKWARD, Dir.LEFT: Dir.FORWARD, Dir.SAME: Dir.SAME, Dir.UNKNOWN: Dir.UNKNOWN},
    }


    # ---- prompts ----
    @classmethod
    def prompt(cls) -> str:
        return "Continuous bearing: degree in [-180, 180]."

    # ---- stringify ----
    def to_string(self, perspective: str = 'ego', kind: str = 'relation',
                  gate_dir: 'DirectionRel' = None) -> str:
        assert perspective in ('ego', 'allo')
        assert kind in ('relation', 'orientation')
        if kind == 'relation':
            return f"{self._dir_to_string(self.pair, perspective)}, {self._format_degree(self.degree)}"
        # orientation
        return self._ori_to_string(self.pair, perspective, None if gate_dir is None else gate_dir.pair)

    @classmethod
    def _dir_to_string(cls, d: Union[Dir, DirPair], perspective: str = 'ego') -> str:
        labels = cls.EGO if perspective == 'ego' else cls.ALLO
        if isinstance(d, Dir): return labels[d]
        h, v = d.horiz, d.vert
        if Dir.UNKNOWN in (h, v): return 'unknown'
        if h == v == Dir.SAME: return 'same'
        if h == Dir.SAME or v == Dir.SAME:
            return f"directly {labels[h] if h != Dir.SAME else labels[v]}"
        return f"{labels[v]}-{labels[h]}"

    @classmethod
    def pair_to_string(cls, d: DirPair, perspective: str = 'ego') -> str:
        return cls._dir_to_string(d, perspective)


# RelationTriple defined near the end after all classes

    @classmethod
    def _ori_to_string(cls, o: DirPair, perspective: str = 'ego', gate_dir: DirPair = None) -> str:
        # special case for gates
        if gate_dir is not None:
            if o.vert != Dir.SAME and o.horiz == Dir.SAME:
                side = {Dir.FORWARD: 'front', Dir.BACKWARD: 'back'}.get(gate_dir.vert)
                if side: return f"gate at {side} wall"
            if o.horiz != Dir.SAME and o.vert == Dir.SAME:
                side = {Dir.RIGHT: 'right', Dir.LEFT: 'left'}.get(gate_dir.horiz)
                if side: return f"gate at {side} wall"
        # object facing
        if o.horiz != Dir.SAME and o.vert == Dir.SAME: primary = o.horiz
        elif o.vert != Dir.SAME and o.horiz == Dir.SAME: primary = o.vert
        else: raise ValueError(f"Invalid orientation: {o}")
        mapping = ( {Dir.FORWARD:'forward', Dir.BACKWARD:'backward', Dir.RIGHT:'right', Dir.LEFT:'left'}
                    if perspective=='ego' else
                    {Dir.FORWARD:'north', Dir.BACKWARD:'south', Dir.RIGHT:'east', Dir.LEFT:'west'} )
        return mapping.get(primary, 'unknown')

    @staticmethod
    def _format_degree(v: float) -> str:
        if abs(v) < 1e-12: return "0°"
        sign = '+' if v > 0 else '-'
        return f"{sign}{abs(v):.0f}°"

    # ---- geometry / transforms ----
    @classmethod
    def transform(cls, pair: DirPair, orientation: tuple) -> DirPair:
        ori = tuple(map(int, orientation))
        if ori not in cls.TRANSFORMS: raise ValueError(f"Invalid orientation: {ori}")
        t = cls.TRANSFORMS[ori]
        h, v = t[pair.horiz], t[pair.vert]
        return DirPair(v, h) if (h in (Dir.FORWARD, Dir.BACKWARD)) or (v in (Dir.RIGHT, Dir.LEFT)) else DirPair(h, v)

    @classmethod
    def get_direction(cls, pos1: tuple, pos2: tuple, orientation: tuple = None) -> 'DirectionRel':
        p1, p2 = np.array(pos1), np.array(pos2)
        diff = p1 - p2
        pair = DirPair(Dir.from_horizontal_delta(diff[0]), Dir.from_vertical_delta(diff[1]))
        pair = cls.transform(pair, orientation) if orientation is not None else pair
        # degree relative to orientation (default (0,1))
        anchor = np.array((0,1) if orientation is None else orientation, dtype=float)
        anchor = anchor / (np.linalg.norm(anchor) or 1.0)
        v = diff.astype(float)
        v = v / (np.linalg.norm(v) or 1.0)
        deg = 0.0 if np.linalg.norm(v) < 1e-12 else float(-np.degrees(np.arctan2(anchor[0]*v[1]-anchor[1]*v[0], np.dot(anchor, v))))
        return cls(pair, deg)

    @classmethod
    def get_relative_orientation(cls, obj_ori: tuple, anchor_ori: tuple) -> DirPair:
        base = cls.get_direction(obj_ori, (0, 0)).pair
        return cls.transform(base, anchor_ori)

    @classmethod
    def from_positions(cls, pos1: tuple, pos2: tuple, anchor_ori: tuple = (0, 1)) -> 'DirectionRel':
        return cls.get_direction(pos1, pos2, anchor_ori)


@dataclass()
class DistanceRel:
    value: float

    def to_string(self, perspective: str = 'ego') -> str:
        d = float(self.value)
        return f"{d:.2f}"

    @classmethod
    def prompt(cls) -> str:
        return "Continuous distance: real-valued Euclidean length."

    @classmethod
    def get_distance(cls, pos1: tuple, pos2: tuple) -> 'DistanceRel':
        p1 = np.array(pos1, float)
        p2 = np.array(pos2, float)
        return cls(float(np.linalg.norm(p1 - p2)))


@dataclass()
class DirectionRelDiscrete(DirectionRel):
    """Discrete bins for direction (inherits continuous, overrides discrete-only API)."""
    # Degree bins
    DEGREE_BINS: ClassVar[list[tuple[float, float]]] = [(0, 15), (15, 30), (30, 45), (45, 180)]
    DEGREE_BIN_LABELS: ClassVar[dict[str, list[str] | str]] = {
        'pos': ['slight-right', 'right', 'sharp-right', 'beyond-fov'],
        'neg': ['slight-left', 'left', 'sharp-left', 'beyond-fov'],
        'zero': 'directly front',
    }

    @classmethod
    def bin_bearing(cls, degree: float) -> Tuple[int, str, str]:
        v = float(degree)
        if abs(v) < 1e-6:
            return -1, str(cls.DEGREE_BIN_LABELS['zero']), 'zero'
        side = 'neg' if v < 0 else 'pos'
        a = abs(v)
        for i, (lo, hi) in enumerate(cls.DEGREE_BINS):
            if lo < a <= hi:
                return i, cls.DEGREE_BIN_LABELS[side][i], side  # type: ignore[index]
        return len(cls.DEGREE_BINS) - 1, cls.DEGREE_BIN_LABELS[side][-1], side  # type: ignore[index]

    @classmethod
    def from_relation(cls, rel: DirectionRel) -> 'DirectionRelDiscrete':
        return cls(pair=rel.pair, degree=rel.degree)  # metadata is derivable via binning

    @classmethod
    def prompt(cls) -> str:
        parts = [f"(0→{cls.DEGREE_BIN_LABELS['zero']})"]
        parts += [f"({lo},{hi}]→{cls.DEGREE_BIN_LABELS['neg'][i]}/{cls.DEGREE_BIN_LABELS['pos'][i]}"
                  for i, (lo, hi) in enumerate(cls.DEGREE_BINS)]
        return "Bearing bins: " + ", ".join(parts) + "."

    def to_string(self, perspective: str = 'ego', kind: str = 'relation') -> str:
        assert kind == 'relation' and perspective == 'ego'
        # compute current bin on demand
        _, label, _ = self.bin_bearing(self.degree)
        return label


@dataclass()
class DistanceRelDiscrete(DistanceRel):
    """Discrete bins for distance (inherits continuous)."""

    # Distance bins moved here from DistanceRel
    DISTANCE_BINS: ClassVar[list[tuple[float, float]]] = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0)]  # (lo, hi]
    DISTANCE_BIN_LABELS: ClassVar[list[str]] = ['near', 'mid distance', 'far', 'very far']
    DIST_ZERO_LABEL: ClassVar[str] = 'same distance'

    @classmethod
    def bin_distance(cls, value: float) -> Tuple[int, str]:
        d = float(value)
        if d <= 1e-6: return -1,cls.DIST_ZERO_LABEL
        for i, (lo, hi) in enumerate(cls.DISTANCE_BINS):
            if lo < d <= hi: return i, cls.DISTANCE_BIN_LABELS[i]
        raise ValueError(f"Invalid distance: {d}")

    @classmethod
    def from_value(cls, value: float) -> 'DistanceRelDiscrete':
        return cls(float(value))

    @classmethod
    def prompt(cls) -> str:
        parts = [f"=0→{cls.DIST_ZERO_LABEL}"] + [f"({lo},{hi}]→{label}"
                 for (lo, hi), label in zip(cls.DISTANCE_BINS, cls.DISTANCE_BIN_LABELS)]
        return "Distance bins: " + ", ".join(parts) + "."

    def to_string(self, perspective: str = 'ego') -> str:
        bin_idx, label = self.bin_distance(self.value)
        return label


@dataclass(frozen=False)
class PairwiseRelationship:
    direction: Optional[DirectionRel] = None
    dist: Optional[DistanceRel] = None

    @property
    def bearing(self) -> DirectionRel:
        return self.direction  # type: ignore[return-value]

    # --- external API helpers (hide DirectionRel/DistanceRel) ---
    @property
    def dir_pair(self) -> Optional[DirPair]:
        return None if self.direction is None else self.direction.pair

    @property
    def degree(self) -> float:
        return 0.0 if self.direction is None else float(self.direction.degree)

    @property
    def distance_value(self) -> float:
        return 0.0 if self.dist is None else float(self.dist.value)

    @staticmethod
    def pair_to_string(d: DirPair, perspective: str = 'ego') -> str:
        return DirectionRel._dir_to_string(d, perspective)

    @staticmethod
    def format_degree(v: float) -> str:
        return DirectionRel._format_degree(v)

    @staticmethod
    def distance_to_string(value: float, perspective: str = 'ego') -> str:
        return DistanceRel(float(value)).to_string(perspective)

    @staticmethod
    def rotate_pair_90(p: DirPair) -> DirPair:
        t = DirectionRel.TRANSFORMS[(-1, 0)]
        h, v = t[p.horiz], t[p.vert]
        if h in (Dir.FORWARD, Dir.BACKWARD) or v in (Dir.RIGHT, Dir.LEFT):
            h, v = v, h
        return DirPair(h, v)

    def to_string(self, perspective: str = 'ego') -> str:
        if self.direction is None and self.dist is None:
            return ""
        if self.direction is None:
            return self.distance_to_string(self.distance_value, perspective)
        if self.dist is None:
            return self.direction.to_string(perspective, 'relation')
        return f"{self.direction.to_string(perspective, 'relation')}, {self.dist.to_string(perspective)}"

    @classmethod
    def relationship(cls, pos1: tuple, pos2: tuple, anchor_ori: Optional[tuple] = None, full: bool = True) -> 'PairwiseRelationship':
        d = DirectionRel.get_direction(pos1, pos2, (anchor_ori if anchor_ori is not None else (0, 1)))
        if not full:
            return cls(direction=d, dist=None)
        return cls(direction=d, dist=DistanceRel.get_distance(pos1, pos2))


    @classmethod
    def get_direction(cls, pos1: tuple, pos2: tuple, anchor_ori: Optional[tuple] = None) -> DirectionRel:
        return DirectionRel.get_direction(pos1, pos2, anchor_ori)

    @classmethod
    def get_orientation(cls, obj_ori: tuple, anchor_ori: tuple) -> DirectionRel:
        dp = DirectionRel.get_relative_orientation(tuple(obj_ori), tuple(anchor_ori))
        return DirectionRel(dp, 0.0)

    @classmethod
    def get_bearing_degree(cls, pos1: tuple, pos2: tuple, anchor_ori: tuple = (0, 1)) -> float:
        return DirectionRel.from_positions(pos1, pos2, anchor_ori).degree

    @classmethod
    def get_distance(cls, pos1: tuple, pos2: tuple) -> DistanceRel:
        return DistanceRel.get_distance(pos1, pos2)

    @classmethod
    def prompt(cls) -> str:
        return "Relationship reporting: precise direction (pair, degree), distance."


@dataclass()
class PairwiseRelationshipDiscrete(PairwiseRelationship):
    direction: DirectionRelDiscrete  # type: ignore[assignment]
    dist: DistanceRelDiscrete  # type: ignore[assignment]

    @classmethod
    def prompt(cls) -> str:
        return (
            "Discrete relationship reporting:\n"
            f"- {DirectionRelDiscrete.prompt()}\n"
            f"- {DistanceRelDiscrete.prompt()}"
        )
    
    @classmethod
    def relationship(cls, pos1: tuple, pos2: tuple, anchor_ori: Optional[tuple] = None) -> 'PairwiseRelationshipDiscrete':
        rel = PairwiseRelationship.relationship(pos1, pos2, anchor_ori=anchor_ori, full=True)
        d = DirectionRelDiscrete.from_relation(rel.direction)
        s = DistanceRelDiscrete.from_value(rel.dist.value if rel.dist else 0.0)
        return cls(direction=d, dist=s)

    def to_string(self, perspective: str = 'ego') -> str:
        return f"{self.direction.to_string(perspective, 'relation')}, {self.dist.to_string(perspective)}"
    
    


@dataclass()
class LocalRelationship:
    side: Optional[str]
    proximity: Optional[str]

    @classmethod
    def from_positions(cls, a_pos: tuple, b_pos: tuple, anchor_pos: tuple, anchor_ori: tuple) -> 'LocalRelationship':

        def _norm_angle(x: float) -> float:
            return (x + 180.0) % 360.0 - 180.0
        
        rel_a = PairwiseRelationshipDiscrete.relationship(tuple(a_pos), tuple(anchor_pos), anchor_ori=tuple(anchor_ori))
        rel_b = PairwiseRelationshipDiscrete.relationship(tuple(b_pos), tuple(anchor_pos), anchor_ori=tuple(anchor_ori))
        
        # Check if direction or distance bins are the same (indicating locality)
        i_a, _, s_a = DirectionRelDiscrete.bin_bearing(rel_a.direction.degree)
        i_b, _, s_b = DirectionRelDiscrete.bin_bearing(rel_b.direction.degree)
        dir_same = (i_a == i_b) and (s_a == s_b)
        j_a, _ = DistanceRelDiscrete.bin_distance(rel_a.dist.value)
        j_b, _ = DistanceRelDiscrete.bin_distance(rel_b.dist.value)
        dist_same = (j_a == j_b)

        if not dir_same and not dist_same:
            return None

        a_bearing = PairwiseRelationship.get_bearing_degree(a_pos, anchor_pos, anchor_ori)
        b_bearing = PairwiseRelationship.get_bearing_degree(b_pos, anchor_pos, anchor_ori)
        a_dist = PairwiseRelationship.get_distance(a_pos, anchor_pos).value
        b_dist = PairwiseRelationship.get_distance(b_pos, anchor_pos).value
        side = 'right' if _norm_angle(a_bearing - b_bearing) > 0 else 'left' if _norm_angle(a_bearing - b_bearing) < 0 else 'same direction'
        prox = 'closer' if a_dist < b_dist else 'farther' if a_dist > b_dist else 'same distance'
        return cls(side=side, proximity=prox)

    def to_string(self, a_name: str, b_name: str) -> str:
        return (f"{a_name} is {self.side} of {b_name}, {self.proximity} to agent")



@dataclass()
class RelationTriple:
    """A -> B with relation, optional anchor name and orientation."""
    obj_a: str
    obj_b: str
    relation: Union[PairwiseRelationship, LocalRelationship]
    anchor_name: Optional[str] = None
    anchor_ori: Optional[tuple] = None




def relationship_applies(obj1, obj2, relationship, anchor_pos: Optional[tuple] = None, anchor_ori: tuple = (0, 1)) -> bool:
    """Check if a relationship applies. Only Pairwise(Discrete) or LocalRelationship allowed."""
    p1 = getattr(obj1, 'pos', obj1)
    p2 = getattr(obj2, 'pos', obj2)

    if isinstance(relationship, PairwiseRelationshipDiscrete):
        rel = PairwiseRelationshipDiscrete.relationship(tuple(p1), tuple(p2), anchor_ori=anchor_ori)
        i1, _, s1 = DirectionRelDiscrete.bin_bearing(relationship.degree)
        i2, _, s2 = DirectionRelDiscrete.bin_bearing(rel.degree)
        j1, _ = DistanceRelDiscrete.bin_distance(relationship.distance_value)
        j2, _ = DistanceRelDiscrete.bin_distance(rel.distance_value)
        return i1 == i2 and s1 == s2 and j1 == j2
    

    if isinstance(relationship, PairwiseRelationship):
        cur = PairwiseRelationship.relationship(tuple(p1), tuple(p2), anchor_ori=anchor_ori, full=True)
        # direction-only
        if relationship.dist is None and relationship.direction is not None:
            same_pair = cur.dir_pair == relationship.dir_pair
            deg_close = abs(cur.degree - relationship.degree) <= 1e-6
            return same_pair and deg_close
        # distance-only
        if relationship.direction is None and relationship.dist is not None:
            return abs(cur.distance_value - relationship.distance_value) <= 1e-6
        # full
        same_pair = cur.dir_pair == relationship.dir_pair
        deg_close = abs(cur.degree - relationship.degree) <= 1e-6
        dist_close = abs(cur.distance_value - relationship.distance_value) <= 1e-6
        return same_pair and deg_close and dist_close


    if isinstance(relationship, LocalRelationship):
        assert anchor_pos is not None, "anchor_pos required for LocalRelationship check"
        cur = LocalRelationship.from_positions(tuple(p1), tuple(p2), tuple(anchor_pos), tuple(anchor_ori))
        return (cur is not None) and (cur.side == relationship.side) and (cur.proximity == relationship.proximity)

    raise ValueError(f"Invalid relationship type: {type(relationship)}")


if __name__ == "__main__":
    print(PairwiseRelationshipDiscrete.prompt())


