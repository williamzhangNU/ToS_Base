from enum import Enum, auto
from typing import Union, Optional, ClassVar, Tuple, Protocol
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


"""Relationship primitives for spatial reasoning."""


# ---- Bin System Protocol ----
class BinSystem(Protocol):
    """Angle binning system without perspective argument.

    Implementations should provide class attributes:
    - BINS: list of (lo_deg, hi_deg, label)
    - LABELS: list of labels (same order as BINS)
    """
    BINS: ClassVar[list]
    LABELS: ClassVar[list]

    @abstractmethod
    def bin(self, degree: float) -> Tuple[int, str]:
        """Return (bin_id, bin_label) for given degree."""
        pass
    
    @classmethod
    @abstractmethod
    def prompt(cls) -> str:
        """Return description of the bin system."""
        pass


class DistanceBinSystem(Protocol):
    """Distance binning system with unified attributes."""
    BINS: ClassVar[list]
    LABELS: ClassVar[list]

    @abstractmethod
    def bin(self, value: float) -> Tuple[int, str]:
        """Return (bin_id, bin_label) for given distance value."""
        pass
    
    @classmethod
    @abstractmethod
    def prompt(cls) -> str:
        """Return description of the distance bin system."""
        pass


class EgoFrontBins:
    """Ego-centric front-focused bins."""
    BINS = [
        (-5, 5),
        (5, 25),  
        (25, 45),
        (-25, -5),
        (-45, -25),
        (45, 180),
        (-180, -45),
    ]
    LABELS = ['front', 'front-slight-right', 'front-right', 'front-slight-left', 'front-left', 'beyond-fov', 'beyond-fov']
    
    def bin(self, degree: float) -> Tuple[int, str]:
        v = float(degree)
        for i, (lo, hi) in enumerate(self.BINS):
            if lo <= v <= hi:
                return i, self.LABELS[i]
        return len(self.LABELS) - 1, self.LABELS[-1]
    
    @classmethod
    def prompt(cls) -> str:
        parts = []
        for (lo, hi), label in zip(cls.BINS, cls.LABELS):
            parts.append(f"[{lo}°,{hi}°]→{label}" if lo == -5 else f"({lo}°,{hi}°]→{label}")
        return f"Bearing bins (egocentric): {', '.join(parts)}."


class _CardinalBinsBase:
    """Cardinal direction bins (8 directions) without perspective args."""
    LABELS: ClassVar[list] = []
    # Boundaries: [-22.5, +22.5], [22.5, 67.5], ..., [292.5, 337.5]
    BINS: ClassVar[list] = [
        (-22.5, 22.5), (22.5, 67.5), (67.5, 112.5), (112.5, 157.5),
        (157.5, 202.5), (202.5, 247.5), (247.5, 292.5), (292.5, 337.5),
    ]

    def bin(self, degree: float) -> Tuple[int, str]:
        v = float(degree)
        w = (v + 360.0) % 360.0
        idx = int(((w + 22.5) // 45.0) % 8)
        return idx, self.LABELS[idx]

    @classmethod
    def prompt(cls) -> str:
        parts = [f"[{lo}°,{hi}°]→{label}" for (lo, hi), label in zip(cls.BINS, cls.LABELS)]
        return f"Bearing bins: {', '.join(parts)}."


class CardinalBinsAllo(_CardinalBinsBase):
    LABELS = ['north', 'north east', 'east', 'south east', 'south', 'south west', 'west', 'north west']


class CardinalBinsEgo(_CardinalBinsBase):
    LABELS = [
        "around 12 o'clock",  # front
        'around 1:30',         # front-right
        "around 3 o'clock",   # right
        'around 4:30',         # back-right
        "around 6 o'clock",   # back
        'around 7:30',         # back-left
        "around 9 o'clock",   # left
        'around 10:30'         # front-left
    ]


class StandardDistanceBins:
    """Standard distance bins."""
    BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 32.0), (32.0, 64.0)]
    ZERO_LABEL = 'same distance'
    LABELS = ['very near', 'near', 'mid distance', 'slightly far', 'far', 'very far', 'extremely far']
    
    def bin(self, value: float) -> Tuple[int, str]:
        d = float(value)
        if d <= 1e-6:
            return -1, self.ZERO_LABEL
        for i, (lo, hi) in enumerate(self.BINS):
            if lo < d <= hi:
                return i, self.LABELS[i]
        raise ValueError(f"Invalid distance: {d}")
    
    @classmethod
    def prompt(cls) -> str:
        parts = [f"=0→{cls.ZERO_LABEL}"] + [f"({lo},{hi}]→{label}"
                 for (lo, hi), label in zip(cls.BINS, cls.LABELS)]
        return f"Distance bins: {', '.join(parts)}."


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


@dataclass(frozen=True)
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
    
    def __eq__(self, other):
        if type(other) is not DirectionRel:
            return False
        return self.pair == other.pair and abs(self.degree - other.degree) < 1e-6
    
    def __hash__(self):
        # Round degree to 6 decimal places for consistent hashing
        return hash((self.pair, round(self.degree, 6)))

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
    def pair_to_string(cls, d: DirPair, perspective: str = 'ego') -> str:
        return cls._dir_to_string(d, perspective)

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
    def _ori_to_string(cls, o: DirPair, perspective: str = 'ego', gate_dir: DirPair = None) -> str:
        # special case for gates
        if gate_dir is not None:
            if o.vert != Dir.SAME and o.horiz == Dir.SAME:
                side = {Dir.FORWARD: 'front', Dir.BACKWARD: 'back'}.get(gate_dir.vert)
                if side: return f"on {side} wall"
            if o.horiz != Dir.SAME and o.vert == Dir.SAME:
                side = {Dir.RIGHT: 'right', Dir.LEFT: 'left'}.get(gate_dir.horiz)
                if side: return f"on {side} wall"
        # object facing
        if o.horiz != Dir.SAME and o.vert == Dir.SAME: primary = o.horiz
        elif o.vert != Dir.SAME and o.horiz == Dir.SAME: primary = o.vert
        else: raise ValueError(f"Invalid orientation: {o}")
        mapping = ( {Dir.FORWARD:'forward', Dir.BACKWARD:'backward', Dir.RIGHT:'right', Dir.LEFT:'left'}
                    if perspective=='ego' else
                    {Dir.FORWARD:'north', Dir.BACKWARD:'south', Dir.RIGHT:'east', Dir.LEFT:'west'} )
        return f"facing {mapping.get(primary, 'unknown')}"

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
    
    def __eq__(self, other):
        if type(other) is not DistanceRel:
            return False
        return abs(self.value - other.value) < 1e-6
    
    def __hash__(self):
        # Round value to 6 decimal places for consistent hashing
        return hash(round(self.value, 6))

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
class DirectionRelBinned(DirectionRel):
    """Direction with modular bin system."""
    bin_system: BinSystem = None
    bin_id: int = 0
    bin_label: str = ""
    
    def __post_init__(self):
        if self.bin_system is not None:
            self.bin_id, self.bin_label = self.bin_system.bin(self.degree)

    @classmethod
    def from_relation(cls, rel: DirectionRel, bin_system: BinSystem) -> 'DirectionRelBinned':
        bin_id, bin_label = bin_system.bin(rel.degree)
        return cls(pair=rel.pair, degree=rel.degree, bin_system=bin_system, 
                  bin_id=bin_id, bin_label=bin_label)

    def to_string(self) -> str:
        return self.bin_label


@dataclass()
class DistanceRelBinned(DistanceRel):
    """Discrete bins for distance using modular bin system."""
    bin_system: DistanceBinSystem = None
    bin_id: int = 0
    bin_label: str = ""
    
    def __post_init__(self):
        if self.bin_system is not None:
            self.bin_id, self.bin_label = self.bin_system.bin(self.value)

    @classmethod
    def from_value(cls, value: float, bin_system: DistanceBinSystem = None) -> 'DistanceRelBinned':
        bin_system = bin_system or StandardDistanceBins()
        bin_id, bin_label = bin_system.bin(value)
        return cls(value=value, bin_system=bin_system, bin_id=bin_id, bin_label=bin_label)

    @classmethod
    def bin_distance(cls, value: float) -> Tuple[int, str]:
        # For backward compatibility
        bin_system = StandardDistanceBins()
        return bin_system.bin(value)

    def to_string(self) -> str:
        return self.bin_label


@dataclass(frozen=False)
class PairwiseRelationship:
    direction: Optional[DirectionRel] = None
    dist: Optional[DistanceRel] = None
    
    def __eq__(self, other):
        if type(other) is not PairwiseRelationship:
            return False
        return self.direction == other.direction and self.dist == other.dist
    
    def __hash__(self):
        return hash((self.direction, self.dist))

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
    def distance_to_string(value: float) -> str:
        return DistanceRel(float(value)).to_string()

    @staticmethod
    def rotate_pair_90(p: DirPair) -> DirPair:
        t = DirectionRel.TRANSFORMS[(-1, 0)]
        h, v = t[p.horiz], t[p.vert]
        if h in (Dir.FORWARD, Dir.BACKWARD) or v in (Dir.RIGHT, Dir.LEFT):
            h, v = v, h
        return DirPair(h, v)

    def to_string(self) -> str:
        if self.direction is None and self.dist is None:
            return ""
        if self.direction is None:
            return self.distance_to_string(self.distance_value)
        if self.dist is None:
            # default textualization: allocentric
            return self.direction.to_string('allo', 'relation')
        return f"{self.direction.to_string('allo', 'relation')}, {self.dist.to_string()}"

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
    direction: DirectionRelBinned  # type: ignore[assignment]
    dist: DistanceRelBinned  # type: ignore[assignment]
    
    def __eq__(self, other):
        if type(other) is not PairwiseRelationshipDiscrete:
            return False
        return (self.direction.bin_id == other.direction.bin_id and 
                self.dist.bin_id == other.dist.bin_id)
    
    def __hash__(self):
        return hash((self.direction.bin_id, self.dist.bin_id))

    @classmethod
    def prompt(cls, bin_system: BinSystem = None, distance_bin_system: DistanceBinSystem = None) -> str:
        bin_system = bin_system or EgoFrontBins()
        distance_bin_system = distance_bin_system or StandardDistanceBins()
        return (
            f"Discrete relationship reporting:\n"
            f"- {bin_system.prompt()}\n"
            f"- {distance_bin_system.prompt()}"
        )
    
    @classmethod
    def relationship(cls, pos1: tuple, pos2: tuple, anchor_ori: Optional[tuple] = None, 
                    bin_system: BinSystem = None, distance_bin_system: DistanceBinSystem = None) -> 'PairwiseRelationshipDiscrete':
        bin_system = bin_system or EgoFrontBins()
        distance_bin_system = distance_bin_system or StandardDistanceBins()
        rel = PairwiseRelationship.relationship(pos1, pos2, anchor_ori=anchor_ori, full=True)
        d = DirectionRelBinned.from_relation(rel.direction, bin_system)
        s = DistanceRelBinned.from_value(rel.dist.value if rel.dist else 0.0, distance_bin_system)
        return cls(direction=d, dist=s)

    def to_string(self) -> str:
        return f"{self.direction.to_string()}, {self.dist.to_string()}"
    
    


@dataclass()
class ProximityRelationship:
    """Object-to-object proximity relationship for nearby objects in agent's FOV."""
    pairwise_rel: PairwiseRelationshipDiscrete
    
    # Proximity threshold - objects must be within this distance to have proximity relationship
    PROXIMITY_THRESHOLD: ClassVar[float] = 2.0 # within near distance
    
    def __eq__(self, other):
        if type(other) is not ProximityRelationship:
            return False
        return self.pairwise_rel == other.pairwise_rel
    
    def __hash__(self):
        return hash(self.pairwise_rel)
    
    @classmethod
    def from_positions(cls, a_pos: tuple, b_pos: tuple, perspective_ori: tuple = (0, 1)) -> Optional['ProximityRelationship']:
        """Create proximity relationship between two close objects (both must be in agent's FOV)."""
        # Check if objects are close enough to each other
        a_to_b_dist = np.linalg.norm(np.array(a_pos) - np.array(b_pos))
        if a_to_b_dist > cls.PROXIMITY_THRESHOLD:
            return None
            
        # Create pairwise relationship between the two objects using agent's perspective
        cardinal_bins = CardinalBinsEgo()
        distance_bins = StandardDistanceBins()
        pairwise_rel = PairwiseRelationshipDiscrete.relationship(a_pos, b_pos, perspective_ori, cardinal_bins, distance_bins)
        
        return cls(pairwise_rel=pairwise_rel)
    
    @classmethod
    def prompt(cls, bin_system: BinSystem = None, distance_bin_system: DistanceBinSystem = None) -> str:
        bin_system = bin_system or CardinalBinsEgo()
        distance_bin_system = distance_bin_system or StandardDistanceBins()
        return (
            f"Proximity relationship reporting:\n"
            f"Object-to-object relationships for nearby objects (≤{cls.PROXIMITY_THRESHOLD} units).\n"
            f"- {bin_system.prompt()}\n"
            f"- {distance_bin_system.prompt()}"
        )
    
    def to_string(self, a_name: str, b_name: str) -> str:
        rel_str = self.pairwise_rel.to_string()
        return f"from {b_name}'s view, {a_name} is {rel_str}"



@dataclass()
class RelationTriple:
    """subject -> anchor with relation and perspective (anchor's orientation).
    Relationship is evaluated as subject relative to anchor from anchor's perspective.
    """
    subject: str
    anchor: str
    relation: Union[PairwiseRelationship, PairwiseRelationshipDiscrete, ProximityRelationship]
    orientation: Optional[tuple] = None  # anchor object's orientation










if __name__ == "__main__":
    relationship = PairwiseRelationshipDiscrete.relationship((4, 6), (0, 0), anchor_ori=(1, 0), bin_system=CardinalBinsEgo(), distance_bin_system=StandardDistanceBins())
    print(relationship)
    print(relationship.to_string())