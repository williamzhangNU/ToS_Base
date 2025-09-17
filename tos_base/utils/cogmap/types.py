from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BaseCogMetrics:
    overall: float = 0.0
    valid: bool = True

    def to_dict(self) -> Dict[str, float]:
        return {"overall": float(self.overall)}

    @classmethod
    def invalid(cls):
        return cls(overall=0.0, valid=False)


@dataclass
class MapCogMetrics(BaseCogMetrics):
    dir: float = 0.0
    facing: float = 0.0
    pos: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "dir": float(self.dir),
            "facing": float(self.facing),
            "pos": float(self.pos),
            "overall": float(self.overall),
        }

    @staticmethod
    def average(items: List['MapCogMetrics']) -> 'MapCogMetrics':
        valid_items = [i for i in items if isinstance(i, MapCogMetrics) and i.valid]
        if not valid_items:
            return MapCogMetrics.invalid()
        import numpy as np
        return MapCogMetrics(
            dir=float(np.mean([i.dir for i in valid_items])),
            facing=float(np.mean([i.facing for i in valid_items])),
            pos=float(np.mean([i.pos for i in valid_items])),
            overall=float(np.mean([i.overall for i in valid_items])),
            valid=True,
        )

    @classmethod
    def invalid(cls) -> 'MapCogMetrics':
        return cls(dir=0.0, facing=0.0, pos=0.0, overall=0.0, valid=False)


@dataclass
class RelationMetrics(BaseCogMetrics):
    dir: float = 0.0
    dist: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"dir": float(self.dir), "dist": float(self.dist), "overall": float(self.overall)}

    @classmethod
    def invalid(cls) -> 'RelationMetrics':
        return cls(dir=0.0, dist=0.0, overall=0.0, valid=False)

    @staticmethod
    def average(items: List['RelationMetrics']) -> 'RelationMetrics':
        valid_items = [i for i in items if isinstance(i, RelationMetrics) and i.valid]
        if not valid_items:
            return RelationMetrics.invalid()
        import numpy as np
        return RelationMetrics(
            dir=float(np.mean([i.dir for i in valid_items])),
            dist=float(np.mean([i.dist for i in valid_items])),
            overall=float(np.mean([i.overall for i in valid_items])),
            valid=True,
        )


@dataclass
class ConsistencySummary:
    local_vs_global: Optional[MapCogMetrics] = None
    rooms_vs_global_avg: Optional[MapCogMetrics] = None
    rooms_vs_global_per_room: Dict[str, MapCogMetrics] = field(default_factory=dict)
    map_vs_relations: float = 0.0
    relations_consistency: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "local_vs_global": (self.local_vs_global.to_dict() if self.local_vs_global and self.local_vs_global.valid else {}),
            "rooms_vs_global": {
                "average": (self.rooms_vs_global_avg.to_dict() if self.rooms_vs_global_avg and self.rooms_vs_global_avg.valid else {}),
                "per_room": {k: v.to_dict() for k, v in self.rooms_vs_global_per_room.items()},
            },
            "map_vs_relations": float(self.map_vs_relations),
            "relations_consistency": float(self.relations_consistency),
        }


__all__ = [
    "BaseCogMetrics",
    "MapCogMetrics",
    "RelationMetrics",
    "ConsistencySummary",
]


