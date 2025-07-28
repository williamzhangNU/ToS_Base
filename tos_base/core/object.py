from typing import List, Dict, Union
import numpy as np
from typing import Any
from dataclasses import dataclass, field


@dataclass
class Object:
    """
    Represents an object or agent in a 2D environment with position and orientation.

    Attributes:
        name (str): The identifier for the object
        pos (np.ndarray): A 2D coordinate representing position
        ori (np.ndarray): A 2D unit vector representing orientation
            - (1, 0)  → 0 degrees
            - (0, 1)  → 90 degrees
            - (-1, 0) → 180 degrees
            - (0, -1) → 270 degrees
        has_orientation (bool): Whether this object has meaningful orientation

    Raises:
        ValueError: If the orientation vector is not one of the valid orientations
                   (only for objects with has_orientation=True)
    """

    name: str
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    ori: np.ndarray = field(default_factory=lambda: np.array([0, 1]))
    has_orientation: bool = True

    def __post_init__(self):
        assert len(self.pos) == 2, "Position must be a 2D vector"
        assert len(self.ori) == 2, "Orientation must be a 2D vector"
        if self.has_orientation:
            self._validate()


    def _validate(self) -> None:
        VALID_ORIENTATIONS = [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([-1, 0]),
            np.array([0, -1])
        ]
        if not any(np.array_equal(self.ori, valid_ori) for valid_ori in VALID_ORIENTATIONS):
            raise ValueError(
                f"Orientation must be one of {[o.tolist() for o in VALID_ORIENTATIONS]}, "
                f"got {self.ori.tolist()}"
            )

    def to_dict(self) -> Dict[str, Union[str, List[float], bool]]:
        return {
            'name': self.name,
            'pos': self.pos.tolist(),
            'ori': self.ori.tolist(),
            'has_orientation': self.has_orientation
        }

    @classmethod
    def from_dict(cls, obj_dict: Dict[str, Union[str, List[float], bool]]) -> 'Object':
        return cls(
            name=obj_dict['name'],
            pos=np.array(obj_dict['pos']),
            ori=np.array(obj_dict['ori']),
            has_orientation=obj_dict.get('has_orientation', True)
        )

    def __repr__(self) -> str:
        return (
            f"\nObject(\n"
            f"    name={self.name},\n"
            f"    pos={self.pos.tolist()},\n"
            f"    ori={self.ori.tolist()},\n"
            f"    has_orientation={self.has_orientation}\n"
            f")"
        )
    
    def copy(self):
        return Object(
            name=self.name,
            pos=self.pos.copy(),
            ori=self.ori.copy(),
            has_orientation=self.has_orientation
        )

@dataclass
class Agent(Object):
    name: str = 'agent'
    pos: np.ndarray = field(default_factory=lambda: np.array([0, 0]), init=True)
    ori: np.ndarray = field(default_factory=lambda: np.array([0, 1]), init=True)
    has_orientation: bool = field(default=True, init=True)


    """@classmethod
    def from_dict(cls, obj_dict: Dict[str, Any]) -> 'Agent':
        # only call __init__ with name, then set the rest
        inst = cls(name=obj_dict['name'])
        inst.pos = np.array(obj_dict['pos'])
        inst.ori = np.array(obj_dict['ori'])
        inst.has_orientation = obj_dict.get('has_orientation', True)
        return inst"""