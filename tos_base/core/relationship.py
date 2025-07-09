from enum import Enum, auto
from typing import Union
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


class DirectionSystem:
    """
    Manages spatial relationships between objects in the environment.
    
    Provides conversion between internal direction representations and
    human-readable strings in both egocentric and allocentric perspectives.
    """
    
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
    
    # Orientation transformations for 90, 180, and 270 degrees rotation
    TRANSFORMS = {
        # Identity (facing north)
        (0, 1): {dir_: dir_ for dir_ in Dir},
        
        # Facing east (90° counterclockwise)
        (1, 0): {
            Dir.FORWARD: Dir.LEFT,
            Dir.BACKWARD: Dir.RIGHT,
            Dir.RIGHT: Dir.FORWARD,
            Dir.LEFT: Dir.BACKWARD,
            Dir.SAME: Dir.SAME,
            Dir.UNKNOWN: Dir.UNKNOWN,
        },
        
        # Facing south (180° rotation)
        (0, -1): {
            Dir.FORWARD: Dir.BACKWARD,
            Dir.BACKWARD: Dir.FORWARD,
            Dir.RIGHT: Dir.LEFT,
            Dir.LEFT: Dir.RIGHT,
            Dir.SAME: Dir.SAME,
            Dir.UNKNOWN: Dir.UNKNOWN,
        },
        
        # Facing west (270° counterclockwise)
        (-1, 0): {
            Dir.FORWARD: Dir.RIGHT,
            Dir.BACKWARD: Dir.LEFT,
            Dir.RIGHT: Dir.BACKWARD,
            Dir.LEFT: Dir.FORWARD,
            Dir.SAME: Dir.SAME,
            Dir.UNKNOWN: Dir.UNKNOWN,
        },
    }
    
    @classmethod
    def to_string(cls, direction: Union[Dir, DirPair], perspective: str = 'ego') -> str:
        """
        Convert direction to a human-readable string based on perspective.
        
        Args:
            direction: A single direction or a direction pair
            perspective: 'ego' (egocentric) or 'allo' (allocentric)
            
        Returns:
            String representation of the direction
        """
        assert perspective in ('ego', 'allo'), f"Invalid perspective: {perspective}"
            
        labels = cls.EGO_LABELS if perspective == 'ego' else cls.ALLO_LABELS
        
        if isinstance(direction, Dir):
            return labels[direction]
        elif isinstance(direction, DirPair):
            h_str = labels[direction.horiz]
            v_str = labels[direction.vert]
            
            return f"({h_str}, {v_str})"
        else:
            raise TypeError(f"Expected Dir or DirPair, got {type(direction)}")
    
    @classmethod
    def transform(cls, dir_pair: DirPair, orientation: tuple) -> DirPair:
        """
        Transform direction based on an orientation.
        
        Args:
            dir_pair: Direction pair to transform
            orientation: Tuple of (x, y) representing orientation vector
            
        Returns:
            Transformed direction pair
        """
        ori_tuple = tuple(map(int, orientation))
        if ori_tuple not in cls.TRANSFORMS:
            raise ValueError(f"Invalid orientation: {ori_tuple}")
            
        transform = cls.TRANSFORMS[ori_tuple]
        
        # Transform and handle axis swapping if needed
        horiz_transformed = transform[dir_pair.horiz]
        vert_transformed = transform[dir_pair.vert]
        
        # Handle axis swapping
        if (horiz_transformed in (Dir.FORWARD, Dir.BACKWARD) or 
            vert_transformed in (Dir.RIGHT, Dir.LEFT)):
            return DirPair(vert_transformed, horiz_transformed)
            
        return DirPair(horiz_transformed, vert_transformed)
    
    @classmethod
    def get_direction(cls, pos1: tuple, pos2: tuple, orientation: tuple = None) -> DirPair:
        """
        Direction of pos1 relative to pos2
        
        Args:
            pos1: Reference position (x, y)
            pos2: Target position (x, y)
            orientation: Optional orientation to transform the result
            
        Returns:
            Direction pair representing the relationship
        """
        # Convert inputs to numpy arrays
        p1 = np.array(pos1)
        p2 = np.array(pos2)
        
        # Calculate difference vector
        diff = p1 - p2
        
        # Create direction pair
        h_dir = Dir.from_horizontal_delta(diff[0])
        v_dir = Dir.from_delta(diff[1])
        dir_pair = DirPair(h_dir, v_dir)
        
        # Apply orientation transform if provided
        if orientation is not None:
            return cls.transform(dir_pair, orientation)
            
        return dir_pair
    
    @classmethod
    def get_relative_orientation(cls, obj_ori: tuple, anchor_ori: tuple) -> DirPair:
        """
        Get object's orientation relative to an anchor orientation.
        
        Args:
            obj_ori: Object's orientation vector
            anchor_ori: Reference orientation vector
            
        Returns:
            Direction pair describing relative orientation
        """
        # Calculate direction as if object was at origin
        base_dir = cls.get_direction(obj_ori, (0, 0))
        
        # Transform based on anchor orientation
        return cls.transform(base_dir, anchor_ori)