from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
import re
import numpy as np

"""
Base action definitions and common functionality.
Contains the abstract base class and result types for all actions.
"""


@dataclass
class ActionResult:
    """Result of action execution"""
    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


class BaseAction(ABC):
    """Base class for all actions"""
    
    # Class attributes to be overridden by subclasses
    format_desc = ""
    description = ""
    example = ""
    format_pattern = ""
    
    # Shared field of view for all actions
    _field_of_view: int = 90
    
    def __init__(self, parameters=None):
        self.parameters = parameters
    
    @classmethod
    def set_field_of_view(cls, field_of_view: int):
        """Set the field of view for all actions"""
        assert field_of_view in [90, 180], "Field of view must be 90 or 180 degrees"
        cls._field_of_view = field_of_view
    
    @classmethod
    def get_field_of_view(cls) -> int:
        """Get the current field of view"""
        return cls._field_of_view
    
    @abstractmethod
    def success_message(self, **kwargs) -> str:
        """Return success message for this action"""
        pass
    
    @abstractmethod
    def error_message(self, error_type: str) -> str:
        """Return error message for this action"""
        pass
    
    @abstractmethod
    def execute(self, room, **kwargs) -> ActionResult:
        """Execute action on room state.
        
        Args:
            room: Room to execute action on
            **kwargs: Additional execution context (e.g., coordinate system info)
            
        Returns:
            ActionResult containing success status, message, and additional data
        """
        pass
    
    @staticmethod
    def _is_visible(from_obj, to_obj, field_of_view: int = None) -> bool:
        """
        Check if to_obj is visible from from_obj
        Args:
            from_obj: Object viewing from
            to_obj: Object being viewed
            field_of_view: Field of view in degrees (90 or 180). If None, uses class default.
            
        Returns:
            bool: True if to_obj is visible from from_obj's perspective
            
        Notes:
            - For 90-degree field of view: objects within 45° left and right of orientation
            - For 180-degree field of view: objects within 90° left and right of orientation
        """
        if field_of_view is None:
            field_of_view = BaseAction._field_of_view
        
        assert field_of_view in [90, 180], "Invalid field of view"
        direction_vec = to_obj.pos - from_obj.pos
        if np.allclose(direction_vec, 0):
            return True
        direction_norm = direction_vec / np.linalg.norm(direction_vec)
        ori_norm = from_obj.ori / np.linalg.norm(from_obj.ori)
        
        return np.dot(direction_norm, ori_norm) >= (0.707 - 1e-3) if field_of_view == 90 else np.dot(direction_norm, ori_norm) >= (0.0 - 1e-3)
    
    @staticmethod
    def _get_rotation_matrix(degrees: int) -> np.ndarray:
        """Get rotation matrix for specified degrees.
        NOTE agent rotates clockwise <==> other object rotates counterclockwise
        """
        rotations = {0: [[1,0],[0,1]], 90: [[0,1],[-1,0]], 180: [[-1,0],[0,-1]], 270: [[0,-1],[1,0]]}
        return np.array(rotations[degrees])
    
    @staticmethod
    def is_final() -> bool:
        """Check if this is a final action (ends the sequence)"""
        return False
    
    @staticmethod
    def is_term() -> bool:
        """Check if this is a termination action"""
        return False
    
    @classmethod
    def parse(cls, action_str: str):
        """Parse action string and return instance if matches"""
        if match := re.match(cls.format_pattern, action_str):
            return cls(*match.groups())
        return None

    def get_feedback(self, success: bool, error_type: str = None, **kwargs) -> str:
        """Generate feedback based on execution result"""
        return self.success_message(**kwargs) if success else self.error_message(error_type) 