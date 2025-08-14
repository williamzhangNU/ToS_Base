from enum import Enum
from typing import Dict, Type, TYPE_CHECKING
import numpy as np

from ..core.room import Room
from ..core.object import Agent
if TYPE_CHECKING:
    from .tasks import BaseEvaluationTask

class EvalTaskType(Enum):
    """Enum for all available evaluation task types."""
    
    # Task type definitions: (short_name, class_name)
    DIR = ("dir", "DirectionEvaluationTask")
    ROT = ("rot", "RotEvaluationTask")
    ROT_DUAL = ("rot_dual", "RotDualEvaluationTask")
    POV = ("pov", "PovEvaluationTask")
    E2A = ("e2a", "E2AEvaluationTask")
    LOCALIZATION = ("loc", "LocalizationEvaluationTask")
    FALSE_BELIEF = ("false_belief", "FalseBeliefEvaluationTask")
    
    def __init__(self, short_name: str, class_name: str):
        self.short_name = short_name
        self.class_name = class_name
    
    @classmethod
    def get_short_names(cls) -> list[str]:
        """Get all short names for task types."""
        return [task.short_name for task in cls]
    
    @classmethod
    def get_class_names(cls) -> list[str]:
        """Get all class names for task types."""
        return [task.class_name for task in cls]
    
    @classmethod
    def get_task_map(cls) -> Dict[str, 'Type[BaseEvaluationTask]']:
        """Get mapping from short names to task classes."""
        # Import here to avoid circular imports
        from .direction import DirectionEvaluationTask, PovEvaluationTask
        from .rotation import RotEvaluationTask, RotDualEvaluationTask
        from .e2a import E2AEvaluationTask
        from .localization import LocalizationEvaluationTask
        from .false_belief import FalseBeliefEvaluationTask
        
        task_map = {
            cls.DIR.short_name: DirectionEvaluationTask,
            cls.ROT.short_name: RotEvaluationTask,
            cls.ROT_DUAL.short_name: RotDualEvaluationTask,
            cls.POV.short_name: PovEvaluationTask,
            cls.E2A.short_name: E2AEvaluationTask,
            cls.LOCALIZATION.short_name: LocalizationEvaluationTask,
            cls.FALSE_BELIEF.short_name: FalseBeliefEvaluationTask,
        }
        return task_map
    
    @classmethod
    def get_class_map(cls) -> Dict[str, 'Type[BaseEvaluationTask]']:
        """Get mapping from class names to task classes."""
        task_map = cls.get_task_map()
        return {task.class_name: task_class for task, task_class in 
                zip(cls, task_map.values())}
    
    @classmethod
    def from_short_name(cls, short_name: str) -> 'EvalTaskType':
        """Get task type from short name."""
        for task in cls:
            if task.short_name == short_name:
                return task
        raise ValueError(f"Unknown task short name: {short_name}")
    
    @classmethod
    def from_class_name(cls, class_name: str) -> 'EvalTaskType':
        """Get task type from class name."""
        for task in cls:
            if task.class_name == class_name:
                return task
        raise ValueError(f"Unknown task class name: {class_name}")
    
    @classmethod
    def create_task(cls, task_name: str, np_random: np.random.Generator, room: 'Room', agent: 'Agent', config: dict = None) -> 'BaseEvaluationTask':
        """Create an evaluation task instance from task name."""
        task_map = cls.get_task_map()
        if task_name in task_map:
            task_class = task_map[task_name]
            return task_class(np_random, room, agent, config or {})
        else:
            raise ValueError(f"Unknown evaluation task: {task_name}") 