from enum import Enum
from typing import Any, Dict, Optional, Tuple, Type, TYPE_CHECKING
import numpy as np

from ..core.room import Room
from ..core.object import Agent
from ..utils.eval_utilities import evaluate_task_answer
if TYPE_CHECKING:
    from .tasks import BaseEvaluationTask

class EvalTaskType(Enum):
    """Enum for all available evaluation task types."""
    
    # Task type definitions: (short_name, class_name)
    DIR = ("dir", "DirectionEvaluationTask")
    ROT = ("rot", "RotEvaluationTask")
    ROT_DUAL = ("rot_dual", "RotDualEvaluationTask")
    POV = ("pov", "PovEvaluationTask")
    BWD_POV_TEXT = ("bwd_pov_text", "BackwardPovTextEvaluationTask")
    BWD_POV_VISION = ("bwd_pov_vision", "BackwardPovVisionEvaluationTask")
    E2A = ("e2a", "AlloMappingEvaluationTask")
    FWD_LOC = ("fwd_loc", "Action2LocationEvaluationTask")
    BWD_LOC_TEXT = ("bwd_loc_text", "Location2ActionTextEvaluationTask")
    BWD_LOC_VISION = ("bwd_loc_vision", "Location2ActionVisionEvaluationTask")
    FWD_FOV = ("fwd_fov", "Action2ViewEvaluationTask")
    BWD_NAV_TEXT = ("bwd_nav_text", "View2ActionTextEvaluationTask")
    BWD_NAV_VISION = ("bwd_nav_vision", "View2ActionVisionEvaluationTask")
    BWD_NAV_REV = ("bwd_nav_rev", "View2ActionRevEvaluationTask")
    FALSE_BELIEF = ("false_belief", "FalseBeliefDirectionPov")
    DIR_ANCHOR = ("dir_anchor", "DirectionPov")
    
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
        from .direction import DirectionEvaluationTask, PovEvaluationTask, BackwardPovTextEvaluationTask, BackwardPovVisionEvaluationTask, DirectionPov
        from .rotation import RotEvaluationTask, RotDualEvaluationTask
        from .e2a import AlloMappingEvaluationTask
        from .localization import Action2LocationEvaluationTask, Location2ActionTextEvaluationTask, Location2ActionVisionEvaluationTask
        from .false_belief import FalseBeliefDirectionPov
        from .navigation_tasks import Action2ViewEvaluationTask, View2ActionTextEvaluationTask, View2ActionVisionEvaluationTask, View2ActionRevEvaluationTask
        
        task_map = {
            cls.DIR.short_name: DirectionEvaluationTask,
            cls.ROT.short_name: RotEvaluationTask,
            cls.ROT_DUAL.short_name: RotDualEvaluationTask,
            cls.POV.short_name: PovEvaluationTask,
            cls.DIR_ANCHOR.short_name: DirectionPov,
            cls.E2A.short_name: AlloMappingEvaluationTask,
            cls.FWD_LOC.short_name: Action2LocationEvaluationTask,
            cls.BWD_LOC_TEXT.short_name: Location2ActionTextEvaluationTask,
            cls.BWD_LOC_VISION.short_name: Location2ActionVisionEvaluationTask,
            cls.FALSE_BELIEF.short_name: FalseBeliefDirectionPov,
            cls.FWD_FOV.short_name: Action2ViewEvaluationTask,
            cls.BWD_NAV_TEXT.short_name: View2ActionTextEvaluationTask,
            cls.BWD_NAV_VISION.short_name: View2ActionVisionEvaluationTask,
            cls.BWD_NAV_REV.short_name: View2ActionRevEvaluationTask,
            cls.BWD_POV_TEXT.short_name: BackwardPovTextEvaluationTask,
            cls.BWD_POV_VISION.short_name: BackwardPovVisionEvaluationTask,
        }
        return task_map
    
    @classmethod
    def get_class_map(cls) -> Dict[str, 'Type[BaseEvaluationTask]']:
        """Get mapping from class names to task classes."""
        task_map = cls.get_task_map()
        return {task.class_name: task_class for task, task_class in 
                zip(cls, task_map.values())}

    @classmethod
    def resolve_class_name(cls, task_name: str) -> str:
        """Resolve short or long task identifier to class name."""
        if task_name in cls.get_short_names():
            return cls.from_short_name(task_name).class_name
        if task_name in cls.get_class_names():
            return task_name
        raise ValueError(f"Unknown task identifier: {task_name}")

    @classmethod
    def evaluate_prediction(
        cls,
        task_name: str,
        pred: Any,
        answer: Any,
        choices: Optional[list[str]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate a prediction for the given task identifier."""
        class_name = cls.resolve_class_name(task_name)
        return evaluate_task_answer(class_name, pred, answer, choices or [])
    
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
    def create_task(cls, task_name: str, np_random: np.random.Generator, room: 'Room', agent: 'Agent', config: dict = None, history_manager = None) -> 'BaseEvaluationTask':
        """Create an evaluation task instance from task name."""
        task_map = cls.get_task_map()
        if task_name in task_map:
            task_class = task_map[task_name]
            return task_class(np_random, room, agent, config or {}, history_manager)
        else:
            raise ValueError(f"Unknown evaluation task: {task_name}") 
        

if __name__ == "__main__":
    from ..utils.room_utils import RoomPlotter, RoomGenerator
    from tqdm import tqdm


    task_name = 'bwd_loc_text'
    for seed in tqdm(range(0, 1)):
        np_random = np.random.default_rng(seed)
        room, agent = RoomGenerator.generate_room(
            room_size=(30, 30),
            n_objects=10,
            np_random=np_random,
            room_name='room',
            level=2,
            main=6,
        )
        # print(f'room: {room}')
        # print(f'agent: {agent}')
        RoomPlotter.plot(room, agent, mode='img', save_path='room.png')
        task = EvalTaskType.create_task(task_name, np_random=np_random, room=room, agent=agent)
        print(task.generate_question(), task.answer)
        user_pred = "(1, -5)" # task.answer
        score, info = EvalTaskType.evaluate_prediction(task_name, user_pred, task.answer, task.choices)
        print(f"Evaluation result: {score}, details: {info}")