"""
Base evaluation definitions (data and abstract base classes).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np
from dataclasses import dataclass

from ..core.room import Room
from ..core.object import Agent, Object
from ..utils.eval_utilities import multi_choice_eval_fn
from ..actions import RotateAction, ObserveAction

@dataclass
class EvaluationData:
    question: str
    answer: str
    reasoning: str
    task_type: str
    choices: List[str] = None

    def __post_init__(self):
        # Lazy import to avoid circular dependency during module import
        from .task_types import EvalTaskType  # type: ignore
        valid_task_types = EvalTaskType.get_class_names()
        assert self.task_type in valid_task_types, f"Invalid task type: {self.task_type}"
        if self.choices is None:
            self.choices = []
    
    def evaluate(self, pred: Any) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate an answer to the given question using multi-choice evaluation"""
        return multi_choice_eval_fn(pred, self.answer), {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the evaluation data to a dictionary"""
        return {
            'question': self.question,
            'answer': self.answer,
            'reasoning': self.reasoning,
            'task_type': self.task_type,
            'choices': self.choices,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationData':
        """Initialize the evaluation data from a dictionary"""
        return cls(**data)


class BaseEvaluationTask(ABC):
    """Abstract base class for all spatial evaluation tasks."""
    
    def __init__(self, np_random: np.random.Generator, room: Room, agent: Agent, config: Dict[str, Any] = None):
        """Initialize the evaluation task"""
        self.config = config or {}
        self.np_random = np_random
        self.room = room.copy()
        self.agent = agent.copy()
        self.eval_data = EvaluationData(
            question="",
            answer="",
            reasoning="",
            task_type=self.__class__.__name__,
            choices=[]
        )

    @property
    def answer(self) -> Any:
        return self.eval_data.answer
    
    @property
    def question(self) -> str:
        return self.eval_data.question
    
    @property
    def reasoning(self) -> str:
        return self.eval_data.reasoning
    
    @property
    def choices(self) -> List[str]:
        return self.eval_data.choices
    
    def _generate_reasoning(self) -> str:
        """Generate reasoning for the evaluation task"""
        return f"Reasoning for {self.__class__.__name__}"
    
    def format_choices(self, choices: List[str], correct_index: int) -> Tuple[str, str]:
        """Format choices as lines and return (choices_text, correct_label)."""
        return "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]), chr(65 + correct_index)
    
    @abstractmethod
    def generate_choices(self, correct_answer: Any) -> Tuple[List[str], int]:
        """Return (choices, correct_index)."""
        raise NotImplementedError
    
    @abstractmethod
    def generate_question(self) -> str:
        """Generate evaluation question based on the current room/agent state."""
        if self.room is None:
            raise ValueError("Room must be set before generating question")
        raise NotImplementedError
    
    def evaluate(self, pred: Any) -> Tuple[bool, Dict[str, Any]]:
        return self.eval_data.evaluate(pred)
    
    def to_string(self) -> str:
        return f"{self.__class__.__name__}()"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the evaluation task to a dictionary"""
        return {
            'question': self.question,
            'answer': self.answer,
            'reasoning': self.reasoning,
            'choices': self.choices,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEvaluationTask':
        raise NotImplementedError

    @classmethod
    def create_task_from_dict(cls, data: Dict[str, Any]) -> 'BaseEvaluationTask':
        from .task_types import EvalTaskType  # Lazy import
        task_types = EvalTaskType.get_class_map()
        task_type = data.get('type', cls.__name__)
        return task_types.get(task_type, cls).from_dict(data)



class SpatialManipulationTaskBase(BaseEvaluationTask):
    """Base class for tasks that manipulate objects and position agents."""

    def _position_agent_at(self, pos: np.ndarray) -> None:
        self.agent.pos = np.array(pos)
        for rotation in self.np_random.permutation([0, 90, 180, 270]):
            RotateAction(rotation).execute(self.room, self.agent)
            if ObserveAction().execute(self.room, self.agent).data['visible_objects']:
                break

    def _take_full_observations(self, neglect_objects: List[str] = None) -> str:
        messages = []
        obs_result = ObserveAction().execute(self.room, self.agent, neglect_objects=neglect_objects or [])
        messages.append(obs_result.message)
        all_objects = [obj.name for obj in self.room.objects if obj.name not in (neglect_objects or [])]
        visible_objects = obs_result.data['visible_objects']
        if len(visible_objects) < len(all_objects):
            messages.extend([
                RotateAction(180).execute(self.room, self.agent).message,
                ObserveAction().execute(self.room, self.agent, neglect_objects=neglect_objects or []).message,
            ])
        return '\n'.join(messages)

    def _get_diagonal_position(self, target_obj: Object, ref_obj1: Object, ref_obj2: Object, neglect_trivial: bool = False) -> np.ndarray:
        joints = [(ref_obj1.pos[0], ref_obj2.pos[1]), (ref_obj2.pos[0], ref_obj1.pos[1])]
        if not neglect_trivial:
            joints.extend([tuple(ref_obj1.pos), tuple(ref_obj2.pos)])
        joint = self.np_random.choice(list(set(joints)))
        joint_dir = np.array(joint) - target_obj.pos
        jx, jy = joint
        for _ in range(200):
            p = self.room.get_random_point(self.np_random)
            if (p[0] - jx) * joint_dir[0] >= 0 and (p[1] - jy) * joint_dir[1] >= 0:
                return p
        return self.room.get_random_point(self.np_random)