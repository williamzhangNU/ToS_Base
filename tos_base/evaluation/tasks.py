"""
The script defines the different evaluation metrics for the SpatialGym.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, List
import numpy as np
from typing_extensions import override
import copy
from dataclasses import dataclass

from .task_types import EvalTaskType

from ..core.room import Room, BaseRoom
from ..core.object import Agent
from ..utils.eval_utilities import (
    multi_choice_eval_fn,
)
from ..core.constant import CANDIDATE_OBJECTS
from ..core.graph import DirectionalGraph
from ..core.object import Object, Agent
from ..core.relationship import DirPair, Dir, DirectionRel, TotalRelationship
from ..actions import MoveAction, RotateAction, ObserveAction, BaseAction

@dataclass
class EvaluationData:
    question: str
    answer: str
    reasoning: str
    task_type: str
    choices: List[str] = None

    def __post_init__(self):
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
    """Base class for all spatial evaluation tasks"""
    
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
    
    def format_choices(self, choices: List[str], correct_index: int) -> str:
        """Format choices as A. B. C. D. ... and return choice text"""
        return "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]), chr(65 + correct_index)
    
    @abstractmethod
    def generate_choices(self, correct_answer: Any) -> Tuple[List[str], str]:
        """Generate choices with only one correct answer"""
        pass
    
    @abstractmethod
    def generate_question(self) -> str:
        """Generate evaluation questions based on the room state"""
        if self.room is None:
            raise ValueError("Room must be set before generating question")
        pass
    
    def evaluate(self, pred: Any) -> Tuple[bool, Dict[str, Any]]:
        return self.eval_data.evaluate(pred)
    
    def to_string(self) -> str:
        """Convert the evaluation task to a string"""
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
        """Initialize the evaluation task from a dictionary"""
        return cls(
            question=data['question'],
            answer=data['answer'],
            reasoning=data['reasoning'],
        )

    @classmethod
    def create_task_from_dict(cls, data: Dict[str, Any]) -> 'BaseEvaluationTask':
        """Initialize a single evaluation task from a dictionary with type information"""
        task_types = EvalTaskType.get_class_map()
        task_type = data.get('type', cls.__name__)
        return task_types.get(task_type, cls).from_dict(data)


class DirectionEvaluationTask(BaseEvaluationTask):
    """Unified direction task: supports pairwise (dir) and perspective-taking (pov)."""

    MODE = 'dir'

    QUESTION_TEMPLATE_DIR = (
        "From a top-down view, what is the spatial relationship between the pair {obj_pairs_str}?\n"
        "For the pair (A, B), the relationship (<horizontal>, <vertical>) means A is <horizontal> of B and A is <vertical> of B.\n"
        "where <horizontal>: west, east, same; <vertical>: north, south, same\n"
        "Choose the correct answer:\n"
        "{choices_text}\n\n"
        "IMPORTANT: You must respond with ONLY the single letter choice (E.g., A, B, C, D) in your answer\n\n"
    )
    QUESTION_TEMPLATE_POV = (
        "Imagine you are at the same position and orientation as the {anchor_obj_name}.\n"
        "From this perspective, what is the direction of the {obj_name}?\n\n"
        "For object A, the relationship (<horizontal>, <vertical>) means A is <horizontal> and <vertical> of you.\n"
        "Choose the correct answer:\n"
        "{choices_text}\n\n"
        "IMPORTANT: You must respond with ONLY the single letter choice (E.g., A, B, C, D) in your answer\n\n"
    )

    def generate_question(self) -> str:
        if self.room is None:
            raise ValueError("Room must be set before generating question")

        mode = self.config.get('mode', getattr(self, 'MODE', 'dir'))

        if mode == 'pov':
            # pick an object and an oriented anchor
            obj_idx = self.np_random.integers(0, len(self.room.objects))
            oriented_indices = [i for i, obj in enumerate(self.room.objects) if obj.has_orientation]
            assert len(oriented_indices) > 0, "No objects with orientation found for perspective taking task"
            anchor_idx = self.np_random.choice(oriented_indices)
            while obj_idx == anchor_idx:
                obj_idx = self.np_random.integers(0, len(self.room.objects))
            obj_name = self.room.objects[obj_idx].name
            anchor_name = self.room.objects[anchor_idx].name
            obj_pos = self.room.get_object_by_name(obj_name).pos
            anchor_obj = self.room.get_object_by_name(anchor_name)
            dir_rel = TotalRelationship.get_direction(tuple(obj_pos), tuple(anchor_obj.pos), anchor_ori=tuple(anchor_obj.ori))
            correct_answer = dir_rel.to_string('ego')
            choices, correct_idx = self.generate_choices(correct_answer)
            choices_text, correct_label = self.format_choices(choices, correct_idx)
            self.eval_data.question = self.QUESTION_TEMPLATE_POV.format(
                anchor_obj_name=anchor_name,
                obj_name=obj_name,
                choices_text=choices_text,
            )
        else:  # dir
            n = len(self.room.objects)
            pairs = [(i, j) if self.np_random.random() >= 0.5 else (j, i)
                     for i in range(n) for j in range(i + 1, n)]
            self.np_random.shuffle(pairs)
            i, j = pairs[0]
            obj1, obj2 = self.room.objects[i], self.room.objects[j]
            dir_rel = TotalRelationship.get_direction(tuple(obj1.pos), tuple(obj2.pos))
            correct_answer = dir_rel.to_string('allo')
            choices, correct_idx = self.generate_choices(correct_answer)
            choices_text, correct_label = self.format_choices(choices, correct_idx)
            self.eval_data.question = self.QUESTION_TEMPLATE_DIR.format(
                obj_pairs_str=f"({obj1.name}, {obj2.name})",
                choices_text=choices_text,
            )
        
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question

    def generate_choices(self, correct_answer: str) -> Tuple[List[str], int]:
        h_dirs = [Dir.LEFT, Dir.RIGHT, Dir.SAME]
        v_dirs = [Dir.FORWARD, Dir.BACKWARD, Dir.SAME]
        choices = [correct_answer]
        while len(choices) < 4:
            wrong_dir = DirPair(self.np_random.choice(h_dirs), self.np_random.choice(v_dirs))
            choice = f"{DirectionRel.pair_to_string(wrong_dir, perspective='allo' if self.MODE == 'dir' else 'ego')}"
            if choice not in choices:
                choices.append(choice)
        self.np_random.shuffle(choices)
        correct_idx = choices.index(correct_answer)
        return choices, correct_idx

class PovEvaluationTask(DirectionEvaluationTask):
    """Thin alias for POV mode of DirectionEvaluationTask."""
    MODE = 'pov'


class RotEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for rotation questions.
    
    Q: What is the sequence of objects when agent turns around at its original position?
    A: [<obj1>, <obj2>, ...]
    
    Agent turns clockwise/counterclockwise and lists objects in order of encounter.

    TODO:
    1. for movement, need to gaurentee no ambiguity when generate room / evaluate answer
    2. Before rotation, the agent can turn to face some object
    """

    
    QUESTION_TEMPLATE = (
        "You return to your starting position and facing north.\n"
        "perform a full 360° rotation by turning {turn_direction} in place.\n"
        "Identify the order in which objects come directly into view (straight ahead).\n\n"
        "Instructions:\n"
        "- Exclude any object at your exact position\n"
        "- An object is counted only when you face it directly\n\n"
        "Choose the correct sequence:\n"
        "{choices_text}\n\n"
        "IMPORTANT: You must respond with ONLY the single letter choice (E.g., A, B, C, D) in your answer\n\n"
    )
    MOVEMENT_TEMPLATE = (
        "You moved to the same position as {move_obj_name}.\n"
    )
    TURN_TEMPLATE = (
        "You turned clockwise {degree} degrees.\n"
    )
    
    def generate_question(self) -> str:

        if self.room is None:
            raise ValueError("Room must be set before generating question")
        turn_direction = self.np_random.choice(['clockwise', 'counterclockwise'])
        if_move = self.config.get('if_move', False)
        if_turn = self.config.get('if_turn', False)
        
        movement_prompt = ""
        turn_prompt = ""
        neglect_objects = []
        if if_move:
            move_obj = self.np_random.choice(self.room.objects)
            movement_prompt = self.MOVEMENT_TEMPLATE.format(move_obj_name=move_obj.name)
            neglect_objects.append(move_obj.name)
            MoveAction(move_obj.name).execute(self.room, self.agent)
        if if_turn:
            degree = self.np_random.choice([90, 180, 270])
            turn_prompt = self.TURN_TEMPLATE.format(degree=degree)
            RotateAction(degree).execute(self.room, self.agent)

        # Compute ordering by relative bearing around the agent (CW positive)
        def bearing_deg(obj: Object) -> Tuple[float, float]:
            deg = TotalRelationship.get_degree(tuple(obj.pos), tuple(self.agent.pos), anchor_ori=tuple(self.agent.ori)).value
            # Wrap to [0, 360) in chosen rotation direction
            angle = (deg % 360.0) if turn_direction == 'clockwise' else ((-deg) % 360.0)
            dist = TotalRelationship.get_distance(tuple(obj.pos), tuple(self.agent.pos)).value
            return angle, dist

        objects = [obj for obj in self.room.objects if not np.array_equal(obj.pos, self.agent.pos)]
        objects.sort(key=bearing_deg)
        correct_answer = [obj.name for obj in objects]
        
        # Generate choices
        choices, correct_idx = self.generate_choices(correct_answer)
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = movement_prompt + turn_prompt + self.QUESTION_TEMPLATE.format(
            turn_direction=turn_direction,
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question
    
    def generate_choices(self, correct_answer: list) -> Tuple[List[str], int]:
        """Generate 4 object sequence choices"""
        correct_answer_str = ", ".join(correct_answer)
        choices = [correct_answer_str]
        
        for _ in range(3):
            wrong_list = correct_answer.copy()
            assert len(wrong_list) >= 3, "Need at least 3 objects for this task"
            while ", ".join(wrong_list) in choices:
                self.np_random.shuffle(wrong_list)
            choices.append(", ".join(wrong_list))
        
        self.np_random.shuffle(choices)
        correct_idx = choices.index(correct_answer_str)
        return choices, correct_idx
    
    @override
    def to_string(self) -> str:
        return f"{self.__class__.__name__}({self.config.get('turn_direction', 'clockwise')})"


class RotDualEvaluationTask(BaseEvaluationTask):
    """
    Dual evaluation task for rotation questions.
    
    Q: Given a sequence of objects appearing in front during rotation, what direction did you rotate?
    A: "clockwise" or "counterclockwise"
    
    Reverse of RotEvaluationTask - provides sequence, asks for direction.

    TODO: make it four-choice
    """

    QUESTION_TEMPLATE = (
        "You return to your starting position and facing north.\n"
        "you performed a complete 360° rotation in place.\n"
        "During the rotation, these objects appeared directly in front of you in this order:\n"
        "{object_sequence}\n\n"
        "Based on this sequence, in which direction did you rotate?\n\n"
        "Choose the correct answer:\n"
        "{choices_text}\n\n"
        "IMPORTANT: You must respond with ONLY the single letter choice (E.g., A, B, C, D) in your answer\n\n"
    )
    
    def generate_question(self) -> str:

        if self.room is None:
            raise ValueError("Room must be set before generating question")
        
        # Randomly choose rotation direction
        turn_direction = self.np_random.choice(['clockwise', 'counterclockwise'])
        
        # Sort objects by relative bearing around the agent
        def bearing_deg(obj: Object) -> Tuple[float, float]:
            deg = TotalRelationship.get_degree(tuple(obj.pos), tuple(self.agent.pos), anchor_ori=tuple(self.agent.ori)).value
            angle = (deg % 360.0) if turn_direction == 'clockwise' else ((-deg) % 360.0)
            dist = TotalRelationship.get_distance(tuple(obj.pos), tuple(self.agent.pos)).value
            return angle, dist

        objects = [obj for obj in self.room.objects if not np.array_equal(obj.pos, self.agent.pos)]
        objects.sort(key=bearing_deg)
        
        # Create sequence string
        object_names = [obj.name for obj in objects]
        object_sequence = ", ".join(object_names)
        
        # Generate choices
        choices, correct_idx = self.generate_choices(turn_direction)
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        
        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            object_sequence=object_sequence,
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        
        return self.eval_data.question
    
    def generate_choices(self, correct_answer: str) -> Tuple[List[str], int]:
        """Generate 4 rotation direction choices"""
        opposite = 'counterclockwise' if correct_answer == 'clockwise' else 'clockwise'
        choices = [correct_answer, opposite]
        
        self.np_random.shuffle(choices)
        correct_idx = choices.index(correct_answer)
        return choices, correct_idx

class E2AEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for coordinate and orientation identification.
    
    Q: Given object names, what are their coordinates and orientations?
    A: [(coord, orientation), ...]
    
    Tests conversion from object identification to coordinates and orientations.

    TODO debug
    """

    QUESTION_TEMPLATE = (
        "You return to your starting position and facing north.\n"
        "Consider the global map coordinates (x right, y up).\n"
        "What are the coordinates and orientations of these objects: {object_names}?\n\n"
        "Answer format: [obj at (x, y) facing orientation, ...] where orientation is north/east/south/west\n"
        "Choose the correct answer:\n"
        "{choices_text}\n\n"
        "IMPORTANT: You must respond with ONLY the single letter choice (E.g., A, B, C, D) in your answer\n\n"
    )
    
    def generate_question(self) -> str:
        if self.room is None:
            raise ValueError("Room must be set before generating question")
        
        # Objects for listing
        objects = list(self.room.objects) + [self.agent]
        
        self.np_random.shuffle(objects)
        object_names = [obj.name for obj in objects]
        
        # Create correct answer: (object_name, coord, orientation)
        correct_answer = [(obj.name, tuple(obj.pos.astype(int)), self._get_orientation_string(obj)) 
                         for obj in objects]
        
        # Generate choices
        choices, correct_idx = self.generate_choices(correct_answer, objects)
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        
        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            object_names=", ".join(object_names),
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question
    
    def _get_orientation_string(self, obj: Object) -> str:
        """Convert object orientation to north/east/south/west string"""
        return {(0, 1): "north", (1, 0): "east", (0, -1): "south", (-1, 0): "west"}[tuple(obj.ori)]
    
    def generate_choices(self, correct_answer: List[Tuple], objects: List[Object]) -> Tuple[List[str], int]:
        """Generate 4 coordinate and orientation choices"""
        correct_str = self._format_answer(correct_answer)
        choices = [correct_str]
        
        # Create ground truth graph for comparison
        coords = [obj.pos for obj in objects]
        gt_v_matrix, gt_h_matrix = DirectionalGraph.create_graph_from_coordinates(coords)
        
        for _ in range(3):
            wrong_answer = self._generate_wrong_choice(correct_answer, objects, gt_v_matrix, gt_h_matrix)
            choices.append(self._format_answer(wrong_answer))
        
        self.np_random.shuffle(choices)
        correct_idx = choices.index(correct_str)
        return choices, correct_idx
    
    def _format_answer(self, answer: List[Tuple]) -> str:
        """Format answer tuples to string"""
        return ", ".join([f"{name} at {coord} facing {orientation}" for name, coord, orientation in answer])
    
    def _generate_wrong_choice(self, correct_answer: List[Tuple], objects: List[Object],
                              gt_v_matrix: np.ndarray, gt_h_matrix: np.ndarray) -> List[Tuple]:
        """Generate a wrong choice that differs in coordinates or orientations"""
        
        # Randomly choose to change coordinates or orientations
        orientations = ["north", "east", "south", "west"]
        for _ in range(20):
            wrong_coords = [tuple(self.room.get_random_point(self.np_random)) for _ in range(len(objects))]
            wrong_v_matrix, wrong_h_matrix = DirectionalGraph.create_graph_from_coordinates(wrong_coords)
            if not (np.array_equal(wrong_v_matrix, gt_v_matrix) and np.array_equal(wrong_h_matrix, gt_h_matrix)):
                return [(name, coord, self.np_random.choice(orientations)) for (name, _, orientation), coord in zip(correct_answer, wrong_coords)]
        raise ValueError("Failed to generate wrong choice")
        

class SpatialManipulationTaskBase(BaseEvaluationTask):
    """Base class for tasks that manipulate objects and position agents"""
    
    def _position_agent_at(self, pos: np.ndarray) -> None:
        """Position agent at specified position and find good observation angle"""
        # decoupled: do not mutate room; just adjust internal agent pose for observation
        self.agent.pos = np.array(pos)
        
        # Find rotation with visible objects
        for rotation in self.np_random.permutation([0, 90, 180, 270]):
            RotateAction(rotation).execute(self.room, self.agent)
            if ObserveAction().execute(self.room, self.agent).data['visible_objects']:
                break
    
    def _take_full_observations(self, neglect_objects: List[str] = None) -> str:
        """Take observations with optional 180-degree rotation for full coverage"""
        messages = []
        obs_result = ObserveAction().execute(self.room, self.agent, neglect_objects=neglect_objects or [])
        messages.append(obs_result.message)
        
        # Rotate 180 and observe again if needed
        all_objects = [obj.name for obj in self.room.objects if obj.name not in (neglect_objects or [])]
        visible_objects = obs_result.data['visible_objects']
        if len(visible_objects) < len(all_objects):
            messages.extend([RotateAction(180).execute(self.room, self.agent).message, 
                           ObserveAction().execute(self.room, self.agent, neglect_objects=neglect_objects or []).message])
        
        return '\n'.join(messages)
    
    def _get_diagonal_position(self, target_obj: Object, ref_obj1: Object, ref_obj2: Object, neglect_trivial: bool = False) -> np.ndarray:
        """
        Calculate position that can deduce target object's position based on reference objects
        Final position is in the opposite quadrant of the joint (diagonal)
        TODO: add more constraints, make more challenging
        """
        joints = [(ref_obj1.pos[0], ref_obj2.pos[1]), (ref_obj2.pos[0], ref_obj1.pos[1])]
        if not neglect_trivial:
            joints.extend([tuple(ref_obj1.pos), tuple(ref_obj2.pos)])
        joint = self.np_random.choice(list(set(joints)))
        joint_dir = np.array(joint) - target_obj.pos
        jx, jy = joint
        # sample until hitting the opposite quadrant w.r.t. joint
        for _ in range(200):
            p = self.room.get_random_point(self.np_random)
            if (p[0] - jx) * joint_dir[0] >= 0 and (p[1] - jy) * joint_dir[1] >= 0:
                return p
        # fallback
        p = self.room.get_random_point(self.np_random)
        return p


class LocalizationEvaluationTask(SpatialManipulationTaskBase):
    """
    Evaluation task for localizing a target object from a diagonal position.
    
    1. Select target object and two reference objects
    2. Generate coordinate joints from reference objects  
    3. Choose diagonal position relative to target based on joint
    4. Move agent, rotate, observe (excluding target)
    5. Ask for target object direction

    TODO constraint on agent position
    """

    QUESTION_TEMPLATE = (
        "You observe the room from another view with new location and orientation.\n"
        "{observations}\n"
        "Based on your observations, what is the direction and orientation of the {target_name} from your current perspective?\n\n"
        "Hint: first reason about your current position and facing direction."
        "Answer format: (<horiz>, <vert>), <orientation>\n"
        "Choose the correct answer:\n"
        "{choices_text}\n\n"
        "IMPORTANT: You must respond with ONLY the single letter choice (E.g., A, B, C, D) in your answer\n\n"
    )
    
    def generate_question(self) -> str:
        if self.room is None:
            raise ValueError("Room must be set before generating question")
        
        # Step 1: Select target object
        target_obj = self.np_random.choice(self.room.objects)
        target_name = target_obj.name
        
        # Step 2: Select two other objects and create joints
        other_objects_candidates = [obj for obj in self.room.objects if obj.name != target_name and obj.has_orientation]
        assert len(other_objects_candidates) >= 2, "Need at least 2 objects with orientation for this task"
        
        ref_obj1, ref_obj2 = self.np_random.choice(other_objects_candidates, size=2, replace=False)
        
        # Step 3: Calculate diagonal position
        diagonal_pos = self._get_diagonal_position(target_obj, ref_obj1, ref_obj2)
        
        # Step 4: Position agent and take observations
        self._position_agent_at(diagonal_pos)
        observations = self._take_full_observations(neglect_objects=[target_name])
        
        # Calculate answer (direction + orientation)
        dir_rel = TotalRelationship.get_direction(tuple(self.agent.pos), tuple(target_obj.pos), anchor_ori=tuple(self.agent.ori))
        dir_str = dir_rel.to_string('ego')
        ori_rel = TotalRelationship.get_orientation(tuple(target_obj.ori), tuple(self.agent.ori))
        correct_answer = [dir_str, ori_rel.to_string('ego', kind='orientation')]
        
        # Generate choices
        choices, correct_idx = self.generate_choices(correct_answer)
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        
        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            observations=observations,
            target_name=target_name,
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        
        return self.eval_data.question
    
    def generate_choices(self, correct_answer: tuple) -> Tuple[List[str], int]:
        """Generate 4 localization choices"""
        h_dirs = [Dir.LEFT, Dir.RIGHT, Dir.SAME]
        v_dirs = [Dir.FORWARD, Dir.BACKWARD, Dir.SAME]
        orientations = ['forward', 'backward', 'right', 'left']
        choices = [f'{correct_answer[0]}, {correct_answer[1]}']
        
        while len(choices) < 4:
            wrong_dir = DirPair(self.np_random.choice(h_dirs), self.np_random.choice(v_dirs))
            choice = f"{DirectionRel.pair_to_string(wrong_dir, perspective='ego')}, {orientations[self.np_random.choice(range(len(orientations)))]}"
            if choice not in choices:
                choices.append(choice)
        
        self.np_random.shuffle(choices)
        correct_idx = choices.index(f'{correct_answer[0]}, {correct_answer[1]}')
        return choices, correct_idx


class FalseBeliefEvaluationTask(SpatialManipulationTaskBase):
    """
    Evaluation task for detecting object movement or rotation.
    
    Config options:
    - 'action_type': 'rotation' or 'movement' (default: 'rotation')
    """

    ROTATION_TEMPLATE = (
        "One object in the room has rotated.\n"
        "You observe the room from another view with new location and orientation.\n"
        "{observations}\n"
        "Which object rotated and by how many degrees clockwise?\n\n"
        "Answer format: <object_name>, <degrees>\n"
        "Choose the correct answer:\n"
        "{choices_text}\n\n"
        "IMPORTANT: You must respond with ONLY the single letter choice (E.g., A, B, C, D) in your answer\n\n"
    )
    
    MOVEMENT_TEMPLATE = (
        "One object in the room has moved.\n" 
        "You observe the room from another view with new location and orientation.\n"
        "{observations}\n"
        "Which object moved?\n\n"
        "Answer format: <object_name>\n"
        "Choose the correct answer:\n"
        "{choices_text}\n\n"
        "IMPORTANT: You must respond with ONLY the single letter choice (E.g., A, B, C, D) in your answer\n\n"
    )
    
    def generate_question(self) -> str:
        if self.room is None:
            raise ValueError("Room must be set before generating question")
        
        action_type = self.config.get('action_type', 'rotation')
        
        # Apply action based on type
        if action_type == 'movement':
            correct_answer, agent_pos = self._apply_movement()
            template = self.MOVEMENT_TEMPLATE
            self._position_agent_at(agent_pos)
        else:  # rotation
            correct_answer = self._apply_rotation()
            template = self.ROTATION_TEMPLATE
            self._position_agent_random()
        
        # Take observations and generate question
        observations = self._take_full_observations()
        choices, correct_idx = self.generate_choices(correct_answer)
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        
        self.eval_data.question = template.format(
            observations=observations,
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question
    
    def _apply_movement(self) -> Tuple[str, np.ndarray]:
        """Apply object movement and return moved object name and agent position"""
        target_obj = self.np_random.choice(self.room.objects)
        
        # Get reference objects for positioning
        other_objects = [obj for obj in self.room.objects if obj.name != target_obj.name and obj.has_orientation]
        ref_obj1, ref_obj2 = self.np_random.choice(other_objects, size=2, replace=False)
        
        # Calculate agent position and move target to opposite quadrant
        agent_pos = self._get_diagonal_position(target_obj, ref_obj1, ref_obj2, neglect_trivial=True)
        
        # Move target to opposite quadrant relative to agent
        rel_x, rel_y = target_obj.pos - agent_pos
        # move target to opposite quadrant relative to agent by sampling
        for _ in range(200):
            p = self.room.get_random_point(self.np_random)
            if (p[0] - agent_pos[0]) * rel_x >= 0 and (p[1] - agent_pos[1]) * rel_y >= 0:
                target_obj.pos = p
                break
        return target_obj.name, agent_pos
    
    def _apply_rotation(self) -> Tuple[str, str]:
        """Apply object rotation and return (object_name, degrees_str)"""
        oriented_objects = [obj for obj in self.room.objects if obj.has_orientation]
        assert len(oriented_objects) >= 2, "Need at least 2 objects with orientation for this task"
        target_obj = self.np_random.choice(oriented_objects)
        rotation_degrees = self.np_random.choice([90, 180, 270])
        
        rotations = {90: [[0,-1],[1,0]], 180: [[-1,0],[0,-1]], 270: [[0,1],[-1,0]]}
        target_obj.ori = target_obj.ori @ rotations[rotation_degrees]
        
        return target_obj.name, str(rotation_degrees)
    
    def _position_agent_random(self):
        """Position agent randomly for rotation-only tasks"""
        self._position_agent_at(self.room.get_random_point(self.np_random))
    
    def generate_choices(self, correct_answer: Any) -> Tuple[List[str], int]:
        """Generate 4 false belief choices"""
        choices = [str(correct_answer) if isinstance(correct_answer, str) else f'{correct_answer[0]}, {correct_answer[1]}']
        objects = [obj.name for obj in self.room.objects]
        
        while len(choices) < (4 if self.config.get('action_type', 'rotation') == 'rotation' else len(objects)):
            if isinstance(correct_answer, tuple):  # Rotation: (object, degrees)
                wrong_obj = self.np_random.choice(objects)
                wrong_deg = self.np_random.choice(['0', '90', '180', '270'])
                choice = f"{wrong_obj}, {wrong_deg}"
            else:  # Movement: object name
                choice = self.np_random.choice(objects)
            
            if choice not in choices:
                choices.append(choice)
        
        self.np_random.shuffle(choices)
        correct_idx = choices.index(str(correct_answer) if isinstance(correct_answer, str) else f'{correct_answer[0]}, {correct_answer[1]}')
        return choices, correct_idx




def test_task(task_class, room, agent, np_random):
    task = task_class(np_random=np_random, room=room, agent=agent)
    question = task.generate_question()
    print(question)
    print(f"Correct answer: {task.answer}")
    correct, info = task.evaluate('B')
    print(f"Self-evaluation: {correct}, Info: {info}")


if __name__ == "__main__":
    from ..core import CANDIDATE_OBJECTS
    from ..utils.room_utils import RoomGenerator
    from gymnasium.utils import seeding

    # Simple test setup
    room_config = {
        'room_size': [10, 10],
        'n_objects': 3,
        'candidate_objects': CANDIDATE_OBJECTS,
    }
    np_random = seeding.np_random(2)[0]
    room, agent = RoomGenerator.generate_room(**room_config, np_random=np_random)
    print(f"Room: {room}")
    
    BaseAction.set_field_of_view(90)

    # test_task(DirectionEvaluationTask, room, agent, np_random)

    # test_task(RotEvaluationTask, room, agent, np_random)

    # test_task(RotDualEvaluationTask, room, agent, np_random)


    # test_task(PovEvaluationTask, room, agent, np_random)

    test_task(E2AEvaluationTask, room, agent, np_random)

    # test_task(LocalizationEvaluationTask, room, agent, np_random)

    # test_task(FalseBeliefEvaluationTask, room, agent, np_random)