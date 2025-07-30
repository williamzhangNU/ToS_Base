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

from ..core.room import Room
from ..utils.eval_utilities import (
    multi_choice_eval_fn,
)
from ..core.constant import CANDIDATE_OBJECTS
from ..core.graph import DirectionalGraph
from ..core.object import Object, Agent
from ..core.relationship import DirPair, Dir, DirectionSystem
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
    
    def __init__(self, np_random: np.random.Generator, config: Dict[str, Any] = None, room: Room = None):
        """Initialize the evaluation task"""
        self.config = config or {}
        self.np_random = np_random
        self.room = room.copy() if room else None
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
    
    def _generate_reasoning(self, room: Room) -> str:
        """Generate reasoning for the evaluation task"""
        return f"Testing spatial reasoning for {self.__class__.__name__}"
    
    def format_choices(self, choices: List[str], correct_index: int) -> str:
        """Format choices as A. B. C. D. ... and return choice text"""
        return "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]), chr(65 + correct_index)
    
    @abstractmethod
    def generate_choices(self, correct_answer: Any, room: Room) -> Tuple[List[str], str]:
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
    """
    Evaluation task for checking spatial relationship between one randomly chosen object pair.
    
    Q: spatial relationship between a randomly chosen pair
    A: (<horiz>, <vert>)
    
    Randomly chooses one pair from all possible object pairs and asks for their relationship.
    """

    QUESTION_TEMPLATE = (
        "From a top-down view, what is the spatial relationship between the pair {obj_pairs_str}?\n"
        "For the pair (A, B), the relationship (<horizontal>, <vertical>) means A is <horizontal> of B and A is <vertical> of B.\n"
        "where <horizontal>: west, east, same; <vertical>: north, south, same\n"
        "Choose the correct answer:\n"
        "{choices_text}\n\n"
        "IMPORTANT: You must respond with ONLY the single letter choice (E.g., A, B, C, D) in your answer\n\n"
        "Answer: "
    )
    
    def generate_question(self) -> str:
        """Generate a question that asks about spatial relationship between one randomly chosen pair"""
        if self.room is None:
            raise ValueError("Room must be set before generating question")
        n = len(self.room.all_objects)
        
        # Generate all pairs with random order
        pairs = [(i, j) if self.np_random.random() >= 0.5 else (j, i) 
                for i in range(n) for j in range(i+1, n)]
        self.np_random.shuffle(pairs)

        pair = pairs[0]
        obj1, obj2 = self.room.all_objects[pair[0]], self.room.all_objects[pair[1]]
        _, correct_answer = self.room.get_direction(obj1.name, obj2.name, perspective='allo')
        
        # Generate choices
        choices, correct_idx = self.generate_choices(correct_answer, self.room)
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            obj_pairs_str=f"({obj1.name}, {obj2.name})",
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning(room)
        
        return self.eval_data.question
    
    def generate_choices(self, correct_answer: str, room: Room) -> Tuple[List[str], int]:
        """Generate 4 direction choices"""
        h_dirs = ['west', 'east', 'same']
        v_dirs = ['north', 'south', 'same']
        choices = [correct_answer]
        while len(choices) < 4:
            random_h = self.np_random.choice(h_dirs)
            random_v = self.np_random.choice(v_dirs)
            choice = f"({random_h}, {random_v})"
            if choice not in choices:
                choices.append(choice)
        
        self.np_random.shuffle(choices)
        correct_idx = choices.index(str(correct_answer))
        return choices, correct_idx


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
        "You will perform a full 360-degree rotation by continuously turning {turn_direction} in place.\n"
        "As you rotate, imagine a narrow spotlight beam projecting straight ahead from your current viewpoint.\n"
        "Your task is to identify the sequence of objects that become visible during the full rotation.\n\n"
        "Instructions:\n"
        "- Object at your current position is not included\n"
        "- Objects are visible ONLY when you turn to face them directly\n\n"
        "Choose the correct sequence:\n"
        "{choices_text}\n\n"
        "IMPORTANT: You must respond with ONLY the single letter choice (E.g., A, B, C, D) in your answer\n\n"
        "Answer: "
    )
    MOVEMENT_TEMPLATE = (
        "You moved to the same position as {move_obj_name}.\n"
    )
    TURN_TEMPLATE = (
        "You turned clockwise {degree} degrees.\n"
    )
    
    def generate_question(self) -> str:

        def get_angle(pos: np.ndarray) -> float:
            """Get angle from positive y-axis"""
            angle = np.arctan2(pos[0], pos[1])
            if angle < 0:
                angle += 2 * np.pi  
            return angle
        
        if self.room is None:
            raise ValueError("Room must be set before generating question")
        self.room = self.room.copy()
        # turn_direction = self.config.get('turn_direction', 'clockwise')
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
            MoveAction(move_obj.name).execute(self.room)
        if if_turn:
            degree = self.np_random.choice([90, 180, 270])
            turn_prompt = self.TURN_TEMPLATE.format(degree=degree)
            RotateAction(degree).execute(self.room)

        objects = self.room.objects.copy()
        direct_front_objects = [obj for obj in objects if obj.pos[0] == 0 and obj.pos[1] >= 0]
        other_objects = [obj for obj in objects if obj.name not in [obj.name for obj in direct_front_objects]]
        other_objects.sort(key=lambda x: get_angle(x.pos), reverse=(turn_direction == 'counterclockwise'))

        correct_answer = [obj.name for obj in direct_front_objects] + [obj.name for obj in other_objects]
        
        # Generate choices
        choices, correct_idx = self.generate_choices(correct_answer, self.room)
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = movement_prompt + turn_prompt + self.QUESTION_TEMPLATE.format(
            turn_direction=turn_direction,
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning(self.room)
        return self.eval_data.question
    
    def generate_choices(self, correct_answer: list, room: Room) -> Tuple[List[str], int]:
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


class CircularRotEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for circular movement around the center.
    
    1. Agent moves far from center so distance > any object's distance to center
    2. Agent faces center and rotates around it
    3. Uses same logic as mental rotation at original position
    """

    QUESTION_TEMPLATE = (
        "Assume your original position is the center of the room.\n"
        "Move {direction} from the center until you are farther away than any object in the room.\n"
        "Now face the center and walk {turn_direction} in a circle around it.\n"
        "Imagine a narrow spotlight beam projecting straight ahead from your current viewpoint.\n"
        "Your task is to identify the sequence of objects that become visible during the full rotation.\n\n"
        "Instructions:\n"
        "- As you circle around, only count objects that are directly in front of you AND between you and the center.\n"
        "- Objects are visible ONLY when you turn to face them directly\n\n"
        "Choose the correct sequence:\n"
        "{choices_text}\n\n"
        "IMPORTANT: You must respond with ONLY the single letter choice (E.g., A, B, C, D) in your answer\n\n"
        "Answer: "
    )
    
    def generate_question(self) -> str:
        if self.room is None:
            raise ValueError("Room must be set before generating question")
        if self.room.agent is None:
            raise ValueError("Agent must be present for circular rotation task")
            
        self.room = self.room.copy()
        
        # Choose direction to move from center
        direction = self.np_random.choice(['front', 'right', 'left', 'back'])
        turn_direction = self.config.get('turn_direction', 'clockwise')
        RotateAction({'front': 0, 'right': 90, 'left': 270, 'back': 180}[direction]).execute(self.room)
        
        # Use same rotation logic as RotEvaluationTask from center
        def get_angle(pos: np.ndarray) -> float:
            """Get angle from positive y-axis"""
            angle = np.arctan2(pos[0], pos[1])
            if angle < 0:
                angle += 2 * np.pi  
            return angle
        
        objects = self.room.objects.copy()
        objects.sort(key=lambda x: get_angle(x.pos), reverse=(turn_direction == 'counterclockwise'))
        
        correct_answer = [obj.name for obj in objects]
        
        # Generate choices
        choices, correct_idx = self.generate_choices(correct_answer, self.room)
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        
        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            direction=direction,
            turn_direction=turn_direction,
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = f"Agent rotates {turn_direction} around center from {direction}"
        
        return self.eval_data.question
    
    def generate_choices(self, correct_answer: list, room: Room) -> Tuple[List[str], int]:
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
        "You performed a complete 360-degree rotation at your position.\n"
        "During the rotation, these objects appeared directly in front of you in this order:\n"
        "{object_sequence}\n\n"
        "Based on this sequence, in which direction did you rotate?\n\n"
        "Choose the correct answer:\n"
        "{choices_text}\n\n"
        "IMPORTANT: You must respond with ONLY the single letter choice (E.g., A, B, C, D) in your answer\n\n"
        "Answer: "
    )
    
    def generate_question(self) -> str:

        def get_angle(pos: np.ndarray) -> float:
            """Get angle from positive y-axis"""
            angle = np.arctan2(pos[0], pos[1])
            if angle < 0:
                angle += 2 * np.pi  
            return angle
        
        if self.room is None:
            raise ValueError("Room must be set before generating question")
        
        # Randomly choose rotation direction
        turn_direction = self.np_random.choice(['clockwise', 'counterclockwise'])
        
        # Sort objects based on rotation direction
        objects = self.room.objects
        objects.sort(key=lambda x: get_angle(x.pos), reverse=(turn_direction == 'counterclockwise'))
        
        # Create sequence string
        object_names = [obj.name for obj in objects]
        object_sequence = ", ".join(object_names)
        
        # Generate choices
        choices, correct_idx = self.generate_choices(turn_direction, self.room)
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        
        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            object_sequence=object_sequence,
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning(self.room)
        
        return self.eval_data.question
    
    def generate_choices(self, correct_answer: str, room: Room) -> Tuple[List[str], int]:
        """Generate 4 rotation direction choices"""
        opposite = 'counterclockwise' if correct_answer == 'clockwise' else 'clockwise'
        choices = [correct_answer, opposite]
        
        self.np_random.shuffle(choices)
        correct_idx = choices.index(correct_answer)
        return choices, correct_idx


class PovEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for perspective taking questions.
    
    Q: Ask spatial relationship between object a from the perspective of object c
    A: <dir>
    
    Tests ability to take another object's viewpoint for spatial reasoning.
    """

    QUESTION_TEMPLATE = (
        "Imagine you are at the same position and orientation as the {anchor_obj_name}.\n"
        "From this perspective, what is the direction of the {obj_name}?\n\n"
        "For object A, the relationship (<horizontal>, <vertical>) means A is <horizontal> and <vertical> of you.\n"
        "Choose the correct answer:\n"
        "{choices_text}\n\n"
        "IMPORTANT: You must respond with ONLY the single letter choice (E.g., A, B, C, D) in your answer\n\n"
        "Answer: "
    )
    
    def generate_question(self) -> str:
        if self.room is None:
            raise ValueError("Room must be set before generating question")
        self.room = self.room.copy()
        
        # Choose three different objects
        obj_idx = self.np_random.integers(0, len(self.room.all_objects))
        
        # Choose anchor object that has orientation
        oriented_objects = [i for i, obj in enumerate(self.room.all_objects) if obj.has_orientation]
        assert len(oriented_objects) > 0, "No objects with orientation found for perspective taking task"
        anchor_obj_idx = self.np_random.choice(oriented_objects)
        
        while obj_idx == anchor_obj_idx:
            obj_idx = self.np_random.integers(0, len(self.room.all_objects))
        obj_name, anchor_obj_name = self.room.all_objects[obj_idx].name, self.room.all_objects[anchor_obj_idx].name

        _, correct_answer = self.room.get_direction(obj_name, anchor_obj_name, anchor_name=anchor_obj_name)
        
        # Generate choices
        choices, correct_idx = self.generate_choices(correct_answer, self.room)
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            anchor_obj_name=anchor_obj_name,
            obj_name=obj_name,
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning(self.room)
        return self.eval_data.question
    
    def generate_choices(self, correct_answer: str, room: Room) -> Tuple[List[str], int]:
        """Generate 4 direction choices"""
        h_dirs = ['west', 'east', 'same']
        v_dirs = ['north', 'south', 'same']
        choices = [correct_answer]
        while len(choices) < 4:
            random_h = self.np_random.choice(h_dirs)
            random_v = self.np_random.choice(v_dirs)
            choice = f"({random_h}, {random_v})"
            if choice not in choices:
                choices.append(choice)
        
        self.np_random.shuffle(choices)
        correct_idx = choices.index(str(correct_answer))
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
        "You are at position (0, 0) and facing north.\n"
        "Your front is the positive y-axis, your right is the positive x-axis.\n"
        "What are the coordinates and orientations of these objects: {object_names}?\n\n"
        "Answer format: [obj at (x, y) facing orientation, ...] where orientation is north/east/south/west\n"
        "Choose the correct answer:\n"
        "{choices_text}\n\n"
        "IMPORTANT: You must respond with ONLY the single letter choice (E.g., A, B, C, D) in your answer\n\n"
        "Answer: "
    )
    
    def generate_question(self) -> str:
        if self.room is None:
            raise ValueError("Room must be set before generating question")
        
        # Get objects excluding agent and objects at same position as agent
        agent_pos = self.room.agent.pos if self.room.agent else None
        objects = [obj for obj in self.room.objects 
                  if agent_pos is None or not np.array_equal(obj.pos, agent_pos)]
        
        self.np_random.shuffle(objects)
        object_names = [obj.name for obj in objects]
        
        # Create correct answer: (object_name, coord, orientation)
        correct_answer = [(obj.name, tuple(obj.pos.astype(int)), self._get_orientation_string(obj)) 
                         for obj in objects]
        
        # Generate choices
        choices, correct_idx = self.generate_choices(correct_answer, objects, self.room)
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        
        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            object_names=", ".join(object_names),
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning(self.room)
        return self.eval_data.question
    
    def _get_orientation_string(self, obj: Object) -> str:
        """Convert object orientation to north/east/south/west string"""
        return {(0, 1): "north", (1, 0): "east", (0, -1): "south", (-1, 0): "west"}[tuple(obj.ori)]
    
    def generate_choices(self, correct_answer: List[Tuple], objects: List[Object], room: Room) -> Tuple[List[str], int]:
        """Generate 4 coordinate and orientation choices"""
        correct_str = self._format_answer(correct_answer)
        choices = [correct_str]
        
        # Create ground truth graph for comparison
        coords = [obj.pos for obj in objects]
        gt_v_matrix, gt_h_matrix = DirectionalGraph.create_graph_from_coordinates(coords)
        
        for _ in range(3):
            wrong_answer = self._generate_wrong_choice(correct_answer, objects, room, gt_v_matrix, gt_h_matrix)
            choices.append(self._format_answer(wrong_answer))
        
        self.np_random.shuffle(choices)
        correct_idx = choices.index(correct_str)
        return choices, correct_idx
    
    def _format_answer(self, answer: List[Tuple]) -> str:
        """Format answer tuples to string"""
        return ", ".join([f"{name} at {coord} facing {orientation}" for name, coord, orientation in answer])
    
    def _generate_wrong_choice(self, correct_answer: List[Tuple], objects: List[Object], room: Room,
                              gt_v_matrix: np.ndarray, gt_h_matrix: np.ndarray) -> List[Tuple]:
        """Generate a wrong choice that differs in coordinates or orientations"""
        
        # Randomly choose to change coordinates or orientations
        orientations = ["north", "east", "south", "west"]
        min_x, max_x, min_y, max_y = room.get_boundary()
        for _ in range(10):  # Try up to 10 times to find different coordinates
            wrong_coords = [(self.np_random.integers(int(min_x), int(max_x) + 1),
                            self.np_random.integers(int(min_y), int(max_y) + 1)) for _ in range(len(objects))]
            
            # Check if coordinates create different spatial relationships
            wrong_v_matrix, wrong_h_matrix = DirectionalGraph.create_graph_from_coordinates(wrong_coords)
            if not (np.array_equal(wrong_v_matrix, gt_v_matrix) and np.array_equal(wrong_h_matrix, gt_h_matrix)):
                return [(name, coord, self.np_random.choice(orientations)) for (name, _, orientation), coord in zip(correct_answer, wrong_coords)]
        raise ValueError("Failed to generate wrong choice")
        

class SpatialManipulationTaskBase(BaseEvaluationTask):
    """Base class for tasks that manipulate objects and position agents"""
    
    def _position_agent_at(self, room: Room, pos: np.ndarray) -> None:
        """Position agent at specified position and find good observation angle"""
        room.add_object(Object(name='tmp_obj', pos=pos))
        MoveAction('tmp_obj').execute(room, move_anyway=True)
        room.remove_object('tmp_obj')
        
        # Find rotation with visible objects
        for rotation in self.np_random.permutation([0, 90, 180, 270]):
            RotateAction(rotation).execute(room)
            if ObserveAction().execute(room).data['visible_objects']:
                break
    
    def _take_full_observations(self, room: Room, neglect_objects: List[str] = None) -> str:
        """Take observations with optional 180-degree rotation for full coverage"""
        messages = []
        obs_result = ObserveAction().execute(room, neglect_objects=neglect_objects or [])
        messages.append(obs_result.message)
        
        # Rotate 180 and observe again if needed
        all_objects = [obj.name for obj in room.objects if obj.name not in (neglect_objects or [])]
        visible_objects = obs_result.data['visible_objects']
        if len(visible_objects) < len(all_objects):
            messages.extend([RotateAction(180).execute(room).message, 
                           ObserveAction().execute(room, neglect_objects=neglect_objects or []).message])
        
        return '\n'.join(messages)
    
    def _get_diagonal_position(self, room: Room, target_obj: Object, ref_obj1: Object, ref_obj2: Object, neglect_trivial: bool = False) -> np.ndarray:
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
        min_x, max_x, min_y, max_y = room.get_boundary()
        jx, jy = joint
        
        # Choose opposite quadrant
        x_bounds = (jx, max_x) if joint_dir[0] > 0 else (min_x, jx)
        y_bounds = (jy, max_y) if joint_dir[1] > 0 else (min_y, jy)
        return np.round(np.array([self.np_random.uniform(*x_bounds), self.np_random.uniform(*y_bounds)]), 1)


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
        "You observe the room from another view.\n"
        "{observations}\n"
        "Based on your observations, what is the direction and orientation of the {target_name} from your current perspective?\n\n"
        "Answer format: (<horiz>, <vert>), <orientation>\n"
        "Choose the correct answer:\n"
        "{choices_text}\n\n"
        "IMPORTANT: You must respond with ONLY the single letter choice (E.g., A, B, C, D) in your answer\n\n"
        "Answer: "
    )
    
    def generate_question(self, room: Room) -> str:
        room = room.copy()
        
        if room.agent is None:
            raise ValueError("Agent must be present for this task")
        
        # Step 1: Select target object
        target_obj = self.np_random.choice(room.objects)
        target_name = target_obj.name
        
        # Step 2: Select two other objects and create joints
        other_objects_candidates = [obj for obj in room.objects if obj.name != target_name and obj.has_orientation]
        assert len(other_objects_candidates) >= 2, "Need at least 2 objects with orientation for this task"
        
        ref_obj1, ref_obj2 = self.np_random.choice(other_objects_candidates, size=2, replace=False)
        
        # Step 3: Calculate diagonal position
        diagonal_pos = self._get_diagonal_position(room, target_obj, ref_obj1, ref_obj2)
        
        # Step 4: Position agent and take observations
        self._position_agent_at(room, diagonal_pos)
        observations = self._take_full_observations(room, neglect_objects=[target_name])
        
        # Calculate answer (direction + orientation)
        dir_pair, _ = room.get_direction(target_name, room.agent.name, perspective='ego')
        _, orientation_str = room.get_orientation(target_name, room.agent.name)
        correct_answer = [
            DirectionSystem.to_string(dir_pair.horiz, perspective='ego'), 
            DirectionSystem.to_string(dir_pair.vert, perspective='ego'), 
            orientation_str
        ]
        
        # Generate choices
        choices, correct_idx = self.generate_choices(correct_answer, room)
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        
        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            observations=observations,
            target_name=target_name,
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning(room)
        
        return self.eval_data.question
    
    def generate_choices(self, correct_answer: tuple, room: Room) -> Tuple[List[str], int]:
        """Generate 4 localization choices"""
        h_dirs = ['left', 'right', 'same']
        v_dirs = ['front', 'back', 'same']
        orientations = ['forward', 'backward', 'right', 'left']
        choices = [f'({correct_answer[0]}, {correct_answer[1]}), {correct_answer[2]}']
        
        while len(choices) < 4:
            wrong_h, wrong_v, wrong_o = self.np_random.choice(h_dirs), self.np_random.choice(v_dirs), self.np_random.choice(orientations)
            choice = f"({wrong_h}, {wrong_v}), {wrong_o}"
            if choice not in choices:
                choices.append(choice)
        
        self.np_random.shuffle(choices)
        correct_idx = choices.index(f'({correct_answer[0]}, {correct_answer[1]}), {correct_answer[2]}')
        return choices, correct_idx


class FalseBeliefEvaluationTask(SpatialManipulationTaskBase):
    """
    Evaluation task for detecting object movement or rotation.
    
    Config options:
    - 'action_type': 'rotation' or 'movement' (default: 'rotation')
    """

    ROTATION_TEMPLATE = (
        "One object in the room has rotated.\n"
        "You observe the room from another view.\n"
        "{observations}\n"
        "Which object rotated and by how many degrees clockwise?\n\n"
        "Answer format: <object_name>, <degrees>\n"
        "Choose the correct answer:\n"
        "{choices_text}\n\n"
        "IMPORTANT: You must respond with ONLY the single letter choice (E.g., A, B, C, D) in your answer\n\n"
        "Answer: "
    )
    
    MOVEMENT_TEMPLATE = (
        "One object in the room has moved.\n" 
        "You observe the room from another view.\n"
        "{observations}\n"
        "Which object moved?\n\n"
        "Answer format: <object_name>\n"
        "Choose the correct answer:\n"
        "{choices_text}\n\n"
        "IMPORTANT: You must respond with ONLY the single letter choice (E.g., A, B, C, D) in your answer\n\n"
        "Answer: "
    )
    
    def generate_question(self) -> str:
        if self.room is None:
            raise ValueError("Room must be set before generating question")
        
        if self.room.agent is None:
            raise ValueError("Agent must be present for this task")
        
        action_type = self.config.get('action_type', 'rotation')
        
        # Apply action based on type
        if action_type == 'movement':
            correct_answer, agent_pos = self._apply_movement(self.room)
            template = self.MOVEMENT_TEMPLATE
            self._position_agent_at(self.room, agent_pos)
        else:  # rotation
            correct_answer = self._apply_rotation(self.room)
            template = self.ROTATION_TEMPLATE
            self._position_agent_random(self.room)
        
        # Take observations and generate question
        observations = self._take_full_observations(self.room)
        choices, correct_idx = self.generate_choices(correct_answer, self.room)
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        
        self.eval_data.question = template.format(
            observations=observations,
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning(self.room)
        return self.eval_data.question
    
    def _apply_movement(self, room: Room) -> Tuple[str, np.ndarray]:
        """Apply object movement and return moved object name and agent position"""
        target_obj = self.np_random.choice(room.objects)
        
        # Get reference objects for positioning
        other_objects = [obj for obj in room.objects if obj.name != target_obj.name and obj.has_orientation]
        ref_obj1, ref_obj2 = self.np_random.choice(other_objects, size=2, replace=False)
        
        # Calculate agent position and move target to opposite quadrant
        agent_pos = self._get_diagonal_position(room, target_obj, ref_obj1, ref_obj2, neglect_trivial=True)
        
        # Move target to opposite quadrant relative to agent
        rel_x, rel_y = target_obj.pos - agent_pos
        min_x, max_x, min_y, max_y = room.get_boundary()
        
        x_range = (agent_pos[0], max_x) if rel_x < 0 else (min_x, agent_pos[0])
        y_range = (agent_pos[1], max_y) if rel_y < 0 else (min_y, agent_pos[1])
        
        target_obj.pos = np.array([self.np_random.uniform(*x_range), self.np_random.uniform(*y_range)])
        return target_obj.name, agent_pos
    
    def _apply_rotation(self, room: Room) -> Tuple[str, str]:
        """Apply object rotation and return (object_name, degrees_str)"""
        oriented_objects = [obj for obj in room.objects if obj.has_orientation]
        assert len(oriented_objects) >= 2, "Need at least 2 objects with orientation for this task"
        target_obj = self.np_random.choice(oriented_objects)
        rotation_degrees = self.np_random.choice([90, 180, 270])
        
        rotations = {90: [[0,-1],[1,0]], 180: [[-1,0],[0,-1]], 270: [[0,1],[-1,0]]}
        target_obj.ori = target_obj.ori @ rotations[rotation_degrees]
        
        return target_obj.name, str(rotation_degrees)
    
    def _position_agent_random(self, room: Room):
        """Position agent randomly for rotation-only tasks"""
        min_x, max_x, min_y, max_y = room.get_boundary()
        agent_pos = np.array([self.np_random.uniform(min_x, max_x), self.np_random.uniform(min_y, max_y)])
        self._position_agent_at(room, agent_pos)
    
    def generate_choices(self, correct_answer: Any, room: Room) -> Tuple[List[str], int]:
        """Generate 4 false belief choices"""
        choices = [str(correct_answer) if isinstance(correct_answer, str) else f'{correct_answer[0]}, {correct_answer[1]}']
        objects = [obj.name for obj in room.objects]
        
        while len(choices) < (4 if self.config['action_type'] == 'rotation' else len(objects)):
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




def test_task(task_class, room, np_random):
    task = task_class(np_random=np_random)
    question = task.generate_question(room)
    print(question)
    print(f"Correct answer: {task.answer}")
    correct, info = task.evaluate('B')
    print(f"Self-evaluation: {correct}, Info: {info}")


if __name__ == "__main__":
    from ..core import CANDIDATE_OBJECTS
    from ..utils.room_utils import generate_room
    from gymnasium.utils import seeding

    # Simple test setup
    room_config = {
        'room_range': [-10, 10],
        'n_objects': 3,
        'candidate_objects': CANDIDATE_OBJECTS,
        'generation_type': 'pov',
    }
    np_random = seeding.np_random(2)[0]
    room = generate_room(**room_config, np_random=np_random)
    print(f"Room: {room}")
    
    BaseAction.set_field_of_view(180)

    test_task(DirectionEvaluationTask, room, np_random)

    test_task(RotEvaluationTask, room, np_random)

    test_task(RotDualEvaluationTask, room, np_random)

    test_task(CircularRotEvaluationTask, room, np_random)

    test_task(PovEvaluationTask, room, np_random)

    test_task(E2AEvaluationTask, room, np_random)

    test_task(LocalizationEvaluationTask, room, np_random)

    test_task(FalseBeliefEvaluationTask, room, np_random)