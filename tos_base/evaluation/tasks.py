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
    
    def __init__(self, np_random: np.random.Generator, config: Dict[str, Any] = None):
        """Initialize the evaluation task"""
        self.config = config or {}
        self.np_random = np_random
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
    def generate_question(self, room: Room) -> str:
        """Generate evaluation questions based on the room state"""
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
        "Choose the correct answer:\n"
        "{choices_text}\n\n"
        "IMPORTANT: You must respond with ONLY the single letter choice (E.g., A, B, C, D) in your answer\n\n"
        "Answer: "
    )
    
    def generate_question(self, room: Room) -> str:
        """Generate a question that asks about spatial relationship between one randomly chosen pair"""
        room = room.copy()
        n = len(room.all_objects)
        
        # Generate all pairs with random order
        pairs = [(i, j) if self.np_random.random() >= 0.5 else (j, i) 
                for i in range(n) for j in range(i+1, n)]
        self.np_random.shuffle(pairs)

        pair = pairs[0]
        obj1, obj2 = room.all_objects[pair[0]], room.all_objects[pair[1]]
        _, correct_answer = room.get_direction(obj1.name, obj2.name, perspective='allo')
        
        # Generate choices
        choices, correct_idx = self.generate_choices(correct_answer, room)
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
            if choice == str(correct_answer):
                continue
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
    
    def generate_question(self, room: Room) -> str:

        def get_angle(pos: np.ndarray) -> float:
            """Get angle from positive y-axis"""
            angle = np.arctan2(pos[0], pos[1])
            if angle < 0:
                angle += 2 * np.pi  
            return angle
        
        room = room.copy()
        turn_direction = self.config.get('turn_direction', 'clockwise')
        if_move = self.config.get('if_move', False)
        if_turn = self.config.get('if_turn', False)
        
        movement_prompt = ""
        turn_prompt = ""
        neglect_objects = []
        if if_move:
            move_obj = self.np_random.choice(room.objects)
            movement_prompt = self.MOVEMENT_TEMPLATE.format(move_obj_name=move_obj.name)
            neglect_objects.append(move_obj.name)
            MoveAction(move_obj.name).execute(room)
        if if_turn:
            degree = self.np_random.choice([90, 180, 270])
            turn_prompt = self.TURN_TEMPLATE.format(degree=degree)
            RotateAction(degree).execute(room)

        objects = room.objects.copy()
        direct_front_objects = [obj for obj in objects if obj.pos[0] == 0 and obj.pos[1] >= 0]
        other_objects = [obj for obj in objects if obj.name not in [obj.name for obj in direct_front_objects]]
        other_objects.sort(key=lambda x: get_angle(x.pos), reverse=(turn_direction == 'counterclockwise'))

        correct_answer = [obj.name for obj in direct_front_objects] + [obj.name for obj in other_objects]
        
        # Generate choices
        choices, correct_idx = self.generate_choices(correct_answer, room)
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = movement_prompt + turn_prompt + self.QUESTION_TEMPLATE.format(
            turn_direction=turn_direction,
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning(room)
        return self.eval_data.question
    
    def generate_choices(self, correct_answer: list, room: Room) -> Tuple[List[str], int]:
        """Generate 4 object sequence choices"""
        correct_answer_str = ", ".join(correct_answer)
        choices = [correct_answer_str]
        
        for _ in range(3):
            wrong_list = correct_answer.copy()
            while ", ".join(wrong_list) == correct_answer_str:
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
    
    def generate_question(self, room: Room) -> str:
        if room.agent is None:
            raise ValueError("Agent must be present for circular rotation task")
            
        room = room.copy()
        
        # Choose direction to move from center
        direction = self.np_random.choice(['front', 'right', 'left', 'back'])
        turn_direction = self.config.get('turn_direction', 'clockwise')
        RotateAction({'front': 0, 'right': 90, 'left': 270, 'back': 180}[direction]).execute(room)
        
        # Use same rotation logic as RotEvaluationTask from center
        def get_angle(pos: np.ndarray) -> float:
            """Get angle from positive y-axis"""
            angle = np.arctan2(pos[0], pos[1])
            if angle < 0:
                angle += 2 * np.pi  
            return angle
        
        objects = room.objects.copy()
        objects.sort(key=lambda x: get_angle(x.pos), reverse=(turn_direction == 'counterclockwise'))
        
        correct_answer = [obj.name for obj in objects]
        
        # Generate choices
        choices, correct_idx = self.generate_choices(correct_answer, room)
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
            while ", ".join(wrong_list) == correct_answer_str:
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
    
    def generate_question(self, room: Room) -> str:

        def get_angle(pos: np.ndarray) -> float:
            """Get angle from positive y-axis"""
            angle = np.arctan2(pos[0], pos[1])
            if angle < 0:
                angle += 2 * np.pi  
            return angle
        
        room = room.copy()
        
        # Randomly choose rotation direction
        turn_direction = self.np_random.choice(['clockwise', 'counterclockwise'])
        
        # Sort objects based on rotation direction
        objects = room.objects
        objects.sort(key=lambda x: get_angle(x.pos), reverse=(turn_direction == 'counterclockwise'))
        
        # Create sequence string
        object_names = [obj.name for obj in objects]
        object_sequence = ", ".join(object_names)
        
        # Generate choices
        choices, correct_idx = self.generate_choices(turn_direction, room)
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        
        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            object_sequence=object_sequence,
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning(room)
        
        return self.eval_data.question
    
    def generate_choices(self, correct_answer: str, room: Room) -> Tuple[List[str], int]:
        """Generate 4 rotation direction choices"""
        opposite = 'counterclockwise' if correct_answer == 'clockwise' else 'clockwise'
        choices = [correct_answer, opposite, opposite, opposite]  # 1 correct, 3 wrong
        
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
    
    def generate_question(self, room: Room) -> str:
        room = room.copy()
        
        # Choose three different objects
        obj_idx = self.np_random.integers(0, len(room.all_objects))
        
        # Choose anchor object that has orientation
        oriented_objects = [i for i, obj in enumerate(room.all_objects) if obj.has_orientation]
        assert len(oriented_objects) > 0, "No objects with orientation found for perspective taking task"
        anchor_obj_idx = self.np_random.choice(oriented_objects)
        
        while obj_idx == anchor_obj_idx:
            obj_idx = self.np_random.integers(0, len(room.all_objects))
        obj_name, anchor_obj_name = room.all_objects[obj_idx].name, room.all_objects[anchor_obj_idx].name

        _, correct_answer = room.get_direction(obj_name, anchor_obj_name, anchor_name=anchor_obj_name)
        
        # Generate choices
        choices, correct_idx = self.generate_choices(correct_answer, room)
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            anchor_obj_name=anchor_obj_name,
            obj_name=obj_name,
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
            if choice == str(correct_answer):
                continue
            choices.append(choice)
        
        self.np_random.shuffle(choices)
        correct_idx = choices.index(str(correct_answer))
        return choices, correct_idx


class E2AEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for allo2ego questions.
    
    Q: Given coordinates, what objects are at these positions?
    A: [<obj1>, <obj2>, ...]
    
    Tests conversion from allocentric coordinates to object identification.
    """

    QUESTION_TEMPLATE = (
        "Assume your front is the positive y-axis, your right is the positive x-axis.\n"
        "You observe objects at these coordinates: {coordinates}.\n"
        "What objects are at these coordinates in the same order?\n\n"
        "Choose the correct answer:\n"
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
    
    def generate_question(self, room: Room) -> str:
        room = room.copy()
    
        if_move, if_turn = self.config.get('if_move', False), self.config.get('if_turn', False)
        movement_prompt = ""
        turn_prompt = ""
        # Step 1: Agent moves to a randomly chosen object
        if if_move:
            chosen_obj = self.np_random.choice(room.objects)
            move_action = MoveAction(chosen_obj.name)
            move_action.execute(room, move_anyway=True)
            movement_prompt = self.MOVEMENT_TEMPLATE.format(move_obj_name=chosen_obj.name)
                
        # Step 2: Agent rotates by a random degree
        if if_turn:
            degree = self.np_random.choice([0, 90, 180, 270])
            rotate_action = RotateAction(degree)
            rotate_action.execute(room)
            turn_prompt = self.TURN_TEMPLATE.format(degree=degree)
        
        # Step 3: Get objects and add offset to coordinates
        objects = [obj for obj in copy.deepcopy(room.all_objects) 
                  if room.agent is None or obj.name != room.agent.name]
        self.np_random.shuffle(objects)
        
        # Add random offset to all coordinates
        offset = self.np_random.uniform(-5, 5, size=2).astype(int)
        offset_coords = [tuple(obj.pos + offset) for obj in objects]

        correct_answer = [obj.name for obj in objects]
        
        # Generate choices
        choices, correct_idx = self.generate_choices(correct_answer, room)
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = movement_prompt + turn_prompt + self.QUESTION_TEMPLATE.format(
            coordinates=str(offset_coords),
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning(room)
        return self.eval_data.question
    
    def generate_choices(self, correct_answer: list, room: Room) -> Tuple[List[str], int]:
        """Generate 4 object sequence choices"""
        correct_answer_str = ", ".join(correct_answer)
        choices = [correct_answer_str]
        
        for _ in range(3):
            wrong_list = correct_answer.copy()
            while ", ".join(wrong_list) == correct_answer_str:
                self.np_random.shuffle(wrong_list)
            choices.append(", ".join(wrong_list))
        
        self.np_random.shuffle(choices)
        correct_idx = choices.index(correct_answer_str)
        return choices, correct_idx


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
        correct_answer = (
            DirectionSystem.to_string(dir_pair.horiz, perspective='ego'), 
            DirectionSystem.to_string(dir_pair.vert, perspective='ego'), 
            orientation_str
        )
        
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
        choices = [str(correct_answer)]
        
        while len(choices) < 4:
            wrong_h = self.np_random.choice(h_dirs)
            wrong_v = self.np_random.choice(v_dirs)
            wrong_o = self.np_random.choice(orientations)
            choice = f"({wrong_h}, {wrong_v}, {wrong_o})"
            if choice == str(correct_answer):
                continue
            choices.append(choice)
        
        self.np_random.shuffle(choices)
        correct_idx = choices.index(str(correct_answer))
        return choices, correct_idx


class FalseBeliefEvaluationTask(SpatialManipulationTaskBase):
    """
    Evaluation task for detecting object movement and/or rotation.
    
    Config options:
    - 'action_type': 'rotation', 'movement', or 'both' (default: 'rotation')

    TODO prove correctness of the task
    """

    ROTATION_TEMPLATE = (
        "One object in the room has rotated.\n"
        "You observe the room from another view.\n"
        "{observations}\n"
        "Which object rotated and by how many degrees clockwise?\n\n"
        "Choose the correct answer:\n"
        "{choices_text}\n\n"
        "Answer: "
    )
    
    MOVEMENT_TEMPLATE = (
        "One object in the room has moved.\n" 
        "You observe the room from another view.\n"
        "{observations}\n"
        "Which object moved?\n\n"
        "Choose the correct answer:\n"
        "{choices_text}\n\n"
        "Answer: "
    )
    
    BOTH_TEMPLATE = (
        "One object moved and one object rotated (may be the same object).\n"
        "You observe the room from another view.\n"
        "{observations}\n"
        "Which object moved and which object rotated and by how many degrees clockwise?\n\n"
        "Choose the correct answer:\n"
        "{choices_text}\n\n"
        "Answer: "
    )
    
    def generate_question(self, room: Room) -> str:
        room = room.copy()
        
        if room.agent is None:
            raise ValueError("Agent must be present for this task")
        
        action_type = self.config.get('action_type', 'rotation')
        moved_obj_name = None
        rotated_obj_name = None
        rotation_degrees = None
        
        # Step 1: Apply actions based on type
        agent_pos = None
        if action_type in ['movement', 'both']:
            moved_obj_name, agent_pos = self._apply_movement(room)
        
        if action_type in ['rotation', 'both']:
            rotated_obj_name, rotation_degrees = self._apply_rotation(room)
        
        # Step 2: Position agent
        if agent_pos is not None:
            self._position_agent_at(room, agent_pos)
        else:
            self._position_agent_random(room)
        
        # Step 3: Take observations
        observations = self._take_full_observations(room)
        
        # Step 4: Generate question and answer based on action type
        if action_type == 'rotation':
            template = self.ROTATION_TEMPLATE
            correct_answer = (rotated_obj_name, str(rotation_degrees))
        elif action_type == 'movement':
            template = self.MOVEMENT_TEMPLATE
            correct_answer = moved_obj_name
        else:  # both
            template = self.BOTH_TEMPLATE
            correct_answer = (moved_obj_name, rotated_obj_name, str(rotation_degrees))
        
        # Generate choices
        choices, correct_idx = self.generate_choices(correct_answer, room)
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        
        self.eval_data.question = template.format(
            observations=observations,
            choices_text=choices_text
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning(room)
        return self.eval_data.question
    
    def _apply_movement(self, room: Room) -> Tuple[str, np.ndarray]:
        """Apply object movement relative to agent's diagonal position"""
        # Select target object
        target_obj = self.np_random.choice(room.objects)
        
        # Select two reference objects with orientation
        other_objects = [obj for obj in room.objects 
                        if obj.name != target_obj.name and obj.has_orientation]
        if len(other_objects) < 2:
            other_objects = [obj for obj in room.objects if obj.name != target_obj.name]
        
        ref_obj1, ref_obj2 = self.np_random.choice(other_objects, size=2, replace=False)
        
        # Calculate agent's diagonal position first
        agent_pos = self._get_diagonal_position(room, target_obj, ref_obj1, ref_obj2, neglect_trivial=True)
        
        # Move target to different quadrant relative to agent position
        rel_x = target_obj.pos[0] - agent_pos[0]
        rel_y = target_obj.pos[1] - agent_pos[1]
        
        # Choose opposite quadrant
        chosen_quadrant = (-1 if rel_x >= 0 else 1, -1 if rel_y >= 0 else 1)
        
        # Calculate new position in chosen quadrant relative to agent
        min_x, max_x, min_y, max_y = room.get_boundary()
        x_range = (agent_pos[0], max_x) if chosen_quadrant[0] > 0 else (min_x, agent_pos[0])
        y_range = (agent_pos[1], max_y) if chosen_quadrant[1] > 0 else (min_y, agent_pos[1])
        
        new_pos = np.array([self.np_random.uniform(*x_range), self.np_random.uniform(*y_range)])
        target_obj.pos = new_pos
        
        return target_obj.name, agent_pos
    
    def _apply_rotation(self, room: Room) -> Tuple[str, int]:
        """Apply object rotation"""
        oriented_objects = [obj for obj in room.objects if obj.has_orientation]
        target_obj = self.np_random.choice(oriented_objects)
        rotation_degrees = self.np_random.choice([90, 180, 270])
        
        rotations = {90: [[0,-1],[1,0]], 180: [[-1,0],[0,-1]], 270: [[0,1],[-1,0]]}
        target_obj.ori = target_obj.ori @ rotations[rotation_degrees]
        
        return target_obj.name, rotation_degrees
    
    def _position_agent_random(self, room: Room):
        """Position agent randomly for rotation-only tasks"""
        min_x, max_x, min_y, max_y = room.get_boundary()
        agent_pos = np.array([self.np_random.uniform(min_x, max_x), self.np_random.uniform(min_y, max_y)])
        self._position_agent_at(room, agent_pos)
    
    def generate_choices(self, correct_answer: Any, room: Room) -> Tuple[List[str], int]:
        """Generate 4 false belief choices"""
        choices = [str(correct_answer)]
        objects = [obj.name for obj in room.objects]
        
        while len(choices) < 4:
            if isinstance(correct_answer, tuple):
                if len(correct_answer) == 2:  # Rotation: (object, degrees)
                    wrong_obj = self.np_random.choice(objects)
                    wrong_deg = self.np_random.choice(['90', '180', '270'])
                    choice = f"({wrong_obj}, {wrong_deg})"
                else:  # Both: (moved_obj, rotated_obj, degrees)  
                    obj1 = self.np_random.choice(objects)
                    obj2 = self.np_random.choice(objects)
                    deg = self.np_random.choice(['90', '180', '270'])
                    choice = f"({obj1}, {obj2}, {deg})"
            else:  # Movement: object name
                choice = self.np_random.choice(objects)
            
            if choice == str(correct_answer):
                continue
            choices.append(choice)
        
        self.np_random.shuffle(choices)
        correct_idx = choices.index(str(correct_answer))
        return choices, correct_idx






























if __name__ == "__main__":
    from ..core import CANDIDATE_OBJECTS
    from ..utils.room_utils import generate_room
    from gymnasium.utils import seeding

    # Simple test setup
    room_config = {
        'room_range': [-10, 10],
        'n_objects': 4,
        'candidate_objects': CANDIDATE_OBJECTS,
        'generation_type': 'pov',
    }
    np_random = seeding.np_random(12)[0]
    room = generate_room(**room_config, np_random=np_random)
    
    BaseAction.set_field_of_view(180)

    # Test DirectionEvaluationTask
    print("Testing DirectionEvaluationTask:")
    direction_task = DirectionEvaluationTask(np_random=np_random)
    question = direction_task.generate_question(room)
    print(question)
    print(f"Correct answer: {direction_task.answer}")
    
    # Test evaluation
    correct, info = direction_task.evaluate(direction_task.answer)
    print(f"Self-evaluation: {correct}, Info: {info}")