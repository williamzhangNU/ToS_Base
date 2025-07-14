"""
The script defines the different evaluation metrics for the SpatialGym.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, List
import numpy as np
from typing_extensions import override
import copy
from dataclasses import dataclass

from ..core.room import Room
from ..utils.eval_utilities import (
    dir_eval_fn,
    obj_seq_eval_fn,
    deg_seq_eval_fn,
    list_dir_eval_fn,
    e2a_eval_fn,
    obj_presence_eval_fn,
    multi_choice_eval_fn,
)
from ..core.constant import CANDIDATE_OBJECTS
from ..core.graph import DirectionalGraph
from ..core.object import Object, Agent
from ..core.relationship import DirPair, Dir, DirectionSystem
from ..actions import MoveAction, RotateAction

@dataclass
class EvaluationData:
    question: str
    answer: str
    reasoning: str
    task_type: str

    def __post_init__(self):
        valid_task_types = [
            'AllPairsEvaluationTask',
            'DirEvaluationTask',
            'RotEvaluationTask',
            'PovEvaluationTask',
            'ReverseDirEvaluationTask',
            'E2AEvaluationTask',
            'ObjectPresenceEvaluationTask',
        ]
        assert self.task_type in valid_task_types, f"Invalid task type: {self.task_type}"
    
    def evaluate(self, pred: Any) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate an answer to the given question"""
        if self.task_type == 'AllPairsEvaluationTask':
            correct_count = list_dir_eval_fn(pred, self.answer)
            total_answers = len(self.answer)
            score = correct_count / total_answers if total_answers > 0 else 0.0
            
            info = {
                "score": score,
                "correct_count": correct_count,
                "total_count": total_answers
            }
            return correct_count == total_answers, info
        
        elif self.task_type in ['DirEvaluationTask', 'PovEvaluationTask']:
            return dir_eval_fn(pred, self.answer), {}
        
        elif self.task_type == 'RotEvaluationTask':
            return obj_seq_eval_fn(pred, self.answer), {}
        
        elif self.task_type == 'ReverseDirEvaluationTask':
            return multi_choice_eval_fn(pred, self.answer), {}
        
        elif self.task_type == 'E2AEvaluationTask':
            return e2a_eval_fn(pred, self.answer)
        
        elif self.task_type == 'ObjectPresenceEvaluationTask':
            return obj_presence_eval_fn(pred, self.answer)
        
        return False, {"error": "Unknown task type"}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the evaluation data to a dictionary"""
        return {
            'question': self.question,
            'answer': self.answer,
            'reasoning': self.reasoning,
            'task_type': self.task_type,
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
            task_type=self.__class__.__name__
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
    
    def _generate_reasoning(self, room: Room) -> str:
        """Generate reasoning for the evaluation task"""
        return f"Testing spatial reasoning for {self.__class__.__name__}"
    
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
        task_types = {
            'AllPairsEvaluationTask': AllPairsEvaluationTask,
            'DirEvaluationTask': DirEvaluationTask,
            'RotEvaluationTask': RotEvaluationTask,
            'PovEvaluationTask': PovEvaluationTask,
            'ReverseDirEvaluationTask': ReverseDirEvaluationTask,
            'E2AEvaluationTask': E2AEvaluationTask,
            'ObjectPresenceEvaluationTask': ObjectPresenceEvaluationTask,
        }
        
        task_type = data.get('type', cls.__name__)
        return task_types.get(task_type, cls).from_dict(data)


class ObjectPresenceEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for identifying what objects are present in the room.
    
    Q: What objects are in the room?
    A: List of object names (excluding agent)
    """

    QUESTION_TEMPLATE = (
        "What objects are present in the room? "
        "List all the objects you can see, separated by commas."
    )
    
    def generate_question(self, room: Room) -> str:
        """Generate a question asking about object presence in the room"""
        
        # Get all object names excluding the agent
        object_names = [obj.name for obj in room.objects]
        
        self.eval_data.question = self.QUESTION_TEMPLATE
        self.eval_data.answer = object_names
        self.eval_data.reasoning = f"The room contains {len(object_names)} objects: {', '.join(object_names)}"
        return self.eval_data.question


class AllPairsEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for checking all spatial relationships allocentrically between object pairs.
    
    Q: spatial relationship between all pairs of objects
    A: [<dir>, <dir>, ...]
    
    For N objects, generates N*(N-1)/2 distinct relationships with randomly shuffled object orders.
    Answer format: 1. (<horiz>, <vert>), 2. (<horiz>, <vert>), ...
    """

    QUESTION_TEMPLATE = (
        "From a top-down view, determine the spatial relationships between the following object pairs:\n"
        "{obj_pairs_str}\n"
        "For each pair (A, B), provide the relationship in the format: (<horizontal>, <vertical>), meaning A is <horizontal> to B and <vertical> to B.\n"
        "\n"
        "Answer format:\n"
        "1. (<horizontal>, <vertical>)\n"
        "2. (<horizontal>, <vertical>)\n"
        "..."
    )
    
    def generate_question(self, room: Room) -> str:
        """Generate a question that asks about all spatial relationships between pairs of objects"""
        room = room.copy()
        answer = []
        n = len(room.all_objects)
        
        # Generate all pairs with random order
        pairs = [(i, j) if self.np_random.random() >= 0.5 else (j, i) 
                for i in range(n) for j in range(i+1, n)]
        self.np_random.shuffle(pairs)
        
        rel_questions = []
        for i, j in pairs:
            obj1, obj2 = room.all_objects[i], room.all_objects[j]
            _, dir_pair_str = room.get_direction(obj1.name, obj2.name, perspective='allo')
            answer.append(dir_pair_str)
            rel_questions.append(f"({obj1.name}, {obj2.name})")
        
        rel_questions_str = "\n".join([f"{i}. {question}" for i, question in enumerate(rel_questions, 1)])
        
        self.eval_data.question = self.QUESTION_TEMPLATE.format(obj_pairs_str=rel_questions_str)
        self.eval_data.answer = answer
        self.eval_data.reasoning = self._generate_reasoning(room)
        return self.eval_data.question


class DirEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for direction questions.
    
    Q: Ask spatial relationship between two objects (a, b)
    A: <dir>
    
    Movement types:
    1. static: Add new object <dir> to anchor_obj
    2. object_move: Move target_obj <dir> to anchor_obj  
    3. agent_move: Move agent <dir> to anchor_obj
    4. agent_turn: Rotate agent <degree>
    """
    
    def generate_question(self, room: Room) -> str:
        room = room.copy()
        movement = self.config.get('movement', 'static')
        
        # Validate movement requirements
        if movement in ['agent_move', 'agent_turn'] and room.agent is None:
            raise ValueError(f"Agent must be in the room for {movement}")

        graph = DirectionalGraph(room.all_objects, is_explore=False)
        graph.is_explore = True

        # Handle different movement types
        if movement == 'static':
            target_name, target_obj_idx, obs = self._handle_static_movement(room)
        elif movement == 'object_move':
            target_name, target_obj_idx, obs = self._handle_object_movement(room)
        elif movement == 'agent_move':
            target_name, target_obj_idx, obs = self._handle_agent_movement(room)
        elif movement == 'agent_turn':
            return self._handle_agent_turn(room, graph)
        else:
            raise ValueError(f"Invalid movement type: {movement}")

        # Update graph and generate question
        anchor_obj_idx, anchor_name, new_pos, dir_pair, dir_pair_str = self._get_movement_details(room)
        obs += f"{target_name} moves {dir_pair_str} to {anchor_name}."

        if movement == 'static':
            graph.add_node(anchor_obj_idx, dir_pair)
        else:
            graph.move_node(target_obj_idx, anchor_obj_idx, dir_pair)

        query_obj_idx, query_obj = self._get_query_object(room, target_name, anchor_name)
        question = f"{obs} {target_name} is what direction to {query_obj.name}?"
        
        dir_pair_query = graph.get_direction(target_obj_idx, query_obj_idx)
        answer = DirectionSystem.to_string(dir_pair_query, perspective='ego' if room.agent else 'allo')
        
        self.eval_data.question = question
        self.eval_data.answer = answer
        self.eval_data.reasoning = self._generate_reasoning(room)
        return question

    def _handle_static_movement(self, room: Room) -> Tuple[str, int, str]:
        """Handle static movement (new object placement)"""
        target_name = room.objects[0].name
        while room.has_object(target_name):
            target_name = self.np_random.choice(CANDIDATE_OBJECTS)
        target_obj_idx = len(room.all_objects)
        obs = f"A new object {target_name} is placed in the room.\n"
        return target_name, target_obj_idx, obs

    def _handle_object_movement(self, room: Room) -> Tuple[str, int, str]:
        """Handle object movement"""
        non_agent_indices = [i for i, obj in enumerate(room.all_objects) 
                           if room.agent is None or obj.name != room.agent.name]
        target_obj_idx = self.np_random.choice(non_agent_indices)
        target_name = room.all_objects[target_obj_idx].name
        obs = f"{target_name} is moved to a new position.\n"
        return target_name, target_obj_idx, obs

    def _handle_agent_movement(self, room: Room) -> Tuple[str, int, str]:
        """Handle agent movement"""
        target_name = room.agent.name
        target_obj_idx = next(i for i, obj in enumerate(room.all_objects) if obj == room.agent)
        obs = f"{target_name} moves to a new position.\n"
        return target_name, target_obj_idx, obs

    def _handle_agent_turn(self, room: Room, graph: DirectionalGraph) -> str:
        """Handle agent turn movement"""
        target_obj_idx = self.np_random.integers(0, len(room.all_objects))
        target_name = room.all_objects[target_obj_idx].name
        
        degree = self.np_random.choice([90, 180, 270])
        graph.rotate_axis(degree)
        obs = f"You turn {degree} degrees."

        query_obj_idx, query_obj = self._get_query_object(room, target_name, "")
        question = f"{obs} {target_name} is what direction to {query_obj.name}?"
        
        dir_pair_query = graph.get_direction(target_obj_idx, query_obj_idx)
        answer = DirectionSystem.to_string(dir_pair_query, perspective='ego' if room.agent else 'allo')
        
        self.eval_data.question = question
        self.eval_data.answer = answer
        self.eval_data.reasoning = self._generate_reasoning(room)
        return question

    def _get_movement_details(self, room: Room) -> Tuple[int, str, np.ndarray, Any, str]:
        """Get movement details for non-turn movements"""
        anchor_obj_idx = self.np_random.integers(0, len(room.all_objects))
        anchor_obj = room.all_objects[anchor_obj_idx]
        anchor_name = anchor_obj.name
        
        min_x, max_x, min_y, max_y = room.get_boundary()
        new_pos = np.array([self.np_random.uniform(min_x, max_x), self.np_random.uniform(min_y, max_y)])
        
        dir_pair = DirectionSystem.get_direction(new_pos, anchor_obj.pos, anchor_obj.ori)
        dir_pair_str = DirectionSystem.to_string(dir_pair, perspective='ego' if room.agent else 'allo')
        
        return anchor_obj_idx, anchor_name, new_pos, dir_pair, dir_pair_str

    def _get_query_object(self, room: Room, target_name: str, anchor_name: str) -> Tuple[int, Object]:
        """Get a query object that's different from target and anchor"""
        query_obj_idx = self.np_random.integers(0, len(room.all_objects))
        while room.all_objects[query_obj_idx].name in [target_name, anchor_name]:
            query_obj_idx = self.np_random.integers(0, len(room.all_objects))
        return query_obj_idx, room.all_objects[query_obj_idx]


class ReverseDirEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for reverse direction questions.
    
    Q: which object is also <dir> to <new_obj>?
    A: <obj2>
    
    Adds a new object and guarantees inferable direction.
    Given new_obj --> anchor_obj, asks which (target) object is also new_obj --> target_obj.
    """

    QUESTION_TEMPLATE = (
        "A new object {new_obj_name} is {dir_pair_str} to {anchor_obj_name}.\n"
        "Then {new_obj_name} is also {dir_pair_str} to which object?\n"
        "Format your answer as a single object name, e.g., 'chair'"
    )
    
    def generate_question(self, room: Room) -> str:
        room = room.copy()
        
        # Get new object name
        new_obj_name = room.objects[0].name
        while room.has_object(new_obj_name):
            new_obj_name = self.np_random.choice(CANDIDATE_OBJECTS)

        # Choose anchor and target objects
        anchor_obj_idx = self.np_random.integers(0, len(room.all_objects))
        target_obj_idx = self.np_random.integers(0, len(room.all_objects))
        while target_obj_idx == anchor_obj_idx:
            anchor_obj_idx = self.np_random.integers(0, len(room.all_objects))
        
        target_obj = room.all_objects[target_obj_idx]
        anchor_obj = room.all_objects[anchor_obj_idx]

        # Get direction and find inferable objects
        dir_pair, dir_pair_str = room.get_direction(anchor_obj.name, target_obj.name)

        graph = DirectionalGraph(room.all_objects, is_explore=False)

        # Find all objects that anchor is also dir_pair to
        answer = []
        for i in range(len(room.all_objects)):
            if i != anchor_obj_idx:  # Skip the anchor object itself
                anchor_dir_pair = graph.get_direction(anchor_obj_idx, i)
                if (anchor_dir_pair.horiz == dir_pair.horiz and 
                    anchor_dir_pair.vert == dir_pair.vert):
                    answer.append(room.all_objects[i].name)

        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            new_obj_name=new_obj_name, 
            dir_pair_str=dir_pair_str, 
            anchor_obj_name=anchor_obj.name
        )
        self.eval_data.answer = answer
        self.eval_data.reasoning = self._generate_reasoning(room)
        return self.eval_data.question


class RotEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for rotation questions.
    
    Q: What is the sequence of objects when agent turns around at its original position?
    A: [<obj1>, <obj2>, ...]
    
    Agent turns clockwise/counterclockwise and lists objects in order of encounter.

    TODO:
    1. for movement, need to gaurentee no ambiguity when generate room / evaluate answer
    """

    QUESTION_TEMPLATE = (
        "You will turn {turn_direction} 360 degrees. "
        "During your rotation, you will encounter objects in a specific sequence as they come into your direct line of sight. "
        "List them in the order you encounter them during the rotation: <obj1>, <obj2>, <obj3>, ..."
        "Do not list objects that are at the same position as you."
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

        objects = room.objects
        objects.sort(key=lambda x: get_angle(x.pos), reverse=(turn_direction == 'counterclockwise'))

        self.eval_data.question = movement_prompt + turn_prompt + self.QUESTION_TEMPLATE.format(turn_direction=turn_direction)
        self.eval_data.answer = [obj.name for obj in objects if obj.name not in neglect_objects]
        self.eval_data.reasoning = self._generate_reasoning(room)
        return self.eval_data.question
    
    @override
    def to_string(self) -> str:
        return f"{self.__class__.__name__}({self.config.get('turn_direction', 'clockwise')})"


class PovEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for perspective taking questions.
    
    Q: Ask spatial relationship between object a from the perspective of object c
    A: <dir>
    
    Tests ability to take another object's viewpoint for spatial reasoning.
    """

    QUESTION_TEMPLATE = (
        "Imagine you are {anchor_obj_name}. From your perspective, what direction is {obj_name} relative to you?\n"
        "Answer in the format: (horizontal, vertical)\n"
        "Example: (left, front) or (right, back)"
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

        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            anchor_obj_name=anchor_obj_name,
            obj_name=obj_name,
        )
        _, dir_pair_str = room.get_direction(obj_name, anchor_obj_name, anchor_name=anchor_obj_name)
        self.eval_data.answer = dir_pair_str
        self.eval_data.reasoning = self._generate_reasoning(room)
        return self.eval_data.question


class E2AEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for ego2allo questions.
    
    Q: After turn <ori>, what are the coordinates of the objects?
    A: [(<obj1_x>, <obj1_y>), (<obj2_x>, <obj2_y>), ...]
    
    Tests conversion from egocentric view to allocentric coordinates.
    Agent explores from ego-centric view and evaluates allocentric mapping.
    """

    QUESTION_TEMPLATE = (
        "You originally face north, now you are facing {agent_ori} and at {agent_pos}. Here are the objects: {object_sequence_str}\n"
        "What are the position and coordinates of each object?\n"
        "Use yourself as origin (0,0), with positive y-axis in your facing direction.\n"
        "Format: [(x1, y1), (x2, y2), ...]"
    )
    def generate_question(self, room: Room) -> str:
        room = room.copy()
        
        # Step 1: Agent moves to a randomly chosen object
        chosen_obj = self.np_random.choice(room.objects)
        move_action = MoveAction(chosen_obj.name)
        move_action.execute(room)
        
        # Step 2: Agent rotates by a random degree
        degree = self.np_random.choice([0, 90, 180, 270])
        if degree != 0:
            rotate_action = RotateAction(degree)
            rotate_action.execute(room)
        
        # Step 3: Generate question and answer
        orientations = ["north", "east", "south", "west"]
        agent_ori_str = orientations[degree // 90]

        objects = [obj for obj in copy.deepcopy(room.all_objects) 
                  if room.agent is None or obj.name != room.agent.name]
        self.np_random.shuffle(objects)
        object_sequence_str = ", ".join([obj.name for obj in objects])

        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            object_sequence_str=object_sequence_str,
            agent_ori=agent_ori_str,
            agent_pos=chosen_obj.name
        )
        self.eval_data.answer = [tuple(obj.pos) for obj in objects]
        self.eval_data.reasoning = self._generate_reasoning(room)
        return self.eval_data.question


if __name__ == "__main__":
    from ..core import CANDIDATE_OBJECTS
    from ..utils.room_utils import generate_room
    from gymnasium.utils import seeding

    room_config = {
        'room_range': [-10, 10],
        'n_objects': 4,
        'candidate_objects': CANDIDATE_OBJECTS,
        'generation_type': 'pov',
    }
    np_random = seeding.np_random(21)[0]
    room = generate_room(**room_config, np_random=np_random)
    print(room)

    # Test all pairs evaluation task
    print("\n" + "="*50)
    print("Testing AllPairsEvaluationTask:")
    print("="*50)
    all_pairs_task = AllPairsEvaluationTask(np_random=np_random)
    all_pairs_question = all_pairs_task.generate_question(room)
    print(all_pairs_question)
    print(f"Expected answer: {all_pairs_task.answer}")
    correct, info = all_pairs_task.evaluate("1. (south, west)\n2. (north, west)\n3. (south, west)\n4. (south, east)\n5. (north, east)\n6. (north, east)")
    print(correct)
    print(info)
    
    # # Test evaluation with a sample answer
    # print("\n" + "="*50)
    # print("Testing ReverseDirEvaluationTask:")
    # print("="*50)
    # sample_answer = "1. (right, front)\n2. (left, back)\n3. (same, front)"
    # correct, info = all_pairs_task.evaluate(sample_answer)
    # print(f"Sample answer evaluation: {correct}, Info: {info}")

    
    # # Test direction evaluation task
    # print("\n" + "="*50)
    # print("Testing ReverseDirEvaluationTask:")
    # print("="*50)
    # task = DirEvaluationTask(np_random=np_random, config={'movement': 'agent_turn'})
    # question = task.generate_question(room)
    # print(question)
    # print(task.answer)
    # correct, info = task.evaluate("(unknown, front)")
    # print(correct)

    # # Test reverse direction evaluation task
    # print("\n" + "="*50)
    # print("Testing ReverseDirEvaluationTask:")
    # print("="*50)
    # reverse_dir_task = ReverseDirEvaluationTask(np_random=np_random)
    # reverse_dir_question = reverse_dir_task.generate_question(room)
    # print(reverse_dir_question)
    # print(f"Expected answer: {reverse_dir_task.answer}")

    # # Test rotation evaluation task
    # print("\n" + "="*50)
    # print("Testing RotEvaluationTask:")
    # print("="*50)
    # rotation_config = {
    #     'turn_direction': 'counterclockwise',
    #     'if_move': True,
    #     'if_turn': True
    # }
    # rot_task = RotEvaluationTask(np_random=np_random, config=rotation_config)
    # rot_question = rot_task.generate_question(room)
    # print(rot_question)
    # print(f"Expected answer: {rot_task.answer}")

    # Test pov evaluation task
    print("\n" + "="*50)
    print("Testing PovEvaluationTask:")
    print("="*50)
    pov_task = PovEvaluationTask(np_random=np_random)
    pov_question = pov_task.generate_question(room)
    print(pov_question)
    print(f"Expected answer: {pov_task.answer}")

    # # Test object presence evaluation task
    # print("\n" + "="*50)
    # print("Testing ObjectPresenceEvaluationTask:")
    # print("="*50)
    # obj_presence_task = ObjectPresenceEvaluationTask(np_random=np_random)
    # obj_presence_question = obj_presence_task.generate_question(room)
    # print(obj_presence_question)
    # print(f"Expected answer: {obj_presence_task.answer}")
    
    # # Test with a correct answer
    # pred_answer = ", ".join(obj_presence_task.answer)
    # correct, info = obj_presence_task.evaluate(pred_answer)
    # print(f"Test with correct answer: {correct}, Info: {info}")
    
    # # Test with a partial answer
    # partial_answer = ", ".join(obj_presence_task.answer[:2])  # Only first 2 objects
    # correct, info = obj_presence_task.evaluate(partial_answer)
    # print(f"Test with partial answer: {correct}, Info: {info}")

    # # Test e2a evaluation task
    # print("\n" + "="*50)
    # print("Testing E2AEvaluationTask:")
    # print("="*50)
    # e2a_task = E2AEvaluationTask(np_random=np_random)
    # e2a_question = e2a_task.generate_question(room)
    # print(e2a_question)
    # print(f"Expected answer: {e2a_task.answer}")
    # pred_answer = "[(1, 8), (0, 0), (6, -4), (3, -5)]"
    # correct, info = e2a_task.evaluate(pred_answer)
    # print(correct)
    # print(info)