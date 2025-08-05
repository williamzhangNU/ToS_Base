from typing import Optional, List
import re
import numpy as np

from .base import BaseAction, ActionResult

"""
Specific action implementations for spatial exploration.
Contains all concrete action classes and the ActionSequence parser.
"""


ACTION_INSTRUCTION = """\
You can move in the room and turn around to observe the room.

Available Actions:
{actions}

Answer with following format:
Actions: [<movement_action1>, <movement_action2>, ... , <final_action>]

Rules:
- First use one or more movement actions, then end with exactly one final action.
- Actions will be executed in order.
- You have a field of view for observation: {field_of_view} degrees.

Examples:
{examples}

"""


class MoveAction(BaseAction):
    """Move to a target object"""
    
    format_desc = "Move(object_name)"
    description = (
        "Move to the same position as the object. "
        "Your orientation does NOT change when you move."
        "You can ONLY move to object within your field of view. "
        "You can ONLY move to objects by name, not directions or others. "
        "Invalid examples: Move(left), Move(forward), Move(back). "
    )
    example = "Move(table)"
    format_pattern = r"^Move\(([A-Za-z0-9_-]+)\)$"
    
    def __init__(self, target: str):
        super().__init__(target)
        self.target = target

    def _move_agent_to_pos(self, room, target_pos):
        """Move agent to target position, shift coordinate system to keep agent at origin"""
        for obj in [o for o in room.all_objects if o.name != room.agent.name]:
            obj.pos = obj.pos - target_pos
    
    def success_message(self, **kwargs) -> str:
        return f"You moved at {self.target}."
        # return f"You moved to the same position as {self.target}."
    
    def error_message(self, error_type: str) -> str:
        errors = {"not_found": "object not found", "not_visible": "object not visible"}
        return f"Cannot move to '{self.target}': {errors.get(error_type, 'execution failed')}."
    
    def execute(self, room, **kwargs) -> ActionResult:
        """Execute move action on room state."""
        if not room.has_object(self.target):
            return ActionResult(False, self.get_feedback(False, "not_found"), str(self), 'move', {'target_name': self.target})
        
        target_obj = room.get_object_by_name(self.target)
        if not kwargs.get('move_anyway', False) and not self._is_visible(room.agent, target_obj):
            return ActionResult(False, self.get_feedback(False, "not_visible"), str(self), 'move', {'target_name': self.target})        
        self._move_agent_to_pos(room, target_obj.pos)

        return ActionResult(True, self.get_feedback(True), str(self), 'move', {'target_name': self.target})
    
    def __repr__(self):
        return f"Move({self.target})"


class RotateAction(BaseAction):
    """Rotate by specified degrees"""
    
    format_desc = "Rotate(degrees)"
    description = "Rotate clockwise by specified degrees relative to your current orientation, only valid degrees are 0, 90, 180, 270."
    example = "Rotate(90)"
    format_pattern = r"^Rotate\(([0-9-]+)\)$"
    VALID_DEGREES = [0, 90, 180, 270]
    
    def __init__(self, degrees: int):
        super().__init__(degrees)
        self.degrees = int(degrees)

    def _rotate_agent(self, room, degrees: int):
        """Rotate agent by specified degrees, shift coordinate system to keep agent at origin"""
        rotation_matrix = self._get_rotation_matrix(degrees)
        for obj in [o for o in room.all_objects if o.name != room.agent.name]:
            obj.pos = obj.pos @ rotation_matrix
            obj.ori = obj.ori @ rotation_matrix
    
    def success_message(self, **kwargs) -> str:
        return f"You rotated clockwise {self.degrees}°."
    
    def error_message(self, error_type: str) -> str:
        if error_type == "invalid_degree":
            return f"Cannot rotate by {self.degrees}°: only {self.VALID_DEGREES} allowed."
        return f"Cannot rotate by {self.degrees}°: execution failed."
    
    def execute(self, room, **kwargs) -> ActionResult:
        """Execute rotate action on room state."""
        if self.degrees is None or self.degrees not in self.VALID_DEGREES:
            return ActionResult(False, self.get_feedback(False, "invalid_degree"), str(self), 'rotate', {'degrees': self.degrees})
        self._rotate_agent(room, self.degrees)   
        return ActionResult(True, self.get_feedback(True), str(self), 'rotate', {'degrees': self.degrees})
    
    def __repr__(self):
        return f"Rotate({self.degrees})"


class ReturnAction(BaseAction):
    """Return to anchor position"""
    
    format_desc = "Return()"
    description = "Return to the starting anchor position"
    example = "Return()"
    format_pattern = r"^Return\(\)$"
    
    def success_message(self, **kwargs) -> str:
        return "You returned to anchor."
    
    def error_message(self, error_type: str) -> str:
        return "Cannot return to anchor: execution failed."
    
    def execute(self, room, **kwargs) -> ActionResult:
        """Execute return action on room state."""
        ori_to_deg = {(0, 1): 0, (0, -1): 180, (1, 0): 90, (-1, 0): 270}
        target_deg = ori_to_deg[tuple(room.initial_pos.ori)]
        
        MoveAction(room.initial_pos.name).execute(room)
        RotateAction(target_deg).execute(room)
        
        return ActionResult(True, self.get_feedback(True), str(self), 'return', {'target_name': room.initial_pos.name, 'degrees': target_deg})
    
    def __repr__(self):
        return "Return()"


class ObserveAction(BaseAction):
    """Observe spatial relationships of all objects in view"""
    
    format_desc = "Observe()"
    description = (
        "Observe spatial relationships of all objects in the field of view relative to your current position. "
        "You can only observe objects that are within your field of view."
    )
    example = "Observe()"
    format_pattern = r"^Observe\(\)$"

    directional_template = "{obj_name} is {dir_str} of you"
    orientation_template = "{obj_name} faces {orientation}"
    
    def __init__(self):
        super().__init__()
    
    def success_message(self, **kwargs) -> str:
        return f"You observe: {kwargs.get('answer', 'nothing')}."
    
    def error_message(self, error_type: str) -> str:
        return "Cannot observe: execution failed."
    
    def execute(self, room, **kwargs) -> ActionResult:
        """Execute observe action on room state. NOTE Also neglect same position objects."""
        neglect_objects = kwargs.get('neglect_objects', []) + [obj.name for obj in room.objects if np.allclose(obj.pos, room.agent.pos)]
        with_orientation = kwargs.get('with_orientation', True)
        visible_objects = [obj for obj in room.objects if self._is_visible(room.agent, obj) and obj.name not in neglect_objects]
        
        if not visible_objects:
            answer = "Nothing in your field of view."
            return ActionResult(True, self.get_feedback(True, answer=answer), str(self), 'observe', {
                'answer': answer, 'visible_objects': [], 'relationships': []
            })

        relationships = []
        for obj in visible_objects:
            _, dir_str = room.get_direction(obj.name, room.agent.name, perspective='ego')
            # answer_str = f"{obj.name} is {dir_str} of you"
            answer_str = f"{obj.name} is {dir_str}"
            if with_orientation and obj.has_orientation:
                _, orientation = room.get_orientation(obj.name, room.agent.name)
                answer_str += f" and faces {orientation}"
            relationships.append(answer_str)
        final_answer = ", ".join(relationships)
        
        return ActionResult(True, self.get_feedback(True, answer=final_answer), str(self), 'observe', {
            'answer': final_answer,
            'visible_objects': [obj.name for obj in visible_objects],
            'relationships': relationships
        })
    
    @staticmethod
    def is_final() -> bool:
        return True
    
    def __repr__(self):
        return "Observe()"


class TermAction(BaseAction):
    """Terminate exploration"""
    
    format_desc = "Term()"
    description = "Terminate the exploration phase. Term() must be alone with no movement actions except for Return()."
    example = "Term()"
    format_pattern = r"^Term\(\)$"
    
    def success_message(self, **kwargs) -> str:
        return "Exploration terminated."
    
    def error_message(self, error_type: str) -> str:
        return "Cannot terminate exploration: execution failed."
    
    def execute(self, room, **kwargs) -> ActionResult:
        """Execute term action on room state."""
        return ActionResult(True, self.get_feedback(True), str(self), 'term', {'terminated': True})
    
    @staticmethod
    def is_final() -> bool:
        return True
    
    @staticmethod
    def is_term() -> bool:
        return True
    
    def __repr__(self):
        return "Term()"

# Action registry for easy lookup
ACTION_CLASSES = [MoveAction, RotateAction, ReturnAction, ObserveAction, TermAction]


class ActionSequence:
    """Sequence of actions for spatial exploration"""
    
    def __init__(self, motion_actions: List[BaseAction] = None, final_action: BaseAction = None):
        self.motion_actions = motion_actions or []
        self.final_action = final_action
    
    def __repr__(self):
        motions = ", ".join(str(action) for action in self.motion_actions)
        return f"ActionSequence(motions=[{motions}], final={self.final_action})"

    @classmethod
    def parse(cls, action_str: str) -> Optional['ActionSequence']:
        m = re.search(r'\[(.*)\]', action_str.strip())
        if not m:
            return None
        # extract top-level actions like Move(table), Rotate(90), Term()
        action_strs = re.findall(r'([A-Za-z]+\(.*?\))', m.group(1))
        if not action_strs:
            return None

        motions = []
        final_action = None
        for i, act_s in enumerate(action_strs):
            act = cls._parse_single_action(act_s.strip())
            if not act:
                return None
            if i == len(action_strs) - 1:
                if not act.is_final():
                    return None
                final_action = act
            else:
                if act.is_final():
                    return None
                motions.append(act)
        if isinstance(final_action, TermAction):
            if any(not isinstance(a, ReturnAction) for a in motions):
                return None
        return cls(motions, final_action)
    
    @staticmethod
    def _parse_single_action(action_str: str) -> Optional[BaseAction]:
        """Parse a single action string using registered action classes"""
        for action_class in ACTION_CLASSES:
            if action := action_class.parse(action_str):
                return action
        return None
    
    @staticmethod
    def get_usage_instructions() -> str:
        """Get usage instructions for action sequences"""
        motion_actions = [cls for cls in ACTION_CLASSES if not cls.is_final()]
        final_actions = [cls for cls in ACTION_CLASSES if cls.is_final()]
        
        action_desc = (
            "Movement Actions:\n" +
            "\n".join(f"- {cls.format_desc}: {cls.description}" for cls in motion_actions) +
            "\n\n" +
            "Final Actions:\n" +
            "\n".join(f"- {cls.format_desc}: {cls.description}" for cls in final_actions)
        )
        examples = (
            f"Valid Example:\nActions: [Move(table), Rotate(90), Observe()]\n\n" +
            f"Valid Example:\nActions: [Observe()]\n\n" +
            f"Invalid Example (no final action):\nActions: [Move(table)]\n\n"+
            f"Invalid Example (more than one final action):\nActions: [Observe(), Rotate(90), Observe()]\n\n"
            
        )
        
        return ACTION_INSTRUCTION.format(
            actions=action_desc,
            examples=examples,
            field_of_view=BaseAction.get_field_of_view()
        )


if __name__ == "__main__":
    # Test action parsing and execution
    from ..core import Room, Object, Agent
    import numpy as np
    
    # Create test room
    agent = Agent()
    table = Object("table", np.array([1, 2]), np.array([0, 1]))
    chair = Object("chair", np.array([-2, 1]), np.array([0, 1]))
    room = Room(agent, [table, chair], name="test_room")

    print(f"Room: {room}")
    
    print("=== Testing Action Parsing ===")
    
    # Test individual action parsing
    test_actions = [
        "Move(table)",
        "Rotate(90)",
        "Rotate(-45)",
        "Observe()",
        "Query(table, chair)",
        "Return()",
        "Term()",
        "InvalidAction()",
        "Move()",
        "Rotate(abc)",
    ]
    
    print("Individual Action Parsing Tests:")
    for action_str in test_actions:
        action = ActionSequence._parse_single_action(action_str)
        print(f"  '{action_str}' -> {action}")
    
    print("\n=== Testing Action Sequence Parsing ===")
    
    # Test action sequence parsing
    test_sequences = [
        "Actions: [Move(table), Rotate(90), Observe()]",
        "Actions: [Observe()]",
        "Actions: [Move(table), Query(table, chair)]",
        "Actions: [Return(), Term()]",
        "Actions: [Term()]",
        "Actions: [Move(table)]",  # Invalid: no final action
        "Actions: [Observe(), Query(table, chair)]",  # Invalid: multiple final actions
        "Actions: [Move(invalid_object), Observe(), Observe()]",
        "Actions: [Rotate(90), Rotate(-45), Return()]",
    ]
    
    print("Action Sequence Parsing Tests:")
    for seq_str in test_sequences:
        sequence = ActionSequence.parse(seq_str)
        if sequence:
            print(f"  '{seq_str}' -> SUCCESS")
            print(f"    Motion: {sequence.motion_actions}")
            print(f"    Final: {sequence.final_action}")
        else:
            print(f"  '{seq_str}' -> ERROR")
    
    