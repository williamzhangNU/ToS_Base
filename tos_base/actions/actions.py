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
Movement: [<movement_action1>, <movement_action2>, ...]
Final: <final_action>

Format Notes:
- Use movement actions in Movement, and final action in Final.
- If no movement is needed, use [] in Movement.
- Actions in Movement will be executed in order.
- Exactly 2 lines in movement and final order, separated these two lines by a newline.

Examples:
{examples}

Rules:
- Term() must be alone (no movement actions)
- Field of view: {field_of_view} degrees.
"""


class MoveAction(BaseAction):
    """Move to a target object"""
    
    format_desc = "Move(object_name)"
    description = (
        "Move to the same position as the object. "
        "You can ONLY move to object within your field of view. "
        "You can ONLY move to objects by name, not directions or others. "
        "Invalid examples: Move(left), Move(forward), Move(back)"
    )
    example = "Move(table)"
    format_pattern = r"^Move\(([A-Za-z0-9_-]+)\)$"
    
    def __init__(self, target: str):
        super().__init__(target)
        self.target = target

    def _move_agent_to_pos(self, room, target_pos):
        """Move agent to target position, shift coordinate system to keep agent at origin"""
        for obj in room.objects:
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
            return ActionResult(False, self.get_feedback(False, "not_found"))
        
        target_obj = room.get_object_by_name(self.target)
        if not kwargs.get('move_anyway', False) and not self._is_visible(room.agent, target_obj):
            return ActionResult(False, self.get_feedback(False, "not_visible"))
        
        self._move_agent_to_pos(room, target_obj.pos)

        return ActionResult(True, self.get_feedback(True), {'target_name': self.target})
    
    def __repr__(self):
        return f"Move({self.target})"


class RotateAction(BaseAction):
    """Rotate by specified degrees"""
    
    format_desc = "Rotate(degrees)"
    description = "Rotate by specified degrees, only valid degrees are 0, 90, 180, 270."
    example = "Rotate(90)"
    format_pattern = r"^Rotate\(([0-9-]+)\)$"
    VALID_DEGREES = [0, 90, 180, 270]
    
    def __init__(self, degrees: int):
        super().__init__(degrees)
        self.degrees = int(degrees)

    def _rotate_agent(self, room, degrees: int):
        """Rotate agent by specified degrees, shift coordinate system to keep agent at origin"""
        rotation_matrix = self._get_rotation_matrix(degrees)
        for obj in room.objects:
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
            return ActionResult(False, self.get_feedback(False, "invalid_degree"))
        
        self._rotate_agent(room, self.degrees)
            
        return ActionResult(True, self.get_feedback(True), {'degrees': self.degrees})
    
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
        agent_anchor = kwargs['agent_anchor']
        
        ori_to_deg = {(0, 1): 0, (0, -1): 180, (1, 0): 90, (-1, 0): 270}
        target_deg = ori_to_deg[tuple(agent_anchor.ori)]
        
        MoveAction(agent_anchor.name).execute(room)
        RotateAction(target_deg).execute(room)
        
        return ActionResult(True, self.get_feedback(True), {'target_name': agent_anchor.name, 'degrees': target_deg})
    
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
        """Execute observe action on room state."""
        neglect_objects = kwargs.get('neglect_objects', [])
        with_orientation = kwargs.get('with_orientation', True)
        visible_objects = [obj for obj in room.objects if self._is_visible(room.agent, obj) and obj.name not in neglect_objects]
        
        if not visible_objects:
            answer = "Nothing to observe in the current field of view"
            return ActionResult(True, self.get_feedback(True, answer=answer), {
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
        
        return ActionResult(True, self.get_feedback(True, answer=final_answer), {
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
    description = "Terminate the exploration phase"
    example = "Term()"
    format_pattern = r"^Term\(\)$"
    
    def success_message(self, **kwargs) -> str:
        return "Exploration terminated."
    
    def error_message(self, error_type: str) -> str:
        return "Cannot terminate exploration: execution failed."
    
    def execute(self, room, **kwargs) -> ActionResult:
        """Execute term action on room state."""
        return ActionResult(True, self.get_feedback(True), {'terminated': True})
    
    @staticmethod
    def is_final() -> bool:
        return True
    
    @staticmethod
    def is_term() -> bool:
        return True
    
    def __repr__(self):
        return "Term()"


class QueryAction(BaseAction):
    """Query spatial relationship of a specific object"""
    
    format_desc = "Query(object_name)"
    description = "Query spatial relationship of a specific object relative to your current position"
    example = "Query(table)"
    format_pattern = r"^Query\(([A-Za-z0-9_-]+)\)$"
    
    def __init__(self, target=None):
        super().__init__(target)
        self.target = target
    
    def success_message(self, **kwargs) -> str:
        return f"Queried: {kwargs.get('answer', 'N/A')}"
    
    def error_message(self, error_type: str) -> str:
        errors = {"not_found": "object not found", "not_visible": "not visible"}
        return f"Cannot query '{self.target}': {errors.get(error_type, 'execution failed')}."
    
    def execute(self, room, **kwargs) -> ActionResult:
        """Execute query action on room state."""
        if not room.has_object(self.target):
            return ActionResult(False, self.get_feedback(False, "not_found"))
        
        target_obj = room.get_object_by_name(self.target)
        if not self._is_visible(room.agent, target_obj):
            return ActionResult(False, self.get_feedback(False, "not_visible"))
        
        dir_pair, dir_str = room.get_direction(self.target, room.agent.name, perspective='ego')
        answer = f"{self.target} is {dir_str}"
        
        return ActionResult(True, self.get_feedback(True, answer=answer), {
            'answer': answer, 'target_object': self.target, 
            'direction_pair': dir_pair, 'direction_string': dir_str
        })
    
    def is_final(self) -> bool:
        return True
    
    def __repr__(self):
        return f"Query({self.target})"


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
        """Parse action string into ActionSequence"""
        lines = [line.strip() for line in action_str.strip().split('\n') if line.strip()]
        
        # Must have exactly 2 lines: Movement then Final
        if len(lines) != 2 or not lines[0].startswith('Movement:') or not lines[1].startswith('Final:'):
            return None
        motion_actions = []
        
        # Parse Movement line
        bracket_content = lines[0][len('Movement:'):].strip()
        if not (bracket_content.startswith('[') and bracket_content.endswith(']')):
            return None
            
        actions_str = bracket_content[1:-1].strip()
        if actions_str:
            for item in [i.strip() for i in actions_str.split(',') if i.strip()]:
                action = cls._parse_single_action(item)
                if not action or action.is_final():
                    return None
                motion_actions.append(action)
        
        # Parse Final line
        final_action_str = lines[1][len('Final:'):].strip()
        final_action = cls._parse_single_action(final_action_str)
        
        if not final_action or not final_action.is_final():
            return None
            
        if isinstance(final_action, TermAction) and motion_actions:
            return None
            
        return cls(motion_actions, final_action)
    
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
            f"Valid Example:\nMovement: [Move(table), Rotate(90)]\nFinal: Observe()\n\n" +
            f"Valid Example (no movement):\nMovement: []\nFinal: Observe()\n\n" +
            f"Invalid Example (wrong order):\nFinal: Observe()\nMovement: []\n\n" +
            f"Invalid Example (missing separator):\nMovement: [Move(table)] Final: Observe()"
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
    agent_anchor = Object("agent_anchor", np.array([0, 0]), np.array([0, 1]))
    room = Room([table, chair, agent_anchor], "test_room", agent)

    print(f"Room: {room}")
    
    print("=== Testing Action Parsing ===")
    
    # # Test Case 1: Simple final action only
    # test1 = "Movement: []\nFinal: Observe()"
    # result1 = ActionSequence.parse(test1)
    # print(f"Test 1 - Simple final: {'✓ PASS' if result1 else '✗ FAIL'}")
    # print(f"  Input: {test1.replace(chr(10), ' | ')}")
    # print(f"  Result: {result1}")
    
    # # Test Case 2: Movement with final action
    # test2 = "Movement: [Move(table), Rotate(90)]\nFinal: Observe()"
    # result2 = ActionSequence.parse(test2)
    # print(f"\nTest 2 - Movement + Final: {'✓ PASS' if result2 else '✗ FAIL'}")
    # print(f"  Input: {test2.replace(chr(10), ' | ')}")
    # print(f"  Result: {result2}")
    
    # # Test Case 3: Term action (should be alone)
    # test3 = "Movement: []\nFinal: Term()"
    # result3 = ActionSequence.parse(test3)
    # print(f"\nTest 3 - Term action: {'✓ PASS' if result3 else '✗ FAIL'}")
    # print(f"  Input: {test3.replace(chr(10), ' | ')}")
    # print(f"  Result: {result3}")
    
    # # Test Case 4: Wrong order (should fail)
    # test4 = "Final: Observe()\nMovement: []"
    # result4 = ActionSequence.parse(test4)
    # print(f"\nTest 4 - Wrong order: {'✓ PASS' if not result4 else '✗ FAIL'}")
    # print(f"  Input: {test4.replace(chr(10), ' | ')}")
    # print(f"  Expected: None (should fail)")
    # print(f"  Result: {result4}")
    

    # # Test Case 4: Wrong order (should fail)
    # test5 = "Movement: [Observe()] \nFinal: Term()"
    # result5 = ActionSequence.parse(test5)
    # print(f"\nTest 5 - Wrong order: {'✓ PASS' if not result5 else '✗ FAIL'}")
    # print(f"  Input: {test5.replace(chr(10), ' | ')}")
    # print(f"  Expected: None (should fail)")
    # print(f"  Result: {result5}")

    print("=== Testing Action Execution ===")
    
    # Test Case 1: Simple final action only
    test1 = "Movement: []\nFinal: Observe()"
    result1 = ActionSequence.parse(test1)
    print(f"Test 1 - Simple final: {'✓ PASS' if result1 else '✗ FAIL'}")
    print(f"  Input: {test1.replace(chr(10), ' | ')}")
    print(f"  Result: {result1}")
    
    # Test Case 2: Observe with orientation
    print("\nTest 2 - Observe with orientation")
    action = ObserveAction()
    result = action.execute(room, with_orientation=True)
    print(f"  Result: {result.message}")
    # expected_msg = "table is (right, front) of you and faces away from you, chair is (left, front) of you and faces away from you"
    # assert expected_msg in result.message
    # print(f"  ✓ PASS")
    