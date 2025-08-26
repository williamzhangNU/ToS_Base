from typing import Optional, List
import re
import numpy as np

from .base import BaseAction, ActionResult
from ..core.object import Gate
from ..core.relationship import PairwiseRelationship, PairwiseRelationshipDiscrete, ProximityRelationship, RelationTriple, OrientationRel, DegreeRel

"""
Specific action implementations for spatial exploration.
Contains all concrete action classes and the ActionSequence parser.
"""


ACTION_INSTRUCTION = """\
You can move within and across rooms, turn, observe, and traverse doors.

Available Actions:
{actions}

Answer format:
Actions: [<movement_1>, <movement_2>, ... (movement) , <movement_n>, <final_action>] or [<query_1>, <query_2>, ... (query) , <query_n>]

Rules:
- You may perform zero, one or more movement actions.
- Either:
  - Provide **exactly one** final action (and it **must be last**), OR
  - Provide one or more **query actions only** (no movement/final actions).
- Observe action only reports from your current position. If you move multiple times, the final Observe action gives the view only from your last position.
- Actions execute in order. Field of view: {field_of_view}°.

Costs:
{costs}

You need to explore the environment using **minimal cost**.

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
    format_pattern = r"^Move\(([A-Za-z0-9_ -]+)\)$"
    cost = 0
    
    def __init__(self, target: str):
        super().__init__(target)
        self.target = target
    
    def success_message(self, **kwargs) -> str:
        return f"You moved at {self.target}."
    
    def error_message(self, error_type: str) -> str:
        errors = {"not_found": "object not found", "not_visible": "object not visible"}
        return f"Cannot move to '{self.target}': {errors.get(error_type, 'execution failed')}."
    
    def execute(self, room, agent, **kwargs) -> ActionResult:
        """Execute move action on room state."""
        if not room.has_object(self.target):
            return ActionResult(False, self.get_feedback(False, "not_found"), str(self), 'move', {'target_name': self.target})
        
        target_obj = room.get_object_by_name(self.target)
        if not kwargs.get('move_anyway', False) and not self._is_visible(agent, target_obj):
            return ActionResult(False, self.get_feedback(False, "not_visible"), str(self), 'move', {'target_name': self.target})        
        agent.pos  = target_obj.pos # only change pos, not ori or room_id

        return ActionResult(True, self.get_feedback(True), str(self), 'move', {'target_name': self.target})
    
    def __repr__(self):
        return f"Move({self.target})"

class RotateAction(BaseAction):
    """Rotate by specified degrees"""
    
    format_desc = "Rotate(degrees)"
    description = "Rotate by specified degrees relative to your current orientation. Positive = clockwise, negative = counterclockwise. Valid: -270, -180, -90, 0, 90, 180, 270."
    example = "Rotate(-90)"
    format_pattern = r"^Rotate\(([0-9-]+)\)$"
    VALID_DEGREES = [0, 90, 180, 270, -90, -180, -270]
    
    def __init__(self, degrees: int):
        super().__init__(degrees)
        self.degrees = int(degrees)
        
    def success_message(self, **kwargs) -> str:
        if self.degrees == 0:
            return "You rotated 0°."
        direction = 'clockwise' if self.degrees > 0 else 'counterclockwise'
        return f"You rotated {direction} {abs(self.degrees)}°."
    
    def error_message(self, error_type: str) -> str:
        if error_type == "invalid_degree":
            return f"Cannot rotate by {self.degrees}°: only {self.VALID_DEGREES} allowed."
        return f"Cannot rotate by {self.degrees}°: execution failed."
    
    def execute(self, room, agent, **kwargs) -> ActionResult:
        """Execute rotate action on room state."""
        if self.degrees is None or self.degrees not in self.VALID_DEGREES:
            return ActionResult(False, self.get_feedback(False, "invalid_degree"), str(self), 'rotate', {'degrees': self.degrees})
        agent.ori = agent.ori @ self._get_rotation_matrix(self.degrees)  
        return ActionResult(True, self.get_feedback(True), str(self), 'rotate', {'degrees': self.degrees})
    
    def __repr__(self):
        return f"Rotate({self.degrees})"


class ReturnAction(BaseAction):
    """Return to anchor position"""
    
    format_desc = "Return()"
    description = "Return to the starting position and orientation."
    example = "Return()"
    format_pattern = r"^Return\(\)$"
    cost = 0
    
    def success_message(self, **kwargs) -> str:
        return "You returned to starting position and orientation."
    
    def error_message(self, error_type: str) -> str:
        return "Cannot return to anchor: execution failed."
    
    def execute(self, room, agent, **kwargs) -> ActionResult:
        """Execute return action on room state."""
        # move to initial position
        agent.pos = agent.init_pos.copy()
        # rotate to initial orientation (compute delta)
        # report clockwise degrees from north; ensure consistency with _ori_to_deg in agent_proxy
        deg = {(0, 1): 0, (1, 0): 90, (0, -1): 180, (-1, 0): 270}[tuple(agent.ori)]
        agent.ori = agent.init_ori.copy()
        # restore room id if tracked
        if agent.init_room_id is not None:
            agent.room_id = agent.init_room_id
        return ActionResult(True, self.get_feedback(True), str(self), 'return', {'target_name': 'initial_pos', 'degrees': deg})
    
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
    cost = 1
    directional_template = "{obj_name}: {dir_str}"
    orientation_template = "{obj_name} facing {orientation}"
    # MODE: 'dir' for direction-only, 'full' for (dir, deg, dist)
    MODE: str = 'dir'
    
    def __init__(self):
        super().__init__()

    
    def _collect_obj_observations(self, agent, visible_objects, anchor_name: str, mode: str = 'dir', with_orientation: bool = True, discrete: bool = False):
        relationships: List[str] = []
        relation_triples: List[RelationTriple] = []
        for obj in visible_objects:
            answer_str = ""
            if discrete:
                rel = PairwiseRelationshipDiscrete.relationship(tuple(obj.pos), tuple(agent.pos), anchor_ori=tuple(agent.ori))
            else:
                rel = PairwiseRelationship.relationship(tuple(obj.pos), tuple(agent.pos), anchor_ori=tuple(agent.ori), full=(mode == 'full'))
            pairwise_str = rel.to_string()

            # orientation (gate/object) via OrientationRel only
            if isinstance(obj, Gate):
                ori_pair = OrientationRel.get_relative_orientation(tuple(obj.get_ori_for_room(int(agent.room_id))), tuple(agent.ori))
                ori_str = OrientationRel.to_string(ori_pair, 'ego', 'orientation', if_gate=True)
            else:
                ori_pair = OrientationRel.get_relative_orientation(tuple(obj.ori), tuple(agent.ori))
                ori_str = OrientationRel.to_string(ori_pair, 'ego', 'orientation')
            answer_str = f"{obj.name}: {pairwise_str}, {ori_str}"
            relationships.append(answer_str)
            relation_triples.append(RelationTriple(subject=obj.name, anchor=anchor_name, relation=rel, orientation=tuple(agent.ori)))
        
        final_answer = "\n" + "\n".join(f"• {rel}" for rel in relationships)
        return final_answer, relationships, relation_triples
    
    def success_message(self, **kwargs) -> str:
        return f"You observe: {kwargs.get('answer', 'nothing')}."
    
    def error_message(self, error_type: str) -> str:
        return "Cannot observe: execution failed."
    
    def execute(self, room, agent, **kwargs) -> ActionResult:
        """Execute observe action on room state. NOTE Also neglect same position objects."""
        
        # neglect objects in the same position
        neglect_objects = kwargs.get('neglect_objects', []) + [obj.name for obj in room.all_objects if np.allclose(obj.pos, agent.pos)]
        with_orientation = kwargs.get('with_orientation', True)
        visible_objects = [obj for obj in room.all_objects if self._is_visible(agent, obj) and obj.name not in neglect_objects]
        
        if not visible_objects:
            answer = "Nothing in view."
            return ActionResult(True, self.get_feedback(True, answer=answer), str(self), 'observe', {
                'answer': answer, 'visible_objects': [], 'relationships': []
            })

        anchor_name = self.get_anchor_name(room, agent) if not kwargs.get('free_position', False) else 'free_position'
        final_answer, relationships, relation_triples = self._collect_obj_observations(agent=agent, visible_objects=visible_objects, anchor_name=anchor_name, mode=self.MODE, with_orientation=with_orientation, discrete=False)

        
        return ActionResult(True, self.get_feedback(True, answer=final_answer), str(self), 'observe', {
            'answer': final_answer,
            'visible_objects': [obj.name for obj in visible_objects],
            'relationships': relationships,
            'relation_triples': relation_triples
        })
    
    @staticmethod
    def is_final() -> bool:
        return True
    
    def __repr__(self):
        return "Observe()"


class ObserveRelAction(ObserveAction):
    """Observe full relationships (dir, degree, distance) of all visible objects"""
    format_desc = "ObserveRel()"
    description = "Observe full relationships (direction, signed degree, distance) for all visible objects."
    example = "ObserveRel()"
    format_pattern = r"^ObserveRel\(\)$"
    MODE = 'full'
    cost = 1
    @staticmethod
    def is_final() -> bool:
        return True

    def __repr__(self):
        return "ObserveRel()"

class ObserveDirAction(ObserveAction):
    """Observe direction-only relationships of all visible objects"""
    format_desc = "ObserveDir()"
    description = "Observe direction-only relationships for all visible objects."
    example = "ObserveDir()"
    format_pattern = r"^ObserveDir\(\)$"
    MODE = 'dir'
    cost = 1
    @staticmethod
    def is_final() -> bool:
        return True

    def __repr__(self):
        return "ObserveDir()"


class ObserveApproxAction(ObserveAction):
    """Observe with approximate relations and local (near) pair descriptions"""
    format_desc = "ObserveApprox()"
    description = "Observe with approximate values; also report near pairs (left/right, closer/farther)."
    example = "ObserveApprox()"
    format_pattern = r"^ObserveApprox\(\)$"
    MODE = 'full'
    cost = 1
    @staticmethod
    def is_final() -> bool:
        return True

    def __repr__(self):
        return "ObserveApprox()"
    
    def _collect_local_relationships(self, agent, visible_objects, anchor_name: str):
        # proximity-based pair relations using discrete relationship binning
        relationships, relation_triples = [], []
        n = len(visible_objects)
        for i in range(n):
            for j in range(i + 1, n):
                a_obj, b_obj = visible_objects[i], visible_objects[j]
                # NOTE always use b_obj.ori for orientation
                prox_rel = ProximityRelationship.from_positions(tuple(a_obj.pos), tuple(b_obj.pos), tuple(b_obj.ori))
                if prox_rel is not None:
                    relationships.append(prox_rel.to_string(a_obj.name, b_obj.name))
                    relation_triples.append(RelationTriple(subject=a_obj.name, anchor=b_obj.name, relation=prox_rel, orientation=tuple(b_obj.ori)))
        final_answer = "\n".join(f"• {rel}" for rel in relationships)
        return final_answer, relationships, relation_triples

    def execute(self, room, agent, **kwargs) -> ActionResult:
        neglect_objects = kwargs.get('neglect_objects', []) + [obj.name for obj in room.all_objects if np.allclose(obj.pos, agent.pos)]
        with_orientation = kwargs.get('with_orientation', True)
        visible_objects = [obj for obj in room.all_objects if self._is_visible(agent, obj) and obj.name not in neglect_objects]
        if not visible_objects:
            answer = "Nothing in view."
            return ActionResult(True, self.get_feedback(True, answer=answer), str(self), 'observe_approx', {
                'answer': answer, 'visible_objects': [], 'relationships': [], 'local_relationships': []
            })

        anchor_name = self.get_anchor_name(room, agent) if not kwargs.get('free_position', False) else 'free_position'
        pairwise_answer, relationships, pairwise_relation_triples = self._collect_obj_observations(agent=agent, visible_objects=visible_objects, anchor_name=anchor_name, mode='full', with_orientation=with_orientation, discrete=True)
        local_answer, local_relationships, local_relation_triples = self._collect_local_relationships(agent, visible_objects, anchor_name)

        final_answer = f"{pairwise_answer}" + ((f"\nLocal relations:\n{local_answer}") if local_answer else "")
        return ActionResult(True, self.get_feedback(True, answer=final_answer), str(self), 'observe_approx', {
            'answer': final_answer,
            'visible_objects': [obj.name for obj in visible_objects],
            'relationships': relationships,
            'local_relationships': local_relationships,
            'relation_triples': pairwise_relation_triples + local_relation_triples
        })

class TermAction(BaseAction):
    """Terminate exploration"""
    
    format_desc = "Term()"
    description = "Terminate the exploration phase. Term() must be alone with no movement actions except for Return()."
    example = "Term()"
    format_pattern = r"^Term\(\)$"
    cost = 0
    def success_message(self, **kwargs) -> str:
        return "Exploration terminated."
    
    def error_message(self, error_type: str) -> str:
        return "Cannot terminate exploration: execution failed."
    
    def execute(self, room, agent, **kwargs) -> ActionResult:
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


class GoThroughDoorAction(BaseAction):
    """Go through a gate (door) connecting two rooms.
    Valid only if you are at the door position and facing the doorway:
    face opposite to the door's orientation for your current room (or the same as
    the door's orientation for the other room).
    """

    format_desc = "GoThroughDoor(door_name)"
    description = (
        "Go to the connected room. You must be at the door and face the doorway "
        "On success you appear in the other room facing into it. Your orientation will NOT change."
    )
    example = "GoThroughDoor(door1)"
    format_pattern = r"^GoThroughDoor\(([A-Za-z0-9_ -]+)\)$"
    cost = 0
    def __init__(self, door_name: str):
        super().__init__(door_name)
        self.door_name = door_name

    def success_message(self, **kwargs) -> str:
        rid = kwargs.get('room_id')
        return f"You pass through {self.door_name} into room {rid}." if rid is not None else f"You pass through {self.door_name}."

    def error_message(self, error_type: str) -> str:
        errors = {
            "not_found": "door not found",
            "not_gate": "target is not a door",
            "not_at_door": "you are not at the door position",
            "not_facing": "you are not facing the door",
        }
        return f"Cannot go through '{self.door_name}': {errors.get(error_type, 'execution failed')}."

    def execute(self, room, agent, **kwargs) -> ActionResult:
        if not room.has_object(self.door_name):
            return ActionResult(False, self.get_feedback(False, "not_found"), str(self), 'go_through_door', {})
        door = room.get_object_by_name(self.door_name)
        assert isinstance(door, Gate), "Door must be a Gate object"
        # Treat as door if it is listed in room.gates by name
        if self.door_name not in [g.name for g in room.gates]:
            return ActionResult(False, self.get_feedback(False, "not_gate"), str(self), 'go_through_door', {})
        

        # Must face the doorway and at the door position
        cur_rid = int(agent.room_id)
        door_connected_room_ids = [int(x) for x in door.room_id]
        assert len(door_connected_room_ids) == 2 and cur_rid in door_connected_room_ids, "Door must connect two rooms and you must be at one of the connected rooms"
        next_room_id = door_connected_room_ids[0] if door_connected_room_ids[1] == cur_rid else door_connected_room_ids[1]
        req_ori_other = door.get_ori_for_room(next_room_id)
        if not np.allclose(agent.pos, door.pos):
            return ActionResult(False, self.get_feedback(False, "not_at_door"), str(self), 'go_through_door', {})
        if not np.array_equal(agent.ori, req_ori_other):
            return ActionResult(False, self.get_feedback(False, "not_facing"), str(self), 'go_through_door', {})

        # move to next room with position and orientation unchanged
        agent.room_id = next_room_id
        return ActionResult(True, self.get_feedback(True, room_id=int(agent.room_id)), str(self), 'go_through_door', {"door": self.door_name, "room_id": int(agent.room_id)})


class QueryAction(BaseAction):
    """Query accurate spatial relationship between an object and the agent anchor"""

    format_desc = "Query(obj)"
    description = "Return accurate spatial relationship between the object from agent's perspective."
    example = "Query(table)"
    format_pattern = r"^Query\(([A-Za-z0-9_ -]+)\)$"
    cost = 5
    def __init__(self, obj: str):
        super().__init__(obj)
        self.obj = obj

    def success_message(self, **kwargs) -> str:
        return f"You query {self.obj}: {kwargs.get('answer','unknown')}"

    def error_message(self, error_type: str) -> str:
        return f"Cannot query: {error_type}"

    def execute(self, room, agent, **kwargs) -> ActionResult:
        if self.obj != 'initial_pos' and (not room.has_object(self.obj)):
            return ActionResult(False, self.get_feedback(False, "object not found"), str(self), 'query', {})
        # compute relationship from agent's CURRENT pose
        obj_pos = room.get_object_by_name(self.obj).pos if self.obj != 'initial_pos' else agent.init_pos
        rel = PairwiseRelationship.relationship(tuple(obj_pos), tuple(agent.pos), anchor_ori=tuple(agent.ori), full=True)
        ans = rel.to_string()
        return ActionResult(True, self.get_feedback(True, answer=ans), str(self), 'query',{
            "answer": ans,
            "object": self.obj,
            'relation_triples': [RelationTriple(subject=self.obj, anchor=self.get_anchor_name(room, agent), relation=rel, orientation=tuple(agent.ori))]
        })

    @staticmethod
    def is_final() -> bool: return False
    @staticmethod
    def is_query() -> bool: return True
    def __repr__(self): return f"Query({self.obj})"


# class QueryRelAction(BaseAction):
#     """Query accurate allocentric relationship between two objects"""

#     format_desc = "QueryRel(obj1, obj2)"
#     description = "Return accurate allocentric spatial relationship between two objects."
#     example = "QueryRel(table, chair)"
#     format_pattern = r"^QueryRel\(([A-Za-z0-9_ -]+),\s*([A-Za-z0-9_ -]+)\)$"
#     cost = 5
#     def __init__(self, obj1: str, obj2: str):
#         super().__init__((obj1, obj2))
#         self.obj1, self.obj2 = obj1, obj2

#     def success_message(self, **kwargs) -> str:
#         return f"Relationship: {self.obj1}→{self.obj2}: {kwargs.get('answer','unknown')}"

#     def error_message(self, error_type: str) -> str:
#         return f"Cannot query relationship: {error_type}"

#     def execute(self, room, agent, **kwargs) -> ActionResult:
#         if self.obj1 != 'initial_pos' and self.obj2 != 'initial_pos' and (not room.has_object(self.obj1) or not room.has_object(self.obj2)):
#             return ActionResult(False, self.get_feedback(False, "object not found"), str(self), 'query', {})
#         o1_pos = room.get_object_by_name(self.obj1).pos if self.obj1 != 'initial_pos' else agent.init_pos
#         o2_pos = room.get_object_by_name(self.obj2).pos if self.obj2 != 'initial_pos' else agent.init_pos
#         rel = PairwiseRelationship.relationship(tuple(o1_pos), tuple(o2_pos), anchor_ori=tuple(agent.init_ori), full=True)
#         ans = rel.to_string()
#         return ActionResult(True, self.get_feedback(True, answer=ans), str(self), 'query', {"answer": ans, "objects": [self.obj1, self.obj2], "pair": [self.obj1, self.obj2]})

#     @staticmethod
#     def is_final() -> bool: return False
#     @staticmethod
#     def is_query() -> bool: return True
#     def __repr__(self): return f"QueryRel({self.obj1}, {self.obj2})"










# Action registry for easy lookup
# Expose all observe variants; default flows may still prefer ObserveApprox
ACTION_CLASSES = [
    MoveAction, RotateAction, ReturnAction, GoThroughDoorAction,
    ObserveApproxAction, TermAction, QueryAction
]


class ActionSequence:
    """Sequence of actions for spatial exploration"""
    
    def __init__(self, motion_actions: List[BaseAction] = None, final_action: BaseAction = None, query_actions: List[BaseAction] = None):
        self.motion_actions = motion_actions or []
        self.final_action = final_action
        self.query_actions = query_actions or []
    
    def __repr__(self):
        motions = ", ".join(str(action) for action in self.motion_actions)
        queries = ", ".join(str(action) for action in self.query_actions)
        return f"ActionSequence(motions=[{motions}], queries=[{queries}], final={self.final_action})"

    @classmethod
    def parse(cls, action_str: str) -> Optional['ActionSequence']:
        m = re.search(r'\[(.*)\]', action_str.strip())
        if not m:
            return None
        # extract top-level actions like Move(table), Rotate(90), Term()
        action_strs = re.findall(r'([A-Za-z]+\([^()]*\))', m.group(1))
        if not action_strs:
            return None

        # Parse all actions
        parsed_actions = []
        for act_s in action_strs:
            act = cls._parse_single_action(act_s.strip())
            if not act:
                return None
            parsed_actions.append(act)

        # If any query action present, enforce query-only sequence
        if any(a.is_query() for a in parsed_actions):
            if not all(a.is_query() for a in parsed_actions):
                return None
            return cls([], None, parsed_actions)

        # Otherwise, standard: zero or more motions then exactly one final
        motions, final_action = [], None
        for i, act in enumerate(parsed_actions):
            if i == len(parsed_actions) - 1:
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
        motion_actions = [cls for cls in ACTION_CLASSES if not cls.is_final() and not cls.is_query()]
        final_actions = [cls for cls in ACTION_CLASSES if cls.is_final()]
        query_actions = [cls for cls in ACTION_CLASSES if cls.is_query()]
        
        action_desc = (
            "Movement Actions:\n" +
            "\n".join(f"- {cls.format_desc}: {cls.description}" for cls in motion_actions) +
            "\n\n" +
            "Final Actions:\n" +
            "\n".join(f"- {cls.format_desc}: {cls.description}" for cls in final_actions) +
            "\n\n" +
            "Query Actions (use alone, can list multiple):\n" +
            "\n".join(f"- {cls.format_desc}: {cls.description}" for cls in query_actions)
        )
        examples = (
            f"Valid Example:\nActions: [Move(table), Rotate(90), Observe()]\n\n" +
            f"Valid Example:\nActions: [Observe()]\n\n" +
            f"Valid Example (queries only):\nActions: [Query(table), Query(lamp)]\n\n" +
            f"Invalid Example (no final action):\nActions: [Move(table)]\n\n"+
            f"Invalid Example (more than one final action):\nActions: [Observe(), Rotate(90), Observe()]\n\n" +
            f"Invalid Example (termination with other actions):\nActions: [Move(table), Term()]\n\n" +
            f"Invalid Example (mixing queries with others):\nActions: [Move(table), Query(a)]\n\n"
        )
        
        return ACTION_INSTRUCTION.format(
            actions=action_desc,
            examples=examples,
            field_of_view=BaseAction.get_field_of_view(),
            costs="\n".join(f"- {cls.format_desc}: {cls.cost}" for cls in ACTION_CLASSES)
        )


if __name__ == "__main__":
    # simple smoke tests
    import numpy as np
    from ..core.object import Object, Agent
    from ..core.room import Room
    from ..managers.exploration_manager import ExplorationManager

    objs = [
        Object('table', np.array([1, 2]), np.array([0, 1])),
        Object('chair', np.array([2, 2]), np.array([0, 1])),
        Object('lamp', np.array([1, 3]), np.array([0, 1])),
    ]
    mask = np.ones((6, 6), dtype=np.int8)
    room = Room(objects=objs, mask=mask, name='r')
    agent = Agent(pos=np.array([1, 1]), ori=np.array([0, 1]), room_id=1, init_room_id=1)
    mgr = ExplorationManager(room, agent)

    # ObserveApprox
    seq = ActionSequence.parse("Actions: [ObserveApprox()]")
    info, results = mgr.execute_action_sequence(seq)
    print('ObserveApprox ->', results[0].message)

    # Query-only: two queries
    seq = ActionSequence.parse("Actions: [Query(table), Query(lamp)]")
    info, results = mgr.execute_action_sequence(seq)
    print('Query ->', "; ".join(r.message for r in results))
    print('Counts:', mgr.action_counts, 'Cost:', mgr.action_cost)