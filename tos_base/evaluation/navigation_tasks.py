"""Forward/Backward navigation tasks with shared helpers.

ForwardFOVEvaluationTask: predict final observation from an action sequence.
BackwardNavEvaluationTask: infer action sequence from a final observation.
"""

from typing import Any, List, Tuple
import numpy as np

from .tasks import BaseEvaluationTask, retry_generate_question
from ..core.object import Agent, Gate
from ..core.relationship import PairwiseRelationshipDiscrete, EgoFrontBins, StandardDistanceBins
from ..actions import ObserveAction, RotateAction, MoveAction
from ..managers.exploration_manager import ExplorationManager
from ..utils.utils import hash

# Nav action descriptor: ('rotate', degrees) or ('jumpto', object_name)
NavAction = Tuple[str, Any]

# ---- Small helpers ----
def _closest_cardinal(vec: np.ndarray) -> np.ndarray:
    """Return closest cardinal direction vector."""
    basis = [np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0])]
    dots = [float(np.dot(vec, b)) for b in basis]
    return basis[int(np.argmax(dots))]


def _ordinal(n: int) -> str:
    n = int(n)
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _nearfar_phrase(index: int, total: int) -> str:
    index = int(index)
    total = int(total)
    if index == 1:
        return "nearest"
    if index == total:
        return "farthest"
    return f"{_ordinal(index)} nearest"


def _ori_to_deg(ori: Tuple[int, int]) -> int:
    """Convert orientation tuple to degrees."""
    mapping = {(0, 1): 0, (1, 0): 90, (0, -1): 180, (-1, 0): 270}
    return mapping[tuple(int(x) for x in ori)]


def _rotate_ori(ori: Tuple[int, int], degrees: int) -> Tuple[int, int]:
    mapping = {0: (0, 1), 90: (1, 0), 180: (0, -1), 270: (-1, 0)}
    cur = _ori_to_deg(ori)
    new_deg = (cur + degrees) % 360
    return mapping.get(new_deg, ori)


def _rotation_delta(current: Tuple[int, int], desired: Tuple[int, int]) -> int:
    cur = _ori_to_deg(current)
    des = _ori_to_deg(desired)
    return (des - cur + 540) % 360 - 180


class BaseNavEvaluationTask(BaseEvaluationTask):
    """Shared navigation helpers for both tasks."""

    def _agent_from_init(self) -> Agent:
        """Reset agent to initial position and orientation."""
        a = self.agent.copy()
        a.pos = self.agent.init_pos.copy()
        a.ori = self.agent.init_ori.copy()
        a.room_id = getattr(self.agent, 'init_room_id', None) or a.room_id
        if a.room_id is None:
            info = self.room.get_cell_info(int(a.pos[0]), int(a.pos[1]))
            a.room_id = info.get('room_id', a.room_id)
        return a

    def _move_simple(self, agent: Agent, name: str) -> None:
        """Move agent to object position."""
        obj = self.room.get_object_by_name(name)
        agent.pos = obj.pos.copy()
        agent.room_id = obj.room_id

    def _current_rooms(self, agent: Agent) -> List[int]:
        """Get current room IDs for agent."""
        rid = getattr(agent, 'room_id', None)
        if isinstance(rid, list):
            return [int(x) for x in rid]
        if rid is None:
            info = self.room.get_cell_info(int(agent.pos[0]), int(agent.pos[1]))
            rid = info.get('room_id')
        return [int(rid)] if rid is not None else []

    def _candidates_in_rooms(self, rooms: List[int]) -> List[str]:
        """Get all object names in given rooms."""
        names: List[str] = []
        for rid in rooms:
            names.extend(self.room.objects_by_room.get(int(rid), []))
            names.extend(self.room.gates_by_room.get(int(rid), []))
        return list(dict.fromkeys(names))

    def _generate_plan(self, steps: int = 2) -> List[NavAction]:
        """Generate navigation plan that keeps JumpTo targets in view."""
        a = self._agent_from_init()
        plan: List[NavAction] = []
        last_was_gate = False
        other_rooms_after_gate: List[int] = []
        
        for _ in range(int(steps)):
            rooms = self._current_rooms(a)
            cand = [n for n in self._candidates_in_rooms(rooms) 
                   if not np.allclose(self.room.get_object_by_name(n).pos, a.pos)]
            if not cand:
                break
            
            gate_cand = [n for n in cand if isinstance(self.room.get_object_by_name(n), Gate)]
            non_gate = [n for n in cand if n not in gate_cand]
            
            if last_was_gate:
                objects_in_other_rooms = [n for n in non_gate 
                                         if self.room.get_object_by_name(n).room_id in other_rooms_after_gate]
                pool = objects_in_other_rooms or non_gate or gate_cand
                name = str(self.np_random.choice(pool))
            else:
                if gate_cand and int(self.np_random.integers(0, 10)) < 6:
                    name = str(self.np_random.choice(gate_cand))
                else:
                    name = str(self.np_random.choice(non_gate or cand))
            
            target = self.room.get_object_by_name(name)
            desired_ori = _closest_cardinal(target.pos - a.pos)
            delta = _rotation_delta(tuple(a.ori), tuple(desired_ori))
            if int(delta) != 0:
                plan.append(('rotate', int(delta)))
                a.ori = np.array(_rotate_ori(tuple(a.ori), int(delta)))

            plan.append(('jumpto', name))
            self._move_simple(a, name)
            
            if isinstance(self.room.get_object_by_name(name), Gate):
                last_was_gate = True
                gobj = self.room.get_object_by_name(name)
                other_rooms_after_gate = [int(r) for r in list(gobj.room_id) if int(r) not in rooms]
            else:
                last_was_gate = False
                other_rooms_after_gate = []
        # Align to a valid viewing orientation with at least one visible object
        valid_oris = []
        for ori in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            tmp = a.copy()
            tmp.ori = np.array(ori)
            if ObserveAction().execute(self.room, tmp).data.get('visible_objects', []):
                valid_oris.append(ori)
        final_ori = tuple(valid_oris[int(self.np_random.integers(0, len(valid_oris)))] if valid_oris else (0, 1))
        delta = _rotation_delta(tuple(a.ori), tuple(final_ori))
        if int(delta) != 0:
            plan.append(('rotate', int(delta)))
            a.ori = np.array(_rotate_ori(tuple(a.ori), int(delta)))
        return plan

    def _execute_plan(self, plan: List[NavAction]) -> Agent:
        """Execute navigation plan and return final agent state."""
        mgr = ExplorationManager(self.room.copy(), self._agent_from_init())
        for action_type, value in plan:
            if action_type == 'rotate':
                mgr.execute_success_action(RotateAction(int(value)))
            elif action_type == 'jumpto':
                target = mgr.exploration_room.get_object_by_name(str(value))
                assert MoveAction._is_visible(mgr.agent, target), f"Target '{value}' must be visible before JumpTo."
                mgr.observed_items.add(str(value))
                mgr.execute_success_action(MoveAction(str(value)), move_anyway=True)
            else:
                raise ValueError(f"Unknown navigation action: {action_type}")
        return mgr.agent.copy()

    def _get_visible_objects(self, agent: Agent) -> List[Tuple[str, str, str]]:
        """Get visible objects with their relations. Returns list of (name, direction, distance)."""
        res = ObserveAction().execute(self.room, agent)
        triples = res.data.get('relation_triples', [])
        objects = []
        for tr in triples:
            if isinstance(tr.relation, PairwiseRelationshipDiscrete):
                objects.append((tr.subject, tr.relation.direction.bin_label, tr.relation.dist.bin_label))
        return objects

    def _describe_target(self, mgr: ExplorationManager, target_name: str) -> str:
        bin_sys = EgoFrontBins()
        dist_sys = StandardDistanceBins()
        target = self.room.get_object_by_name(target_name)
        rel_t = PairwiseRelationshipDiscrete.relationship(
            tuple(target.pos),
            tuple(mgr.agent.pos),
            anchor_ori=tuple(mgr.agent.ori),
            bin_system=bin_sys,
            distance_bin_system=dist_sys,
        )
        dir_label = rel_t.direction.bin_label
        dist_label = rel_t.dist.bin_label

        dir_group = []
        dist_group = []
        visible = mgr.execute_success_action(ObserveAction()).data.get('visible_objects', [])
        for name in visible:
            obj = self.room.get_object_by_name(name)
            rel = PairwiseRelationshipDiscrete.relationship(
                tuple(obj.pos),
                tuple(mgr.agent.pos),
                anchor_ori=tuple(mgr.agent.ori),
                bin_system=bin_sys,
                distance_bin_system=dist_sys,
            )
            if int(rel.direction.bin_id) == int(rel_t.direction.bin_id):
                dir_group.append((obj, float(rel.direction.degree)))
            if int(rel.dist.bin_id) == int(rel_t.dist.bin_id):
                dist_group.append((obj, float(np.linalg.norm(np.array(obj.pos) - np.array(mgr.agent.pos)))))

        dir_phrase = None
        if len(dir_group) > 1:
            dir_group.sort(key=lambda item: item[1])
            idx = 1 + next(i for i, (obj, _) in enumerate(dir_group) if obj.name == target.name)
            dir_phrase = f"{_ordinal(idx)} from left"

        dist_phrase = None
        if len(dist_group) > 1:
            dist_group.sort(key=lambda item: item[1])
            idx = 1 + next(i for i, (obj, _) in enumerate(dist_group) if obj.name == target.name)
            dist_phrase = f"{_nearfar_phrase(idx, len(dist_group))} one"

        if dir_phrase or dist_phrase:
            descriptors = " also ".join(filter(None, (dir_phrase, dist_phrase)))
            return f"Among objects which are {dir_label}, {dist_label}, you jump to the {descriptors}."

        return f"Jump to the object at {dir_label}, {dist_label}."

    def _plan_to_text(self, plan: List[NavAction]) -> str:
        mgr = ExplorationManager(self.room.copy(), self._agent_from_init())
        steps = []
        for idx, (act, value) in enumerate(plan, start=1):
            if act == 'rotate':
                mgr.execute_success_action(RotateAction(int(value)))
                label = f"Rotate({int(value)})"
            elif act == 'jumpto':
                name = str(value)
                target = mgr.exploration_room.get_object_by_name(name)
                description = self._describe_target(mgr, name)
                assert MoveAction._is_visible(mgr.agent, target), f"Target '{name}' must be visible before JumpTo."
                mgr.observed_items.add(name)
                mgr.execute_success_action(MoveAction(name), move_anyway=True)
                label = description
            else:
                raise ValueError(f"Unknown navigation action: {act}")
            steps.append(f"{idx}. {label}")
        return "\n".join(steps)

    def _sample_plan_with_visible(self, steps: int, max_attempts: int = 5) -> Tuple[List[NavAction], Agent, List[Tuple[str, str, str]]]:
        attempts = max(1, min(int(self.config.get('plan_retry', max_attempts)), max_attempts))
        for _ in range(attempts):
            plan = self._generate_plan(steps)
            agent = self._execute_plan(plan)
            visible = self._get_visible_objects(agent)
            if visible:
                return plan, agent, visible
        raise ValueError("Failed to generate navigation plan with visible objects")

class ForwardFOVEvaluationTask(BaseNavEvaluationTask):
    """Predict final observation from an action sequence."""
    QUESTION_TEMPLATE = (
        "You return to your starting position and face north.\n"
        "You will execute the following action sequence:\n"
        "{actions}\n\n"
        "After executing the actions, what is the egocentric relation of {target}?\n\n"
        "Answer format: {target} is at <direction>, <distance>\n"
        "Example: {target} is at front, near\n"
    )

    @retry_generate_question
    def generate_question(self) -> str:
        steps = int(self.config.get('steps', 2))
        plan, end_agent, visible = self._sample_plan_with_visible(steps)
        actions_str = self._plan_to_text(plan)

        self.np_random.shuffle(visible)
        target_name, direction, distance = visible[0]
        answer = f"{target_name} is at {direction}, {distance}"

        self.eval_data.question = self.QUESTION_TEMPLATE.format(actions=actions_str, target=target_name)
        self.eval_data.answer = answer
        self.eval_data.choices = []
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question

class BackwardNavEvaluationTask(BaseNavEvaluationTask):
    """Infer action sequence from final observation."""
    QUESTION_TEMPLATE = (
        "You return to your starting position and face north.\n"
        "Then you have executed an action sequence and changed to a new location and facing direction.\n"
        "You observe the following:\n"
        "{final_obs}\n\n"
        "What action sequence led to this final view?\n\n"
        "Answer format: JumpTo(object1), JumpTo(object2), Rotate(degrees)\n"
        "Example: JumpTo(lamp), JumpTo(chair), Rotate(90)\n"
    )

    @retry_generate_question
    def generate_question(self) -> str:
        steps = int(self.config.get('steps', 2))
        plan, end_agent, visible = self._sample_plan_with_visible(steps)
        self.np_random.shuffle(visible)
        visible = visible[:3]
        obs_parts = [f"{name} is at {direction}, {distance}" 
                    for name, direction, distance in visible]
        final_obs = "; ".join(obs_parts)
        
        # Store expected final state and object positions for evaluation
        init_agent = self._agent_from_init()
        object_positions = {obj.name: tuple(map(int, obj.pos)) for obj in self.room.all_objects}
        
        answer = {
            'final_pos': tuple(map(int, end_agent.pos)),
            'final_ori': tuple(map(int, end_agent.ori)),
            'init_pos': tuple(map(int, init_agent.pos)),
            'init_ori': tuple(map(int, init_agent.ori)),
            'object_positions': object_positions,
            'final_observation': [
                {
                    'name': name,
                    'direction': direction,
                    'distance': distance,
                }
                for name, direction, distance in visible
            ],
        }
        
        self.eval_data.question = self.QUESTION_TEMPLATE.format(final_obs=final_obs)
        self.eval_data.answer = answer
        self.eval_data.choices = []
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question


