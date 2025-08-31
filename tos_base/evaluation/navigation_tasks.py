"""Forward/Backward navigation tasks with shared helpers.

ForwardFOVEvaluationTask: predict final observation from an action sequence.
BackwardNavEvaluationTask: infer action sequence from a final observation.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from .tasks import BaseEvaluationTask
from ..core.object import Agent, Object
from ..core.relationship import EgoFrontBins, StandardDistanceBins, PairwiseRelationshipDiscrete
from ..actions import BaseAction, ObserveApproxAction, RotateAction, MoveAction
from ..managers.exploration_manager import ExplorationManager
from ..utils.action_utils import action_results_to_text


"""Small, shared helpers. Use actions (Observe/Rotate/Move) via ExplorationManager."""

def _closest_cardinal(vec: np.ndarray) -> np.ndarray:
    basis = [np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0])]
    dots = [float(np.dot(vec, b)) for b in basis]
    return basis[int(np.argmax(dots))]


def _ori_to_name(ori: Tuple[int, int]) -> str:
    mapping = {(0, 1): "north", (1, 0): "east", (0, -1): "south", (-1, 0): "west"}
    return mapping.get(tuple(int(x) for x in ori), "north")


def _ordinal(n: int) -> str:
    return f"{int(n)}th" if n > 3 else "1st" if n == 1 else "2nd" if n == 2 else "3rd" if n == 3 else f"{int(n)}th"


def _nearfar_phrase(i: int, k: int) -> str:
    if i == 1: return "nearest"
    if i == k: return "farthest"
    return f"{_ordinal(i)} nearest"


def _ori_to_deg(ori: Tuple[int, int]) -> int:
    mapping = {(0, 1): 0, (1, 0): 90, (0, -1): 180, (-1, 0): 270}
    return mapping[tuple(int(x) for x in ori)]


class BaseNavEvaluationTask(BaseEvaluationTask):
    """Shared navigation helpers for both tasks."""

    def _agent_from_init(self) -> Agent:
        a = self.agent.copy(); a.pos = self.agent.init_pos.copy(); a.ori = self.agent.init_ori.copy()
        a.room_id = self.agent.init_room_id if getattr(self.agent, 'init_room_id', None) is not None else a.room_id
        if a.room_id is None:
            info = self.room.get_cell_info(int(a.pos[0]), int(a.pos[1])); a.room_id = info.get('room_id', a.room_id)
        return a

    def _new_mgr(self, agent: Agent | None = None) -> ExplorationManager:
        return ExplorationManager(self.room.copy(), (agent or self._agent_from_init()))

    def _visible_names(self, mgr: ExplorationManager) -> List[str]:
        return mgr.execute_success_action(ObserveApproxAction()).data['visible_objects']

    def _rotate_to_face(self, mgr: ExplorationManager, target: Object):
        vec = target.pos - mgr.agent.pos
        if np.allclose(vec, 0):
            return None
        desired = _closest_cardinal(vec)
        cur, des = _ori_to_deg(tuple(int(x) for x in mgr.agent.ori)), _ori_to_deg(tuple(int(x) for x in desired))
        delta = (des - cur + 540) % 360 - 180
        if int(delta) == 0:
            return None
        return mgr.execute_success_action(RotateAction(int(delta)))

    def _describe_target(self, mgr: ExplorationManager, target: Object) -> str:
        bin_sys, dist_sys = EgoFrontBins(), StandardDistanceBins()
        rel_t = PairwiseRelationshipDiscrete.relationship(tuple(target.pos), tuple(mgr.agent.pos), anchor_ori=tuple(mgr.agent.ori), bin_system=bin_sys, distance_bin_system=dist_sys)
        dir_label, dist_label = rel_t.direction.bin_label, rel_t.dist.bin_label
        # Build groups via current Observe; indices only when multiple
        dir_group, dist_group = [], []
        for name in self._visible_names(mgr):
            o = self.room.get_object_by_name(name)
            rel = PairwiseRelationshipDiscrete.relationship(tuple(o.pos), tuple(mgr.agent.pos), anchor_ori=tuple(mgr.agent.ori), bin_system=bin_sys, distance_bin_system=dist_sys)
            deg = float(rel.direction.degree)
            if int(rel.direction.bin_id) == int(rel_t.direction.bin_id):
                dir_group.append((o, deg))
            if int(rel.dist.bin_id) == int(rel_t.dist.bin_id):
                dval = float(np.linalg.norm(np.array(o.pos) - np.array(mgr.agent.pos)))
                dist_group.append((o, dval))
        extras: List[str] = []
        if len(dir_group) > 1:
            dir_group.sort(key=lambda x: x[1])
            idx = 1 + next(i for i, (o, _) in enumerate(dir_group) if o.name == target.name)
            extras.append(f"at {dir_label} it's {_ordinal(idx)} from left")
        if len(dist_group) > 1:
            dist_group.sort(key=lambda x: x[1])
            i = 1 + next(i for i, (o, _) in enumerate(dist_group) if o.name == target.name)
            extras.append(f"at {dist_label} it's {_nearfar_phrase(i, len(dist_group))}")
        base = f"move to the object at {dir_label}, {dist_label}"
        return base + ("; " + "; ".join(extras) if extras else "") + "."

    def build_action_sequence(self, sequence: List[str], final_ori: Tuple[int, int]) -> Tuple[List[List], Agent]:
        """Return per-step ActionResults groups and end agent. Each target yields [Rotate?, Move]. Final group is Rotate if needed."""
        mgr = self._new_mgr()
        per_step: List[List] = []
        for name in sequence:
            target = mgr.exploration_room.get_object_by_name(name)
            group: List = []
            rot_res = self._rotate_to_face(mgr, target)
            if rot_res is not None:
                group.append(rot_res)
            # compute description BEFORE moving
            desc = self._describe_target(mgr, target)
            obs_names = set(mgr.execute_success_action(ObserveApproxAction()).data['visible_objects'])
            move_res = mgr.execute_success_action(MoveAction(name), observed_items=obs_names)
            move_res.message = desc
            group.append(move_res)
            per_step.append(group)
        cur, des = _ori_to_deg(tuple(int(x) for x in mgr.agent.ori)), _ori_to_deg(tuple(int(x) for x in final_ori))
        delta = (des - cur + 540) % 360 - 180
        if int(delta) != 0:
            per_step.append([mgr.execute_success_action(RotateAction(int(delta)))])
        return per_step, mgr.agent.copy()

    def action_sequence_to_string(self, per_step: List[List], include_final_face: bool) -> str:
        """Render numbered lines; optionally drop the last rotate group."""
        groups = list(per_step)
        if (not include_final_face) and groups and len(groups[-1]) == 1 and getattr(groups[-1][0], 'action_type', '') == 'rotate':
            groups = groups[:-1]
        return "\n".join(f"{i+1}. {action_results_to_text(group)}" for i, group in enumerate(groups))

    def _current_rooms(self, agent: Agent) -> List[int]:
        rid = getattr(agent, 'room_id', None)
        if isinstance(rid, list):
            return [int(x) for x in rid]
        if rid is None:
            info = self.room.get_cell_info(int(agent.pos[0]), int(agent.pos[1])); rid = info.get('room_id')
        return [int(rid)] if rid is not None else []

    def _candidates_in_rooms(self, rooms: List[int]) -> List[str]:
        names: List[str] = []
        for rid in rooms:
            names.extend(self.room.objects_by_room.get(int(rid), []))
            names.extend(self.room.gates_by_room.get(int(rid), []))
        return list(dict.fromkeys(names))

    def _generate_plan(self, steps: int = 2) -> Tuple[List[str], Tuple[int, int]]:
        """Sample next targets from current rooms (or gates), rotating then moving each step.
        Final orientation guarantees ≥1 object in FOV.
        """
        mgr = self._new_mgr()
        seq: List[str] = []
        for _ in range(int(steps)):
            rooms = self._current_rooms(mgr.agent)
            cand = [n for n in self._candidates_in_rooms(rooms) if not np.allclose(self.room.get_object_by_name(n).pos, mgr.agent.pos)]
            if not cand:
                break
            name = str(self.np_random.choice(cand))
            target = mgr.exploration_room.get_object_by_name(name)
            _ = self._rotate_to_face(mgr, target)
            observed = set(mgr.execute_success_action(ObserveApproxAction()).data['visible_objects'])
            _ = mgr.execute_success_action(MoveAction(name), observed_items=observed)
            seq.append(name)
        # choose final orientation with ≥1 visible object
        valid_oris = []
        for ori in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            tmp_mgr = self._new_mgr(mgr.agent.copy()); tmp_mgr.agent.ori = np.array(ori)
            if self._visible_names(tmp_mgr):
                valid_oris.append(ori)
        final_ori = tuple(valid_oris[int(self.np_random.integers(0, len(valid_oris)))] if valid_oris else (0, 1))
        return seq, final_ori

    def _final_obs_text(self, end_agent: Agent) -> str:
        mgr = self._new_mgr(end_agent.copy())
        res = mgr.execute_success_action(ObserveApproxAction())
        return action_results_to_text([res])

    def _pairs_all(self, end_agent: Agent) -> List[str]:
        """Return all pairwise relationship strings (exclude local) from end agent's perspective."""
        mgr = self._new_mgr(end_agent.copy())
        return mgr.execute_success_action(ObserveApproxAction()).data['relationships']

    def _pairs_text(self, end_agent: Agent, max_items: int = 3) -> str:
        """Shuffle and join up to max_items pairwise lines in one sentence."""
        pairs = self._pairs_all(end_agent)
        self.np_random.shuffle(pairs)
        pick = pairs[:max(0, min(max_items, len(pairs)))]
        return "; ".join(pick)

    def _is_wrong_forward(self, candidate_text: str, correct_all: set[str]) -> bool:
        parts = [p.strip() for p in candidate_text.split(';') if p.strip()]
        return not all(p in correct_all for p in parts)

    def _is_wrong_backward(self, seq: List[str], final_ori: Tuple[int, int], correct_all: set[str]) -> bool:
        """True if executing (seq, final_ori) yields a DIFFERENT set of pairwise relationships."""
        _, end_agent = self.build_action_sequence(seq, final_ori)
        return set(self._pairs_all(end_agent)) != set(correct_all)

    def _random_forward_obs(self, end_agent: Agent) -> str:
        """Make a random slight variant: move to a random visible object's pos or randomize orientation, then return pairwise text."""
        a = end_agent.copy()
        vis_names = self._visible_names(self._new_mgr(a.copy()))
        if vis_names:
            pick = str(self.np_random.choice(vis_names))
            a.pos = self.room.get_object_by_name(pick).pos.copy()
        a.ori = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)][int(self.np_random.integers(0, 4))])
        return self._pairs_text(a, max_items=3)

    def _random_backward_action_string(self) -> Tuple[str, List[str], Tuple[int, int]]:
        """Generate a random short sequence and final orientation, then render action string with final face line."""
        seq, final_ori = self._generate_plan(self.steps)
        per_step, _ = self.build_action_sequence(seq, final_ori)
        return self.action_sequence_to_string(per_step, include_final_face=True), seq, final_ori

class ForwardFOVEvaluationTask(BaseNavEvaluationTask):
    """Predict final observation from an action sequence."""

    QUESTION_TEMPLATE = (
        "You will execute a short action sequence. Always rotate to bring the next target into view before moving.\n"
        "Actions:\n{actions}\n\n"
        "What will you observe at the end?\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    def generate_choices(self, correct_answer: Any) -> Tuple[List[str], int]:
        """Randomly mix wrong strategies; forward options are pairwise-only (no local/prefix), ≤3 lines."""
        end_agent: Agent = self._ctx['end_agent']
        final_ori: Tuple[int, int] = tuple(self._ctx['final_ori'])
        correct_obs = str(correct_answer)
        choices, seen = [correct_obs], {correct_obs}
        # orientation variants (pairwise only)
        wrong_ori: List[str] = []
        for ori in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if tuple(ori) == tuple(final_ori):
                continue
            a = end_agent.copy(); a.ori = np.array(ori)
            wrong_ori.append(self._pairs_text(a, max_items=3))
        # position variants near→far (pairwise only)
        wrong_pos: List[str] = []
        final_pos = end_agent.pos.copy()
        others = [o for o in self.room.all_objects if not np.allclose(o.pos, final_pos)]
        others.sort(key=lambda o: float(np.linalg.norm(o.pos - final_pos)))
        for o in others:
            b = end_agent.copy(); b.pos = o.pos.copy()
            if int(self.np_random.integers(0, 2)) == 1:
                b.ori = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)][int(self.np_random.integers(0, 4))])
            wrong_pos.append(self._pairs_text(b, max_items=3))
        pool = wrong_ori + wrong_pos
        self.np_random.shuffle(pool)
        # ensure wrong not subset of correct full set; backfill if needed
        correct_all = set(self._pairs_all(end_agent))
        for s in pool:
            if len(choices) == 4: break
            if s and (s not in seen) and self._is_wrong_forward(s, correct_all):
                choices.append(s); seen.add(s)
        while len(choices) < 4:
            s = self._random_forward_obs(end_agent)
            if s and (s not in seen) and self._is_wrong_forward(s, correct_all):
                choices.append(s); seen.add(s)
        self.np_random.shuffle(choices)
        choices = choices[:4]
        return choices, int(choices.index(correct_obs))

    def generate_question(self) -> str:
        self.steps = int(self.config.get('steps', 2))
        seq, final_ori = self._generate_plan(self.steps)
        per_step, end_agent = self.build_action_sequence(seq, final_ori)
        actions_str = self.action_sequence_to_string(per_step, include_final_face=False)
        correct_obs = self._pairs_text(end_agent, max_items=3)
        self._ctx = {'end_agent': end_agent.copy(), 'final_ori': tuple(final_ori)}
        choices, correct_idx = self.generate_choices(correct_obs)
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        self.eval_data.question = self.QUESTION_TEMPLATE.format(actions=actions_str, choices_text=choices_text)
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question

class BackwardNavEvaluationTask(ForwardFOVEvaluationTask):
    """Infer action sequence from final observation."""

    QUESTION_TEMPLATE = (
        "You see the final observation below (after some actions from the start).\n"
        "{final_obs}\n\n"
        "Which action sequence led to this final view? Always rotate before moving.\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    def _wrong_by_orientation(self, seq: List[str], final_ori: Tuple[int, int], avoid: Tuple[int, int]) -> Optional[Tuple[str, List[str], Tuple[int, int]]]:
        for ori in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if tuple(ori) == tuple(avoid): continue
            per_step, _ = self.build_action_sequence(seq, ori)
            return self.action_sequence_to_string(per_step, include_final_face=True), list(seq), tuple(ori)
        return None

    def _wrong_by_final_object(self, seq: List[str], final_ori: Tuple[int, int]) -> Optional[Tuple[str, List[str], Tuple[int, int]]]:
        if not seq: return None
        pool = [o.name for o in self.room.all_objects if o.name != seq[-1]]
        self.np_random.shuffle(pool)
        for n in pool:
            alt = list(seq[:-1] + [n])
            per_step, _ = self.build_action_sequence(alt, final_ori)
            return self.action_sequence_to_string(per_step, include_final_face=True), alt, tuple(final_ori)
        return None

    def generate_choices(self, correct_answer: Any) -> Tuple[List[str], int]:
        seq, final_ori = list(self._ctx['seq']), tuple(self._ctx['final_ori'])
        correct = str(correct_answer)
        # compute the correct set of pairwise relationships once
        correct_all = set(self._pairs_all(self.build_action_sequence(seq, final_ori)[1]))

        wrong: List[str] = []
        strategies = [
            lambda: self._wrong_by_orientation(seq, final_ori, avoid=final_ori),
            lambda: self._wrong_by_final_object(seq, final_ori),
            lambda: self._random_backward_action_string(),
        ]
        self.np_random.shuffle(strategies)

        while len(wrong) < 3 and strategies:
            cand = strategies.pop()()
            if not cand: continue
            txt, s, o = cand
            if txt not in wrong and self._is_wrong_backward(s, o, correct_all):
                wrong.append(txt)

        choices = [correct] + wrong[:3]
        self.np_random.shuffle(choices)
        return choices, int(choices.index(correct))

    def generate_question(self) -> str:
        self.steps = int(self.config.get('max_steps', 2))
        seq, final_ori = self._generate_plan(self.steps)
        per_step, end_agent = self.build_action_sequence(seq, final_ori)
        final_obs = self._final_obs_text(end_agent)
        correct_actions = self.action_sequence_to_string(per_step, include_final_face=True)
        self._ctx = {'seq': list(seq), 'final_ori': tuple(final_ori)}
        choices, correct_idx = self.generate_choices(correct_actions)
        choices_text, correct_label = self.format_choices(choices, correct_idx)
        self.eval_data.question = self.QUESTION_TEMPLATE.format(final_obs=final_obs, choices_text=choices_text)
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question


