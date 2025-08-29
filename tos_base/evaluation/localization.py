"""Localization task: infer your 2D coordinate from a new view."""

from typing import List, Tuple, Dict, Any
import numpy as np

from .tasks import BaseEvaluationTask
from ..core.object import Object
from ..core.relationship import StandardDistanceBins, PairwiseRelationshipDiscrete, EgoFrontBins
from ..actions import BaseAction
from ..utils.action_utils import action_results_to_text
from ..actions import ObserveApproxAction

class LocalizationEvaluationTask(BaseEvaluationTask):
    """Localize your own coordinate (x, y). TODO treat ... as origin, including orientation?"""

    QUESTION_TEMPLATE = (
        "You observe the room from a new location and orientation.\n"
        "{observations}\n"
        "Treat {origin_name} as the origin (0, 0).\n"
        "What is your 2D coordinate (x, y)?\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    def __init__(self, np_random, room, agent, config: Dict[str, Any] = None):
        super().__init__(np_random, room, agent, config)
        self.only_main_room: bool = bool(self.config.get('only_main_room', False))
        self._bin = EgoFrontBins()
        self._dist = StandardDistanceBins()
        self._ctx: Dict[str, Any] = {}

    def _take_observations(self, neglect_objects: List[str] = None) -> str:
        obs_result = ObserveApproxAction().execute(self.room, self.agent, neglect_objects=neglect_objects or [], free_position=True)
        return action_results_to_text([obs_result])

    def _pick_room(self) -> int:
        if self.only_main_room:
            return 1
        rids = [int(r) for r in self.room.objects_by_room.keys() if isinstance(r, int) and r > 0]
        self.np_random.shuffle(rids)
        for rid in rids:
            names = self.room.objects_by_room.get(int(rid), [])
            if len(names) < 2:
                continue
            objs = [self.room.get_object_by_name(n) for n in names]
            ok = False
            for i in range(len(objs)):
                for j in range(i + 1, len(objs)):
                    d = float(np.linalg.norm(objs[i].pos - objs[j].pos))
                    if d > 1.0 + 1e-6:
                        ok = True; break
                if ok: break
            if ok:
                return int(rid)
        return 1

    def _sample_valid_agent_pose(self) -> Tuple[Tuple[int, int], Tuple[int, int], int, List[Object], List[Object]]:
        """Pick a room and a pose: >=1 visible object and >=1 hidden object in that room."""
        rid = self._pick_room()
        xmin, xmax, ymin, ymax = self.room.get_boundary(room_id=rid)
        coords = [(x, y) for x in range(xmin, xmax + 1) for y in range(ymin, ymax + 1)]
        self.np_random.shuffle(coords)
        for pos in coords:
            if self.room.get_cell_info(pos[0], pos[1])['object_name']:
                continue
            for ori in self.np_random.permutation([(0,1), (1,0), (0,-1), (-1,0)]):
                tmp = self.agent.copy()
                tmp.pos, tmp.ori, tmp.room_id = np.array(pos), np.array(ori), rid
                in_room = [o for o in self.room.objects if int(o.room_id) == rid]
                vis = [o for o in in_room if BaseAction._is_visible(tmp, o) and not np.allclose(o.pos, tmp.pos)]
                hid = [o for o in in_room if o not in vis and not np.allclose(o.pos, tmp.pos)]
                if vis and hid:
                    return pos, ori, rid, vis, hid
        raise ValueError("No valid pose found")

    def generate_question(self) -> str:
        pos, ori, rid, visible_objs, hidden_objs = self._sample_valid_agent_pose()
        self.agent.pos, self.agent.ori, self.agent.room_id = np.array(pos), np.array(ori), int(rid)
        origin_obj = self.np_random.choice(hidden_objs)
        observations = self._take_observations()

        # correct answer: your coord relative to origin
        origin_pos = tuple(origin_obj.pos)
        correct_coord = (int(self.agent.pos[0]) - origin_pos[0], int(self.agent.pos[1]) - origin_pos[1])
        # store ctx for choices
        self._ctx = {
            'rid': int(rid),
            'origin_pos': origin_pos,
            'visible_names': [o.name for o in visible_objs],
            'agent_ori': tuple(self.agent.ori),
        }
        choices, correct_idx = self.generate_choices(correct_coord)
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            observations=observations,
            origin_name=origin_obj.name,
            choices_text=choices_text,
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question

    def generate_choices(self, correct_coord: Tuple[int, int]) -> Tuple[List[str], int]:
        rid = int(self._ctx['rid'])
        origin_pos = tuple(self._ctx['origin_pos'])
        agent_ori = tuple(self._ctx['agent_ori'])
        visible_names = list(self._ctx['visible_names'])

        # true discrete rels: object -> agent (from agent orientation)
        true_rels = {}
        for name in visible_names:
            obj = self.room.get_object_by_name(name)
            rel = PairwiseRelationshipDiscrete.relationship(tuple(obj.pos), tuple(self.agent.pos), anchor_ori=agent_ori, bin_system=self._bin, distance_bin_system=self._dist)
            true_rels[name] = (int(rel.direction.bin_id), int(rel.dist.bin_id))

        def fmt(p: Tuple[int, int]) -> str:
            return f"({int(p[0] - origin_pos[0])}, {int(p[1] - origin_pos[1])})"

        correct_text = f"({int(correct_coord[0])}, {int(correct_coord[1])})"
        out, seen = [correct_text], {correct_text}

        # sample wrong points inside the room until 3 satisfy the mismatch criterion
        xmin, xmax, ymin, ymax = self.room.get_boundary(room_id=rid)
        candidates = [(x, y) for x in range(xmin, xmax + 1) for y in range(ymin, ymax + 1)]
        self.np_random.shuffle(candidates)
        for x, y in candidates:
            if len(out) >= 4:
                break
            if (x, y) == tuple(self.agent.pos) or np.linalg.norm(np.array((x, y)) - np.array(self.agent.pos)) < 2:
                continue
            if self.room.get_cell_info(x, y)['object_name']:
                continue
            # compare discrete rels
            mismatch = False
            for name in visible_names:
                obj = self.room.get_object_by_name(name)
                rel = PairwiseRelationshipDiscrete.relationship(tuple(obj.pos), (int(x), int(y)), anchor_ori=agent_ori, bin_system=self._bin, distance_bin_system=self._dist)
                pair = (int(rel.direction.bin_id), int(rel.dist.bin_id))
                if pair != true_rels[name]:
                    mismatch = True
                    break
            if mismatch:
                s = fmt((x, y))
                if s not in seen:
                    out.append(s); seen.add(s)

        # if still not enough (degenerate rooms), fill random unique points
        if len(out) < 4:
            for x, y in candidates:
                if len(out) >= 4:
                    break
                s = fmt((x, y))
                if s not in seen and (x, y) != tuple(self.agent.pos):
                    out.append(s); seen.add(s)

        self.np_random.shuffle(out)
        return out, out.index(correct_text)


