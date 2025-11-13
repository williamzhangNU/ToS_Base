"""Localization task: infer your 2D coordinate from a new view."""

from typing import List, Tuple
import numpy as np
import json

from .tasks import BaseEvaluationTask, retry_generate_question
from ..core.object import Object
from ..core.relationship import PairwiseRelationshipDiscrete
from ..actions import BaseAction
from ..actions import ObserveAction
from ..utils.utils import hash


def _visible_relations(room, agent) -> List[Tuple[str, str, str]]:
    res = ObserveAction().execute(room, agent.copy(), free_position=True)
    triples = res.data.get('relation_triples', [])
    out: List[Tuple[str, str, str]] = []
    for tr in triples:
        rel = getattr(tr, "relation", None)
        if isinstance(rel, PairwiseRelationshipDiscrete):
            out.append((tr.subject, rel.direction.bin_label, rel.dist.bin_label))
    return out
def _ori_to_name(ori: Tuple[int, int]) -> str:
    mapping = {(0, 1): "north", (1, 0): "east", (0, -1): "south", (-1, 0): "west"}
    return mapping.get(tuple(int(x) for x in ori), "north")

class BaseLocEvaluationTask(BaseEvaluationTask):
    """Base class for localization tasks."""
    def _pick_room(self) -> int:
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
                # Filter to objects within distance 5 of tmp position
                nearby_hid = [o for o in hid if np.linalg.norm(o.pos - tmp.pos) <= 5]
                if nearby_hid:
                    hid = nearby_hid
                if len(vis) >= 1 and hid:
                    return pos, ori, rid, vis, hid
        raise ValueError("No valid pose found")


class BackwardLocTextEvaluationTask(BaseLocEvaluationTask):
    """Localize your own coordinate (x, y) and orientation."""
    ACTION_TEMPLATE = (
        "You change to a new location and facing direction\n"
        "{observations}\n"
    )
    QUESTION_TEMPLATE = (
        "Treat {origin_name} as the origin (0, 0), and your starting facing direction is north.\n"
        "What is your current 2D coordinate (x, y) and facing direction?\n\n"
        "Answer format: (x, y), facing <direction>\n"
        "Example: (2, -1), facing east\n"
    )

    @retry_generate_question
    def generate_question(self) -> dict:
        pos, ori, rid, _, hidden_objs = self._sample_valid_agent_pose()
        self.agent.pos, self.agent.ori, self.agent.room_id = np.array(pos), np.array(ori), int(rid)
        origin_obj = self.np_random.choice(hidden_objs)
        observations = self._take_observations()

        origin_pos = tuple(origin_obj.pos)
        correct_coord = (
            int(self.agent.pos[0]) - int(origin_pos[0]),
            int(self.agent.pos[1]) - int(origin_pos[1]),
        )
        correct_orientation = _ori_to_name(tuple(self.agent.ori))

        self.eval_data.action = self.ACTION_TEMPLATE.format(observations= observations )
        self.eval_data.question = self.eval_data.action + self.QUESTION_TEMPLATE.format(
            origin_name=origin_obj.name,
        )
        self.eval_data.answer = {
            'coord': correct_coord,
            'orientation': correct_orientation,
        }
        self.eval_data.choices = []
        self.eval_data.id = hash(json.dumps(self.eval_data.answer) + self.eval_data.question)
        return self.eval_data.question
    
class BackwardLocVisionEvaluationTask(BaseLocEvaluationTask):
    """Localize your own coordinate (x, y) and orientation."""
    ACTION_TEMPLATE = (
        "You change to a new location and facing direction\n"
        "{observations}\n"
    )
    QUESTION_TEMPLATE = (
        "Treat {origin_name} as the origin (0, 0), and your starting facing direction is north.\n"
        "What is your current 2D coordinate (x, y) and facing direction?\n\n"
        "Answer format: (x, y), facing <direction>\n"
        "Example: (2, -1), facing east\n"
    )

    @retry_generate_question
    def generate_question(self) -> dict:
        pos, ori, rid, _, hidden_objs = self._sample_valid_agent_pose()
        self.agent.pos, self.agent.ori, self.agent.room_id = np.array(pos), np.array(ori), int(rid)
        origin_obj = self.np_random.choice(hidden_objs)
        observations = self._take_observations()
        origin_pos = tuple(origin_obj.pos)
        correct_coord = (
            int(self.agent.pos[0]) - int(origin_pos[0]),
            int(self.agent.pos[1]) - int(origin_pos[1]),
        )
        correct_orientation = _ori_to_name(tuple(self.agent.ori))

        self.eval_data.action = self.ACTION_TEMPLATE.format(observations= "<image>")
        self.eval_data.question = self.eval_data.action + self.QUESTION_TEMPLATE.format(
            origin_name=origin_obj.name,
        )
        self.eval_data.answer = {
            'coord': correct_coord,
            'orientation': correct_orientation,
        }
        self.eval_data.choices = []
        self.eval_data.id = hash(json.dumps(self.eval_data.answer) + self.eval_data.question)
        return self.eval_data.question

class ForwardLocEvaluationTask(BaseLocEvaluationTask):
    ACTION_TEMPLATE = (
        "Treat {origin_name} as the origin (0, 0), and your starting facing direction is north.\n"
        "You move to {loc} and face {direction}.\n"
    )
    QUESTION_TEMPLATE = (
        "What is the egocentric relation of {target}?\n\n"
        "Answer format: {target} is at <direction>, <distance>\n"
        "Example: {target} is at front, near\n"
    )

    @retry_generate_question
    def generate_question(self) -> dict:
        pos, ori, rid, _, hidden_objs = self._sample_valid_agent_pose()
        self.agent.pos, self.agent.ori, self.agent.room_id = np.array(pos), np.array(ori), int(rid)
        origin_obj = self.np_random.choice(hidden_objs)

        # question fields
        origin_pos = tuple(origin_obj.pos)
        loc_rel = (int(self.agent.pos[0]) - origin_pos[0], int(self.agent.pos[1]) - origin_pos[1])
        dir_name = _ori_to_name(tuple(self.agent.ori))

        # compute correct observation text (pairwise-only, compact)
        rels = _visible_relations(self.room, self.agent)
        self.np_random.shuffle(rels)
        if not rels:
            raise ValueError("No visible relations found")
        target_name, direction, distance = rels[0]

        self.eval_data.action = self.ACTION_TEMPLATE.format(
            origin_name=origin_obj.name,
            loc=f"({int(loc_rel[0])}, {int(loc_rel[1])})",
            direction=dir_name,
        )
        self.eval_data.question = self.eval_data.action + self.QUESTION_TEMPLATE.format(
            target=target_name,
        )
        self.eval_data.answer = f"{target_name} is at {direction}, {distance}"
        self.eval_data.choices = []
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question

