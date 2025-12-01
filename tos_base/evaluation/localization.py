"""Localization task: infer your 2D coordinate from a new view."""

from typing import List, Tuple
import numpy as np
import json

from .tasks import BaseEvaluationTask, retry_generate_question
from ..core.object import Object, Gate
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

    def _get_origin(self) -> Tuple[Tuple[int, int], str]:
        """Determine origin position and name."""
        init_pos = self.agent.init_pos
        init_room_info = self.room.get_cell_info(int(init_pos[0]), int(init_pos[1]))
        init_room_id = init_room_info.get('room_id')
        
        current_room_id = self.agent.room_id
        
        if current_room_id == init_room_id:
            return tuple(map(int, init_pos)), "your starting position"
        
        # Find a door in the current room
        names = self.room.objects_by_room.get(int(current_room_id), [])
        if hasattr(self.room, 'gates_by_room'):
            names.extend(self.room.gates_by_room.get(int(current_room_id), []))
            
        gates = []
        for name in names:
            obj = self.room.get_object_by_name(name)
            if isinstance(obj, Gate):
                gates.append(obj)
        
        if gates:
            gate = self.np_random.choice(gates)
            return tuple(map(int, gate.pos)), f"the {gate.name}"
            
        return tuple(map(int, init_pos)), "your starting position"


class BaseLocation2ActionEvaluationTask(BaseLocEvaluationTask):
    """Base class for Location2Action (Backward Localization) tasks."""
    ACTION_TEMPLATE = (
        "You move to a new location and your current facing direction is {orientation}.\n"
        "{observations}\n"
    )
    QUESTION_TEMPLATE = (
        "Treat {origin_name} as the origin (0, 0), and your starting facing direction is north.\n"
        "What is your current 2D coordinate (x, y)?\n\n"
        "Answer format: (x, y)\n"
        "Example: (2, -1)\n"
    )

    def _get_observations(self) -> str:
        raise NotImplementedError

    @retry_generate_question
    def generate_question(self) -> dict:
        pos, ori, rid, _, _ = self._sample_valid_agent_pose()
        self.agent.pos, self.agent.ori, self.agent.room_id = np.array(pos), np.array(ori), int(rid)
        
        origin_pos, origin_name = self._get_origin()
        
        observations = self._get_observations()

        correct_coord = (
            int(self.agent.pos[0]) - int(origin_pos[0]),
            int(self.agent.pos[1]) - int(origin_pos[1]),
        )
        correct_orientation = _ori_to_name(tuple(self.agent.ori))

        self.eval_data.action = self.ACTION_TEMPLATE.format(
            orientation=correct_orientation,
            observations=observations
        )
        
        self.eval_data.question = self.eval_data.action + self.QUESTION_TEMPLATE.format(origin_name=origin_name)
        self.eval_data.answer = {'coord': correct_coord}
        self.eval_data.choices = []
        self.eval_data.id = hash(json.dumps(self.eval_data.answer) + self.eval_data.question)
        return self.eval_data.question


class Location2ActionTextEvaluationTask(BaseLocation2ActionEvaluationTask):
    """Localize your own coordinate (x, y) and orientation using text observations."""
    def _get_observations(self) -> str:
        return self._take_observations()


class Location2ActionVisionEvaluationTask(BaseLocation2ActionEvaluationTask):
    """Localize your own coordinate (x, y) and orientation using vision."""
    def _get_observations(self) -> str:
        return "You observe: <image>"

class Action2LocationEvaluationTask(BaseLocEvaluationTask):
    ACTION_TEMPLATE = (
        "Treat {origin_name} as the origin (0, 0), and your starting facing direction is north.\n"
        "You move to {loc} and face {direction}.\n"
    )
    QUESTION_TEMPLATE = (
        "What is the egocentric relation of {target}?\n\n"
        "Answer format: <direction>, <distance>\n"
        "Example: front, near\n"
    )

    @retry_generate_question
    def generate_question(self) -> dict:
        pos, ori, rid, _, hidden_objs = self._sample_valid_agent_pose()
        self.agent.pos, self.agent.ori, self.agent.room_id = np.array(pos), np.array(ori), int(rid)
        
        origin_pos, origin_name = self._get_origin()

        # question fields
        loc_rel = (int(self.agent.pos[0]) - origin_pos[0], int(self.agent.pos[1]) - origin_pos[1])
        dir_name = _ori_to_name(tuple(self.agent.ori))

        # compute correct observation text (pairwise-only, compact)
        rels = _visible_relations(self.room, self.agent)
        self.np_random.shuffle(rels)
        if not rels:
            raise ValueError("No visible relations found")
        target_name, direction, distance = rels[0]

        self.eval_data.action = self.ACTION_TEMPLATE.format(
            origin_name=origin_name,
            loc=f"({int(loc_rel[0])}, {int(loc_rel[1])})",
            direction=dir_name,
        )
        self.eval_data.question = self.eval_data.action + self.QUESTION_TEMPLATE.format(
            target=target_name,
        )
        self.eval_data.answer = f"{direction}, {distance}"
        self.eval_data.choices = []
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question


if __name__ == "__main__":
    from ..utils.room_utils import RoomPlotter, RoomGenerator
    from .task_types import EvalTaskType
    from tqdm import tqdm
    import numpy as np

    def test_task(task_name: str):
        print(f"\nTesting task: {task_name}")
        for seed in tqdm(range(0, 1)):
            np_random = np.random.default_rng(seed)
            room, agent = RoomGenerator.generate_room(
                room_size=(30, 30),
                n_objects=10,
                np_random=np_random,
                room_name='room',
                level=2,
                main=6,
            )
            try:
                task = EvalTaskType.create_task(task_name, np_random=np_random, room=room, agent=agent)
                print(f"Question: {task.generate_question()}")
                print(f"Answer: {task.answer}")
                
                # Test correct answer
                score, info = EvalTaskType.evaluate_prediction(task_name, task.answer, task.answer, task.choices)
                print(f"Correct Answer Evaluation: {score}, details: {info}")
                assert score == 1.0, f"Failed correct answer test for {task_name}"

                # Test robustness
                if isinstance(task.answer, str):
                    # Case insensitivity
                    robust_answer = task.answer.upper()
                    score, info = EvalTaskType.evaluate_prediction(task_name, robust_answer, task.answer, task.choices)
                    print(f"Robust Answer (Upper) Evaluation: {score}, details: {info}")
                    assert score == 1.0, f"Failed robust answer (Upper) test for {task_name}"
                    
                    # Extra whitespace
                    robust_answer = task.answer.replace(" ", "  ")
                    score, info = EvalTaskType.evaluate_prediction(task_name, robust_answer, task.answer, task.choices)
                    print(f"Robust Answer (Spaces) Evaluation: {score}, details: {info}")
                    assert score == 1.0, f"Failed robust answer (Spaces) test for {task_name}"

                    # Swapped order (for direction tasks)
                    if "," in task.answer:
                        parts = [p.strip() for p in task.answer.split(",")]
                        if len(parts) == 2:
                            swapped_answer = f"{parts[1]}, {parts[0]}"
                            score, info = EvalTaskType.evaluate_prediction(task_name, swapped_answer, task.answer, task.choices)
                            print(f"Robust Answer (Swapped) Evaluation: {score}, details: {info}")
                            assert score == 1.0, f"Failed robust answer (Swapped) test for {task_name}"

                # Test incorrect answer
                incorrect_answer = "wrong answer"
                score, info = EvalTaskType.evaluate_prediction(task_name, incorrect_answer, task.answer, task.choices)
                print(f"Incorrect Answer Evaluation: {score}, details: {info}")
                assert score < 1.0, f"Failed incorrect answer test for {task_name}"

            except ValueError as e:
                print(f"Skipping seed {seed} for {task_name}: {e}")

    task_names = ['fwd_loc', 'bwd_loc_text', 'bwd_loc_vision']
    for task_name in task_names:
        test_task(task_name)

