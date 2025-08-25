"""False belief task: detect a changed object (rotation or movement)."""

from typing import List, Tuple, Any
import numpy as np

from .tasks import BaseEvaluationTask
from ..core.relationship import CardinalBinsAllo


class FalseBeliefEvaluationTask(BaseEvaluationTask):
    """Identify which object moved, which rotated, and rotation degrees (clockwise)."""

    QUESTION_TEMPLATE = (
        "The environment changed. You turned 360° with 90° FOV and observed:\n"
        "{observations}\n\n"
        "Which object moved, which object rotated, and by how many degrees (clockwise) did it rotate?\n"
        "Answer format: moved=<name>; rotated=<name>; deg=<0|90|180|270>\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    def generate_question(self) -> str:
        # 1) Pick a room with ≥3 objects AND ≥3 oriented objects; place agent randomly
        rids = [int(r) for r in self.room.objects_by_room.keys() if isinstance(r, int) and r > 0]
        self.np_random.shuffle(rids)
        rid = -1
        for r in rids:
            names_r = self.room.objects_by_room.get(int(r), [])
            objs_r = [self.room.get_object_by_name(n) for n in names_r]
            if len(objs_r) >= 3 and sum(1 for o in objs_r if o.has_orientation) >= 3:
                rid = int(r); break
        if rid == -1: raise ValueError("No room with ≥3 objects AND ≥3 oriented objects")

        xmin, xmax, ymin, ymax = self.room.get_boundary(room_id=rid)
        coords = [(x, y) for x in range(xmin, xmax + 1) for y in range(ymin, ymax + 1)]
        self.np_random.shuffle(coords)
        pos = None
        for p in coords:
            if not self.room.get_cell_info(p[0], p[1])['object_name']:
                pos = p; break
        if pos is None:
            pos = self.room.get_random_point(self.np_random)
        ori = self.np_random.choice([(0,1),(1,0),(0,-1),(-1,0)])
        self.agent.pos, self.agent.ori, self.agent.room_id = np.array(pos), np.array(ori), int(rid)

        # 2) Apply one movement and one rotation (distinct objects if possible)
        names = self.room.objects_by_room.get(int(rid), [])
        objs = [self.room.get_object_by_name(n) for n in names]
        moved_name = self._apply_movement(objs, rid)
        oriented_objects = [o for o in objs if o.has_orientation and o.name != moved_name]
        if len(oriented_objects) == 0:
            oriented_objects = [o for o in objs if o.has_orientation]
        rotated_name, deg = self._apply_rotation(oriented_objects)


        # 3) 360° observations and ask a 3-part question
        observations = self._take_full_observations()
        choices, correct_idx = self.generate_choices((moved_name, rotated_name, deg))
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = self.QUESTION_TEMPLATE.format(observations=observations, choices_text=choices_text)
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question

    def _apply_movement(self, objs: List[Any], rid: int) -> str:
        target_obj = self.np_random.choice(objs)
        new_p = self._sample_point_with_discrete_change(
            reference_pos=tuple(map(int, target_obj.pos)),
            anchor_pos=tuple(map(int, self.agent.pos)),
            room_id=int(rid),
            min_distance=2.0,
            bin_system=CardinalBinsAllo(),
            anchor_ori=(0,1),
            must_be_free=True,
            max_trials=800,
        )
        target_obj.pos = np.array(new_p)
        return target_obj.name

    def _apply_rotation(self, oriented_objects: List[Any]) -> Tuple[str, int]:
        assert len(oriented_objects) >= 1, "Need oriented object(s)"
        target_obj = self.np_random.choice(oriented_objects)
        rotation_degrees = int(self.np_random.choice([0, 90, 180, 270]))
        rotations = {90: [[0, -1], [1, 0]], 180: [[-1, 0], [0, -1]], 270: [[0, 1], [-1, 0]]}
        target_obj.ori = target_obj.ori @ rotations[rotation_degrees]
        return target_obj.name, rotation_degrees

    def _position_agent_random(self) -> None:
        self._position_agent_at(self.room.get_random_point(self.np_random))

    def generate_choices(self, correct_answer: Any) -> Tuple[List[str], int]:
        moved_name, rotated_name, deg = correct_answer
        rid = int(self.agent.room_id)
        objects = [n for n in self.room.objects_by_room.get(rid, [])]
        self.np_random.shuffle(objects)

        def fmt(m, r, d):
            return f"moved={m}; rotated={r}; deg={d}"

        correct = fmt(moved_name, rotated_name, deg)
        choices, seen = [correct], {correct}

        # build wrongs by perturbing one or two fields
        while len(choices) < 4:
            m, r, d = moved_name, rotated_name, deg
            mode = int(self.np_random.integers(0, 3))  # 0: move only, 1: rotate only, 2: deg only
            if mode == 0:
                m = self.np_random.choice([o for o in objects if o != moved_name])
            elif mode == 1:
                r = self.np_random.choice([o for o in objects if o != rotated_name])
            else:
                d = int(self.np_random.choice([x for x in [0, 90, 180, 270] if x != deg]))
            s = fmt(m, r, d)
            if s not in seen:
                choices.append(s); seen.add(s)
        self.np_random.shuffle(choices)
        return choices, choices.index(correct)


