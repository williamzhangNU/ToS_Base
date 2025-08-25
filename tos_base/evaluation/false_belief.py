"""False belief task: detect a changed object (rotation or movement)."""

from typing import List, Tuple, Any
import numpy as np

from .tasks import SpatialManipulationTaskBase
from ..core.relationship import CardinalBinsAllo


class FalseBeliefEvaluationTask(SpatialManipulationTaskBase):
    """Identify which object changed (rotated or moved)."""

    QUESTION_TEMPLATE = (
        "One object in the room has changed (rotated or moved).\n"
        "You observe by turning around 360° with 90° field of view.\n"
        "{observations}\n"
        "Which object changed?\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    def generate_question(self) -> str:
        # 1) Pick a room with ≥3 objects, place agent randomly
        rids = [int(r) for r in self.room.objects_by_room.keys() if isinstance(r, int) and r > 0]
        self.np_random.shuffle(rids)
        rid = 1
        for r in rids:
            if len(self.room.objects_by_room.get(int(r), [])) >= 3:
                rid = int(r); break

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

        # 2) Apply change to a random object (rotation or movement)
        names = self.room.objects_by_room.get(int(rid), [])
        objs = [self.room.get_object_by_name(n) for n in names]
        changed = self._apply_movement(objs, rid) if self.config.get('action_type', 'rotation') else self._apply_movement(objs, rid)


        # 3) 360° observations and ask which object changed
        observations = self._take_full_observations()
        choices, correct_idx = self.generate_choices(changed)
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = self.QUESTION_TEMPLATE.format(observations=observations, choices_text=choices_text)
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question

    def _apply_movement(self, objs: List[Any], rid: int) -> str:
        target_obj = self.np_random.choice(objs)
        new_p = self.sample_point_with_discrete_change(
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

    def _apply_rotation(self, objs: List[Any]) -> str:
        oriented_objects = [obj for obj in objs if obj.has_orientation]
        assert len(oriented_objects) >= 2, "Need at least 2 objects with orientation"
        target_obj = self.np_random.choice(oriented_objects)
        rotation_degrees = self.np_random.choice([90, 180, 270])
        rotations = {90: [[0, -1], [1, 0]], 180: [[-1, 0], [0, -1]], 270: [[0, 1], [-1, 0]]}
        target_obj.ori = target_obj.ori @ rotations[rotation_degrees]
        return target_obj.name

    def _position_agent_random(self) -> None:
        self._position_agent_at(self.room.get_random_point(self.np_random))

    def generate_choices(self, correct_answer: Any) -> Tuple[List[str], int]:
        correct_name = str(correct_answer)
        # choices only from the selected room
        rid = int(self.agent.room_id)
        objects = [n for n in self.room.objects_by_room.get(rid, [])]
        distractors = [n for n in objects if n != correct_name]
        self.np_random.shuffle(distractors)
        choices = [correct_name] + distractors[:3]
        self.np_random.shuffle(choices)
        return choices, choices.index(correct_name)


