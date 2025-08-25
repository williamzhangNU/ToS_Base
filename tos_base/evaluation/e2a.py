"""E2A: object coordinates and orientations identification task."""

from typing import List, Tuple
import numpy as np

from .tasks import BaseEvaluationTask
from ..core.object import Object
from ..core.relationship import CardinalBinsAllo


class E2AEvaluationTask(BaseEvaluationTask):
    """Given object names, choose correct coordinates and orientations."""

    QUESTION_TEMPLATE = (
        "Treat your starting position as the origin (0, 0), facing north.\n"
        "Consider the global map coordinates (x right, y up).\n"
        "What are the coordinates and orientations of these objects: {object_names}?\n\n"
        "Answer format: [obj at (x, y) facing orientation, ...] where orientation is north/east/south/west\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    def generate_question(self) -> str:
        # select ≥3 objects from the whole environment
        names = [o.name for o in self.room.objects]
        self.np_random.shuffle(names)
        pick_names = names[:max(3, min(5, len(names)))]
        objects = [self.room.get_object_by_name(n) for n in pick_names]
        object_names = [obj.name for obj in objects]

        # transform points so that initial agent position is origin
        def rel(p):
            return (int(p[0]) - self.agent.init_pos[0], int(p[1]) - self.agent.init_pos[1])
        correct_answer = [(obj.name, rel(obj.pos), self._get_orientation_string(obj)) for obj in objects]

        # 2) choices: include challenging wrong options using discrete-change helper
        choices, correct_idx = self.generate_choices(correct_answer, objects)
        choices_text, correct_label = self.format_choices(choices, correct_idx)

        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            object_names=", ".join(object_names),
            choices_text=choices_text,
        )
        self.eval_data.answer = correct_label
        self.eval_data.choices = choices
        self.eval_data.reasoning = self._generate_reasoning()
        return self.eval_data.question

    def _get_orientation_string(self, obj: Object) -> str:
        return {(0, 1): "north", (1, 0): "east", (0, -1): "south", (-1, 0): "west"}[tuple(obj.ori)]

    def generate_choices(self, correct_answer: List[Tuple], objects: List[Object]) -> Tuple[List[str], int]:
        correct_str = self._format_answer(correct_answer)
        choices = [correct_str]

        abs_coords = [tuple(map(int, obj.pos)) for obj in objects]

        # wrong options: change a random subset (>=1) of selected objects
        origin = tuple(map(int, self.agent.init_pos))
        for _ in range(3):
            k = int(self.np_random.integers(1, max(2, len(objects))))
            idxs = list(range(len(objects)))
            self.np_random.shuffle(idxs)
            idxs = idxs[:k]
            new_coords = list(abs_coords)
            for i in idxs:
                rid_i = int(objects[i].room_id) if getattr(objects[i], 'room_id', None) is not None else 1
                p = self.sample_point_with_discrete_change(
                    reference_pos=abs_coords[i],
                    anchor_pos=origin,
                    room_id=rid_i,
                    min_distance=2.0,
                    bin_system=CardinalBinsAllo(),
                    anchor_ori=(0,1),
                    must_be_free=False,
                ) or abs_coords[i]
                new_coords[i] = p
            wrong = []
            for (name, _, ori), newc in zip(correct_answer, new_coords):
                wrong.append((name, (int(newc[0]) - origin[0], int(newc[1]) - origin[1]), ori))
            choices.append(self._format_answer(wrong))

        self.np_random.shuffle(choices)
        correct_idx = choices.index(correct_str)
        return choices, correct_idx

    def _format_answer(self, answer: List[Tuple]) -> str:
        return ", ".join([f"{name} at {coord} facing {orientation}" for name, coord, orientation in answer])



