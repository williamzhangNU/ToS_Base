"""E2A: object coordinates and orientations identification task."""

from typing import List, Tuple
import numpy as np

from .tasks import BaseEvaluationTask
from ..core.object import Object
from ..core.graph import DirectionalGraph


class E2AEvaluationTask(BaseEvaluationTask):
    """Given object names, choose correct coordinates and orientations."""

    QUESTION_TEMPLATE = (
        "You return to your starting position and facing north.\n"
        "Consider the global map coordinates (x right, y up).\n"
        "What are the coordinates and orientations of these objects: {object_names}?\n\n"
        "Answer format: [obj at (x, y) facing orientation, ...] where orientation is north/east/south/west\n\n"
        "Choose the correct answer:\n{choices_text}\n\n"
        "IMPORTANT: Answer with ONLY the letter (A, B, C, ...).\n\n"
    )

    def generate_question(self) -> str:
        objects = list(self.room.objects) + [self.agent]
        self.np_random.shuffle(objects)
        object_names = [obj.name for obj in objects]

        correct_answer = [(obj.name, tuple(obj.pos.astype(int)), self._get_orientation_string(obj)) for obj in objects]

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

        coords = [obj.pos for obj in objects]
        gt_v_matrix, gt_h_matrix = DirectionalGraph.create_graph_from_coordinates(coords)

        for _ in range(3):
            wrong_answer = self._generate_wrong_choice(correct_answer, objects, gt_v_matrix, gt_h_matrix)
            choices.append(self._format_answer(wrong_answer))

        self.np_random.shuffle(choices)
        correct_idx = choices.index(correct_str)
        return choices, correct_idx

    def _format_answer(self, answer: List[Tuple]) -> str:
        return ", ".join([f"{name} at {coord} facing {orientation}" for name, coord, orientation in answer])

    def _generate_wrong_choice(self, correct_answer: List[Tuple], objects: List[Object], gt_v_matrix: np.ndarray, gt_h_matrix: np.ndarray) -> List[Tuple]:
        orientations = ["north", "east", "south", "west"]
        for _ in range(20):
            wrong_coords = [tuple(self.room.get_random_point(self.np_random)) for _ in range(len(objects))]
            wrong_v_matrix, wrong_h_matrix = DirectionalGraph.create_graph_from_coordinates(wrong_coords)
            if not (np.array_equal(wrong_v_matrix, gt_v_matrix) and np.array_equal(wrong_h_matrix, gt_h_matrix)):
                return [
                    (name, coord, self.np_random.choice(orientations))
                    for (name, _, _), coord in zip(correct_answer, wrong_coords)
                ]
        return correct_answer


