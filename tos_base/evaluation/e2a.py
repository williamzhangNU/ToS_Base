"""E2A: object coordinates and orientations identification task."""

from typing import List, Tuple
import numpy as np

from .tasks import BaseEvaluationTask, retry_generate_question
from ..core.object import Object
from ..utils.utils import hash

class AlloMappingEvaluationTask(BaseEvaluationTask):
    """Report allocentric coordinates for selected objects."""

    QUESTION_TEMPLATE = (
        "Treat your starting position as the origin (0, 0) while facing north.\n"
        "Report allocentric coordinates using (x right, y up).\n"
        "Objects: {object_list}.\n"
        "Answer format: (x0, y0); (x1, y1); ... in the same order.\n"
        "Example: (1, 0); (-2, 3); (0, -1)\n"
    )

    @retry_generate_question
    def generate_question(self) -> str:
        self._selected_objects = self._pick_objects()
        self.eval_data.question = self.QUESTION_TEMPLATE.format(
            object_list=", ".join(obj.name for obj in self._selected_objects)
        )
        self.eval_data.answer = self._collect_coordinates(self._selected_objects)
        self.eval_data.choices = []
        self.eval_data.id = hash(self.eval_data.question)
        return self.eval_data.question

    def _pick_objects(self) -> List[Object]:
        pool = list(self.room.objects)
        self.np_random.shuffle(pool)
        count = int(self.np_random.integers(3, min(6, len(pool)) + 1))
        return pool[:count]

    def _collect_coordinates(self, objects: List[Object]) -> List[Tuple[int, int]]:
        ox, oy = map(int, self.agent.init_pos)
        coords: List[Tuple[int, int]] = []
        for obj in objects:
            x, y = map(int, obj.pos)
            coords.append((x - ox, y - oy))
        return coords


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
                score, info = EvalTaskType.evaluate_prediction(task_name, str(task.answer), task.answer, task.choices)
                print(f"Correct Answer Evaluation: {score}, details: {info}")
                assert score == 1.0, f"Failed correct answer test for {task_name}, {score}"

                # Test incorrect answer
                incorrect_answer = [(99, 99)]
                score, info = EvalTaskType.evaluate_prediction(task_name, str(incorrect_answer), task.answer, task.choices)
                print(f"Incorrect Answer Evaluation: {score}, details: {info}")
                assert score < 1.0, f"Failed incorrect answer test for {task_name}"

            except ValueError as e:
                print(f"Skipping seed {seed} for {task_name}: {e}")

    task_names = ['e2a']
    for task_name in task_names:
        test_task(task_name)
